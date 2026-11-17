// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Candidate 1 (Recommended): `RoaringIndex + AhoCorasick + RegexAutomata` Hybrid Columnar Engine.
//!
//! Splits predicate evaluation by operator class:
//! 1. Single-pass `AHashMap` lookup for `Exists` and `Equals` (`String`, `Int`, `Double`, `Bool`, `Bytes`,
//!    plus compile-time hex-to-bytes decoding for `trace_id` and `span_id`).
//! 2. Single-pass scan per attribute table (`LogAttrs`, `ResourceAttrs`, `ScopeAttrs`) regardless of the
//!    number of distinct watched attribute keys.
//! 3. Exact cross-type `i64`/`f64` numeric comparison for `Gt`, `Gte`, `Lt`, `Lte`.
//! 4. Pure-Rust SIMD `AhoCorasick` (`MatchKind::Standard`) for `Contains` and mandatory literal substrings
//!    extracted from `Regex` via `regex_syntax::hir::literal::Extractor`, invoking `regex_automata::meta::Regex`
//!    (`multi_line(false)`) only on candidate strings whose prefilter literal matched.
//! 5. Dictionary-value caching on `DictionaryArray<UInt8Type | UInt16Type>` columns.

use ahash::AHashMap;
use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use arrow::array::{
    Array, ArrayRef, BooleanArray, DictionaryArray, Float64Array, RecordBatch, StringArray,
    StructArray, UInt8Array,
};
use arrow::datatypes::{UInt8Type, UInt16Type};
use otel_arrow_dfe_pdata::otap::OtapArrowRecords;
use otel_arrow_dfe_pdata::otap::filter::IdBitmapPool;
use otel_arrow_dfe_pdata::otlp::attributes::AttributeValueType;
use otel_arrow_dfe_pdata::proto::opentelemetry::arrow::v1::ArrowPayloadType;
use otel_arrow_dfe_pdata::schema::consts;
use parking_lot::Mutex;
use regex_automata::Input;
use regex_automata::meta::{Cache as MetaCache, Regex as MetaRegex};
use regex_syntax::hir::{Hir, HirKind, literal::Extractor};
use roaring::RoaringBitmap;
use smallvec::SmallVec;

use crate::model::{
    Action, AttributePath, LogFieldSelector, LogMatcher, LogRecordField, Policy, PolicyDefinition,
    PolicyError, Predicate, ScalarValue, ScopeField, eq_i64_f64,
};
use crate::otap_ops::{
    CompiledTransformPolicy, ExtractedValue, LogsBatchView, apply_spec_transforms_masked,
    extract_anyval_from_struct, extract_bytes_from_array, extract_fixed_bytes_from_array,
    extract_i64_from_array, extract_parent_id_u16, extract_str_from_array,
    filter_otap_logs_by_bitmap, format_non_zero_hex,
};
use crate::{EvaluationOutput, PolicyEvaluationEngine};

const MAX_REGEX_SHARD_SIZE: usize = 128;

#[derive(Debug)]
struct CompiledSingleRegex {
    matcher_id: u32,
    regex: MetaRegex,
    cache: Mutex<MetaCache>,
}

impl CompiledSingleRegex {
    fn compile(matcher_id: u32, pattern: &str) -> Result<Self, PolicyError> {
        let syntax = regex_automata::util::syntax::Config::new().multi_line(false);
        let regex = MetaRegex::builder()
            .syntax(syntax)
            .build(pattern)
            .map_err(|e| PolicyError::Compile(format!("invalid regex '{pattern}': {e}")))?;
        let cache = Mutex::new(regex.create_cache());
        Ok(Self {
            matcher_id,
            regex,
            cache,
        })
    }

    #[inline]
    fn is_match(&self, haystack: &str) -> bool {
        let mut cache = self.cache.lock();
        self.regex
            .search_with(&mut cache, &Input::new(haystack))
            .is_some()
    }
}

#[derive(Debug)]
struct UnfilteredRegexShard {
    shard_regex: MetaRegex,
    shard_cache: Mutex<MetaCache>,
    members: Vec<CompiledSingleRegex>,
}

impl UnfilteredRegexShard {
    fn compile(patterns: Vec<(u32, String)>) -> Result<Self, PolicyError> {
        let syntax = regex_automata::util::syntax::Config::new().multi_line(false);
        let raw_strs: Vec<&str> = patterns.iter().map(|(_, p)| p.as_str()).collect();
        let shard_regex = MetaRegex::builder()
            .syntax(syntax)
            .build_many(&raw_strs)
            .map_err(|e| PolicyError::Compile(format!("regex shard build error: {e}")))?;
        let shard_cache = Mutex::new(shard_regex.create_cache());
        let mut members = Vec::with_capacity(patterns.len());
        for (mid, pat) in patterns {
            members.push(CompiledSingleRegex::compile(mid, &pat)?);
        }
        Ok(Self {
            shard_regex,
            shard_cache,
            members,
        })
    }

    #[inline]
    fn eval(&self, haystack: &str, out_matchers: &mut SmallVec<[u32; 8]>) {
        if self.members.len() == 1 {
            if self.members[0].is_match(haystack) {
                out_matchers.push(self.members[0].matcher_id);
            }
            return;
        }
        let shard_hit = {
            let mut cache = self.shard_cache.lock();
            self.shard_regex
                .search_with(&mut cache, &Input::new(haystack))
                .is_some()
        };
        if shard_hit {
            for m in &self.members {
                if m.is_match(haystack) {
                    out_matchers.push(m.matcher_id);
                }
            }
        }
    }
}

#[derive(Debug, Default)]
struct AcPatternEntry {
    /// Matchers satisfied directly when this AC literal matches (`Contains(String)`).
    direct_matchers: SmallVec<[u32; 4]>,
    /// Indices into `FieldWatchers::prefiltered_regexes` that must be verified when this literal matches.
    prefiltered_regex_indices: SmallVec<[u32; 4]>,
}

/// All compiled watchers targeting a single field or attribute key.
#[derive(Debug, Default)]
struct FieldWatchers {
    exists_matchers: SmallVec<[u32; 4]>,
    exact_str: AHashMap<String, SmallVec<[u32; 4]>>,
    exact_int: AHashMap<i64, SmallVec<[u32; 4]>>,
    exact_bool_true: SmallVec<[u32; 4]>,
    exact_bool_false: SmallVec<[u32; 4]>,
    exact_bytes: AHashMap<Vec<u8>, SmallVec<[u32; 4]>>,
    numeric_comparisons: Vec<(u32, Predicate)>,
    empty_str_contains: SmallVec<[u32; 2]>,
    empty_bytes_contains: SmallVec<[u32; 2]>,
    bytes_contains: Vec<(u32, Vec<u8>)>,
    ac_automaton: Option<AhoCorasick>,
    ac_entries: Vec<AcPatternEntry>,
    prefiltered_regexes: Vec<CompiledSingleRegex>,
    unfiltered_shards: Vec<UnfilteredRegexShard>,
}

#[derive(Default)]
struct FieldWatchersBuilder {
    exists_matchers: SmallVec<[u32; 4]>,
    exact_str: AHashMap<String, SmallVec<[u32; 4]>>,
    exact_int: AHashMap<i64, SmallVec<[u32; 4]>>,
    exact_bool_true: SmallVec<[u32; 4]>,
    exact_bool_false: SmallVec<[u32; 4]>,
    exact_bytes: AHashMap<Vec<u8>, SmallVec<[u32; 4]>>,
    numeric_comparisons: Vec<(u32, Predicate)>,
    empty_str_contains: SmallVec<[u32; 2]>,
    empty_bytes_contains: SmallVec<[u32; 2]>,
    bytes_contains: Vec<(u32, Vec<u8>)>,
    ac_literal_map: AHashMap<Vec<u8>, usize>,
    ac_literals: Vec<Vec<u8>>,
    ac_entries: Vec<AcPatternEntry>,
    prefiltered_regexes: Vec<CompiledSingleRegex>,
    unfiltered_regexes: Vec<(u32, String)>,
}

impl FieldWatchersBuilder {
    fn intern_ac_literal(&mut self, lit: Vec<u8>) -> usize {
        if let Some(&idx) = self.ac_literal_map.get(&lit) {
            idx
        } else {
            let idx = self.ac_literals.len();
            self.ac_literal_map.insert(lit.clone(), idx);
            self.ac_literals.push(lit);
            self.ac_entries.push(AcPatternEntry::default());
            idx
        }
    }

    fn add_matcher(
        &mut self,
        matcher_id: u32,
        predicate: &Predicate,
        is_hex_id_field: Option<usize>,
    ) -> Result<(), PolicyError> {
        match predicate {
            Predicate::Exists => {
                self.exists_matchers.push(matcher_id);
            }
            Predicate::Equals(scalar) => match scalar {
                ScalarValue::String(s) => {
                    if let Some(hex_byte_len) = is_hex_id_field {
                        // Decode 32-char or 16-char lowercase hex into raw bytes at compile time if valid
                        if s.len() == hex_byte_len * 2
                            && s.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
                            && let Ok(decoded) = hex::decode(s)
                            && decoded.iter().any(|&b| b != 0)
                        {
                            self.exact_bytes.entry(decoded).or_default().push(matcher_id);
                        }
                    }
                    self.exact_str.entry(s.clone()).or_default().push(matcher_id);
                }
                ScalarValue::Int(i) => {
                    self.exact_int.entry(*i).or_default().push(matcher_id);
                }
                ScalarValue::Double(d) => {
                    // Also store in numeric_comparisons for exact cross-type i64/f64 equality
                    self.numeric_comparisons
                        .push((matcher_id, Predicate::Equals(ScalarValue::Double(*d))));
                }
                ScalarValue::Bool(true) => self.exact_bool_true.push(matcher_id),
                ScalarValue::Bool(false) => self.exact_bool_false.push(matcher_id),
                ScalarValue::Bytes(b) => {
                    self.exact_bytes.entry(b.clone()).or_default().push(matcher_id);
                }
            },
            Predicate::Gt(_) | Predicate::Gte(_) | Predicate::Lt(_) | Predicate::Lte(_) => {
                self.numeric_comparisons.push((matcher_id, predicate.clone()));
            }
            Predicate::Contains(scalar) => match scalar {
                ScalarValue::String(s) => {
                    if s.is_empty() {
                        self.empty_str_contains.push(matcher_id);
                    } else {
                        let ac_idx = self.intern_ac_literal(s.as_bytes().to_vec());
                        self.ac_entries[ac_idx].direct_matchers.push(matcher_id);
                    }
                }
                ScalarValue::Bytes(b) => {
                    if b.is_empty() {
                        self.empty_bytes_contains.push(matcher_id);
                    } else {
                        self.bytes_contains.push((matcher_id, b.clone()));
                    }
                }
                _ => {}
            },
            Predicate::Regex(pat) => {
                // Parse HIR with multi_line(false) to inspect literals
                let hir_opt = regex_syntax::ParserBuilder::new()
                    .multi_line(false)
                    .build()
                    .parse(pat)
                    .ok();

                if let Some(ref hir) = hir_opt
                    && let Some(best_lit) = extract_best_mandatory_literal(hir)
                    && best_lit.len() >= 2
                {
                    let re_idx = self.prefiltered_regexes.len() as u32;
                    self.prefiltered_regexes
                        .push(CompiledSingleRegex::compile(matcher_id, pat)?);
                    let ac_idx = self.intern_ac_literal(best_lit);
                    self.ac_entries[ac_idx]
                        .prefiltered_regex_indices
                        .push(re_idx);
                } else {
                    self.unfiltered_regexes.push((matcher_id, pat.clone()));
                }
            }
        }
        Ok(())
    }

    fn build(self) -> Result<FieldWatchers, PolicyError> {
        let ac_automaton = if self.ac_literals.is_empty() {
            None
        } else {
            Some(
                AhoCorasickBuilder::new()
                    .match_kind(MatchKind::Standard)
                    .build(&self.ac_literals)
                    .map_err(|e| PolicyError::Compile(format!("AhoCorasick build error: {e}")))?,
            )
        };

        let mut unfiltered_shards = Vec::new();
        for chunk in self.unfiltered_regexes.chunks(MAX_REGEX_SHARD_SIZE) {
            unfiltered_shards.push(UnfilteredRegexShard::compile(chunk.to_vec())?);
        }

        Ok(FieldWatchers {
            exists_matchers: self.exists_matchers,
            exact_str: self.exact_str,
            exact_int: self.exact_int,
            exact_bool_true: self.exact_bool_true,
            exact_bool_false: self.exact_bool_false,
            exact_bytes: self.exact_bytes,
            numeric_comparisons: self.numeric_comparisons,
            empty_str_contains: self.empty_str_contains,
            empty_bytes_contains: self.empty_bytes_contains,
            bytes_contains: self.bytes_contains,
            ac_automaton,
            ac_entries: self.ac_entries,
            prefiltered_regexes: self.prefiltered_regexes,
            unfiltered_shards,
        })
    }
}

/// Extracts the longest mandatory contiguous literal byte slice from a regex `Hir`, if any.
fn extract_best_mandatory_literal(hir: &Hir) -> Option<Vec<u8>> {
    // First try regex_syntax's Extractor for prefixes/suffixes
    let mut extractor = Extractor::new();
    extractor.limit_total(16);
    let seq = extractor.extract(hir);
    if let Some(lits) = seq.literals()
        && lits.len() == 1
        && lits[0].as_bytes().len() >= 2
    {
        return Some(lits[0].as_bytes().to_vec());
    }

    // Walk top-level Concat / Capture / Repetition(min >= 1) to find the longest literal node
    let mut best: Option<Vec<u8>> = None;
    collect_mandatory_literals(hir, &mut best);
    best
}

fn collect_mandatory_literals(hir: &Hir, best: &mut Option<Vec<u8>>) {
    match hir.kind() {
        HirKind::Literal(lit) => {
            if lit.0.len() >= 2 && best.as_ref().is_none_or(|b| lit.0.len() > b.len()) {
                *best = Some(lit.0.to_vec());
            }
        }
        HirKind::Capture(cap) => collect_mandatory_literals(&cap.sub, best),
        HirKind::Repetition(rep) if rep.min >= 1 => {
            collect_mandatory_literals(&rep.sub, best);
        }
        HirKind::Concat(subs) => {
            let mut cur_run = Vec::new();
            for sub in subs {
                if let HirKind::Literal(lit) = sub.kind() {
                    cur_run.extend_from_slice(&lit.0);
                } else {
                    if cur_run.len() >= 2 && best.as_ref().is_none_or(|b| cur_run.len() > b.len()) {
                        *best = Some(cur_run.clone());
                    }
                    cur_run.clear();
                    collect_mandatory_literals(sub, best);
                }
            }
            if cur_run.len() >= 2 && best.as_ref().is_none_or(|b| cur_run.len() > b.len()) {
                *best = Some(cur_run);
            }
        }
        _ => {}
    }
}

impl FieldWatchers {
    #[inline]
    fn has_str_watchers(&self) -> bool {
        !self.exists_matchers.is_empty()
            || !self.exact_str.is_empty()
            || !self.empty_str_contains.is_empty()
            || self.ac_automaton.is_some()
            || !self.unfiltered_shards.is_empty()
    }

    #[inline]
    fn eval_str_into(&self, s: &str, out: &mut SmallVec<[u32; 8]>) {
        out.extend_from_slice(&self.exists_matchers);
        out.extend_from_slice(&self.empty_str_contains);
        if let Some(exact_ids) = self.exact_str.get(s) {
            out.extend_from_slice(exact_ids);
        }
        if let Some(ref ac) = self.ac_automaton {
            // Track which AC pattern IDs and prefiltered regex indices have already been tested on this string
            let mut tested_re: SmallVec<[u32; 8]> = SmallVec::new();
            let mut seen_ac: SmallVec<[usize; 8]> = SmallVec::new();
            for m in ac.find_overlapping_iter(s) {
                let pat_idx = m.pattern().as_usize();
                if seen_ac.contains(&pat_idx) {
                    continue;
                }
                seen_ac.push(pat_idx);
                let entry = &self.ac_entries[pat_idx];
                out.extend_from_slice(&entry.direct_matchers);
                for &re_idx in &entry.prefiltered_regex_indices {
                    if !tested_re.contains(&re_idx) {
                        tested_re.push(re_idx);
                        let cre = &self.prefiltered_regexes[re_idx as usize];
                        if cre.is_match(s) {
                            out.push(cre.matcher_id);
                        }
                    }
                }
            }
        }
        for shard in &self.unfiltered_shards {
            shard.eval(s, out);
        }
    }

    #[inline]
    fn eval_int_into(&self, val: i64, out: &mut SmallVec<[u32; 8]>) {
        out.extend_from_slice(&self.exists_matchers);
        if let Some(ids) = self.exact_int.get(&val) {
            out.extend_from_slice(ids);
        }
        for (mid, pred) in &self.numeric_comparisons {
            let matched = match pred {
                Predicate::Equals(ScalarValue::Double(d)) => eq_i64_f64(val, *d),
                Predicate::Equals(ScalarValue::Int(i)) => val == *i,
                Predicate::Gt(nv) | Predicate::Gte(nv) | Predicate::Lt(nv) | Predicate::Lte(nv) => {
                    pred.matches_ordering(nv.cmp_candidate_i64(val))
                }
                _ => false,
            };
            if matched {
                out.push(*mid);
            }
        }
    }

    #[inline]
    fn eval_double_into(&self, val: f64, out: &mut SmallVec<[u32; 8]>) {
        out.extend_from_slice(&self.exists_matchers);
        if val.is_nan() {
            return;
        }
        // Check if exact_int contains an integer equal to `val`
        if !self.exact_int.is_empty()
            && val.fract() == 0.0
            && val >= (i64::MIN as f64)
            && val < 9_223_372_036_854_775_808.0_f64
        {
            let as_i64 = val as i64;
            if eq_i64_f64(as_i64, val)
                && let Some(ids) = self.exact_int.get(&as_i64)
            {
                out.extend_from_slice(ids);
            }
        }
        for (mid, pred) in &self.numeric_comparisons {
            let matched = match pred {
                Predicate::Equals(ScalarValue::Double(d)) => !d.is_nan() && val == *d,
                Predicate::Equals(ScalarValue::Int(i)) => eq_i64_f64(*i, val),
                Predicate::Gt(nv) | Predicate::Gte(nv) | Predicate::Lt(nv) | Predicate::Lte(nv) => {
                    pred.matches_ordering(nv.cmp_candidate_f64(val))
                }
                _ => false,
            };
            if matched {
                out.push(*mid);
            }
        }
    }

    #[inline]
    fn eval_bool_into(&self, val: bool, out: &mut SmallVec<[u32; 8]>) {
        out.extend_from_slice(&self.exists_matchers);
        if val {
            out.extend_from_slice(&self.exact_bool_true);
        } else {
            out.extend_from_slice(&self.exact_bool_false);
        }
    }

    #[inline]
    fn eval_bytes_into(&self, val: &[u8], out: &mut SmallVec<[u32; 8]>) {
        out.extend_from_slice(&self.exists_matchers);
        out.extend_from_slice(&self.empty_bytes_contains);
        if let Some(ids) = self.exact_bytes.get(val) {
            out.extend_from_slice(ids);
        }
        for (mid, needle) in &self.bytes_contains {
            if memchr::memmem::find(val, needle).is_some() {
                out.push(*mid);
            }
        }
    }

    fn estimated_bytes(&self) -> usize {
        let ac_bytes = self
            .ac_automaton
            .as_ref()
            .map_or(0, AhoCorasick::memory_usage);
        let str_bytes: usize = self
            .exact_str
            .keys()
            .map(|k| k.len() + 32)
            .sum();
        let re_bytes = (self.prefiltered_regexes.len() + self.unfiltered_shards.len()) * 512;
        ac_bytes + str_bytes + re_bytes + self.ac_entries.len() * 32
    }
}

#[derive(Debug)]
struct ComplexMatcher {
    matcher_id: u32,
    field: LogFieldSelector,
    predicate: Predicate,
    compiled_re: Option<CompiledSingleRegex>,
}

#[derive(Debug)]
struct CompiledPolicyRule {
    pos_matchers: SmallVec<[u32; 4]>,
    neg_matchers: SmallVec<[u32; 2]>,
}

impl CompiledPolicyRule {
    #[inline]
    fn evaluate_rows(&self, matcher_bitmaps: &[RoaringBitmap], all_rows: &RoaringBitmap) -> RoaringBitmap {
        if self.pos_matchers.is_empty() && self.neg_matchers.is_empty() {
            return all_rows.clone();
        }
        let mut acc = if let Some(&first_pos) = self.pos_matchers.first() {
            let bm = &matcher_bitmaps[first_pos as usize];
            if bm.is_empty() {
                return RoaringBitmap::new();
            }
            bm.clone()
        } else {
            all_rows.clone()
        };
        for &pos_id in self.pos_matchers.iter().skip(1) {
            acc &= &matcher_bitmaps[pos_id as usize];
            if acc.is_empty() {
                return acc;
            }
        }
        for &neg_id in &self.neg_matchers {
            acc -= &matcher_bitmaps[neg_id as usize];
            if acc.is_empty() {
                return acc;
            }
        }
        acc
    }
}

/// Candidate 1: `RoaringIndex + AhoCorasick + RegexAutomata` Hybrid Columnar Engine.
#[derive(Debug)]
pub struct HybridPolicyEngine {
    num_matchers: usize,
    body_watchers: Option<FieldWatchers>,
    sev_text_watchers: Option<FieldWatchers>,
    sev_num_watchers: Option<FieldWatchers>,
    event_name_watchers: Option<FieldWatchers>,
    trace_id_watchers: Option<FieldWatchers>,
    span_id_watchers: Option<FieldWatchers>,
    scope_name_watchers: Option<FieldWatchers>,
    scope_version_watchers: Option<FieldWatchers>,
    scope_schema_url_watchers: Option<FieldWatchers>,
    log_attr_watchers: AHashMap<String, FieldWatchers>,
    res_attr_watchers: AHashMap<String, FieldWatchers>,
    scope_attr_watchers: AHashMap<String, FieldWatchers>,
    complex_matchers: Vec<ComplexMatcher>,
    keep_filter_rules: Vec<CompiledPolicyRule>,
    drop_filter_rules: Vec<CompiledPolicyRule>,
    transform_rules: Vec<CompiledPolicyRule>,
    transforms: Vec<CompiledTransformPolicy>,
    heap_bytes: usize,
}

impl HybridPolicyEngine {
    /// Compiles a slice of `Policy` definitions into an immutable `HybridPolicyEngine`.
    pub fn compile(policies: &[Policy]) -> Result<Self, PolicyError> {
        let mut num_matchers = 0u32;
        let mut body_b = FieldWatchersBuilder::default();
        let mut has_body = false;
        let mut sev_text_b = FieldWatchersBuilder::default();
        let mut has_sev_text = false;
        let mut sev_num_b = FieldWatchersBuilder::default();
        let mut has_sev_num = false;
        let mut event_name_b = FieldWatchersBuilder::default();
        let mut has_event_name = false;
        let mut trace_id_b = FieldWatchersBuilder::default();
        let mut has_trace_id = false;
        let mut span_id_b = FieldWatchersBuilder::default();
        let mut has_span_id = false;
        let mut scope_name_b = FieldWatchersBuilder::default();
        let mut has_scope_name = false;
        let mut scope_ver_b = FieldWatchersBuilder::default();
        let mut has_scope_ver = false;
        let mut scope_url_b = FieldWatchersBuilder::default();
        let mut has_scope_url = false;

        let mut log_attr_b: AHashMap<String, FieldWatchersBuilder> = AHashMap::new();
        let mut res_attr_b: AHashMap<String, FieldWatchersBuilder> = AHashMap::new();
        let mut scope_attr_b: AHashMap<String, FieldWatchersBuilder> = AHashMap::new();
        let mut complex_matchers: Vec<ComplexMatcher> = Vec::new();

        let mut register_matcher = |m: &LogMatcher| -> Result<(u32, bool), PolicyError> {
            let mid = num_matchers;
            num_matchers += 1;

            // Route multi-segment paths, BodyField, or array Contains to complex_matchers
            match &m.field {
                LogFieldSelector::RecordField(rf) => match rf {
                    LogRecordField::Body => {
                        has_body = true;
                        body_b.add_matcher(mid, &m.predicate, None)?;
                    }
                    LogRecordField::SeverityText => {
                        has_sev_text = true;
                        sev_text_b.add_matcher(mid, &m.predicate, None)?;
                    }
                    LogRecordField::SeverityNumber => {
                        has_sev_num = true;
                        sev_num_b.add_matcher(mid, &m.predicate, None)?;
                    }
                    LogRecordField::EventName => {
                        has_event_name = true;
                        event_name_b.add_matcher(mid, &m.predicate, None)?;
                    }
                    LogRecordField::TraceId => {
                        has_trace_id = true;
                        trace_id_b.add_matcher(mid, &m.predicate, Some(16))?;
                    }
                    LogRecordField::SpanId => {
                        has_span_id = true;
                        span_id_b.add_matcher(mid, &m.predicate, Some(8))?;
                    }
                    LogRecordField::Unspecified => {}
                },
                LogFieldSelector::ScopeField(sf) => match sf {
                    ScopeField::Name => {
                        has_scope_name = true;
                        scope_name_b.add_matcher(mid, &m.predicate, None)?;
                    }
                    ScopeField::Version => {
                        has_scope_ver = true;
                        scope_ver_b.add_matcher(mid, &m.predicate, None)?;
                    }
                    ScopeField::SchemaUrl => {
                        has_scope_url = true;
                        scope_url_b.add_matcher(mid, &m.predicate, None)?;
                    }
                    ScopeField::Unspecified => {}
                },
                LogFieldSelector::LogAttribute(AttributePath { path }) if path.len() == 1 => {
                    log_attr_b
                        .entry(path[0].clone())
                        .or_default()
                        .add_matcher(mid, &m.predicate, None)?;
                    // Also register in complex_matchers if Contains is used (in case the attribute value is a CBOR Slice)
                    if matches!(m.predicate, Predicate::Contains(_)) {
                        complex_matchers.push(ComplexMatcher {
                            matcher_id: mid,
                            field: m.field.clone(),
                            predicate: m.predicate.clone(),
                            compiled_re: None,
                        });
                    }
                }
                LogFieldSelector::ResourceAttribute(AttributePath { path }) if path.len() == 1 => {
                    res_attr_b
                        .entry(path[0].clone())
                        .or_default()
                        .add_matcher(mid, &m.predicate, None)?;
                    if matches!(m.predicate, Predicate::Contains(_)) {
                        complex_matchers.push(ComplexMatcher {
                            matcher_id: mid,
                            field: m.field.clone(),
                            predicate: m.predicate.clone(),
                            compiled_re: None,
                        });
                    }
                }
                LogFieldSelector::ScopeAttribute(AttributePath { path }) if path.len() == 1 => {
                    scope_attr_b
                        .entry(path[0].clone())
                        .or_default()
                        .add_matcher(mid, &m.predicate, None)?;
                    if matches!(m.predicate, Predicate::Contains(_)) {
                        complex_matchers.push(ComplexMatcher {
                            matcher_id: mid,
                            field: m.field.clone(),
                            predicate: m.predicate.clone(),
                            compiled_re: None,
                        });
                    }
                }
                _ => {
                    let compiled_re = if let Predicate::Regex(pat) = &m.predicate {
                        Some(CompiledSingleRegex::compile(mid, pat)?)
                    } else {
                        None
                    };
                    complex_matchers.push(ComplexMatcher {
                        matcher_id: mid,
                        field: m.field.clone(),
                        predicate: m.predicate.clone(),
                        compiled_re,
                    });
                }
            }
            Ok((mid, m.negate))
        };

        let mut keep_filter_rules = Vec::new();
        let mut drop_filter_rules = Vec::new();
        let mut raw_transforms: Vec<&Policy> = Vec::new();

        for policy in policies {
            if !policy.enabled {
                continue;
            }
            match &policy.definition {
                PolicyDefinition::LogFilter(lf) => {
                    if lf.matches.is_empty() || lf.action == Action::Unspecified {
                        continue;
                    }
                    let mut pos_matchers = SmallVec::new();
                    let mut neg_matchers = SmallVec::new();
                    for m in &lf.matches {
                        let (mid, neg) = register_matcher(m)?;
                        if neg {
                            neg_matchers.push(mid);
                        } else {
                            pos_matchers.push(mid);
                        }
                    }
                    let rule = CompiledPolicyRule {
                        pos_matchers,
                        neg_matchers,
                    };
                    match lf.action {
                        Action::Keep => keep_filter_rules.push(rule),
                        Action::Drop => drop_filter_rules.push(rule),
                        Action::Unspecified => {}
                    }
                }
                PolicyDefinition::LogTransform(_) => {
                    raw_transforms.push(policy);
                }
                PolicyDefinition::MetricFilter(_) | PolicyDefinition::TraceSampling(_) => {}
            }
        }

        // Sort transform policies by ascending lexicographical Policy.id per policy.md:431-440
        raw_transforms.sort_by(|a, b| a.id.cmp(&b.id));
        let mut transform_rules = Vec::with_capacity(raw_transforms.len());
        let mut transforms = Vec::with_capacity(raw_transforms.len());

        for policy in raw_transforms {
            if let PolicyDefinition::LogTransform(lt) = &policy.definition {
                let mut pos_matchers = SmallVec::new();
                let mut neg_matchers = SmallVec::new();
                for m in &lt.matches {
                    let (mid, neg) = register_matcher(m)?;
                    if neg {
                        neg_matchers.push(mid);
                    } else {
                        pos_matchers.push(mid);
                    }
                }
                transform_rules.push(CompiledPolicyRule {
                    pos_matchers,
                    neg_matchers,
                });
                transforms.push(CompiledTransformPolicy::compile(policy.id.clone(), lt)?);
            }
        }

        let body_watchers = if has_body { Some(body_b.build()?) } else { None };
        let sev_text_watchers = if has_sev_text {
            Some(sev_text_b.build()?)
        } else {
            None
        };
        let sev_num_watchers = if has_sev_num {
            Some(sev_num_b.build()?)
        } else {
            None
        };
        let event_name_watchers = if has_event_name {
            Some(event_name_b.build()?)
        } else {
            None
        };
        let trace_id_watchers = if has_trace_id {
            Some(trace_id_b.build()?)
        } else {
            None
        };
        let span_id_watchers = if has_span_id {
            Some(span_id_b.build()?)
        } else {
            None
        };
        let scope_name_watchers = if has_scope_name {
            Some(scope_name_b.build()?)
        } else {
            None
        };
        let scope_version_watchers = if has_scope_ver {
            Some(scope_ver_b.build()?)
        } else {
            None
        };
        let scope_schema_url_watchers = if has_scope_url {
            Some(scope_url_b.build()?)
        } else {
            None
        };

        let mut log_attr_watchers = AHashMap::with_capacity(log_attr_b.len());
        for (k, b) in log_attr_b {
            log_attr_watchers.insert(k, b.build()?);
        }
        let mut res_attr_watchers = AHashMap::with_capacity(res_attr_b.len());
        for (k, b) in res_attr_b {
            res_attr_watchers.insert(k, b.build()?);
        }
        let mut scope_attr_watchers = AHashMap::with_capacity(scope_attr_b.len());
        for (k, b) in scope_attr_b {
            scope_attr_watchers.insert(k, b.build()?);
        }

        let mut heap_bytes =
            (keep_filter_rules.len() + drop_filter_rules.len() + transform_rules.len()) * 32;
        for w in [
            &body_watchers,
            &sev_text_watchers,
            &sev_num_watchers,
            &event_name_watchers,
            &trace_id_watchers,
            &span_id_watchers,
            &scope_name_watchers,
            &scope_version_watchers,
            &scope_schema_url_watchers,
        ]
        .into_iter()
        .flatten()
        {
            heap_bytes += w.estimated_bytes();
        }
        for w in log_attr_watchers
            .values()
            .chain(res_attr_watchers.values())
            .chain(scope_attr_watchers.values())
        {
            heap_bytes += w.estimated_bytes() + 32;
        }

        Ok(Self {
            num_matchers: num_matchers as usize,
            body_watchers,
            sev_text_watchers,
            sev_num_watchers,
            event_name_watchers,
            trace_id_watchers,
            span_id_watchers,
            scope_name_watchers,
            scope_version_watchers,
            scope_schema_url_watchers,
            log_attr_watchers,
            res_attr_watchers,
            scope_attr_watchers,
            complex_matchers,
            keep_filter_rules,
            drop_filter_rules,
            transform_rules,
            transforms,
            heap_bytes,
        })
    }

    fn evaluate_all_matchers(&self, view: &LogsBatchView<'_>) -> Vec<RoaringBitmap> {
        let mut bitmaps = vec![RoaringBitmap::new(); self.num_matchers];
        if self.num_matchers == 0 {
            return bitmaps;
        }
        let num_rows = view.num_rows;

        // 1. Body StructArray
        if let Some(ref w) = self.body_watchers
            && let Some(body_col) = view.root.column_by_name(consts::BODY)
        {
            eval_body_column(body_col, num_rows, w, &mut bitmaps);
        }

        // 2. SeverityText
        if let Some(ref w) = self.sev_text_watchers
            && let Some(col) = view.root.column_by_name(consts::SEVERITY_TEXT)
        {
            eval_str_column_root(col, num_rows, true, w, &mut bitmaps);
        }

        // 3. SeverityNumber
        if let Some(ref w) = self.sev_num_watchers
            && let Some(col) = view.root.column_by_name(consts::SEVERITY_NUMBER)
        {
            eval_int_column_root(col, num_rows, true, w, &mut bitmaps);
        }

        // 4. EventName
        if let Some(ref w) = self.event_name_watchers
            && let Some(col) = view.root.column_by_name(consts::EVENT_NAME)
        {
            eval_str_column_root(col, num_rows, true, w, &mut bitmaps);
        }

        // 5. TraceId
        if let Some(ref w) = self.trace_id_watchers
            && let Some(col) = view.root.column_by_name(consts::TRACE_ID)
        {
            eval_hex_id_column_root(col, num_rows, w, &mut bitmaps);
        }

        // 6. SpanId
        if let Some(ref w) = self.span_id_watchers
            && let Some(col) = view.root.column_by_name(consts::SPAN_ID)
        {
            eval_hex_id_column_root(col, num_rows, w, &mut bitmaps);
        }

        // 7. ScopeSchemaUrl
        if let Some(ref w) = self.scope_schema_url_watchers
            && let Some(col) = view.root.column_by_name(consts::SCHEMA_URL)
        {
            eval_str_column_root(col, num_rows, true, w, &mut bitmaps);
        }

        // 8. Scope Name & Version
        if (self.scope_name_watchers.is_some() || self.scope_version_watchers.is_some())
            && let Some(scope_col) = view.root.column_by_name(consts::SCOPE)
            && let Some(scope_struct) = scope_col.as_any().downcast_ref::<StructArray>()
        {
            if let Some(ref w) = self.scope_name_watchers
                && let Some(col) = scope_struct.column_by_name(consts::NAME)
            {
                eval_str_column_root(col, num_rows, true, w, &mut bitmaps);
            }
            if let Some(ref w) = self.scope_version_watchers
                && let Some(col) = scope_struct.column_by_name(consts::VERSION)
            {
                eval_str_column_root(col, num_rows, true, w, &mut bitmaps);
            }
        }

        // 9. Single-pass LogAttrs scan
        if !self.log_attr_watchers.is_empty()
            && let Some(rb) = view.records.get(ArrowPayloadType::LogAttrs)
        {
            eval_attr_table_single_pass(rb, &self.log_attr_watchers, &mut |pid, m_ids| {
                if let Some(&row) = view.log_id_to_row.get(pid as usize)
                    && row != u32::MAX
                {
                    for &mid in m_ids {
                        bitmaps[mid as usize].insert(row);
                    }
                }
            });
        }

        // 10. Single-pass ResourceAttrs scan
        if !self.res_attr_watchers.is_empty()
            && let Some(rb) = view.records.get(ArrowPayloadType::ResourceAttrs)
        {
            eval_attr_table_single_pass(rb, &self.res_attr_watchers, &mut |pid, m_ids| {
                if let Some(rows_bm) = view.resource_id_to_rows.get(&pid) {
                    for &mid in m_ids {
                        bitmaps[mid as usize] |= rows_bm;
                    }
                }
            });
        }

        // 11. Single-pass ScopeAttrs scan
        if !self.scope_attr_watchers.is_empty()
            && let Some(rb) = view.records.get(ArrowPayloadType::ScopeAttrs)
        {
            eval_attr_table_single_pass(rb, &self.scope_attr_watchers, &mut |pid, m_ids| {
                if let Some(rows_bm) = view.scope_id_to_rows.get(&pid) {
                    for &mid in m_ids {
                        bitmaps[mid as usize] |= rows_bm;
                    }
                }
            });
        }

        // 12. Complex / nested CBOR / array matchers
        for cm in &self.complex_matchers {
            for row in 0..num_rows {
                let Some(val) = view.extract_field_at_row(&cm.field, row) else {
                    continue;
                };
                let matched = match &cm.predicate {
                    Predicate::Exists => true,
                    Predicate::Equals(s) => val.equals_scalar(s),
                    Predicate::Contains(s) => val.contains_scalar(s),
                    Predicate::Regex(_) => {
                        if let (ExtractedValue::Str(s), Some(re)) = (&val, &cm.compiled_re) {
                            re.is_match(s.as_ref())
                        } else {
                            false
                        }
                    }
                    Predicate::Gt(nv)
                    | Predicate::Gte(nv)
                    | Predicate::Lt(nv)
                    | Predicate::Lte(nv) => match val {
                        ExtractedValue::Int(i) => {
                            cm.predicate.matches_ordering(nv.cmp_candidate_i64(i))
                        }
                        ExtractedValue::Double(d) => {
                            cm.predicate.matches_ordering(nv.cmp_candidate_f64(d))
                        }
                        _ => false,
                    },
                };
                if matched {
                    bitmaps[cm.matcher_id as usize].insert(row as u32);
                }
            }
        }

        bitmaps
    }
}

fn eval_str_column_root(
    col: &ArrayRef,
    num_rows: usize,
    empty_is_absent: bool,
    w: &FieldWatchers,
    bitmaps: &mut [RoaringBitmap],
) {
    if !w.has_str_watchers() {
        return;
    }
    if let Some(sa) = col.as_any().downcast_ref::<StringArray>() {
        let mut matched: SmallVec<[u32; 8]> = SmallVec::new();
        for row in 0..num_rows {
            if !sa.is_valid(row) {
                continue;
            }
            let s = sa.value(row);
            if empty_is_absent && s.is_empty() {
                continue;
            }
            matched.clear();
            w.eval_str_into(s, &mut matched);
            for &mid in &matched {
                bitmaps[mid as usize].insert(row as u32);
            }
        }
    } else if let Some(da) = col.as_any().downcast_ref::<DictionaryArray<UInt8Type>>() {
        if let Some(vals) = da.values().as_any().downcast_ref::<StringArray>() {
            let dict_len = vals.len();
            let mut dict_matches: Vec<SmallVec<[u32; 8]>> = vec![SmallVec::new(); dict_len];
            for (k, slot) in dict_matches.iter_mut().enumerate() {
                if vals.is_valid(k) {
                    let s = vals.value(k);
                    if !(empty_is_absent && s.is_empty()) {
                        w.eval_str_into(s, slot);
                    }
                }
            }
            let keys = da.keys();
            for row in 0..num_rows {
                if da.is_valid(row) {
                    let k = keys.value(row) as usize;
                    for &mid in &dict_matches[k] {
                        bitmaps[mid as usize].insert(row as u32);
                    }
                }
            }
        }
    } else if let Some(da) = col.as_any().downcast_ref::<DictionaryArray<UInt16Type>>()
        && let Some(vals) = da.values().as_any().downcast_ref::<StringArray>()
    {
        let dict_len = vals.len();
        let mut dict_matches: Vec<SmallVec<[u32; 8]>> = vec![SmallVec::new(); dict_len];
        for (k, slot) in dict_matches.iter_mut().enumerate() {
            if vals.is_valid(k) {
                let s = vals.value(k);
                if !(empty_is_absent && s.is_empty()) {
                    w.eval_str_into(s, slot);
                }
            }
        }
        let keys = da.keys();
        for row in 0..num_rows {
            if da.is_valid(row) {
                let k = keys.value(row) as usize;
                for &mid in &dict_matches[k] {
                    bitmaps[mid as usize].insert(row as u32);
                }
            }
        }
    }
}

fn eval_int_column_root(
    col: &ArrayRef,
    num_rows: usize,
    zero_is_absent: bool,
    w: &FieldWatchers,
    bitmaps: &mut [RoaringBitmap],
) {
    let mut matched: SmallVec<[u32; 8]> = SmallVec::new();
    for row in 0..num_rows {
        let Some(v) = extract_i64_from_array(col, row) else {
            continue;
        };
        if zero_is_absent && v == 0 {
            continue;
        }
        matched.clear();
        w.eval_int_into(v, &mut matched);
        for &mid in &matched {
            bitmaps[mid as usize].insert(row as u32);
        }
    }
}

fn eval_hex_id_column_root(
    col: &ArrayRef,
    num_rows: usize,
    w: &FieldWatchers,
    bitmaps: &mut [RoaringBitmap],
) {
    let needs_hex_str =
        !w.empty_str_contains.is_empty() || w.ac_automaton.is_some() || !w.unfiltered_shards.is_empty();
    let mut matched: SmallVec<[u32; 8]> = SmallVec::new();
    for row in 0..num_rows {
        let Some(bytes) = extract_fixed_bytes_from_array(col, row) else {
            continue;
        };
        if bytes.iter().all(|&b| b == 0) {
            continue;
        }
        matched.clear();
        matched.extend_from_slice(&w.exists_matchers);
        if let Some(ids) = w.exact_bytes.get(bytes) {
            matched.extend_from_slice(ids);
        }
        if needs_hex_str && let Some(hex_str) = format_non_zero_hex(bytes) {
            matched.extend_from_slice(&w.empty_str_contains);
            if let Some(ref ac) = w.ac_automaton {
                let mut tested_re: SmallVec<[u32; 8]> = SmallVec::new();
                let mut seen_ac: SmallVec<[usize; 8]> = SmallVec::new();
                for m in ac.find_overlapping_iter(&hex_str) {
                    let pat_idx = m.pattern().as_usize();
                    if seen_ac.contains(&pat_idx) {
                        continue;
                    }
                    seen_ac.push(pat_idx);
                    let entry = &w.ac_entries[pat_idx];
                    matched.extend_from_slice(&entry.direct_matchers);
                    for &re_idx in &entry.prefiltered_regex_indices {
                        if !tested_re.contains(&re_idx) {
                            tested_re.push(re_idx);
                            let cre = &w.prefiltered_regexes[re_idx as usize];
                            if cre.is_match(&hex_str) {
                                matched.push(cre.matcher_id);
                            }
                        }
                    }
                }
            }
            for shard in &w.unfiltered_shards {
                shard.eval(&hex_str, &mut matched);
            }
        }
        for &mid in &matched {
            bitmaps[mid as usize].insert(row as u32);
        }
    }
}

fn eval_body_column(
    body_col: &ArrayRef,
    num_rows: usize,
    w: &FieldWatchers,
    bitmaps: &mut [RoaringBitmap],
) {
    let Some(sa) = body_col.as_any().downcast_ref::<StructArray>() else {
        eval_str_column_root(body_col, num_rows, false, w, bitmaps);
        return;
    };
    let Some(type_col) = sa.column_by_name(consts::ATTRIBUTE_TYPE) else {
        return;
    };
    let Some(type_arr) = type_col.as_any().downcast_ref::<UInt8Array>() else {
        return;
    };

    // Fast path: if all valid rows have body.type == Str, evaluate the `str` child column directly
    // (preserving dictionary caching on `logs.body.str`!).
    let all_str_or_empty = (0..num_rows).all(|r| {
        !sa.is_valid(r)
            || !type_arr.is_valid(r)
            || type_arr.value(r) == AttributeValueType::Str as u8
            || type_arr.value(r) == AttributeValueType::Empty as u8
    });

    if all_str_or_empty {
        if let Some(str_col) = sa.column_by_name(consts::ATTRIBUTE_STR) {
            if let Some(da) = str_col.as_any().downcast_ref::<DictionaryArray<UInt8Type>>()
                && let Some(vals) = da.values().as_any().downcast_ref::<StringArray>()
            {
                let mut dict_matches: Vec<SmallVec<[u32; 8]>> = vec![SmallVec::new(); vals.len()];
                for (k, slot) in dict_matches.iter_mut().enumerate() {
                    if vals.is_valid(k) {
                        w.eval_str_into(vals.value(k), slot);
                    }
                }
                let keys = da.keys();
                for row in 0..num_rows {
                    if sa.is_valid(row)
                        && type_arr.is_valid(row)
                        && type_arr.value(row) == AttributeValueType::Str as u8
                        && da.is_valid(row)
                    {
                        let k = keys.value(row) as usize;
                        for &mid in &dict_matches[k] {
                            bitmaps[mid as usize].insert(row as u32);
                        }
                    }
                }
                return;
            }
            if let Some(da) = str_col.as_any().downcast_ref::<DictionaryArray<UInt16Type>>()
                && let Some(vals) = da.values().as_any().downcast_ref::<StringArray>()
            {
                let mut dict_matches: Vec<SmallVec<[u32; 8]>> = vec![SmallVec::new(); vals.len()];
                for (k, slot) in dict_matches.iter_mut().enumerate() {
                    if vals.is_valid(k) {
                        w.eval_str_into(vals.value(k), slot);
                    }
                }
                let keys = da.keys();
                for row in 0..num_rows {
                    if sa.is_valid(row)
                        && type_arr.is_valid(row)
                        && type_arr.value(row) == AttributeValueType::Str as u8
                        && da.is_valid(row)
                    {
                        let k = keys.value(row) as usize;
                        for &mid in &dict_matches[k] {
                            bitmaps[mid as usize].insert(row as u32);
                        }
                    }
                }
                return;
            }
            if let Some(str_arr) = str_col.as_any().downcast_ref::<StringArray>() {
                let mut matched: SmallVec<[u32; 8]> = SmallVec::new();
                for row in 0..num_rows {
                    if sa.is_valid(row)
                        && type_arr.is_valid(row)
                        && type_arr.value(row) == AttributeValueType::Str as u8
                        && str_arr.is_valid(row)
                    {
                        matched.clear();
                        w.eval_str_into(str_arr.value(row), &mut matched);
                        for &mid in &matched {
                            bitmaps[mid as usize].insert(row as u32);
                        }
                    }
                }
                return;
            }
        }
    }

    // Polymorphic fallback across mixed body types
    let mut matched: SmallVec<[u32; 8]> = SmallVec::new();
    for row in 0..num_rows {
        let Some(ev) = extract_anyval_from_struct(sa, row) else {
            continue;
        };
        matched.clear();
        match ev {
            ExtractedValue::Str(s) => w.eval_str_into(s.as_ref(), &mut matched),
            ExtractedValue::Int(i) => w.eval_int_into(i, &mut matched),
            ExtractedValue::Double(d) => w.eval_double_into(d, &mut matched),
            ExtractedValue::Bool(b) => w.eval_bool_into(b, &mut matched),
            ExtractedValue::Bytes(b) => w.eval_bytes_into(b.as_ref(), &mut matched),
            ExtractedValue::Array(_) | ExtractedValue::Map(_) => {
                matched.extend_from_slice(&w.exists_matchers);
            }
        }
        for &mid in &matched {
            bitmaps[mid as usize].insert(row as u32);
        }
    }
}

fn eval_attr_table_single_pass<F>(
    rb: &RecordBatch,
    watchers: &AHashMap<String, FieldWatchers>,
    emit_parent_matches: &mut F,
) where
    F: FnMut(u16, &[u32]),
{
    let num_rows = rb.num_rows();
    if num_rows == 0 {
        return;
    }
    let Some(key_col) = rb.column_by_name(consts::ATTRIBUTE_KEY) else {
        return;
    };
    let Some(type_col) = rb.column_by_name(consts::ATTRIBUTE_TYPE) else {
        return;
    };
    let Some(type_arr) = type_col.as_any().downcast_ref::<UInt8Array>() else {
        return;
    };

    let str_col = rb.column_by_name(consts::ATTRIBUTE_STR);
    let int_col = rb.column_by_name(consts::ATTRIBUTE_INT);
    let double_col = rb
        .column_by_name(consts::ATTRIBUTE_DOUBLE)
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
    let bool_col = rb
        .column_by_name(consts::ATTRIBUTE_BOOL)
        .and_then(|c| c.as_any().downcast_ref::<BooleanArray>());
    let bytes_col = rb.column_by_name(consts::ATTRIBUTE_BYTES);

    let mut matched: SmallVec<[u32; 8]> = SmallVec::new();

    for r in 0..num_rows {
        let Some(key_str) = extract_str_from_array(key_col, r) else {
            continue;
        };
        let Some(w) = watchers.get(key_str) else {
            continue;
        };
        let Some(pid) = extract_parent_id_u16(rb, r) else {
            continue;
        };
        if !type_arr.is_valid(r) {
            continue;
        }
        let Ok(vtype) = AttributeValueType::try_from(type_arr.value(r)) else {
            continue;
        };

        matched.clear();
        match vtype {
            AttributeValueType::Empty => continue,
            AttributeValueType::Str => {
                if let Some(col) = str_col
                    && let Some(s) = extract_str_from_array(col, r)
                {
                    w.eval_str_into(s, &mut matched);
                }
            }
            AttributeValueType::Int => {
                if let Some(col) = int_col
                    && let Some(v) = extract_i64_from_array(col, r)
                {
                    w.eval_int_into(v, &mut matched);
                }
            }
            AttributeValueType::Double => {
                if let Some(da) = double_col
                    && da.is_valid(r)
                {
                    w.eval_double_into(da.value(r), &mut matched);
                }
            }
            AttributeValueType::Bool => {
                if let Some(ba) = bool_col
                    && ba.is_valid(r)
                {
                    w.eval_bool_into(ba.value(r), &mut matched);
                }
            }
            AttributeValueType::Bytes => {
                if let Some(col) = bytes_col
                    && let Some(b) = extract_bytes_from_array(col, r)
                {
                    w.eval_bytes_into(b, &mut matched);
                }
            }
            AttributeValueType::Map | AttributeValueType::Slice => {
                matched.extend_from_slice(&w.exists_matchers);
            }
        }
        if !matched.is_empty() {
            emit_parent_matches(pid, &matched);
        }
    }
}

impl PolicyEvaluationEngine for HybridPolicyEngine {
    fn name(&self) -> &'static str {
        "Candidate1_Hybrid_Roaring_AC"
    }

    fn evaluate_filter(&self, batch: &OtapArrowRecords) -> Result<RoaringBitmap, PolicyError> {
        let Some(view) = LogsBatchView::try_new(batch)? else {
            return Ok(RoaringBitmap::new());
        };
        let mut all_rows = RoaringBitmap::new();
        all_rows.insert_range(0..view.num_rows as u32);

        if self.drop_filter_rules.is_empty() {
            return Ok(all_rows);
        }

        let matcher_bitmaps = self.evaluate_all_matchers(&view);
        let mut dropped = RoaringBitmap::new();
        for rule in &self.drop_filter_rules {
            dropped |= rule.evaluate_rows(&matcher_bitmaps, &all_rows);
        }
        if dropped.is_empty() {
            return Ok(all_rows);
        }

        let mut kept_exempt = RoaringBitmap::new();
        for rule in &self.keep_filter_rules {
            kept_exempt |= rule.evaluate_rows(&matcher_bitmaps, &all_rows);
        }

        // ACTION_KEEP overrides ACTION_DROP (`policy.md:730-739`)
        dropped -= &kept_exempt;
        all_rows -= &dropped;
        Ok(all_rows)
    }

    fn evaluate(&self, batch: OtapArrowRecords) -> Result<EvaluationOutput, PolicyError> {
        let Some(root) = batch.get(ArrowPayloadType::Logs) else {
            return Ok(EvaluationOutput {
                records: Some(batch),
                kept_rows: 0,
                dropped_rows: 0,
                transform_hits: 0,
            });
        };
        let orig_rows = root.num_rows();
        if orig_rows == 0 {
            return Ok(EvaluationOutput {
                records: Some(batch),
                kept_rows: 0,
                dropped_rows: 0,
                transform_hits: 0,
            });
        }

        let keep_rows = self.evaluate_filter(&batch)?;
        let kept_count = keep_rows.len();
        let dropped_count = (orig_rows as u64).saturating_sub(kept_count);
        if kept_count == 0 {
            return Ok(EvaluationOutput {
                records: None,
                kept_rows: 0,
                dropped_rows: dropped_count,
                transform_hits: 0,
            });
        }

        let mut pool = IdBitmapPool::new();
        let mut filtered = filter_otap_logs_by_bitmap(&batch, &keep_rows, orig_rows, &mut pool)?;

        let mut transform_hits = 0u64;
        if !self.transforms.is_empty()
            && let Some(tx_view) = LogsBatchView::try_new(&filtered)?
        {
            let mut tx_all_rows = RoaringBitmap::new();
            tx_all_rows.insert_range(0..tx_view.num_rows as u32);
            let tx_matcher_bitmaps = self.evaluate_all_matchers(&tx_view);
            let mut policy_matches = Vec::with_capacity(self.transform_rules.len());
            for rule in &self.transform_rules {
                policy_matches.push(rule.evaluate_rows(&tx_matcher_bitmaps, &tx_all_rows));
            }
            transform_hits =
                apply_spec_transforms_masked(&mut filtered, &self.transforms, &policy_matches)?;
        }

        Ok(EvaluationOutput {
            records: Some(filtered),
            kept_rows: kept_count,
            dropped_rows: dropped_count,
            transform_hits,
        })
    }

    fn estimated_heap_bytes(&self) -> usize {
        self.heap_bytes
    }
}
