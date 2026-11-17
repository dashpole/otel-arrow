// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Candidate 2: Sharded `regex-automata` (`meta::Regex`) + Columnar Dictionary-Cached Engine.
//!
//! Mirrors `policy-rs`'s multi-pattern regex-database architecture in 100% safe pure Rust:
//! - Compiles string predicates (`Equals(String)`, `Contains(String)`, `Regex`) per target field
//!   into shards of up to [`MAX_PATTERNS_PER_SHARD`] patterns using
//!   `regex_automata::meta::Regex::build_many` with `multi_line(false)` syntax semantics.
//! - Evaluates each unique `(LogFieldSelector, Predicate)` at most once per batch, caching
//!   multi-pattern automaton lookups across Arrow `DictionaryArray<UInt8Type | UInt16Type>` values.
//! - Scans OTAP attribute tables (`LogAttrs`, `ResourceAttrs`, `ScopeAttrs`) in a single pass
//!   per table with dictionary-key filtering and per-field string dictionary memoization.

use std::mem::size_of;

use ahash::AHashMap;
use arrow::array::{
    Array, ArrayRef, DictionaryArray, FixedSizeBinaryArray, RecordBatch, StringArray, StructArray,
    UInt8Array,
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
use regex_automata::util::syntax::Config as SyntaxConfig;
use roaring::RoaringBitmap;

use crate::model::{
    Action, AttributePath, LogFieldSelector, LogMatcher, LogRecordField, NumericValue, Policy,
    PolicyDefinition, PolicyError, Predicate, ScalarValue, ScopeField, eq_i64_f64,
};
use crate::otap_ops::{
    CompiledTransformPolicy, ExtractedValue, LogsBatchView, apply_spec_transforms_masked,
    extract_anyval_from_attr_batch, extract_anyval_from_struct, extract_fixed_bytes_from_array,
    extract_i64_from_array, extract_parent_id_u16, extract_str_from_array,
    filter_otap_logs_by_bitmap, format_non_zero_hex, navigate_extracted_path,
};
use crate::{EvaluationOutput, PolicyEvaluationEngine};

/// Maximum number of string patterns compiled into a single `regex_automata::meta::Regex` shard.
pub const MAX_PATTERNS_PER_SHARD: usize = 128;

/// Hashable canonical representation of [`ScalarValue`] for `O(1)` predicate deduplication.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CanonicalScalarKey {
    String(String),
    Int(i64),
    Bool(bool),
    DoubleBits(u64),
    Bytes(Vec<u8>),
}

impl From<&ScalarValue> for CanonicalScalarKey {
    fn from(value: &ScalarValue) -> Self {
        match value {
            ScalarValue::String(s) => Self::String(s.clone()),
            ScalarValue::Int(i) => Self::Int(*i),
            ScalarValue::Bool(b) => Self::Bool(*b),
            ScalarValue::Double(d) => Self::DoubleBits(canonical_f64_bits(*d)),
            ScalarValue::Bytes(b) => Self::Bytes(b.clone()),
        }
    }
}

/// Hashable canonical representation of [`NumericValue`] for `O(1)` predicate deduplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CanonicalNumericKey {
    Int(i64),
    DoubleBits(u64),
}

impl From<&NumericValue> for CanonicalNumericKey {
    fn from(value: &NumericValue) -> Self {
        match value {
            NumericValue::Int(i) => Self::Int(*i),
            NumericValue::Double(d) => Self::DoubleBits(canonical_f64_bits(*d)),
        }
    }
}

/// Hashable canonical representation of [`Predicate`] for `O(1)` predicate deduplication.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CanonicalPredicateKey {
    Exists,
    Equals(CanonicalScalarKey),
    Regex(String),
    Gt(CanonicalNumericKey),
    Gte(CanonicalNumericKey),
    Lt(CanonicalNumericKey),
    Lte(CanonicalNumericKey),
    Contains(CanonicalScalarKey),
}

impl From<&Predicate> for CanonicalPredicateKey {
    fn from(pred: &Predicate) -> Self {
        match pred {
            Predicate::Exists => Self::Exists,
            Predicate::Equals(s) => Self::Equals(CanonicalScalarKey::from(s)),
            Predicate::Regex(r) => Self::Regex(r.clone()),
            Predicate::Gt(n) => Self::Gt(CanonicalNumericKey::from(n)),
            Predicate::Gte(n) => Self::Gte(CanonicalNumericKey::from(n)),
            Predicate::Lt(n) => Self::Lt(CanonicalNumericKey::from(n)),
            Predicate::Lte(n) => Self::Lte(CanonicalNumericKey::from(n)),
            Predicate::Contains(s) => Self::Contains(CanonicalScalarKey::from(s)),
        }
    }
}

#[inline]
fn canonical_f64_bits(v: f64) -> u64 {
    if v == 0.0 {
        0.0_f64.to_bits()
    } else {
        v.to_bits()
    }
}

/// Individual verification strategy when a multi-pattern shard triggers on a haystack.
#[derive(Debug)]
enum StringMatcherVerify {
    Exact(String),
    Contains(String),
    Regex {
        re: MetaRegex,
        cache: Mutex<MetaCache>,
    },
}

impl StringMatcherVerify {
    #[inline]
    fn is_match(&self, haystack: &str, input: &Input<'_>) -> bool {
        match self {
            Self::Exact(s) => haystack == s.as_str(),
            Self::Contains(s) => {
                if s.is_empty() {
                    true
                } else {
                    memchr::memmem::find(haystack.as_bytes(), s.as_bytes()).is_some()
                }
            }
            Self::Regex { re, cache } => {
                let mut guard = cache.lock();
                re.search_with(&mut guard, input).is_some()
            }
        }
    }

    fn estimated_heap_bytes(&self) -> usize {
        match self {
            Self::Exact(s) | Self::Contains(s) => s.capacity(),
            Self::Regex { re, cache } => re.memory_usage() + cache.lock().memory_usage(),
        }
    }
}

/// Single compiled string pattern entry within a [`RegexShard`].
#[derive(Debug)]
struct StringPatternEntry {
    pred_id: usize,
    automaton_pattern: String,
    verify: StringMatcherVerify,
}

/// Multi-pattern `regex_automata::meta::Regex` shard covering up to [`MAX_PATTERNS_PER_SHARD`] patterns.
#[derive(Debug)]
struct RegexShard {
    shard_re: MetaRegex,
    shard_cache: Mutex<MetaCache>,
    entries: Vec<StringPatternEntry>,
    pattern_bytes: usize,
}

impl RegexShard {
    fn compile(entries: Vec<StringPatternEntry>) -> Result<Self, PolicyError> {
        let patterns: Vec<&str> = entries
            .iter()
            .map(|e| e.automaton_pattern.as_str())
            .collect();
        let pattern_bytes: usize = patterns.iter().map(|p| p.len()).sum();
        let syntax_cfg = SyntaxConfig::new().multi_line(false);
        let shard_re = MetaRegex::builder()
            .syntax(syntax_cfg)
            .build_many(&patterns)
            .map_err(|e| PolicyError::Compile(format!("regex-automata shard build error: {e}")))?;
        let shard_cache = Mutex::new(shard_re.create_cache());
        Ok(Self {
            shard_re,
            shard_cache,
            entries,
            pattern_bytes,
        })
    }

    #[inline]
    fn evaluate_haystack(&self, haystack: &str, matched_preds: &mut Vec<usize>) {
        let input = Input::new(haystack);
        let first_match_idx = {
            let mut cache = self.shard_cache.lock();
            self.shard_re
                .search_with(&mut cache, &input)
                .map(|m| m.pattern().as_usize())
        };
        let Some(first_idx) = first_match_idx else {
            return;
        };
        if let Some(entry) = self.entries.get(first_idx) {
            matched_preds.push(entry.pred_id);
        }
        if self.entries.len() > 1 {
            for (idx, entry) in self.entries.iter().enumerate() {
                if idx == first_idx {
                    continue;
                }
                if entry.verify.is_match(haystack, &input) {
                    matched_preds.push(entry.pred_id);
                }
            }
        }
    }

    fn estimated_heap_bytes(&self) -> usize {
        let cache_bytes = self.shard_cache.lock().memory_usage();
        let entries_bytes: usize = self
            .entries
            .iter()
            .map(|e| {
                size_of::<StringPatternEntry>()
                    + e.automaton_pattern.capacity()
                    + e.verify.estimated_heap_bytes()
            })
            .sum();
        size_of::<Self>() + self.shard_re.memory_usage() + cache_bytes + self.pattern_bytes + entries_bytes
    }
}

/// Sharded multi-pattern regex database for all string predicates targeting a single field.
#[derive(Debug, Default)]
struct FieldRegexDatabase {
    shards: Vec<RegexShard>,
}

impl FieldRegexDatabase {
    fn compile(entries: Vec<StringPatternEntry>) -> Result<Self, PolicyError> {
        if entries.is_empty() {
            return Ok(Self { shards: Vec::new() });
        }
        let mut shards = Vec::with_capacity(entries.len().div_ceil(MAX_PATTERNS_PER_SHARD));
        let mut iter = entries.into_iter();
        loop {
            let chunk: Vec<StringPatternEntry> = iter.by_ref().take(MAX_PATTERNS_PER_SHARD).collect();
            if chunk.is_empty() {
                break;
            }
            shards.push(RegexShard::compile(chunk)?);
        }
        Ok(Self { shards })
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.shards.is_empty()
    }

    #[inline]
    fn evaluate_haystack(&self, haystack: &str, matched_preds: &mut Vec<usize>) {
        for shard in &self.shards {
            shard.evaluate_haystack(haystack, matched_preds);
        }
    }

    fn estimated_heap_bytes(&self) -> usize {
        self.shards.iter().map(RegexShard::estimated_heap_bytes).sum()
    }
}

/// Non-string scalar or numeric range predicate attached to a field.
#[derive(Debug, Clone)]
struct NonStringPredicateEntry {
    pred_id: usize,
    predicate: Predicate,
}

/// Compiled evaluation plan for a single unique [`LogFieldSelector`].
#[derive(Debug)]
struct CompiledFieldPlan {
    field: LogFieldSelector,
    used_in_filter: bool,
    used_in_transform: bool,
    exists_pred_id: Option<usize>,
    regex_db: FieldRegexDatabase,
    non_string_preds: Vec<NonStringPredicateEntry>,
    /// `Contains(ScalarValue::String)` predicates that can also match CBOR `ExtractedValue::Array` elements.
    array_string_contains: Vec<(usize, String)>,
}

impl CompiledFieldPlan {
    #[inline]
    fn matches_mode(&self, filter_only: bool, transform_only: bool) -> bool {
        if filter_only {
            self.used_in_filter
        } else if transform_only {
            self.used_in_transform
        } else {
            self.used_in_filter || self.used_in_transform
        }
    }

    #[inline]
    fn eval_int_value(&self, val: i64, matched_preds: &mut Vec<usize>) {
        if let Some(pid) = self.exists_pred_id {
            matched_preds.push(pid);
        }
        for entry in &self.non_string_preds {
            let matched = match &entry.predicate {
                Predicate::Equals(ScalarValue::Int(t)) => val == *t,
                Predicate::Equals(ScalarValue::Double(t)) => eq_i64_f64(val, *t),
                Predicate::Gt(n) | Predicate::Gte(n) | Predicate::Lt(n) | Predicate::Lte(n) => {
                    entry.predicate.matches_ordering(n.cmp_candidate_i64(val))
                }
                _ => false,
            };
            if matched {
                matched_preds.push(entry.pred_id);
            }
        }
    }

    #[inline]
    fn eval_extracted_value(&self, val: &ExtractedValue<'_>, matched_preds: &mut Vec<usize>) {
        match val {
            ExtractedValue::Str(s) => {
                if let Some(pid) = self.exists_pred_id {
                    matched_preds.push(pid);
                }
                if !self.regex_db.is_empty() {
                    self.regex_db.evaluate_haystack(s.as_ref(), matched_preds);
                }
            }
            ExtractedValue::Int(i) => {
                self.eval_int_value(*i, matched_preds);
            }
            ExtractedValue::Double(d) => {
                if let Some(pid) = self.exists_pred_id {
                    matched_preds.push(pid);
                }
                for entry in &self.non_string_preds {
                    let matched = match &entry.predicate {
                        Predicate::Equals(ScalarValue::Double(t)) => {
                            !d.is_nan() && !t.is_nan() && *d == *t
                        }
                        Predicate::Equals(ScalarValue::Int(t)) => eq_i64_f64(*t, *d),
                        Predicate::Gt(n)
                        | Predicate::Gte(n)
                        | Predicate::Lt(n)
                        | Predicate::Lte(n) => {
                            entry.predicate.matches_ordering(n.cmp_candidate_f64(*d))
                        }
                        _ => false,
                    };
                    if matched {
                        matched_preds.push(entry.pred_id);
                    }
                }
            }
            ExtractedValue::Bool(b) => {
                if let Some(pid) = self.exists_pred_id {
                    matched_preds.push(pid);
                }
                for entry in &self.non_string_preds {
                    if let Predicate::Equals(ScalarValue::Bool(t)) = &entry.predicate
                        && *b == *t
                    {
                        matched_preds.push(entry.pred_id);
                    }
                }
            }
            ExtractedValue::Bytes(bytes) => {
                if let Some(pid) = self.exists_pred_id {
                    matched_preds.push(pid);
                }
                for entry in &self.non_string_preds {
                    let matched = match &entry.predicate {
                        Predicate::Equals(ScalarValue::Bytes(t)) => {
                            bytes.as_ref() == t.as_slice()
                        }
                        Predicate::Contains(ScalarValue::Bytes(t)) => {
                            if t.is_empty() {
                                true
                            } else {
                                memchr::memmem::find(bytes.as_ref(), t.as_slice()).is_some()
                            }
                        }
                        _ => false,
                    };
                    if matched {
                        matched_preds.push(entry.pred_id);
                    }
                }
            }
            ExtractedValue::Array(elems) => {
                if let Some(pid) = self.exists_pred_id {
                    matched_preds.push(pid);
                }
                for (pid, target_s) in &self.array_string_contains {
                    if elems.iter().any(|e| {
                        matches!(e, ExtractedValue::Str(s) if s.as_ref() == target_s.as_str())
                    }) {
                        matched_preds.push(*pid);
                    }
                }
                for entry in &self.non_string_preds {
                    if let Predicate::Contains(scalar) = &entry.predicate
                        && elems.iter().any(|e| e.equals_scalar(scalar))
                    {
                        matched_preds.push(entry.pred_id);
                    }
                }
            }
            ExtractedValue::Map(_) => {
                if let Some(pid) = self.exists_pred_id {
                    matched_preds.push(pid);
                }
            }
        }
    }

    fn estimated_heap_bytes(&self) -> usize {
        size_of::<Self>()
            + self.regex_db.estimated_heap_bytes()
            + self.non_string_preds.capacity() * size_of::<NonStringPredicateEntry>()
            + self
                .array_string_contains
                .iter()
                .map(|(_, s)| size_of::<(usize, String)>() + s.capacity())
                .sum::<usize>()
    }
}

/// Compiled matcher reference within a filter or transform policy.
#[derive(Debug, Clone, Copy)]
struct CompiledMatcherRef {
    pred_id: usize,
    negate: bool,
}

/// Compiled `LogFilterPolicy` using deduplicated predicate IDs.
#[derive(Debug, Clone)]
struct CompiledFilterPolicy {
    matchers: Vec<CompiledMatcherRef>,
    action: Action,
}

/// Builder for accumulating predicates on a single [`LogFieldSelector`] prior to shard compilation.
#[derive(Default)]
struct FieldPlanBuilder {
    used_in_filter: bool,
    used_in_transform: bool,
    exists_pred_id: Option<usize>,
    string_entries: Vec<StringPatternEntry>,
    non_string_preds: Vec<NonStringPredicateEntry>,
    array_string_contains: Vec<(usize, String)>,
}

/// Candidate 2 policy evaluation engine combining sharded `regex_automata::meta::Regex`
/// multi-pattern automatons with dictionary-cached columnar OTAP scans.
#[derive(Debug)]
pub struct ShardedRegexAutomataEngine {
    num_predicates: usize,
    root_field_plans: Vec<CompiledFieldPlan>,
    log_attr_plans: Vec<CompiledFieldPlan>,
    res_attr_plans: Vec<CompiledFieldPlan>,
    scope_attr_plans: Vec<CompiledFieldPlan>,
    filter_policies: Vec<CompiledFilterPolicy>,
    transform_matcher_sets: Vec<Vec<CompiledMatcherRef>>,
    transforms: Vec<CompiledTransformPolicy>,
}

impl ShardedRegexAutomataEngine {
    /// Compiles a slice of [`Policy`] definitions into a [`ShardedRegexAutomataEngine`].
    pub fn compile(policies: &[Policy]) -> Result<Self, PolicyError> {
        // Validate regex syntax across all policies upfront (including disabled policies)
        let syntax_cfg = SyntaxConfig::new().multi_line(false);
        for policy in policies {
            match &policy.definition {
                PolicyDefinition::LogFilter(f) => {
                    for m in &f.matches {
                        if let Predicate::Regex(pat) = &m.predicate {
                            MetaRegex::builder()
                                .syntax(syntax_cfg)
                                .build(pat)
                                .map_err(|e| {
                                    PolicyError::Compile(format!("invalid regex '{pat}': {e}"))
                                })?;
                        }
                    }
                }
                PolicyDefinition::LogTransform(tx) => {
                    for m in &tx.matches {
                        if let Predicate::Regex(pat) = &m.predicate {
                            MetaRegex::builder()
                                .syntax(syntax_cfg)
                                .build(pat)
                                .map_err(|e| {
                                    PolicyError::Compile(format!("invalid regex '{pat}': {e}"))
                                })?;
                        }
                    }
                }
                PolicyDefinition::MetricFilter(_) | PolicyDefinition::TraceSampling(_) => {}
            }
        }

        let mut pred_dedup: AHashMap<(LogFieldSelector, CanonicalPredicateKey), usize> =
            AHashMap::new();
        let mut field_builders: AHashMap<LogFieldSelector, FieldPlanBuilder> = AHashMap::new();
        let mut field_order: Vec<LogFieldSelector> = Vec::new();
        let mut num_predicates = 0usize;

        let mut register_matcher = |matcher: &LogMatcher,
                                    is_filter: bool,
                                    is_transform: bool|
         -> Result<CompiledMatcherRef, PolicyError> {
            let key = (
                matcher.field.clone(),
                CanonicalPredicateKey::from(&matcher.predicate),
            );
            let is_new_field = !field_builders.contains_key(&matcher.field);
            if is_new_field {
                field_order.push(matcher.field.clone());
            }
            let builder = field_builders.entry(matcher.field.clone()).or_default();
            builder.used_in_filter |= is_filter;
            builder.used_in_transform |= is_transform;

            if let Some(&existing_id) = pred_dedup.get(&key) {
                return Ok(CompiledMatcherRef {
                    pred_id: existing_id,
                    negate: matcher.negate,
                });
            }

            let pred_id = num_predicates;
            num_predicates += 1;
            pred_dedup.insert(key, pred_id);

            match &matcher.predicate {
                Predicate::Exists => {
                    builder.exists_pred_id = Some(pred_id);
                }
                Predicate::Equals(ScalarValue::String(s)) => {
                    let automaton_pattern = format!(r"\A(?:{})\z", regex_syntax::escape(s));
                    builder.string_entries.push(StringPatternEntry {
                        pred_id,
                        automaton_pattern,
                        verify: StringMatcherVerify::Exact(s.clone()),
                    });
                }
                Predicate::Contains(ScalarValue::String(s)) => {
                    let automaton_pattern = regex_syntax::escape(s);
                    builder.string_entries.push(StringPatternEntry {
                        pred_id,
                        automaton_pattern,
                        verify: StringMatcherVerify::Contains(s.clone()),
                    });
                    builder.array_string_contains.push((pred_id, s.clone()));
                }
                Predicate::Regex(pat) => {
                    let re = MetaRegex::builder()
                        .syntax(syntax_cfg)
                        .build(pat)
                        .map_err(|e| PolicyError::Compile(format!("invalid regex '{pat}': {e}")))?;
                    let cache = Mutex::new(re.create_cache());
                    builder.string_entries.push(StringPatternEntry {
                        pred_id,
                        automaton_pattern: pat.clone(),
                        verify: StringMatcherVerify::Regex { re, cache },
                    });
                }
                other => {
                    builder.non_string_preds.push(NonStringPredicateEntry {
                        pred_id,
                        predicate: other.clone(),
                    });
                }
            }

            Ok(CompiledMatcherRef {
                pred_id,
                negate: matcher.negate,
            })
        };

        let mut filter_policies = Vec::new();
        let mut raw_transforms: Vec<&Policy> = Vec::new();

        for policy in policies {
            if !policy.enabled {
                continue;
            }
            match &policy.definition {
                PolicyDefinition::LogFilter(f) => {
                    let mut matchers = Vec::with_capacity(f.matches.len());
                    for m in &f.matches {
                        matchers.push(register_matcher(m, true, false)?);
                    }
                    filter_policies.push(CompiledFilterPolicy {
                        matchers,
                        action: f.action,
                    });
                }
                PolicyDefinition::LogTransform(_) => {
                    raw_transforms.push(policy);
                }
                PolicyDefinition::MetricFilter(_) | PolicyDefinition::TraceSampling(_) => {}
            }
        }

        // Sort enabled transform policies by ascending lexicographical Policy.id
        raw_transforms.sort_by(|a, b| a.id.cmp(&b.id));

        let mut transform_matcher_sets = Vec::with_capacity(raw_transforms.len());
        let mut transforms = Vec::with_capacity(raw_transforms.len());

        for policy in raw_transforms {
            if let PolicyDefinition::LogTransform(tx) = &policy.definition {
                let mut matchers = Vec::with_capacity(tx.matches.len());
                for m in &tx.matches {
                    matchers.push(register_matcher(m, false, true)?);
                }
                transform_matcher_sets.push(matchers);
                transforms.push(CompiledTransformPolicy::compile(policy.id.clone(), tx)?);
            }
        }

        let mut root_field_plans = Vec::new();
        let mut log_attr_plans = Vec::new();
        let mut res_attr_plans = Vec::new();
        let mut scope_attr_plans = Vec::new();

        for field in field_order {
            let Some(builder) = field_builders.remove(&field) else {
                continue;
            };
            let regex_db = FieldRegexDatabase::compile(builder.string_entries)?;
            let plan = CompiledFieldPlan {
                field: field.clone(),
                used_in_filter: builder.used_in_filter,
                used_in_transform: builder.used_in_transform,
                exists_pred_id: builder.exists_pred_id,
                regex_db,
                non_string_preds: builder.non_string_preds,
                array_string_contains: builder.array_string_contains,
            };
            match &plan.field {
                LogFieldSelector::LogAttribute(_) => log_attr_plans.push(plan),
                LogFieldSelector::ResourceAttribute(_) => res_attr_plans.push(plan),
                LogFieldSelector::ScopeAttribute(_) => scope_attr_plans.push(plan),
                LogFieldSelector::RecordField(_)
                | LogFieldSelector::ScopeField(_)
                | LogFieldSelector::BodyField(_) => root_field_plans.push(plan),
            }
        }

        Ok(Self {
            num_predicates,
            root_field_plans,
            log_attr_plans,
            res_attr_plans,
            scope_attr_plans,
            filter_policies,
            transform_matcher_sets,
            transforms,
        })
    }

    fn evaluate_predicates_for_view(
        &self,
        view: &LogsBatchView<'_>,
        pred_bitmaps: &mut [RoaringBitmap],
        filter_only: bool,
        transform_only: bool,
    ) {
        let mut matched_preds = Vec::with_capacity(8);

        for plan in &self.root_field_plans {
            if !plan.matches_mode(filter_only, transform_only) {
                continue;
            }
            eval_root_field_plan(view, plan, pred_bitmaps, &mut matched_preds);
        }

        if let Some(log_attrs) = view.records.get(ArrowPayloadType::LogAttrs) {
            eval_attr_table_plans(
                log_attrs,
                &self.log_attr_plans,
                filter_only,
                transform_only,
                AttrParentTarget::LogRows(&view.log_id_to_row),
                pred_bitmaps,
                &mut matched_preds,
            );
        }

        if let Some(res_attrs) = view.records.get(ArrowPayloadType::ResourceAttrs) {
            eval_attr_table_plans(
                res_attrs,
                &self.res_attr_plans,
                filter_only,
                transform_only,
                AttrParentTarget::ParentBitmapMap(&view.resource_id_to_rows),
                pred_bitmaps,
                &mut matched_preds,
            );
        }

        if let Some(scope_attrs) = view.records.get(ArrowPayloadType::ScopeAttrs) {
            eval_attr_table_plans(
                scope_attrs,
                &self.scope_attr_plans,
                filter_only,
                transform_only,
                AttrParentTarget::ParentBitmapMap(&view.scope_id_to_rows),
                pred_bitmaps,
                &mut matched_preds,
            );
        }
    }

    fn compute_filter_keep_bitmap(
        &self,
        num_rows: usize,
        pred_bitmaps: &[RoaringBitmap],
    ) -> RoaringBitmap {
        let all_rows = RoaringBitmap::from_sorted_iter(0..num_rows as u32)
            .unwrap_or_else(|_| (0..num_rows as u32).collect());
        if self.filter_policies.is_empty() {
            return all_rows;
        }

        let mut drop_rows = RoaringBitmap::new();
        let mut keep_rows = RoaringBitmap::new();

        for fp in &self.filter_policies {
            if fp.matchers.is_empty() || fp.action == Action::Unspecified {
                continue;
            }
            let matched = eval_matcher_conjunction(&fp.matchers, &all_rows, pred_bitmaps);
            if matched.is_empty() {
                continue;
            }
            match fp.action {
                Action::Drop => drop_rows |= &matched,
                Action::Keep => keep_rows |= &matched,
                Action::Unspecified => {}
            }
        }

        if drop_rows.is_empty() {
            return all_rows;
        }
        drop_rows -= &keep_rows;
        let mut final_keep = all_rows;
        final_keep -= &drop_rows;
        final_keep
    }

    fn compute_transform_match_bitmaps(
        &self,
        num_rows: usize,
        pred_bitmaps: &[RoaringBitmap],
    ) -> Vec<RoaringBitmap> {
        let all_rows = RoaringBitmap::from_sorted_iter(0..num_rows as u32)
            .unwrap_or_else(|_| (0..num_rows as u32).collect());
        let mut out = Vec::with_capacity(self.transform_matcher_sets.len());
        for matchers in &self.transform_matcher_sets {
            if matchers.is_empty() {
                out.push(all_rows.clone());
            } else {
                out.push(eval_matcher_conjunction(matchers, &all_rows, pred_bitmaps));
            }
        }
        out
    }
}

#[inline]
fn eval_matcher_conjunction(
    matchers: &[CompiledMatcherRef],
    all_rows: &RoaringBitmap,
    pred_bitmaps: &[RoaringBitmap],
) -> RoaringBitmap {
    // Seed from the first positive matcher if available to avoid cloning `all_rows` unnecessarily.
    let mut acc = if let Some(first_pos_idx) = matchers.iter().position(|m| !m.negate) {
        let first_bm = &pred_bitmaps[matchers[first_pos_idx].pred_id];
        if first_bm.is_empty() {
            return RoaringBitmap::new();
        }
        let mut bm = first_bm.clone();
        for (idx, m) in matchers.iter().enumerate() {
            if idx == first_pos_idx {
                continue;
            }
            if m.negate {
                bm -= &pred_bitmaps[m.pred_id];
            } else {
                bm &= &pred_bitmaps[m.pred_id];
            }
            if bm.is_empty() {
                break;
            }
        }
        bm
    } else {
        let mut bm = all_rows.clone();
        for m in matchers {
            bm -= &pred_bitmaps[m.pred_id];
            if bm.is_empty() {
                break;
            }
        }
        bm
    };
    if acc.is_empty() {
        acc.clear();
    }
    acc
}

fn eval_root_field_plan(
    view: &LogsBatchView<'_>,
    plan: &CompiledFieldPlan,
    pred_bitmaps: &mut [RoaringBitmap],
    matched_preds: &mut Vec<usize>,
) {
    let num_rows = view.num_rows;
    match &plan.field {
        LogFieldSelector::RecordField(rf) => match rf {
            LogRecordField::Unspecified => {}
            LogRecordField::SeverityText => {
                if let Some(col) = view.root.column_by_name(consts::SEVERITY_TEXT) {
                    eval_string_column_with_dict_cache(
                        col,
                        num_rows,
                        true,
                        plan,
                        pred_bitmaps,
                        matched_preds,
                    );
                }
            }
            LogRecordField::EventName => {
                if let Some(col) = view.root.column_by_name(consts::EVENT_NAME) {
                    eval_string_column_with_dict_cache(
                        col,
                        num_rows,
                        true,
                        plan,
                        pred_bitmaps,
                        matched_preds,
                    );
                }
            }
            LogRecordField::SeverityNumber => {
                if let Some(col) = view.root.column_by_name(consts::SEVERITY_NUMBER) {
                    for row in 0..num_rows {
                        if let Some(v) = extract_i64_from_array(col, row)
                            && v != 0
                        {
                            matched_preds.clear();
                            plan.eval_int_value(v, matched_preds);
                            for &pid in matched_preds.iter() {
                                pred_bitmaps[pid].insert(row as u32);
                            }
                        }
                    }
                }
            }
            LogRecordField::TraceId => {
                if let Some(col) = view.root.column_by_name(consts::TRACE_ID) {
                    eval_hex_id_column(col, num_rows, plan, pred_bitmaps, matched_preds);
                }
            }
            LogRecordField::SpanId => {
                if let Some(col) = view.root.column_by_name(consts::SPAN_ID) {
                    eval_hex_id_column(col, num_rows, plan, pred_bitmaps, matched_preds);
                }
            }
            LogRecordField::Body => {
                if let Some(body_col) = view.root.column_by_name(consts::BODY) {
                    eval_body_column(body_col, num_rows, plan, pred_bitmaps, matched_preds);
                }
            }
        },
        LogFieldSelector::ScopeField(sf) => match sf {
            ScopeField::Unspecified => {}
            ScopeField::SchemaUrl => {
                if let Some(col) = view.root.column_by_name(consts::SCHEMA_URL) {
                    eval_string_column_with_dict_cache(
                        col,
                        num_rows,
                        true,
                        plan,
                        pred_bitmaps,
                        matched_preds,
                    );
                }
            }
            ScopeField::Name => {
                if let Some(scope_col) = view.root.column_by_name(consts::SCOPE)
                    && let Some(scope_struct) = scope_col.as_any().downcast_ref::<StructArray>()
                    && let Some(col) = scope_struct.column_by_name(consts::NAME)
                {
                    eval_string_column_with_dict_cache(
                        col,
                        num_rows,
                        true,
                        plan,
                        pred_bitmaps,
                        matched_preds,
                    );
                }
            }
            ScopeField::Version => {
                if let Some(scope_col) = view.root.column_by_name(consts::SCOPE)
                    && let Some(scope_struct) = scope_col.as_any().downcast_ref::<StructArray>()
                    && let Some(col) = scope_struct.column_by_name(consts::VERSION)
                {
                    eval_string_column_with_dict_cache(
                        col,
                        num_rows,
                        true,
                        plan,
                        pred_bitmaps,
                        matched_preds,
                    );
                }
            }
        },
        LogFieldSelector::BodyField(AttributePath { path }) => {
            if path.is_empty() {
                return;
            }
            let Some(body_col) = view.root.column_by_name(consts::BODY) else {
                return;
            };
            let Some(body_struct) = body_col.as_any().downcast_ref::<StructArray>() else {
                return;
            };
            for row in 0..num_rows {
                if let Some(val) = extract_anyval_from_struct(body_struct, row)
                    && let Some(nested) = navigate_extracted_path(val, path)
                {
                    matched_preds.clear();
                    plan.eval_extracted_value(&nested, matched_preds);
                    for &pid in matched_preds.iter() {
                        pred_bitmaps[pid].insert(row as u32);
                    }
                }
            }
        }
        LogFieldSelector::LogAttribute(_)
        | LogFieldSelector::ResourceAttribute(_)
        | LogFieldSelector::ScopeAttribute(_) => {}
    }
}

fn eval_string_column_with_dict_cache(
    col: &ArrayRef,
    num_rows: usize,
    empty_is_absent: bool,
    plan: &CompiledFieldPlan,
    pred_bitmaps: &mut [RoaringBitmap],
    matched_preds: &mut Vec<usize>,
) {
    if let Some(da) = col.as_any().downcast_ref::<DictionaryArray<UInt8Type>>() {
        let Some(vals) = da.values().as_any().downcast_ref::<StringArray>() else {
            return;
        };
        let dict_len = vals.len();
        let mut dict_matches: Vec<Option<Vec<usize>>> = vec![None; dict_len];
        for (k, slot) in dict_matches.iter_mut().enumerate().take(dict_len) {
            if !vals.is_valid(k) {
                continue;
            }
            let s = vals.value(k);
            if empty_is_absent && s.is_empty() {
                continue;
            }
            let mut m = Vec::new();
            if let Some(pid) = plan.exists_pred_id {
                m.push(pid);
            }
            if !plan.regex_db.is_empty() {
                plan.regex_db.evaluate_haystack(s, &mut m);
            }
            *slot = Some(m);
        }
        let keys = da.keys();
        for row in 0..num_rows {
            if !da.is_valid(row) {
                continue;
            }
            let k = keys.value(row) as usize;
            if let Some(Some(preds)) = dict_matches.get(k) {
                for &pid in preds {
                    pred_bitmaps[pid].insert(row as u32);
                }
            }
        }
        return;
    }

    if let Some(da) = col.as_any().downcast_ref::<DictionaryArray<UInt16Type>>() {
        let Some(vals) = da.values().as_any().downcast_ref::<StringArray>() else {
            return;
        };
        let dict_len = vals.len();
        let mut dict_matches: Vec<Option<Vec<usize>>> = vec![None; dict_len];
        for (k, slot) in dict_matches.iter_mut().enumerate().take(dict_len) {
            if !vals.is_valid(k) {
                continue;
            }
            let s = vals.value(k);
            if empty_is_absent && s.is_empty() {
                continue;
            }
            let mut m = Vec::new();
            if let Some(pid) = plan.exists_pred_id {
                m.push(pid);
            }
            if !plan.regex_db.is_empty() {
                plan.regex_db.evaluate_haystack(s, &mut m);
            }
            *slot = Some(m);
        }
        let keys = da.keys();
        for row in 0..num_rows {
            if !da.is_valid(row) {
                continue;
            }
            let k = keys.value(row) as usize;
            if let Some(Some(preds)) = dict_matches.get(k) {
                for &pid in preds {
                    pred_bitmaps[pid].insert(row as u32);
                }
            }
        }
        return;
    }

    if let Some(sa) = col.as_any().downcast_ref::<StringArray>() {
        for row in 0..num_rows {
            if !sa.is_valid(row) {
                continue;
            }
            let s = sa.value(row);
            if empty_is_absent && s.is_empty() {
                continue;
            }
            if let Some(pid) = plan.exists_pred_id {
                pred_bitmaps[pid].insert(row as u32);
            }
            if !plan.regex_db.is_empty() {
                matched_preds.clear();
                plan.regex_db.evaluate_haystack(s, matched_preds);
                for &pid in matched_preds.iter() {
                    pred_bitmaps[pid].insert(row as u32);
                }
            }
        }
    }
}

fn eval_hex_id_column(
    col: &ArrayRef,
    num_rows: usize,
    plan: &CompiledFieldPlan,
    pred_bitmaps: &mut [RoaringBitmap],
    matched_preds: &mut Vec<usize>,
) {
    if let Some(da) = col.as_any().downcast_ref::<DictionaryArray<UInt8Type>>()
        && let Some(vals) = da.values().as_any().downcast_ref::<FixedSizeBinaryArray>()
    {
        let dict_len = vals.len();
        let mut dict_matches: Vec<Option<Vec<usize>>> = vec![None; dict_len];
        for (k, slot) in dict_matches.iter_mut().enumerate().take(dict_len) {
            if !vals.is_valid(k) {
                continue;
            }
            let Some(hex_str) = format_non_zero_hex(vals.value(k)) else {
                continue;
            };
            let mut m = Vec::new();
            if let Some(pid) = plan.exists_pred_id {
                m.push(pid);
            }
            if !plan.regex_db.is_empty() {
                plan.regex_db.evaluate_haystack(&hex_str, &mut m);
            }
            *slot = Some(m);
        }
        let keys = da.keys();
        for row in 0..num_rows {
            if !da.is_valid(row) {
                continue;
            }
            let k = keys.value(row) as usize;
            if let Some(Some(preds)) = dict_matches.get(k) {
                for &pid in preds {
                    pred_bitmaps[pid].insert(row as u32);
                }
            }
        }
        return;
    }

    if let Some(da) = col.as_any().downcast_ref::<DictionaryArray<UInt16Type>>()
        && let Some(vals) = da.values().as_any().downcast_ref::<FixedSizeBinaryArray>()
    {
        let dict_len = vals.len();
        let mut dict_matches: Vec<Option<Vec<usize>>> = vec![None; dict_len];
        for (k, slot) in dict_matches.iter_mut().enumerate().take(dict_len) {
            if !vals.is_valid(k) {
                continue;
            }
            let Some(hex_str) = format_non_zero_hex(vals.value(k)) else {
                continue;
            };
            let mut m = Vec::new();
            if let Some(pid) = plan.exists_pred_id {
                m.push(pid);
            }
            if !plan.regex_db.is_empty() {
                plan.regex_db.evaluate_haystack(&hex_str, &mut m);
            }
            *slot = Some(m);
        }
        let keys = da.keys();
        for row in 0..num_rows {
            if !da.is_valid(row) {
                continue;
            }
            let k = keys.value(row) as usize;
            if let Some(Some(preds)) = dict_matches.get(k) {
                for &pid in preds {
                    pred_bitmaps[pid].insert(row as u32);
                }
            }
        }
        return;
    }

    for row in 0..num_rows {
        let Some(bytes) = extract_fixed_bytes_from_array(col, row) else {
            continue;
        };
        if bytes.iter().all(|&b| b == 0) {
            continue;
        }
        if let Some(pid) = plan.exists_pred_id {
            pred_bitmaps[pid].insert(row as u32);
        }
        if !plan.regex_db.is_empty()
            && let Some(hex_str) = format_non_zero_hex(bytes)
        {
            matched_preds.clear();
            plan.regex_db.evaluate_haystack(&hex_str, matched_preds);
            for &pid in matched_preds.iter() {
                pred_bitmaps[pid].insert(row as u32);
            }
        }
    }
}

fn eval_body_column(
    body_col: &ArrayRef,
    num_rows: usize,
    plan: &CompiledFieldPlan,
    pred_bitmaps: &mut [RoaringBitmap],
    matched_preds: &mut Vec<usize>,
) {
    let Some(body_struct) = body_col.as_any().downcast_ref::<StructArray>() else {
        eval_string_column_with_dict_cache(
            body_col,
            num_rows,
            false,
            plan,
            pred_bitmaps,
            matched_preds,
        );
        return;
    };

    let Some(type_col) = body_struct.column_by_name(consts::ATTRIBUTE_TYPE) else {
        return;
    };
    let Some(type_arr) = type_col.as_any().downcast_ref::<UInt8Array>() else {
        return;
    };
    let str_col = body_struct.column_by_name(consts::ATTRIBUTE_STR);

    // Pre-build lazy dictionary cache if `body.str` is dictionary-encoded
    let dict_u8 = str_col.and_then(|c| c.as_any().downcast_ref::<DictionaryArray<UInt8Type>>());
    let dict_u16 = str_col.and_then(|c| c.as_any().downcast_ref::<DictionaryArray<UInt16Type>>());
    let dict_vals_len = dict_u8
        .map(|d| d.values().len())
        .or_else(|| dict_u16.map(|d| d.values().len()))
        .unwrap_or(0);
    let mut str_dict_cache: Vec<Option<Vec<usize>>> = if dict_vals_len > 0 {
        vec![None; dict_vals_len]
    } else {
        Vec::new()
    };

    for row in 0..num_rows {
        if !body_struct.is_valid(row) || !type_arr.is_valid(row) {
            continue;
        }
        let Ok(vtype) = AttributeValueType::try_from(type_arr.value(row)) else {
            continue;
        };
        match vtype {
            AttributeValueType::Empty => continue,
            AttributeValueType::Str => {
                let Some(sc) = str_col else { continue };
                if !sc.is_valid(row) {
                    continue;
                }
                if let Some(da) = dict_u8 {
                    let k = da.keys().value(row) as usize;
                    if let Some(preds) = eval_or_get_str_dict_entry(
                        da.values(),
                        k,
                        plan,
                        &mut str_dict_cache,
                    ) {
                        for &pid in preds {
                            pred_bitmaps[pid].insert(row as u32);
                        }
                    }
                } else if let Some(da) = dict_u16 {
                    let k = da.keys().value(row) as usize;
                    if let Some(preds) = eval_or_get_str_dict_entry(
                        da.values(),
                        k,
                        plan,
                        &mut str_dict_cache,
                    ) {
                        for &pid in preds {
                            pred_bitmaps[pid].insert(row as u32);
                        }
                    }
                } else if let Some(s) = extract_str_from_array(sc, row) {
                    if let Some(pid) = plan.exists_pred_id {
                        pred_bitmaps[pid].insert(row as u32);
                    }
                    if !plan.regex_db.is_empty() {
                        matched_preds.clear();
                        plan.regex_db.evaluate_haystack(s, matched_preds);
                        for &pid in matched_preds.iter() {
                            pred_bitmaps[pid].insert(row as u32);
                        }
                    }
                }
            }
            _ => {
                if let Some(val) = extract_anyval_from_struct(body_struct, row) {
                    matched_preds.clear();
                    plan.eval_extracted_value(&val, matched_preds);
                    for &pid in matched_preds.iter() {
                        pred_bitmaps[pid].insert(row as u32);
                    }
                }
            }
        }
    }
}

#[inline]
fn eval_or_get_str_dict_entry<'a>(
    dict_values: &ArrayRef,
    k: usize,
    plan: &CompiledFieldPlan,
    cache: &'a mut [Option<Vec<usize>>],
) -> Option<&'a [usize]> {
    if k >= cache.len() {
        return None;
    }
    if cache[k].is_none() {
        let sa = dict_values.as_any().downcast_ref::<StringArray>()?;
        if !sa.is_valid(k) {
            return None;
        }
        let s = sa.value(k);
        let mut m = Vec::new();
        if let Some(pid) = plan.exists_pred_id {
            m.push(pid);
        }
        if !plan.regex_db.is_empty() {
            plan.regex_db.evaluate_haystack(s, &mut m);
        }
        cache[k] = Some(m);
    }
    cache[k].as_deref()
}

enum AttrParentTarget<'a> {
    LogRows(&'a [u32]),
    ParentBitmapMap(&'a AHashMap<u16, RoaringBitmap>),
}

impl<'a> AttrParentTarget<'a> {
    #[inline]
    fn record_matches(&self, pid: u16, preds: &[usize], pred_bitmaps: &mut [RoaringBitmap]) {
        match self {
            Self::LogRows(log_id_to_row) => {
                if let Some(&row) = log_id_to_row.get(pid as usize)
                    && row != u32::MAX
                {
                    for &pred_id in preds {
                        pred_bitmaps[pred_id].insert(row);
                    }
                }
            }
            Self::ParentBitmapMap(map) => {
                if let Some(rows_bm) = map.get(&pid) {
                    for &pred_id in preds {
                        pred_bitmaps[pred_id] |= rows_bm;
                    }
                }
            }
        }
    }
}

struct ActiveAttrPlan<'a> {
    plan: &'a CompiledFieldPlan,
    rest_path: &'a [String],
    str_dict_cache: Vec<Option<Vec<usize>>>,
}

fn eval_attr_table_plans(
    rb: &RecordBatch,
    plans: &[CompiledFieldPlan],
    filter_only: bool,
    transform_only: bool,
    target: AttrParentTarget<'_>,
    pred_bitmaps: &mut [RoaringBitmap],
    matched_preds: &mut Vec<usize>,
) {
    let num_attr_rows = rb.num_rows();
    if num_attr_rows == 0 || plans.is_empty() {
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
    let str_dict_u8 = str_col.and_then(|c| c.as_any().downcast_ref::<DictionaryArray<UInt8Type>>());
    let str_dict_u16 =
        str_col.and_then(|c| c.as_any().downcast_ref::<DictionaryArray<UInt16Type>>());
    let str_dict_len = str_dict_u8
        .map(|d| d.values().len())
        .or_else(|| str_dict_u16.map(|d| d.values().len()))
        .unwrap_or(0);

    let mut active_plans: Vec<ActiveAttrPlan<'_>> = Vec::new();
    let mut key_to_plan_indices: AHashMap<&str, Vec<usize>> = AHashMap::new();

    for plan in plans {
        if !plan.matches_mode(filter_only, transform_only) {
            continue;
        }
        let path = match &plan.field {
            LogFieldSelector::LogAttribute(p)
            | LogFieldSelector::ResourceAttribute(p)
            | LogFieldSelector::ScopeAttribute(p) => &p.path,
            _ => continue,
        };
        let Some(first_key) = path.first() else {
            continue;
        };
        let rest_path = &path[1..];
        let idx = active_plans.len();
        let cache_len = if rest_path.is_empty() {
            str_dict_len
        } else {
            0
        };
        active_plans.push(ActiveAttrPlan {
            plan,
            rest_path,
            str_dict_cache: vec![None; cache_len],
        });
        key_to_plan_indices
            .entry(first_key.as_str())
            .or_default()
            .push(idx);
    }

    if active_plans.is_empty() {
        return;
    }

    // Pre-index dictionary keys for ATTRIBUTE_KEY if dictionary-encoded
    let key_dict_u8 = key_col.as_any().downcast_ref::<DictionaryArray<UInt8Type>>();
    let key_dict_u16 = key_col.as_any().downcast_ref::<DictionaryArray<UInt16Type>>();

    let key_dict_lookup: Vec<Option<Vec<usize>>> = if let Some(da) = key_dict_u8 {
        build_key_dict_lookup(da.values(), &key_to_plan_indices)
    } else if let Some(da) = key_dict_u16 {
        build_key_dict_lookup(da.values(), &key_to_plan_indices)
    } else {
        Vec::new()
    };

    for attr_row in 0..num_attr_rows {
        let plan_indices_opt: Option<&[usize]> = if let Some(da) = key_dict_u8 {
            if !da.is_valid(attr_row) {
                continue;
            }
            let k = da.keys().value(attr_row) as usize;
            key_dict_lookup.get(k).and_then(Option::as_deref)
        } else if let Some(da) = key_dict_u16 {
            if !da.is_valid(attr_row) {
                continue;
            }
            let k = da.keys().value(attr_row) as usize;
            key_dict_lookup.get(k).and_then(Option::as_deref)
        } else {
            let Some(k_str) = extract_str_from_array(key_col, attr_row) else {
                continue;
            };
            key_to_plan_indices.get(k_str).map(Vec::as_slice)
        };

        let Some(plan_indices) = plan_indices_opt else {
            continue;
        };
        if !type_arr.is_valid(attr_row) {
            continue;
        }
        let Ok(vtype) = AttributeValueType::try_from(type_arr.value(attr_row)) else {
            continue;
        };
        if vtype == AttributeValueType::Empty {
            continue;
        }
        let Some(pid) = extract_parent_id_u16(rb, attr_row) else {
            continue;
        };

        for &plan_idx in plan_indices {
            let active = &mut active_plans[plan_idx];
            if active.rest_path.is_empty() && vtype == AttributeValueType::Str {
                let Some(sc) = str_col else { continue };
                if !sc.is_valid(attr_row) {
                    continue;
                }
                if let Some(da) = str_dict_u8 {
                    let k = da.keys().value(attr_row) as usize;
                    if let Some(preds) = eval_or_get_str_dict_entry(
                        da.values(),
                        k,
                        active.plan,
                        &mut active.str_dict_cache,
                    ) {
                        target.record_matches(pid, preds, pred_bitmaps);
                    }
                } else if let Some(da) = str_dict_u16 {
                    let k = da.keys().value(attr_row) as usize;
                    if let Some(preds) = eval_or_get_str_dict_entry(
                        da.values(),
                        k,
                        active.plan,
                        &mut active.str_dict_cache,
                    ) {
                        target.record_matches(pid, preds, pred_bitmaps);
                    }
                } else if let Some(s) = extract_str_from_array(sc, attr_row) {
                    matched_preds.clear();
                    if let Some(exists_id) = active.plan.exists_pred_id {
                        matched_preds.push(exists_id);
                    }
                    if !active.plan.regex_db.is_empty() {
                        active.plan.regex_db.evaluate_haystack(s, matched_preds);
                    }
                    target.record_matches(pid, matched_preds, pred_bitmaps);
                }
            } else if let Some(val) = extract_anyval_from_attr_batch(rb, attr_row) {
                if active.rest_path.is_empty() {
                    matched_preds.clear();
                    active.plan.eval_extracted_value(&val, matched_preds);
                    target.record_matches(pid, matched_preds, pred_bitmaps);
                } else if let Some(nested) = navigate_extracted_path(val, active.rest_path) {
                    matched_preds.clear();
                    active.plan.eval_extracted_value(&nested, matched_preds);
                    target.record_matches(pid, matched_preds, pred_bitmaps);
                }
            }
        }
    }
}

fn build_key_dict_lookup(
    dict_values: &ArrayRef,
    key_to_plan_indices: &AHashMap<&str, Vec<usize>>,
) -> Vec<Option<Vec<usize>>> {
    let Some(sa) = dict_values.as_any().downcast_ref::<StringArray>() else {
        return Vec::new();
    };
    let len = sa.len();
    let mut out = Vec::with_capacity(len);
    for k in 0..len {
        if sa.is_valid(k)
            && let Some(indices) = key_to_plan_indices.get(sa.value(k))
        {
            out.push(Some(indices.clone()));
        } else {
            out.push(None);
        }
    }
    out
}

impl PolicyEvaluationEngine for ShardedRegexAutomataEngine {
    fn name(&self) -> &'static str {
        "Candidate2_Sharded_RegexAutomata"
    }

    fn evaluate_filter(&self, batch: &OtapArrowRecords) -> Result<RoaringBitmap, PolicyError> {
        let Some(view) = LogsBatchView::try_new(batch)? else {
            return Ok(RoaringBitmap::new());
        };
        if self.filter_policies.is_empty() {
            return Ok(
                RoaringBitmap::from_sorted_iter(0..view.num_rows as u32)
                    .unwrap_or_else(|_| (0..view.num_rows as u32).collect()),
            );
        }
        let mut pred_bitmaps = vec![RoaringBitmap::new(); self.num_predicates];
        self.evaluate_predicates_for_view(&view, &mut pred_bitmaps, true, false);
        Ok(self.compute_filter_keep_bitmap(view.num_rows, &pred_bitmaps))
    }

    fn evaluate(&self, mut batch: OtapArrowRecords) -> Result<EvaluationOutput, PolicyError> {
        let Some(root) = batch.get(ArrowPayloadType::Logs) else {
            return Ok(EvaluationOutput {
                records: Some(batch),
                kept_rows: 0,
                dropped_rows: 0,
                transform_hits: 0,
            });
        };
        let num_rows = root.num_rows();
        if num_rows == 0 {
            return Ok(EvaluationOutput {
                records: Some(batch),
                kept_rows: 0,
                dropped_rows: 0,
                transform_hits: 0,
            });
        }

        // Fast path when neither filter policies nor transform policies are active.
        if self.filter_policies.is_empty() && self.transforms.is_empty() {
            return Ok(EvaluationOutput {
                records: Some(batch),
                kept_rows: num_rows as u64,
                dropped_rows: 0,
                transform_hits: 0,
            });
        }

        // Case 1: Transform-only evaluation.
        if self.filter_policies.is_empty() {
            let mut pred_bitmaps = vec![RoaringBitmap::new(); self.num_predicates];
            let policy_matches = {
                let Some(view) = LogsBatchView::try_new(&batch)? else {
                    return Ok(EvaluationOutput {
                        records: None,
                        kept_rows: 0,
                        dropped_rows: 0,
                        transform_hits: 0,
                    });
                };
                self.evaluate_predicates_for_view(&view, &mut pred_bitmaps, false, true);
                self.compute_transform_match_bitmaps(view.num_rows, &pred_bitmaps)
            };
            let transform_hits =
                apply_spec_transforms_masked(&mut batch, &self.transforms, &policy_matches)?;
            return Ok(EvaluationOutput {
                records: Some(batch),
                kept_rows: num_rows as u64,
                dropped_rows: 0,
                transform_hits,
            });
        }

        // Case 2: Filter policies exist (with or without transform policies).
        let mut pred_bitmaps = vec![RoaringBitmap::new(); self.num_predicates];
        let keep_bitmap = {
            let Some(view) = LogsBatchView::try_new(&batch)? else {
                return Ok(EvaluationOutput {
                    records: None,
                    kept_rows: 0,
                    dropped_rows: 0,
                    transform_hits: 0,
                });
            };
            self.evaluate_predicates_for_view(&view, &mut pred_bitmaps, true, false);
            self.compute_filter_keep_bitmap(view.num_rows, &pred_bitmaps)
        };

        let kept_rows = keep_bitmap.len();
        let dropped_rows = (num_rows as u64).saturating_sub(kept_rows);
        if kept_rows == 0 {
            return Ok(EvaluationOutput {
                records: None,
                kept_rows: 0,
                dropped_rows,
                transform_hits: 0,
            });
        }

        if kept_rows < num_rows as u64 {
            let mut pool = IdBitmapPool::new();
            batch = filter_otap_logs_by_bitmap(&batch, &keep_bitmap, num_rows, &mut pool)?;
        }

        if self.transforms.is_empty() {
            return Ok(EvaluationOutput {
                records: Some(batch),
                kept_rows,
                dropped_rows,
                transform_hits: 0,
            });
        }

        let policy_matches = if kept_rows == num_rows as u64 {
            // No rows were dropped: evaluate only transform-exclusive field plans and reuse shared bitmaps.
            if let Some(view) = LogsBatchView::try_new(&batch)? {
                let mut matched_preds = Vec::with_capacity(8);
                for plan in &self.root_field_plans {
                    if plan.used_in_transform && !plan.used_in_filter {
                        eval_root_field_plan(&view, plan, &mut pred_bitmaps, &mut matched_preds);
                    }
                }
                if let Some(log_attrs) = view.records.get(ArrowPayloadType::LogAttrs) {
                    let tx_only: Vec<_> = self
                        .log_attr_plans
                        .iter()
                        .filter(|p| p.used_in_transform && !p.used_in_filter)
                        .collect();
                    if !tx_only.is_empty() {
                        eval_attr_table_plans_refs(
                            log_attrs,
                            &tx_only,
                            AttrParentTarget::LogRows(&view.log_id_to_row),
                            &mut pred_bitmaps,
                            &mut matched_preds,
                        );
                    }
                }
                if let Some(res_attrs) = view.records.get(ArrowPayloadType::ResourceAttrs) {
                    let tx_only: Vec<_> = self
                        .res_attr_plans
                        .iter()
                        .filter(|p| p.used_in_transform && !p.used_in_filter)
                        .collect();
                    if !tx_only.is_empty() {
                        eval_attr_table_plans_refs(
                            res_attrs,
                            &tx_only,
                            AttrParentTarget::ParentBitmapMap(&view.resource_id_to_rows),
                            &mut pred_bitmaps,
                            &mut matched_preds,
                        );
                    }
                }
                if let Some(scope_attrs) = view.records.get(ArrowPayloadType::ScopeAttrs) {
                    let tx_only: Vec<_> = self
                        .scope_attr_plans
                        .iter()
                        .filter(|p| p.used_in_transform && !p.used_in_filter)
                        .collect();
                    if !tx_only.is_empty() {
                        eval_attr_table_plans_refs(
                            scope_attrs,
                            &tx_only,
                            AttrParentTarget::ParentBitmapMap(&view.scope_id_to_rows),
                            &mut pred_bitmaps,
                            &mut matched_preds,
                        );
                    }
                }
            }
            self.compute_transform_match_bitmaps(num_rows, &pred_bitmaps)
        } else {
            let mut tx_pred_bitmaps = vec![RoaringBitmap::new(); self.num_predicates];
            let Some(view) = LogsBatchView::try_new(&batch)? else {
                return Ok(EvaluationOutput {
                    records: Some(batch),
                    kept_rows,
                    dropped_rows,
                    transform_hits: 0,
                });
            };
            self.evaluate_predicates_for_view(&view, &mut tx_pred_bitmaps, false, true);
            self.compute_transform_match_bitmaps(view.num_rows, &tx_pred_bitmaps)
        };

        let transform_hits =
            apply_spec_transforms_masked(&mut batch, &self.transforms, &policy_matches)?;

        Ok(EvaluationOutput {
            records: Some(batch),
            kept_rows,
            dropped_rows,
            transform_hits,
        })
    }

    fn estimated_heap_bytes(&self) -> usize {
        let plans_bytes: usize = self
            .root_field_plans
            .iter()
            .chain(self.log_attr_plans.iter())
            .chain(self.res_attr_plans.iter())
            .chain(self.scope_attr_plans.iter())
            .map(CompiledFieldPlan::estimated_heap_bytes)
            .sum();
        let filter_bytes: usize = self
            .filter_policies
            .iter()
            .map(|fp| {
                size_of::<CompiledFilterPolicy>()
                    + fp.matchers.capacity() * size_of::<CompiledMatcherRef>()
            })
            .sum();
        let tx_matcher_bytes: usize = self
            .transform_matcher_sets
            .iter()
            .map(|ms| size_of::<Vec<CompiledMatcherRef>>() + ms.capacity() * size_of::<CompiledMatcherRef>())
            .sum();
        let tx_bytes: usize = self
            .transforms
            .iter()
            .map(|tx| {
                size_of::<CompiledTransformPolicy>()
                    + tx.policy_id.capacity()
                    + tx.rename.capacity() * size_of::<crate::model::LogRename>()
                    + tx.add.capacity() * size_of::<crate::model::LogAdd>()
                    + tx.remove.capacity() * size_of::<crate::model::LogRemove>()
                    + tx.redact.capacity() * size_of::<crate::otap_ops::CompiledLogRedact>()
            })
            .sum();
        size_of::<Self>() + plans_bytes + filter_bytes + tx_matcher_bytes + tx_bytes
    }
}

fn eval_attr_table_plans_refs(
    rb: &RecordBatch,
    plans: &[&CompiledFieldPlan],
    target: AttrParentTarget<'_>,
    pred_bitmaps: &mut [RoaringBitmap],
    matched_preds: &mut Vec<usize>,
) {
    let num_attr_rows = rb.num_rows();
    if num_attr_rows == 0 || plans.is_empty() {
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
    let str_dict_u8 = str_col.and_then(|c| c.as_any().downcast_ref::<DictionaryArray<UInt8Type>>());
    let str_dict_u16 =
        str_col.and_then(|c| c.as_any().downcast_ref::<DictionaryArray<UInt16Type>>());
    let str_dict_len = str_dict_u8
        .map(|d| d.values().len())
        .or_else(|| str_dict_u16.map(|d| d.values().len()))
        .unwrap_or(0);

    let mut active_plans: Vec<ActiveAttrPlan<'_>> = Vec::with_capacity(plans.len());
    let mut key_to_plan_indices: AHashMap<&str, Vec<usize>> = AHashMap::new();

    for &plan in plans {
        let path = match &plan.field {
            LogFieldSelector::LogAttribute(p)
            | LogFieldSelector::ResourceAttribute(p)
            | LogFieldSelector::ScopeAttribute(p) => &p.path,
            _ => continue,
        };
        let Some(first_key) = path.first() else {
            continue;
        };
        let rest_path = &path[1..];
        let idx = active_plans.len();
        let cache_len = if rest_path.is_empty() {
            str_dict_len
        } else {
            0
        };
        active_plans.push(ActiveAttrPlan {
            plan,
            rest_path,
            str_dict_cache: vec![None; cache_len],
        });
        key_to_plan_indices
            .entry(first_key.as_str())
            .or_default()
            .push(idx);
    }

    if active_plans.is_empty() {
        return;
    }

    let key_dict_u8 = key_col.as_any().downcast_ref::<DictionaryArray<UInt8Type>>();
    let key_dict_u16 = key_col.as_any().downcast_ref::<DictionaryArray<UInt16Type>>();

    let key_dict_lookup: Vec<Option<Vec<usize>>> = if let Some(da) = key_dict_u8 {
        build_key_dict_lookup(da.values(), &key_to_plan_indices)
    } else if let Some(da) = key_dict_u16 {
        build_key_dict_lookup(da.values(), &key_to_plan_indices)
    } else {
        Vec::new()
    };

    for attr_row in 0..num_attr_rows {
        let plan_indices_opt: Option<&[usize]> = if let Some(da) = key_dict_u8 {
            if !da.is_valid(attr_row) {
                continue;
            }
            let k = da.keys().value(attr_row) as usize;
            key_dict_lookup.get(k).and_then(Option::as_deref)
        } else if let Some(da) = key_dict_u16 {
            if !da.is_valid(attr_row) {
                continue;
            }
            let k = da.keys().value(attr_row) as usize;
            key_dict_lookup.get(k).and_then(Option::as_deref)
        } else {
            let Some(k_str) = extract_str_from_array(key_col, attr_row) else {
                continue;
            };
            key_to_plan_indices.get(k_str).map(Vec::as_slice)
        };

        let Some(plan_indices) = plan_indices_opt else {
            continue;
        };
        if !type_arr.is_valid(attr_row) {
            continue;
        }
        let Ok(vtype) = AttributeValueType::try_from(type_arr.value(attr_row)) else {
            continue;
        };
        if vtype == AttributeValueType::Empty {
            continue;
        }
        let Some(pid) = extract_parent_id_u16(rb, attr_row) else {
            continue;
        };

        for &plan_idx in plan_indices {
            let active = &mut active_plans[plan_idx];
            if active.rest_path.is_empty() && vtype == AttributeValueType::Str {
                let Some(sc) = str_col else { continue };
                if !sc.is_valid(attr_row) {
                    continue;
                }
                if let Some(da) = str_dict_u8 {
                    let k = da.keys().value(attr_row) as usize;
                    if let Some(preds) = eval_or_get_str_dict_entry(
                        da.values(),
                        k,
                        active.plan,
                        &mut active.str_dict_cache,
                    ) {
                        target.record_matches(pid, preds, pred_bitmaps);
                    }
                } else if let Some(da) = str_dict_u16 {
                    let k = da.keys().value(attr_row) as usize;
                    if let Some(preds) = eval_or_get_str_dict_entry(
                        da.values(),
                        k,
                        active.plan,
                        &mut active.str_dict_cache,
                    ) {
                        target.record_matches(pid, preds, pred_bitmaps);
                    }
                } else if let Some(s) = extract_str_from_array(sc, attr_row) {
                    matched_preds.clear();
                    if let Some(exists_id) = active.plan.exists_pred_id {
                        matched_preds.push(exists_id);
                    }
                    if !active.plan.regex_db.is_empty() {
                        active.plan.regex_db.evaluate_haystack(s, matched_preds);
                    }
                    target.record_matches(pid, matched_preds, pred_bitmaps);
                }
            } else if let Some(val) = extract_anyval_from_attr_batch(rb, attr_row) {
                if active.rest_path.is_empty() {
                    matched_preds.clear();
                    active.plan.eval_extracted_value(&val, matched_preds);
                    target.record_matches(pid, matched_preds, pred_bitmaps);
                } else if let Some(nested) = navigate_extracted_path(val, active.rest_path) {
                    matched_preds.clear();
                    active.plan.eval_extracted_value(&nested, matched_preds);
                    target.record_matches(pid, matched_preds, pred_bitmaps);
                }
            }
        }
    }
}
