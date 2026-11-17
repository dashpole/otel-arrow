// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Candidate 4 (`CompiledOplPolicyEngine`) and Baseline B (`IdiomaticOplTextEngine`).
//!
//! Translates `Policy` definitions (`LogFilterPolicy` and `LogTransformPolicy`) into
//! `otel-arrow-dfe-query-engine` (`OplParser` / `Pipeline`) stages to measure native OPL
//! DataFusion expression-tree evaluation throughput and its crossover point against
//! dedicated multi-pattern policy engines.

use arrow::array::{Array, UInt16Array};
use otel_arrow_contrib_data_engine_parser_abstractions::Parser;
use otel_arrow_dfe_pdata::otap::OtapArrowRecords;
use otel_arrow_dfe_pdata::otap::filter::IdBitmapPool;
use otel_arrow_dfe_pdata::proto::opentelemetry::arrow::v1::ArrowPayloadType;
use otel_arrow_dfe_pdata::schema::consts;
use otel_arrow_dfe_query_engine::parser::default_parser_options;
use otel_arrow_dfe_query_engine::pipeline::Pipeline;
use otel_arrow_dfe_query_engine::pipeline::state::ExecutionState;
use otel_arrow_dfe_query_engine_languages::opl::parser::OplParser;
use parking_lot::Mutex;
use roaring::RoaringBitmap;

use crate::candidate1_hybrid::HybridPolicyEngine;
use crate::model::{
    Action, AttributePath, LogFieldSelector, LogFilterPolicy, LogMatcher, LogRecordField,
    NumericValue, Policy, PolicyDefinition, PolicyError, Predicate, ScalarValue, ScopeField,
};
use crate::otap_ops::{LogsBatchView, filter_otap_logs_by_bitmap};
use crate::{EvaluationOutput, PolicyEvaluationEngine};

struct OplRuntimePipeline {
    query_text: String,
    pipeline: Mutex<(Pipeline, ExecutionState)>,
}

impl std::fmt::Debug for OplRuntimePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OplRuntimePipeline")
            .field("query_text", &self.query_text)
            .finish()
    }
}

impl OplRuntimePipeline {
    fn compile(query_text: String) -> Result<Self, PolicyError> {
        let parsed = OplParser::parse_with_options(&query_text, default_parser_options())
            .map_err(|e| PolicyError::Compile(format!("OPL parse error on '{query_text}': {e:?}")))?;
        let pipeline = Pipeline::try_new(parsed.pipeline)
            .map_err(|e| PolicyError::Compile(format!("OPL pipeline build error: {e}")))?;
        Ok(Self {
            query_text,
            pipeline: Mutex::new((pipeline, ExecutionState::default())),
        })
    }

    fn execute_matched_rows(
        &self,
        rt: &tokio::runtime::Runtime,
        batch: &OtapArrowRecords,
        view: &LogsBatchView<'_>,
    ) -> Result<RoaringBitmap, PolicyError> {
        let mut guard = self.pipeline.lock();
        let (ref mut pipeline, ref mut exec_state) = *guard;
        let out_batch = rt
            .block_on(pipeline.execute_with_state(batch.clone(), exec_state))
            .map_err(|e| PolicyError::Evaluation(format!("OPL execution error: {e}")))?;

        let mut matched_rows = RoaringBitmap::new();
        let Some(out_root) = out_batch.get(ArrowPayloadType::Logs) else {
            return Ok(matched_rows);
        };
        if out_root.num_rows() == 0 {
            return Ok(matched_rows);
        }
        if let Some(id_col) = out_root.column_by_name(consts::ID)
            && let Some(id_arr) = id_col.as_any().downcast_ref::<UInt16Array>()
        {
            for r in 0..out_root.num_rows() {
                if id_arr.is_valid(r) {
                    let lid = id_arr.value(r) as usize;
                    if let Some(&orig_row) = view.log_id_to_row.get(lid)
                        && orig_row != u32::MAX
                    {
                        matched_rows.insert(orig_row);
                    }
                }
            }
        }
        Ok(matched_rows)
    }
}

fn escape_opl_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn format_opl_field(field: &LogFieldSelector) -> Option<String> {
    match field {
        LogFieldSelector::RecordField(rf) => match rf {
            LogRecordField::Body => Some("body".to_owned()),
            LogRecordField::SeverityText => Some("severity_text".to_owned()),
            LogRecordField::SeverityNumber => Some("severity_number".to_owned()),
            LogRecordField::EventName => Some("event_name".to_owned()),
            // TraceId and SpanId are FixedSizeBinary(16/8) in OTAP; OPL does not auto-hex-encode them for string ops
            LogRecordField::TraceId | LogRecordField::SpanId | LogRecordField::Unspecified => None,
        },
        LogFieldSelector::ScopeField(sf) => match sf {
            ScopeField::Name => Some("instrumentation_scope.name".to_owned()),
            ScopeField::Version => Some("instrumentation_scope.version".to_owned()),
            ScopeField::SchemaUrl => Some("schema_url".to_owned()),
            ScopeField::Unspecified => None,
        },
        LogFieldSelector::LogAttribute(AttributePath { path }) if path.len() == 1 => {
            Some(format!("attributes[{}]", escape_opl_string(&path[0])))
        }
        LogFieldSelector::ResourceAttribute(AttributePath { path }) if path.len() == 1 => Some(
            format!("resource.attributes[{}]", escape_opl_string(&path[0])),
        ),
        LogFieldSelector::ScopeAttribute(AttributePath { path }) if path.len() == 1 => Some(
            format!(
                "instrumentation_scope.attributes[{}]",
                escape_opl_string(&path[0])
            ),
        ),
        _ => None,
    }
}

fn format_opl_matcher(m: &LogMatcher) -> Option<String> {
    // Negated matchers on optional attributes have SQL 3-valued NULL semantics in OPL, so only compile
    // positive matchers directly into OPL `where` clauses.
    if m.negate {
        return None;
    }
    let field_expr = format_opl_field(&m.field)?;
    let expr = match &m.predicate {
        Predicate::Equals(scalar) => match scalar {
            ScalarValue::String(s) => format!("({field_expr} == {})", escape_opl_string(s)),
            ScalarValue::Int(i) => format!("({field_expr} == {i})"),
            ScalarValue::Double(d) => {
                if d.is_nan() || d.is_infinite() || d.abs() >= 9_007_199_254_740_992.0 {
                    return None;
                }
                format!("({field_expr} == {d:?})")
            }
            ScalarValue::Bool(b) => format!("({field_expr} == {b})"),
            ScalarValue::Bytes(_) => return None,
        },
        Predicate::Gt(nv) | Predicate::Gte(nv) | Predicate::Lt(nv) | Predicate::Lte(nv) => {
            let op = match &m.predicate {
                Predicate::Gt(_) => ">",
                Predicate::Gte(_) => ">=",
                Predicate::Lt(_) => "<",
                Predicate::Lte(_) => "<=",
                _ => unreachable!(),
            };
            match nv {
                NumericValue::Int(i) => {
                    if i.unsigned_abs() >= 9_007_199_254_740_992 {
                        return None;
                    }
                    format!("({field_expr} {op} {i})")
                }
                NumericValue::Double(d) => {
                    if d.is_nan() || d.is_infinite() || d.abs() >= 9_007_199_254_740_992.0 {
                        return None;
                    }
                    format!("({field_expr} {op} {d:?})")
                }
            }
        }
        Predicate::Contains(ScalarValue::String(s)) => {
            if s.is_empty() {
                return None;
            }
            // In OPL, `contains` on `LogAttribute` or `RecordField` is supported for flat string columns
            if matches!(m.field, LogFieldSelector::LogAttribute(_)) {
                // OPL supports `contains` on attributes, but if an attribute is CBOR Slice, fallback handles it
                return None;
            }
            format!("contains({field_expr}, {})", escape_opl_string(s))
        }
        Predicate::Regex(pat) => {
            // If pattern uses `$` without `(?m)`, DataFusion's regex engine matches standard Rust regex,
            // but reject if pattern is empty
            if pat.is_empty() {
                return None;
            }
            format!("matches({field_expr}, r{})", escape_opl_string(pat))
        }
        _ => return None,
    };
    Some(expr)
}

fn join_balanced_or(exprs: &[String]) -> String {
    match exprs.len() {
        0 => String::new(),
        1 => exprs[0].clone(),
        2 => format!("({} or {})", exprs[0], exprs[1]),
        n => {
            let mid = n / 2;
            let left = join_balanced_or(&exprs[..mid]);
            let right = join_balanced_or(&exprs[mid..]);
            format!("({left} or {right})")
        }
    }
}

fn format_opl_filter_policy(lf: &LogFilterPolicy) -> Option<String> {
    if lf.matches.is_empty() {
        return None;
    }
    let mut parts = Vec::with_capacity(lf.matches.len());
    for m in &lf.matches {
        parts.push(format_opl_matcher(m)?);
    }
    if parts.len() == 1 {
        Some(parts.pop().unwrap())
    } else {
        Some(format!("({})", parts.join(" and ")))
    }
}

/// Candidate 4: Compiled OPL / `otel-arrow-dfe-query-engine` Pipeline Engine.
#[derive(Debug)]
pub struct CompiledOplPolicyEngine {
    rt: tokio::runtime::Runtime,
    opl_drop_pipeline: Option<OplRuntimePipeline>,
    opl_keep_pipeline: Option<OplRuntimePipeline>,
    fallback_engine: Option<HybridPolicyEngine>,
    transforms_engine: Option<HybridPolicyEngine>,
    has_transforms: bool,
    heap_bytes: usize,
}

impl CompiledOplPolicyEngine {
    /// Compiles `policies` into an OPL `Pipeline` backed by `otel-arrow-dfe-query-engine`.
    pub fn compile(policies: &[Policy]) -> Result<Self, PolicyError> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .map_err(|e| PolicyError::Compile(format!("tokio runtime error: {e}")))?;

        let mut drop_exprs = Vec::new();
        let mut keep_exprs = Vec::new();
        let mut needs_fallback = false;
        let mut has_transforms = false;

        // OPL's recursive-descent parser and DataFusion planner stack-overflow or degrade above 256 boolean branches
        let enabled_filters: Vec<&LogFilterPolicy> = policies
            .iter()
            .filter(|p| p.enabled)
            .filter_map(|p| match &p.definition {
                PolicyDefinition::LogFilter(lf) if lf.action != Action::Unspecified => Some(lf),
                PolicyDefinition::LogTransform(_) => {
                    has_transforms = true;
                    None
                }
                _ => None,
            })
            .collect();

        if enabled_filters.len() <= 1024 {
            for lf in &enabled_filters {
                if let Some(expr) = format_opl_filter_policy(lf) {
                    match lf.action {
                        Action::Drop => drop_exprs.push(expr),
                        Action::Keep => keep_exprs.push(expr),
                        Action::Unspecified => {}
                    }
                } else {
                    needs_fallback = true;
                    break;
                }
            }
        } else {
            needs_fallback = true;
        }

        let (opl_drop_pipeline, opl_keep_pipeline) = if !needs_fallback {
            let drop_pipe = if drop_exprs.is_empty() {
                None
            } else {
                let q = format!("logs | where {}", join_balanced_or(&drop_exprs));
                match OplRuntimePipeline::compile(q) {
                    Ok(p) => Some(p),
                    Err(_) => {
                        needs_fallback = true;
                        None
                    }
                }
            };
            let keep_pipe = if !needs_fallback && !keep_exprs.is_empty() {
                let q = format!("logs | where {}", join_balanced_or(&keep_exprs));
                match OplRuntimePipeline::compile(q) {
                    Ok(p) => Some(p),
                    Err(_) => {
                        needs_fallback = true;
                        None
                    }
                }
            } else {
                None
            };
            (drop_pipe, keep_pipe)
        } else {
            (None, None)
        };

        let fallback_engine = if needs_fallback {
            Some(HybridPolicyEngine::compile(policies)?)
        } else {
            None
        };
        let transforms_engine = if has_transforms {
            let tx_only: Vec<Policy> = policies
                .iter()
                .filter(|p| p.enabled && matches!(p.definition, PolicyDefinition::LogTransform(_)))
                .cloned()
                .collect();
            Some(HybridPolicyEngine::compile(&tx_only)?)
        } else {
            None
        };

        let heap_bytes = opl_drop_pipeline.as_ref().map_or(0, |p| p.query_text.len() * 16 + 4096)
            + opl_keep_pipeline.as_ref().map_or(0, |p| p.query_text.len() * 16 + 4096)
            + fallback_engine.as_ref().map_or(0, PolicyEvaluationEngine::estimated_heap_bytes);

        Ok(Self {
            rt,
            opl_drop_pipeline,
            opl_keep_pipeline,
            fallback_engine,
            transforms_engine,
            has_transforms,
            heap_bytes,
        })
    }
}

impl PolicyEvaluationEngine for CompiledOplPolicyEngine {
    fn name(&self) -> &'static str {
        "Candidate4_Compiled_OPL_Pipeline"
    }

    fn evaluate_filter(&self, batch: &OtapArrowRecords) -> Result<RoaringBitmap, PolicyError> {
        if let Some(ref fb) = self.fallback_engine {
            return fb.evaluate_filter(batch);
        }
        let Some(view) = LogsBatchView::try_new(batch)? else {
            return Ok(RoaringBitmap::new());
        };
        let mut all_rows = RoaringBitmap::new();
        all_rows.insert_range(0..view.num_rows as u32);

        let Some(ref drop_pipe) = self.opl_drop_pipeline else {
            return Ok(all_rows);
        };
        let mut dropped = drop_pipe.execute_matched_rows(&self.rt, batch, &view)?;
        if dropped.is_empty() {
            return Ok(all_rows);
        }
        if let Some(ref keep_pipe) = self.opl_keep_pipeline {
            let kept_exempt = keep_pipe.execute_matched_rows(&self.rt, batch, &view)?;
            dropped -= &kept_exempt;
        }
        all_rows -= &dropped;
        Ok(all_rows)
    }

    fn evaluate(&self, batch: OtapArrowRecords) -> Result<EvaluationOutput, PolicyError> {
        if let Some(ref fb) = self.fallback_engine {
            return fb.evaluate(batch);
        }
        if self.has_transforms
            && let Some(ref tx_eng) = self.transforms_engine
        {
            // Execute filter via OPL pipeline, then transforms via spec transform executor
            let Some(root) = batch.get(ArrowPayloadType::Logs) else {
                return Ok(EvaluationOutput {
                    records: Some(batch),
                    kept_rows: 0,
                    dropped_rows: 0,
                    transform_hits: 0,
                });
            };
            let orig_rows = root.num_rows();
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
            let filtered = filter_otap_logs_by_bitmap(&batch, &keep_rows, orig_rows, &mut pool)?;
            // Run only transforms on the surviving rows
            let tx_out = tx_eng.evaluate(filtered)?;
            return Ok(EvaluationOutput {
                records: tx_out.records,
                kept_rows: kept_count,
                dropped_rows: dropped_count,
                transform_hits: tx_out.transform_hits,
            });
        }

        let Some(root) = batch.get(ArrowPayloadType::Logs) else {
            return Ok(EvaluationOutput {
                records: Some(batch),
                kept_rows: 0,
                dropped_rows: 0,
                transform_hits: 0,
            });
        };
        let orig_rows = root.num_rows();
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
        let filtered = filter_otap_logs_by_bitmap(&batch, &keep_rows, orig_rows, &mut pool)?;
        Ok(EvaluationOutput {
            records: Some(filtered),
            kept_rows: kept_count,
            dropped_rows: dropped_count,
            transform_hits: 0,
        })
    }

    fn estimated_heap_bytes(&self) -> usize {
        self.heap_bytes
    }
}

/// Baseline B: Idiomatic OPL Text Pipeline (`logs | where ...`).
#[derive(Debug)]
pub struct IdiomaticOplTextEngine {
    inner: CompiledOplPolicyEngine,
}

impl IdiomaticOplTextEngine {
    /// Compiles `policies` into an idiomatic OPL query string and parses it into a `Pipeline`.
    pub fn compile(policies: &[Policy]) -> Result<Self, PolicyError> {
        Ok(Self {
            inner: CompiledOplPolicyEngine::compile(policies)?,
        })
    }
}

impl PolicyEvaluationEngine for IdiomaticOplTextEngine {
    fn name(&self) -> &'static str {
        "BaselineB_Idiomatic_OPL_Text"
    }

    fn evaluate_filter(&self, batch: &OtapArrowRecords) -> Result<RoaringBitmap, PolicyError> {
        self.inner.evaluate_filter(batch)
    }

    fn evaluate(&self, batch: OtapArrowRecords) -> Result<EvaluationOutput, PolicyError> {
        self.inner.evaluate(batch)
    }

    fn estimated_heap_bytes(&self) -> usize {
        self.inner.estimated_heap_bytes()
    }
}
