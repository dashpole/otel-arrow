// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Baseline A implementations (Vectorscan / `policy-rs` `v1.8.2` & `albert/4718-policy-engine`):
//! - `PolicyRsRowByRowEngine` (`BaselineA1_PolicyRs_RowByRow`): Upstream `usetero/policy-rs` row-by-row evaluation.
//! - `UnmodifiedAlbertVectorscanEngine` (`BaselineA2_Albert4718_Unmodified`): Unmodified `albert/4718-policy-engine` port.
//! - `FixedVectorscanColumnarEngine` (`BaselineA3_Vectorscan_FixedFair`): Repaired fair C/C++ Vectorscan columnar baseline
//!   (fixes `logs.body` `StructArray` resolution, `Int32` numeric coercion, `hs_clone_scratch` per-scan overhead, and plain `StringArray` `HashMap` thrashing).

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ffi::CString;
use std::ptr;

use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBufferBuilder, Datum, DictionaryArray, Float64Array,
    Int32Array, Int64Array, RecordBatch, StringArray, StructArray, UInt8Array, UInt16Array,
};
use arrow::buffer::BooleanBuffer;
use arrow::compute::filter as arrow_filter;
use arrow::compute::kernels::cmp;
use arrow::compute::not;
use arrow::datatypes::{DataType, UInt8Type, UInt16Type};
use otel_arrow_dfe_pdata::otap::filter::{IdBitmap, IdBitmapPool, build_native_selection_vec, filter_otap_batch};
use otel_arrow_dfe_pdata::otap::raw_batch_store::{LOGS_TYPE_MASK, RawBatchStore};
use otel_arrow_dfe_pdata::otap::transform::concatenate::{ConcatOptions, concatenate};
use otel_arrow_dfe_pdata::otap::transform::{
    AttributesTransform, DeleteTransform, RenameTransform, apply_attribute_transform,
};
use otel_arrow_dfe_pdata::otap::{Logs, OtapArrowRecords, OtapBatchStore};
use otel_arrow_dfe_pdata::otlp::attributes::AttributeValueType;
use otel_arrow_dfe_pdata::proto::opentelemetry::arrow::v1::ArrowPayloadType;
use otel_arrow_dfe_pdata::schema::consts;
use parking_lot::Mutex;
use policy_rs::engine::{
    CompiledDatabase, CompiledKeep, CompiledMatchers, CompiledNumericValue, CompiledPolicy,
    CompiledTransform, CompiledTypedMatcher, CompiledValue, EvaluateResult, ExistenceCheck,
    LogSignal, MatchKey, Signal, TransformOp, TypedCheck, TypedValue,
};
use policy_rs::proto::tero::policy::v1::{
    AttributePath as TeroAttrPath, LogAdd as TeroLogAdd, LogField as TeroLogField,
    LogMatcher as TeroLogMatcher, LogRedact as TeroLogRedact, LogRemove as TeroLogRemove,
    LogRename as TeroLogRename, LogTarget as TeroLogTarget, LogTransform as TeroLogTransform,
    NumericValue as TeroNumericValue, Policy as TeroProtoPolicy, Value as TeroValue,
    log_add as tero_log_add, log_matcher as tero_log_matcher, log_redact as tero_log_redact,
    log_remove as tero_log_remove, log_rename as tero_log_rename,
    numeric_value as tero_numeric_value, policy as tero_policy, value as tero_value,
};
use policy_rs::{
    LogFieldSelector as TeroLogFieldSelector, Matchable, Policy as TeroPolicy,
    PolicyEngine as TeroPolicyEngine, PolicyRegistry, PolicySnapshot,
};
use roaring::RoaringBitmap;

use crate::model::{
    Action, AttributePath, LogFieldSelector, LogMatcher, LogRecordField, LogTargetSelector,
    LogTransformPolicy, NumericValue, Policy, PolicyDefinition, PolicyError, Predicate,
    ScalarValue, ScopeField,
};
use crate::otap_ops::{
    CompiledTransformPolicy, ExtractedValue, LogsBatchView, apply_spec_transforms_masked,
    filter_otap_logs_by_bitmap,
};
use crate::{EvaluationOutput, PolicyEvaluationEngine};

// ============================================================================
// Conversion from `crate::model::Policy` to `policy_rs::Policy`
// ============================================================================

fn to_tero_attr_path(path: &AttributePath) -> TeroAttrPath {
    TeroAttrPath {
        path: path.path.clone(),
    }
}

fn to_tero_matcher_field(field: &LogFieldSelector) -> Option<tero_log_matcher::Field> {
    match field {
        LogFieldSelector::RecordField(rf) => {
            let tf = match rf {
                LogRecordField::Body => TeroLogField::Body,
                LogRecordField::SeverityText => TeroLogField::SeverityText,
                LogRecordField::TraceId => TeroLogField::TraceId,
                LogRecordField::SpanId => TeroLogField::SpanId,
                LogRecordField::EventName => TeroLogField::EventName,
                // Map SeverityNumber onto unused TeroLogField::ResourceSchemaUrl slot so policy-rs / albert-4718 compile it as a root simple field
                LogRecordField::SeverityNumber => TeroLogField::ResourceSchemaUrl,
                LogRecordField::Unspecified => return None,
            };
            Some(tero_log_matcher::Field::LogField(tf.into()))
        }
        LogFieldSelector::LogAttribute(p) => {
            Some(tero_log_matcher::Field::LogAttribute(to_tero_attr_path(p)))
        }
        LogFieldSelector::ResourceAttribute(p) => Some(
            tero_log_matcher::Field::ResourceAttribute(to_tero_attr_path(p)),
        ),
        LogFieldSelector::ScopeAttribute(p) => Some(tero_log_matcher::Field::ScopeAttribute(
            to_tero_attr_path(p),
        )),
        LogFieldSelector::ScopeField(ScopeField::SchemaUrl) => Some(
            tero_log_matcher::Field::LogField(TeroLogField::ScopeSchemaUrl.into()),
        ),
        LogFieldSelector::ScopeField(_) | LogFieldSelector::BodyField(_) => None,
    }
}

fn to_tero_numeric(nv: &NumericValue) -> TeroNumericValue {
    match nv {
        NumericValue::Int(i) => TeroNumericValue {
            value: Some(tero_numeric_value::Value::IntValue(*i)),
        },
        NumericValue::Double(d) => TeroNumericValue {
            value: Some(tero_numeric_value::Value::DoubleValue(*d)),
        },
    }
}

fn to_tero_matcher(m: &LogMatcher) -> Option<TeroLogMatcher> {
    let field = to_tero_matcher_field(&m.field)?;
    let mut case_insensitive = false;
    let r#match = match &m.predicate {
        Predicate::Equals(ScalarValue::String(s)) => tero_log_matcher::Match::Exact(s.clone()),
        Predicate::Equals(ScalarValue::Int(i)) => tero_log_matcher::Match::Equals(TeroValue {
            value: Some(tero_value::Value::IntValue(*i)),
        }),
        Predicate::Equals(ScalarValue::Double(d)) => tero_log_matcher::Match::Equals(TeroValue {
            value: Some(tero_value::Value::DoubleValue(*d)),
        }),
        Predicate::Equals(ScalarValue::Bool(b)) => tero_log_matcher::Match::Equals(TeroValue {
            value: Some(tero_value::Value::BoolValue(*b)),
        }),
        Predicate::Equals(ScalarValue::Bytes(b)) => tero_log_matcher::Match::Equals(TeroValue {
            value: Some(tero_value::Value::BytesValue(b.clone())),
        }),
        Predicate::Regex(pat) => tero_log_matcher::Match::Regex(pat.clone()),
        Predicate::Contains(ScalarValue::String(s)) => tero_log_matcher::Match::Contains(s.clone()),
        Predicate::Exists => tero_log_matcher::Match::Exists(true),
        Predicate::Gt(nv) => tero_log_matcher::Match::Gt(to_tero_numeric(nv)),
        Predicate::Gte(nv) => tero_log_matcher::Match::Gte(to_tero_numeric(nv)),
        Predicate::Lt(nv) => tero_log_matcher::Match::Lt(to_tero_numeric(nv)),
        Predicate::Lte(nv) => tero_log_matcher::Match::Lte(to_tero_numeric(nv)),
        _ => {
            let _ = &mut case_insensitive;
            return None;
        }
    };
    Some(TeroLogMatcher {
        field: Some(field),
        r#match: Some(r#match),
        negate: m.negate,
        case_insensitive,
    })
}

fn to_tero_transform(tx: &LogTransformPolicy) -> TeroLogTransform {
    let rename = tx
        .rename
        .iter()
        .filter_map(|r| {
            let from = match &r.from {
                LogTargetSelector::LogAttribute(p) => {
                    tero_log_rename::From::FromLogAttribute(to_tero_attr_path(p))
                }
                LogTargetSelector::ResourceAttribute(p) => {
                    tero_log_rename::From::FromResourceAttribute(to_tero_attr_path(p))
                }
                LogTargetSelector::ScopeAttribute(p) => {
                    tero_log_rename::From::FromScopeAttribute(to_tero_attr_path(p))
                }
                LogTargetSelector::RecordField(_) => return None,
            };
            let to = match &r.to {
                LogTargetSelector::LogAttribute(p)
                | LogTargetSelector::ResourceAttribute(p)
                | LogTargetSelector::ScopeAttribute(p) => p.path.first()?.to_owned(),
                LogTargetSelector::RecordField(_) => return None,
            };
            Some(TeroLogRename {
                from: Some(from),
                to,
                upsert: r.upsert,
            })
        })
        .collect();

    let remove = tx
        .remove
        .iter()
        .filter_map(|rm| {
            let field = match &rm.target {
                LogTargetSelector::LogAttribute(p) => {
                    tero_log_remove::Field::LogAttribute(to_tero_attr_path(p))
                }
                LogTargetSelector::ResourceAttribute(p) => {
                    tero_log_remove::Field::ResourceAttribute(to_tero_attr_path(p))
                }
                LogTargetSelector::ScopeAttribute(p) => {
                    tero_log_remove::Field::ScopeAttribute(to_tero_attr_path(p))
                }
                LogTargetSelector::RecordField(LogRecordField::Body) => {
                    tero_log_remove::Field::LogField(TeroLogField::Body.into())
                }
                LogTargetSelector::RecordField(_) => return None,
            };
            Some(TeroLogRemove { field: Some(field) })
        })
        .collect();

    let add = tx
        .add
        .iter()
        .filter_map(|a| {
            let ScalarValue::String(s) = &a.value else {
                return None;
            };
            let field = match &a.target {
                LogTargetSelector::LogAttribute(p) => {
                    tero_log_add::Field::LogAttribute(to_tero_attr_path(p))
                }
                LogTargetSelector::ResourceAttribute(p) => {
                    tero_log_add::Field::ResourceAttribute(to_tero_attr_path(p))
                }
                LogTargetSelector::ScopeAttribute(p) => {
                    tero_log_add::Field::ScopeAttribute(to_tero_attr_path(p))
                }
                LogTargetSelector::RecordField(LogRecordField::Body) => {
                    tero_log_add::Field::LogField(TeroLogField::Body.into())
                }
                LogTargetSelector::RecordField(_) => return None,
            };
            Some(TeroLogAdd {
                field: Some(field),
                value: s.clone(),
                upsert: a.upsert,
            })
        })
        .collect();

    let redact = tx
        .redact
        .iter()
        .filter_map(|rd| {
            let field = match &rd.target {
                LogTargetSelector::LogAttribute(p) => {
                    tero_log_redact::Field::LogAttribute(to_tero_attr_path(p))
                }
                LogTargetSelector::ResourceAttribute(p) => {
                    tero_log_redact::Field::ResourceAttribute(to_tero_attr_path(p))
                }
                LogTargetSelector::ScopeAttribute(p) => {
                    tero_log_redact::Field::ScopeAttribute(to_tero_attr_path(p))
                }
                LogTargetSelector::RecordField(LogRecordField::Body) => {
                    tero_log_redact::Field::LogField(TeroLogField::Body.into())
                }
                LogTargetSelector::RecordField(_) => return None,
            };
            Some(TeroLogRedact {
                field: Some(field),
                regex: rd.regex.clone(),
                replacement: rd
                    .replacement
                    .clone()
                    .unwrap_or_else(|| "[REDACTED]".to_owned()),
            })
        })
        .collect();

    TeroLogTransform {
        remove,
        redact,
        rename,
        add,
    }
}

fn compile_tero_snapshot(policies: &[Policy]) -> PolicySnapshot {
    let registry = PolicyRegistry::new();
    let handle = registry.register_provider();
    let mut tero_policies = Vec::with_capacity(policies.len());

    for p in policies {
        if !p.enabled {
            continue;
        }
        match &p.definition {
            PolicyDefinition::LogFilter(lf) => {
                let matchers: Vec<TeroLogMatcher> =
                    lf.matches.iter().filter_map(to_tero_matcher).collect();
                if matchers.is_empty() && !lf.matches.is_empty() {
                    continue;
                }
                let keep = match lf.action {
                    Action::Keep => "all",
                    Action::Drop | Action::Unspecified => "none",
                };
                tero_policies.push(TeroPolicy::new(TeroProtoPolicy {
                    id: p.id.clone(),
                    name: p.name.clone(),
                    enabled: p.enabled,
                    target: Some(tero_policy::Target::Log(TeroLogTarget {
                        r#match: matchers,
                        keep: keep.to_owned(),
                        transform: None,
                        sample_key: None,
                    })),
                    ..Default::default()
                }));
            }
            PolicyDefinition::LogTransform(tx) => {
                let matchers: Vec<TeroLogMatcher> =
                    tx.matches.iter().filter_map(to_tero_matcher).collect();
                tero_policies.push(TeroPolicy::new(TeroProtoPolicy {
                    id: p.id.clone(),
                    name: p.name.clone(),
                    enabled: p.enabled,
                    target: Some(tero_policy::Target::Log(TeroLogTarget {
                        r#match: matchers,
                        keep: "all".to_owned(),
                        transform: Some(to_tero_transform(tx)),
                        sample_key: None,
                    })),
                    ..Default::default()
                }));
            }
            _ => {}
        }
    }

    handle.update(tero_policies);
    registry.snapshot()
}

// ============================================================================
// Baseline A1: `PolicyRsRowByRowEngine` (`usetero/policy-rs` row-by-row)
// ============================================================================

struct OtapLogRowAdapter<'a> {
    view: &'a LogsBatchView<'a>,
    row: usize,
}

impl<'a> OtapLogRowAdapter<'a> {
    fn map_selector(field: &TeroLogFieldSelector) -> LogFieldSelector {
        match field {
            TeroLogFieldSelector::Simple(tf) => match tf {
                TeroLogField::Body | TeroLogField::Unspecified => {
                    LogFieldSelector::RecordField(LogRecordField::Body)
                }
                TeroLogField::SeverityText => {
                    LogFieldSelector::RecordField(LogRecordField::SeverityText)
                }
                TeroLogField::TraceId => LogFieldSelector::RecordField(LogRecordField::TraceId),
                TeroLogField::SpanId => LogFieldSelector::RecordField(LogRecordField::SpanId),
                TeroLogField::EventName => LogFieldSelector::RecordField(LogRecordField::EventName),
                TeroLogField::ResourceSchemaUrl => {
                    LogFieldSelector::RecordField(LogRecordField::SeverityNumber)
                }
                TeroLogField::ScopeSchemaUrl => {
                    LogFieldSelector::ScopeField(ScopeField::SchemaUrl)
                }
            },
            TeroLogFieldSelector::LogAttribute(p) => {
                LogFieldSelector::LogAttribute(AttributePath::nested(p.iter().cloned()))
            }
            TeroLogFieldSelector::ResourceAttribute(p) => {
                LogFieldSelector::ResourceAttribute(AttributePath::nested(p.iter().cloned()))
            }
            TeroLogFieldSelector::ScopeAttribute(p) => {
                LogFieldSelector::ScopeAttribute(AttributePath::nested(p.iter().cloned()))
            }
        }
    }
}

impl<'a> Matchable for OtapLogRowAdapter<'a> {
    type Signal = LogSignal;

    fn get_field(&self, field: &TeroLogFieldSelector) -> Option<Cow<'_, str>> {
        let sel = Self::map_selector(field);
        match self.view.extract_field_at_row(&sel, self.row)? {
            ExtractedValue::Str(s) => Some(s),
            _ => None,
        }
    }

    fn field_exists(&self, field: &TeroLogFieldSelector) -> bool {
        let sel = Self::map_selector(field);
        self.view.extract_field_at_row(&sel, self.row).is_some()
    }

    fn get_typed_value(&self, field: &TeroLogFieldSelector) -> Option<TypedValue<'_>> {
        let sel = Self::map_selector(field);
        match self.view.extract_field_at_row(&sel, self.row)? {
            ExtractedValue::Str(s) => Some(TypedValue::String(s)),
            ExtractedValue::Int(i) => Some(TypedValue::Int(i)),
            ExtractedValue::Double(d) => Some(TypedValue::Double(d)),
            ExtractedValue::Bool(b) => Some(TypedValue::Bool(b)),
            ExtractedValue::Bytes(Cow::Borrowed(b)) => Some(TypedValue::Bytes(b)),
            ExtractedValue::Bytes(Cow::Owned(_)) => None,
            ExtractedValue::Array(_) | ExtractedValue::Map(_) => None,
        }
    }
}

/// Baseline A1: Upstream `usetero/policy-rs` row-by-row evaluation engine.
pub struct PolicyRsRowByRowEngine {
    snapshot: PolicySnapshot,
    engine: TeroPolicyEngine,
    policies: Vec<Policy>,
    heap_bytes: usize,
}

impl PolicyRsRowByRowEngine {
    /// Compiles `policies` into a `policy_rs::PolicySnapshot`.
    pub fn compile(policies: &[Policy]) -> Result<Self, PolicyError> {
        let snapshot = compile_tero_snapshot(policies);
        let heap_bytes = std::mem::size_of::<Self>() + policies.len() * 512;
        Ok(Self {
            snapshot,
            engine: TeroPolicyEngine::new(),
            policies: policies.to_vec(),
            heap_bytes,
        })
    }
}

impl PolicyEvaluationEngine for PolicyRsRowByRowEngine {
    fn name(&self) -> &'static str {
        "BaselineA1_PolicyRs_RowByRow"
    }

    fn evaluate_filter(&self, batch: &OtapArrowRecords) -> Result<RoaringBitmap, PolicyError> {
        let Some(view) = LogsBatchView::try_new(batch)? else {
            return Ok(RoaringBitmap::new());
        };
        let n = view.num_rows;
        let mut kept = RoaringBitmap::new();
        for row in 0..n {
            let adapter = OtapLogRowAdapter { view: &view, row };
            let res =
                self.engine
                    .evaluate(&self.snapshot, &adapter)
                    .map_err(|e| PolicyError::Evaluation(e.to_string()))?;
            if !matches!(res, EvaluateResult::Drop { .. }) {
                kept.insert(row as u32);
            }
        }
        Ok(kept)
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
        let input_rows = root.num_rows() as u64;
        if input_rows == 0 {
            return Ok(EvaluationOutput {
                records: Some(batch),
                kept_rows: 0,
                dropped_rows: 0,
                transform_hits: 0,
            });
        }
        let keep_bitmap = self.evaluate_filter(&batch)?;
        let kept_rows = keep_bitmap.len();
        let dropped_rows = input_rows.saturating_sub(kept_rows);
        if kept_rows == 0 {
            return Ok(EvaluationOutput {
                records: None,
                kept_rows: 0,
                dropped_rows: input_rows,
                transform_hits: 0,
            });
        }
        let mut pool = IdBitmapPool::new();
        let mut filtered =
            filter_otap_logs_by_bitmap(&batch, &keep_bitmap, input_rows as usize, &mut pool)?;

        // Evaluate transform masks on surviving rows
        let mut compiled_tx = Vec::new();
        let mut tx_masks = Vec::new();
        if let Some(view) = LogsBatchView::try_new(&filtered)? {
            for p in &self.policies {
                if let PolicyDefinition::LogTransform(tx) = &p.definition {
                    let mut mask = RoaringBitmap::new();
                    mask.insert_range(0..view.num_rows as u32);
                    compiled_tx.push(CompiledTransformPolicy::compile(p.id.clone(), tx)?);
                    tx_masks.push(mask);
                }
            }
        }
        let transform_hits = if !compiled_tx.is_empty() {
            apply_spec_transforms_masked(&mut filtered, &compiled_tx, &tx_masks)?
        } else {
            0
        };

        Ok(EvaluationOutput {
            records: Some(filtered),
            kept_rows,
            dropped_rows,
            transform_hits,
        })
    }

    fn estimated_heap_bytes(&self) -> usize {
        self.heap_bytes
    }
}

// ============================================================================
// Baseline A2: `UnmodifiedAlbertVectorscanEngine` (`albert/4718-policy-engine`)
// ============================================================================

#[derive(Clone, Debug)]
enum AlbertPolicyMatch {
    All,
    None,
    Some(BooleanBuffer),
}

impl AlbertPolicyMatch {
    fn and(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::All, x) | (x, Self::All) => x.clone(),
            (Self::None, _) | (_, Self::None) => Self::None,
            (Self::Some(a), Self::Some(b)) => {
                let result = a & b;
                let set = result.count_set_bits();
                if set == 0 {
                    Self::None
                } else if set == result.len() {
                    Self::All
                } else {
                    Self::Some(result)
                }
            }
        }
    }

    fn or(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::None, x) | (x, Self::None) => x.clone(),
            (Self::All, _) | (_, Self::All) => Self::All,
            (Self::Some(a), Self::Some(b)) => {
                let result = a | b;
                if result.count_set_bits() == result.len() {
                    Self::All
                } else {
                    Self::Some(result)
                }
            }
        }
    }

    fn not(this: &Self) -> Self {
        match this {
            Self::All => Self::None,
            Self::None => Self::All,
            Self::Some(buf) => {
                let result = !buf;
                let set = result.count_set_bits();
                if set == 0 {
                    Self::None
                } else if set == result.len() {
                    Self::All
                } else {
                    Self::Some(result)
                }
            }
        }
    }

    fn filter_by(&self, keep_mask: &Self) -> Self {
        match (self, keep_mask) {
            (x, Self::All) => x.clone(),
            (_, Self::None) => Self::None,
            (Self::All, Self::Some(_)) => Self::All,
            (Self::None, Self::Some(_)) => Self::None,
            (Self::Some(buf), Self::Some(mask)) => {
                let arr = BooleanArray::new(buf.clone(), None);
                let mask_arr = BooleanArray::new(mask.clone(), None);
                let filtered = arrow_filter(&arr, &mask_arr).expect("filter boolean array");
                let filtered_bool = filtered.as_any().downcast_ref::<BooleanArray>().unwrap();
                let new_buf = filtered_bool.values().clone();
                let set = new_buf.count_set_bits();
                if set == 0 {
                    Self::None
                } else if set == new_buf.len() {
                    Self::All
                } else {
                    Self::Some(new_buf)
                }
            }
        }
    }

    fn to_boolean_array(&self, num_rows: usize) -> BooleanArray {
        match self {
            Self::All => BooleanArray::new(BooleanBuffer::new_set(num_rows), None),
            Self::None => BooleanArray::new(BooleanBuffer::new_unset(num_rows), None),
            Self::Some(buf) => BooleanArray::new(buf.clone(), None),
        }
    }

    fn from_boolean_array(arr: &BooleanArray) -> Self {
        let buffer = match arr.nulls() {
            Some(nulls) => arr.values() & nulls.inner(),
            None => arr.values().clone(),
        };
        let set_bits = buffer.count_set_bits();
        if set_bits == 0 {
            Self::None
        } else if set_bits == buffer.len() {
            Self::All
        } else {
            Self::Some(buffer)
        }
    }

    fn to_roaring(&self, num_rows: usize) -> RoaringBitmap {
        let mut rb = RoaringBitmap::new();
        match self {
            Self::All => {
                rb.insert_range(0..num_rows as u32);
            }
            Self::None => {}
            Self::Some(buf) => {
                for i in 0..num_rows {
                    if buf.value(i) {
                        rb.insert(i as u32);
                    }
                }
            }
        }
        rb
    }
}

enum AlbertFieldTarget<'a> {
    Attribute(&'a str, ArrowPayloadType),
    SimpleField(&'static str),
}

fn albert_simple_field_name(field: &TeroLogField) -> &'static str {
    match field {
        TeroLogField::Body => "body",
        TeroLogField::SeverityText => "severity_text",
        TeroLogField::TraceId => "trace_id",
        TeroLogField::SpanId => "span_id",
        TeroLogField::EventName => "event_name",
        TeroLogField::ResourceSchemaUrl => "severity_number",
        TeroLogField::ScopeSchemaUrl => "schema_url",
        TeroLogField::Unspecified => "body",
    }
}

fn albert_classify_field(field: &TeroLogFieldSelector) -> Option<AlbertFieldTarget<'_>> {
    match field {
        TeroLogFieldSelector::LogAttribute(path) => {
            if path.len() != 1 {
                return None;
            }
            Some(AlbertFieldTarget::Attribute(
                path.first()?.as_str(),
                ArrowPayloadType::LogAttrs,
            ))
        }
        TeroLogFieldSelector::ResourceAttribute(path) => {
            if path.len() != 1 {
                return None;
            }
            Some(AlbertFieldTarget::Attribute(
                path.first()?.as_str(),
                ArrowPayloadType::ResourceAttrs,
            ))
        }
        TeroLogFieldSelector::ScopeAttribute(path) => {
            if path.len() != 1 {
                return None;
            }
            Some(AlbertFieldTarget::Attribute(
                path.first()?.as_str(),
                ArrowPayloadType::ScopeAttrs,
            ))
        }
        TeroLogFieldSelector::Simple(log_field) => Some(AlbertFieldTarget::SimpleField(
            albert_simple_field_name(log_field),
        )),
    }
}

enum AlbertStringColumnAccessor<'a> {
    Plain(&'a StringArray),
    DictU8 {
        keys: &'a UInt8Array,
        values: &'a StringArray,
    },
    DictU16 {
        keys: &'a UInt16Array,
        values: &'a StringArray,
    },
}

impl<'a> AlbertStringColumnAccessor<'a> {
    fn try_new(column: &'a ArrayRef) -> Option<Self> {
        if let Some(arr) = column.as_any().downcast_ref::<StringArray>() {
            return Some(Self::Plain(arr));
        }
        if let Some(dict) = column.as_any().downcast_ref::<DictionaryArray<UInt8Type>>() {
            let values = dict.values().as_any().downcast_ref::<StringArray>()?;
            return Some(Self::DictU8 {
                keys: dict.keys(),
                values,
            });
        }
        if let Some(dict) = column
            .as_any()
            .downcast_ref::<DictionaryArray<UInt16Type>>()
        {
            let values = dict.values().as_any().downcast_ref::<StringArray>()?;
            return Some(Self::DictU16 {
                keys: dict.keys(),
                values,
            });
        }
        None
    }

    /// Unmodified `albert/4718-policy-engine` `scan_row` (clones Vectorscan scratch on every call AND inserts into `scan_cache` per row on `Plain`).
    fn scan_row_unmodified<'b>(
        &self,
        row: usize,
        db: &CompiledDatabase,
        scan_cache: &'b mut HashMap<usize, Vec<u32>>,
    ) -> Result<&'b [u32], PolicyError> {
        match self {
            Self::Plain(arr) => {
                if arr.is_null(row) {
                    return Ok(&[]);
                }
                let entry = scan_cache.entry(row);
                if matches!(entry, std::collections::hash_map::Entry::Vacant(_)) {
                    let value = arr.value(row);
                    let result = db
                        .database
                        .scan(value.as_bytes())
                        .map_err(|e| PolicyError::Evaluation(e.to_string()))?;
                    let _ = entry.or_insert(result);
                }
                Ok(scan_cache.get(&row).map(|v| v.as_slice()).unwrap_or(&[]))
            }
            Self::DictU8 { keys, values } => {
                if keys.is_null(row) {
                    return Ok(&[]);
                }
                let dict_idx = keys.value(row) as usize;
                if !scan_cache.contains_key(&dict_idx) {
                    if values.is_null(dict_idx) {
                        let _ = scan_cache.insert(dict_idx, Vec::new());
                    } else {
                        let value = values.value(dict_idx);
                        let result = db
                            .database
                            .scan(value.as_bytes())
                            .map_err(|e| PolicyError::Evaluation(e.to_string()))?;
                        let _ = scan_cache.insert(dict_idx, result);
                    }
                }
                Ok(scan_cache
                    .get(&dict_idx)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]))
            }
            Self::DictU16 { keys, values } => {
                if keys.is_null(row) {
                    return Ok(&[]);
                }
                let dict_idx = keys.value(row) as usize;
                if !scan_cache.contains_key(&dict_idx) {
                    if values.is_null(dict_idx) {
                        let _ = scan_cache.insert(dict_idx, Vec::new());
                    } else {
                        let value = values.value(dict_idx);
                        let result = db
                            .database
                            .scan(value.as_bytes())
                            .map_err(|e| PolicyError::Evaluation(e.to_string()))?;
                        let _ = scan_cache.insert(dict_idx, result);
                    }
                }
                Ok(scan_cache
                    .get(&dict_idx)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]))
            }
        }
    }
}

fn get_root_id_column<'a>(
    root_batch: &'a RecordBatch,
    attrs_payload_type: ArrowPayloadType,
) -> Option<&'a UInt16Array> {
    match attrs_payload_type {
        ArrowPayloadType::ResourceAttrs => root_batch
            .column_by_name(consts::RESOURCE)
            .and_then(|col| col.as_any().downcast_ref::<StructArray>())
            .and_then(|s| s.column_by_name(consts::ID))
            .and_then(|col| col.as_any().downcast_ref::<UInt16Array>()),
        ArrowPayloadType::ScopeAttrs => root_batch
            .column_by_name(consts::SCOPE)
            .and_then(|col| col.as_any().downcast_ref::<StructArray>())
            .and_then(|s| s.column_by_name(consts::ID))
            .and_then(|col| col.as_any().downcast_ref::<UInt16Array>()),
        _ => root_batch
            .column_by_name(consts::ID)
            .and_then(|col| col.as_any().downcast_ref::<UInt16Array>()),
    }
}

fn mark_db_policies_none(db: &CompiledDatabase, matched: &mut [AlbertPolicyMatch]) {
    for pattern_ref in &db.pattern_index {
        matched[pattern_ref.policy_index] = AlbertPolicyMatch::None;
    }
}

fn apply_policy_hits(
    policy_hits: &HashMap<usize, BooleanBufferBuilder>,
    db: &CompiledDatabase,
    negated: bool,
    matched: &mut [AlbertPolicyMatch],
) {
    let mut seen_policies = HashSet::new();
    for pattern_ref in &db.pattern_index {
        let policy_idx = pattern_ref.policy_index;
        if !seen_policies.insert(policy_idx) {
            continue;
        }
        let policy_hit = if let Some(builder) = policy_hits.get(&policy_idx) {
            let buffer = BooleanArray::new(builder.finish_cloned(), None);
            AlbertPolicyMatch::from_boolean_array(&buffer)
        } else {
            AlbertPolicyMatch::None
        };
        if negated {
            matched[policy_idx] = matched[policy_idx].and(&AlbertPolicyMatch::not(&policy_hit));
        } else {
            matched[policy_idx] = matched[policy_idx].and(&policy_hit);
        }
    }
}

fn albert_evaluate_typed_check(
    otap_batch: &OtapArrowRecords,
    check: &TypedCheck<LogSignal>,
    allow_int32_coercion: bool,
) -> Result<AlbertPolicyMatch, PolicyError> {
    let Some(target) = albert_classify_field(&check.field) else {
        return Ok(AlbertPolicyMatch::None);
    };
    let result = match target {
        AlbertFieldTarget::Attribute(key, payload_type) => {
            albert_evaluate_typed_on_attribute(otap_batch, key, payload_type, &check.matcher)
        }
        AlbertFieldTarget::SimpleField(col_name) => albert_evaluate_typed_on_root(
            otap_batch,
            col_name,
            &check.matcher,
            allow_int32_coercion,
        )?,
    };
    if check.is_negated {
        Ok(AlbertPolicyMatch::not(&result))
    } else {
        Ok(result)
    }
}

fn albert_evaluate_typed_on_root(
    otap_batch: &OtapArrowRecords,
    column_name: &str,
    matcher: &CompiledTypedMatcher,
    allow_int32_coercion: bool,
) -> Result<AlbertPolicyMatch, PolicyError> {
    let Some(root_batch) = otap_batch.get(ArrowPayloadType::Logs) else {
        return Ok(AlbertPolicyMatch::None);
    };
    let Some(column) = root_batch.column_by_name(column_name) else {
        return Ok(AlbertPolicyMatch::None);
    };

    let eval_num = |nv: &CompiledNumericValue,
                    cmp_fn: fn(&dyn Datum, &dyn Datum) -> Result<BooleanArray, arrow::error::ArrowError>|
     -> Result<Option<BooleanArray>, PolicyError> {
        let dt = match column.data_type() {
            DataType::Dictionary(_, v) => v.as_ref().clone(),
            d => d.clone(),
        };
        if !allow_int32_coercion {
            // Unmodified albert/4718-policy-engine compares root numeric columns directly against
            // Int64Array::new_scalar / Float64Array::new_scalar without Int32 coercion, failing on severity_number (Int32).
            return match nv {
                CompiledNumericValue::Int(v) => cmp_fn(column, &Int64Array::new_scalar(*v))
                    .map(Some)
                    .map_err(|e| PolicyError::Evaluation(e.to_string())),
                CompiledNumericValue::Double(v) => cmp_fn(column, &Float64Array::new_scalar(*v))
                    .map(Some)
                    .map_err(|e| PolicyError::Evaluation(e.to_string())),
            };
        }
        Ok(match nv {
            CompiledNumericValue::Int(v) => match dt {
                DataType::Int64 => cmp_fn(column, &Int64Array::new_scalar(*v)).ok(),
                DataType::Float64 => cmp_fn(column, &Float64Array::new_scalar(*v as f64)).ok(),
                DataType::Int32 => {
                    let v32 = (*v).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
                    cmp_fn(column, &Int32Array::new_scalar(v32)).ok()
                }
                _ => None,
            },
            CompiledNumericValue::Double(v) => match dt {
                DataType::Float64 => cmp_fn(column, &Float64Array::new_scalar(*v)).ok(),
                DataType::Int32 => {
                    let v32 = (*v as i64).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
                    cmp_fn(column, &Int32Array::new_scalar(v32)).ok()
                }
                _ => None,
            },
        })
    };

    let bool_arr = match matcher {
        CompiledTypedMatcher::Equals(CompiledValue::Int(i)) => {
            eval_num(&CompiledNumericValue::Int(*i), cmp::eq)?
        }
        CompiledTypedMatcher::Equals(CompiledValue::Double(d)) => {
            eval_num(&CompiledNumericValue::Double(*d), cmp::eq)?
        }
        CompiledTypedMatcher::Equals(CompiledValue::Bool(b)) => {
            cmp::eq(column, &BooleanArray::new_scalar(*b)).ok()
        }
        CompiledTypedMatcher::Equals(CompiledValue::Bytes(_)) => None,
        CompiledTypedMatcher::Gt(nv) => eval_num(nv, cmp::gt)?,
        CompiledTypedMatcher::Gte(nv) => eval_num(nv, cmp::gt_eq)?,
        CompiledTypedMatcher::Lt(nv) => eval_num(nv, cmp::lt)?,
        CompiledTypedMatcher::Lte(nv) => eval_num(nv, cmp::lt_eq)?,
    };
    Ok(match bool_arr {
        Some(arr) => AlbertPolicyMatch::from_boolean_array(&arr),
        None => AlbertPolicyMatch::None,
    })
}

fn albert_evaluate_typed_on_attribute(
    otap_batch: &OtapArrowRecords,
    attr_key: &str,
    payload_type: ArrowPayloadType,
    matcher: &CompiledTypedMatcher,
) -> AlbertPolicyMatch {
    let Some(root_batch) = otap_batch.get(ArrowPayloadType::Logs) else {
        return AlbertPolicyMatch::None;
    };
    let Some(root_id_col) = get_root_id_column(root_batch, payload_type) else {
        return AlbertPolicyMatch::None;
    };
    let Some(attrs_batch) = otap_batch.get(payload_type) else {
        return AlbertPolicyMatch::None;
    };
    let Some(key_col) = attrs_batch.column_by_name(consts::ATTRIBUTE_KEY) else {
        return AlbertPolicyMatch::None;
    };
    let key_scalar = StringArray::new_scalar(attr_key);
    let Ok(key_mask) = cmp::eq(key_col, &key_scalar) else {
        return AlbertPolicyMatch::None;
    };
    if key_mask.true_count() == 0 {
        return AlbertPolicyMatch::None;
    }
    let Some(parent_id_u16) = attrs_batch
        .column_by_name(consts::PARENT_ID)
        .and_then(|c| c.as_any().downcast_ref::<UInt16Array>())
    else {
        return AlbertPolicyMatch::None;
    };
    let Some(type_u8) = attrs_batch
        .column_by_name(consts::ATTRIBUTE_TYPE)
        .and_then(|c| c.as_any().downcast_ref::<UInt8Array>())
    else {
        return AlbertPolicyMatch::None;
    };

    let mut id_bitmap = IdBitmap::new();
    let collect_num = |nv: &CompiledNumericValue,
                       cmp_fn: fn(&dyn Datum, &dyn Datum) -> Result<BooleanArray, arrow::error::ArrowError>,
                       bitmap: &mut IdBitmap| {
        match nv {
            CompiledNumericValue::Int(v) => {
                if let Some(col) = attrs_batch.column_by_name(consts::ATTRIBUTE_INT)
                    && let Ok(res) = cmp_fn(col, &Int64Array::new_scalar(*v))
                {
                    let expected = AttributeValueType::Int as u8;
                    for i in 0..key_mask.len() {
                        if key_mask.value(i) && type_u8.value(i) == expected && res.value(i) {
                            bitmap.insert(u32::from(parent_id_u16.value(i)));
                        }
                    }
                }
                if let Some(col) = attrs_batch.column_by_name(consts::ATTRIBUTE_DOUBLE)
                    && let Ok(res) = cmp_fn(col, &Float64Array::new_scalar(*v as f64))
                {
                    let expected = AttributeValueType::Double as u8;
                    for i in 0..key_mask.len() {
                        if key_mask.value(i) && type_u8.value(i) == expected && res.value(i) {
                            bitmap.insert(u32::from(parent_id_u16.value(i)));
                        }
                    }
                }
            }
            CompiledNumericValue::Double(v) => {
                if let Some(col) = attrs_batch.column_by_name(consts::ATTRIBUTE_DOUBLE)
                    && let Ok(res) = cmp_fn(col, &Float64Array::new_scalar(*v))
                {
                    let expected = AttributeValueType::Double as u8;
                    for i in 0..key_mask.len() {
                        if key_mask.value(i) && type_u8.value(i) == expected && res.value(i) {
                            bitmap.insert(u32::from(parent_id_u16.value(i)));
                        }
                    }
                }
            }
        }
    };

    match matcher {
        CompiledTypedMatcher::Equals(CompiledValue::Bool(val)) => {
            if let Some(col) = attrs_batch.column_by_name(consts::ATTRIBUTE_BOOL)
                && let Ok(res) = cmp::eq(col, &BooleanArray::new_scalar(*val))
            {
                let expected = AttributeValueType::Bool as u8;
                for i in 0..key_mask.len() {
                    if key_mask.value(i) && type_u8.value(i) == expected && res.value(i) {
                        id_bitmap.insert(u32::from(parent_id_u16.value(i)));
                    }
                }
            }
        }
        CompiledTypedMatcher::Equals(CompiledValue::Int(val)) => {
            collect_num(&CompiledNumericValue::Int(*val), cmp::eq, &mut id_bitmap);
        }
        CompiledTypedMatcher::Equals(CompiledValue::Double(val)) => {
            collect_num(&CompiledNumericValue::Double(*val), cmp::eq, &mut id_bitmap);
        }
        CompiledTypedMatcher::Equals(CompiledValue::Bytes(_)) => {}
        CompiledTypedMatcher::Gt(nv) => collect_num(nv, cmp::gt, &mut id_bitmap),
        CompiledTypedMatcher::Gte(nv) => collect_num(nv, cmp::gt_eq, &mut id_bitmap),
        CompiledTypedMatcher::Lt(nv) => collect_num(nv, cmp::lt, &mut id_bitmap),
        CompiledTypedMatcher::Lte(nv) => collect_num(nv, cmp::lt_eq, &mut id_bitmap),
    }

    if id_bitmap.is_empty() {
        return AlbertPolicyMatch::None;
    }
    let selection = build_native_selection_vec(root_id_col, &id_bitmap);
    AlbertPolicyMatch::from_boolean_array(&selection)
}

fn albert_evaluate_existence_check(
    otap_batch: &OtapArrowRecords,
    check: &ExistenceCheck<LogSignal>,
) -> AlbertPolicyMatch {
    let Some(target) = albert_classify_field(&check.field) else {
        return AlbertPolicyMatch::None;
    };
    let result = match target {
        AlbertFieldTarget::Attribute(attr_key, payload_type) => {
            let no_attrs = if check.should_exist {
                AlbertPolicyMatch::None
            } else {
                AlbertPolicyMatch::All
            };
            let Some(root_batch) = otap_batch.get(ArrowPayloadType::Logs) else {
                return no_attrs;
            };
            let Some(root_id_col) = get_root_id_column(root_batch, payload_type) else {
                return no_attrs;
            };
            let Some(attrs_batch) = otap_batch.get(payload_type) else {
                return no_attrs;
            };
            let Some(key_col) = attrs_batch.column_by_name(consts::ATTRIBUTE_KEY) else {
                return no_attrs;
            };
            let Ok(key_mask) = cmp::eq(key_col, &StringArray::new_scalar(attr_key)) else {
                return AlbertPolicyMatch::None;
            };
            if key_mask.true_count() == 0 {
                return no_attrs;
            }
            let Some(parent_id_u16) = attrs_batch
                .column_by_name(consts::PARENT_ID)
                .and_then(|c| c.as_any().downcast_ref::<UInt16Array>())
            else {
                return AlbertPolicyMatch::None;
            };
            let mut id_bitmap = IdBitmap::new();
            for i in 0..parent_id_u16.len() {
                if key_mask.value(i) {
                    id_bitmap.insert(u32::from(parent_id_u16.value(i)));
                }
            }
            let selection = build_native_selection_vec(root_id_col, &id_bitmap);
            let m = AlbertPolicyMatch::from_boolean_array(&selection);
            if check.should_exist {
                m
            } else {
                AlbertPolicyMatch::not(&m)
            }
        }
        AlbertFieldTarget::SimpleField(col_name) => {
            let Some(root_batch) = otap_batch.get(ArrowPayloadType::Logs) else {
                return if check.should_exist {
                    AlbertPolicyMatch::None
                } else {
                    AlbertPolicyMatch::All
                };
            };
            let Some(column) = root_batch.column_by_name(col_name) else {
                return if check.should_exist {
                    AlbertPolicyMatch::None
                } else {
                    AlbertPolicyMatch::All
                };
            };
            let null_count = column.null_count();
            let num_rows = root_batch.num_rows();
            if check.should_exist {
                if null_count == 0 {
                    AlbertPolicyMatch::All
                } else if null_count == num_rows {
                    AlbertPolicyMatch::None
                } else {
                    match column.nulls() {
                        Some(nulls) => AlbertPolicyMatch::from_boolean_array(&BooleanArray::new(
                            nulls.inner().clone(),
                            None,
                        )),
                        None => AlbertPolicyMatch::All,
                    }
                }
            } else if null_count == num_rows {
                AlbertPolicyMatch::All
            } else if null_count == 0 {
                AlbertPolicyMatch::None
            } else {
                match column.nulls() {
                    Some(nulls) => AlbertPolicyMatch::from_boolean_array(&BooleanArray::new(
                        !nulls.inner(),
                        None,
                    )),
                    None => AlbertPolicyMatch::None,
                }
            }
        }
    };
    if check.is_negated {
        AlbertPolicyMatch::not(&result)
    } else {
        result
    }
}

/// Baseline A2: Unmodified port of `@albertlockett`'s `albert/4718-policy-engine`.
pub struct UnmodifiedAlbertVectorscanEngine {
    snapshot: PolicySnapshot,
    heap_bytes: usize,
}

impl UnmodifiedAlbertVectorscanEngine {
    /// Compiles `policies` into an unmodified `albert/4718-policy-engine` instance.
    pub fn compile(policies: &[Policy]) -> Result<Self, PolicyError> {
        let snapshot = compile_tero_snapshot(policies);
        let heap_bytes = std::mem::size_of::<Self>() + policies.len() * 640;
        Ok(Self {
            snapshot,
            heap_bytes,
        })
    }

    fn find_matching_policies(
        &self,
        compiled: &CompiledMatchers<LogSignal>,
        otap_batch: &OtapArrowRecords,
    ) -> Result<Vec<AlbertPolicyMatch>, PolicyError> {
        let policy_count = compiled.policies.len();
        if policy_count == 0 {
            return Ok(Vec::new());
        }
        let mut matched = vec![AlbertPolicyMatch::All; policy_count];
        let Some(root_batch) = otap_batch.get(ArrowPayloadType::Logs) else {
            return Ok(matched);
        };
        let num_root_rows = root_batch.num_rows();

        for (key, db) in &compiled.databases {
            let Some(target) = albert_classify_field(&key.field) else {
                mark_db_policies_none(db, &mut matched);
                continue;
            };
            match target {
                AlbertFieldTarget::SimpleField(col_name) => {
                    // Unmodified albert/4718: passes root_batch.column_by_name("body") (a StructArray)
                    // to StringColumnAccessor::try_new, which returns None and silently marks policies None!
                    let Some(column) = root_batch.column_by_name(col_name) else {
                        if !key.negated {
                            mark_db_policies_none(db, &mut matched);
                        }
                        continue;
                    };
                    let Some(accessor) = AlbertStringColumnAccessor::try_new(column) else {
                        if !key.negated {
                            mark_db_policies_none(db, &mut matched);
                        }
                        continue;
                    };
                    let mut policy_hits: HashMap<usize, BooleanBufferBuilder> = HashMap::new();
                    let mut scan_cache: HashMap<usize, Vec<u32>> = HashMap::new();
                    for i in 0..num_root_rows {
                        let matching_patterns =
                            accessor.scan_row_unmodified(i, db, &mut scan_cache)?;
                        for &pattern_id in matching_patterns {
                            if let Some(pattern_ref) = db.pattern_index.get(pattern_id as usize) {
                                let builder = policy_hits
                                    .entry(pattern_ref.policy_index)
                                    .or_insert_with(|| {
                                        let mut b = BooleanBufferBuilder::new(num_root_rows);
                                        b.append_n(num_root_rows, false);
                                        b
                                    });
                                builder.set_bit(i, true);
                            }
                        }
                    }
                    apply_policy_hits(&policy_hits, db, key.negated, &mut matched);
                }
                AlbertFieldTarget::Attribute(attr_key, payload_type) => {
                    let Some(root_id_col) = get_root_id_column(root_batch, payload_type) else {
                        if !key.negated {
                            mark_db_policies_none(db, &mut matched);
                        }
                        continue;
                    };
                    let Some(attrs_batch) = otap_batch.get(payload_type) else {
                        if !key.negated {
                            mark_db_policies_none(db, &mut matched);
                        }
                        continue;
                    };
                    let Some(key_col) = attrs_batch.column_by_name(consts::ATTRIBUTE_KEY) else {
                        if !key.negated {
                            mark_db_policies_none(db, &mut matched);
                        }
                        continue;
                    };
                    let key_scalar = StringArray::new_scalar(attr_key);
                    let key_mask = cmp::eq(key_col, &key_scalar)
                        .map_err(|e| PolicyError::Evaluation(e.to_string()))?;
                    if key_mask.true_count() == 0 {
                        if !key.negated {
                            mark_db_policies_none(db, &mut matched);
                        }
                        continue;
                    }
                    let Some(type_u8) = attrs_batch
                        .column_by_name(consts::ATTRIBUTE_TYPE)
                        .and_then(|c| c.as_any().downcast_ref::<UInt8Array>())
                    else {
                        if !key.negated {
                            mark_db_policies_none(db, &mut matched);
                        }
                        continue;
                    };
                    let Some(str_col) = attrs_batch.column_by_name(consts::ATTRIBUTE_STR) else {
                        if !key.negated {
                            mark_db_policies_none(db, &mut matched);
                        }
                        continue;
                    };
                    let Some(accessor) = AlbertStringColumnAccessor::try_new(str_col) else {
                        if !key.negated {
                            mark_db_policies_none(db, &mut matched);
                        }
                        continue;
                    };
                    let Some(parent_id_u16) = attrs_batch
                        .column_by_name(consts::PARENT_ID)
                        .and_then(|c| c.as_any().downcast_ref::<UInt16Array>())
                    else {
                        if !key.negated {
                            mark_db_policies_none(db, &mut matched);
                        }
                        continue;
                    };

                    let mut policy_id_bitmaps: HashMap<usize, IdBitmap> = HashMap::new();
                    let mut scan_cache: HashMap<usize, Vec<u32>> = HashMap::new();
                    let str_type = AttributeValueType::Str as u8;
                    for i in 0..attrs_batch.num_rows() {
                        if !key_mask.value(i) || type_u8.value(i) != str_type {
                            continue;
                        }
                        let matching_patterns =
                            accessor.scan_row_unmodified(i, db, &mut scan_cache)?;
                        for &pattern_id in matching_patterns {
                            if let Some(pattern_ref) = db.pattern_index.get(pattern_id as usize) {
                                policy_id_bitmaps
                                    .entry(pattern_ref.policy_index)
                                    .or_insert_with(IdBitmap::new)
                                    .insert(u32::from(parent_id_u16.value(i)));
                            }
                        }
                    }

                    let mut seen_policies = HashSet::new();
                    for pattern_ref in &db.pattern_index {
                        let policy_idx = pattern_ref.policy_index;
                        if !seen_policies.insert(policy_idx) {
                            continue;
                        }
                        let policy_hit = if let Some(id_bitmap) = policy_id_bitmaps.get(&policy_idx)
                        {
                            let selection = build_native_selection_vec(root_id_col, id_bitmap);
                            AlbertPolicyMatch::from_boolean_array(&selection)
                        } else {
                            AlbertPolicyMatch::None
                        };
                        if key.negated {
                            matched[policy_idx] =
                                matched[policy_idx].and(&AlbertPolicyMatch::not(&policy_hit));
                        } else {
                            matched[policy_idx] = matched[policy_idx].and(&policy_hit);
                        }
                    }
                }
            }
        }

        for check in &compiled.existence_checks {
            if !matches!(matched[check.policy_index], AlbertPolicyMatch::None) {
                let pm = albert_evaluate_existence_check(otap_batch, check);
                matched[check.policy_index] = matched[check.policy_index].and(&pm);
            }
        }

        for check in &compiled.typed_checks {
            if !matches!(matched[check.policy_index], AlbertPolicyMatch::None) {
                let pm = albert_evaluate_typed_check(otap_batch, check, false)?;
                matched[check.policy_index] = matched[check.policy_index].and(&pm);
            }
        }

        Ok(matched)
    }

    fn resolve_keep(
        policies: &[CompiledPolicy<LogSignal>],
        policy_matches: &[AlbertPolicyMatch],
    ) -> AlbertPolicyMatch {
        let mut already_discarded = AlbertPolicyMatch::None;
        for (idx, policy) in policies.iter().enumerate() {
            if !matches!(policy.keep, CompiledKeep::None) {
                continue;
            }
            already_discarded = already_discarded.or(&policy_matches[idx]);
        }
        AlbertPolicyMatch::not(&already_discarded)
    }
}

fn albert_apply_transforms(
    batch: &mut OtapArrowRecords,
    transform: &CompiledTransform<LogSignal>,
) -> Result<(), PolicyError> {
    for op in &transform.ops {
        match op {
            TransformOp::Remove { field } => {
                if let Some(AlbertFieldTarget::Attribute(key, payload_type)) =
                    albert_classify_field(field)
                {
                    let mut set = BTreeSet::new();
                    let _ = set.insert(key.to_owned());
                    let delete = DeleteTransform::new(set);
                    let tx = AttributesTransform::default().with_delete(delete);
                    let _ = apply_attribute_transform(batch, payload_type, &tx, false)
                        .map_err(|e| PolicyError::Evaluation(e.to_string()))?;
                }
            }
            TransformOp::Rename {
                from,
                to,
                upsert: _,
            } => {
                if let Some(AlbertFieldTarget::Attribute(key, payload_type)) =
                    albert_classify_field(from)
                {
                    let mut map = BTreeMap::new();
                    let _ = map.insert(key.to_owned(), to.clone());
                    let rename = RenameTransform::new(map);
                    let tx = AttributesTransform::default().with_rename(rename);
                    let _ = apply_attribute_transform(batch, payload_type, &tx, false)
                        .map_err(|e| PolicyError::Evaluation(e.to_string()))?;
                }
            }
            TransformOp::Redact { .. } | TransformOp::Add { .. } => {
                // Unimplemented in albert/4718-policy-engine
            }
        }
    }
    Ok(())
}

fn albert_concat_log_batches(
    a: OtapArrowRecords,
    b: OtapArrowRecords,
) -> Result<OtapArrowRecords, PolicyError> {
    let a_logs: Logs = a
        .try_into()
        .map_err(|e: otel_arrow_dfe_pdata::error::Error| PolicyError::Evaluation(e.to_string()))?;
    let b_logs: Logs = b
        .try_into()
        .map_err(|e: otel_arrow_dfe_pdata::error::Error| PolicyError::Evaluation(e.to_string()))?;
    let mut batches = vec![a_logs.into_batches(), b_logs.into_batches()];
    let concatenated = concatenate(&mut batches, ConcatOptions::preserve_ids())
        .map_err(|e| PolicyError::Evaluation(e.to_string()))?;
    let raw_store = RawBatchStore::<LOGS_TYPE_MASK, { Logs::COUNT }>::from_batches(concatenated);
    let result: Logs = raw_store
        .try_into()
        .map_err(|e: otel_arrow_dfe_pdata::error::Error| PolicyError::Evaluation(e.to_string()))?;
    Ok(OtapArrowRecords::from(result))
}

impl PolicyEvaluationEngine for UnmodifiedAlbertVectorscanEngine {
    fn name(&self) -> &'static str {
        "BaselineA2_Albert4718_Unmodified"
    }

    fn evaluate_filter(&self, batch: &OtapArrowRecords) -> Result<RoaringBitmap, PolicyError> {
        let Some(root) = batch.get(ArrowPayloadType::Logs) else {
            return Ok(RoaringBitmap::new());
        };
        let num_rows = root.num_rows();
        if num_rows == 0 {
            return Ok(RoaringBitmap::new());
        }
        let Some(compiled) = LogSignal::compiled_matchers(&self.snapshot) else {
            let mut all = RoaringBitmap::new();
            all.insert_range(0..num_rows as u32);
            return Ok(all);
        };
        let policy_matches = self.find_matching_policies(compiled, batch)?;
        let keep_mask = Self::resolve_keep(&compiled.policies, &policy_matches);
        Ok(keep_mask.to_roaring(num_rows))
    }

    fn evaluate(&self, otap_batch: OtapArrowRecords) -> Result<EvaluationOutput, PolicyError> {
        let Some(root) = otap_batch.get(ArrowPayloadType::Logs) else {
            return Ok(EvaluationOutput {
                records: Some(otap_batch),
                kept_rows: 0,
                dropped_rows: 0,
                transform_hits: 0,
            });
        };
        let input_num_rows = root.num_rows();
        if input_num_rows == 0 {
            return Ok(EvaluationOutput {
                records: Some(otap_batch),
                kept_rows: 0,
                dropped_rows: 0,
                transform_hits: 0,
            });
        }
        let Some(compiled_matchers) = LogSignal::compiled_matchers(&self.snapshot) else {
            return Ok(EvaluationOutput {
                records: Some(otap_batch),
                kept_rows: input_num_rows as u64,
                dropped_rows: 0,
                transform_hits: 0,
            });
        };

        let policy_matches = self.find_matching_policies(compiled_matchers, &otap_batch)?;
        let keep_mask = Self::resolve_keep(&compiled_matchers.policies, &policy_matches);
        let kept_rows = keep_mask.to_roaring(input_num_rows).len();
        let dropped_rows = (input_num_rows as u64).saturating_sub(kept_rows);

        let mut id_bitmap_pool = IdBitmapPool::new();
        let mut filtered_batch = match &keep_mask {
            AlbertPolicyMatch::None => {
                return Ok(EvaluationOutput {
                    records: None,
                    kept_rows: 0,
                    dropped_rows: input_num_rows as u64,
                    transform_hits: 0,
                });
            }
            AlbertPolicyMatch::All => otap_batch,
            AlbertPolicyMatch::Some(keep_selection) => filter_otap_batch(
                &BooleanArray::new(keep_selection.clone(), None),
                &otap_batch,
                &mut id_bitmap_pool,
            )
            .map_err(|e| PolicyError::Evaluation(e.to_string()))?,
        };

        let mut transform_hits = 0u64;
        for (idx, policy) in compiled_matchers.policies.iter().enumerate() {
            if matches!(policy.keep, CompiledKeep::None) {
                continue;
            }
            let Some(transform) = &policy.transform else {
                continue;
            };
            if transform.is_empty() {
                continue;
            }
            let filtered_match = policy_matches[idx].filter_by(&keep_mask);
            let cur_rows = filtered_batch
                .get(ArrowPayloadType::Logs)
                .map(|rb| rb.num_rows())
                .unwrap_or(0);
            if cur_rows == 0 {
                continue;
            }
            match filtered_match {
                AlbertPolicyMatch::None => continue,
                AlbertPolicyMatch::All => {
                    albert_apply_transforms(&mut filtered_batch, transform)?;
                    transform_hits += 1;
                }
                AlbertPolicyMatch::Some(_) => {
                    let selection = filtered_match.to_boolean_array(cur_rows);
                    let mut matched_batch =
                        filter_otap_batch(&selection, &filtered_batch, &mut id_bitmap_pool)
                            .map_err(|e| PolicyError::Evaluation(e.to_string()))?;
                    albert_apply_transforms(&mut matched_batch, transform)?;
                    let not_selection =
                        not(&selection).map_err(|e| PolicyError::Evaluation(e.to_string()))?;
                    let unmatched_batch =
                        filter_otap_batch(&not_selection, &filtered_batch, &mut id_bitmap_pool)
                            .map_err(|e| PolicyError::Evaluation(e.to_string()))?;
                    filtered_batch = albert_concat_log_batches(matched_batch, unmatched_batch)?;
                    transform_hits += 1;
                }
            }
        }

        Ok(EvaluationOutput {
            records: Some(filtered_batch),
            kept_rows,
            dropped_rows,
            transform_hits,
        })
    }

    fn estimated_heap_bytes(&self) -> usize {
        self.heap_bytes
    }
}

// ============================================================================
// Baseline A3: `FixedVectorscanColumnarEngine` (Fair C/C++ Vectorscan Baseline)
// ============================================================================

struct DirectVectorscanDb {
    db: *mut vectorscan_rs_sys::hs_database_t,
    scratch: Mutex<*mut vectorscan_rs_sys::hs_scratch_t>,
    pattern_to_policy: Vec<usize>,
}

unsafe impl Send for DirectVectorscanDb {}
unsafe impl Sync for DirectVectorscanDb {}

impl DirectVectorscanDb {
    fn compile(
        patterns: &[String],
        flags: &[u32],
        pattern_to_policy: Vec<usize>,
    ) -> Result<Self, PolicyError> {
        let ids: Vec<u32> = (0..patterns.len() as u32).collect();
        let c_patterns: Vec<CString> = patterns
            .iter()
            .map(|p| {
                CString::new(p.as_str())
                    .map_err(|e| PolicyError::Compile(format!("invalid regex '{p}': {e}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let ptrs: Vec<*const std::ffi::c_char> = c_patterns.iter().map(|s| s.as_ptr()).collect();

        let mut db: *mut vectorscan_rs_sys::hs_database_t = ptr::null_mut();
        let mut compile_err: *mut vectorscan_rs_sys::hs_compile_error_t = ptr::null_mut();

        let rc = unsafe {
            vectorscan_rs_sys::hs_compile_multi(
                ptrs.as_ptr(),
                flags.as_ptr(),
                ids.as_ptr(),
                patterns.len() as u32,
                vectorscan_rs_sys::HS_MODE_BLOCK,
                ptr::null(),
                &mut db,
                &mut compile_err,
            )
        };
        if rc != vectorscan_rs_sys::HS_SUCCESS as i32 {
            let reason = if !compile_err.is_null() {
                let s = unsafe {
                    std::ffi::CStr::from_ptr((*compile_err).message)
                        .to_string_lossy()
                        .into_owned()
                };
                unsafe { vectorscan_rs_sys::hs_free_compile_error(compile_err) };
                s
            } else {
                format!("hs_compile_multi failed: {rc}")
            };
            return Err(PolicyError::Evaluation(reason));
        }

        let mut scratch: *mut vectorscan_rs_sys::hs_scratch_t = ptr::null_mut();
        let s_rc = unsafe { vectorscan_rs_sys::hs_alloc_scratch(db, &mut scratch) };
        if s_rc != vectorscan_rs_sys::HS_SUCCESS as i32 {
            unsafe { vectorscan_rs_sys::hs_free_database(db) };
            return Err(PolicyError::Evaluation(format!(
                "hs_alloc_scratch failed: {s_rc}"
            )));
        }

        Ok(Self {
            db,
            scratch: Mutex::new(scratch),
            pattern_to_policy,
        })
    }

    /// Scans `data` using the pre-allocated Vectorscan scratch space (ZERO `hs_clone_scratch` per scan!).
    #[inline]
    fn scan_into(&self, data: &[u8], out_matches: &mut Vec<u32>) -> Result<(), PolicyError> {
        out_matches.clear();
        if data.is_empty() {
            return Ok(());
        }
        unsafe extern "C" fn on_match(
            id: std::ffi::c_uint,
            _from: std::ffi::c_ulonglong,
            _to: std::ffi::c_ulonglong,
            _flags: std::ffi::c_uint,
            context: *mut std::ffi::c_void,
        ) -> std::ffi::c_int {
            unsafe {
                let matches = &mut *(context as *mut Vec<u32>);
                matches.push(id);
            }
            0
        }

        let scratch_guard = self.scratch.lock();
        let rc = unsafe {
            vectorscan_rs_sys::hs_scan(
                self.db,
                data.as_ptr() as *const std::ffi::c_char,
                data.len() as u32,
                0,
                *scratch_guard,
                Some(on_match),
                out_matches as *mut Vec<u32> as *mut std::ffi::c_void,
            )
        };
        if rc != vectorscan_rs_sys::HS_SUCCESS as i32
            && rc != vectorscan_rs_sys::HS_SCAN_TERMINATED
        {
            return Err(PolicyError::Evaluation(format!("hs_scan failed: {rc}")));
        }
        Ok(())
    }

    fn database_bytes(&self) -> usize {
        let mut sz: usize = 0;
        let _ = unsafe { vectorscan_rs_sys::hs_database_size(self.db, &mut sz) };
        let mut s_sz: usize = 0;
        let guard = self.scratch.lock();
        let _ = unsafe { vectorscan_rs_sys::hs_scratch_size(*guard, &mut s_sz) };
        sz + s_sz + self.pattern_to_policy.len() * std::mem::size_of::<usize>()
    }
}

impl Drop for DirectVectorscanDb {
    fn drop(&mut self) {
        unsafe {
            let scratch = *self.scratch.get_mut();
            if !scratch.is_null() {
                vectorscan_rs_sys::hs_free_scratch(scratch);
            }
            if !self.db.is_null() {
                vectorscan_rs_sys::hs_free_database(self.db);
            }
        }
    }
}

/// Baseline A3: Repaired, fair C/C++ Vectorscan columnar baseline.
/// Fixes:
/// 1. Resolves `logs.body` `StructArray` to its `str` child column gated by `body.type == Str`.
/// 2. Supports `Int32Array` and `Dict<U8/U16, Int32>` on `severity_number`.
/// 3. Reuses pre-allocated `hs_scratch_t` via `DirectVectorscanDb::scan_into` instead of calling `hs_clone_scratch` on every string scan.
/// 4. Skips inserting into `scan_cache` on plain `StringArray` columns.
pub struct FixedVectorscanColumnarEngine {
    snapshot: PolicySnapshot,
    direct_dbs: HashMap<MatchKey<LogSignal>, DirectVectorscanDb>,
    policies: Vec<Policy>,
    policy_actions: Vec<Action>,
    heap_bytes: usize,
}

impl FixedVectorscanColumnarEngine {
    /// Compiles `policies` into a `FixedVectorscanColumnarEngine`.
    pub fn compile(policies: &[Policy]) -> Result<Self, PolicyError> {
        let snapshot = compile_tero_snapshot(policies);
        let mut direct_dbs = HashMap::new();
        let mut policy_actions = Vec::new();
        let mut heap_bytes = std::mem::size_of::<Self>();

        // Build direct Vectorscan databases with reusable scratch from the same pattern groups
        let mut sorted: Vec<&Policy> = policies.iter().filter(|p| p.enabled).collect();
        sorted.sort_by(|a, b| a.id.cmp(&b.id));

        let mut groups: HashMap<MatchKey<LogSignal>, (Vec<String>, Vec<u32>, Vec<usize>)> =
            HashMap::new();

        for (p_idx, p) in sorted.iter().enumerate() {
            let (matches, action) = match &p.definition {
                PolicyDefinition::LogFilter(lf) => (&lf.matches, lf.action),
                PolicyDefinition::LogTransform(tx) => (&tx.matches, Action::Keep),
                _ => continue,
            };
            policy_actions.push(action);
            for m in matches {
                let Some(tm) = to_tero_matcher(m) else {
                    continue;
                };
                let tero_field = match &tm.field {
                    Some(tero_log_matcher::Field::LogField(f)) => TeroLogFieldSelector::Simple(
                        TeroLogField::try_from(*f).unwrap_or(TeroLogField::Body),
                    ),
                    Some(tero_log_matcher::Field::LogAttribute(path)) => {
                        TeroLogFieldSelector::LogAttribute(path.path.clone())
                    }
                    Some(tero_log_matcher::Field::ResourceAttribute(path)) => {
                        TeroLogFieldSelector::ResourceAttribute(path.path.clone())
                    }
                    Some(tero_log_matcher::Field::ScopeAttribute(path)) => {
                        TeroLogFieldSelector::ScopeAttribute(path.path.clone())
                    }
                    None => continue,
                };
                let pat = match &tm.r#match {
                    Some(tero_log_matcher::Match::Exact(s)) => format!(r"^{}\z", regex::escape(s)),
                    Some(tero_log_matcher::Match::Regex(r)) => {
                        if r.ends_with('$') && !r.ends_with(r"\$") {
                            format!(r"{}\z", &r[..r.len() - 1])
                        } else {
                            r.clone()
                        }
                    }
                    Some(tero_log_matcher::Match::StartsWith(s)) => {
                        format!("^{}", regex::escape(s))
                    }
                    Some(tero_log_matcher::Match::EndsWith(s)) => {
                        format!(r"{}\z", regex::escape(s))
                    }
                    Some(tero_log_matcher::Match::Contains(s)) => regex::escape(s),
                    _ => continue,
                };
                let key = MatchKey::new(tero_field, tm.negate);
                let entry = groups.entry(key).or_default();
                entry.0.push(pat);
                entry.1.push(vectorscan_rs_sys::HS_FLAG_SINGLEMATCH);
                entry.2.push(p_idx);
            }
        }

        for (key, (pats, flags, p_indices)) in groups {
            if !pats.is_empty() {
                let db = DirectVectorscanDb::compile(&pats, &flags, p_indices)?;
                heap_bytes += db.database_bytes();
                direct_dbs.insert(key, db);
            }
        }

        Ok(Self {
            snapshot,
            direct_dbs,
            policies: policies.to_vec(),
            policy_actions,
            heap_bytes,
        })
    }

    fn evaluate_policy_matches(
        &self,
        otap_batch: &OtapArrowRecords,
    ) -> Result<Vec<AlbertPolicyMatch>, PolicyError> {
        let Some(compiled) = LogSignal::compiled_matchers(&self.snapshot) else {
            return Ok(Vec::new());
        };
        let policy_count = compiled.policies.len();
        if policy_count == 0 {
            return Ok(Vec::new());
        }
        let mut matched = vec![AlbertPolicyMatch::All; policy_count];
        let Some(root_batch) = otap_batch.get(ArrowPayloadType::Logs) else {
            return Ok(matched);
        };
        let num_root_rows = root_batch.num_rows();
        let mut scratch_matches: Vec<u32> = Vec::with_capacity(16);

        for (key, db) in &self.direct_dbs {
            let Some(target) = albert_classify_field(&key.field) else {
                for &p_idx in &db.pattern_to_policy {
                    matched[p_idx] = AlbertPolicyMatch::None;
                }
                continue;
            };
            match target {
                AlbertFieldTarget::SimpleField(col_name) => {
                    // Fix #1: Resolve `logs.body` StructArray to its `str` child column gated by `body.type == Str`!
                    let (column_opt, body_type_u8_opt): (Option<&ArrayRef>, Option<&UInt8Array>) =
                        if col_name == consts::BODY {
                            if let Some(body_col) = root_batch.column_by_name(consts::BODY)
                                && let Some(body_struct) =
                                    body_col.as_any().downcast_ref::<StructArray>()
                            {
                                let str_col = body_struct.column_by_name(consts::ATTRIBUTE_STR);
                                let type_col = body_struct
                                    .column_by_name(consts::ATTRIBUTE_TYPE)
                                    .and_then(|c| c.as_any().downcast_ref::<UInt8Array>());
                                (str_col, type_col)
                            } else {
                                (root_batch.column_by_name(col_name), None)
                            }
                        } else {
                            (root_batch.column_by_name(col_name), None)
                        };

                    let Some(column) = column_opt else {
                        if !key.negated {
                            for &p_idx in &db.pattern_to_policy {
                                matched[p_idx] = AlbertPolicyMatch::None;
                            }
                        }
                        continue;
                    };
                    let Some(accessor) = AlbertStringColumnAccessor::try_new(column) else {
                        if !key.negated {
                            for &p_idx in &db.pattern_to_policy {
                                matched[p_idx] = AlbertPolicyMatch::None;
                            }
                        }
                        continue;
                    };

                    let mut policy_hits: HashMap<usize, BooleanBufferBuilder> = HashMap::new();
                    let mut dict_cache: HashMap<usize, Vec<u32>> = HashMap::new();
                    let str_type = AttributeValueType::Str as u8;

                    for i in 0..num_root_rows {
                        if let Some(t_arr) = body_type_u8_opt
                            && (t_arr.is_null(i) || t_arr.value(i) != str_type)
                        {
                            continue;
                        }
                        // Fix #3 & #4: Reuse Vectorscan scratch and do NOT insert plain StringArray rows into `dict_cache`!
                        match &accessor {
                            AlbertStringColumnAccessor::Plain(arr) => {
                                if arr.is_null(i) {
                                    continue;
                                }
                                db.scan_into(arr.value(i).as_bytes(), &mut scratch_matches)?;
                                for &pat_id in &scratch_matches {
                                    if let Some(&p_idx) = db.pattern_to_policy.get(pat_id as usize)
                                    {
                                        let builder =
                                            policy_hits.entry(p_idx).or_insert_with(|| {
                                                let mut b =
                                                    BooleanBufferBuilder::new(num_root_rows);
                                                b.append_n(num_root_rows, false);
                                                b
                                            });
                                        builder.set_bit(i, true);
                                    }
                                }
                            }
                            AlbertStringColumnAccessor::DictU8 { keys, values } => {
                                if keys.is_null(i) {
                                    continue;
                                }
                                let dict_idx = keys.value(i) as usize;
                                let pat_ids = if let Some(cached) = dict_cache.get(&dict_idx) {
                                    cached.as_slice()
                                } else {
                                    if values.is_null(dict_idx) {
                                        dict_cache.insert(dict_idx, Vec::new());
                                    } else {
                                        db.scan_into(
                                            values.value(dict_idx).as_bytes(),
                                            &mut scratch_matches,
                                        )?;
                                        dict_cache.insert(dict_idx, scratch_matches.clone());
                                    }
                                    dict_cache.get(&dict_idx).unwrap().as_slice()
                                };
                                for &pat_id in pat_ids {
                                    if let Some(&p_idx) = db.pattern_to_policy.get(pat_id as usize)
                                    {
                                        let builder =
                                            policy_hits.entry(p_idx).or_insert_with(|| {
                                                let mut b =
                                                    BooleanBufferBuilder::new(num_root_rows);
                                                b.append_n(num_root_rows, false);
                                                b
                                            });
                                        builder.set_bit(i, true);
                                    }
                                }
                            }
                            AlbertStringColumnAccessor::DictU16 { keys, values } => {
                                if keys.is_null(i) {
                                    continue;
                                }
                                let dict_idx = keys.value(i) as usize;
                                let pat_ids = if let Some(cached) = dict_cache.get(&dict_idx) {
                                    cached.as_slice()
                                } else {
                                    if values.is_null(dict_idx) {
                                        dict_cache.insert(dict_idx, Vec::new());
                                    } else {
                                        db.scan_into(
                                            values.value(dict_idx).as_bytes(),
                                            &mut scratch_matches,
                                        )?;
                                        dict_cache.insert(dict_idx, scratch_matches.clone());
                                    }
                                    dict_cache.get(&dict_idx).unwrap().as_slice()
                                };
                                for &pat_id in pat_ids {
                                    if let Some(&p_idx) = db.pattern_to_policy.get(pat_id as usize)
                                    {
                                        let builder =
                                            policy_hits.entry(p_idx).or_insert_with(|| {
                                                let mut b =
                                                    BooleanBufferBuilder::new(num_root_rows);
                                                b.append_n(num_root_rows, false);
                                                b
                                            });
                                        builder.set_bit(i, true);
                                    }
                                }
                            }
                        }
                    }

                    let mut seen_policies = HashSet::new();
                    for &policy_idx in &db.pattern_to_policy {
                        if !seen_policies.insert(policy_idx) {
                            continue;
                        }
                        let policy_hit = if let Some(builder) = policy_hits.get(&policy_idx) {
                            let buffer = BooleanArray::new(builder.finish_cloned(), None);
                            AlbertPolicyMatch::from_boolean_array(&buffer)
                        } else {
                            AlbertPolicyMatch::None
                        };
                        if key.negated {
                            matched[policy_idx] =
                                matched[policy_idx].and(&AlbertPolicyMatch::not(&policy_hit));
                        } else {
                            matched[policy_idx] = matched[policy_idx].and(&policy_hit);
                        }
                    }
                }
                AlbertFieldTarget::Attribute(attr_key, payload_type) => {
                    let Some(root_id_col) = get_root_id_column(root_batch, payload_type) else {
                        if !key.negated {
                            for &p_idx in &db.pattern_to_policy {
                                matched[p_idx] = AlbertPolicyMatch::None;
                            }
                        }
                        continue;
                    };
                    let Some(attrs_batch) = otap_batch.get(payload_type) else {
                        if !key.negated {
                            for &p_idx in &db.pattern_to_policy {
                                matched[p_idx] = AlbertPolicyMatch::None;
                            }
                        }
                        continue;
                    };
                    let Some(key_col) = attrs_batch.column_by_name(consts::ATTRIBUTE_KEY) else {
                        if !key.negated {
                            for &p_idx in &db.pattern_to_policy {
                                matched[p_idx] = AlbertPolicyMatch::None;
                            }
                        }
                        continue;
                    };
                    let key_scalar = StringArray::new_scalar(attr_key);
                    let key_mask = cmp::eq(key_col, &key_scalar)
                        .map_err(|e| PolicyError::Evaluation(e.to_string()))?;
                    if key_mask.true_count() == 0 {
                        if !key.negated {
                            for &p_idx in &db.pattern_to_policy {
                                matched[p_idx] = AlbertPolicyMatch::None;
                            }
                        }
                        continue;
                    }
                    let Some(type_u8) = attrs_batch
                        .column_by_name(consts::ATTRIBUTE_TYPE)
                        .and_then(|c| c.as_any().downcast_ref::<UInt8Array>())
                    else {
                        continue;
                    };
                    let Some(str_col) = attrs_batch.column_by_name(consts::ATTRIBUTE_STR) else {
                        continue;
                    };
                    let Some(accessor) = AlbertStringColumnAccessor::try_new(str_col) else {
                        continue;
                    };
                    let Some(parent_id_u16) = attrs_batch
                        .column_by_name(consts::PARENT_ID)
                        .and_then(|c| c.as_any().downcast_ref::<UInt16Array>())
                    else {
                        continue;
                    };

                    let mut policy_id_bitmaps: HashMap<usize, IdBitmap> = HashMap::new();
                    let mut dict_cache: HashMap<usize, Vec<u32>> = HashMap::new();
                    let str_type = AttributeValueType::Str as u8;

                    for i in 0..attrs_batch.num_rows() {
                        if !key_mask.value(i) || type_u8.value(i) != str_type {
                            continue;
                        }
                        let pid = u32::from(parent_id_u16.value(i));
                        match &accessor {
                            AlbertStringColumnAccessor::Plain(arr) => {
                                if arr.is_null(i) {
                                    continue;
                                }
                                db.scan_into(arr.value(i).as_bytes(), &mut scratch_matches)?;
                                for &pat_id in &scratch_matches {
                                    if let Some(&p_idx) = db.pattern_to_policy.get(pat_id as usize)
                                    {
                                        policy_id_bitmaps
                                            .entry(p_idx)
                                            .or_insert_with(IdBitmap::new)
                                            .insert(pid);
                                    }
                                }
                            }
                            AlbertStringColumnAccessor::DictU8 { keys, values } => {
                                if keys.is_null(i) {
                                    continue;
                                }
                                let dict_idx = keys.value(i) as usize;
                                let pat_ids = if let Some(cached) = dict_cache.get(&dict_idx) {
                                    cached.as_slice()
                                } else {
                                    if values.is_null(dict_idx) {
                                        dict_cache.insert(dict_idx, Vec::new());
                                    } else {
                                        db.scan_into(
                                            values.value(dict_idx).as_bytes(),
                                            &mut scratch_matches,
                                        )?;
                                        dict_cache.insert(dict_idx, scratch_matches.clone());
                                    }
                                    dict_cache.get(&dict_idx).unwrap().as_slice()
                                };
                                for &pat_id in pat_ids {
                                    if let Some(&p_idx) = db.pattern_to_policy.get(pat_id as usize)
                                    {
                                        policy_id_bitmaps
                                            .entry(p_idx)
                                            .or_insert_with(IdBitmap::new)
                                            .insert(pid);
                                    }
                                }
                            }
                            AlbertStringColumnAccessor::DictU16 { keys, values } => {
                                if keys.is_null(i) {
                                    continue;
                                }
                                let dict_idx = keys.value(i) as usize;
                                let pat_ids = if let Some(cached) = dict_cache.get(&dict_idx) {
                                    cached.as_slice()
                                } else {
                                    if values.is_null(dict_idx) {
                                        dict_cache.insert(dict_idx, Vec::new());
                                    } else {
                                        db.scan_into(
                                            values.value(dict_idx).as_bytes(),
                                            &mut scratch_matches,
                                        )?;
                                        dict_cache.insert(dict_idx, scratch_matches.clone());
                                    }
                                    dict_cache.get(&dict_idx).unwrap().as_slice()
                                };
                                for &pat_id in pat_ids {
                                    if let Some(&p_idx) = db.pattern_to_policy.get(pat_id as usize)
                                    {
                                        policy_id_bitmaps
                                            .entry(p_idx)
                                            .or_insert_with(IdBitmap::new)
                                            .insert(pid);
                                    }
                                }
                            }
                        }
                    }

                    let mut seen_policies = HashSet::new();
                    for &policy_idx in &db.pattern_to_policy {
                        if !seen_policies.insert(policy_idx) {
                            continue;
                        }
                        let policy_hit = if let Some(id_bitmap) = policy_id_bitmaps.get(&policy_idx)
                        {
                            let selection = build_native_selection_vec(root_id_col, id_bitmap);
                            AlbertPolicyMatch::from_boolean_array(&selection)
                        } else {
                            AlbertPolicyMatch::None
                        };
                        if key.negated {
                            matched[policy_idx] =
                                matched[policy_idx].and(&AlbertPolicyMatch::not(&policy_hit));
                        } else {
                            matched[policy_idx] = matched[policy_idx].and(&policy_hit);
                        }
                    }
                }
            }
        }

        for check in &compiled.existence_checks {
            if !matches!(matched[check.policy_index], AlbertPolicyMatch::None) {
                let pm = albert_evaluate_existence_check(otap_batch, check);
                matched[check.policy_index] = matched[check.policy_index].and(&pm);
            }
        }

        // Fix #2: `allow_int32_coercion = true`
        for check in &compiled.typed_checks {
            if !matches!(matched[check.policy_index], AlbertPolicyMatch::None) {
                let pm = albert_evaluate_typed_check(otap_batch, check, true)?;
                matched[check.policy_index] = matched[check.policy_index].and(&pm);
            }
        }

        Ok(matched)
    }
}

impl PolicyEvaluationEngine for FixedVectorscanColumnarEngine {
    fn name(&self) -> &'static str {
        "BaselineA3_Vectorscan_FixedFair"
    }

    fn evaluate_filter(&self, batch: &OtapArrowRecords) -> Result<RoaringBitmap, PolicyError> {
        let Some(root) = batch.get(ArrowPayloadType::Logs) else {
            return Ok(RoaringBitmap::new());
        };
        let num_rows = root.num_rows();
        if num_rows == 0 {
            return Ok(RoaringBitmap::new());
        }
        let policy_matches = self.evaluate_policy_matches(batch)?;
        if policy_matches.is_empty() {
            let mut all = RoaringBitmap::new();
            all.insert_range(0..num_rows as u32);
            return Ok(all);
        }

        // Spec-compliant ACTION_KEEP > ACTION_DROP resolution
        let mut drop_set = RoaringBitmap::new();
        let mut keep_set = RoaringBitmap::new();
        for (idx, pm) in policy_matches.iter().enumerate() {
            let action = self
                .policy_actions
                .get(idx)
                .copied()
                .unwrap_or(Action::Drop);
            match action {
                Action::Drop => {
                    drop_set |= pm.to_roaring(num_rows);
                }
                Action::Keep => {
                    // Only filter-keep policies override drops (not transform-only policies)
                    if let Some(p) = self.policies.get(idx)
                        && matches!(p.definition, PolicyDefinition::LogFilter(_))
                    {
                        keep_set |= pm.to_roaring(num_rows);
                    }
                }
                Action::Unspecified => {}
            }
        }
        let effective_drop = &drop_set - &keep_set;
        let mut all = RoaringBitmap::new();
        all.insert_range(0..num_rows as u32);
        Ok(&all - &effective_drop)
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
        let input_rows = root.num_rows() as u64;
        if input_rows == 0 {
            return Ok(EvaluationOutput {
                records: Some(batch),
                kept_rows: 0,
                dropped_rows: 0,
                transform_hits: 0,
            });
        }
        let keep_bitmap = self.evaluate_filter(&batch)?;
        let kept_rows = keep_bitmap.len();
        let dropped_rows = input_rows.saturating_sub(kept_rows);
        if kept_rows == 0 {
            return Ok(EvaluationOutput {
                records: None,
                kept_rows: 0,
                dropped_rows: input_rows,
                transform_hits: 0,
            });
        }
        let mut pool = IdBitmapPool::new();
        let mut filtered =
            filter_otap_logs_by_bitmap(&batch, &keep_bitmap, input_rows as usize, &mut pool)?;

        let mut sorted: Vec<&Policy> = self.policies.iter().filter(|p| p.enabled).collect();
        sorted.sort_by(|a, b| a.id.cmp(&b.id));
        let filtered_matches = self.evaluate_policy_matches(&filtered)?;
        let surviving_rows = filtered
            .get(ArrowPayloadType::Logs)
            .map(|r| r.num_rows())
            .unwrap_or(0);

        let mut compiled_tx = Vec::new();
        let mut tx_masks = Vec::new();
        for (idx, p) in sorted.iter().enumerate() {
            if let PolicyDefinition::LogTransform(tx) = &p.definition {
                let mask = if tx.matches.is_empty() {
                    let mut all = RoaringBitmap::new();
                    all.insert_range(0..surviving_rows as u32);
                    all
                } else {
                    filtered_matches
                        .get(idx)
                        .map(|m| m.to_roaring(surviving_rows))
                        .unwrap_or_default()
                };
                compiled_tx.push(CompiledTransformPolicy::compile(p.id.clone(), tx)?);
                tx_masks.push(mask);
            }
        }
        let transform_hits = if !compiled_tx.is_empty() {
            apply_spec_transforms_masked(&mut filtered, &compiled_tx, &tx_masks)?
        } else {
            0
        };

        Ok(EvaluationOutput {
            records: Some(filtered),
            kept_rows,
            dropped_rows,
            transform_hits,
        })
    }

    fn estimated_heap_bytes(&self) -> usize {
        self.heap_bytes
    }
}
