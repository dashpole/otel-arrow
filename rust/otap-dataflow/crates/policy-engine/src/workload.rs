// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Deterministic OTAP `Logs` batch and `Policy` workload generator across batch sizes,
//! dictionary cardinalities, policy counts (`N = 1..50,000`), and workload profiles (`W1..W4b`).

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BinaryArray, BooleanArray, DictionaryArray, FixedSizeBinaryArray,
    Float64Array, Int32Array, Int64Array, RecordBatch, StringArray, StructArray, UInt8Array,
    UInt16Array,
};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Field, Schema, UInt8Type, UInt16Type};
use otel_arrow_dfe_pdata::otap::{Logs, OtapArrowRecords};
use otel_arrow_dfe_pdata::otlp::attributes::AttributeValueType;
use otel_arrow_dfe_pdata::proto::opentelemetry::arrow::v1::ArrowPayloadType;
use otel_arrow_dfe_pdata::schema::consts;

use crate::model::{
    Action, AttributePath, LogAdd, LogFieldSelector, LogMatcher, LogRecordField, LogRedact,
    LogRemove, LogRename, LogTargetSelector, LogTransformPolicy, NumericValue, Policy, Predicate,
    ScalarValue,
};

/// Dictionary encoding cardinality mode for synthetic OTAP `Logs` batches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DictCardinality {
    /// 16 unique values (`DictionaryArray<UInt8Type>`).
    Low16,
    /// 256 unique values (`DictionaryArray<UInt16Type>`).
    Medium256,
    /// Plain non-dictionary `StringArray` with unique per-row strings.
    Plain,
}

impl DictCardinality {
    /// Short label for benchmark tables.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Low16 => "dict16",
            Self::Medium256 => "dict256",
            Self::Plain => "plain",
        }
    }
}

/// Workload profile (`W1`, `W2`, `W3`, `W4a`, `W4b`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadProfile {
    /// W1: Realistic production mix (60% structured eq/exists, 15% numeric range, 15% contains, 10% regex).
    W1ProductionMix,
    /// W2: Literal-rich substring (`contains`) + anchored/literal-bearing `regex`.
    W2LiteralRegex,
    /// W3: Adversarial unanchored character-class regexes with no extractable literal > 2 bytes.
    W3AdversarialRegex,
    /// W4a: 50% Filter + 50% Transform (`Rename` + `Remove`) for cross-baseline comparison.
    W4aFilterRenameRemove,
    /// W4b: 50% Filter + 50% Full 4-Stage Spec Transform (`Rename -> Add -> Remove -> Redact`).
    W4bFullSpecTransforms,
}

impl WorkloadProfile {
    /// Short label for benchmark tables.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::W1ProductionMix => "W1_prod_mix",
            Self::W2LiteralRegex => "W2_literal_regex",
            Self::W3AdversarialRegex => "W3_adversarial_regex",
            Self::W4aFilterRenameRemove => "W4a_filter_rename_remove",
            Self::W4bFullSpecTransforms => "W4b_full_4stage_transform",
        }
    }
}

const SERVICES: &[&str] = &[
    "checkout-service",
    "payment-gateway",
    "auth-service",
    "inventory-api",
    "recommendation-engine",
    "shipping-worker",
    "notification-hub",
    "fraud-detector",
];

const SEVERITIES: &[(&str, i32)] = &[
    ("TRACE", 1),
    ("DEBUG", 5),
    ("INFO", 9),
    ("WARN", 13),
    ("ERROR", 17),
    ("FATAL", 21),
];

const ROUTES: &[&str] = &[
    "/api/v1/orders/1042",
    "/api/v2/orders/9981",
    "/api/v1/checkout/session",
    "/api/v1/auth/token",
    "/healthz",
    "/metrics",
    "/api/v2/inventory/reserve",
    "/api/v1/payments/charge",
];

/// Builds a deterministic `OtapArrowRecords` (`Logs`) batch with `num_rows` log records.
#[must_use]
pub fn generate_logs_batch(num_rows: usize, cardinality: DictCardinality) -> OtapArrowRecords {
    let ids: Vec<u16> = (0..num_rows).map(|i| (i % 65535) as u16).collect();
    let res_ids: Vec<u16> = (0..num_rows).map(|i| (i % SERVICES.len()) as u16).collect();
    let scope_ids: Vec<u16> = (0..num_rows).map(|i| (i % 4) as u16).collect();

    let sev_texts: Vec<&str> = (0..num_rows)
        .map(|i| SEVERITIES[i % SEVERITIES.len()].0)
        .collect();
    let sev_nums: Vec<i32> = (0..num_rows)
        .map(|i| SEVERITIES[i % SEVERITIES.len()].1)
        .collect();
    let event_names: Vec<String> = (0..num_rows)
        .map(|i| format!("event.domain.action_{}", i % 32))
        .collect();
    let schema_urls: Vec<&str> = (0..num_rows)
        .map(|i| {
            if i % 2 == 0 {
                "https://opentelemetry.io/schemas/1.28.0"
            } else {
                "https://opentelemetry.io/schemas/1.27.0"
            }
        })
        .collect();

    // Generate body strings according to cardinality
    let unique_count = match cardinality {
        DictCardinality::Low16 => 16,
        DictCardinality::Medium256 => 256,
        DictCardinality::Plain => num_rows.max(1),
    };
    let unique_bodies: Vec<String> = (0..unique_count)
        .map(|u| match u % 8 {
            0 => format!(
                "POST /api/v1/orders/{} user_id=550e8400-e29b-41d4-a716-44665544{:04x} status=200",
                1000 + u,
                u
            ),
            1 => format!(
                "ERROR [auth] connection reset while verifying Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.{:06x}",
                u
            ),
            2 => format!(
                "WARN deadline exceeded contacting upstream 10.{}.{}.{} after 250ms",
                (u >> 4) & 255,
                u & 255,
                (u * 7) & 255
            ),
            3 => format!(
                "DEBUG cache hit for session {:08x}-{:04x}-{:04x} in region us-east1",
                u * 17 + 0x1000_0000,
                u & 0xffff,
                (u * 3) & 0xffff
            ),
            4 => format!(
                "INFO processed payment charge ssn=123-45-{:04} amount={}.99",
                u % 10000,
                10 + (u % 500)
            ),
            5 => format!(
                "ERROR OutOfMemory in worker pool shard={} active_tasks={}",
                u % 16,
                64 + (u % 128)
            ),
            6 => format!(
                "POST /api/v2/orders/{} client_ip=192.168.{}.{} trace_ok=true",
                2000 + u,
                (u >> 3) & 255,
                u & 255
            ),
            _ => format!("INFO heartbeat ok seq={} node=worker-{}", u, u % 32),
        })
        .collect();

    let body_str_col: ArrayRef = match cardinality {
        DictCardinality::Low16 => {
            let keys = UInt16Array::from_iter_values((0..num_rows).map(|i| (i % 16) as u16));
            let values = StringArray::from(
                unique_bodies
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
            );
            Arc::new(DictionaryArray::<UInt16Type>::try_new(keys, Arc::new(values)).unwrap())
        }
        DictCardinality::Medium256 => {
            let keys = UInt16Array::from_iter_values((0..num_rows).map(|i| (i % 256) as u16));
            let values = StringArray::from(
                unique_bodies
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
            );
            Arc::new(DictionaryArray::<UInt16Type>::try_new(keys, Arc::new(values)).unwrap())
        }
        DictCardinality::Plain => {
            let values = StringArray::from(
                (0..num_rows)
                    .map(|i| unique_bodies[i % unique_bodies.len()].as_str())
                    .collect::<Vec<_>>(),
            );
            Arc::new(values)
        }
    };

    let sev_text_col: ArrayRef = match cardinality {
        DictCardinality::Low16 => {
            let keys =
                UInt8Array::from_iter_values((0..num_rows).map(|i| (i % SEVERITIES.len()) as u8));
            let values =
                StringArray::from(SEVERITIES.iter().map(|(s, _)| *s).collect::<Vec<_>>());
            Arc::new(DictionaryArray::<UInt8Type>::try_new(keys, Arc::new(values)).unwrap())
        }
        DictCardinality::Medium256 => {
            let keys = UInt16Array::from_iter_values(
                (0..num_rows).map(|i| (i % SEVERITIES.len()) as u16),
            );
            let values =
                StringArray::from(SEVERITIES.iter().map(|(s, _)| *s).collect::<Vec<_>>());
            Arc::new(DictionaryArray::<UInt16Type>::try_new(keys, Arc::new(values)).unwrap())
        }
        DictCardinality::Plain => Arc::new(StringArray::from(sev_texts)),
    };

    let unique_events: Vec<String> = (0..32)
        .map(|i| format!("event.domain.action_{i}"))
        .collect();
    let event_name_col: ArrayRef = match cardinality {
        DictCardinality::Low16 => {
            let keys = UInt8Array::from_iter_values((0..num_rows).map(|i| (i % 16) as u8));
            let values = StringArray::from(
                unique_events[..16]
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
            );
            Arc::new(DictionaryArray::<UInt8Type>::try_new(keys, Arc::new(values)).unwrap())
        }
        DictCardinality::Medium256 => {
            let keys = UInt16Array::from_iter_values((0..num_rows).map(|i| (i % 32) as u16));
            let values = StringArray::from(
                unique_events
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
            );
            Arc::new(DictionaryArray::<UInt16Type>::try_new(keys, Arc::new(values)).unwrap())
        }
        DictCardinality::Plain => Arc::new(StringArray::from(event_names)),
    };

    let body_types = UInt8Array::from_iter_values(std::iter::repeat_n(
        AttributeValueType::Str as u8,
        num_rows,
    ));
    let body_struct = StructArray::new(
        vec![
            Arc::new(Field::new(consts::ATTRIBUTE_TYPE, DataType::UInt8, false)),
            Arc::new(Field::new(
                consts::ATTRIBUTE_STR,
                body_str_col.data_type().clone(),
                true,
            )),
        ]
        .into(),
        vec![Arc::new(body_types), body_str_col],
        Some(NullBuffer::from(vec![true; num_rows])),
    );

    // Trace IDs (16 bytes) and Span IDs (8 bytes)
    let trace_ids: Vec<[u8; 16]> = (0..num_rows)
        .map(|i| {
            let mut b = [0u8; 16];
            b[0] = 0x0a;
            b[14] = ((i >> 8) & 0xff) as u8;
            b[15] = ((i & 0xff) as u8).max(1);
            b
        })
        .collect();
    let span_ids: Vec<[u8; 8]> = (0..num_rows)
        .map(|i| {
            let mut b = [0u8; 8];
            b[0] = 0x0b;
            b[7] = ((i & 0xff) as u8).max(1);
            b
        })
        .collect();
    let trace_id_arr = FixedSizeBinaryArray::try_from_iter(trace_ids.iter().map(|b| b.as_slice())).unwrap();
    let span_id_arr = FixedSizeBinaryArray::try_from_iter(span_ids.iter().map(|b| b.as_slice())).unwrap();

    // Resource StructArray
    let res_struct = StructArray::new(
        vec![Arc::new(Field::new(consts::ID, DataType::UInt16, false))].into(),
        vec![Arc::new(UInt16Array::from(res_ids))],
        None,
    );

    // Scope StructArray
    let scope_names: Vec<&str> = (0..num_rows)
        .map(|i| match i % 4 {
            0 => "otel.http.logger",
            1 => "otel.grpc.logger",
            2 => "otel.db.logger",
            _ => "otel.worker.logger",
        })
        .collect();
    let scope_versions: Vec<&str> = vec!["1.0.0"; num_rows];
    let scope_struct = StructArray::new(
        vec![
            Arc::new(Field::new(consts::ID, DataType::UInt16, false)),
            Arc::new(Field::new(consts::NAME, DataType::Utf8, true)),
            Arc::new(Field::new(consts::VERSION, DataType::Utf8, true)),
        ]
        .into(),
        vec![
            Arc::new(UInt16Array::from(scope_ids)),
            Arc::new(StringArray::from(scope_names)),
            Arc::new(StringArray::from(scope_versions)),
        ],
        None,
    );

    let root_schema = Arc::new(Schema::new(vec![
        Field::new(consts::ID, DataType::UInt16, false),
        Field::new(consts::BODY, body_struct.data_type().clone(), true),
        Field::new(consts::SEVERITY_NUMBER, DataType::Int32, true),
        Field::new(consts::SEVERITY_TEXT, sev_text_col.data_type().clone(), true),
        Field::new(consts::EVENT_NAME, event_name_col.data_type().clone(), true),
        Field::new(consts::TRACE_ID, DataType::FixedSizeBinary(16), true),
        Field::new(consts::SPAN_ID, DataType::FixedSizeBinary(8), true),
        Field::new(consts::SCHEMA_URL, DataType::Utf8, true),
        Field::new(consts::RESOURCE, res_struct.data_type().clone(), true),
        Field::new(consts::SCOPE, scope_struct.data_type().clone(), true),
    ]));
    let root_rb = RecordBatch::try_new(
        root_schema,
        vec![
            Arc::new(UInt16Array::from(ids)),
            Arc::new(body_struct),
            Arc::new(Int32Array::from(sev_nums)),
            sev_text_col,
            event_name_col,
            Arc::new(trace_id_arr),
            Arc::new(span_id_arr),
            Arc::new(StringArray::from(schema_urls)),
            Arc::new(res_struct),
            Arc::new(scope_struct),
        ],
    )
    .unwrap();

    // Build LogAttrs (4 attributes per log row: http.route, env, http.status_code, latency_ms)
    let attr_rows = num_rows * 4;
    let mut la_pids = Vec::with_capacity(attr_rows);
    let mut la_keys = Vec::with_capacity(attr_rows);
    let mut la_types = Vec::with_capacity(attr_rows);
    let mut la_strs: Vec<Option<String>> = Vec::with_capacity(attr_rows);
    let mut la_ints: Vec<Option<i64>> = Vec::with_capacity(attr_rows);
    let mut la_doubles: Vec<Option<f64>> = Vec::with_capacity(attr_rows);
    let mut la_bools: Vec<Option<bool>> = Vec::with_capacity(attr_rows);
    let mut la_bytes: Vec<Option<&[u8]>> = Vec::with_capacity(attr_rows);

    for r in 0..num_rows {
        let pid = (r % 65535) as u16;
        // 1. http.route (Str)
        la_pids.push(pid);
        la_keys.push("http.route");
        la_types.push(AttributeValueType::Str as u8);
        la_strs.push(Some(ROUTES[r % ROUTES.len()].to_owned()));
        la_ints.push(None);
        la_doubles.push(None);
        la_bools.push(None);
        la_bytes.push(None);

        // 2. env (Str)
        la_pids.push(pid);
        la_keys.push("env");
        la_types.push(AttributeValueType::Str as u8);
        la_strs.push(Some(if r % 3 == 0 { "prod" } else { "staging" }.to_owned()));
        la_ints.push(None);
        la_doubles.push(None);
        la_bools.push(None);
        la_bytes.push(None);

        // 3. http.status_code (Int)
        la_pids.push(pid);
        la_keys.push("http.status_code");
        la_types.push(AttributeValueType::Int as u8);
        la_strs.push(None);
        la_ints.push(Some(match r % 5 {
            0 => 200,
            1 => 201,
            2 => 400,
            3 => 404,
            _ => 503,
        }));
        la_doubles.push(None);
        la_bools.push(None);
        la_bytes.push(None);

        // 4. latency_ms (Double)
        la_pids.push(pid);
        la_keys.push("latency_ms");
        la_types.push(AttributeValueType::Double as u8);
        la_strs.push(None);
        la_ints.push(None);
        la_doubles.push(Some((r % 500) as f64 + 1.5));
        la_bools.push(None);
        la_bytes.push(None);
    }

    let attr_schema = Arc::new(Schema::new(vec![
        Field::new(consts::PARENT_ID, DataType::UInt16, false),
        Field::new(consts::ATTRIBUTE_KEY, DataType::Utf8, false),
        Field::new(consts::ATTRIBUTE_TYPE, DataType::UInt8, false),
        Field::new(consts::ATTRIBUTE_STR, DataType::Utf8, true),
        Field::new(consts::ATTRIBUTE_INT, DataType::Int64, true),
        Field::new(consts::ATTRIBUTE_DOUBLE, DataType::Float64, true),
        Field::new(consts::ATTRIBUTE_BOOL, DataType::Boolean, true),
        Field::new(consts::ATTRIBUTE_BYTES, DataType::Binary, true),
    ]));
    let log_attrs_rb = RecordBatch::try_new(
        attr_schema.clone(),
        vec![
            Arc::new(UInt16Array::from(la_pids)),
            Arc::new(StringArray::from(la_keys)),
            Arc::new(UInt8Array::from(la_types)),
            Arc::new(StringArray::from(la_strs)),
            Arc::new(Int64Array::from(la_ints)),
            Arc::new(Float64Array::from(la_doubles)),
            Arc::new(BooleanArray::from(la_bools)),
            Arc::new(BinaryArray::from_opt_vec(la_bytes)),
        ],
    )
    .unwrap();

    // Build ResourceAttrs (service.name per resource.id)
    let res_count = SERVICES.len();
    let res_pids: Vec<u16> = (0..res_count as u16).collect();
    let res_keys: Vec<&str> = vec!["service.name"; res_count];
    let res_types: Vec<u8> = vec![AttributeValueType::Str as u8; res_count];
    let res_strs: Vec<Option<&str>> = SERVICES.iter().map(|&s| Some(s)).collect();
    let res_attrs_rb = RecordBatch::try_new(
        attr_schema,
        vec![
            Arc::new(UInt16Array::from(res_pids)),
            Arc::new(StringArray::from(res_keys)),
            Arc::new(UInt8Array::from(res_types)),
            Arc::new(StringArray::from(res_strs)),
            Arc::new(Int64Array::from(vec![None; res_count])),
            Arc::new(Float64Array::from(vec![None; res_count])),
            Arc::new(BooleanArray::from(vec![None; res_count])),
            Arc::new(BinaryArray::from_opt_vec(vec![None; res_count])),
        ],
    )
    .unwrap();

    let mut records = OtapArrowRecords::Logs(Logs::default());
    records.set(ArrowPayloadType::Logs, root_rb).unwrap();
    records
        .set(ArrowPayloadType::LogAttrs, log_attrs_rb)
        .unwrap();
    records
        .set(ArrowPayloadType::ResourceAttrs, res_attrs_rb)
        .unwrap();
    records
}

/// Generates `num_policies` deterministic `Policy` definitions for `profile`.
#[must_use]
pub fn generate_policies(num_policies: usize, profile: WorkloadProfile) -> Vec<Policy> {
    let mut policies = Vec::with_capacity(num_policies);
    for i in 0..num_policies {
        let id = format!("policy_{i:06}");
        match profile {
            WorkloadProfile::W1ProductionMix => {
                // 60% structured equality/existence, 15% numeric range, 15% contains, 10% regex
                let bucket = i % 20;
                let action = if i % 7 == 0 {
                    Action::Keep
                } else {
                    Action::Drop
                };
                let matcher = if bucket < 6 {
                    // Structured equality on log_attribute["http.route"] or event_name
                    LogMatcher {
                        field: LogFieldSelector::LogAttribute(AttributePath::single("http.route")),
                        predicate: Predicate::Equals(ScalarValue::String(format!(
                            "/api/v1/orders/{}",
                            1000 + i
                        ))),
                        negate: false,
                    }
                } else if bucket < 12 {
                    // Structured equality on event_name (64 unique values across policies, exercising predicate deduplication)
                    LogMatcher {
                        field: LogFieldSelector::RecordField(LogRecordField::EventName),
                        predicate: Predicate::Equals(ScalarValue::String(format!(
                            "event.domain.action_{}",
                            i % 64
                        ))),
                        negate: false,
                    }
                } else if bucket < 15 {
                    // Numeric range on log_attribute["http.status_code"]
                    LogMatcher {
                        field: LogFieldSelector::LogAttribute(AttributePath::single(
                            "http.status_code",
                        )),
                        predicate: Predicate::Gte(NumericValue::Int(500 + (i as i64 % 50))),
                        negate: false,
                    }
                } else if bucket < 18 {
                    // Body substring contains
                    let needle = match i % 3 {
                        0 => format!("connection reset #{}", i / 3),
                        1 => format!("deadline exceeded #{}", i / 3),
                        _ => format!("OutOfMemory shard={}", i % 32),
                    };
                    LogMatcher {
                        field: LogFieldSelector::RecordField(LogRecordField::Body),
                        predicate: Predicate::Contains(ScalarValue::String(needle)),
                        negate: false,
                    }
                } else {
                    // Body regex with literal prefix
                    LogMatcher {
                        field: LogFieldSelector::RecordField(LogRecordField::Body),
                        predicate: Predicate::Regex(format!(
                            r"POST /api/v[12]/orders/{}[0-9]*",
                            1000 + (i % 500)
                        )),
                        negate: false,
                    }
                };
                policies.push(Policy::log_filter(id, vec![matcher], action));
            }
            WorkloadProfile::W2LiteralRegex => {
                let matcher = if i % 2 == 0 {
                    LogMatcher {
                        field: LogFieldSelector::RecordField(LogRecordField::Body),
                        predicate: Predicate::Contains(ScalarValue::String(format!(
                            "orders/{} user_id=",
                            1000 + i
                        ))),
                        negate: false,
                    }
                } else {
                    LogMatcher {
                        field: LogFieldSelector::RecordField(LogRecordField::Body),
                        predicate: Predicate::Regex(format!(
                            r"ERROR \[auth\] connection reset.*{:04x}",
                            i % 4096
                        )),
                        negate: false,
                    }
                };
                policies.push(Policy::log_filter(id, vec![matcher], Action::Drop));
            }
            WorkloadProfile::W3AdversarialRegex => {
                // Character-class regexes with no extractable literal > 2 bytes
                let pat = format!(
                    r"[a-f0-9]{{8}}-{:02x}[a-f0-9]{{2}}-[a-f0-9]{{4}}",
                    i & 0xff
                );
                let matcher = LogMatcher {
                    field: LogFieldSelector::RecordField(LogRecordField::Body),
                    predicate: Predicate::Regex(pat),
                    negate: false,
                };
                policies.push(Policy::log_filter(id, vec![matcher], Action::Drop));
            }
            WorkloadProfile::W4aFilterRenameRemove => {
                if i % 2 == 0 {
                    let matcher = LogMatcher {
                        field: LogFieldSelector::LogAttribute(AttributePath::single("http.route")),
                        predicate: Predicate::Equals(ScalarValue::String(format!(
                            "/api/v1/orders/{}",
                            1000 + i
                        ))),
                        negate: false,
                    };
                    policies.push(Policy::log_filter(id, vec![matcher], Action::Drop));
                } else {
                    let matcher = LogMatcher {
                        field: LogFieldSelector::RecordField(LogRecordField::SeverityText),
                        predicate: Predicate::Equals(ScalarValue::String(
                            SEVERITIES[i % SEVERITIES.len()].0.to_owned(),
                        )),
                        negate: false,
                    };
                    let tx = LogTransformPolicy {
                        matches: vec![matcher],
                        rename: vec![LogRename {
                            from: LogTargetSelector::LogAttribute(AttributePath::single("env")),
                            to: LogTargetSelector::LogAttribute(AttributePath::single(
                                "deployment.environment",
                            )),
                            upsert: true,
                        }],
                        add: Vec::new(),
                        remove: vec![LogRemove {
                            target: LogTargetSelector::LogAttribute(AttributePath::single(
                                "latency_ms",
                            )),
                        }],
                        redact: Vec::new(),
                    };
                    policies.push(Policy::log_transform(id, tx));
                }
            }
            WorkloadProfile::W4bFullSpecTransforms => {
                if i % 2 == 0 {
                    let action = if i % 6 == 0 {
                        Action::Keep
                    } else {
                        Action::Drop
                    };
                    let matcher = LogMatcher {
                        field: LogFieldSelector::LogAttribute(AttributePath::single(
                            "http.status_code",
                        )),
                        predicate: Predicate::Equals(ScalarValue::Int(500 + (i as i64 % 20))),
                        negate: false,
                    };
                    policies.push(Policy::log_filter(id, vec![matcher], action));
                } else {
                    let matcher = LogMatcher {
                        field: LogFieldSelector::RecordField(LogRecordField::SeverityText),
                        predicate: Predicate::Equals(ScalarValue::String(
                            SEVERITIES[i % SEVERITIES.len()].0.to_owned(),
                        )),
                        negate: false,
                    };
                    let tx = LogTransformPolicy {
                        matches: vec![matcher],
                        rename: vec![LogRename {
                            from: LogTargetSelector::LogAttribute(AttributePath::single("env")),
                            to: LogTargetSelector::LogAttribute(AttributePath::single(
                                "deployment.environment",
                            )),
                            upsert: false,
                        }],
                        add: vec![LogAdd {
                            target: LogTargetSelector::LogAttribute(AttributePath::single(
                                "policy.applied",
                            )),
                            value: ScalarValue::String(id.clone()),
                            upsert: true,
                        }],
                        remove: vec![LogRemove {
                            target: LogTargetSelector::LogAttribute(AttributePath::single(
                                "latency_ms",
                            )),
                        }],
                        redact: vec![LogRedact {
                            target: LogTargetSelector::RecordField(LogRecordField::Body),
                            replacement: Some("ssn=[REDACTED-$1]".to_owned()),
                            regex: Some(r"ssn=([0-9]{3})-[0-9]{2}-[0-9]{4}".to_owned()),
                        }],
                    };
                    policies.push(Policy::log_transform(id, tx));
                }
            }
        }
    }
    policies
}
