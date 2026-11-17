// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive multi-dimensional benchmark suite for `otel-arrow-dfe-policy-engine` (`open-telemetry/otel-arrow#4726`).
//! Measures throughput (`M logs/s`, `GiB/s`), `p50`/`p99` latency (`us`), compile time (`ms`),
//! compiled heap footprint (`KiB`), and per-batch heap allocations (`allocs/batch`, `bytes/batch`)
//! across Candidates 1-4 and Baselines A1-A3, B.

#![allow(
    unused_qualifications,
    unsafe_code,
    unused_results,
    variant_size_differences,
    missing_docs
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use arrow::array::RecordBatch;
use otel_arrow_dfe_pdata::otap::OtapArrowRecords;
use otel_arrow_dfe_pdata::proto::opentelemetry::arrow::v1::ArrowPayloadType;
use otel_arrow_dfe_policy_engine::PolicyEvaluationEngine;
use otel_arrow_dfe_policy_engine::baseline_vectorscan::{
    FixedVectorscanColumnarEngine, PolicyRsRowByRowEngine, UnmodifiedAlbertVectorscanEngine,
};
use otel_arrow_dfe_policy_engine::candidate1_hybrid::HybridPolicyEngine;
use otel_arrow_dfe_policy_engine::candidate2_regex_automata::ShardedRegexAutomataEngine;
use otel_arrow_dfe_policy_engine::candidate3_trigram::TrigramPolicyEngine;
use otel_arrow_dfe_policy_engine::candidate4_opl::{
    CompiledOplPolicyEngine, IdiomaticOplTextEngine,
};
use otel_arrow_dfe_policy_engine::workload::{
    DictCardinality, WorkloadProfile, generate_logs_batch, generate_policies,
};

struct CountingAlloc;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static NET_LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        NET_LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        NET_LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
}

fn reset_alloc_counters() -> (usize, usize) {
    (
        ALLOC_COUNT.load(Ordering::Relaxed),
        ALLOC_BYTES.load(Ordering::Relaxed),
    )
}

fn batch_arrow_bytes(batch: &OtapArrowRecords) -> usize {
    let mut total = 0usize;
    let rb_bytes = |rb: &RecordBatch| -> usize { rb.get_array_memory_size() };
    if let Some(rb) = batch.get(ArrowPayloadType::Logs) {
        total += rb_bytes(rb);
    }
    if let Some(rb) = batch.get(ArrowPayloadType::LogAttrs) {
        total += rb_bytes(rb);
    }
    if let Some(rb) = batch.get(ArrowPayloadType::ResourceAttrs) {
        total += rb_bytes(rb);
    }
    if let Some(rb) = batch.get(ArrowPayloadType::ScopeAttrs) {
        total += rb_bytes(rb);
    }
    total
}

struct BenchRow {
    workload: &'static str,
    cardinality: &'static str,
    batch_size: usize,
    num_policies: usize,
    engine: &'static str,
    mode: &'static str,
    compile_ms: f64,
    heap_kib: f64,
    p50_us: f64,
    p99_us: f64,
    m_logs_per_sec: f64,
    gib_per_sec: f64,
    allocs_per_batch: usize,
    alloc_bytes_per_batch: usize,
}

fn run_single_engine_bench<F>(
    workload: WorkloadProfile,
    cardinality: DictCardinality,
    batch: &OtapArrowRecords,
    batch_bytes: usize,
    batch_size: usize,
    num_policies: usize,
    mode_full_eval: bool,
    compile_fn: F,
) -> Option<BenchRow>
where
    F: FnOnce() -> Result<Box<dyn PolicyEvaluationEngine>, otel_arrow_dfe_policy_engine::PolicyError>,
{
    let live_before = NET_LIVE_BYTES.load(Ordering::Relaxed);
    let t0 = Instant::now();
    let engine = compile_fn().ok()?;
    let compile_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let live_after = NET_LIVE_BYTES.load(Ordering::Relaxed);
    let net_alloc_heap = live_after.saturating_sub(live_before);
    let heap_bytes = engine.estimated_heap_bytes().max(net_alloc_heap);
    let heap_kib = heap_bytes as f64 / 1024.0;

    let is_opl = engine.name().contains("OPL");
    let is_row_by_row = engine.name().contains("RowByRow");
    let warmup_iters = if is_row_by_row || (is_opl && num_policies >= 100) || num_policies >= 10_000 || batch_size >= 10_000 {
        1
    } else {
        4
    };
    for _ in 0..warmup_iters {
        if mode_full_eval {
            let _ = std::hint::black_box(engine.evaluate(batch.clone()).ok()?);
        } else {
            let _ = std::hint::black_box(engine.evaluate_filter(batch).ok()?);
        }
    }

    // Measure single-batch allocation overhead (without cloning batch for filter mode)
    let (c0, b0) = reset_alloc_counters();
    if mode_full_eval {
        let cloned = batch.clone();
        let (c_inner0, b_inner0) = reset_alloc_counters();
        let _ = std::hint::black_box(engine.evaluate(cloned).ok()?);
        let (c_inner1, b_inner1) = reset_alloc_counters();
        let _ = (c0, b0);
        let allocs_per_batch = c_inner1.saturating_sub(c_inner0);
        let alloc_bytes_per_batch = b_inner1.saturating_sub(b_inner0);

        let iters = if (is_opl && num_policies >= 100) || num_policies >= 10_000 || batch_size >= 10_000 {
            10
        } else if num_policies >= 1_000 {
            25
        } else {
            60
        };
        let mut samples_us = Vec::with_capacity(iters);
        for _ in 0..iters {
            let cloned_batch = batch.clone();
            let start = Instant::now();
            let _ = std::hint::black_box(engine.evaluate(cloned_batch).ok()?);
            samples_us.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        }
        samples_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50_us = samples_us[samples_us.len() / 2];
        let p99_us = samples_us[(samples_us.len() * 99) / 100];
        let mean_us: f64 = samples_us.iter().sum::<f64>() / samples_us.len() as f64;
        let m_logs_per_sec = (batch_size as f64) / mean_us;
        let gib_per_sec =
            (batch_bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / (mean_us / 1_000_000.0);

        let row = BenchRow {
            workload: workload.label(),
            cardinality: cardinality.label(),
            batch_size,
            num_policies,
            engine: engine.name(),
            mode: "filter+transform",
            compile_ms,
            heap_kib,
            p50_us,
            p99_us,
            m_logs_per_sec,
            gib_per_sec,
            allocs_per_batch,
            alloc_bytes_per_batch,
        };
        eprintln!(
            "  [{} | {} | B={} | N={} | {}] {} -> compile={:.2}ms heap={:.1}KiB p50={:.1}us p99={:.1}us {:.2}M/s ({:.2}GiB/s) allocs={}",
            row.workload,
            row.cardinality,
            row.batch_size,
            row.num_policies,
            row.mode,
            row.engine,
            row.compile_ms,
            row.heap_kib,
            row.p50_us,
            row.p99_us,
            row.m_logs_per_sec,
            row.gib_per_sec,
            row.allocs_per_batch
        );
        Some(row)
    } else {
        let _ = std::hint::black_box(engine.evaluate_filter(batch).ok()?);
        let (c1, b1) = reset_alloc_counters();
        let allocs_per_batch = c1.saturating_sub(c0);
        let alloc_bytes_per_batch = b1.saturating_sub(b0);

        let iters = if is_row_by_row || (is_opl && num_policies >= 1_000) {
            3
        } else if (is_opl && num_policies >= 100)
            || num_policies >= 10_000
            || batch_size >= 10_000
        {
            15
        } else if num_policies >= 1_000 {
            30
        } else {
            80
        };
        let mut samples_us = Vec::with_capacity(iters);
        for _ in 0..iters {
            let start = Instant::now();
            let _ = std::hint::black_box(engine.evaluate_filter(batch).ok()?);
            samples_us.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        }
        samples_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50_us = samples_us[samples_us.len() / 2];
        let p99_us = samples_us[(samples_us.len() * 99) / 100];
        let mean_us: f64 = samples_us.iter().sum::<f64>() / samples_us.len() as f64;
        let m_logs_per_sec = (batch_size as f64) / mean_us;
        let gib_per_sec =
            (batch_bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / (mean_us / 1_000_000.0);

        let row = BenchRow {
            workload: workload.label(),
            cardinality: cardinality.label(),
            batch_size,
            num_policies,
            engine: engine.name(),
            mode: "filter_only",
            compile_ms,
            heap_kib,
            p50_us,
            p99_us,
            m_logs_per_sec,
            gib_per_sec,
            allocs_per_batch,
            alloc_bytes_per_batch,
        };
        eprintln!(
            "  [{} | {} | B={} | N={} | {}] {} -> compile={:.2}ms heap={:.1}KiB p50={:.1}us p99={:.1}us {:.2}M/s ({:.2}GiB/s) allocs={}",
            row.workload,
            row.cardinality,
            row.batch_size,
            row.num_policies,
            row.mode,
            row.engine,
            row.compile_ms,
            row.heap_kib,
            row.p50_us,
            row.p99_us,
            row.m_logs_per_sec,
            row.gib_per_sec,
            row.allocs_per_batch
        );
        Some(row)
    }
}

fn main() {
    let mut rows: Vec<BenchRow> = Vec::new();

    // 1. Primary Scaling Matrix across W1, W2, W3 (Batch=1000, Dict256 & Plain, N = 1, 10, 100, 1000, 10000)
    for &profile in &[
        WorkloadProfile::W1ProductionMix,
        WorkloadProfile::W2LiteralRegex,
        WorkloadProfile::W3AdversarialRegex,
    ] {
        for &card in &[
            DictCardinality::Low16,
            DictCardinality::Medium256,
            DictCardinality::Plain,
        ] {
            let batch_size = 1000usize;
            let batch = generate_logs_batch(batch_size, card);
            let batch_bytes = batch_arrow_bytes(&batch);

            let policy_counts: &[usize] = if card == DictCardinality::Low16 {
                &[10, 100, 1000]
            } else {
                &[1, 10, 100, 1000, 10000]
            };

            for &n in policy_counts {
                let policies = generate_policies(n, profile);
                eprintln!(
                    "Running profile={} card={} B={} N={} ...",
                    profile.label(),
                    card.label(),
                    batch_size,
                    n
                );

                if let Some(r) = run_single_engine_bench(
                    profile,
                    card,
                    &batch,
                    batch_bytes,
                    batch_size,
                    n,
                    false,
                    || Ok(Box::new(HybridPolicyEngine::compile(&policies)?)),
                ) {
                    rows.push(r);
                }

                if let Some(r) = run_single_engine_bench(
                    profile,
                    card,
                    &batch,
                    batch_bytes,
                    batch_size,
                    n,
                    false,
                    || Ok(Box::new(ShardedRegexAutomataEngine::compile(&policies)?)),
                ) {
                    rows.push(r);
                }

                if let Some(r) = run_single_engine_bench(
                    profile,
                    card,
                    &batch,
                    batch_bytes,
                    batch_size,
                    n,
                    false,
                    || Ok(Box::new(TrigramPolicyEngine::compile(&policies)?)),
                ) {
                    rows.push(r);
                }

                // Candidate 4 & Baseline B (OPL DataFusion query-engine): run up to N = 100 across all profiles,
                // plus N = 1000 on W1_prod_mix dict256 to measure the high-N OPL scaling point.
                if n <= 100
                    || (profile == WorkloadProfile::W1ProductionMix
                        && card == DictCardinality::Medium256
                        && n == 1000)
                {
                    if let Some(r) = run_single_engine_bench(
                        profile,
                        card,
                        &batch,
                        batch_bytes,
                        batch_size,
                        n,
                        false,
                        || Ok(Box::new(CompiledOplPolicyEngine::compile(&policies)?)),
                    ) {
                        rows.push(r);
                    }
                    if n <= 100 {
                        if let Some(r) = run_single_engine_bench(
                            profile,
                            card,
                            &batch,
                            batch_bytes,
                            batch_size,
                            n,
                            false,
                            || Ok(Box::new(IdiomaticOplTextEngine::compile(&policies)?)),
                        ) {
                            rows.push(r);
                        }
                    }
                }

                // Baselines A1, A2, A3 (Vectorscan / policy-rs)
                // Note: PolicyRsRowByRowEngine takes ~80ms/batch at N=10 and ~1.27s/batch at N=100 due to
                // per-row hs_clone_scratch and relational attribute table lookups.
                if n <= 10 || (n == 100 && card == DictCardinality::Medium256) {
                    if let Some(r) = run_single_engine_bench(
                        profile,
                        card,
                        &batch,
                        batch_bytes,
                        batch_size,
                        n,
                        false,
                        || Ok(Box::new(PolicyRsRowByRowEngine::compile(&policies)?)),
                    ) {
                        rows.push(r);
                    }
                }
                if profile != WorkloadProfile::W1ProductionMix {
                    // Note: UnmodifiedAlbertVectorscanEngine panics on W1's Int32 severity_number comparison
                    // and silently no-ops on body StructArray, which we record on W2/W3 at N=100 for reference.
                    if n == 100 && card == DictCardinality::Medium256 {
                        if let Some(r) = run_single_engine_bench(
                            profile,
                            card,
                            &batch,
                            batch_bytes,
                            batch_size,
                            n,
                            false,
                            || {
                                Ok(Box::new(UnmodifiedAlbertVectorscanEngine::compile(
                                    &policies,
                                )?))
                            },
                        ) {
                            rows.push(r);
                        }
                    }
                }
                if n <= 1000 || profile == WorkloadProfile::W1ProductionMix {
                    if let Some(r) = run_single_engine_bench(
                        profile,
                        card,
                        &batch,
                        batch_bytes,
                        batch_size,
                        n,
                        false,
                        || Ok(Box::new(FixedVectorscanColumnarEngine::compile(&policies)?)),
                    ) {
                        rows.push(r);
                    }
                }
            }
        }
    }

    // 2. Batch-Size Sensitivity Sweep (B = 100, 1000, 10000 on W1, Plain & Dict256, N = 100 and N = 1000)
    for &batch_size in &[100usize, 10000] {
        for &card in &[DictCardinality::Medium256, DictCardinality::Plain] {
            let batch = generate_logs_batch(batch_size, card);
            let batch_bytes = batch_arrow_bytes(&batch);
            for &n in &[100usize, 1000] {
                let policies = generate_policies(n, WorkloadProfile::W1ProductionMix);
                for compile_fn in [
                    (&|| {
                        Ok(Box::new(HybridPolicyEngine::compile(&policies)?)
                            as Box<dyn PolicyEvaluationEngine>)
                    })
                        as &dyn Fn() -> Result<
                            Box<dyn PolicyEvaluationEngine>,
                            otel_arrow_dfe_policy_engine::PolicyError,
                        >,
                    &|| Ok(Box::new(ShardedRegexAutomataEngine::compile(&policies)?)),
                    &|| Ok(Box::new(TrigramPolicyEngine::compile(&policies)?)),
                    &|| Ok(Box::new(FixedVectorscanColumnarEngine::compile(&policies)?)),
                ] {
                    if let Some(r) = run_single_engine_bench(
                        WorkloadProfile::W1ProductionMix,
                        card,
                        &batch,
                        batch_bytes,
                        batch_size,
                        n,
                        false,
                        compile_fn,
                    ) {
                        rows.push(r);
                    }
                }
            }
        }
    }

    // 3. Extreme Scale Stress Test (N = 50,000 policies on W1, B = 1000, Dict256)
    {
        let batch_size = 1000usize;
        let card = DictCardinality::Medium256;
        let batch = generate_logs_batch(batch_size, card);
        let batch_bytes = batch_arrow_bytes(&batch);
        let n = 50_000usize;
        let policies = generate_policies(n, WorkloadProfile::W1ProductionMix);
        eprintln!("Running extreme scale N=50,000 on W1...");
        if let Some(r) = run_single_engine_bench(
            WorkloadProfile::W1ProductionMix,
            card,
            &batch,
            batch_bytes,
            batch_size,
            n,
            false,
            || Ok(Box::new(HybridPolicyEngine::compile(&policies)?)),
        ) {
            rows.push(r);
        }
        if let Some(r) = run_single_engine_bench(
            WorkloadProfile::W1ProductionMix,
            card,
            &batch,
            batch_bytes,
            batch_size,
            n,
            false,
            || Ok(Box::new(TrigramPolicyEngine::compile(&policies)?)),
        ) {
            rows.push(r);
        }
    }

    // 4. End-to-End Filter + Transform Pipeline (W4a and W4b, B = 1000, Dict256, N = 10, 100, 1000)
    for &profile in &[
        WorkloadProfile::W4aFilterRenameRemove,
        WorkloadProfile::W4bFullSpecTransforms,
    ] {
        let card = DictCardinality::Medium256;
        let batch_size = 1000usize;
        let batch = generate_logs_batch(batch_size, card);
        let batch_bytes = batch_arrow_bytes(&batch);
        for &n in &[10usize, 100, 1000] {
            let policies = generate_policies(n, profile);
            eprintln!("Running transform pipeline {} N={}...", profile.label(), n);

            if let Some(r) = run_single_engine_bench(
                profile,
                card,
                &batch,
                batch_bytes,
                batch_size,
                n,
                true,
                || Ok(Box::new(HybridPolicyEngine::compile(&policies)?)),
            ) {
                rows.push(r);
            }
            if let Some(r) = run_single_engine_bench(
                profile,
                card,
                &batch,
                batch_bytes,
                batch_size,
                n,
                true,
                || Ok(Box::new(ShardedRegexAutomataEngine::compile(&policies)?)),
            ) {
                rows.push(r);
            }
            if let Some(r) = run_single_engine_bench(
                profile,
                card,
                &batch,
                batch_bytes,
                batch_size,
                n,
                true,
                || Ok(Box::new(TrigramPolicyEngine::compile(&policies)?)),
            ) {
                rows.push(r);
            }
            if n <= 100 {
                if let Some(r) = run_single_engine_bench(
                    profile,
                    card,
                    &batch,
                    batch_bytes,
                    batch_size,
                    n,
                    true,
                    || Ok(Box::new(CompiledOplPolicyEngine::compile(&policies)?)),
                ) {
                    rows.push(r);
                }
            }
            if profile == WorkloadProfile::W4aFilterRenameRemove && n <= 1000 {
                if let Some(r) = run_single_engine_bench(
                    profile,
                    card,
                    &batch,
                    batch_bytes,
                    batch_size,
                    n,
                    true,
                    || Ok(Box::new(FixedVectorscanColumnarEngine::compile(&policies)?)),
                ) {
                    rows.push(r);
                }
            }
        }
    }

    // Format Markdown table output
    let mut md = String::new();
    md.push_str("# Raw Benchmark Output (`otel-arrow-dfe-policy-engine`)\n\n");
    md.push_str("| Workload | Card | Batch | Policies (N) | Mode | Engine | Compile (ms) | Heap (KiB) | p50 (us) | p99 (us) | Throughput (M logs/s) | GiB/s | Allocs/Batch | Alloc Bytes/Batch |\n");
    md.push_str("| :--- | :--- | ---: | ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for r in &rows {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {:.2} | {:.1} | {:.1} | {:.1} | {:.2} | {:.2} | {} | {} |\n",
            r.workload,
            r.cardinality,
            r.batch_size,
            r.num_policies,
            r.mode,
            r.engine,
            r.compile_ms,
            r.heap_kib,
            r.p50_us,
            r.p99_us,
            r.m_logs_per_sec,
            r.gib_per_sec,
            r.allocs_per_batch,
            r.alloc_bytes_per_batch,
        ));
    }

    let out_path = std::env::var_os("BENCHMARK_OUTPUT_PATH").map_or_else(
        || {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../../..")
                .join("BENCHMARK_RESULTS_RAW.md")
        },
        std::path::PathBuf::from,
    );
    std::fs::write(&out_path, &md).expect("write BENCHMARK_RESULTS_RAW.md");
    println!("{md}");
}
