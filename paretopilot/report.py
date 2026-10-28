import csv
import html
import math
from .workload import shapes
from .util import read_json


def write_baseline_report(root, config, baseline):
    inputs, output = shapes(config["operator"])
    title = f"MatMul baseline: {inputs[0]} x {inputs[1]} -> {output}" if config["operator"]["kind"] == "matmul" else "Add baseline"
    duration = baseline["duration_us"]
    note = "Baseline measurement only. Candidate tuning and speedup were not evaluated."
    if baseline.get("execution") == "kernel":
        if baseline.get("timing_method") == "kernel_task_duration_us":
            note += " Timing: direct kernel profiler Task Duration(us) P50. Event intervals are diagnostic only."
        else:
            note += " Timing: device-event interval P50, including launch submission gaps; not OM Task Duration(us)."
    identity = baseline.get("operator_identity", {})
    routing = read_json(root/"routing.json") if (root/"routing.json").is_file() else {}
    observed = f'Observed operator: {identity.get("Op Type", "unavailable")}; node: {identity.get("Op Name", "unavailable")}. '
    observed += routing.get("reason", "")
    roofline_note = "Roofline uses configured hardware assumptions and shape minimum traffic unless measured traffic is supplied. Cache reuse and compiler lowering can change actual traffic."
    svg = roofline_svg(config, baseline, [])
    (root/"roofline.svg").write_text(svg, encoding="utf-8")
    (root/"report.md").write_text(f"# {title}\n\n{note}\n\n{observed}\n\nTarget task median: {duration:.4f} us.\n\n{roofline_note}\n", encoding="utf-8")
    (root/"report.html").write_text(
        f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{html.escape(title)}</title>'
        f'<body><h1>{html.escape(title)}</h1><p>{note}</p><p>{html.escape(observed)}</p><p>Target task median: {duration:.4f} us.</p>'
        f'<p>{roofline_note}</p>{svg}</body></html>', encoding="utf-8")


def roofline_svg(config, baseline, trials):
    peak = config["hardware"]["peak_flops"]/1e12
    bandwidth = config["hardware"]["bandwidth_bytes_s"]/1e12
    good = [baseline] + [t for t in trials if t["status"] == "ok"]
    xs = [t["roofline"]["arithmetic_intensity"] for t in good]
    performance = [t["roofline"]["flops"]/(t["duration_us"]*1e6) for t in good]
    xmin, xmax = min(xs+[peak/bandwidth])/10, max(xs+[peak/bandwidth])*10
    ymin, ymax = min(performance)/10, max([peak, *performance])*2
    ymin = min(ymin, ymax/100)
    def x(v): return 65+620*math.log(v/xmin)/math.log(xmax/xmin)
    def y(v): return 340-280*math.log(max(v,ymin)/ymin)/math.log(ymax/ymin)
    curve = ' '.join(f'{x(v):.2f},{y(min(peak,bandwidth*v)):.2f}' for v in [xmin*(xmax/xmin)**(i/100) for i in range(101)])
    points = ''.join(f'<circle cx="{x(t["roofline"]["arithmetic_intensity"]):.2f}" cy="{y(t["roofline"]["flops"]/(t["duration_us"]*1e6)):.2f}" r="4" fill="{ "#e45756" if i == 0 else "#1976d2" }"><title>{html.escape(t["role"])}: {t["duration_us"]:.4f} us</title></circle>' for i,t in enumerate(good))
    ticks = ''
    for i in range(5):
        vx, vy = xmin*(xmax/xmin)**(i/4), ymin*(ymax/ymin)**(i/4)
        ticks += f'<text x="{x(vx):.1f}" y="360" text-anchor="middle" font-size="11">{vx:.2g}</text><text x="60" y="{y(vy)+4:.1f}" text-anchor="end" font-size="11">{vy:.2g}</text>'
    basis = ', '.join(sorted({t['roofline']['traffic_basis'] for t in good}))
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="430"><rect width="100%" height="100%" fill="white"/><path d="M65 60V340H700" fill="none" stroke="#555"/>{ticks}<polyline points="{curve}" fill="none" stroke="#444" stroke-width="2"/>{points}<text x="180" y="385">Arithmetic intensity (FLOP/byte), log scale</text><text x="70" y="35">TFLOP/s, log scale; red = baseline; blue = candidates</text><text x="70" y="407">Traffic basis: {basis}; configured hardware ceilings are assumptions.</text></svg>'


def write_report(root, result):
    from .util import read_json
    config = read_json(root/"manifest.json")["config"]
    primary = result.get("primary_baseline") == "source_baseline"
    plot_baseline = result["source_baseline"] if primary else result["baseline"]
    references = [result["baseline"]] if primary else [result["source_baseline"]] if "source_baseline" in result else []
    (root/"roofline.svg").write_text(roofline_svg(config, plot_baseline, references+result["history"]), encoding="utf-8")
    with (root/"trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["generation", "status", "duration_us", "params", "error", "timing_method"])
        for t in result["history"]:
            writer.writerow([t["generation"], t["status"], t.get("duration_us"), str(t["params"]), t.get("error", ""), t.get("timing_method", "task_duration_us" if result["mode"] == "ascend" else "synthetic")])
    text = f"""# ParetoPilot kernel tuning

Backend: **{result['mode']}**. Agent: **{result['agent']['kind']}**.

Selected: **{result['selected']}**. {result['confirmation']['reason']}.

Selected parameters: {result['selected_params']}.

Baseline{' (default source)' if primary else ''}: {plot_baseline['duration_us']:.4f} us. Successful candidates: {sum(t['status']=='ok' for t in result['history'])}/{len(result['history'])}.

Metric: OM target-op Task Duration(us) median. PyTorch is the correctness reference.

Experiments: {len(result['history'])}. See summary.json, trials.csv and agent/ for complete evidence.

![Roofline](roofline.svg)

Confirmation compares interleaved fresh baseline/candidate builds. The range threshold is a heuristic, not a confidence interval.
"""
    if result["mode"] == "simulation":
        text += "\nAll timings and correctness outcomes are synthetic. No hardware improvement has been measured.\n"
    if result.get("source_baseline", {}).get("execution") == "kernel":
        profiler_search = result["source_baseline"].get("timing_method") == "kernel_task_duration_us"
        text = text.replace("Metric: OM target-op Task Duration(us) median.",
                            ("Search metric: direct kernel profiler Task Duration(us) P50; Event intervals are diagnostic only. " if profiler_search else
                             "Search metric: direct kernel device-event interval P50 (includes launch submission gaps), not OM task duration. ") +
                            "Stock reference and final confirmation use OM Task Duration(us) P50. Compare only timings from the same method.")
        text += "\nSearch uses a persistent direct kernel launcher; gain acceptance uses fresh ATC/OM confirmation only.\n"
    if "source_baseline" in result:
        final = result["confirmation"]
        source_operator = result["source_baseline"]["selection"].get("operator", "MatMulV3")
        text += (f"\nOfficial source: CANN ops-nn 8.5.0, {source_operator}; commit {result['source_baseline']['selection']['commit']}.\n"
                 "\nThe installed stock baseline and this source build are different implementations.\n"
                 "\nConfirmation task medians (us), not build/stage wall-clock seconds:\n\n"
                 f"- Stock CANN: {final.get('baseline_us', 'unavailable')}\n"
                 f"- Official source defaults: {final.get('source_baseline_us', 'unavailable')}\n"
                 f"- Tuned source candidate: {final.get('candidate_us', 'unavailable')}\n"
                 f"- Candidate speedup over stock: {final.get('speedup', 'unavailable')}x\n"
                 f"- Candidate speedup over source defaults: {final.get('source_speedup', 'unavailable')}x\n"
                 f"- Tuning gain accepted: {final.get('tuning_accepted', False)}\n"
                 "\nDefault core_limit=0 preserves device core count; tiling_policy=0 preserves upstream knowledge-bank lookup with heuristic fallback.\n"
                 "\nSource/default/candidate are rebuilt and measured in alternating order. Identical default parameters never establish tuning gain.\n")
    if result.get("primary_baseline") == "source_baseline":
        final = result["confirmation"]
        stock_type = result["baseline"].get("operator_identity", {}).get("Op Type", "MatMul")
        text += ("\nPrimary optimization baseline: default BatchMatMulV3 from the same pinned source. "
                 f"Stock {stock_type} is a performance reference only and does not gate acceptance or become the selected source implementation.\n"
                 f"\nPrimary V3 baseline: {final.get('primary_baseline_us', result['source_baseline']['duration_us'])} us; "
                 f"V3 tuning speedup: {final.get('primary_speedup', 'unavailable')}x.\n"
                 "\nHost controls: core_limit=0 keeps the physical core budget; tiling_policy=0 keeps upstream KB lookup; "
                 "batch_strategy=0 keeps upstream automatic L1/multibatch choices, 1 keeps common tiling. "
                 "base_m/base_n=0 retain automatic tiles; explicit pairs use SetFixSplit(M,N,-1) with heuristic/common policy. "
                 "baseK remains automatic. "
                 "No tile dimensions are overwritten after dependency calculation.\n")
    if "domain" in result:
        domain, cache = result["domain"], result.get("result_cache", {})
        text += (f"\nShape-generated domains: {domain['legal_candidates']} legal candidates; "
                 f"budget {domain['budget']}. See domain.json and effective_config.json.\n"
                 f"Historical parameter cache: {cache.get('status')}; search skipped after fresh retest: "
                 f"{cache.get('search_skipped', False)}. Final acceptance uses fresh OM confirmation.\n")
    (root/"report.md").write_text(text, encoding="utf-8")
    rows = ''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [t['generation'],t['status'],t.get('duration_us',''),t['params'],t.get('error','')])+'</tr>' for t in result['history'])
    page = '<!doctype html><meta charset="utf-8"><title>ParetoPilot</title><style>body{font:16px system-ui;margin:32px;max-width:1200px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap}</style>'
    page += f'<h1>ParetoPilot</h1><pre>{html.escape(text)}</pre><img src="roofline.svg" alt="Roofline"><table><tr><th>Generation</th><th>Status</th><th>Task us</th><th>Parameters</th><th>Error</th></tr>{rows}</table>'
    (root/"report.html").write_text(page, encoding="utf-8")
