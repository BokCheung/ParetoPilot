import csv
import html
import math


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
    (root/"roofline.svg").write_text(roofline_svg(config, result["baseline"], result["history"]), encoding="utf-8")
    with (root/"trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["generation", "status", "duration_us", "params", "error"])
        for t in result["history"]:
            writer.writerow([t["generation"], t["status"], t.get("duration_us"), str(t["params"]), t.get("error", "")])
    text = f"""# ParetoPilot kernel tuning

Backend: **{result['mode']}**. Agent: **{result['agent']['kind']}**.

Selected: **{result['selected']}**. {result['confirmation']['reason']}.

Selected parameters: {result['selected_params']}.

Baseline: {result['baseline']['duration_us']:.4f} us. Successful candidates: {sum(t['status']=='ok' for t in result['history'])}/{len(result['history'])}.

Metric: OM target-op Task Duration(us) median. PyTorch is the correctness reference.

Experiments: {len(result['history'])}. See summary.json, trials.csv and agent/ for complete evidence.

![Roofline](roofline.svg)

Confirmation compares interleaved fresh baseline/candidate builds. The range threshold is a heuristic, not a confidence interval.
"""
    if result["mode"] == "simulation":
        text += "\nAll timings and correctness outcomes are synthetic. No hardware improvement has been measured.\n"
    (root/"report.md").write_text(text, encoding="utf-8")
    rows = ''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [t['generation'],t['status'],t.get('duration_us',''),t['params'],t.get('error','')])+'</tr>' for t in result['history'])
    page = '<!doctype html><meta charset="utf-8"><title>ParetoPilot</title><style>body{font:16px system-ui;margin:32px;max-width:1200px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap}</style>'
    page += f'<h1>ParetoPilot</h1><pre>{html.escape(text)}</pre><img src="roofline.svg" alt="Roofline"><table><tr><th>Generation</th><th>Status</th><th>Task us</th><th>Parameters</th><th>Error</th></tr>{rows}</table>'
    (root/"report.html").write_text(page, encoding="utf-8")
