#!/usr/bin/env python3
"""Steady-state decode throughput per reader from vLLM logs (only intervals with a full batch: Running >= 20),
token budget per page, and full-archive cost under several max_tokens caps. Writes cost_model.json."""
import json, re, statistics as st, sys
# usage: python bench/cost_model.py <reader_out_dir> <embed_bench.json>
D = sys.argv[1] if len(sys.argv) > 1 else "out3"
EMB = sys.argv[2] if len(sys.argv) > 2 else "out/embed_bench.json"
PAGES, OVER = 12_200_000, 1.15
PRICE = {"4090_community": 0.34, "4090_secure": 0.74}
emb = json.load(open(EMB))["throughput"]
res = {"readers": {}, "embed": {}}
for t in ("2b", "4b", "8b"):
    s = json.load(open(f"{D}/vlmstats_{t}.json")); pages = json.load(open(f"{D}/vlm_{t}.json"))
    gen = []; run = []
    for line in open(f"{D}/serve_{t}.log", errors="replace"):
        m = re.search(r"generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs", line)
        if m and int(m.group(3)) > 0 and float(m.group(1)) > 0:     # saturated: requests queued, GPU at its batch limit
            gen.append(float(m.group(1))); run.append(int(m.group(2)))
    n_len = s["finish"].get("length", 0); n_ok = s["ok"]
    good_tok = (s["out_tok"] - n_len * 8000) / max(1, n_ok)         # runaway pages hit the 8000 cap exactly
    ss = st.median(gen) if gen else None
    r = {"valid_frac": round(n_ok / s["pages"], 3), "runaway_frac": round(n_len / s["pages"], 3),
         "out_tok_per_valid_page": round(good_tok), "in_tok_per_page": round(s["in_tok"] / s["pages"]),
         "steady_gen_tok_s_median": ss, "n_saturated_intervals": len(gen), "median_running_seqs": st.median(run) if run else None,
         "measured_pages_per_s_whole_run": s["pages_per_s"], "wall_s": s["wall_s"]}
    for cap in (8000, 4000, 3000):
        tok = (n_ok * good_tok + n_len * cap) / s["pages"]               # avg decode tokens per page incl. failures
        pps = ss / tok if ss else None
        h = PAGES / pps / 3600 * OVER if pps else None
        r[f"cap{cap}"] = {"tok_per_page": round(tok), "pages_per_s": round(pps, 3) if pps else None,
                          "gpu_h": round(h) if h else None, **({k: round(h * v) for k, v in PRICE.items()} if h else {})}
    h0 = PAGES / s["pages_per_s"] / 3600 * OVER
    r["whole_run_rate"] = {"gpu_h": round(h0), **{k: round(h0 * v) for k, v in PRICE.items()}}
    res["readers"][t] = r
for k in ("text", "image@280", "interleaved@280"):
    h = PAGES / emb[k] / 3600 * OVER
    res["embed"][k] = {"pages_per_s": emb[k], "gpu_h": round(h), **{n: round(h * v) for n, v in PRICE.items()}}
json.dump(res, open(f"{D}/cost_model.json", "w"), indent=1)
print(json.dumps(res, indent=1))
