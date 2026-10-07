#!/usr/bin/env python3
"""
Run the pipeline's page-analysis step (Qwen3-VL via a local vLLM server, schema.EXTRACTION_PROMPT, JSON-schema
constrained) over the benchmark pages, so embed_bench.py can build page chunks from the VISION output instead
of archive.org's ABBYY OCR. Measures real VLM throughput on this GPU as a side effect.

    python vlm_pages.py --pages /root/pages/pages.json --out /root/out/pages_vlm.json --concurrency 8
Each page gains: vlm = {transcription, visual_elements, entities, legibility, ...} or vlm_error.
"""
import argparse, base64, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schema import PAGE_SCHEMA, EXTRACTION_PROMPT, validate_page, check_grounding, extract_json

REP_PEN = 1.05   # small VLMs loop on dense typed pages ("App App App..."); same value for every model compared

def call(endpoint, model, img, max_tokens):
    b64 = base64.b64encode(open(img, "rb").read()).decode()
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens, "repetition_penalty": REP_PEN,
            "response_format": {"type": "json_schema", "json_schema": {"name": "page", "schema": PAGE_SCHEMA}},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": EXTRACTION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]}
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    u = d.get("usage") or {}
    ch = d["choices"][0]
    return (ch["message"]["content"] or "", time.time() - t, u.get("prompt_tokens"), u.get("completion_tokens"),
            ch.get("finish_reason"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--concurrency", type=int, default=8); ap.add_argument("--max-tokens", type=int, default=8000)
    a = ap.parse_args()
    pages = json.load(open(a.pages))
    stats: dict = {"ok": 0, "invalid": 0, "error": 0, "in_tok": 0, "out_tok": 0}

    def one(p):
        try:
            txt, sec, it, ot, fin = call(a.endpoint, a.model, p["image"], a.max_tokens)
        except Exception as e:
            p["vlm_error"] = repr(e)[:200]; stats["error"] += 1; return
        stats["in_tok"] += it or 0; stats["out_tok"] += ot or 0
        p["finish_reason"] = fin; stats.setdefault("finish", {}); stats["finish"][fin] = stats["finish"].get(fin, 0) + 1
        obj = extract_json(txt)
        ok, errs = validate_page(obj) if obj is not None else (False, ["no json"])
        p["vlm_seconds"] = round(sec, 2)
        if ok:
            g_ok, ungrounded = check_grounding(obj)
            p["vlm"] = obj; p["ungrounded_entities"] = ungrounded; stats["ok"] += 1
            stats["ungrounded_pages"] = stats.get("ungrounded_pages", 0) + (0 if g_ok else 1)
        else:
            p["vlm_error"] = "; ".join(map(str, errs))[:300]; p["vlm_raw"] = txt[:2000]; stats["invalid"] += 1

    t = time.time()
    with ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(one, pages))
    wall = time.time() - t
    stats.update(pages=len(pages), wall_s=round(wall, 1), pages_per_s=round(len(pages) / wall, 3),
                 concurrency=a.concurrency, model=a.model)
    vis = [v for p in pages if p.get("vlm") for v in p["vlm"].get("visual_elements", [])]
    from collections import Counter
    stats["visual_types"] = dict(Counter(v.get("type") for v in vis))
    stats["pages_with_visuals"] = sum(1 for p in pages if p.get("vlm", {}).get("visual_elements"))
    json.dump(pages, open(a.out, "w"), indent=1)
    json.dump(stats, open(a.out.replace("vlm_", "vlmstats_"), "w"), indent=1)
    print(json.dumps(stats, indent=1))

if __name__ == "__main__":
    main()
