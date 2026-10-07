#!/usr/bin/env python3
"""
Compare page-reader models on the same pages (outputs of vlm_pages.py), and write embed-ready page files.

Agreement is measured against the largest model (reference) and against archive.org ABBYY OCR. Neither is ground
truth: high agreement with BOTH means the small model reads what the others read; it does not prove correctness.

    python vlm_compare.py --ref /root/out/vlm_8b.json /root/out/vlm_2b.json /root/out/vlm_4b.json --out /root/out
Writes <out>/vlm_compare.json and, per model, <out>/embed_pages_<tag>.json where
    text = transcription + visual element descriptions   (the page chunk the pipeline would index)
    abbyy kept for independent query generation; citation = official cia.gov PDF #page=N.
"""
import argparse, difflib, json, os, re
from collections import Counter

def words(t):
    return re.findall(r"[a-z0-9]+", (t or "").lower())

def wsim(a, b):
    a, b = words(a), words(b)
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()

def ents(v):
    return {(e.get("type"), " ".join(words(e.get("name")))) for e in (v or {}).get("entities") or [] if e.get("name")}

def vtypes(v):
    return Counter(x.get("type") for x in (v or {}).get("visual_elements") or [])

def chunk_text(p):
    v = p.get("vlm") or {}
    vis = "; ".join(f"{x.get('type')}: {x.get('description') or x.get('text') or ''}".strip(": ")
                    for x in v.get("visual_elements") or [])
    t = v.get("transcription") or ""
    return t + (f"\n[visual elements] {vis}" if vis else "")

def f1(a, b):
    if not a and not b:
        return 1.0
    tp = len(a & b)
    return 0.0 if tp == 0 else 2 * tp / (len(a) + len(b))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True); ap.add_argument("others", nargs="+"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tag = lambda path: os.path.basename(path).replace("vlm_", "").replace(".json", "")
    ref = {(p["doc_id"], p["page"]): p for p in json.load(open(a.ref))}
    report = {}
    for path in [a.ref] + a.others:
        t = tag(path); pages = json.load(open(path)); stats = json.load(open(path.replace("vlm_", "vlmstats_")))
        ok = [p for p in pages if p.get("vlm")]
        r = {"pages": len(pages), "valid": len(ok), "invalid": stats.get("invalid"), "errors": stats.get("error"),
             "pages_per_s": stats.get("pages_per_s"), "out_tok_per_page": round(stats["out_tok"] / max(1, len(ok)), 1),
             "in_tok_per_page": round(stats["in_tok"] / max(1, len(ok)), 1),
             "ungrounded_pages": stats.get("ungrounded_pages", 0),
             "pages_with_visuals": sum(1 for p in ok if p["vlm"].get("visual_elements")),
             "visual_types": dict(sum((vtypes(p["vlm"]) for p in ok), Counter())),
             "mean_legibility": round(sum(p["vlm"].get("legibility", 0) for p in ok) / max(1, len(ok)), 3)}
        sims_abbyy = [wsim(p["vlm"]["transcription"], p.get("abbyy") or p.get("text")) for p in ok
                      if len(p.get("abbyy") or p.get("text") or "") > 200]
        r["text_sim_vs_abbyy_mean"] = round(sum(sims_abbyy) / max(1, len(sims_abbyy)), 3)
        if path != a.ref:
            both = [(p, ref.get((p["doc_id"], p["page"]))) for p in ok]
            both = [(p, q) for p, q in both if q and q.get("vlm")]
            r["n_vs_ref"] = len(both)
            r["text_sim_vs_ref_mean"] = round(sum(wsim(p["vlm"]["transcription"], q["vlm"]["transcription"]) for p, q in both) / max(1, len(both)), 3)
            r["entity_f1_vs_ref"] = round(sum(f1(ents(p["vlm"]), ents(q["vlm"])) for p, q in both) / max(1, len(both)), 3)
            r["visual_type_agree_vs_ref"] = round(sum(set(vtypes(p["vlm"])) == set(vtypes(q["vlm"])) for p, q in both) / max(1, len(both)), 3)
        report[t] = r
        emb = []
        for p in pages:
            q = dict(p); q["text"] = chunk_text(p) if p.get("vlm") else ""
            q.pop("vlm_raw", None); emb.append(q)
        json.dump(emb, open(os.path.join(a.out, f"embed_pages_{t}.json"), "w"))
    json.dump(report, open(os.path.join(a.out, "vlm_compare.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))

if __name__ == "__main__":
    main()
