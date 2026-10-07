#!/usr/bin/env python3
"""
Durable CREST bench store, committed to the repo so every run builds on the previous ones:

  bench/runs/<run_id>/            raw outputs of one run (reader JSON, stats, embed results, logs)
  bench/store/readings.jsonl      one line per (doc_id, page, model): the full reader output (transcription,
                                  entities, visual_elements, legibility), citation (official cia.gov #page=N),
                                  source, run_id. Append-only; never rewritten.
  bench/store/pages_seen.json     doc_ids already sampled (new samples exclude them)
  bench/store/images/             page images, local only (*.jpg is gitignored); published per run as a
                                  GitHub release asset instead

    python crest_store.py ingest <run_dir> <run_id> [--images <dir>]
    python crest_store.py stats
"""
import json, os, shutil, sys, glob, hashlib
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench")
STORE = os.path.join(ROOT, "store"); RUNS = os.path.join(ROOT, "runs")
READ = os.path.join(STORE, "readings.jsonl"); SEEN = os.path.join(STORE, "pages_seen.json")

def load_keys():
    keys = set()
    if os.path.exists(READ):
        for l in open(READ):
            d = json.loads(l); keys.add((d["doc_id"], d["page"], d["model"]))
    return keys

def ingest(run_dir, run_id, images=None):
    os.makedirs(STORE, exist_ok=True); dst = os.path.join(RUNS, run_id); os.makedirs(dst, exist_ok=True)
    for f in glob.glob(os.path.join(run_dir, "*")):
        if os.path.isfile(f) and not os.path.basename(f).startswith("pip") and not f.endswith(".tar"):  # pip logs are noise; image tars go to a release
            shutil.copy2(f, dst)
    keys = load_keys(); seen = set(json.load(open(SEEN))) if os.path.exists(SEEN) else set()
    added = 0
    with open(READ, "a") as out:
        for f in sorted(glob.glob(os.path.join(run_dir, "vlm_*.json"))):
            if os.path.basename(f) == "vlm_compare.json":
                continue
            stats_f = f.replace("vlm_", "vlmstats_")
            model = json.load(open(stats_f)).get("model") if os.path.exists(stats_f) else os.path.basename(f)
            for p in json.load(open(f)):
                seen.add(p["doc_id"])
                k = (p["doc_id"], p["page"], model)
                if k in keys:
                    continue
                rec = {"doc_id": p["doc_id"], "page": p["page"], "model": model, "run_id": run_id,
                       "stratum": p.get("stratum"), "title": p.get("title"), "citation": p.get("citation"),
                       "document_url": p.get("document_url"), "source": p.get("source"),
                       "image_file": os.path.basename(p.get("image") or ""),
                       "valid": bool(p.get("vlm")), "reading": p.get("vlm"),
                       "raw_if_invalid": None if p.get("vlm") else (p.get("vlm_raw") or "")[:2000],
                       "error": p.get("vlm_error"), "finish_reason": p.get("finish_reason"),
                       "seconds": p.get("vlm_seconds"), "ungrounded": p.get("ungrounded_entities"),
                       "abbyy": p.get("abbyy")}
                assert "archive.org" not in (rec["citation"] or ""), rec["citation"]
                out.write(json.dumps(rec) + "\n"); keys.add(k); added += 1
    json.dump(sorted(seen), open(SEEN, "w"))
    if images:
        idir = os.path.join(STORE, "images"); os.makedirs(idir, exist_ok=True)
        if images.endswith(".tar"):                       # accept the pod's page_images.tar directly
            import tarfile, tempfile
            tmp = tempfile.mkdtemp()
            with tarfile.open(images) as t:
                t.extractall(tmp, filter="data")
            images = tmp
        for f in glob.glob(os.path.join(images, "**", "*.jpg"), recursive=True):
            t = os.path.join(idir, os.path.basename(f))
            if not os.path.exists(t):
                shutil.copy2(f, t)
    print(f"ingested {run_id}: +{added} readings, {len(keys)} total, {len(seen)} docs seen")

def stats():
    from collections import Counter
    c = Counter(); v = Counter()
    for l in open(READ):
        d = json.loads(l); c[d["model"]] += 1; v[d["model"]] += d["valid"]
    for m in c:
        print(f"{m:42s} {c[m]:6d} readings  {v[m]:6d} valid")
    print("images:", len(os.listdir(os.path.join(STORE, "images"))) if os.path.isdir(os.path.join(STORE, "images")) else 0)
    print("docs seen:", len(json.load(open(SEEN))))

if __name__ == "__main__":
    if sys.argv[1] == "ingest":
        ingest(sys.argv[2], sys.argv[3], sys.argv[5] if len(sys.argv) > 5 and sys.argv[4] == "--images" else None)
    else:
        stats()
