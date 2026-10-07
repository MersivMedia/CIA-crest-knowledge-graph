#!/usr/bin/env python3
"""
Benchmark EmbeddingGemma 2 on real CREST pages before re-architecting retrieval around it.

Runs on a GPU box (bf16). For each sampled page it builds three document representations:

  text    the VLM/ABBYY transcription, embedded as "title: {title} | text: {transcription}"
  image   the page scan itself (no text), embedded through the vision encoder
  inter   one interleaved embedding: "title: {title} | text: {transcription} <|image|>" + the scan

and asks: which representation retrieves the right page for a query, and how fast is each?

Queries are built WITHOUT a human or an LLM, so the benchmark is deterministic and cheap:
  - "known-item" queries: a 6-12 word span sampled from the middle of the page's own transcription
    with OCR noise injected (char swaps), mimicking a researcher who half-remembers a phrase
  - "title" queries: the manifest title, which is a short human-written description
  - "visual" queries (photo/NGA strata only): the manifest title of an image-heavy record
These are weak proxies for real research questions; the human-labelled set (M6) replaces them.

Reports, per representation and per MRL dimension (768/512/256/128):
  recall@1, recall@10, MRR@10, overall and per stratum; encode throughput (pages/s) per modality and
  vision-token budget; peak VRAM. Writes embed_bench.json.

    python embed_bench.py --sample embed_sample.json --pages-dir /root/pages --out embed_bench.json
"""
import argparse, json, os, random, re, statistics, sys, time

def noisy(s, rate, rng):
    out = []
    for ch in s:
        r = rng.random()
        if ch.isalpha() and r < rate / 2:
            out.append(rng.choice("abcdefghijklmnopqrstuvwxyz"))
        elif ch.isalpha() and r < rate:
            continue
        else:
            out.append(ch)
    return "".join(out)

def spans(text, rng, n=2, lo=6, hi=12):
    w = re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", text)
    if len(w) < hi + 4:
        return []
    out = []
    for _ in range(n):
        k = rng.randint(lo, hi); i = rng.randint(2, len(w) - k - 2)
        out.append(" ".join(w[i:i + k]))
    return out

def metrics(ranks):
    r1 = sum(1 for r in ranks if r == 1) / len(ranks)
    r10 = sum(1 for r in ranks if r and r <= 10) / len(ranks)
    mrr = sum((1 / r) for r in ranks if r and r <= 10) / len(ranks)
    return dict(n=len(ranks), r1=round(r1, 3), r10=round(r10, 3), mrr10=round(mrr, 3))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)                # pages.json from fetch_pages.py
    ap.add_argument("--query-field", default="text",
                    help="field the phrase queries are cut from. Use 'abbyy' when 'text' is a VLM transcription, so "
                         "queries come from an independent OCR and the test is not circular")
    ap.add_argument("--min-text", type=int, default=200,
                    help="pages with less transcription than this are 'image-only': text rep = title alone")
    ap.add_argument("--out", default="embed_bench.json")
    ap.add_argument("--model", default="google/embeddinggemma-2")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--budgets", default="280,560,1120")      # vision soft-token budgets to time
    ap.add_argument("--noise", type=float, default=0.04)
    ap.add_argument("--distractors", type=int, default=0,
                    help="extra text-only pages (from --distractor-file) added to the index to make retrieval harder")
    ap.add_argument("--distractor-file")
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer
    from PIL import Image
    rng = random.Random(11)

    pages = json.load(open(args.sample))
    pages = [p for p in pages if os.path.exists(p["image"])]
    for p in pages:
        p["has_text"] = len(p.get("text") or "") > args.min_text
        if not p["has_text"]:
            p["text"] = ""
    print(f"{len(pages)} pages, {sum(p['has_text'] for p in pages)} with a transcription, "
          f"{sum(not p['has_text'] for p in pages)} image-only")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    t0 = time.time()
    model = SentenceTransformer(args.model, device="cuda", model_kwargs={"torch_dtype": dtype},
                                config_kwargs={"audio_config": None})       # text + vision only (440M)
    load_s = time.time() - t0
    torch.cuda.reset_peak_memory_stats()
    res = {"n_image_only": sum(not p["has_text"] for p in pages), "model": args.model, "dtype": str(dtype), "gpu": torch.cuda.get_device_name(0),
           "load_s": round(load_s, 1), "n_pages": len(pages)}

    def doc_text(p):
        return f"title: {p['title'] or 'none'} | text: {p['text'][:20000]}"

    # ---------- encode corpus, timed per representation
    def timed(fn, items, label):
        torch.cuda.synchronize(); t = time.time()
        E = fn(items)
        torch.cuda.synchronize(); dt = time.time() - t
        res.setdefault("throughput", {})[label] = round(len(items) / dt, 2)
        print(f"  {label:<24} {len(items) / dt:7.2f} items/s")
        return E

    texts = [doc_text(p) for p in pages]
    imgs = [Image.open(p["image"]).convert("RGB") for p in pages]
    E_text = timed(lambda X: model.encode(X, batch_size=args.batch, convert_to_tensor=True, normalize_embeddings=True),
                   texts, "text")
    E_img = timed(lambda X: model.encode([{"image": im} for im in X], batch_size=args.batch,
                                         convert_to_tensor=True, normalize_embeddings=True), imgs, "image@280")
    E_int = timed(lambda X: model.encode([{"text": f"{t} <|image|>", "image": im} for t, im in X],
                                         batch_size=max(1, args.batch // 2), convert_to_tensor=True,
                                         normalize_embeddings=True), list(zip(texts, imgs)), "interleaved@280")

    # vision budget timing (the processor exposes max_soft_tokens; set it, re-time a slice)
    budgets = [int(b) for b in args.budgets.split(",")]
    proc = getattr(model[0], "processor", None)
    for b in budgets:
        if b == 280 or proc is None:
            continue
        try:
            ip = proc.image_processor
            old = ip.max_soft_tokens
            ip.max_soft_tokens = b; ip.image_seq_length = b
            if hasattr(proc, "image_seq_length"): proc.image_seq_length = b
            sl = imgs[: min(64, len(imgs))]
            E_b = timed(lambda X: model.encode([{"image": im} for im in X], batch_size=args.batch,
                                               convert_to_tensor=True, normalize_embeddings=True), sl, f"image@{b}")
            res.setdefault("budget_e", {})[b] = E_b
            ip.max_soft_tokens = old; ip.image_seq_length = old
            if hasattr(proc, "image_seq_length"): proc.image_seq_length = old
        except Exception as e:
            res.setdefault("budget_errors", {})[b] = repr(e)[:200]
    res.pop("budget_e", None)
    res["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    # optional distractors (text-only) to grow the index
    E_dis = None
    if args.distractors and args.distractor_file:
        dis = json.load(open(args.distractor_file))[: args.distractors]
        E_dis = model.encode([f"title: {d.get('title') or 'none'} | text: {d['text'][:20000]}" for d in dis],
                             batch_size=args.batch, convert_to_tensor=True, normalize_embeddings=True)
        res["distractors"] = len(dis)

    # ---------- queries
    Q = []   # (query_text, gold_index, kind, stratum)
    for i, p in enumerate(pages):
        for s in spans(p.get(args.query_field) or p["text"], rng):
            Q.append((noisy(s, args.noise, rng), i, "phrase", p["stratum"]))
        # title queries only make sense if the title is not itself the indexed text: for image-only pages the
        # text rep IS the title, so a title query there is trivially solvable by text; use them to test image/
        # interleaved reps and report them separately (kind=title_imgonly)
        if p.get("title") and len(p["title"].split()) >= 3 and p["page"] == 0:
            Q.append((p["title"], i, "title" if p["has_text"] else "title_imgonly", p["stratum"]))
    print(f"{len(Q)} queries")
    E_q = model.encode([q[0] for q in Q], prompt_name="SearchQuery", batch_size=args.batch,
                       convert_to_tensor=True, normalize_embeddings=True)

    # ---------- score at each MRL dimension
    import torch.nn.functional as F
    def trunc(E, d):
        return F.normalize(E[:, :d].float(), dim=-1)
    out = {}
    for d in (768, 512, 256, 128):
        q = trunc(E_q, d)
        for name, E in (("text", E_text), ("image", E_img), ("interleaved", E_int)):
            C = trunc(E, d)
            if E_dis is not None:
                C = torch.cat([C, trunc(E_dis, d)])
            S = q @ C.T
            order = S.argsort(dim=1, descending=True)
            ranks = []
            for qi, (_, gold, kind, st) in enumerate(Q):
                pos = (order[qi] == gold).nonzero()
                ranks.append(int(pos[0, 0]) + 1 if len(pos) else None)
            buckets = {"all": ranks}
            for qi, (_, gold, kind, st) in enumerate(Q):
                buckets.setdefault(f"kind={kind}", []).append(ranks[qi])
                buckets.setdefault(f"stratum={st}", []).append(ranks[qi])
            out[f"{name}@{d}"] = {k: metrics(v) for k, v in buckets.items()}
        # late fusion: max of text and image similarity (two vectors per page)
        Ct, Ci = trunc(E_text, d), trunc(E_img, d)
        S = torch.maximum(q @ Ct.T, q @ Ci.T)
        if E_dis is not None:
            S = torch.cat([S, q @ trunc(E_dis, d).T], dim=1)
        order = S.argsort(dim=1, descending=True)
        ranks = []
        for qi, (_, gold, kind, st) in enumerate(Q):
            pos = (order[qi] == gold).nonzero()
            ranks.append(int(pos[0, 0]) + 1 if len(pos) else None)
        b = {"all": ranks}
        for qi, (_, gold, kind, st) in enumerate(Q):
            b.setdefault(f"kind={kind}", []).append(ranks[qi]); b.setdefault(f"stratum={st}", []).append(ranks[qi])
        out[f"fusion_max@{d}"] = {k: metrics(v) for k, v in b.items()}

    # cross-modal sanity: is a page's own image nearer its own text than other pages' text?
    S = trunc(E_img, 768) @ trunc(E_text, 768).T
    own = S.diag(); rank_own = (S > own[:, None]).sum(1) + 1
    res["image_to_own_text"] = {"r1": round(float((rank_own == 1).float().mean()), 3),
                                "r10": round(float((rank_own <= 10).float().mean()), 3),
                                "median_rank": int(rank_own.median())}
    res["retrieval"] = out
    res["n_queries"] = len(Q)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("gpu", "throughput", "peak_vram_gb", "image_to_own_text")}, indent=1))
    for key in ("text@768", "image@768", "interleaved@768", "fusion_max@768", "text@256", "interleaved@256", "text@128"):
        print(f"{key:<20} {out[key]['all']}")

if __name__ == "__main__":
    main()
