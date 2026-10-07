# Benchmark: page readers (Qwen3-VL 2B / 4B / 8B) and EmbeddingGemma 2 on real CREST pages

Run 2026-10-07 on one RTX 4090 (RunPod, secure cloud, $0.74/hr). Total spend **$1.09**.
Scripts: `fetch_pages.py`, `vlm_pages.py`, `vlm_compare.py`, `embed_bench.py`, `bench/` (pod runner, cost model).

## What was tested

- **Sample:** 176 pages from 47 documents in five groups: typed text (64), photo-interpretation
  reports (72), NGA imagery records (11), scientific (16), STARGATE (13). 16 documents came from
  archive.org and 31 from the Wayback copy of the cia.gov PDF.
- **Citation:** every page carries the official reading-room link
  (`https://www.cia.gov/readingroom/docs/<doc>.pdf#page=N`) even when the bytes came from
  archive.org. No archive.org URL appears in any citation field.
- **Readers:** each Qwen3-VL size ran the pipeline's real extraction prompt and JSON schema
  (`schema.py`) at temperature 0 under vLLM 0.31, with `repetition_penalty 1.05` and an
  8,000-token cap. 8B used the official FP8 weights so it fits a 24 GB card.
- **Retrieval:** EmbeddingGemma 2 (`google/embeddinggemma-2`, bf16) embedded each reader's page
  chunk (transcription plus visual-element list) three ways: text only, page image only, and
  text plus image together. Queries are phrases cut from archive.org's **independent** OCR, with
  4% character noise, plus document titles. Queries never come from the text being indexed.

## Page readers

| | 2B | 4B | 8B (FP8) |
|---|---|---|---|
| Valid pages (schema passes) | **24%** | 83% | **85%** |
| Runaway output (hit the token cap) | 76% | 17% | 15% |
| Entities per valid page (median) | 5 | **0** | **8.5** |
| Pages with visual elements listed | 42 | 102 | **148** |
| Maps, photos and diagrams found | 22 | **1** | **56** |
| Agreement with ABBYY OCR text | 0.81 | 0.78 | 0.79 |
| Entity agreement with 8B (F1) | 0.41 | 0.22 | — |
| Decode speed, saturated (tok/s) | 858 | 533 | 544 |

Readings:

- **2B is unusable.** Three pages in four loop ("App App App …") until the cap, so it costs *more*
  per page than 8B despite being four times smaller.
- **4B reads the text but skips the graph.** Its transcriptions match 8B's (0.77 similarity), but it
  returns no entities on most pages and almost never names a map, photo or diagram. The knowledge
  graph and the visual-element index are built from exactly those fields.
- **8B is the only reader that produces all three outputs** the PRD needs: text, entities and
  visual elements.
- **Speed is not proportional to size.** On a 4090 the KV cache, not the weights, limits
  throughput. Each page costs ~3,000 image tokens of input, so only ~13 pages fit in a batch for
  4B and 8B alike. 4B and 8B decode at the same speed. 4B is cheaper per page only because it
  writes less, and what it leaves out is the entities.
- **Grounding check is noisy.** 46 of 150 8B pages had an entity not found verbatim in the
  transcription. Most were normalisations ("CIA" from "CIA-RDP…", spelled-out agency names), not
  inventions. `check_grounding()` needs a normalising matcher before it can gate data.

## EmbeddingGemma 2 retrieval (Recall@10, page found in the top 10)

| Index built from | Text only | Image only | Text + image |
|---|---|---|---|
| ABBYY OCR text (first run, 56 text-heavy pages) | 0.93 | 0.84 | 0.93 |
| 8B reader output (176 pages, mixed) | 0.62 | **0.46** | **0.68** |
| 4B reader output | 0.65 | 0.51 | 0.70 |
| 2B reader output | 0.51 | 0.41 | 0.67 |

- **Image-only embeddings are the weakest option on these scans.** That rules out the
  "EmbeddingGemma only, no reader" design. It would lose roughly a third of the pages that text
  search finds.
- **Text + image is the best index** in every run. The image adds recall on pages with little text.
- Truncating to 256 dimensions cost 5–9 points of Recall@10. Keep 768d with int8 storage.
- **Embedding cost is negligible:** 68 pages/s text, 17 pages/s text + image on one 4090. That is
  about **$76 for the whole archive** (text + image, community 4090).

## Measured cost, full archive (12.2M pages, +15% overhead)

Throughput = saturated decode tokens/s ÷ average output tokens per page, including failed pages
up to the cap. Recommended cap is **4,000 tokens**: the longest valid 8B page used ~3,300.

| Reader, 4,000-token cap | Pages/s per 4090 | GPU-hours | Community 4090 ($0.34) | Secure 4090 ($0.74) |
|---|---|---|---|---|
| 2B | 0.26 | 14,800 | $5,040 | $10,980 |
| 4B | 0.48 | 8,100 | $2,750 | $5,980 |
| **8B FP8** | **0.35** | **11,100** | **$3,770** | **$8,210** |

The PRD's earlier figure ($1,612 on 8×H100) assumed 3.1 pages/s per H100. That was an estimate.
The measured 4090 rate is 0.35 pages/s. H100 throughput is still unmeasured; with ~3× the KV-cache
memory it should batch far more pages, so a one-hour H100 run is the next test.

## Known limits of this benchmark

- One GPU type (4090) and one sample of 176 pages. Costs scale linearly with the per-page token
  figures above; a different document mix moves them.
- The 8B run had only a few minutes at full batch. Its decode rate (544 tok/s) is the least
  certain number here.
- Neither ABBYY nor 8B is ground truth. Agreement shows the readers see the same words, not that
  the words are right.
- Invalid pages were not retried. The pipeline's quarantine-and-replay step adds cost not counted
  beyond the 15% overhead.


---

## Sample 2: H100, newer readers (2026-10-07)

One H100 SXM 80GB (RunPod secure cloud, $3.49/hr), vLLM 0.31. Total spend **$2.54**. Run outputs are in
[`bench/runs/2026-10-07-h100-s2/`](bench/runs/2026-10-07-h100-s2/) and every reading is in `bench/store/readings.jsonl`.
The 215 page images are attached to the GitHub release `bench-2026-10-07-h100-s2`.

**Sample:** 215 pages from 80 documents, none of them read before: text 30 docs, NGA 20, scientific 15, STARGATE 15.
No Ground Photo Caption Cards could be fetched (no archive.org item, no Wayback capture), so that stratum is missing.
OCR-only models were excluded: the reader has to see photos, maps and other visual content.

| Reader | Valid JSON | Pages/s (wall) | Output tok/page | Median names | Median visual elements | Pages with names not found in the page text |
|---|---|---|---|---|---|---|
| Qwen3-VL-8B FP8 (baseline) | 185/215 (86%) | 1.78 | 1,547 | 9 | 4 | 49 |
| Qwen3-VL-8B FP8, images ≤1 MP | 183/215 (85%) | 2.42 | 1,513 | 9 | 4 | 55 |
| Qwen3.5-9B (bf16) | 206/215 (96%) | 2.76 | 1,334 | 7 | 5 | 85 |
| **Qwen3.6-35B-A3B FP8** | **213/215 (99%)** | **2.72** | **1,030** | 7 | 4 | **20** |

Agreement with the 8B baseline (not accuracy, there is no answer key yet):

| Reader | Text similarity | Name F1 | Visual-type agreement |
|---|---|---|---|
| 8B at 1 MP | 0.92 | 0.66 | 0.35 |
| Qwen3.5-9B | 0.90 | 0.43 | 0.23 |
| Qwen3.6-35B-A3B | 0.88 | 0.45 | 0.21 |

**Reading the results**

- **Qwen3.6-35B-A3B is the most reliable reader tested:** 99% valid output, the least output per page, and the fewest
  pages with names that don't appear in the page text (20, against 49 for the baseline and 85 for Qwen3.5-9B).
- **Qwen3.5-9B is fast but invents the most names.** Its speed advantage is small and its grounding is the worst of the four.
- **Shrinking images to 1 MP** makes the 8B 36% faster and keeps text similarity at 0.92, but costs a third of name agreement.
- **Low agreement on names and visual types** means the models disagree a lot. Which one is right needs a manual check
  of a few dozen pages against the images. That check is the next step before choosing a reader.

**Cost (low confidence).** Each run lasted 80–120 s with up to 160 pages in flight at once, so start-up and wind-down
dominate and the speed never reached a steady state. Wall-clock pages/s is used as-is, plus 15% overhead.

| Reader | GPU-hours, 12.2M pages | H100 at $2.69/hr | H100 at $3.49/hr | $ per 1,000 pages |
|---|---|---|---|---|
| Qwen3-VL-8B FP8 | ~2,190 | ~$5,890 | ~$7,640 | $0.42 |
| Qwen3.5-9B | ~1,410 | ~$3,800 | ~$4,920 | $0.27 |
| **Qwen3.6-35B-A3B FP8** | **~1,430** | **~$3,840** | **~$4,990** | **$0.28** |
| *Ref: Qwen3-VL-8B on RTX 4090 (sample 1)* | *~10,000* | *$3,410 at $0.34/hr* | | *$0.24* |

The H100 buys **speed, not savings**: per page it costs about the same as the 4090 path (~$0.27 vs ~$0.24 per 1,000
pages) but finishes about 7× faster per GPU. Qwen3.6 gives better output for roughly the same money. A longer run
(2,000+ pages) is needed for a steady-state speed and a firm number.

**Run notes:** FP8 models failed to start at first, because the pod image's NVCC is older than 12.9 and vLLM's
DeepGEMM JIT asserts on Hopper. Setting `VLLM_USE_DEEP_GEMM=0` fixed it (`bench/pod_h100_fp8.sh`).
