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
