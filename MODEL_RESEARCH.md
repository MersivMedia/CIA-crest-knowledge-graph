# CREST per-page reader: open-weight VLM candidates (researched 2026-10-07)

Baseline: Qwen3-VL-8B-Instruct-FP8 on a 4090 with vLLM 0.31: 0.35 pages/s, 85% schema-valid, about $3,770 for the full archive.
Method: HF API (model metadata, card READMEs), vLLM supported-models page (via r.jina.ai). Every ID below was confirmed to exist via the HF API. Shortlist re-verified independently (HF API + vLLM v0.31.0 model registry: all seven shortlisted architectures are registered).
Params = safetensors total from the HF API. Date = HF createdAt.
No GPU runs: **all quality and speed claims are vendor numbers and are not measured on CREST.**

## Tier 1: drop-in single-pass replacements (JSON + entities + visuals)
| Model | Date | License | Params | vLLM | Notes |
|---|---|---|---|---|---|
| Qwen/Qwen3.5-9B | 2026-02-27 | apache-2.0 | 9.65B dense | Qwen3_5ForConditionalGeneration ✔ | Direct successor to Qwen3-VL-8B (natively multimodal). Fits a 4090 at FP8. **Top A/B candidate.** |
| Qwen/Qwen3.5-4B / 2B | 2026-02-27/28 | apache-2.0 | 4.66B / 2.27B | ✔ | Worth retesting. Qwen3-VL-4B gave few entities, so this needs measuring. |
| Qwen/Qwen3.6-35B-A3B (+ -FP8) | 2026-04-15 | apache-2.0 | 36B total, ~3B active [active count UNVERIFIED from card] | Qwen3_5MoeForConditionalGeneration ✔ | MoE, so decoding is cheap. FP8 is ~36GB: H100 only. A 4090 needs a community INT4. **Best H100 candidate.** |
| Qwen/Qwen3.6-27B, Qwen/Qwen3.8-27B (+FP8) | 2026-04-21 / 2026-08-05 | apache-2.0 | 27.8B dense | ✔ (Qwen3.8 arch name not grepped [UNVERIFIED]) | Newest Qwen VL. Card says "documents". Quality ceiling, about 3x slower than the 9B. H100 only. |
| Qwen/Qwen3.8-Flash-Next | 2026-08-24 | other | 180B | – | Too big for one GPU. Skip. |
| google/gemma-4-E4B-it / E2B-it | 2026-03-02 | apache-2.0 | 8.0B / 5.1B (effective 4B/2B) | Gemma4ForConditionalGeneration ✔ | Cheap. Gemma vision usually uses fewer image tokens [UNVERIFIED for G4]. |
| google/gemma-4-26B-A4B-it | 2026-03-11 | apache-2.0 | 25.8B MoE, 4B active | ✔ | H100. Official QAT int4 checkpoints exist (gemma-4-26B-A4B-it-qat-*) and might fit a 4090. Gemma refusals on intelligence content are a risk. |
| openbmb/MiniCPM-V-4.6 | 2026-04-13 | apache-2.0 | 1.3B (SigLIP2 + Qwen3.5-0.8B) | MiniCPMV ✔, SGLang ✔ | 4x/16x visual-token compression, ~1.5x Qwen throughput (vendor claim). Probably too weak for entities (compare Qwen3-VL-2B). |
| ibm-granite/granite-vision-4.1-4b | 2026-04-16 | apache-2.0 | 4B | [UNVERIFIED in vLLM; card shows transformers+peft] | Built for key-value / table extraction (VAREX bench). |
| allenai/Molmo2-8B | 2025-12-14 | apache-2.0 | 8.7B | Molmo2 ✔ | Older than 6 months. Not doc-focused. |
| mistralai/Ministral-3-8B-Instruct-2512 | 2025-10-31 (upd 2026-07) | apache-2.0 | 8.9B | Mistral3 ✔ | Fallback option. |
| moonshotai Kimi | Kimi-VL-A3B-Thinking-2506 (16B, MIT, KimiVL ✔) is old. Kimi-K2.5/K2.6/K3 (K3 = 2.78T params) are multimodal but far too large. | | | | No small new Kimi VL. |
| zai-org GLM | GLM-5.3-Flash (2026-08-25, MIT, 321B) is too big. GLM-4.xV is old. | | | | Use zai-org/GLM-OCR instead (OCR tier). |
| deepseek-ai/DeepSeek-V4-Flash-Vision-Exp | 2026-08-31 | mit | multi-node (4×GB300) | ✔ recipe | Not single-GPU. |
| Llama / Phi | No new meta-llama VLM since 2026-03. Microsoft only has Mage-VL (2026-07-25, not checked) and Fara (agentic). | | | | Skip. |

## Tier 2: dedicated OCR models (stage 1 of a two-stage pipeline)
| Model | Date | License | Params | vLLM | Reported quality / tokens |
|---|---|---|---|---|---|
| baidu/Qianfan-OCR | 2026-03-18 | apache-2.0 | 4.7B | QianfanOCRForConditionalGeneration ✔ | OmniDocBench v1.5 93.12, olmOCR-bench 79.8, **KIE #1 (87.9)**, so it might also handle entities in one pass. 256 tok per 448² tile, max 4,096. Claims 1.02 pages/s W8A8 on an A100. **Strong single-pass alternative.** |
| datalab-to/chandra-ocr-2 | 2026-03-16 | **openrail** (check terms for commercial use) | ~5B? [UNVERIFIED] | vLLM via chandra_vllm | olmOCR-bench **85.8** (SOTA claim). Its "old scans" column is 51.1, the best open score there. Very relevant to microfilm. |
| baidu/Unlimited-OCR | 2026-06-19 | mit | 3.3B | UnlimitedOCRForCausalLM ✔ | DeepSeek-OCR successor (optical token compression). Low vision tokens per page means good KV economics on a 4090. Tokens per page [UNVERIFIED]. |
| deepseek-ai/DeepSeek-OCR-2 | 2026-01-27 | apache-2.0 | 3.4B MoE (~0.5B active) | DeepseekOCR2ForCausalLM ✔ | Few vision tokens (v1: 100–400 tok/page). OmniDocBench 91.09 (per the Qianfan card). |
| XingChen-AGI/TeleOCR | 2026-08-14 | apache-2.0 | 1.2–1.4B | [UNVERIFIED] | Reports scores on Wild_OmniDocBench (camera/degraded pages) and Dr.DocBench, beating MinerU2.5-Pro and PaddleOCR-VL 1.6 (vendor). |
| PaddlePaddle/HPD-Parsing | 2026-07-21 | apache-2.0 | 1B | custom vLLM 0.17.1 fork | OmniDocBench v1.6 94.91, 4,752 tokens/s peak. Fastest option, but needs a fork. |
| PaddlePaddle/PaddleOCR-VL-1.5 (1.6 mentioned) | 2026-01-28 | apache-2.0 | 0.96B | PaddleOCRVL ✔ | Cheap baseline. |
| zai-org/GLM-OCR | 2026-01-30 | mit | 1.3B | GlmOcrForConditionalGeneration ✔ | |
| lightonai/LightOnOCR-2-1B | 2026-01-16 | apache-2.0 | 1.0B | ✔ | |
| opendatalab/MinerU2.5-Pro-2605-1.2B | 2026-05-20 | [UNVERIFIED] | 1.2B | [UNVERIFIED] | |
| tencent/HunyuanOCR | 2025-11-18 | other | 1.1B | ✔ | |
| allenai/olmOCR-2-7B-1025 | 2025-10 | apache-2.0 | 8.3B | Qwen2.5-VL arch ✔ | Older. olmOCR-bench 82.4. |
| datalab-to/surya-ocr-2 | 2026-05-14 | ? | small | no | Classic OCR, CPU-friendly. |
| PP-OCRv6 det/rec | 2026-06 | apache-2.0 | tiny | no | Non-VLM. Cheapest stage 1. |
| nanonets/Nanonets-OCR2-3B | 2025-10 | – | 3B | ✔ | Older. |
| dots.ocr | rednote-hilab/dots.ocr-1.5 does not exist. dots-studio/dots.ocr (2025-07) is the only one. DotsOCR is in vLLM. | | | | Stale. |
| InternVL | No OpenGVLab release since InternVL3.5 (2025-08). No InternVL4 found. | | | | Stale. |

## Decision 2026-10-07: vision-language readers only

OCR-only models (Tier 2, including Qianfan-OCR) are **out of scope**. The reader has to see the page: photographs,
maps, diagrams, stamps, handwriting and redactions feed `visual_elements` and the graph. A transcription-only stage
would drop exactly that. Tier 2 stays listed for reference only.

## Recommendation (A/B on ~200 CREST pages, same JSON prompt, vLLM)
1. **Qwen3.5-9B** (BF16 on an H100; no official FP8 checkpoint exists, checked 2026-10-07): the most likely straight upgrade.
2. ~~Qianfan-OCR 4.7B~~: excluded (OCR model; see Decision above).
3. **Qwen3.6-35B-A3B FP8** on an H100: cheap MoE decoding, newer generation.
4. ~~Two-stage~~ (excluded, see Decision above): **Chandra-OCR-2** (best on old scans; check the openrail license) or **Unlimited-OCR / DeepSeek-OCR-2** (fewest tokens), followed by a text LLM (e.g. Qwen3.5-4B/9B text-only) for entities. Visual elements then need a separate description pass, or you rely on the OCR models' figure/layout tags.
5. Gemma-4-E4B as a cheap control.

Watch out for: OCR-only models drop the "visual_elements" field. Check prefix caching for the shared prompt. Measure schema-valid rate, not just speed.
