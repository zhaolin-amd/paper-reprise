**Why acc_norm (Hellaswag) is PARTIAL while PPL (Wikitext) is MATCH:**

The paper uses **vLLM** as the inference engine with lm-eval; this reproduction passes a
pre-instantiated HuggingFace model object to `HFLM` instead. When lm-eval receives an
already-instantiated model (not a string path), it skips several initialization steps —
the log shows `Many other model arguments may be ignored`. This affects the log-likelihood
computation that acc_norm relies on.

- **PPL** is teacher-forced (no sampling, no batching across choices) → insensitive to
  this difference → 5/5 MATCH within ±0.06.
- **acc_norm** compares log-likelihoods across multiple choice strings → sensitive to
  batching, tokenization padding, and prefix caching differences between vLLM and HF
  eager mode → systematically 1.1–2.1 below the paper.

The BF16 baseline itself is off by 1.55 (74.96 vs 76.51), which rules out the
quantization implementation as the cause — the gap is entirely in the eval infrastructure.

**To close this gap:** re-run via lm-eval's vLLM backend, passing the model as a string
path (`--model vllm --model_args pretrained=<path>`) so lm-eval initializes the
full pipeline consistently with the paper's setup.
