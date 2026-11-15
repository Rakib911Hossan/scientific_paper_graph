"""transformers-backed local model backend -- alternative to llm/gemini.py's
generate_with_retry() for running the pipeline without the Gemini API.

Deliberately NOT vLLM: vLLM's compiled CUDA extensions pin an exact torch
build (torch==2.11.0 as of vllm 0.25.1) with its own CUDA-runtime-version
requirement, independent of whatever torch/CUDA the host environment (e.g.
Colab) already has -- this produced a chain of environment breakages
(marker-pdf/surya's Docker requirement, then transformers.onnx/
find_pruneable_heads_and_indices removed by the transformers version vLLM
forces, then a missing libcudart.so.13 for the CUDA version vLLM's torch
build expects) with no version combination that avoided all of them at
once. Plain `transformers` has no compiled-extension version pin of its
own -- it just uses whatever torch/CUDA is already installed, which is the
one thing guaranteed to already work in the host environment.

transformers/torch/bitsandbytes are only imported inside _get_model(), so
importing this module (e.g. as part of llm/router.py's module-level
import) doesn't require any of them unless the local backend is actually
selected at call time.

There's no KeyPool/RateLimiter equivalent here: those exist to round-robin
and throttle a metered remote API across multiple keys, which doesn't
apply to a single local GPU.

Trade-off versus the earlier vLLM backend: no guided/constrained JSON
decoding, so response_mime_type == "application/json" is honored only via
prompt instruction (every call site's prompt already asks for JSON) plus
this pipeline's existing per-call retry-on-malformed-JSON logic -- the
same class of imperfect-but-workable guarantee Gemini's own JSON mode
gives in practice (see CLAUDE.md's notes on occasional malformed JSON even
under response_mime_type="application/json"), not a regression specific
to this backend.
"""
import os

# Must be set before torch/CUDA initializes -- reduces allocator fragmentation,
# which matters here specifically because Stage 1 (document summary / thesis
# synthesis) feeds the model the *entire* paper's sentence pool as context, so
# a single prefill attention allocation can be several GiB even on a small
# model; fragmentation from earlier allocations can turn a request that would
# otherwise fit into an OOM. Free to set even when it doesn't help.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_model = None
_tokenizer = None


def _get_model(model_id: str, dtype: str):
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"Loading local model {model_id} (4-bit, compute dtype={dtype}) ...")
        compute_dtype = getattr(torch, dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
        )
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    return _model, _tokenizer


def generate(
    prompt: str,
    generation_config=None,
    max_retries: int = 2,
    model_id: str | None = None,
    max_output_tokens: int = 4096,
    dtype: str | None = None,
    **_ignored,  # accepts and ignores vLLM-only kwargs (max_model_len, gpu_memory_utilization) from older call sites
) -> str:
    """Same call contract as gemini.generate_with_retry(): takes a
    google-genai types.GenerateContentConfig (or anything with the same
    attribute names) so every pipeline call site can construct its
    generation_config exactly as it already does for Gemini -- only the
    dispatch in llm/router.py needs to know which backend is selected.

    dtype defaults to float16, not bfloat16 -- Colab's free-tier T4 is
    Turing-generation and has no native bf16 tensor-core support. Override
    via CLAIM_PIPELINE_LOCAL_DTYPE (e.g. "bfloat16") on a newer GPU
    (A100/L4/etc.).
    """
    import torch

    model_id = model_id or os.environ.get("CLAIM_PIPELINE_LOCAL_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    dtype = dtype or os.environ.get("CLAIM_PIPELINE_LOCAL_DTYPE", "float16")
    model, tokenizer = _get_model(model_id, dtype)

    temperature = getattr(generation_config, "temperature", None)
    temperature = temperature if temperature is not None else 1.0

    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt")

    max_input_tokens = int(os.environ.get("CLAIM_PIPELINE_LOCAL_MAX_INPUT_TOKENS", "8192"))
    input_len = inputs["input_ids"].shape[1]
    if input_len > max_input_tokens:
        # Truncate from the FRONT (drop oldest tokens, keep the tail) -- every
        # one of this pipeline's prompts puts its actual output-format
        # instructions ("return JSON with...") at the end, after the bulk
        # content (e.g. Stage 1's full sentence pool). Truncating from the
        # front keeps that instruction tail (and the chat template's
        # add_generation_prompt suffix, which is also at the very end)
        # intact, at the cost of the model only seeing the tail of the
        # source content -- a real quality hit, not just a formatting risk,
        # but this pipeline's local-backend accuracy bar accepts that
        # tradeoff over an outright OOM crash. Root cause is structural
        # (Stage 1 feeds the entire sentence pool as one prompt, sized for
        # a server-side model, not a single consumer GPU) -- this is damage
        # control, not a fix for that.
        print(f"Truncating input from {input_len} to {max_input_tokens} tokens (CLAIM_PIPELINE_LOCAL_MAX_INPUT_TOKENS)")
        inputs["input_ids"] = inputs["input_ids"][:, -max_input_tokens:]
        inputs["attention_mask"] = inputs["attention_mask"][:, -max_input_tokens:]

    inputs = inputs.to(model.device)

    last_err = None
    for attempt in range(max_retries):
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_output_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        except Exception as e:
            last_err = e
            print(f"Local generation failed (attempt {attempt + 1}/{max_retries}): {e}")
            torch.cuda.empty_cache()  # release fragmented-but-unused memory before retrying
    raise RuntimeError(f"Local model generation failed after {max_retries} attempts") from last_err
