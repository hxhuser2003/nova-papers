"""One-off: merge a trained LoRA checkpoint into its base model and push the
result to the Hugging Face Hub, so it can be served with plain
`vllm serve <repo>` -- no --enable-lora, no --lora-modules.

Needed because vLLM's LoRA serving path proved unreliable in Colab on a T4
GPU (repeated SafetensorError: header too large, isolated to the moment
--enable-lora was added; see docs/CONCURRENCY_REPAIR_PLAN.md). This sidesteps
that code path entirely by baking the LoRA weights into an ordinary full
model checkpoint, served the same way the (working) base model always was.

Usage: python -m scripts.merge_lora_and_push <checkpoint-dir> <hf-repo-id> [--base-model NAME] [--public]
e.g.:  python -m scripts.merge_lora_and_push checkpoints/p1-data1000-step1000 your-username/nova-papers-p1
"""
import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("repo_id")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--public", action="store_true", help="push a public repo (default: private)")
    args = p.parse_args()

    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto")
    model = PeftModel.from_pretrained(base, args.checkpoint)
    merged = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    private = not args.public
    merged.push_to_hub(args.repo_id, private=private)
    tokenizer.push_to_hub(args.repo_id, private=private)
    print(f"Pushed to https://huggingface.co/{args.repo_id} (private={private})")


if __name__ == "__main__":
    main()
