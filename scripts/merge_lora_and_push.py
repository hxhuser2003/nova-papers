"""One-off: merge a trained LoRA checkpoint into its base model, so it can be
served with plain `vllm serve <path-or-repo>` -- no --enable-lora, no
--lora-modules.

Needed because vLLM's LoRA serving path proved unreliable in Colab on a T4
GPU (repeated SafetensorError: header too large, isolated to the moment
--enable-lora was added; see docs/CONCURRENCY_REPAIR_PLAN.md). This sidesteps
that code path entirely by baking the LoRA weights into an ordinary full
model checkpoint, served the same way the (working) base model always was.

Always saves locally to --output-dir first (no HuggingFace account needed --
tar that directory, transfer it to Colab however you already transferred the
LoRA adapter, e.g. via Google Drive, and point vllm serve at the local copy).
Passing --repo-id additionally pushes the merged model to the HF Hub, for the
"vllm serve <repo-id>" route instead, if you'd rather not transfer ~6GB by
hand.

Usage: python -m scripts.merge_lora_and_push <checkpoint-dir> --output-dir <local-dir> [--repo-id you/name] [--base-model NAME] [--public]
e.g.:  python -m scripts.merge_lora_and_push checkpoints/p1-data1000-step1000 --output-dir checkpoints/p1-data1000-step1000-merged
       python -m scripts.merge_lora_and_push checkpoints/p1-data1000-step1000 --output-dir /tmp/merged --repo-id your-username/nova-papers-p1
"""
import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--output-dir", required=True, help="local directory to save the merged model + tokenizer to")
    p.add_argument("--repo-id", default=None, help="optional: also push the merged model to this HF Hub repo")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--public", action="store_true", help="if --repo-id is given, push it public (default: private)")
    args = p.parse_args()

    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto")
    model = PeftModel.from_pretrained(base, args.checkpoint)
    merged = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Merged model saved locally to {args.output_dir}")

    if args.repo_id:
        private = not args.public
        merged.push_to_hub(args.repo_id, private=private)
        tokenizer.push_to_hub(args.repo_id, private=private)
        print(f"Also pushed to https://huggingface.co/{args.repo_id} (private={private})")


if __name__ == "__main__":
    main()
