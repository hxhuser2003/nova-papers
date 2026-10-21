"""Live HTTP token/logprob probe. Run on the allocated GPU node."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.vllm_training_client import RolloutModelClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--output', default='eval/training/live-token-probe.json')
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': 'Reply with exactly <answer>Yes</answer>.'}],
        tokenize=True, add_generation_prompt=True,
    )
    with RolloutModelClient(args.base_url, args.model, temperature=1.0) as client:
        result = client.generate(prompt, stop=['</answer>'], max_tokens=64)
        if max(result['token_ids']) >= len(tokenizer):
            raise ValueError('Server token IDs exceed local tokenizer vocabulary')
        # A second call uses the exact first continuation as a token prefix.
        second = client.generate(prompt + result['token_ids'], max_tokens=4)
    report = dict(model=args.model, prompt_tokens=len(prompt), first=result, second=second,
                  decoded=tokenizer.decode(result['token_ids'], skip_special_tokens=False),
                  status='live_http_token_ids_and_logprobs_valid')
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(f'Live token probe passed: {path}')


if __name__ == '__main__':
    main()
