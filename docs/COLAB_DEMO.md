# Colab web demo (P1 checkpoint)

How to run the Nova Papers web app (React frontend + FastAPI backend) on a free Google Colab T4, serving the P1 GRPO checkpoint (`p1-data1000-step1000`, Qwen2.5-3B-Instruct + LoRA r=8), and expose it at a temporary public URL.

Notebook: [`docs/colab_demo_p1.ipynb`](colab_demo_p1.ipynb). Verified end to end on 2026-11-26 (Colab T4, Python 3.13): the app picked the right paper, searched, read a paragraph and answered with citations.

## What you need

- A Google account (free Colab is enough) with the runtime set to **T4 GPU**.
- `p1-checkpoint.zip`: a folder `p1-checkpoint/` holding `adapter_config.json` and `adapter_model.safetensors` (about 55 MB zipped).
- About 25–30 minutes for the first run. Most of it is merging the model (~8 min to write it) and starting vLLM (~3 min).

## Steps

Open the notebook in Colab (*File → Upload notebook*) and run it top to bottom.

| Step | What it does | Time |
| --- | --- | --- |
| 1 | Clone the repo, `pip install` vLLM and the backend's dependencies | 3–5 min |
| 2 | Install Node 22 and build the frontend into `frontend/dist` | 2–3 min |
| 3 | Upload `p1-checkpoint.zip` to `/content` and unzip it (manual upload) | 1–2 min |
| 4 | Merge the adapter into the base model, serve it with vLLM as `nova-p1` | 10–15 min |
| 5 | Write `.env` and start the backend on port 8127 (serves the built frontend too) | <1 min |
| 6 | Start a `cloudflared` quick tunnel and print the public URL | <1 min |

Step 5 should print a health check like `{"model":"nova-p1",...,"corpus_ready":true,"paper_count":1585}`.

## Why the adapter is merged instead of served with `--enable-lora`

With the vLLM version `pip` installs on Colab today, `vllm serve ... --enable-lora --lora-modules nova-p1=...` fails on a T4 at engine start-up (`RuntimeError: Engine core initialization failed`). The base model alone starts fine, which matches the earlier finding recorded in `scripts/merge_lora_and_push.py`: the LoRA serving path is the unreliable part on T4.

So step 4 bakes the adapter into the base weights with `peft` (`merge_and_unload`) and serves the result as an ordinary model. The weights are the same, so answers are the same as serving the adapter. `scripts/merge_lora_and_push.py` does the same merge on CARC; you can also merge there and upload the merged folder (~6 GB) instead of merging in Colab.

## Known issues and fixes

| Symptom | Cause | Fix |
| --- | --- | --- |
| Red pip `ERROR: ... dependency resolver` lines in step 1 | vLLM upgrades packages that Colab's preinstalled tools (gradio, cudf, google-adk) pin | Harmless; ignore |
| `ImportError: Found an incompatible version of torchao` when merging | Colab ships torchao 0.10; current `peft` requires a newer one | The notebook runs `pip uninstall -y torchao` first; the demo does not use torchao |
| `Cannot use FA version 2 is not supported` in `vllm.log` | T4 has no FlashAttention 2 | Harmless; vLLM falls back to another attention backend |
| `npm run build` fails, or Vite complains about the Node version | Colab's built-in Node is older than Vite 8 needs (≥ 20.19) | Step 2 installs Node 22 first |
| `grep: vllm.log: binary file matches` while reading the log | The log contains progress-bar control characters | Use `grep -a` |
| The answer looks unrelated to what you asked | The app picks the paper from the question; generic questions match many papers | Name the paper in the question (see below) |

## Asking questions

QASPER questions assume you already know the paper ("What datasets are used?"), so in the app they can match the wrong one. Put the paper's title or distinctive terms in the question:

> In the paper "PO-EMO: Conceptualization, Annotation, and Modeling of Aesthetic Emotions", what datasets are used?

The notebook's "Demo questions" cell prints random sample questions with their paper title and gold answer. An answer takes roughly 10–30 seconds. The reply shows the answer first, the cited paragraphs (click to inspect), and below them the agent's trace (paper picked → search → read → drafting answer).

## Limits

- **Temporary.** The URL lives only while the Colab runtime runs; it changes every time, and everything under `/content` (including the merged model) is lost on disconnect, so a new session repeats all steps.
- **No authentication.** Anyone with the URL can use the GPU. Share it privately and disconnect the runtime (*Runtime → Disconnect and delete runtime*) when done.
- **Not a production deployment.** Free Colab disconnects idle sessions; use it for demos, not as a standing service.

## Other checkpoints

Change the checkpoint folder in steps 3–4 (`/content/ckpt/p1-checkpoint`). For a 7B checkpoint, also change the base model to `Qwen/Qwen2.5-7B-Instruct`; 7B in fp16 is about 15 GB of weights, which does not fit a T4's 15 GB alongside the KV cache, so use a larger GPU (L4 / A100) for 7B.
