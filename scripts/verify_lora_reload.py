"""Reload an existing LoRA without optimization; verify device monitoring too."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from contextlib import nullcontext
from scripts.gpu_monitor import resolve_uuid
from scripts.summarize_training_metrics import summarize


def check_metadata(path):
    path = Path(path)
    config = json.loads((path/'adapter_config.json').read_text())
    report = json.loads((path/'training_verification.json').read_text())
    if config.get('peft_type') != 'LORA' or config.get('base_model_name_or_path') != report.get('model'):
        raise ValueError('Adapter and training report base models do not agree')
    if report.get('global_steps', 0) <= 0 or report.get('changed_trainable_tensors', 0) <= 0:
        raise ValueError('Training report does not confirm parameter updates')
    if not (path/'adapter_model.safetensors').stat().st_size:
        raise ValueError('Empty adapter weights')
    return config, report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    checkpoint=Path(args.checkpoint).resolve()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    config, training = check_metadata(checkpoint)
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from peft.utils.save_and_load import get_peft_model_state_dict
    from huggingface_hub import snapshot_download
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability()[0] >= 8
    weights=load_file(str(checkpoint/'adapter_model.safetensors'),device='cpu')
    assert weights and all(torch.isfinite(t).all().item() for t in weights.values()), 'Non-finite adapter weights'
    digest=hashlib.sha256((checkpoint/'adapter_model.safetensors').read_bytes()).hexdigest()
    snapshot=snapshot_download(config['base_model_name_or_path'],revision=config.get('revision') or 'main',local_files_only=True)
    report=dict(checkpoint=str(checkpoint),adapter_sha256=digest,base_snapshot=snapshot,
        original_revision_pinned=bool(config.get('revision')),training=training,
        scope='PEFT reload and inference only; no optimization, no vLLM allocator validation',status='started')
    def save(): (out/'reload-verification.json').write_text(json.dumps(report,indent=2))
    save()
    monitor=None
    start=time.time()
    torch.cuda.reset_peak_memory_stats()
    try:
        listing=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True)
        gpu=resolve_uuid(torch.cuda.get_device_properties(0).uuid,listing.splitlines())
        report['gpu_uuid']=gpu
        with (out/'device-memory.csv').open('w') as samples, (out/'monitor-errors.log').open('w') as errors:
            monitor=subprocess.Popen(['nvidia-smi','--id='+gpu,'--query-gpu=timestamp,memory.used',
                '--format=csv,noheader,nounits','--loop-ms=200'],stdout=samples,stderr=errors)
            tokenizer=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True)
            base=AutoModelForCausalLM.from_pretrained(snapshot,dtype=torch.bfloat16,local_files_only=True).to('cuda')
            model=PeftModel.from_pretrained(base,str(checkpoint),is_trainable=False).eval()
            loaded=get_peft_model_state_dict(model)
            assert set(loaded)==set(weights), 'Loaded adapter key mismatch'
            assert all(torch.equal(loaded[k].detach().cpu(),v.to(dtype=loaded[k].dtype)) for k,v in weights.items()), 'Loaded adapter differs from saved weights'
            report['verified_adapter_tensors']=len(weights)
            report['generations']=[]
            with torch.inference_mode():
                for prompt in ['Reply with exactly <answer>Yes</answer>.', 'What does a research paper abstract summarize? Answer briefly.']:
                    text=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
                    inputs=tokenizer(text,return_tensors='pt').to('cuda')
                    first_logits={}
                    for enabled in (False,True):
                        with nullcontext() if enabled else model.disable_adapter():
                            logits=model(**inputs).logits[:,-1,:].float()
                            assert torch.isfinite(logits).all().item(), 'Non-finite logits'
                            first_logits[enabled]=logits.cpu()
                            generated=model.generate(**inputs,max_new_tokens=64,do_sample=False,pad_token_id=tokenizer.eos_token_id)
                            continuation=generated[0,inputs['input_ids'].shape[1]:]
                            assert continuation.numel()>0
                            report['generations'].append(dict(prompt=prompt,adapter_enabled=enabled,
                                generated_tokens=continuation.numel(),output=tokenizer.decode(continuation,skip_special_tokens=True)))
                    report.setdefault('adapter_logit_max_abs_difference',[]).append((first_logits[True]-first_logits[False]).abs().max().item())
            assert any(v>0 for v in report['adapter_logit_max_abs_difference']), 'Adapter had no measurable effect on probe logits'
            torch.cuda.synchronize()
            end=time.time()
            row=dict(phase='reload_and_inference',step=0,seconds=end-start,wall_start=start,wall_end=end,
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            (out/'phase_metrics.jsonl').write_text(json.dumps(row)+'\n')
            assert monitor.poll() is None, 'Device monitor exited early'
            monitor.terminate(); monitor.wait(timeout=10); monitor=None
        result=summarize(out/'phase_metrics.jsonl',out/'device-memory.csv')
        (out/'phase-summary.json').write_text(json.dumps(result,indent=2))
        assert result['_device_sampling']['status']=='valid_samples', 'Invalid device samples'
        assert result['reload_and_inference']['device_sampled_peak_mib'] is not None, 'No samples during inference'
        report['status']='passed'
    except Exception as exc:
        report['status']='failed'; report['error']=repr(exc)
        raise
    finally:
        if monitor is not None:
            if monitor.poll() is None: monitor.terminate()
            monitor.wait(timeout=10)
        save()
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
