import json
import pytest
from scripts.verify_lora_reload import check_metadata

def write(tmp_path, base='Qwen/Qwen2.5-3B-Instruct', changed=504):
    (tmp_path/'adapter_config.json').write_text(json.dumps(dict(base_model_name_or_path=base, peft_type='LORA')))
    (tmp_path/'training_verification.json').write_text(json.dumps(dict(model='Qwen/Qwen2.5-3B-Instruct',global_steps=2,changed_trainable_tensors=changed)))
    (tmp_path/'adapter_model.safetensors').write_bytes(b'placeholder')

def test_metadata_agrees(tmp_path):
    write(tmp_path)
    assert check_metadata(tmp_path)[0]['base_model_name_or_path']=='Qwen/Qwen2.5-3B-Instruct'

@pytest.mark.parametrize('base,changed',[('other',504),('Qwen/Qwen2.5-3B-Instruct',0)])
def test_metadata_rejects_mismatch_or_no_update(tmp_path,base,changed):
    write(tmp_path,base,changed)
    with pytest.raises(ValueError): check_metadata(tmp_path)
