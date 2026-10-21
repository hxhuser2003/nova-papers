import json
from scripts import summarize_training_metrics as summary


def test_summary_keeps_phase_memory_and_duration_separate(tmp_path):
    phases = tmp_path / 'phases.jsonl'
    phases.write_text('\n'.join(json.dumps(r) for r in [
        dict(phase='rollout_and_scoring', seconds=3, wall_start=0, wall_end=3,
             torch_peak_allocated_bytes=100, torch_peak_reserved_bytes=200),
        dict(phase='parameter_update', seconds=2, wall_start=3, wall_end=5,
             torch_peak_allocated_bytes=300, torch_peak_reserved_bytes=400)]))
    result = summary.summarize(phases)
    assert result['rollout_and_scoring']['seconds'] == 3
    assert result['parameter_update']['seconds'] == 2
    assert result['parameter_update']['torch_peak_allocated_bytes'] == 300
    assert result['parameter_update']['device_sampled_peak_mib'] is None
