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


def test_invalid_monitor_text_is_reported_not_a_training_failure(tmp_path):
    phases = tmp_path/'phases.jsonl'
    phases.write_text(json.dumps(dict(phase='parameter_update', seconds=2, wall_start=0, wall_end=5,
        torch_peak_allocated_bytes=300, torch_peak_reserved_bytes=400)))
    device = tmp_path/'device.csv'
    device.write_text('No devices were found\n')
    result = summary.summarize(phases, device)
    assert result['parameter_update']['seconds'] == 2
    assert result['parameter_update']['device_sampled_peak_mib'] is None
    assert result['_device_sampling']['status'] == 'unavailable'
    assert result['_device_sampling']['invalid_rows'] == 1


def test_mixed_monitor_rows_report_partial(tmp_path):
    phases = tmp_path/'phases.jsonl'
    phases.write_text(json.dumps(dict(phase='rollout_and_scoring', seconds=2, wall_start=0, wall_end=9999999999,
        torch_peak_allocated_bytes=100, torch_peak_reserved_bytes=200)))
    device = tmp_path/'device.csv'
    device.write_text('2026/10/22 23:00:00.000, 1234\nNo devices were found\n')
    result = summary.summarize(phases, device)
    assert result['rollout_and_scoring']['device_sampled_peak_mib'] == 1234
    assert result['_device_sampling']['status'] == 'partial'
