import json

from backend import training_metrics


def test_rollout_is_not_double_counted_as_parameter_update(tmp_path):
    now = [0.0]
    metrics = training_metrics.PhaseMetrics(tmp_path / 'phases.jsonl', clock=lambda: now[0])
    metrics.start('parameter_update', step=0)
    now[0] = 2.0
    with metrics.phase('rollout_and_scoring', step=0):
        now[0] = 7.0
    now[0] = 10.0
    metrics.stop()
    rows = [json.loads(line) for line in metrics.path.read_text().splitlines()]
    assert [(r['phase'], r['seconds']) for r in rows] == [
        ('parameter_update', 2.0), ('rollout_and_scoring', 5.0), ('parameter_update', 3.0)]
    assert all(r['step'] == 0 for r in rows)
    assert all(r['torch_peak_allocated_bytes'] is None for r in rows)
