"""Aggregate disjoint phase times and optional whole-device samples.

nvidia-smi timestamps use the node's local timezone, as does datetime.timestamp.
Device samples include all processes on the selected GPU, not just this job.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path


def summarize(phases, device_samples=None):
    samples = []
    invalid_rows = 0
    error = None
    if device_samples is not None:
        try:
            with Path(device_samples).open() as stream:
                for row in csv.reader(stream):
                    if not row:
                        continue
                    try:
                        timestamp, used = row
                        used = float(used)
                        if not math.isfinite(used) or used < 0:
                            raise ValueError('Invalid memory sample')
                        stamp = datetime.strptime(timestamp.strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp()
                    except (ValueError, TypeError):
                        invalid_rows += 1
                        continue
                    samples.append((stamp, used))
        except OSError as exc:
            error = str(exc)
    result = {}
    for line in Path(phases).read_text().splitlines():
        row = json.loads(line)
        entry = result.setdefault(row['phase'], dict(seconds=0.0, torch_peak_allocated_bytes=None,
                                                     torch_peak_reserved_bytes=None, device_sampled_peak_mib=None))
        entry['seconds'] += row['seconds']
        for key in ('torch_peak_allocated_bytes', 'torch_peak_reserved_bytes'):
            if row[key] is not None:
                entry[key] = max(entry[key] or 0, row[key])
        values = [v for t, v in samples if row['wall_start'] <= t < row['wall_end']]
        if values:
            entry['device_sampled_peak_mib'] = max(entry['device_sampled_peak_mib'] or 0, max(values))
    result['_device_sampling'] = dict(
        status=('not_requested' if device_samples is None else
                'unavailable' if not samples else 'partial' if invalid_rows else 'valid_samples'),
        valid_rows=len(samples), invalid_rows=invalid_rows, error=error,
        note='Whole-device samples, not process-only memory; null phase peaks mean no samples in that interval.')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phases')
    parser.add_argument('--device-samples')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(summarize(args.phases, args.device_samples), indent=2))


if __name__ == '__main__':
    main()
