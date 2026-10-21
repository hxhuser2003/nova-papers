"""Aggregate disjoint phase times and optional whole-device samples.

nvidia-smi timestamps use the node's local timezone, as does datetime.timestamp.
Device samples include all processes on the selected GPU, not just this job.
"""
from __future__ import annotations
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def summarize(phases, device_samples=None):
    samples = []
    if device_samples is not None:
        with Path(device_samples).open() as stream:
            for row in csv.reader(stream):
                timestamp, used = row
                samples.append((datetime.strptime(timestamp.strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp(), float(used)))
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
