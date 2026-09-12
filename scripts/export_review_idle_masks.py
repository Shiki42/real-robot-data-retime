"""Export retiming-inserted waits for review; no training hook is implied."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from real_robot_data_retime.retime import boolean_ranges


def clock_idle_mask(clock):
    """Mask duplicate source-clock samples, keeping one supervised boundary.

    Leading/interior holds retain the departure sample. Terminal holds retain
    the arrival sample. A constant clock has no execution and is entirely idle.
    Equality is exact: small positive speed-ramp steps remain supervised.
    """
    clock = np.asarray(clock, dtype=float)
    if clock.ndim != 1 or not len(clock) or not np.isfinite(clock).all():
        raise ValueError('Expected a nonempty finite source clock')
    if np.any(np.diff(clock) < 0):
        raise ValueError('Source clock must be monotone')
    idle = np.zeros(len(clock), dtype=bool)
    if len(clock) > 1 and np.all(clock == clock[0]):
        idle[:] = True
        return idle
    for span in boolean_ranges(np.diff(clock) == 0):
        if span.end == len(clock) - 1:
            idle[span.start + 1:span.end + 1] = True
        else:
            idle[span.start:span.end] = True
    return idle


def required_open_wait(left, right, stages):
    """Holding at the lift peak before the drawer is open is supervised."""
    return ((np.asarray(left) == stages['peak_source_frame'])
            & (np.asarray(right) < stages['open_source_frame']))


def export(dataset, manifest, output):
    dataset, output = Path(dataset), Path(output)
    review = json.loads(Path(manifest).read_text())
    records = []
    for row in review['episodes_data']:
        ep = row['output']
        table = pq.read_table(dataset / f'data/chunk-000/file-{ep:03d}.parquet',
                              columns=['retime.left_source_frame', 'retime.right_source_frame', 'retime.synthetic_hold'])
        if len(table) != row['frames']:
            raise ValueError(f'Video and numeric frame counts differ: {ep}')
        receipt = json.loads((dataset / f'meta/retime_receipts/episode_{ep:03d}.json').read_text())
        if receipt['source_episode_index'] != row['source']:
            raise ValueError(f'Video and numeric source IDs differ: {ep}')
        record = dict(output=ep, source=row['source'], frames=len(table))
        with np.load(dataset / f'meta/retime_source_indices/episode_{ep:03d}.npz') as maps:
            required_wait = required_open_wait(maps['left'], maps['right'], receipt['plan']['stages'])
            record['left_required_open_wait'] = [[span.start, span.end] for span in boolean_ranges(required_wait)]
            for arm in ['left', 'right']:
                clock = table[f'retime.{arm}_source_frame'].to_numpy()
                if not np.array_equal(clock, maps[arm]):
                    raise ValueError(f'Stored source clock differs from receipt map: {ep}/{arm}')
                idle = clock_idle_mask(clock) | table['retime.synthetic_hold'].to_numpy()
                if arm == 'left':
                    idle[required_wait] = False
                record[arm] = [[span.start, span.end] for span in boolean_ranges(idle)]
        records.append(record)
    result = dict(schema_version=1, policy='retiming_waits_except_required_drawer_open_wait',
                  supervised_wait='Left arm holding at the lift peak until the right arm opens the drawer remains supervised (loss=1).',
                  training_status='not_connected_to_training', fps=review['fps'],
                  interval_convention='[start, end), zero-based output frames',
                  mask_semantics='idle=true means exclude that arm from action loss; left action[0:7], right action[7:14]',
                  boundary_policy='Keep departure sample of leading/interior plateaus and arrival sample of terminal plateaus. Entirely constant clocks and explicit synthetic holds are idle.',
                  episodes=records)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.dataset, args.manifest, args.output)
    print(f'Exported {len(result["episodes"])} episodes to {args.output}')
