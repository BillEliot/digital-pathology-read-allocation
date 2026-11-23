"""Replay KidneyAI cumulative grade proportions under the primary allocation design."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kidney_data import VALUES, load_panel
from sequential_eligibility_revision import ordinal_error
from sequential_phase_revision import calibrated
from sequential_revision_audit import READERS, draws_for, estimates, fixed_count, metrics, sequential_count


QUOTA, MATERIAL, TOLERANCE = 40, 5, .10


def contributions(y):
    grade, eligible = np.moveaxis(y, -1, 0)
    cutoffs = (grade[..., None] >= np.arange(1, 4)) & eligible[..., None].astype(bool)
    return np.concatenate((cutoffs, eligible[..., None]), axis=-1)


def error_by_draw(values, h, sample, truth):
    selected = values[sample, READERS]
    q = sample.shape[-1] - h
    totals = selected.sum(axis=(1, 2))
    if q:
        prefix = selected[:, :, :h].sum(axis=(1, 2))
        totals = prefix + (len(values) - h) / q * (totals - prefix)
    denominator = totals[:, 3, None]
    estimate = np.divide(totals[:, :3], denominator, out=np.full((len(totals), 3), np.nan),
                         where=denominator > 0)
    return np.max(np.abs(estimate - truth), axis=1)


def check_weights(y, values, truth, rand):
    # Direct comparison with three independent calls to the original score estimator.
    for h in (0, 14, QUOTA):
        sample, _ = fixed_count(len(y), len(rand[0]), QUOTA, h, rand)
        reconstructed = [
            estimates(values[:, :, [cutoff, 3]], h, sample, "pooled")[1]
            for cutoff in range(3)
        ]
        expected = np.max(np.abs(np.column_stack(reconstructed) - truth), axis=1)
        np.testing.assert_allclose(error_by_draw(values, h, sample, truth), expected, atol=1e-13)
        if h in (0, QUOTA):
            np.testing.assert_allclose(expected, ordinal_error(y, sample), atol=1e-13)


def replay(group, y, draws, index):
    values = contributions(y)
    truth = values[:, :, :3].sum(axis=(0, 1)) / values[:, :, 3].sum()
    joint_flag = y[..., 1].all(axis=1)
    rand = draws_for(len(y), draws, 261121, index)
    check_weights(y, values, truth, tuple(part[:32] for part in rand))
    err = np.empty(draws)
    common = np.empty(draws, dtype=int)
    for h, take, _, sample, shared in sequential_count(len(y), draws, QUOTA, MATERIAL, QUOTA, joint_flag, rand):
        err[take] = error_by_draw(values, h, sample, truth)
        common[take] = joint_flag[shared].sum(axis=1)
    seq_pass = (err <= TOLERANCE + 1e-12) & (common >= MATERIAL)
    meta = dict(dataset="KidneyAI", group=group, suffix="cyclic", estimator="phase")
    sequence = dict(**meta, **metrics(err, common, TOLERANCE, MATERIAL))
    fixed = []
    for reserve in range(QUOTA + 1):
        sample, shared = fixed_count(len(y), draws, QUOTA, reserve, rand)
        err = error_by_draw(values, reserve, sample, truth)
        common = joint_flag[shared].sum(axis=1)
        fixed_pass = (err <= TOLERANCE + 1e-12) & (common >= MATERIAL)
        fixed.append(dict(
            **meta, reserve=reserve, gained=int((seq_pass & ~fixed_pass).sum()),
            lost=int((fixed_pass & ~seq_pass).sum()), **metrics(err, common, TOLERANCE, MATERIAL),
        ))
    return fixed, sequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kidney", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    _, slides, labels = load_panel(args.kidney, labels_only=True)
    groups = np.unique(slides)
    assert len(groups) == 30 and len(labels) == 3928
    draws = 64 if args.smoke else 4000
    fixed, seq = [], []
    for index, group in enumerate(groups[:2] if args.smoke else groups):
        rows, sequence = replay(group, VALUES[labels[slides == group]], draws, index)
        fixed.extend(rows)
        seq.append(sequence)
    f, s = pd.DataFrame(fixed), pd.DataFrame(seq)
    # The shared calibration routine emits a summary for each included estimator.
    held, joint, score = calibrated(f, s, estimators=("phase",))
    score = score.drop(columns="bias").rename(columns={"mae": "mean_max_error", "rmse": "rms_max_error"})
    for name, table in dict(fixed_by_frame=f, sequential_by_frame=s,
                            calibration_by_frame=held, joint_summary=joint, score_summary=score).items():
        table.to_csv(args.output / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
