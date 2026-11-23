"""Vary per-reader quotas in the phase-expanded cyclic replay."""

import argparse
from itertools import islice
from pathlib import Path

import pandas as pd

from pilot_allocation import panels
from sequential_phase_revision import calibrated, replay
from sequential_revision_audit import draws_for


QUOTAS = {"KidneyAI": (30, 40, 50), "FDA_HTT": (4, 5, 6)}


def quota_replay(kidney, data, smoke):
    draws = 64 if smoke else 4000
    fixed, sequential = [], []
    # Read the 30 KidneyAI and 64 FDA frames in original seed order.
    compared = islice(panels(kidney, data), 94)
    for index, (dataset, group, y, base_quota, _, tolerance, target) in enumerate(compared):
        if smoke and index not in (0, 1, 30, 31):
            continue
        quotas = QUOTAS[dataset]
        assert quotas[1] == base_quota and target <= min(quotas) <= max(quotas) <= len(y)
        randomization = draws_for(len(y), draws, 261121, index)
        for quota in quotas:
            fixed_rows, sequential_rows, _, _ = replay(y, quota, tolerance, target, randomization)
            meta = dict(dataset=dataset, group=group, suffix="cyclic", quota=quota)
            fixed.extend(dict(**meta, **row) for row in fixed_rows if row["estimator"] == "phase")
            sequential.extend(dict(**meta, **row) for row in sequential_rows if row["estimator"] == "phase")
    s = pd.DataFrame(sequential)
    assert s.groupby("dataset").group.nunique().to_dict() == (
        {"FDA_HTT": 2, "KidneyAI": 2} if smoke else {"FDA_HTT": 64, "KidneyAI": 30}
    )
    return pd.DataFrame(fixed), s


def fit_by_quota(fixed, sequential):
    outputs = [], [], []
    for (dataset, quota), candidates in fixed.groupby(["dataset", "quota"]):
        sequence = sequential[sequential.dataset.eq(dataset) & sequential.quota.eq(quota)]
        for tables, frame in zip(outputs, calibrated(
            candidates.drop(columns="quota"), sequence.drop(columns="quota"), estimators=("phase",)
        ), strict=True):
            tables.append(frame.assign(quota=quota))
    return [pd.concat(tables, ignore_index=True) for tables in outputs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kidney", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fixed, sequential = quota_replay(args.kidney, args.data, args.smoke)
    held, joint, scores = fit_by_quota(fixed, sequential)
    for name, data in dict(fixed_by_frame=fixed, sequential_by_frame=sequential,
                           calibration_by_frame=held, joint_summary=joint, score_summary=scores).items():
        data.to_csv(args.output / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
