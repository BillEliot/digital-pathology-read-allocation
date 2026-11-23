"""Compare held-out fixed and sequential allocation using both score estimators."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pilot_allocation import panels
from sequential_eligibility_revision import SCHEDULES, calibration, split_suffix
from sequential_revision_audit import (
    draws_for, estimates, fixed_count, metrics, panel_combinations, ratio, sequential_count,
)


ESTIMATORS = ("phase", "pooled")


def replay(y, quota, tolerance, target, rand, priorities=None, grid=False):
    size = len(y)
    draws = len(rand[0])
    order, _ = rand
    flag = y[..., 1].all(axis=1)
    truth = ratio(y.sum(axis=(0, 1)))
    if not np.isfinite(truth):
        raise ValueError("The full-frame score must be defined")

    def fixed(h):
        if priorities is not None:
            return split_suffix(order, priorities, quota, h, flag)
        sample, shared = fixed_count(size, draws, quota, h, rand)
        return sample, flag[shared].sum(axis=1)

    def errors(h, sample):
        pooled, phase = estimates(y, h, sample, "pooled")
        return np.stack((phase - truth, pooled - truth))

    def joint(error, common, eps, required):
        return (np.abs(error) <= eps + 1e-12) & (common >= required)

    def sequential(required):
        error = np.empty((2, draws))
        common = np.empty(draws, dtype=int)
        for h, take, _, sample, shared in sequential_count(size, draws, quota, required, quota, flag, rand):
            if priorities is None:
                achieved = flag[shared].sum(axis=1)
            else:
                sample, achieved = split_suffix(order[take], priorities[take], quota, h, flag)
            error[:, take] = errors(h, sample)
            common[take] = achieved
        return error, common

    seq_error, seq_common = sequential(target)
    seq_rows = [
        dict(estimator=name, **metrics(error, seq_common, tolerance, target))
        for name, error in zip(ESTIMATORS, seq_error, strict=True)
    ]
    seq_pass = joint(seq_error, seq_common, tolerance, target)
    main_rows, grid_rows, seq_grid = [], [], []
    tolerances = (tolerance / 2, tolerance, tolerance * 2)
    targets = (target - 1, target, target + 1)
    for h in range(quota + 1):
        sample, common = fixed(h)
        for name, error, success in zip(ESTIMATORS, errors(h, sample), seq_pass, strict=True):
            fixed_pass = joint(error, common, tolerance, target)
            main_rows.append(dict(
                estimator=name, reserve=h, gained=int((success & ~fixed_pass).sum()),
                lost=int((fixed_pass & ~success).sum()),
                **metrics(error, common, tolerance, target),
            ))
            if grid:
                for req in targets:
                    for eps in tolerances:
                        grid_rows.append(dict(estimator=name, reserve=h, target=req, tolerance=eps,
                                              joint_pass=int(joint(error, common, eps, req).sum()), draws=draws))
    if grid:
        for req in targets:
            error, common = (seq_error, seq_common) if req == target else sequential(req)
            for name, z in zip(ESTIMATORS, error, strict=True):
                for eps in tolerances:
                    seq_grid.append(dict(estimator=name, target=req, tolerance=eps,
                                         joint_pass=int(joint(z, common, eps, req).sum()), draws=draws))
    return main_rows, seq_rows, grid_rows, seq_grid


def calibrated(fixed, seq, estimators=ESTIMATORS):
    selected, summary = [], []
    for name in estimators:
        f = fixed[fixed.estimator.eq(name)]
        s = seq[seq.estimator.eq(name)]
        held, ranges, _ = calibration(f, s)
        selected.append(held.assign(estimator=name))
        summary.append(ranges.assign(estimator=name))

    held = pd.concat(selected, ignore_index=True)
    keys = ["dataset", "group", "suffix", "estimator"]
    held = held.merge(fixed, left_on=[*keys, "heldout_reserve"], right_on=[*keys, "reserve"], validate="1:1")
    held = held.merge(seq, on=keys, suffixes=("_fixed", "_sequential"), validate="1:1")
    assert (held.draws_fixed == held.draws_sequential).all()
    assert (held.joint_pass_sequential - held.joint_pass_fixed == held.gained - held.lost).all()

    summary = pd.concat(summary, ignore_index=True).query("benchmark == 'heldout'")
    rows = []
    for key, block in held.groupby(["dataset", "suffix", "estimator"]):
        n = block.draws_fixed
        variance = ((block.gained + block.lost) - (block.gained - block.lost) ** 2 / n) / (n * (n - 1))
        for arm in ("fixed", "sequential"):
            counts = block[f"defined_{arm}"]
            rows.append(dict(
                dataset=key[0], suffix=key[1], estimator=key[2], allocation=arm,
                score_success=(block[f"score_pass_{arm}"] / n).mean(),
                material_success=(block[f"review_pass_{arm}"] / n).mean(),
                joint_success=(block[f"joint_pass_{arm}"] / n).mean(),
                undefined_rate=(1 - counts / n).mean(),
                bias=block[f"sum_error_{arm}"].sum() / counts.sum(),
                mae=block[f"sum_abs_{arm}"].sum() / counts.sum(),
                rmse=np.sqrt(block[f"sum_error2_{arm}"].sum() / counts.sum()),
                paired_mcse=np.sqrt(variance.sum()) / len(block),
            ))
    return held, summary, pd.DataFrame(rows)


def sensitivity(fixed, seq):
    rows = []
    for key, group in fixed.groupby(["dataset", "estimator", "target", "tolerance"]):
        dataset, estimator, target, tolerance = key
        w = group.pivot(index="group", columns="reserve", values="joint_pass").dropna(axis=1)
        s = seq.query("dataset==@dataset and estimator==@estimator and target==@target and tolerance==@tolerance")
        s = s.set_index("group").reindex(w.index)
        assert s.joint_pass.notna().all() and group.draws.nunique() == 1
        counts = w.to_numpy()
        draws = group.draws.iloc[0]
        h = (counts.sum(axis=0) - counts).argmax(axis=1)
        rows.append(dict(dataset=dataset, estimator=estimator, target=target, tolerance=tolerance,
                         held_out=counts[np.arange(len(h)), h].mean() / draws,
                         reselected=counts.mean(axis=0).max() / draws,
                         sequential=s.joint_pass.mean() / draws))
    return pd.DataFrame(rows)


def alternate_panels(data, draws):
    rows = []
    indexes = {}
    for dataset, group, ids, y, quota, tolerance, target, flag in panel_combinations(data):
        index = indexes.setdefault((dataset, group), len(indexes))
        rand = draws_for(len(y), draws, 261128, index)
        reserve = {"FDA_HTT": 5, "CrowdGleason": 2}[dataset]
        truth = ratio(y.sum(axis=(0, 1)))
        if not np.isfinite(truth):
            continue
        sample, shared = fixed_count(len(y), draws, quota, reserve, rand)
        assert flag.shape == (len(y),)
        ready = flag[shared].sum(axis=1) >= target
        def successes(h, sample, ready):
            return np.array([((np.abs(est - truth) <= tolerance + 1e-12) & ready).sum()
                             for est in reversed(estimates(y, h, sample, "pooled"))])
        fixed = successes(reserve, sample, ready)
        seq = sum(
            successes(h, sample, flag[shared].sum(axis=1) >= target)
            for h, _, _, sample, shared in sequential_count(len(y), draws, quota, target, quota, flag, rand)
        )
        for name, f, s in zip(ESTIMATORS, fixed, seq, strict=True):
            rows.append(dict(dataset=dataset, group=group, panel="/".join(map(str, ids)),
                             estimator=name, draws=draws, fixed=f / draws, sequential=s / draws))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kidney", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    draws = 64 if args.smoke else 4000
    outputs = [[], [], [], []]
    for i, (dataset, group, y, quota, _, tol, target) in enumerate(panels(args.kidney, args.data)):
        if args.smoke and i not in (0, 1, 30, 31, 94, 95):
            continue
        rand = draws_for(len(y), draws, 261121, i)
        rng = np.random.default_rng(np.random.SeedSequence([261135, i]))
        priority = rng.random((draws, 3, len(y))).argsort(axis=2)
        for suffix in SCHEDULES:
            cyclic = suffix == "cyclic"
            part = replay(y, quota, tol, target, rand, None if cyclic else priority, grid=cyclic)
            for output, rows in zip(outputs, part, strict=True):
                output.extend(dict(dataset=dataset, group=group, suffix=suffix, **row) for row in rows)
    f, s, fg, sg = map(pd.DataFrame, outputs)
    held, comparison, scores = calibrated(f, s)
    tables = dict(fixed_by_frame=f, sequential_by_frame=s, calibration_by_frame=held,
                  joint_summary=comparison, score_summary=scores, fixed_grid=fg, sequential_grid=sg,
                  threshold_summary=sensitivity(fg, sg))
    if not args.smoke:
        tables["panel_joint_success"] = alternate_panels(args.data, 1000)
    for filename, table in tables.items():
        table.to_csv(args.output / f"{filename}.csv", index=False)


if __name__ == "__main__":
    main()
