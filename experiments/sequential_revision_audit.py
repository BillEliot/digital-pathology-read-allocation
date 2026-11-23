"""Cross allocation and score estimators on identical draws, with stopping diagnostics."""

import argparse
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from overlap_pilot import schedule
from pilot_allocation import load_htt_ratings, panels, shared_slots
from review_supplement import ORDERS

BEST = {"KidneyAI": 15, "FDA_HTT": 5, "CrowdGleason": 2}
READERS = np.arange(3)[None, :, None]


def draws_for(size, draws, seed, index):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    first = rng.random((draws, 3, size)).argsort(axis=2)[:, 0]
    return first, ORDERS[rng.integers(len(ORDERS), size=draws)]


def fixed_count(size, draws, quota, h, rand):
    order, readers = rand
    slots = schedule(size, quota, h)
    sample = order[np.arange(draws)[:, None, None], slots[readers]]
    return sample, order[:, shared_slots(slots)]


def sequential_count(size, draws, quota, target, cap, flag, rand):
    order, readers = rand
    progress = flag[order[:, :cap]].cumsum(axis=1)
    h = np.minimum((progress < target).sum(axis=1) + 1, cap)
    reached = progress[np.arange(draws), h - 1] >= target
    for k in np.unique(h):
        take = h == k
        slots = schedule(size, quota, k)
        sample = order[take][np.arange(take.sum())[:, None, None], slots[readers[take]]]
        yield k, take, reached[take], sample, order[take][:, shared_slots(slots)]


def ratio(totals):
    num, den = np.moveaxis(totals, -1, 0)
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def target_score(totals, estimand):
    if estimand == "equal_reader":
        return ratio(totals).mean(axis=-1)
    return ratio(totals.sum(axis=-2))


def estimates(y, h, sample, estimand):
    values = y[sample, READERS]
    ordinary = target_score(values.sum(axis=2), estimand)
    q = sample.shape[2] - h
    if not q:
        return ordinary, ordinary
    weighted = values[:, :, :h].sum(axis=2) + (len(y) - h) / q * values[:, :, h:].sum(axis=2)
    return ordinary, target_score(weighted, estimand)


def metrics(error, common, tol, required):
    defined = np.isfinite(error)
    observed = error[defined]
    ready = common >= required
    within = np.abs(error) <= tol + 1e-12
    return {
        "draws": len(error), "defined": len(observed), "sum_error": observed.sum(),
        "sum_error2": np.square(observed).sum(), "sum_abs": np.abs(observed).sum(),
        "score_pass": int(within.sum()), "review_pass": int(ready.sum()),
        "joint_pass": int((within & ready).sum()), "sum_common": common.sum(),
    }


def one_frame(dataset, group, y, n, tol, target, flag, rand, *, auxiliary=False):
    draws = len(rand[0])
    full = y.sum(axis=0)
    out, branch, grid, alternatives = [], [], [], []

    def score(values, estimand, label, req, tolerance, modes=None):
        truth = target_score(values.sum(axis=0), estimand)
        if not np.isfinite(truth):
            return
        fixed_h = BEST[dataset]
        sample, shared = fixed_count(len(y), draws, n, fixed_h, rand)
        for method, est in zip(("pooled", "phase"), estimates(values, fixed_h, sample, estimand)):
            if modes is None or method in modes:
                yield label, "fixed", method, est - truth, flag[shared].sum(axis=1)
        for name, cap in (("reserve_followup", n-1), ("full_cap", n)):
            errors = np.empty((2, draws))
            common = np.empty(draws, dtype=int)
            groups = np.empty(draws, dtype="U18")
            for h, take, reached, sample, shared in sequential_count(len(y), draws, n, req, cap, flag, rand):
                errors[:, take] = np.array(estimates(values, h, sample, estimand)) - truth
                common[take] = flag[shared].sum(axis=1)
                endings = ("goal_full", "below_full") if h == n else ("goal_followup", "below_followup")
                groups[take] = np.where(reached, *endings)
            for method, err in zip(("pooled", "phase"), errors):
                if modes is None or method in modes:
                    yield label, name, method, err, common
                    if label == "main":
                        for state in np.unique(groups):
                            take = groups == state
                            branch.append(dict(dataset=dataset, group=group, allocation=name,
                                               estimator=method, state=state,
                                               **metrics(err[take], common[take], tolerance, req)))

    for _, design, estim, err, common in score(y, "pooled", "main", target, tol):
        out.append(dict(dataset=dataset, group=group, allocation=design, estimator=estim,
                        **metrics(err, common, tol, target)))

    truth = target_score(full, "pooled")
    for h in range(n + 1):
        sample, shared = fixed_count(len(y), draws, n, h, rand)
        ordinary, _ = estimates(y, h, sample, "pooled")
        row = metrics(ordinary - truth, flag[shared].sum(axis=1), tol, target)
        alternatives.append(dict(dataset=dataset, group=group, count=h, **row))

    if auxiliary:
        for mode, v, eps in (("equal_reader", y, tol), ("thresholded", y.copy(), .1)):
            if mode == "thresholded":
                if dataset not in {"KidneyAI", "CrowdGleason"}:
                    continue
                v[..., 0] = (v[..., 0] >= 2) & (v[..., 1] > 0)
            for _, design, estim, err, common in score(v, mode, mode, target, eps):
                grid.append(dict(dataset=dataset, group=group, analysis=mode, tolerance=eps, target=target,
                                 allocation=design, estimator=estim, **metrics(err, common, eps, target)))
        for req in (target-1, target, target+1):
            for _, design, estim, err, common in score(y, "pooled", "grid", req, tol):
                for eps in (tol / 2, tol, tol * 2):
                    grid.append(dict(dataset=dataset, group=group, analysis="grid", tolerance=eps, target=req,
                                     allocation=design, estimator=estim, **metrics(err, common, eps, req)))
    return out, branch, grid, alternatives


def fixed_held_out(rows):
    result = []
    for name, z in rows.groupby("dataset"):
        wide = z.assign(p=z.joint_pass / z.draws).pivot(index="group", columns="count", values="p").dropna(axis=1)
        selected = wide.columns.to_numpy()[(wide.sum().to_numpy()[None, :] - wide.to_numpy()).argmax(axis=1)]
        for group, h, values in zip(wide.index, selected, wide.to_dict("records"), strict=True):
            result.append(dict(dataset=name, group=group, holdout_count=h, holdout_joint=values[h]))
    return pd.DataFrame(result)


def panel_combinations(data):
    htt = load_htt_ratings(data)
    for slide, rows in htt.query("modalityID == 'camic'").groupby("WSI"):
        observed = rows.pivot(index="caseID", columns="readerID", values="evaluable")
        densities = rows.pivot(index="caseID", columns="readerID", values="densityTILs")
        for ids in combinations(observed.columns[observed.notna().all()], 3):
            eligible = observed.loc[:, list(ids)].to_numpy(dtype=bool)
            ratings = densities.loc[:, list(ids)].to_numpy(dtype=float)
            y = np.stack((np.where(eligible, ratings / 100, 0), eligible), axis=-1)
            yield "FDA_HTT", slide, ids, y, 5, .05, 2, eligible.all(axis=1)

    with ZipFile(data / "CrowdGleason_Annotations.zip") as source:
        test = pd.read_csv(source.open("test.csv"))
    readers = [f"marker{i}" for i in range(1, 8)]
    groups = test["Patch filename"].str.extract(r"^(.*)_row_\d+_column_\d+(?:\.[^.]+)?$", expand=False)
    complete = test[readers].isin([0, 1, 2, 3]).all(axis=1)
    for group, block in test.loc[complete, readers].groupby(groups[complete]):
        if len(block) < 10:
            continue
        for ids in combinations(readers, 3):
            v = block.loc[:, list(ids)].to_numpy()
            y = np.stack((v, np.ones(v.shape)), axis=-1)
            yield "CrowdGleason", group, ids, y, max(4, min(40, round(.4 * len(v)))), .1, 2, np.ones(len(v), dtype=bool)


def alternate_panels(data):
    rows = []
    indexes = {}
    for dataset, group, ids, y, n, tol, target, flag in panel_combinations(data):
        draws = 1000
        index = indexes.setdefault((dataset, group), len(indexes))
        rand = draws_for(len(y), draws, 261128, index)
        truth = target_score(y.sum(axis=0), "pooled")
        if not np.isfinite(truth):
            continue
        s, share = fixed_count(len(y), draws, n, BEST[dataset], rand)
        fixed, _ = estimates(y, BEST[dataset], s, "pooled")
        f = metrics(fixed-truth, flag[share].sum(axis=1), tol, target)
        seq = np.zeros(2, dtype=int)
        for h, _, _, s, share in sequential_count(len(y), draws, n, target, n, flag, rand):
            common = flag[share].sum(axis=1)
            for j, estimate in enumerate(estimates(y, h, s, "pooled")):
                seq[j] += metrics(estimate-truth, common, tol, target)["joint_pass"]
        rows.append(dict(dataset=dataset, group=group, panel="/".join(map(str, ids)), draws=draws,
                         fixed_joint=f["joint_pass"] / draws, seq_pooled_joint=seq[0] / draws,
                         seq_phase_joint=seq[1] / draws))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kidney", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--draws", type=int, default=4000)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    combined = panels(args.kidney, args.data)
    output = ([], [], [], [])
    for i, (dataset, group, y, quota, _, tol, req) in enumerate(combined):
        if args.smoke and i > 1:
            break
        flag = (y[..., 1] >= 1).all(axis=1)
        rand = draws_for(len(y), args.draws, 261121, i)
        result = one_frame(dataset, group, y, quota, tol, req, flag, rand, auxiliary=True)
        for target, rows in zip(output, result, strict=True):
            target.extend(rows)
    for name, rows in zip(("estimators", "stopping", "sensitivity", "fixed_grid"), output, strict=True):
        pd.DataFrame(rows).to_csv(args.output / f"{name}.csv", index=False)
    fixed_held_out(pd.DataFrame(output[-1])).to_csv(args.output / "fixed_held_out.csv", index=False)
    if not args.smoke:
        alternate_panels(args.data).to_csv(args.output / "reader_panels.csv", index=False)


if __name__ == "__main__":
    main()
