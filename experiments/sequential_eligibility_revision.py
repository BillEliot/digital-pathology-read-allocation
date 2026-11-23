"""Audit evaluability, suffix designs, held-out calibration, and ordinal recovery."""

import argparse
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from pilot_allocation import load_htt_ratings, panels
from sequential_revision_audit import BEST, READERS, draws_for, fixed_count, metrics, ratio, sequential_count


SCHEDULES = ("cyclic", "independent_suffix")


def fda_audit(path):
    ratings = load_htt_ratings(path).query("modalityID=='camic'")
    if ratings.duplicated(["WSI", "caseID", "readerID"]).any():
        raise ValueError("More than one rating per region and reader")
    audit = ratings.groupby(["WSI", "readerID"]).evaluable.agg(rows="size", statuses="count")
    audit["complete"] = audit.statuses.eq(10)
    size = ratings.groupby("WSI").caseID.nunique()
    assert size.eq(10).all()
    audit["missing_statuses"] = 10 - audit.statuses
    keep = ratings.join(audit.complete, on=["WSI", "readerID"])
    keep = keep[keep.complete].copy()
    keep["eligible"] = keep.evaluable.astype(bool)
    valid = keep["densityTILs"].between(0, 100)
    if not valid[keep.eligible].all():
        raise ValueError("Complete FDA reader has an invalid eligible density")
    reader = keep.groupby("readerID").eligible.agg(records="size", eligible="sum")
    reader["excluded"] = reader.records - reader.eligible
    reader["excluded_rate"] = reader.excluded / reader.records
    illustrator = (audit[audit.complete].reset_index().groupby("WSI").readerID.apply(lambda ids: ids.tolist()[:3]))
    return audit.reset_index(), reader.reset_index(), illustrator


def exclusion(dataset, group, y, readers):
    eligible = y[..., 1].astype(bool)
    frame = dict(dataset=dataset, group=group, regions=len(y), records=eligible.size,
                 excluded=int((~eligible).sum()))
    for k in range(4):
        frame[f"regions_with_{k}_eligible"] = int((eligible.sum(axis=1) == k).sum())
    grade = ratio(y.sum(axis=1))
    partly = ((eligible.sum(axis=1) > 0) & ~eligible.all(axis=1))
    jointly = eligible.all(axis=1)
    frame["mixed_region_mean"] = grade[partly].mean() if partly.any() else np.nan
    frame["full_region_mean"] = grade[jointly].mean() if jointly.any() else np.nan
    counts = [dict(dataset=dataset, group=group, reader=r, records=len(y), excluded=int((~eligible[:, i]).sum()))
              for i, r in enumerate(readers)]
    pairs = []
    for focal, peer in permutations(range(3), 2):
        observed = eligible[:, peer]
        for state, flag in (("focal_excluded", ~eligible[:, focal]), ("focal_eligible", eligible[:, focal])):
            scores = y[observed & flag, peer, 0]
            row = dict(dataset=dataset, group=group, focal=readers[focal], peer=readers[peer], state=state,
                       count=len(scores), mean=scores.mean() if len(scores) else np.nan, p50=np.median(scores) if len(scores) else np.nan)
            if dataset != "FDA_HTT":
                for k in range(4):
                    row[f"grade_{k}"] = int((scores == k).sum())
            pairs.append(row)
    return frame, counts, pairs


def split_suffix(order, priorities, quota, h, flag):
    draws = len(order)
    prefix = order[:, :h]
    q = quota - h
    common = flag[prefix].sum(axis=1)
    if not q:
        return np.broadcast_to(prefix[:, None, :], (draws, 3, h)), common
    n = order.shape[1]
    remaining = np.ones((draws, n), dtype=bool)
    remaining[np.arange(draws)[:, None], prefix] = False
    unread = remaining[np.arange(draws)[:, None, None], priorities]
    followup = priorities[unread].reshape(draws, 3, n - h)[..., :q]
    membership = np.zeros((draws, 3, n), dtype=bool)
    membership[np.arange(draws)[:, None, None], READERS, followup] = True
    common += (membership.all(axis=1) & flag).sum(axis=1)
    return np.concatenate((np.broadcast_to(prefix[:, None, :], (draws, 3, h)), followup), axis=2), common


def score_error(y, sample, truth):
    return ratio(y[sample, READERS].sum(axis=(1, 2))) - truth


def replay(dataset, group, y, quota, tol, target, draws, index):
    n = len(y)
    flag = y[..., 1].all(axis=1)
    rand = draws_for(n, draws, 261121, index)
    order, _ = rand
    priority_rng = np.random.default_rng(np.random.SeedSequence([261135, index]))
    priorities = priority_rng.random((draws, 3, n)).argsort(axis=2)
    truth = ratio(y.sum(axis=(0, 1)))
    observed = dict(dataset=dataset, group=group)
    fixed, seq, ordinal = [], [], []
    for policy in SCHEDULES:
        for h in range(quota + 1):
            if policy == "cyclic":
                sample, shared = fixed_count(n, draws, quota, h, rand)
                common = flag[shared].sum(axis=1)
            else:
                sample, common = split_suffix(order, priorities, quota, h, flag)
            fixed.append(dict(**observed, suffix=policy, reserve=h,
                              **metrics(score_error(y, sample, truth), common, tol, target)))
            if policy == "cyclic" and h == BEST[dataset] and dataset != "FDA_HTT":
                ordinal.append(dict(**observed, policy="fixed", **ordinal_metrics(y, sample, common, target)))
        err = np.empty(draws)
        common = np.empty(draws, dtype=int)
        ordinal_err = np.empty(draws)
        for h, take, _, sample, shared in sequential_count(n, draws, quota, target, quota, flag, rand):
            achieved = flag[shared].sum(axis=1)
            if policy == "independent_suffix":
                sample, achieved = split_suffix(order[take], priorities[take], quota, h, flag)
            err[take] = score_error(y, sample, truth)
            common[take] = achieved
            if policy == "cyclic" and dataset != "FDA_HTT":
                ordinal_err[take] = ordinal_error(y, sample)
        seq.append(dict(**observed, suffix=policy, **metrics(err, common, tol, target)))
        if policy == "cyclic" and dataset != "FDA_HTT":
            ordinal.append(dict(**observed, policy="sequential", **metrics(ordinal_err, common, .10, target)))
    return fixed, seq, ordinal


def ordinal_error(y, sample):
    grade, eligible = np.moveaxis(y, -1, 0)
    cutoffs = np.arange(1, 4)
    cumulative = (grade[..., None] >= cutoffs) & eligible[..., None].astype(bool)
    truth = cumulative.sum(axis=(0, 1)) / eligible.sum()
    denominator = eligible[sample, READERS].sum(axis=(1, 2))
    numerators = cumulative[sample, READERS].sum(axis=(1, 2))
    estimates = np.divide(numerators, denominator[:, None], out=np.full(numerators.shape, np.nan),
                          where=denominator[:, None] > 0)
    return np.max(np.abs(estimates - truth), axis=1)


def ordinal_metrics(y, sample, common, target):
    return metrics(ordinal_error(y, sample), common, .10, target)


def calibration(fixed, seq):
    rows, summary, sensitivity = [], [], []
    for (dataset, suffix), z in fixed.groupby(["dataset", "suffix"]):
        wins = z.pivot(index="group", columns="reserve", values="joint_pass").dropna(axis=1)
        counts = wins.to_numpy(dtype=int)
        reserves = wins.columns.to_numpy()
        totals = counts.sum(axis=0)
        held = (totals[None, :] - counts).argmax(axis=1)
        oracle = totals.argmax()
        assert z.draws.nunique() == 1
        draws = int(z.draws.iloc[0])
        f = counts / draws
        subset = seq.query("dataset==@dataset and suffix==@suffix").set_index("group").reindex(wins.index)
        s = (subset.joint_pass / subset.draws).to_numpy()
        assert np.isfinite(s).all()
        index = np.arange(len(s))
        h = f[index, held]
        for group, reserve, fixed_h, fixed_oracle, sequence in zip(wins.index, reserves[held], h, f[:, oracle], s, strict=True):
            rows.append(dict(dataset=dataset, group=group, suffix=suffix, heldout_reserve=reserve, heldout_fixed=fixed_h,
                             oracle_reserve=reserves[oracle], oracle_fixed=fixed_oracle, sequential=sequence,
                             heldout_gain=sequence-fixed_h, oracle_gain=sequence-fixed_oracle))

        rng = np.random.default_rng(261136)
        weights = rng.multinomial(len(s), np.full(len(s), 1/len(s)), size=10000)
        reweighted_totals = weights @ counts
        new_oracle = reweighted_totals.argmax(axis=1)
        boot_oracle = (weights @ s - reweighted_totals[np.arange(len(weights)), new_oracle] / draws) / len(s)
        boot_held = np.zeros(len(weights))
        for j in index:
            selected = (reweighted_totals - weights[:, j, None] * counts[j]).argmax(axis=1)
            boot_held += weights[:, j] * (s[j] - f[j, selected])
        boot_held /= len(s)
        for benchmark, base, nested in (("heldout", h, boot_held), ("oracle", f[:, oracle], boot_oracle)):
            delta = s - base
            cond = weights @ delta / len(s)
            lo, hi = np.quantile(cond, [.025, .975])
            lo_refit, hi_refit = np.quantile(nested, [.025, .975])
            summary.append(dict(dataset=dataset, suffix=suffix, benchmark=benchmark, frames=len(s),
                                fixed=base.mean(), sequential=s.mean(), gain=delta.mean(), low=lo, high=hi,
                                low_refit=lo_refit, high_refit=hi_refit))
            for i, (m, r) in enumerate(zip(cond, nested, strict=True)):
                sensitivity.append(dict(dataset=dataset, suffix=suffix, benchmark=benchmark, draw=i, conditional=m, refit=r))
    return pd.DataFrame(rows), pd.DataFrame(summary), pd.DataFrame(sensitivity)


def half_frame_splits(fixed, seq):
    rng = np.random.default_rng(261131)
    rows = []
    for dataset, z in fixed[fixed.suffix.eq("cyclic")].groupby("dataset"):
        wins = z.pivot(index="group", columns="reserve", values="joint_pass").dropna(axis=1)
        counts = wins.to_numpy(dtype=int)
        draws = int(z.draws.iloc[0])
        sequence = seq.query("dataset==@dataset and suffix=='cyclic'").set_index("group").reindex(wins.index)
        s = (sequence.joint_pass / sequence.draws).to_numpy()
        for iteration in range(500):
            order = rng.permutation(len(wins))
            train, test = order[:len(order)//2], order[len(order)//2:]
            best = counts[train].sum(axis=0).argmax()
            f = counts[test, best].mean() / draws
            seq_rate = s[test].mean()
            rows.append(dict(dataset=dataset, iteration=iteration, training_frames=len(train), test_frames=len(test),
                             reserve=wins.columns[best], fixed_rate=f, sequential_rate=seq_rate, difference=seq_rate-f))
    return pd.DataFrame(rows)


def summarize_exclusions(frames, pairs):
    records = []
    for dataset, f in frames.groupby("dataset"):
        q = pairs[pairs.dataset.eq(dataset)].pivot(
            index=["group", "focal", "peer"], columns="state", values="mean"
        ).dropna()
        effects = (q.focal_excluded - q.focal_eligible).groupby("group").mean()
        mixed = (f.mixed_region_mean-f.full_region_mean).dropna()
        rate = f.excluded / f.records
        rec = dict(dataset=dataset, regions=f.regions.sum(), excluded=f.excluded.sum(), records=f.records.sum(),
                   min_frame_excluded=rate.min(), median_frame_excluded=rate.median(), max_frame_excluded=rate.max(),
                   top_five_frame_share=f.excluded.nlargest(5).sum()/f.excluded.sum() if f.excluded.sum() else 0,
                   peer_pairs=len(q), peer_frames=len(effects), peer_difference=effects.mean(), peer_median=effects.median(),
                   peer_q25=effects.quantile(.25), peer_q75=effects.quantile(.75),
                   mixed_frames=len(mixed), mixed_difference=mixed.mean(), mixed_median=mixed.median())
        for k in range(4):
            rec[f"regions_with_{k}_eligible"] = f[f"regions_with_{k}_eligible"].sum()
        records.append(rec)
    return pd.DataFrame(records)


def edge_cases():
    y = np.ones((1, 7), dtype=int)
    order = np.arange(7)[None, :]
    priorities = np.tile(order[:, None, :], (1, 3, 1))
    readers = np.arange(3)[None, :]
    wrap, shared = fixed_count(7, 1, 6, 2, (order, readers))
    assert wrap.tolist() == [[[0,1,2,3,4,5], [0,1,6,2,3,4], [0,1,5,6,2,3]]]
    assert shared.tolist() == [[0,1,2,3]]
    for flag, status in ((np.array([True, False, True, False, False, False, False]), True),
                         (np.array([False, False, True, False, False, False, False]), False)):
        [(h, _, reached, sample, _)] = list(sequential_count(7, 1, 3, 2, 3, flag, (order, readers)))
        other, actual = split_suffix(order, priorities, 3, h, flag)
        assert h == 3 and bool(reached[0]) == status and (actual[0] >= 2) == status
        np.testing.assert_array_equal(sample, other)
    priorities = np.array([[[6,5,4,3,2,1,0],[1,0,2,3,4,5,6],[2,0,1,6,5,4,3]]])
    sample, actual = split_suffix(order, priorities, 4, 2, y[0].astype(bool))
    assert sample.tolist() == [[[0,1,6,5], [0,1,2,3], [0,1,2,6]]] and actual.tolist() == [2]
    print("Passed cyclic wrap, empty suffix, last-read success, below-target cap and independent suffix tests.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kidney", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    edge_cases()
    a.output.mkdir(parents=True, exist_ok=True)
    draws = 64 if a.smoke else 4000
    fda, fda_readers, example = fda_audit(a.data)
    frame_data, reader_data, pairs_data, all_fixed, all_seq, all_ordinal = [], [], [], [], [], []
    for i, (ds, frame, y, quota, _, tol, target) in enumerate(panels(a.kidney, a.data)):
        if a.smoke and ds == "KidneyAI" and i >= 2:
            continue
        ids = example[frame] if ds == "FDA_HTT" else (
            ["expert1", "expert2", "expert3"] if ds == "KidneyAI" else ["marker1", "marker2", "marker3"]
        )
        e_frame, e_readers, e_pairs = exclusion(ds, frame, y, ids)
        frame_data.append(e_frame)
        reader_data.extend(e_readers)
        pairs_data.extend(e_pairs)
        for output, part in zip((all_fixed, all_seq, all_ordinal), replay(ds, frame, y, quota, tol, target, draws, i), strict=True):
            output.extend(part)
    frames, readers, pairs, fixed, seq, ordinal = map(pd.DataFrame, (frame_data, reader_data, pairs_data, all_fixed, all_seq, all_ordinal))
    held, summary, bootstrap = calibration(fixed, seq)
    partitions = half_frame_splits(fixed, seq)
    h = partitions.groupby("dataset").difference
    half = h.agg(mean="mean", median="median", low=lambda v: v.quantile(.025), high=lambda v: v.quantile(.975))
    ordinal_summary = ordinal.groupby(["dataset", "policy"])[["draws", "defined", "sum_abs", "score_pass", "joint_pass"]].sum()
    ordinal_summary["mean_max_error"] = ordinal_summary.sum_abs / ordinal_summary.defined
    ordinal_summary["joint_rate"] = ordinal_summary.joint_pass / ordinal_summary.draws
    for name, obj in (("fda_panel_screening", fda), ("fda_complete_reader_exclusions", fda_readers), ("exclusion_by_frame", frames),
                      ("exclusion_by_reader_frame", readers), ("peer_score_strata", pairs),
                      ("exclusion_summary", summarize_exclusions(frames, pairs)), ("suffix_fixed_grid", fixed),
                      ("suffix_sequential", seq), ("calibration_by_frame", held), ("calibration_summary", summary),
                      ("calibration_reweighting", bootstrap), ("half_frame_splits", partitions),
                      ("half_frame_distribution", half.reset_index()),
                      ("ordinal_by_frame", ordinal), ("ordinal_summary", ordinal_summary.reset_index())):
        obj.to_csv(a.output / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
