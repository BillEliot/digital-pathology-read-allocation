"""Evaluate alternative score targets and audit allocation behavior on three releases."""

import argparse
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd

from pilot_allocation import panels
from sequential_revision_audit import BEST, READERS, draws_for, estimates, fixed_count, metrics, ratio, sequential_count


def region_mean(values):
    scores = ratio(values)
    count = np.isfinite(scores).sum(axis=-1)
    return np.divide(np.nansum(scores, axis=-1), count, out=np.full(count.shape, np.nan), where=count > 0)


def region_plugin(y, sample):
    """Pool observed records within each region and average defined region scores."""
    n_draws = len(sample)
    counts = np.zeros((n_draws, len(y), 2))
    np.add.at(counts, (np.arange(n_draws)[:, None, None], sample), y[sample, READERS])
    return region_mean(counts)


def audit_frame(dataset, group, y, n, tol, target, random):
    draws = len(random[0])
    flag = y[..., 1].all(axis=1)
    full = y.sum(axis=0)
    census = dict(dataset=dataset, group=group, pooled=ratio(full.sum(axis=0)),
                  equal_reader=np.mean(ratio(full)), equal_region=region_mean(y.sum(axis=1)))
    scores = [("pooled", y, tol), ("region_plugin", y, tol)]
    if dataset != "FDA_HTT":
        for cutoff in (1, 2, 3):
            binary = y.copy()
            binary[..., 0] = (y[..., 0] >= cutoff) & (y[..., 1] > 0)
            scores.append((f"grade_ge_{cutoff}", binary, .10))

    h = BEST[dataset]
    sample, shared = fixed_count(len(y), draws, n, h, random)
    designs = [("fixed", [(h,np.ones(draws,dtype=bool),sample,shared)])]
    for label, cap in (("reserve_followup", n-1), ("full_cap", n)):
        assignments = sequential_count(len(y), draws, n, target, cap, flag, random)
        designs.append((label, [(k,take,s,sh) for k,take,_,s,sh in assignments]))

    rows, behavior, passes = [], [], {}
    for design, paths in designs:
        common = np.empty(draws,dtype=int)
        for k,take,sample,shared in paths:
            common[take] = j = flag[shared].sum(axis=1)
            for j_count, number in zip(*np.unique(j,return_counts=True),strict=True):
                behavior.append(dict(dataset=dataset, group=group, allocation=design, regions=len(y), quota=n,
                                     prefix=k, followup=n-k, shared=shared.shape[1],
                                     jointly_evaluable=j_count, draws=number))
        for name, val, epsilon in scores:
            errors = np.empty(draws)
            truth = census["equal_region"] if name == "region_plugin" else ratio(val.sum(axis=(0,1)))
            for _,take,sample,_ in paths:
                score = region_plugin(y,sample) if name == "region_plugin" else estimates(val,0,sample,"pooled")[0]
                errors[take] = score - truth
            rows.append(dict(dataset=dataset, group=group, allocation=design, target=name, tolerance=epsilon,
                             **metrics(errors,common,epsilon,target)))
            if name == "pooled":
                passes[design] = (np.abs(errors) <= epsilon+1e-12) & (common >= target)
    difference = passes["full_cap"].astype(int) - passes["fixed"]
    paired = dict(dataset=dataset, group=group, n=draws, wins=int((difference==1).sum()),
                  losses=int((difference==-1).sum()), mcse=float(difference.std(ddof=1)/np.sqrt(draws)))
    return census, rows, behavior, paired


def partitions(grid, estimators):
    rng = np.random.default_rng(261131)
    output = []
    for dataset, part in grid.groupby("dataset"):
        scores = part.assign(rate=part.joint_pass/part.draws).pivot(index="group",columns="count",values="rate").dropna(axis=1)
        seq = estimators.query("dataset==@dataset and allocation=='full_cap' and estimator=='pooled'").set_index("group")
        seq = (seq.joint_pass/seq.draws).reindex(scores.index).to_numpy()
        assert np.isfinite(seq).all()
        a = scores.to_numpy()
        groups = len(a)
        for k in range(500):
            order = rng.permutation(groups)
            train, test = order[:groups//2], order[groups//2:]
            idx = a[train].mean(axis=0).argmax()
            fixed_rate = a[test,idx].mean()
            seq_rate = seq[test].mean()
            output.append(dict(dataset=dataset, iteration=k, training_frames=len(train), test_frames=len(test),
                               reserve=scores.columns[idx], fixed_rate=fixed_rate, sequential_rate=seq_rate,
                               difference=seq_rate-fixed_rate))
    return pd.DataFrame(output)


def omit_readers(panels):
    rows = []
    panels = panels[panels.dataset.isin(["FDA_HTT","CrowdGleason"])]
    for dataset, data in panels.groupby("dataset"):
        ids = data.panel.str.split("/")
        readers = sorted({r for triple in ids for r in triple})
        for reader in readers:
            keep = data.loc[~ids.map(lambda triple: reader in triple)]
            stats = keep.groupby("group")[["fixed_joint","seq_pooled_joint","seq_phase_joint"]].mean()
            assert len(stats) == data.group.nunique()
            avg = stats.mean()
            rows.append(dict(dataset=dataset, omitted_reader=reader, frames=len(stats), panels=len(keep),
                             pooled_delta=avg.seq_pooled_joint-avg.fixed_joint,
                             phase_delta=avg.seq_phase_joint-avg.fixed_joint))
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kidney",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows, counts, census_rows, comparisons = [],[],[],[]
    for i,(name,group,y,n,_,tol,target) in enumerate(panels(args.kidney,args.data)):
        if args.smoke and i>=2: break
        draws=64 if args.smoke else 4000
        census, result, samples, paired = audit_frame(name,group,y,n,tol,target,draws_for(len(y),draws,261121,i))
        census_rows.append(census)
        rows.extend(result)
        counts.extend(samples)
        comparisons.append(paired)
    for name, values in (("estimand_by_frame",rows),("replay_behavior",counts),("full_frame_targets",census_rows),
                         ("paired_monte_carlo",comparisons)):
        pd.DataFrame(values).to_csv(args.output/f"{name}.csv",index=False)
    if args.smoke: return
    pair = pd.DataFrame(comparisons).groupby("dataset").agg(
        frames=("group","size"), n=("n","sum"), wins=("wins","sum"), losses=("losses","sum"),
        mcse_squared=("mcse",lambda x:np.square(x).sum()))
    pair["gain_pp"] = (100*(pair.wins-pair.losses))/pair.n
    pair["gain_display"] = [str((Decimal(int(w)-int(l))*100/Decimal(int(n))).quantize(
        Decimal(".01"),rounding=ROUND_HALF_UP)) for w,l,n in zip(pair.wins,pair.losses,pair.n,strict=True)]
    pair["mcse_pp"] = 100*np.sqrt(pair.mcse_squared)/pair.frames
    pair.to_csv(args.output/"paired_monte_carlo_summary.csv")
    partitions(pd.read_csv(args.audit/"fixed_grid.csv"),
               pd.read_csv(args.audit/"estimators.csv")).to_csv(args.output/"half_frame_holdouts.csv",index=False)
    omit_readers(pd.read_csv(args.audit/"reader_panels.csv")).to_csv(args.output/"leave_reader_out.csv",index=False)


if __name__ == "__main__":
    main()
