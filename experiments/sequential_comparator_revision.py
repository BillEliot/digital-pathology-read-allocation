"""Recalibrate fixed reserves and cross the pilot and independent estimators."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pilot_allocation import panels, pilot_choices, remaining_patterns, shared_slots
from review_supplement import ORDERS
from sequential_revision_audit import estimates, fixed_count, metrics, ratio


def threshold_rows(dataset, group, y, n, tol, target, order, flag):
    draws = len(order[0])
    truth = ratio(y.sum(axis=(0,1)))
    for reserve in range(n+1):
        sample, shared = fixed_count(len(y),draws,n,reserve,order)
        error = estimates(y,0,sample,"pooled")[0] - truth
        common = flag[shared].sum(axis=1)
        for req in (target-1,target,target+1):
            for epsilon in (tol/2,tol,tol*2):
                joint = ((np.abs(error)<=epsilon+1e-12) & (common>=req)).sum()
                yield dict(dataset=dataset,group=group,tolerance=epsilon,target=req,reserve=reserve,
                           draws=draws,joint_pass=int(joint))


def complete_extra(dataset, group, y, n, x, tol, target, permutation, ranks, flag):
    truth = ratio(y.sum(axis=(0,1)))
    first = permutation[:,0]
    draws = len(first)
    sampled = permutation[:,:,:n]
    masks = np.zeros((draws,3,len(y)),dtype=bool)
    masks[np.arange(draws)[:,None,None],np.arange(3)[None,:,None],sampled] = True
    common = (masks.all(axis=1) & flag[None,:]).sum(axis=1)
    ordinary, expanded = estimates(y,0,sampled,"pooled")
    np.testing.assert_allclose(ordinary,expanded,rtol=1e-14,atol=1e-14,equal_nan=True)
    for name,error in (("pooled",ordinary-truth),("phase",expanded-truth)):
        yield dict(dataset=dataset,group=group,allocation="independent",estimator=name,
                   shrink=4,beta_prior=.5,**metrics(error,common,tol,target))
    patterns = remaining_patterns(len(y)-x,n-x)
    pilot = y[first[:,:x]]
    prior_seen = flag[first[:,:x]]
    for shrink in (0,4,12):
        for alpha in (.5,1.):
            picked,_,_ = pilot_choices(pilot,patterns,len(y)-x,n-x,tol,target,dataset=="CrowdGleason",
                                        prior_seen,shrink=shrink,beta_prior=alpha)
            error = np.empty((2,draws))
            common = np.empty(draws,dtype=int)
            for h in np.unique(picked):
                keep = picked == h
                selected = first[keep]
                slots = patterns[h]
                suffix = selected[np.arange(keep.sum())[:,None,None],x+slots[ranks[keep]]]
                prefix = np.broadcast_to(selected[:,None,:x],(keep.sum(),3,x))
                sample = np.concatenate((prefix,suffix),axis=-1)
                error[:,keep] = np.array(estimates(y,x,sample,"pooled")) - truth
                common[keep] = prior_seen[keep].sum(axis=1) + flag[selected[:,x+shared_slots(slots)]].sum(axis=1)
            for est,z in zip(("pooled","phase"),error,strict=True):
                yield dict(dataset=dataset,group=group,allocation="pilot",estimator=est,
                           shrink=shrink,beta_prior=alpha,**metrics(z,common,tol,target))


def threshold_summary(fixed, sequential):
    rows=[]
    for key,g in fixed.groupby(["dataset","tolerance","target"]):
        name,tol,target=key
        wins=g.pivot(index="group",columns="reserve",values="joint_pass").dropna(axis=1)
        if g.draws.nunique()!=1:
            raise ValueError(f"Unequal draws in {key}")
        w=wins/g.draws.iloc[0]
        z=sequential.query("dataset==@name and tolerance==@tol and target==@target and analysis=='grid' and allocation=='full_cap' and estimator=='pooled'")
        seq=(z.set_index("group").joint_pass/z.set_index("group").draws).reindex(w.index)
        original={"KidneyAI":15,"FDA_HTT":5,"CrowdGleason":2}[name]
        selected=wins.sum().idxmax()
        which=(wins.sum().to_numpy()[None,:]-wins.to_numpy()).argmax(axis=1)
        heldout=w.to_numpy()[np.arange(len(w)),which]
        if not np.isfinite(seq).all():
            raise ValueError(key)
        rows.append(dict(dataset=name,tolerance=tol,target=target,frames=len(w),
                         original_reserve=original,calibrated_reserve=selected,
                         frozen=w[original].mean(),reselected=w[selected].mean(),
                         held_out=heldout.mean(),seq_full_cap=seq.mean()))
    return pd.DataFrame(rows)


def reader_sensitivity(panels):
    z=panels[panels.dataset.eq("FDA_HTT")].sort_values(["group","panel"])
    combos=z.panel.str.split("/").tolist()
    reader={r:i for i,r in enumerate(sorted({r for panel in combos for r in panel}))}
    ids=np.array([[reader[r] for r in panel] for panel in combos])
    slides,begin=np.unique(z.group,return_index=True)
    outcome=(z.seq_pooled_joint-z.fixed_joint).to_numpy()
    rng=np.random.default_rng(261134)
    output=[]
    for _ in range(40):
        reader_weight=rng.exponential(size=(250,len(reader)))
        triple=reader_weight[:,ids].prod(axis=-1)
        within=np.add.reduceat(triple*outcome,begin,axis=1)/np.add.reduceat(triple,begin,axis=1)
        slide=rng.exponential(size=(250,len(slides)))
        rows=np.stack((within.mean(axis=1),(within*slide).sum(axis=1)/slide.sum(axis=1)),axis=1)
        output.extend(rows)
    return pd.DataFrame(output,columns=["reader_weighted","reader_and_frame_weighted"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kidney",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    draws=64 if a.smoke else 4000
    reserves,extra=[],[]
    for i,(dataset,group,y,n,x,tol,target) in enumerate(panels(a.kidney,a.data)):
        if a.smoke and i>=2: break
        rng=np.random.default_rng(np.random.SeedSequence([261121,i]))
        perm=rng.random((draws,3,len(y))).argsort(axis=2)
        ranks=ORDERS[rng.integers(len(ORDERS),size=draws)]
        flag=y[...,1].all(axis=1)
        reserves.extend(threshold_rows(dataset,group,y,n,tol,target,(perm[:,0],ranks),flag))
        extra.extend(complete_extra(dataset,group,y,n,x,tol,target,perm,ranks,flag))
    fixed,additional=pd.DataFrame(reserves),pd.DataFrame(extra)
    fixed.to_csv(a.output/"fixed_threshold_curves.csv",index=False)
    additional.to_csv(a.output/"pilot_and_independent.csv",index=False)
    if a.smoke: return
    previous=pd.read_csv(a.audit/"estimators.csv").assign(shrink=4,beta_prior=.5)
    main=additional[additional.shrink.eq(4)&additional.beta_prior.eq(.5)]
    pd.concat((previous,main),ignore_index=True).to_csv(a.output/"complete_factorial.csv",index=False)
    threshold_summary(fixed,pd.read_csv(a.audit/"sensitivity.csv")).to_csv(a.output/"recalibrated_grid.csv",index=False)
    reader_sensitivity(pd.read_csv(a.audit/"reader_panels.csv")).to_csv(a.output/"reader_frame_weights.csv",index=False)


if __name__=="__main__":
    main()
