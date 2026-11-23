"""Audit FDA HTT allocations with both zero and positive remaining quotas.

At a nondivisible budget B = p*h + k, the mixed branch can occur only after
h common regions. The first k readers then receive one suffix read each.
Construct this branch from the pinned labels and the archived randomization.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from fda_pilot_replay import Frame, draws_for, frames_from_zip
from fda_pilot_evaluate import COHORTS, DESIGN, cube, heldout_curves, meets


MIN_BUDGET = 8
N_REGIONS = 10
DRAWS = 4096
KEY = ["panel", "slide", "reference", *DESIGN, "quorum"]
BRANCH = ["mixed", "defined", "mixed_score", "mixed_material", "mixed_joint",
          "err", "abs_err", "sq_err", "panel_defined", "panel_err", "panel_abs_err", "panel_sq_err"]


def pooled(y, count):
    return np.divide(y, count, out=np.full_like(y, np.nan, dtype=float), where=count > 0)


def error_totals(estimate, reference, mask, material):
    valid = mask & np.isfinite(estimate) & np.isfinite(reference)
    error = (estimate-reference)[valid]
    passed = np.zeros(len(mask), dtype=bool)
    passed[valid] = np.abs(error) <= 5 + 1e-12
    return (int(valid.sum()), int(passed.sum()), int((mask & material).sum()), int((passed & material).sum()),
            float(error.sum()), float(np.abs(error).sum()), float(error @ error))


def replay_mixed(frame, roster, order, refs, names, panel):
    """Direct calculation on the only prefixes that can create a mixed branch."""
    rows = []
    sizes = range(2,min(7,roster.shape[1])+1) if panel == "variable" else (3,)
    for p in sizes:
        values = frame.scores[roster[:,:p,None], order[:,None,:]]
        eligible = values >= 0
        scores = np.where(eligible,values,0)
        work_mean = pooled(scores.sum(axis=(1,2)),eligible.sum(axis=(1,2)))
        prefix_y = scores.sum(axis=1).cumsum(axis=1)
        prefix_n = eligible.sum(axis=1).cumsum(axis=1)
        region_eligible = eligible.sum(axis=1)
        common_material = {q:(region_eligible>=q).cumsum(axis=1) for q in range(2,min(3,p)+1)}
        all_draws = np.ones(len(order),dtype=bool)
        for h in range(1,N_REGIONS):
            for k in range(1,p):
                budget = p*h+k
                if budget < MIN_BUDGET:
                    continue
                pos = h+np.arange(k)%(N_REGIONS-h)
                suffix = scores[:,np.arange(k),pos]
                suffix_eligible = eligible[:,np.arange(k),pos]
                estimate = pooled(prefix_y[:,h-1]+suffix.sum(axis=1),
                                  prefix_n[:,h-1]+suffix_eligible.sum(axis=1))
                # At most one follow-up read per working reader. Wrapping can
                # put some of these reads on the same suffix region.
                unique_pos = np.unique(pos)
                shares = np.stack([suffix_eligible[:,pos==v].sum(axis=1) for v in unique_pos],axis=1)
                for q,common in common_material.items():
                    material = common[:,h-1]+(shares>=q).sum(axis=1)>=2
                    reached_before_cap = common[:,h-2]>=2 if h>1 else ~all_draws
                    for policy,mask in (("fixed",all_draws),("sequential",~reached_before_cap)):
                        if not mask.any():
                            continue
                        panel_values = error_totals(estimate,work_mean,mask,material)
                        for i,ref in enumerate(names):
                            measures = error_totals(estimate,refs[:,i],mask,material)
                            rows.append((panel,frame.slide,ref,p,budget,policy,h,q,int(mask.sum()),
                                         *measures,panel_values[0],*panel_values[4:]))
    return rows


def reproduce(inputs,var_path,fixed_path):
    fs = frames_from_zip(inputs)
    rows = []
    for f in fs:
        modes = (False,True) if f.size>=10 else (False,)
        for separated in modes:
            roster,order,refs = draws_for(f,DRAWS,261170,reserve_reference=separated)
            names = ("all_reference_paired","independent3") if separated else ("all_reference",)
            rows.extend(replay_mixed(f,roster,order,refs,names,"variable"))
    common_ids = sorted(set.intersection(*(set(f.readers) for f in fs)))
    assert len(common_ids)==5
    for j,team in enumerate(combinations(common_ids,3),1):
        for f in fs:
            frame = Frame(f.slide,f.batch,team,f.scores[[f.readers.index(r) for r in team]])
            roster,order,refs = draws_for(frame,DRAWS,261180)
            rows.extend(replay_mixed(frame,roster,order,refs,("team_reference",),f"team{j:02d}"))
        print(f"Checked fixed team {j:02d}",flush=True)
    measured = pd.DataFrame(rows,columns=KEY+BRANCH)
    old = pd.concat([pd.read_csv(var_path).assign(panel="variable"),pd.read_csv(fixed_path)],ignore_index=True)
    assert old.draws.eq(DRAWS).all() and not old.duplicated(KEY).any()
    df = old.merge(measured,on=KEY,how="left",validate="1:1")
    assert df.mixed.notna().sum()==len(measured)
    df[BRANCH] = df[BRANCH].fillna(0)
    full = df.policy.eq("fixed") & df.budget.mod(df.readers).ne(0) & df.common.eq(df.budget//df.readers)
    assert df.loc[full,"mixed"].eq(df.loc[full,"draws"]).all()
    assert (df.loc[full,["mixed_score","mixed_material","mixed_joint"]].to_numpy()
            ==df.loc[full,["score","material","joint"]].to_numpy()).all()
    cap = df.loc[df.policy.eq("fixed") & df.common.eq(df.budget//df.readers), KEY+["mixed_joint","mixed_score","mixed_material"]]
    seq = df.loc[df.policy.eq("sequential") & df.mixed.gt(0),KEY+["mixed_joint","mixed_score","mixed_material"]].assign(policy="fixed")
    sub = seq.merge(cap,on=KEY,validate="1:1",suffixes=("_seq","_fixed"))
    assert len(sub)==len(seq)
    for x in ("mixed_joint","mixed_score","mixed_material"):
        assert (sub[x+"_seq"]<=sub[x+"_fixed"]).all()
    for src,total in (("score","score"),("material","material"),("joint","joint")):
        assert (df["mixed_"+src]<=df[total]).all()
    assert (df.mixed_joint<=df.mixed).all() and (df.defined<=df.mixed).all()
    assert not df.loc[df.policy.eq("sequential_spare") | df.budget.mod(df.readers).eq(0),"mixed"].any()
    return df, pd.DataFrame([{"slide":f.slide,"batch":f.batch,"complete_readers":f.size,"full_pilot_reads":10*f.size} for f in fs])


def summarize(weights):
    v = {k:float(x) for k,x in weights.items()}
    return {
        "frequency": v["mixed"]/v["draws"],
        "undefined": 1-v["defined"]/v["mixed"] if v["mixed"] else np.nan,
        "bias": v["err"]/v["defined"] if v["defined"] else np.nan,
        "mae": v["abs_err"]/v["defined"] if v["defined"] else np.nan,
        "rmse": np.sqrt(v["sq_err"]/v["defined"]) if v["defined"] else np.nan,
        "panel_bias": v["panel_err"]/v["panel_defined"] if v["panel_defined"] else np.nan,
        "panel_mae": v["panel_abs_err"]/v["panel_defined"] if v["panel_defined"] else np.nan,
        "panel_rmse": np.sqrt(v["panel_sq_err"]/v["panel_defined"]) if v["panel_defined"] else np.nan,
        "branch_joint": v["mixed_joint"]/v["mixed"] if v["mixed"] else np.nan,
        "branch_score": v["mixed_score"]/v["mixed"] if v["mixed"] else np.nan,
        "branch_material": v["mixed_material"]/v["mixed"] if v["mixed"] else np.nan,
        "joint": v["joint"]/v["draws"],
    }


def design_curves(df, frames):
    by_design = []
    for cohort, floor in COHORTS.items():
        slides = frames.loc[frames.complete_readers.ge(floor),"slide"].tolist()
        rmax = min(floor,7)
        pool = df[df.panel.eq("variable") & df.slide.isin(slides)]
        refs = ("all_reference_paired","independent3") if cohort=="independent8" else ("all_reference",)
        for q in (2,3):
            cfg,cal = cube(pool,slides,rmax,q,refs[0])
            for reference in refs:
                c,values = cube(pool,slides,rmax,q,reference)
                assert c==cfg
                grouped = {k:f.set_index("slide").reindex(slides) for k,f in pool[pool.reference.eq(reference) & pool.quorum.eq(q)].groupby(DESIGN)}
                for (r,b,kind), _rates, selection in heldout_curves(cfg,values,cal):
                    cap=b//r
                    if selection is not None:
                        columns = ["draws","joint",*BRANCH]
                        arr = np.stack([grouped[r,b,"fixed",h][columns].to_numpy() for h in range(cap+1)],axis=1)
                        sample = arr[np.arange(len(slides)),selection]
                        old = dict(zip(columns,sample.sum(axis=0)))
                        # Use the same held-out criterion under the constraint
                        # that no mixed branch can happen in any assignment.
                        top=cap-1 if b%r else cap
                        cfg_indices = [cfg.index((r,b,"fixed",h)) for h in range(top+1)]
                        val = cal[:,cfg_indices,2]
                        sel = np.argmax((val.sum(axis=0)-val).round(12),axis=1)
                        safer = arr[np.arange(len(slides)),sel]
                        safe_rate = safer[:,1].sum()/safer[:,0].sum()
                    else:
                        h = cap if kind=="sequential" else max(0,cap-1)
                        old = grouped[r,b,kind,h][["draws","joint",*BRANCH]].sum().to_dict()
                        if kind=="sequential" and b%r:
                            g=grouped[r,b,"sequential_spare",max(0,cap-1)]
                            safe_rate=g.joint.sum()/g.draws.sum()
                        else:
                            safe_rate=old["joint"]/old["draws"]
                    assert np.isclose(old["joint"]/old["draws"],_rates[:,2].mean(),atol=1e-12,rtol=0)
                    by_design.append(dict(cohort=cohort,reference=reference,quorum=q,readers=r,budget=b,
                                          policy=kind,mixed=int(old["mixed"]),draws=int(old["draws"]),
                                          safe_joint=safe_rate,**summarize(old)))
    return pd.DataFrame(by_design)


def trial_audit(trials, sub, frames, panel, quorum, mode, reference, top):
    """Join archived split masks and reselect using only designs that cannot enter the mixed branch."""
    ids = frames.slide.str.removeprefix("HTT-TILS-").str.removesuffix(".ndpi")
    x_to_i = {v:i for i,v in enumerate(ids)}
    cfg,a = cube(sub,frames.slide.tolist(),top,quorum,reference)
    look={c:i for i,c in enumerate(cfg)}
    cols = pd.MultiIndex.from_tuples(cfg,names=DESIGN)
    metrics = ["draws","joint",*BRANCH]
    sub=sub[sub.quorum.eq(quorum) & sub.reference.eq(reference) & sub.slide.isin(frames.slide) & sub.readers.between(quorum,top)]
    stats = np.stack([sub.pivot(index="slide",columns=DESIGN,values=k).reindex(index=frames.slide,columns=cols).to_numpy() for k in metrics],axis=-1)
    safe = np.array([b%r==0 or p=="sequential_spare" or (p=="fixed" and h<b//r) for r,b,p,h in cfg])
    batch = pd.factorize(frames.batch,sort=True)[0]
    freq = np.bincount(batch)/len(batch)
    parts = []
    for (method,X,target),group in trials.groupby(["method","X","target"],sort=True):
        active = np.zeros((len(group),len(frames)),dtype=bool)
        test = np.zeros_like(active)
        for i,(pilot,val) in enumerate(zip(group.pilot_slides,group.validation_slides,strict=True)):
            active[i,[x_to_i[x] for x in pilot.split("|")]] = True
            if pd.isna(val):
                test[i]=~active[i]
            else:
                test[i,[x_to_i[x] for x in val.split("|")]]=True
        assert (active.sum(axis=1)==X).all() and not (test & active).any()
        train = active.astype(float)/X
        if method=="batch_stratified":
            n = active @ np.eye(len(freq))[batch]
            assert (n>0).all()
            train = active * freq[batch]/n[:,batch]
        validate = test/test.sum(axis=1,keepdims=True)
        assert np.allclose(train.sum(axis=1),1)
        train_rates=(train@a[:,:,2]).round(12)
        budget=np.array([x[1] for x in cfg])
        def decisions(mask):
            eligible=(train_rates>=target)&mask
            cheapest=np.min(np.where(eligible,budget,np.inf),axis=1,keepdims=True)
            selected_mask=np.where(eligible.any(axis=1,keepdims=True),eligible&(budget==cheapest),mask)
            return np.argmax(np.where(selected_mask,train_rates,-np.inf),axis=1)
        selected = decisions(np.ones(len(cfg),dtype=bool))
        archived=np.array([look[c] for c in group[DESIGN].itertuples(index=False,name=None)])
        assert np.array_equal(selected,archived),(mode,panel,quorum,method,X,target)
        old_joint = np.einsum("ij,ji->i",validate,a[:,selected,2])
        assert np.allclose(old_joint,group.joint,atol=1e-12,rtol=0)
        constrained = decisions(safe)
        safe_joint = np.einsum("ij,ji->i",validate,a[:,constrained,2])
        old_stats = np.einsum("ij,jik->ik",validate,stats[:,selected])
        assert np.all(stats[:,constrained,metrics.index("mixed")]==0)
        summary = summarize(dict(zip(metrics,old_stats.sum(axis=0))))
        parts.append(dict(mode=mode,panel=panel,quorum=quorum,method=method,X=X,target=target,
                          changed=(selected!=constrained).mean(),
                          old_budget=budget[selected].mean(),safe_budget=budget[constrained].mean(),
                          old_gate90=meets(old_joint,.9).mean(),safe_gate90=meets(safe_joint,.9).mean(),
                          old_gate95=meets(old_joint,.95).mean(),safe_gate95=meets(safe_joint,.95).mean(),
                          safe_joint=safe_joint.mean(),**summary))
    return parts


def trials(df,frames,review):
    out=[]
    common=frames[frames.complete_readers.ge(7)].reset_index(drop=True)
    for mode,file in (("original37","reviewer_benchmark_splits.csv.gz"),("common37","common_validation_splits.csv.gz")):
        data=pd.read_csv(review/file)
        if "validation_slides" not in data:
            data["validation_slides"]=np.nan
        out.extend(trial_audit(data,df[df.panel.eq("variable")],common,"variable",2,mode,"all_reference",7))
    fixed=pd.read_csv(review/"fixed_panel_splits.csv.gz")
    for (panel,q),sub in fixed.groupby(["panel","quorum"]):
        out.extend(trial_audit(sub,df[df.panel.eq(panel)],frames,panel,q,"fixed64","team_reference",3))
    return pd.DataFrame(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for f in ("input","planning","review","output"):
        p.add_argument("--"+f,type=Path,required=True)
    args=p.parse_args()
    df,frames=reproduce(args.input,args.planning/"replay_counts.csv.gz",args.review/"fixed_panel_counts.csv.gz")
    args.output.mkdir(parents=True,exist_ok=True)
    df.to_csv(args.output/"branch_counts.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    curves=design_curves(df,frames)
    curves.to_csv(args.output/"branch_by_design.csv",index=False)
    thresholds=[]
    for key,g in curves.groupby(["cohort","reference","quorum","readers","policy"]):
        for minimum in (.85,.90,.95):
            base=g.loc[meets(g.joint,minimum),"budget"]
            safe=g.loc[meets(g.safe_joint,minimum),"budget"]
            thresholds.append((*key,minimum,base.min(),safe.min()))
    pd.DataFrame(thresholds,columns=["cohort","reference","quorum","readers","policy","target","budget","safe_budget"]).to_csv(
        args.output/"budget_sensitivity.csv",index=False)
    trials(df,frames,args.review).to_csv(args.output/"pilot_sensitivity.csv",index=False)
    print(f"Mixed branch checked against archived endpoints in {len(df):,} slide-design rows.",flush=True)


if __name__=="__main__":
    main()

