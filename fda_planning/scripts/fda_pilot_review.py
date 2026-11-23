"""Reviewer-requested benchmarks, common validation, and genuinely fixed FDA HTT panels."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from numba import set_num_threads

from fda_pilot_evaluate import DESIGN, ENDPOINTS, PILOT_SIZES, choose, cube, meets, strat_order
from fda_pilot_replay import Frame, assignment_outcomes, design_grid, draws_for, frames_from_zip


TARGETS = (.90, .95)
GATES = (.85, .90, .95)
PROJECTS = (37, 64, 100, 250, 500)
LOW_COST = (5, 20, "fixed", 0)
FULL_FIVE = (5, 50, "fixed", 10)
FIXED_BENCHMARKS = {"fixed15": (3, 15, "fixed", 5), "fixed18": (3, 18, "fixed", 6),
                    "full30": (3, 30, "fixed", 10)}


def save(frame, output, name, compact=False):
    if compact:
        frame.to_csv(output / (name + ".csv.gz"), index=False, compression={"method":"gzip", "mtime":0})
    else:
        frame.to_csv(output / (name + ".csv"), index=False)


def short_slide_ids(slides):
    return slides.str.removeprefix("HTT-TILS-").str.removesuffix(".ndpi").to_numpy()


def quota_for_validation(counts, n_test):
    q = np.ones(len(counts), dtype=int)
    for step in range(len(q)+1, n_test+1):
        under = np.where(q < counts-1, step*counts/counts.sum()-q, -np.inf)
        add = np.argmax(under)
        assert np.isfinite(under[add])
        q[add] += 1
    assert sum(q) == n_test and (q < counts).all()
    return q


def shared_splits(frames, n_test, reps, seed):
    groups = list(frames.groupby("batch", sort=True).indices.values())
    label = pd.factorize(frames.batch, sort=True)[0]
    sizes = np.bincount(label)
    n = len(frames)
    fractions = sizes / n
    val_quota = quota_for_validation(sizes, n_test)
    rng = np.random.default_rng(seed)
    ranks = rng.random((reps, n, 2))
    test = np.zeros((reps,n), dtype=bool)
    orders = {name:np.zeros((reps,max(PILOT_SIZES)), dtype=int) for name in ("random","batch_stratified")}
    for rep, (val_rank, train_rank) in enumerate(ranks.transpose(0,2,1)):
        for idx, k in zip(groups, val_quota, strict=True):
            test[rep, idx[np.argsort(val_rank[idx])[:k]]] = True
        pool = np.flatnonzero(~test[rep])
        sample_rank = train_rank[pool]
        subgroups = [np.flatnonzero(label[pool] == i) for i in range(len(groups))]
        orders["random"][rep] = pool[np.argsort(sample_rank)[:max(PILOT_SIZES)]]
        orders["batch_stratified"][rep] = pool[strat_order(sample_rank, subgroups)]
    return test, orders, label, fractions


def evaluate_splits(frames, splits, config, rates, benchmarks, mode, panel, quorum):
    test, orders, label, fractions = splits
    reps, n = test.shape
    n_test = int(test[0].sum())
    ids = short_slide_ids(frames.slide)
    validation_ids = ["|".join(ids[row]) for row in test]
    validate = test.astype(float) / n_test
    idx = {c:i for i,c in enumerate(config)}
    vals = {name:validate @ rates[:,idx[design],2] for name,design in benchmarks.items()}
    blocks = []
    for method, order in orders.items():
        for x in PILOT_SIZES:
            sample = order[:,:x]
            train = np.zeros((reps,n), dtype=bool)
            train[np.arange(reps)[:,None], sample] = True
            assert (train.sum(axis=1) == x).all() and not (train & test).any()
            weight = train.astype(float) / x
            if method == "batch_stratified":
                count = train.astype(int) @ np.eye(len(fractions))[label]
                assert (count > 0).all()
                weight = train * fractions[label] / count[:,label]
            assert np.allclose(weight.sum(axis=1), 1)
            pilot_ids = ["|".join(sorted(ids[row])) for row in sample]
            estimate = weight @ rates[:,:,2]
            charged = train @ frames.full_pilot_reads.to_numpy()
            for target in TARGETS:
                candidates = [choose(estimate[j],config,target) for j in range(reps)]
                selected = np.array([c for c, _ in candidates])
                quality = np.einsum("si,ism->sm", validate, rates[:,selected,:])
                chosen = pd.DataFrame([config[i] for i in selected], columns=DESIGN)
                chosen = chosen.assign(mode=mode, panel=panel, quorum=quorum, method=method, X=x, target=target,
                                       n_test=n_test, rep=np.arange(reps), pilot_slides=pilot_ids,
                                       validation_slides=validation_ids, pilot_cost=charged,
                                       train_joint=estimate[np.arange(reps), selected],
                                       pilot_met=[hit for _,hit in candidates])
                for i,e in enumerate(ENDPOINTS):
                    chosen[e] = quality[:,i]
                for name,value in vals.items():
                    chosen["test_"+name] = value
                blocks.append(chosen)
    return pd.concat(blocks, ignore_index=True)


def original_split_benchmark(parent, frames, config, rates):
    trial = pd.read_csv(parent / "pilot_splits.csv.gz").query(
        "cohort == 'common37' and quorum == 2 and scope == 'all_sizes' and target in [0.90,0.95]").copy()
    ids = short_slide_ids(frames.slide)
    idx = {s:i for i,s in enumerate(ids)}
    test = np.ones((len(trial),len(ids)), dtype=bool)
    for j, ids_used in enumerate(trial.pilot_slides):
        test[j,[idx[s] for s in ids_used.split("|")]] = False
    weight = test/test.sum(axis=1,keepdims=True)
    assert (test.sum(axis=1) == trial.test_size).all()
    assert np.array_equal((~test) @ frames.full_pilot_reads.to_numpy(), trial.pilot_cost)
    bench = weight @ rates[:,config.index(LOW_COST),:]
    for i, e in enumerate(ENDPOINTS):
        trial["cheap_"+e] = bench[:,i]
    np.testing.assert_allclose(trial.test_full5, weight @ rates[:,config.index(FULL_FIVE),2], atol=1e-12)
    return trial.rename(columns={"test_size":"n_test"}).assign(mode="original37", panel="random_within_slide",
                                                                    test_fixed20=bench[:,2])


def summarize(trials):
    rows = []
    keys = ["mode","panel","quorum","method","X","target"]
    for k,g in trials.groupby(keys, sort=True):
        out = dict(zip(keys,k))
        out.update(n_test=g.n_test.iloc[0], reps=len(g), pilot_cost_mean=g.pilot_cost.mean(),
                   budget_mean=g.budget.mean(), budget_q1=g.budget.quantile(.25),
                   budget_median=g.budget.median(), budget_q3=g.budget.quantile(.75),
                   sequential_fraction=g.policy.str.startswith("sequential").mean())
        for e in ENDPOINTS:
            out.update({f"{e}_mean":g[e].mean(),f"{e}_lo":g[e].quantile(.025),f"{e}_hi":g[e].quantile(.975)})
        out.update({f"holdout_{int(gate*100)}":meets(g.joint,gate).mean() for gate in GATES})
        rows.append(out)
    return pd.DataFrame(rows)


def paired_at_x(trials):
    keys = ["mode","panel","quorum","method","target"]
    rows = []
    for key,g in trials.groupby(keys,sort=True):
        baseline = g[g.X.eq(8)].set_index("rep")
        for x in (12,16,24):
            chosen = g[g.X.eq(x)].set_index("rep").reindex(baseline.index)
            assert baseline.validation_slides.equals(chosen.validation_slides)
            change = chosen.joint - baseline.joint
            rows.append((*key,x,change.mean(),change.quantile(.025),change.quantile(.975),
                         (chosen.budget-baseline.budget).mean(),
                         meets(chosen.joint,.90).mean()-meets(baseline.joint,.90).mean(),
                         meets(chosen.joint,.95).mean()-meets(baseline.joint,.95).mean()))
    return pd.DataFrame(rows,columns=[*keys,"X","joint_change","joint_change_lo","joint_change_hi","budget_change",
                                      "holdout90_change","holdout95_change"])


def project_cost(trials, benchmarks, project_sizes=PROJECTS):
    rows = []
    keys = ["mode","panel","quorum","method","X","target"]
    for key,g in trials.groupby(keys,sort=True):
        x = key[4]
        for name,b0 in benchmarks.items():
            base = g["test_"+name]
            joint_difference = g.joint-base
            gain = b0 - g.budget
            # When the denominator is zero or negative, a reading-cost break-even does not exist.
            for compiled in (True,False):
                net_upfront = g.pilot_cost - (x*g.budget if compiled else 0)
                breakeven = np.floor(net_upfront[gain>0]/gain[gain>0])+1
                for gate in GATES:
                    adequacy = meets(g.joint,gate)
                    base_ok = meets(base,gate)
                    both = adequacy & base_ok
                    for m in project_sizes:
                        costs = net_upfront+m*g.budget
                        baseline = m*b0
                        saving = costs < baseline
                        rows.append((*key,name,compiled,gate,m,baseline,costs.mean(),costs.quantile(.025),costs.quantile(.975),
                                     base.mean(),adequacy.mean(),base_ok.mean(),(adequacy & ~base_ok).mean(),
                                     (both & saving).mean(),saving.mean(),(gain>0).mean(),breakeven[both].median(),
                                     joint_difference.mean(),joint_difference.quantile(.025),joint_difference.quantile(.975)))
    return pd.DataFrame(rows,columns=[*keys,"benchmark","reuse","gate","project_slides","benchmark_total","pilot_total_mean",
                                     "pilot_total_lo","pilot_total_hi","benchmark_joint_mean","pilot_met","benchmark_met",
                                     "only_pilot_met","both_met_and_saved","saved","cheaper_production","break_even_median_both",
                                     "joint_difference","joint_difference_lo","joint_difference_hi"])


def fixed_panel_replay(inputs, draws, seed, threads):
    set_num_threads(threads)
    frames = frames_from_zip(inputs)
    readers = sorted(set.intersection(*(set(f.readers) for f in frames)))
    assert len(readers) == 5
    configs, *grid = design_grid(3)
    panel_ids, rows = [], []
    for n, trio in enumerate(combinations(readers,3), start=1):
        panel = f"team{n:02}"
        panel_ids.append((panel,*trio))
        for frame in frames:
            fixed = Frame(frame.slide,frame.batch,trio,frame.scores[[frame.readers.index(r) for r in trio]])
            roster,regions,ref = draws_for(fixed,draws,seed)
            values = assignment_outcomes(fixed.scores,roster,regions,ref,*grid).sum(axis=0)[:,0]
            for c,(r,b,p,h) in enumerate(configs):
                for q in (2,3):
                    rows.append((panel,frame.slide,"team_reference",r,b,p,h,q,draws,*values[c,q-2]))
        print(f"Replayed fixed panel {panel}",flush=True)
    labels = "panel slide reference readers budget policy common quorum draws score material joint".split()
    return pd.DataFrame(rows,columns=labels),pd.DataFrame(panel_ids,columns=["panel","reader1","reader2","reader3"])


def analyze_fixed(data, frames, reps, seed):
    frames = frames.assign(full_pilot_reads=30)
    splits = shared_splits(frames,24,reps,seed)
    cases, curves, crossed = [], [], []
    keys = ["panel","quorum","budget","policy","common"]
    for panel,raw in data.groupby("panel",sort=True):
        for q in (2,3):
            config,a = cube(raw,frames.slide.tolist(),3,q,"team_reference")
            cases.append(evaluate_splits(frames,splits,config,a,FIXED_BENCHMARKS,"fixed64",panel,q))
            for (r,b,method,h),value in zip(config,a.mean(axis=0),strict=True):
                curves.append((panel,q,b,method,h,*value))
            for name in ("fixed","sequential","sequential_spare"):
                for target in TARGETS:
                    ix = [i for i,(r,b,m,h) in enumerate(config) if m == name]
                    mean = a[:,:,2].mean(axis=0)
                    qualifying = [i for i in ix if meets(mean[i],target)]
                    first = min(qualifying,key=lambda i:(config[i][1],-mean[i],config[i][3])) if qualifying else None
                    crossed.append((panel,q,name,target,config[first][1] if first is not None else np.nan,mean[ix].max()))
    return (pd.concat(cases,ignore_index=True),pd.DataFrame(curves,columns=[*keys,*ENDPOINTS]),
            pd.DataFrame(crossed,columns=["panel","quorum","policy","target","budget","max_joint"]))


def main(args):
    output = args.output
    output.mkdir(parents=True,exist_ok=True)
    f = pd.read_csv(args.parent/"frames.csv")
    if not args.fixed_only:
        f37 = f[f.complete_readers.ge(7)].reset_index(drop=True)
        raw = pd.read_csv(args.parent/"replay_counts.csv.gz")
        config,a = cube(raw,f37.slide.tolist(),7,2,"all_reference")
        baselines = {"fixed20":LOW_COST,"full5":FULL_FIVE}
        original = original_split_benchmark(args.parent,f37,config,a)
        matched = evaluate_splits(f37,shared_splits(f37,13,args.reps,args.split_seed),config,a,baselines,
                                  "common37","random_within_slide",2)
        save(original,output,"reviewer_benchmark_splits",True)
        save(matched,output,"common_validation_splits",True)
        save(summarize(pd.concat([original,matched])),output,"variable_panel_summary")
        save(paired_at_x(matched),output,"common_validation_paired_X")
        save(project_cost(pd.concat([original,matched]),{"fixed20":20,"full5":50}),output,"variable_panel_cost")
    raw,identities = fixed_panel_replay(args.inputs,args.draws,args.seed,args.threads)
    trial,curves,crossed = analyze_fixed(raw,f,args.reps,args.split_seed+1)
    save(identities,output,"fixed_panel_identities")
    save(raw,output,"fixed_panel_counts",True)
    save(trial,output,"fixed_panel_splits",True)
    save(curves,output,"fixed_panel_curves")
    save(crossed,output,"fixed_panel_crossings")
    save(summarize(trial),output,"fixed_panel_summary")
    save(paired_at_x(trial),output,"fixed_panel_paired_X")
    prices = {name:cfg[1] for name,cfg in FIXED_BENCHMARKS.items()}
    save(project_cost(trial,prices,PROJECTS[1:]),output,"fixed_panel_cost")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("inputs","parent","output"):
        parser.add_argument("--"+arg,type=Path,required=True)
    parser.add_argument("--draws",type=int,default=4096)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--reps",type=int,default=1000)
    parser.add_argument("--seed",type=int,default=261180)
    parser.add_argument("--split-seed",type=int,default=261181)
    parser.add_argument("--fixed-only",action="store_true")
    main(parser.parse_args())
