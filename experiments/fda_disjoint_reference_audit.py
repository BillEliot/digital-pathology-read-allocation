"""Recheck the FDA HTT 70-reading, seven-reader reference comparison.

Read the pinned release and optionally check the original replay counts. Compute the
full-budget endpoint directly from eligible totals, then enumerate all disjoint
seven-reader working and three-reader reference combinations on each slide.
"""
import argparse
import csv
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
from itertools import combinations
from math import comb
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


HASH = "0bd851c039a43944febed68bb0e5d2624dcef83acc064387d1ac66e07f1f166b"
SEEDS = (261170, 261172)
DRAWS = 4096
GATES = (2.5, 5.0, 10.0)
WORKERS = 7
HELD_OUT = 3


def load_labels(path):
    with ZipFile(path) as z:
        member, = [x for x in z.namelist() if x.endswith("/inst/extdata/pilotHTT_RST.csv")]
        payload = z.read(member)
    if sha256(payload).hexdigest() != HASH:
        raise ValueError("Input label file does not match the pinned HTT release")
    return pd.read_csv(BytesIO(payload))


def complete_frames(routine):
    frames = {}
    for slide, part in routine.groupby("WSI"):
        columns = sorted(part.caseID.unique())
        batch, = part.batch.unique()
        assert len(columns) == 10
        values, flags = [], []
        for _, reader in part.groupby("readerID"):
            if len(reader) != len(columns) or reader.evaluable.isna().any():
                continue
            r = reader.set_index("caseID").loc[columns]
            assessable = r.evaluable.eq(True).to_numpy()
            grade = r.densityTILs.to_numpy(dtype=float)
            if not ((~assessable) | (np.isfinite(grade) & (grade >= 0) & (grade <= 100))).all():
                continue
            values.append(np.where(assessable, grade, 0))
            flags.append(assessable)
        frames[slide] = batch, np.asarray(values), np.asarray(flags)
    return frames


def ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float),
                     where=denominator > 0)


def seed_check(slide, values, eligible):
    rows, n = [], len(values)
    reader_sum, reader_n = values.sum(axis=1), eligible.sum(axis=1)
    all_ref = values.sum()/eligible.sum()
    slide_rng = int(sha256(slide.encode()).hexdigest()[:8], 16)
    for seed in SEEDS:
        # Reproduce only the reader order. At the full budget, region order and
        # the initial reserve cannot alter the set of assignments.
        rng = np.random.default_rng(np.random.SeedSequence([seed, slide_rng]))
        order = np.argsort(rng.random((DRAWS, n)), axis=1)
        reference_ids, work = order[:, :HELD_OUT], order[:, HELD_OUT:HELD_OUT+WORKERS]
        assert not np.any(work[:, :, None] == reference_ids[:, None, :])
        work_ref = ratio(reader_sum[work].sum(axis=1), reader_n[work].sum(axis=1))
        disjoint = ratio(reader_sum[reference_ids].sum(axis=1), reader_n[reference_ids].sum(axis=1))
        regional_n = eligible[work].sum(axis=1)
        for name, ref in (("all_reference_paired", all_ref), ("independent3", disjoint)):
            score = np.abs(work_ref-ref) <= 5 + 1e-12
            material = (regional_n >= 2).sum(axis=1) >= 2
            rows.append(dict(slide=slide, seed=seed, reference=name, draws=DRAWS,
                             score=int(score.sum()), material=int(material.sum()),
                             joint=int((score & material).sum())))
    return rows


def exact_check(slide, batch, values, eligible):
    n = len(values)
    reader_sum, reader_n = values.sum(axis=1), eligible.sum(axis=1)
    all_score, all_records = reader_sum.sum(), reader_n.sum()
    all_ref = all_score/all_records
    work_ids = np.array(list(combinations(range(n),WORKERS)))
    heldout_positions = np.array(list(combinations(range(n-WORKERS),HELD_OUT)))
    totals = defaultdict(float)
    for begin in range(0,len(work_ids),512):
        work = work_ids[begin:begin+512]
        keep = np.ones((len(work),n), dtype=bool)
        keep[np.arange(len(work))[:,None],work] = False
        rest = np.nonzero(keep)[1].reshape((len(work),n-WORKERS))
        rest_n = reader_n[rest].sum(axis=1)
        num,den = reader_sum[work].sum(axis=1),reader_n[work].sum(axis=1)
        working = ratio(num,den)
        other = ratio(reader_sum[rest].sum(axis=1),rest_n)
        selected = rest[:,heldout_positions]
        holdout = ratio(reader_sum[selected].sum(axis=2),reader_n[selected].sum(axis=2))
        valid = np.isfinite(working).all() and np.isfinite(other).all() and np.isfinite(holdout).all()
        assert valid
        records = eligible[work].sum(axis=1)
        material = (records >= 2).sum(axis=1) >= 2
        assert material.all()
        assert np.allclose(working-all_ref, (rest_n/all_records)*(working-other), rtol=0,atol=2e-14)
        dist_all = np.abs(working-all_ref)
        dist_other = np.abs(working-other)
        dist_holdout = np.abs(working[:,None]-holdout)
        per_work = len(heldout_positions)
        totals["pairs"] += len(work)*per_work
        totals["working_record_share"] += (den/all_records).sum()*per_work
        totals["mae_all"] += dist_all.sum()*per_work
        totals["mae_nonworking"] += dist_other.sum()*per_work
        totals["mae_three"] += dist_holdout.sum()
        totals["reference_mae_three_vs_all"] += np.abs(holdout-all_ref).sum()
        for gate in GATES:
            all_pass = dist_all <= gate + 1e-12
            other_pass = dist_other <= gate + 1e-12
            three_pass = dist_holdout <= gate + 1e-12
            totals[f"all_{gate:g}"] += all_pass.sum()*per_work
            totals[f"nonworking_{gate:g}"] += other_pass.sum()*per_work
            totals[f"three_{gate:g}"] += three_pass.sum()
            if gate == 5:
                totals["all_only_5"] += (all_pass[:,None] & ~three_pass).sum()
                totals["three_only_5"] += (~all_pass[:,None] & three_pass).sum()
    pairs = comb(n,WORKERS)*comb(n-WORKERS,HELD_OUT)
    assert pairs == totals.pop("pairs")
    rates = {k:float(v/pairs) for k,v in totals.items()}
    assert abs(rates["all_5"]-rates["three_5"]-rates["all_only_5"]+rates["three_only_5"])<1e-12
    return dict(slide=slide,batch=batch,complete_readers=n,pairs=pairs,all_ref=all_ref,**rates)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    raw = load_labels(args.input)
    screening = raw.groupby("modalityID").agg(records=("WSI","size"),slides=("WSI","nunique"),readers=("readerID","nunique"))
    routine = raw[raw.modalityID.eq("camic")]
    nonroutine = raw[~raw.modalityID.eq("camic")]
    assert len(raw) == 7898 and len(routine)==5100 and raw.WSI.nunique()==routine.WSI.nunique()==64
    assert len(raw[["WSI","caseID"]].drop_duplicates())==len(routine[["WSI","caseID"]].drop_duplicates())==640
    frames = complete_frames(routine)
    assert sum(len(x[1])*10 for x in frames.values()) == 5090
    print(screening)
    print("Reader IDs routine, nonroutine, nonroutine-only:",
          routine.readerID.nunique(),nonroutine.readerID.nunique(),len(set(nonroutine.readerID)-set(routine.readerID)))
    print("Incomplete sets:",routine.groupby(["WSI","readerID"]).size().loc[lambda s:s<10].sort_values().tolist())
    exact, seeded = [], []
    for slide,(batch,values,eligible) in frames.items():
        if len(values) < WORKERS+HELD_OUT:
            continue
        seeded.extend(seed_check(slide,values,eligible))
        exact.append(exact_check(slide,batch,values,eligible))
    assert len(exact)==8 and len(set(r["batch"] for r in exact))==1
    s = pd.DataFrame(seeded)
    s.to_csv(args.output/"seed_recheck.csv",index=False)
    e = pd.DataFrame(exact)
    e.to_csv(args.output/"exact_by_slide.csv",index=False)
    s["joint_rate"] = s.joint/s.draws
    print("Seeded, equal-slide means:")
    print(s.groupby(["seed","reference"]).joint_rate.mean())
    columns = ["working_record_share","all_2.5","nonworking_2.5","three_2.5","all_5","nonworking_5","three_5",
               "all_10","nonworking_10","three_10","mae_all","mae_nonworking","mae_three",
               "reference_mae_three_vs_all","all_only_5","three_only_5"]
    summarized = e[columns].mean()
    summarized.to_json(args.output/"exact_equal_slide.json",indent=2,double_precision=14)
    print("Exact, equal-slide means:")
    print(summarized.to_string())
    print("Exact cases:",e.pairs.sum())
    print(e[["slide","complete_readers","pairs","all_ref","all_5","nonworking_5","three_5","mae_all","mae_three"]].to_string(index=False))
    screening.to_csv(args.output/"screening.csv")


if __name__ == "__main__":
    main()
