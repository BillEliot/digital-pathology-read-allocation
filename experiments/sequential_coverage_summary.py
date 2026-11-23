"""Summarize stopping and coverage using existing cyclic replay histograms."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from overlap_pilot import schedule
from pilot_allocation import load_htt_ratings


TARGETS = {"KidneyAI": 5, "FDA_HTT": 2, "CrowdGleason": 2}
KEYS = ["dataset", "group"]


def empirical_quartiles(histogram):
    counts = histogram.groupby("prefix").draws.sum().sort_index()
    positions = np.ceil(counts.sum() * np.array([.25, .5, .75]))
    return counts.index.to_numpy()[np.searchsorted(counts.cumsum().to_numpy(), positions)]


def add_schedule_sizes(behavior):
    keys = ["regions", "quota", "prefix"]
    memo = []
    for size, quota, h in behavior[keys].drop_duplicates().itertuples(index=False, name=None):
        slots = schedule(size, quota, h)
        frequencies = np.bincount(slots.ravel(), minlength=size)
        distinct = np.count_nonzero(frequencies)
        assert distinct == h + min(size - h, 3 * (quota - h))
        memo.append((size, quota, h, distinct, np.count_nonzero(frequencies == 3)))
    return behavior.merge(
        pd.DataFrame(memo, columns=keys + ["covered", "recomputed_shared"]),
        on=keys,
        validate="many_to_one",
    )


def stop_and_coverage(behavior, calibration, fixed_grid):
    behavior = add_schedule_sizes(behavior)
    assert behavior.shared.eq(behavior.recomputed_shared).all()
    assert (behavior.prefix + behavior.followup).eq(behavior.quota).all()
    behavior["reached"] = behavior.jointly_evaluable.ge(behavior.dataset.map(TARGETS))
    behavior["hit_before_cap"] = behavior.reached & behavior.followup.gt(0)
    behavior["hit_at_cap"] = behavior.reached & behavior.followup.eq(0)
    sequence = behavior.query("allocation=='full_cap'").copy()
    assert sequence.groupby(KEYS).draws.sum().eq(4000).all()
    frames = sequence.groupby(KEYS).agg(regions=("regions", "first"), quota=("quota", "first")).reset_index()
    fixed_keys = calibration.query("suffix=='cyclic'")[KEYS + ["heldout_reserve"]]
    fixed = frames.merge(fixed_keys, on=KEYS, validate="one_to_one").rename(columns={"heldout_reserve": "prefix"})
    fixed = fixed.merge(
        fixed_grid.query("suffix=='cyclic'").rename(columns={"reserve": "prefix"})[KEYS + ["prefix", "draws", "review_pass"]],
        on=KEYS + ["prefix"],
        validate="one_to_one",
    )
    assert len(fixed) == len(frames) == 182 and fixed.draws.eq(4000).all()
    fixed = add_schedule_sizes(fixed).rename(columns={"recomputed_shared": "shared"})
    fixed["allocation"] = "heldout_fixed"
    fixed["reached_count"] = fixed.review_pass
    sequence["reached_count"] = sequence.draws * sequence.reached
    for state in ("hit_before_cap", "hit_at_cap"):
        sequence[state + "_count"] = sequence.draws * sequence[state]
    coverage = pd.concat((sequence, fixed), ignore_index=True)
    for field in ("prefix", "covered", "shared"):
        coverage[f"sum_{field}"] = coverage[field] * coverage.draws
    coverage["sum_covered_pct"] = 100 * coverage.covered / coverage.regions * coverage.draws
    summary = []
    for (dataset, allocation), values in coverage.groupby(["dataset", "allocation"]):
        d = int(values.draws.sum())
        low, middle, high = empirical_quartiles(values)
        row = dict(dataset=dataset, allocation=allocation, frames=values.group.nunique(), draws=d,
                   prefix_mean=values.sum_prefix.sum()/d, prefix_q25=low, prefix_median=middle, prefix_q75=high,
                   covered_mean=values.sum_covered.sum()/d, covered_pct=values.sum_covered_pct.sum()/d,
                   shared_mean=values.sum_shared.sum()/d, review_pct=100*values.reached_count.sum()/d)
        for state in ("hit_before_cap", "hit_at_cap"):
            row[state + "_pct"] = 100*values[state + "_count"].sum(min_count=1)/d
        summary.append(row)

    fields = ["draws", "sum_prefix", "sum_covered", "sum_covered_pct", "sum_shared", "reached_count",
              "hit_before_cap_count", "hit_at_cap_count"]
    by_frame = coverage.groupby(KEYS + ["allocation"])[fields].sum(min_count=1).reset_index()
    base = fixed[KEYS + ["prefix", "covered", "shared"]].rename(
        columns={f: f"fixed_{f}" for f in ("prefix", "covered", "shared")}
    )
    paired = sequence.merge(base, on=KEYS, validate="many_to_one")
    paired["covered_gain"] = paired.covered - paired.fixed_covered
    distributions = paired.groupby(["dataset", "covered_gain"]).draws.sum().reset_index()
    return pd.DataFrame(summary), by_frame, distributions


def platform_counts(raw):
    groups = raw.groupby("modalityID", dropna=False)
    count = groups.agg(records=("caseID", "size"), slides=("WSI", "nunique"),
                       distinct_rois=("caseID","nunique"), readers=("readerID","nunique")).reset_index()
    return count.assign(retained_for_panel=count.modalityID.eq("camic"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--behavior", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--fixed-grid", type=Path, required=True)
    p.add_argument("--estimator", choices=("phase", "pooled"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    calibration, fixed_grid = pd.read_csv(a.calibration), pd.read_csv(a.fixed_grid)
    if a.estimator:
        calibration = calibration[calibration.estimator.eq(a.estimator)]
        fixed_grid = fixed_grid[fixed_grid.estimator.eq(a.estimator)]
    tables = stop_and_coverage(pd.read_csv(a.behavior), calibration, fixed_grid)
    for name, table in zip(("coverage_summary", "coverage_by_frame", "coverage_difference_distribution"), tables, strict=True):
        table.to_csv(a.output / f"{name}.csv", index=False)
    platform_counts(load_htt_ratings(a.data)).to_csv(a.output / "fda_platform_screening.csv", index=False)


if __name__ == "__main__":
    main()
