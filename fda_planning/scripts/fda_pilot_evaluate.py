"""Fit study-level pilot policies, validate on unseen slides, and charge the trial reads."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COHORTS = {"common37": 7, "common56": 6, "all64": 5, "independent8": 10}
ENDPOINTS = ["score", "material", "joint"]
DESIGN = ["readers", "budget", "policy", "common"]
PILOT_SIZES = (8, 12, 16, 24)
SUMMARY_BUDGETS = (12, 15, 18, 20, 24, 30)
TARGETS = (.85, .90, .95)
PROJECT_SIZES = (37, 64, 100, 250, 500)


def cube(data, slides, max_readers, quorum, reference):
    sub = data[data.slide.isin(slides) & data.readers.between(quorum, max_readers) &
               data.quorum.eq(quorum) & data.reference.eq(reference)]
    assert not sub.duplicated(["slide", *DESIGN]).any()
    configs = sub[DESIGN].drop_duplicates().sort_values(["budget", "readers", "policy", "common"])
    cfg = list(configs.itertuples(index=False, name=None))
    cols = pd.MultiIndex.from_tuples(cfg, names=DESIGN)
    matrices = [sub.assign(rate=sub[endpoint] / sub.draws).pivot(index="slide", columns=DESIGN, values="rate")
                .reindex(index=slides, columns=cols).to_numpy() for endpoint in ENDPOINTS]
    out = np.stack(matrices, axis=-1)
    assert np.isfinite(out).all() and (out[:, :, 2] <= np.minimum(out[:, :, 0], out[:, :, 1])).all()
    return cfg, out


def heldout_curves(cfg, data, calibration):
    """A reserve is fitted without the evaluated slide (and shared across references)."""
    r_b = sorted({(r, b) for r, b, _, _ in cfg})
    design_index = {d: i for i, d in enumerate(cfg)}
    for r, budget in r_b:
        cap = budget // r
        fixed = [design_index[(r, budget, "fixed", h)] for h in range(cap + 1)]
        values = calibration[:, fixed, 2]
        other = values.sum(axis=0) - values
        reserve = np.argmax(other.round(12), axis=1)
        assert (reserve <= cap).all()
        row = np.arange(len(data))
        yield (r, budget, "fixed_heldout"), data[row, np.array(fixed)[reserve]], reserve
        for mode, end in (("sequential", cap), ("sequential_spare", max(0, cap - 1))):
            yield (r, budget, mode), data[:, design_index[(r, budget, mode, end)]], None


def low_high(values):
    return np.quantile(values, [.025, .975], axis=0)


def meets(rate, target):
    return np.round(rate, 12) >= target


def curves(data, frames, bootstrap_seed):
    curve_rows, reserves, matched, crossings = [], [], [], []
    for cohort, floor in COHORTS.items():
        f = frames[frames.complete_readers.ge(floor)]
        slides = f.slide.tolist()
        max_readers = min(floor, 7)
        n = len(f)
        rng = np.random.default_rng(bootstrap_seed + floor)
        weights = rng.multinomial(n, np.repeat(1/n, n), size=2000) / n
        references = ("all_reference_paired", "independent3") if cohort == "independent8" else ("all_reference",)
        for q in (2, 3):
            configs, calibration = cube(data, slides, max_readers, q, references[0])
            ref_primary = {}
            for ref in references:
                ref_cfg, eval_data = cube(data, slides, max_readers, q, ref)
                assert ref_cfg == configs
                cur = {}
                for (r, b, method), by_slide, h in heldout_curves(configs, eval_data, calibration):
                    key = (r, b, method)
                    cur[key] = by_slide
                    line = dict(cohort=cohort, slides=n, reference=ref, quorum=q, readers=r, budget=b, policy=method)
                    line.update(zip(ENDPOINTS, by_slide.mean(axis=0)))
                    if h is not None:
                        for slide, reserve in zip(slides, h, strict=True):
                            reserves.append((cohort, ref, q, r, b, slide, reserve))
                    if b in SUMMARY_BUDGETS:
                        lo, hi = low_high(weights @ by_slide)
                        for endpoint, a, z in zip(ENDPOINTS, lo, hi, strict=True):
                            line.update({f"{endpoint}_lo": a, f"{endpoint}_hi": z})
                    curve_rows.append(line)
                for (r, b, method), values in cur.items():
                    if b not in SUMMARY_BUDGETS:
                        continue
                    comparisons = []
                    if method == "sequential":
                        comparisons.append(("sequential - fixed", cur[(r, b, "fixed_heldout")]))
                    if r != 3 and (3, b, method) in cur:
                        comparisons.append(("versus 3 readers", cur[(3, b, method)]))
                    if ref == "independent3":
                        comparisons.append(("independent reference - all-reader reference", ref_primary[r, b, method]))
                    for name, baseline in comparisons:
                        delta = values - baseline
                        lo, hi = low_high(weights @ delta)
                        for endpoint, point, a, z in zip(ENDPOINTS, delta.mean(axis=0), lo, hi, strict=True):
                            matched.append((cohort, ref, q, r, b, method, name, endpoint, point, a, z))
                if ref == references[0]:
                    ref_primary = cur
                for r in range(q, max_readers + 1):
                    for method in ("fixed_heldout", "sequential", "sequential_spare"):
                        options = [(b, vals.mean(axis=0)) for (nr, b, strategy), vals in cur.items()
                                   if nr == r and strategy == method]
                        for target in TARGETS:
                            found = next(((b, metrics) for b, metrics in options if meets(metrics[2], target)), None)
                            best = max(metrics[2] for _, metrics in options)
                            b, rates = found if found else (None, np.full(3, np.nan))
                            crossings.append((cohort, n, ref, q, r, method, target, b, *rates, best))
    index_cols = ["cohort", "reference", "quorum", "readers", "budget"]
    return {
        "budget_curves": pd.DataFrame(curve_rows),
        "heldout_reserves": pd.DataFrame(reserves, columns=[*index_cols, "slide", "common"]),
        "matched_contrasts": pd.DataFrame(matched, columns=[*index_cols, "policy", "contrast", "endpoint", "difference", "lo", "hi"]),
        "budget_crossings": pd.DataFrame(crossings, columns=["cohort", "slides", "reference", "quorum", "readers", "policy", "target", "budget", *ENDPOINTS, "maximum_joint"]),
    }


def strat_order(rank, strata):
    """Guarantee every batch once, then fill nested proportional deficits."""
    k = len(strata)
    sorted_groups = [g[np.argsort(rank[g])] for g in strata]
    sizes = np.array([len(g) for g in strata])
    frac = sizes / len(rank)
    counts = np.ones(k, dtype=int)
    order = [g[0] for g in sorted_groups]
    for count in range(k + 1, max(PILOT_SIZES) + 1):
        deficit = np.where(counts < sizes, count * frac - counts, -np.inf)
        choices = np.flatnonzero(np.isclose(deficit, deficit.max()))
        choice = min(choices, key=lambda j: rank[sorted_groups[j][counts[j]]])
        order.append(sorted_groups[choice][counts[choice]])
        counts[choice] += 1
    return np.array(order)


def choose(rates, configs, target):
    """Candidates sorted by budget, reader count, strategy name, then reserve."""
    rates = rates.round(12)
    qualified = meets(rates, target)
    if qualified.any():
        b = configs[np.argmax(qualified)][1]
        opts = np.array([i for i, (_, budget, _, _) in enumerate(configs) if budget == b and qualified[i]])
        return opts[np.argmax(rates[opts])], True
    return np.argmax(rates), False


def pilot_trials(data, frames, splits, seed):
    records, cost_records = [], []
    for cohort in ("common37", "common56", "all64"):
        top = COHORTS[cohort]
        f = frames[frames.complete_readers.ge(top)].reset_index(drop=True)
        strata = list(f.groupby("batch", sort=True).indices.values())
        n = len(f)
        membership = pd.factorize(f.batch, sort=True)[0]
        sizes = np.bincount(membership)
        rng = np.random.default_rng(seed + top)
        ranks = rng.random((splits, n))
        batch_order = np.stack([strat_order(rank, strata) for rank in ranks])
        orders = {"random": np.argsort(ranks, axis=1), "batch_stratified": batch_order}
        for q in ((2, 3) if cohort == "common37" else (2,)):
            cfg, a = cube(data, f.slide.tolist(), top, q, "all_reference")
            look = {candidate: i for i, candidate in enumerate(cfg)}
            baseline_designs = {
                "three15_all_shared": (3, 15, "fixed", 5),
                **{f"full{r}": (r, r * 10, "fixed", 10) for r in (3, 5, 7) if r <= top},
            }
            baseline_names = list(baseline_designs)
            baseline_idx = [look[c] for c in baseline_designs.values()]
            baseline_costs = np.array([c[1] for c in baseline_designs.values()])
            all_choices = np.arange(len(cfg))
            fixed_three = np.flatnonzero([r == 3 for r, _, _, _ in cfg])
            for method, order in orders.items():
                for x in PILOT_SIZES:
                    samples = order[:, :x]
                    short_ids = f.slide.str.replace("HTT-TILS-", "", regex=False).str.replace(".ndpi", "", regex=False).to_numpy()
                    trial_sets = ["|".join(sorted(short_ids[idx])) for idx in samples]
                    mask = np.zeros((splits, n), dtype=bool)
                    mask[np.arange(splits)[:, None], samples] = True
                    assert (mask.sum(axis=1) == x).all()
                    test = (~mask).astype(float) / (n - x)
                    train = mask.astype(float) / x
                    if method == "batch_stratified":
                        found = mask.astype(float) @ np.eye(len(strata))[membership]
                        assert (found > 0).all()
                        train = mask * (sizes[membership] / n) / found[:, membership]
                    assert np.allclose(train.sum(axis=1), 1) and (train * test == 0).all()
                    train_quality = train @ a[:, :, 2]
                    test_baselines = test @ a[:, baseline_idx, 2]
                    pilot_cost = mask @ f.full_pilot_reads.to_numpy(dtype=int)
                    for scope, eligible in (("all_sizes", all_choices), ("three_only", fixed_three)):
                        candidate_cfg = [cfg[j] for j in eligible]
                        goals = TARGETS if scope == "all_sizes" else (.90,)
                        for target in goals:
                            for rep in range(splits):
                                selected, hit = choose(train_quality[rep, eligible], candidate_cfg, target)
                                c = eligible[selected]
                                r, b, policy, common = cfg[c]
                                rate = test[rep] @ a[:, c, :]
                                record = dict(cohort=cohort, quorum=q, scope=scope, method=method, X=x, rep=rep, target=target,
                                              n_frames=n, test_size=n-x, pilot_slides=trial_sets[rep], pilot_cost=pilot_cost[rep],
                                              readers=r, budget=b, policy=policy, common=common, train_joint=train_quality[rep, c],
                                              pilot_met=hit, score=rate[0], material=rate[1], joint=rate[2])
                                for j, name in enumerate(baseline_names):
                                    record[f"test_{name}"] = test_baselines[rep, j]
                                records.append(record)
                                if scope != "all_sizes" or target != .90:
                                    continue
                                for m in PROJECT_SIZES:
                                    reuse = pilot_cost[rep] + (m - x) * b
                                    separate = pilot_cost[rep] + m * b
                                    for j, name in enumerate(baseline_names):
                                        cost_records.append((cohort, q, method, x, rep, m, name, pilot_cost[rep], b,
                                                             reuse, separate, baseline_costs[j]*m, rate[2], test_baselines[rep, j]))
    columns = ["cohort", "quorum", "method", "X", "rep", "project_slides", "baseline", "pilot_cost", "budget",
               "reuse_total", "extra_total", "baseline_total", "joint", "baseline_joint"]
    return pd.DataFrame(records), pd.DataFrame(cost_records, columns=columns)


def describe_trials(trials):
    keys = ["cohort", "quorum", "scope", "target", "method", "X"]
    records = []
    choices = trials.value_counts(keys + DESIGN).rename("splits").reset_index()
    for key, v in trials.groupby(keys, sort=True):
        row = dict(zip(keys, key, strict=True))
        row.update(splits=len(v), test_size=v.test_size.iloc[0], train_met=v.pilot_met.mean(), heldout_met=meets(v.joint, v.target).mean(),
                   optimism=(v.train_joint - v.joint).mean(), pilot_cost_mean=v.pilot_cost.mean(), pilot_cost_sd=v.pilot_cost.std())
        for endpoint in ENDPOINTS:
            lo, med, hi = v[endpoint].quantile([.025, .50, .975])
            row.update({f"{endpoint}_mean": v[endpoint].mean(), f"{endpoint}_lo": lo, f"{endpoint}_median": med, f"{endpoint}_hi": hi})
        for col in ("budget", "readers"):
            for key_q, a in zip(("lo", "q1", "median", "q3", "hi"), v[col].quantile([.025, .25, .50, .75, .975]), strict=True):
                row[f"{col}_{key_q}"] = a
        mode = v.value_counts(DESIGN).reset_index(name="count").iloc[0]
        row.update(modal_frequency=mode["count"]/len(v), **{f"modal_{k}": mode[k] for k in DESIGN},
                   sequential_fraction=v.policy.str.startswith("sequential").mean())
        row["primary_stable"] = (row["heldout_met"] >= .80 and row["budget_q3"] - row["budget_q1"] <= 2) if key[3] == .90 else None
        records.append(row)
    return pd.DataFrame(records), choices


def describe_costs(costs):
    keys = ["cohort", "quorum", "method", "X", "project_slides", "baseline"]
    rows = []
    for key, v in costs.groupby(keys, sort=True):
        row = dict(zip(keys, key, strict=True))
        selected_met, baseline_met = meets(v.joint, .90), meets(v.baseline_joint, .90)
        both = selected_met & baseline_met
        difference = v.joint - v.baseline_joint
        row.update(baseline_total=v.baseline_total.iloc[0], pilot_cost_mean=v.pilot_cost.mean(), pilot_fraction_of_slides=key[3]/key[4],
                   baseline_met=baseline_met.mean(), selected_met=selected_met.mean(), both_met=both.mean(),
                   joint_difference_mean=difference.mean(), joint_difference_lo=difference.quantile(.025), joint_difference_hi=difference.quantile(.975))
        for col in ("reuse", "extra"):
            total = v[f"{col}_total"]
            net = v.baseline_total - total
            row.update({f"{col}_mean": total.mean(), f"{col}_lo": total.quantile(.025), f"{col}_hi": total.quantile(.975),
                        f"{col}_pilot_fraction": (v.pilot_cost / total).mean(),
                        f"{col}_save_fraction": (net > 0).mean(),
                        f"{col}_both_met_and_save": (both & (net > 0)).mean(),
                        f"{col}_net_given_both": net[both].mean()})
        rows.append(row)
    return pd.DataFrame(rows)


def main(directory, output, splits, seed):
    d = pd.read_csv(directory / "replay_counts.csv.gz")
    f = pd.read_csv(directory / "frames.csv")
    outcomes = curves(d, f, seed)
    trials, costs = pilot_trials(d, f, splits, seed)
    summary, choices = describe_trials(trials)
    outcomes.update(pilot_splits=trials, pilot_summary=summary, pilot_choices=choices,
                    project_costs=describe_costs(costs))
    output.mkdir(parents=True, exist_ok=True)
    for name, table in outcomes.items():
        if name in ("pilot_splits", "heldout_reserves"):
            table.to_csv(output / f"{name}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        else:
            table.to_csv(output / f"{name}.csv", index=False)
        print(f"{name}: {len(table):,} rows", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=261171)
    args = parser.parse_args()
    main(args.replay, args.output, args.splits, args.seed)
