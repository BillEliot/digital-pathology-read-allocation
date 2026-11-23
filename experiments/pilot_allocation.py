"""Replay a paid common-reading pilot before selecting additional overlap."""

import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import betabinom

from kidney_data import VALUES, load_panel
from overlap_pilot import schedule
from review_supplement import ORDERS, PAIRS


def load_htt_ratings(data):
    with ZipFile(data / "FDA_HTT.zip") as source:
        member, = [name for name in source.namelist() if name.endswith("/inst/extdata/pilotHTT_RST.csv")]
        return pd.read_csv(source.open(member))


def panels(kidney, data, *, alternate_htt_panel=False):
    _, slides, labels = load_panel(kidney, labels_only=True)
    for slide in np.unique(slides):
        yield "KidneyAI", slide, VALUES[labels[slides == slide]], 40, 8, .1, 5

    htt = load_htt_ratings(data)
    for slide, rows in htt.query("modalityID == 'camic'").groupby("WSI"):
        observed = rows.pivot(index="caseID", columns="readerID", values="evaluable")
        densities = rows.pivot(index="caseID", columns="readerID", values="densityTILs")
        complete_readers = observed.columns[observed.notna().all()]
        panel = complete_readers[-3:] if alternate_htt_panel else complete_readers[:3]
        if len(panel) != 3 or len(observed) != 10:
            raise ValueError(f"Unexpected FDA HTT panel: {slide}")
        eligible = observed[panel].to_numpy(dtype=bool)
        grades = densities[panel].to_numpy(dtype=float)
        if not np.isfinite(grades[eligible]).all() or not ((grades[eligible] >= 0) & (grades[eligible] <= 100)).all():
            raise ValueError(f"Invalid sTIL density: {slide}")
        y = np.stack((np.where(eligible, grades / 100, 0), eligible), axis=-1)
        yield "FDA_HTT", slide, y, 5, 2, .05, 2

    with ZipFile(data / "CrowdGleason_Annotations.zip") as source:
        test = pd.read_csv(source.open("test.csv"))
    readers = [f"marker{i}" for i in range(1, 8)]
    groups = test["Patch filename"].str.extract(r"^(.*)_row_\d+_column_\d+(?:\.[^.]+)?$", expand=False)
    if test["Patch filename"].duplicated().any() or groups.isna().any():
        raise ValueError("Unexpected CrowdGleason patch identifier")
    complete = test[readers].isin([0, 1, 2, 3]).all(axis=1)
    for group, rows in test.loc[complete, readers[:3]].groupby(groups[complete]):
        if len(rows) < 10:
            continue
        n = max(4, min(40, round(.4 * len(rows))))
        y = np.stack((rows.to_numpy(), np.ones(rows.shape)), axis=-1)
        yield "CrowdGleason", group, y, n, max(2, n // 5), .1, 2


def shared_slots(slots):
    return np.intersect1d(np.intersect1d(slots[0], slots[1]), slots[2])


def remaining_patterns(size, quota):
    return [schedule(size, quota, h) for h in range(quota + 1)]


def pilot_choices(pilot, patterns, remaining, quota, tolerance, required, all_eligible, pilot_eligible,
                  *, shrink=4, beta_prior=.5):
    draws, x, _, _ = pilot.shape
    numerator, denom = pilot.sum(axis=(1, 2)).T
    theta = np.divide(numerator, denom, out=np.zeros(draws), where=denom > 0)
    z = pilot[..., 0] - theta[:, None, None] * pilot[..., 1]
    centered = z - z.mean(axis=1, keepdims=True)
    cov = np.einsum("dxa,dxb->dab", centered, centered) / (x - 1)
    diagonal = np.trace(cov, axis1=1, axis2=2) * (1 / quota - 1 / remaining)
    pair_sum = (x / (x + shrink)) * sum(cov[:, a, b] for a, b in PAIRS)
    share = np.array([len(shared_slots(s)) for s in patterns])
    overlap = np.array([np.mean([len(np.intersect1d(s[a], s[b])) for a, b in PAIRS]) for s in patterns])
    lin_var = diagonal[:, None] + 2 * pair_sum[:, None] * (overlap[None, :] / quota**2 - 1 / remaining)

    # Estimated full denominator is (N / X) times the pilot denominator.
    # The residual-total variance scales with the remaining frame size squared.
    approx_denominator = (remaining + x) * denom / x
    ratio_se = np.divide(remaining * np.sqrt(np.maximum(lin_var, 0)),
                         approx_denominator[:, None],
                         out=np.full_like(lin_var, np.inf),
                         where=approx_denominator[:, None] > 0)
    score = 2 * ndtr(tolerance / np.maximum(ratio_se, 1e-14)) - 1
    already = pilot_eligible.sum(axis=1)
    need = required - already[:, None]
    if all_eligible:
        review = (share[None, :] >= need).astype(float)
    else:
        # The review forecast depends on the pilot only through its integer count.
        a = np.arange(x + 1)[:, None]
        review_table = betabinom.sf(required - a - 1, share[None, :], a + beta_prior, x - a + beta_prior)
        review = review_table[already]
    proxy = score * review
    best = proxy.argmax(axis=1)
    # Where the pilot has no gradable score or cannot inform the joint proxy,
    # the estimated probability of material breaks the tie.
    best = np.where(proxy.max(axis=1) > 0, best, review.argmax(axis=1))
    return best, score[np.arange(draws), best], review[np.arange(draws), best]


def evaluation(y, y_total, common_flag, sample, shared_index, tolerance, required, pilot=None):
    reader = np.arange(3)[None, :, None]
    sampled = y[sample, reader].sum(axis=(1, 2))
    common_count = common_flag[shared_index].sum(axis=1)
    if pilot is not None:
        observed, pilot_index = pilot
        sampled = observed.sum(axis=(1, 2)) + ((len(y) - pilot_index.shape[1]) / sample.shape[2]) * sampled
        common_count += common_flag[pilot_index].sum(axis=1)
    grade, count = sampled.T
    score = np.divide(grade, count, where=count > 0, out=np.full_like(grade, np.nan, dtype=float))
    if y_total[1] == 0:
        raise ValueError("The complete panel has no evaluable records")
    signed_error = score - y_total[0] / y_total[1]
    error = np.abs(signed_error)
    # The tolerance is inclusive. The small slack absorbs floating-point rounding.
    score_pass = error <= tolerance + 1e-12
    review_pass = common_count >= required
    return error, score_pass, review_pass, common_count, signed_error


def replay(dataset, group, y, n, x, tolerance, required, rng, draws, *, challenge=False, common_flag=None):
    size = len(y)
    quota = n - x
    assert 1 < x < n <= size
    if common_flag is None:
        common_flag = y[..., 1].all(axis=1)
    y_total = y.sum(axis=(0, 1))
    permutation = rng.random((draws, 3, size)).argsort(axis=2)
    ordinals = ORDERS[rng.integers(len(ORDERS), size=draws)]
    first = permutation[:, 0]
    choices = np.arange(draws)
    observed = y[first[:, :x]]
    patterns = remaining_patterns(size - x, quota)
    h_selected, p_score, p_review = pilot_choices(
        observed, patterns, size - x, quota, tolerance, required, dataset == "CrowdGleason",
        common_flag[first[:, :x]],
    )
    rows = []
    joint_draws = {}

    def summarize(method, measure, planned_share):
        err, score_pass, review_pass, common, signed = measure
        joint_draws[method] = score_pass & review_pass
        rows.append(dict(dataset=dataset, group=group, units=size, budget=3*n, pilot=x if method.startswith(("pilot", "adaptive")) else 0,
                         method=method, mae=np.nanmean(err) if np.isfinite(err).any() else np.nan,
                         score_bias=np.nanmean(signed) if np.isfinite(signed).any() else np.nan,
                         score_success=score_pass.mean(), joint_success=joint_draws[method].mean(),
                         review_success=review_pass.mean(), missing=np.isnan(err).mean(),
                         gradable_common=common.mean(), planned_shared=np.mean(planned_share),
                         fully_shared_rate=np.mean(np.asarray(planned_share) == n),
                         pilot_score_prediction=p_score.mean() if method == "adaptive_joint" else np.nan,
                         pilot_review_prediction=p_review.mean() if method == "adaptive_joint" else np.nan))

    independent = permutation[:, :, :n]
    masks = np.zeros((draws, 3, size), dtype=bool)
    masks[choices[:, None, None], np.arange(3)[None, :, None], independent] = True
    # All draws can share the same padded width. A sentinel points to an ineligible row.
    counts = masks.all(axis=1)
    sentinel = np.where(counts, np.arange(size), size)
    available = np.r_[common_flag, False]
    summarize("fixed_independent", evaluation(y, y_total, available, independent, sentinel, tolerance, required), counts.sum(axis=1))

    fixed_counts = {
        "spread": 0, "quarter": n//4, "three_eighths": 3*n//8, "seventeen_fortieths": 17*n//40,
        "half": n//2, "three_quarters": 3*n//4, "four_fifths": 4*n//5, "same": n,
    }
    if challenge:
        fixed_counts = {str(h): h for h in range(n + 1)}
    for name, shared in fixed_counts.items():
        slots = schedule(size, n, shared)
        sample = first[choices[:, None, None], slots[ordinals]]
        measure = evaluation(y, y_total, common_flag, sample, first[:, shared_slots(slots)], tolerance, required)
        summarize("fixed_" + name, measure, len(shared_slots(slots)))

    continuations = {"spread": 0, "quarter": quota//4, "half": quota//2, "same": quota}
    if challenge:
        continuations = {str(h): h for h in range(quota + 1)}
    for name, shared in continuations.items():
        slots = patterns[shared]
        sample = first[choices[:, None, None], x + slots[ordinals]]
        measure = evaluation(y, y_total, common_flag, sample, first[:, x + shared_slots(slots)], tolerance, required,
                             (observed, first[:, :x]))
        summarize("pilot_" + name, measure, x + len(shared_slots(slots)))

    result = (np.empty(draws), np.empty(draws, dtype=bool), np.empty(draws, dtype=bool), np.empty(draws), np.empty(draws))
    total_shared = np.empty(draws, dtype=int)
    for h, slots in enumerate(patterns):
        mask = h_selected == h
        if not mask.any():
            continue
        actual = shared_slots(slots)
        sample = first[choices[mask, None, None], x + slots[ordinals[mask]]]
        shifted = first[mask][:, x + actual]
        measures = evaluation(y, y_total, common_flag, sample, shifted, tolerance, required,
                              (observed[mask], first[mask, :x]))
        for output, m in zip(result, measures, strict=True):
            output[mask] = m
        total_shared[mask] = x + len(actual)
    summarize("adaptive_joint", result, total_shared)

    if challenge:
        # Stop common reads once enough eligible material is observed. Always
        # keep one follow-up read, so the conditional totals can be estimated.
        for name, cap in (("sequential_review", n - 1), ("sequential_review_full_cap", n)):
            shared_before_followup = common_flag[first[:, :cap]].cumsum(axis=1)
            stops = np.minimum((shared_before_followup < required).sum(axis=1) + 1, cap)
            sequential = (np.empty(draws), np.empty(draws, dtype=bool),
                          np.empty(draws, dtype=bool), np.empty(draws), np.empty(draws))
            total_common = stops.copy()
            for h in np.unique(stops):
                mask = stops == h
                common = first[mask, :h]
                if h == n:
                    # Use the common-sample ratio when the final shared read
                    # consumes the budget. This arm does not have guaranteed
                    # conditionally unbiased totals in every stopping branch.
                    remaining = np.repeat(common[:, None, :], 3, axis=1)
                    measure = evaluation(y, y_total, common_flag, remaining, common, tolerance, required)
                else:
                    slots = schedule(size - h, n - h, 0)
                    remaining = first[choices[mask, None, None], h + slots[ordinals[mask]]]
                    actual = shared_slots(slots)
                    total_common[mask] += len(actual)
                    common_remaining = first[mask][:, h + actual]
                    measure = evaluation(y, y_total, common_flag, remaining, common_remaining,
                                         tolerance, required, (y[common], common))
                for output, m in zip(sequential, measure, strict=True):
                    output[mask] = m
            summarize(name, sequential, total_common)
            rows[-1]["pilot"] = stops.mean()
    for row in rows:
        differences = joint_draws["adaptive_joint"].astype(np.int8) - joint_draws[row["method"]]
        row["paired_difference_mcse"] = differences.std(ddof=1) / np.sqrt(draws)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kidney", type=Path, default=Path("Dataset"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=261120)
    parser.add_argument("--max-per-dataset", type=int)
    parser.add_argument("--challenge", action="store_true")
    parser.add_argument("--kidney-pilot", type=int, default=8)
    parser.add_argument("--htt-pilot", type=int, default=2)
    parser.add_argument("--dataset", choices=("KidneyAI", "FDA_HTT", "CrowdGleason"))
    parser.add_argument("--alternate-htt-panel", action="store_true")
    args = parser.parse_args()
    if args.draws < 2:
        parser.error("--draws must be at least two for Monte Carlo standard errors")
    rows = []
    counts = {}
    for i, (dataset, group, y, n, x, tolerance, required) in enumerate(panels(args.kidney, args.data, alternate_htt_panel=args.alternate_htt_panel)):
        if args.dataset and dataset != args.dataset:
            continue
        x = {"KidneyAI": args.kidney_pilot, "FDA_HTT": args.htt_pilot}.get(dataset, x)
        count = counts.get(dataset, 0)
        if args.max_per_dataset is not None and count >= args.max_per_dataset:
            continue
        counts[dataset] = count + 1
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, i]))
        rows.extend(replay(dataset, group, y, n, x, tolerance, required, rng, args.draws, challenge=args.challenge))
    by_group = pd.DataFrame(rows)
    summary = by_group.groupby(["dataset", "method"]).mean(numeric_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    by_group.to_csv(args.output / "by_group.csv", index=False)
    summary.to_csv(args.output / "summary.csv")
    comparisons = []
    for dataset, data in by_group.groupby("dataset"):
        wide = data.pivot(index="group", columns="method", values="joint_success")
        boot = np.random.default_rng(args.seed + 1).integers(len(wide), size=(10000, len(wide)))
        mcse = data.pivot(index="group", columns="method", values="paired_difference_mcse")
        references = wide.drop(columns="adaptive_joint")
        if args.challenge and dataset != "CrowdGleason":
            for family in ("fixed", "pilot"):
                stable = wide[[c for c in wide if c.startswith(family + "_")]]
                objective = stable.sum(axis=0).to_numpy()[None, :] - stable.to_numpy()
                best_other = objective.argmax(axis=1)
                references["loo_" + family] = stable.to_numpy()[np.arange(len(stable)), best_other]
        for method in references:
            difference = (wide.adaptive_joint - references[method]).dropna().to_numpy()
            if len(difference) < len(wide):
                continue
            lo, hi = np.quantile(difference[boot].mean(axis=1), [.025, .975])
            paired_mcse = np.linalg.norm(mcse[method].dropna()) / len(difference) if method in mcse else np.nan
            comparisons.append(dict(dataset=dataset, reference=method, groups=len(difference),
                                    gain=difference.mean(), low=lo, high=hi, paired_mcse=paired_mcse))
    pd.DataFrame(comparisons).to_csv(args.output / "paired_joint_differences.csv", index=False)
    print(counts)
    print(summary[["mae", "score_success", "review_success", "joint_success"]].round(4).to_string())


if __name__ == "__main__":
    main()
