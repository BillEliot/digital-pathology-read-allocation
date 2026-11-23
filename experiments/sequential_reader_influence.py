"""Estimate FDA HTT reader influence from the archived paired panel replays."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sequential_comparator_revision import reader_sensitivity


def influence(panels):
    fda = panels.query("dataset == 'FDA_HTT'").copy()
    target = fda.assign(gain=fda.sequential - fda.fixed)
    readers = sorted({r for names in target.panel.str.split("/") for r in names})
    assert len(readers) == 19
    rows, removed, distributions = [], [], []
    for estimator, block in target.groupby("estimator"):
        assert len(block) == 6715 and block.group.nunique() == 64
        assert not block.duplicated(["group", "panel"]).any()
        assert np.isfinite(block[["gain", "fixed", "sequential"]].to_numpy()).all()
        frame_means = block.groupby("group")[["fixed", "sequential", "gain"]].mean().mean()
        ids = block.panel.str.split("/")
        gains = []
        for reader in readers:
            keep = ~ids.map(lambda names: reader in names)
            retained = block.loc[keep]
            assert retained.group.nunique() == block.group.nunique()
            net = retained.groupby("group").gain.mean().mean()
            gains.append(net)
            removed.append(dict(estimator=estimator, omitted_reader=reader, remaining_triples=len(retained),
                                remaining_slides=retained.group.nunique(), gain=net))
        mapped = block.rename(columns={"fixed": "fixed_joint", "sequential": "seq_pooled_joint"})
        weighted = reader_sensitivity(mapped)
        assert np.isfinite(weighted.to_numpy()).all() and len(weighted) == 10_000
        distributions.append(weighted.assign(estimator=estimator, draw=np.arange(len(weighted))))
        quantiles = weighted.quantile([.025, .975])
        rows.append(dict(estimator=estimator, slides=block.group.nunique(), panels=len(block),
                         readers=len(readers), **frame_means.to_dict(),
                         omitted_low=min(gains), omitted_high=max(gains),
                         reader_low=quantiles.loc[.025, "reader_weighted"],
                         reader_high=quantiles.loc[.975, "reader_weighted"],
                         reader_slide_low=quantiles.loc[.025, "reader_and_frame_weighted"],
                         reader_slide_high=quantiles.loc[.975, "reader_and_frame_weighted"]))
    return pd.DataFrame(rows), pd.DataFrame(removed), pd.concat(distributions, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in zip(("reader_influence_summary", "reader_omissions", "reader_reweighting"),
                          influence(pd.read_csv(args.panels)), strict=True):
        data.to_csv(args.output / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
