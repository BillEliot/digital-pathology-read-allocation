"""Replay FDA HTT reading policies against a common finite-panel reference."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from numba import njit, prange, set_num_threads


INPUT_SHA256 = "0bd851c039a43944febed68bb0e5d2624dcef83acc064387d1ac66e07f1f166b"
REGIONS = 10
SCORE_TOLERANCE = 5.0
TARGET_REGIONS = 2
MIN_BUDGET = 8


@dataclass(frozen=True)
class Frame:
    slide: str
    batch: str
    readers: tuple[str, ...]
    scores: np.ndarray

    @property
    def size(self):
        return len(self.readers)


def frames_from_zip(path):
    with ZipFile(path) as archive:
        file, = [n for n in archive.namelist() if n.endswith("/inst/extdata/pilotHTT_RST.csv")]
        assert sha256(archive.read(file)).hexdigest() == INPUT_SHA256
        data = pd.read_csv(archive.open(file))
    frames = []
    routine = data.loc[data.modalityID.eq("camic")]
    assert len(routine) == 5100 and routine.WSI.nunique() == 64
    for slide, rows in routine.groupby("WSI", sort=True):
        batch, = rows.batch.unique()
        eligible = rows.pivot(index="readerID", columns="caseID", values="evaluable")
        scores = rows.pivot(index="readerID", columns="caseID", values="densityTILs").reindex_like(eligible)
        assert eligible.shape[1] == REGIONS
        n = scores.to_numpy(dtype=float)
        e = eligible.eq(True).to_numpy()
        complete = eligible.notna().all(axis=1).to_numpy() & ((~e) | (np.isfinite(n) & (n >= 0) & (n <= 100))).all(axis=1)
        frames.append(Frame(slide, batch, tuple(eligible.index[complete]), np.where(e[complete], n[complete], -1.0)))
    assert [sum(f.size >= n for f in frames) for n in (5, 6, 7, 10)] == [64, 56, 37, 8]
    return frames


def geometry(r, budget, h):
    """For each reader, return its cyclic suffix start and length and masks by position."""
    quotas = np.full(r, budget // r, dtype=np.int64)
    quotas[:budget % r] += 1
    picks = quotas - h
    suffix = REGIONS - h
    starts = np.full(r, h, dtype=np.int64)
    masks = np.zeros(REGIONS, dtype=np.int64)
    masks[:h] = (1 << r) - 1
    if suffix > 0:
        starts[1:] += np.cumsum(picks[:-1]) % suffix
        for reader in range(r):
            locs = h + (starts[reader] - h + np.arange(picks[reader])) % suffix
            masks[locs] |= 1 << reader
    assert sum(int(mask).bit_count() for mask in masks) == budget
    return starts, picks, masks


def design_grid(r):
    rows, starts, lengths, masks, h_values, lookup = [], [], [], [], [], []
    for budget in range(MIN_BUDGET, REGIONS * r + 1):
        if budget < r:
            continue
        cap = budget // r
        offset = len(h_values)
        for h in range(cap + 1):
            s, l, m = geometry(r, budget, h)
            starts.append(s)
            lengths.append(l)
            masks.append(m)
            h_values.append(h)
            rows.append((r, budget, "fixed", h))
            lookup.append((offset, h, 0))
        for policy, limit in (("sequential", cap), ("sequential_spare", max(0, cap - 1))):
            rows.append((r, budget, policy, limit))
            lookup.append((offset, limit, 1))
    return (rows, *(np.array(a, dtype=np.int64) for a in (starts, lengths, masks, h_values, lookup)))


def draws_for(frame, draws, seed, reserve_reference=False):
    slide_key = int(sha256(frame.slide.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(np.random.SeedSequence([seed, slide_key]))
    readers = np.argsort(rng.random((draws, frame.size)), axis=1)
    regions = np.argsort(rng.random((draws, REGIONS)), axis=1)
    raw = frame.scores
    reference = raw[raw >= 0].mean() if (raw >= 0).any() else np.nan
    refs = np.full((draws, 1), reference)
    if reserve_reference:
        separated = raw[readers[:, :3]]
        present = separated >= 0
        total = np.where(present, separated, 0).sum(axis=(1, 2))
        counts = present.sum(axis=(1, 2))
        separate_ref = np.divide(total, counts, out=np.full(draws, np.nan), where=counts > 0)
        refs = np.column_stack((refs, separate_ref))
        readers = readers[:, 3:]
    return readers, regions, refs


@njit(parallel=True)
def assignment_outcomes(scores, readers, regions, refs, starts, lengths, masks, hs, lookup):
    draws, r = readers.shape
    n = regions.shape[1]
    configs, geometries, nrefs = len(lookup), len(hs), refs.shape[1]
    out = np.zeros((draws, configs, nrefs, 2, 3), dtype=np.uint8)
    count_bits = np.zeros(1 << r, dtype=np.int64)
    for i in range(1, len(count_bits)):
        count_bits[i] = count_bits[i >> 1] + (i & 1)
    for d in prange(draws):
        py = np.zeros((r, n + 1))
        pe = np.zeros((r, n + 1))
        availability = np.zeros(n, dtype=np.int64)
        for reader in range(r):
            for i in range(n):
                value = scores[readers[d, reader], regions[d, i]]
                py[reader, i + 1] = py[reader, i] + max(0.0, value)
                pe[reader, i + 1] = pe[reader, i] + (value >= 0.0)
                if value >= 0:
                    availability[i] |= 1 << reader
        earliest = np.full(2, n + 1)
        common_count = np.zeros(2, dtype=np.int64)
        for i in range(n):
            k = count_bits[availability[i]]
            for q in range(2):
                if k >= q + 2:
                    common_count[q] += 1
                    if common_count[q] == TARGET_REGIONS:
                        earliest[q] = i + 1
        a_score = np.full(geometries, np.nan)
        a_material = np.zeros((geometries, 2), dtype=np.uint8)
        for g in range(geometries):
            h = hs[g]
            expand = lengths[g, r - 1] > 0
            ysum, esum = 0.0, 0.0
            for reader in range(r):
                start, count = starts[g, reader], lengths[g, reader]
                end = min(n, start + count)
                y = py[reader, end] - py[reader, start]
                e = pe[reader, end] - pe[reader, start]
                if start + count > n:
                    wrap = h + start + count - n
                    y += py[reader, wrap] - py[reader, h]
                    e += pe[reader, wrap] - pe[reader, h]
                weight = (n - h) / count if expand else 1.0
                ysum += py[reader, h] + weight * y
                esum += pe[reader, h] + weight * e
            if esum > 0:
                a_score[g] = ysum / esum
            count_two, count_three = 0, 0
            for i in range(n):
                k = count_bits[availability[i] & masks[g, i]]
                count_two += k >= 2
                count_three += k >= 3
            a_material[g, 0] = count_two >= TARGET_REGIONS
            a_material[g, 1] = count_three >= TARGET_REGIONS
        for c in range(configs):
            offset, cap, sequential = lookup[c]
            for q in range(2):
                if r < q + 2:
                    continue
                h = min(earliest[q], cap) if sequential else cap
                g = offset + h
                m = a_material[g, q]
                for ref in range(nrefs):
                    s = abs(a_score[g] - refs[d, ref]) <= SCORE_TOLERANCE + 1e-12
                    out[d, c, ref, q, 0] = s
                    out[d, c, ref, q, 1] = m
                    out[d, c, ref, q, 2] = s and m
    return out


def run(inputs, output, draws, seed, threads, limit=None):
    set_num_threads(threads)
    frames = frames_from_zip(inputs)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(slide=f.slide, batch=f.batch, complete_readers=f.size, full_pilot_reads=REGIONS*f.size)
                  for f in frames]).to_csv(output / "frames.csv", index=False)
    grids = {r: design_grid(r) for r in range(2, 8)}
    rows = []
    for idx, frame in enumerate(frames[:limit]):
        for separated in ([False, True] if frame.size >= 10 else [False]):
            readers, regions, refs = draws_for(frame, draws, seed, reserve_reference=separated)
            names = ("all_reference_paired", "independent3") if separated else ("all_reference",)
            for r in range(2, min(7, readers.shape[1]) + 1):
                configs, *grid = grids[r]
                outcomes = assignment_outcomes(frame.scores, readers[:, :r], regions, refs, *grid)
                totals = outcomes.sum(axis=0)
                for i, (nr, budget, policy, common) in enumerate(configs):
                    for j, ref_name in enumerate(names):
                        for q in range(2, min(3, r) + 1):
                            score, material, joint = totals[i, j, q - 2]
                            rows.append((frame.slide, ref_name, nr, budget, policy, common, q, draws, score, material, joint))
        if (idx + 1) % 8 == 0:
            print(f"Replayed {idx + 1} slides", flush=True)
    cols = "slide reference readers budget policy common quorum draws score material joint".split()
    pd.DataFrame(rows, columns=cols).to_csv(output / "replay_counts.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    print(f"Archived {len(rows):,} slide × policy records, {draws:,} draws per record", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--draws", type=int, default=4096)
    p.add_argument("--seed", type=int, default=261170)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    run(args.inputs, args.output, args.draws, args.seed, args.threads, args.limit)
