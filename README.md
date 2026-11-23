# Digital Pathology Read Allocation

Replay scripts for the study *Adaptive Allocation of Reads for Slide Scoring and Joint Review in Digital Pathology*.

## Setup

Use Python 3.12.10 and install the pinned dependencies.

```sh
python -m pip install -r requirements.txt
```

## Data

- **KidneyAI v1** — Place `expert1.pickle`, `expert2.pickle`, and `expert3.pickle` in the KidneyAI input directory.
- **FDA High Throughput Truthing** — Save the ZIP for commit `78252d93c9f77dcd2fc866068bfb69a0df47b749` as `FDA_HTT.zip` in the source data directory.
- **CrowdGleason v1** — Save the released `Annotations.zip` as `CrowdGleason_Annotations.zip` in the same source data directory.

## Run the main replay

From the repository root, replace the input paths with your local paths.

```sh
python experiments/sequential_phase_revision.py --kidney PATH_TO_KIDNEYAI --data PATH_TO_SOURCE_DIR --output runs/phase
```

Add `--smoke` for a quick check. The FDA HTT staffing and pilot-study scripts are in `fda_planning/scripts/`.
