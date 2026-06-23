#!/usr/bin/env python
"""Look up a wandb run id by display name and print it to stdout.

Used by the HNM round driver to find the run id of a freshly-trained expert
or DAgger student (we use deterministic --run_name strings, but wandb still
assigns random run ids that downstream sbatch steps need).

Usage:
    python scripts/hnm_lookup_wandb_run.py <entity> <project> <run_name>

Prints just the run id (e.g. `7pddrm5x`) on success. On failure, exits 1
with an error message on stderr.

Picks the most-recently-created run when multiple runs share the same
display name (we re-train if a round is re-run, but the latest run is the
relevant one).
"""

from __future__ import annotations

import sys

import wandb


def main():
    if len(sys.argv) != 4:
        sys.stderr.write("usage: hnm_lookup_wandb_run.py <entity> <project> <run_name>\n")
        sys.exit(2)
    entity, project, run_name = sys.argv[1:]
    api = wandb.Api(timeout=60)
    runs = list(api.runs(f"{entity}/{project}", filters={"display_name": run_name}))
    if not runs:
        sys.stderr.write(f"hnm_lookup_wandb_run: no runs with display_name={run_name!r} in {entity}/{project}\n")
        sys.exit(1)
    runs.sort(key=lambda r: r.created_at, reverse=True)
    print(runs[0].id)


if __name__ == "__main__":
    main()
