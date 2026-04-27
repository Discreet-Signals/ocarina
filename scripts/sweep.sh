#!/usr/bin/env bash
# Reclaim disk from the workspace's `target/` directory.
#
# Cargo never garbage-collects stale dep artifacts (each rebuild of a
# crate produces a new rlib hash; old ones linger). For a workspace
# that links a big crate like lapce-app this can pile up tens of GB
# in days.
#
# `cargo sweep --time 7` deletes everything in `target/` not touched
# in the last 7 days. Today's incremental cache stays; yesterday-and-
# older crud goes. Run it whenever `du -sh target` looks scary, or
# wire it into a weekly cron / launchd job.
#
# Aggressive flag: `./scripts/sweep.sh --aggressive` drops to 1 day,
# which keeps only the active build state.

set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v cargo-sweep >/dev/null 2>&1; then
    echo "cargo-sweep not found. Install with: cargo install cargo-sweep"
    exit 1
fi

DAYS=7
if [[ "${1:-}" == "--aggressive" ]]; then
    DAYS=1
fi

echo "==> Pre-sweep target size:"
du -sh target 2>/dev/null || echo "   (no target/ yet)"

echo "==> Sweeping artifacts older than ${DAYS} days..."
cargo sweep --time "${DAYS}"

echo "==> Sweeping artifacts from older Rust toolchains..."
cargo sweep --installed

echo "==> Post-sweep target size:"
du -sh target 2>/dev/null || echo "   (target/ removed)"
