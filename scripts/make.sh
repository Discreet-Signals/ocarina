#!/usr/bin/env bash
# Build both shipping artifacts in one go:
#
#   dist/ocarina-sdk-<version>-<triple>.tar.gz   (headless SDK)
#   dist/Ocarina.app                              (macOS editor bundle)
#
# Each step is a thin wrapper over the dedicated script -- run those
# directly if you need their flags (`--no-tar`, `--target <triple>`,
# `--debug`).
#
# SDK first so a fresh `target/release/ocarina` is sitting in the
# cache when the editor build kicks off; the editor doesn't depend
# on the CLI but cargo's incremental cache is happier this way.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> [1/2] Building Ocarina SDK"
./scripts/make-sdk.sh

echo
echo "==> [2/2] Building Ocarina.app"
./scripts/make-app.sh

echo
echo "==> Done."
ls -lh dist/
