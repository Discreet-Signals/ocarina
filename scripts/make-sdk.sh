#!/usr/bin/env bash
# Build the Ocarina SDK tarball.
#
# Output: dist/ocarina-sdk-<version>-<triple>/ + .tar.gz
#
# The SDK is the headless side of distribution: it carries the
# `ocarina` CLI binary plus the engine workspace source that the
# CLI compiles user games against. Build servers and CI machines
# install only this; no editor required.
#
# Layout produced:
#   dist/ocarina-sdk-<v>-<triple>/
#     bin/ocarina           release CLI binary
#     engine/Cargo.toml     stripped workspace manifest
#     engine/crates/...     runtime crate sources (path-deps)
#     VERSION
#     README.md
#
# Usage:
#   ./scripts/make-sdk.sh             # builds + tars
#   ./scripts/make-sdk.sh --no-tar    # just stages the directory
#
# Cross-compiling: pass --target <triple> to forward to cargo. The
# triple ends up in the output filename. Without it, builds for the
# host machine.

set -euo pipefail

cd "$(dirname "$0")/.."
WORKSPACE_ROOT="$(pwd)"

TAR=1
CARGO_TARGET=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-tar) TAR=0 ;;
        --target) CARGO_TARGET="$2"; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

VERSION=$(grep -E '^version[[:space:]]*=' Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
if [[ -z "${VERSION}" ]]; then
    echo "could not parse version from Cargo.toml" >&2
    exit 1
fi

# Default target triple from `rustc -vV` if --target wasn't passed.
if [[ -z "${CARGO_TARGET}" ]]; then
    HOST_TRIPLE=$(rustc -vV | awk '/^host:/ { print $2 }')
    TARGET_TRIPLE="${HOST_TRIPLE}"
    BUILD_DIR="target/release"
else
    TARGET_TRIPLE="${CARGO_TARGET}"
    BUILD_DIR="target/${CARGO_TARGET}/release"
fi

# Friendlier name for end users: collapse triples like
# x86_64-apple-darwin -> macos-x86_64. Falls through to the raw
# triple for less common targets.
case "${TARGET_TRIPLE}" in
    x86_64-apple-darwin)        DIST_TRIPLE="macos-x86_64" ;;
    aarch64-apple-darwin)       DIST_TRIPLE="macos-aarch64" ;;
    x86_64-unknown-linux-gnu)   DIST_TRIPLE="linux-x86_64" ;;
    aarch64-unknown-linux-gnu)  DIST_TRIPLE="linux-aarch64" ;;
    x86_64-pc-windows-msvc)     DIST_TRIPLE="windows-x86_64" ;;
    *)                           DIST_TRIPLE="${TARGET_TRIPLE}" ;;
esac

OUT_NAME="ocarina-sdk-${VERSION}-${DIST_TRIPLE}"
OUT="dist/${OUT_NAME}"

echo "==> Building ocarina-cli (release, ${TARGET_TRIPLE})"
if [[ -n "${CARGO_TARGET}" ]]; then
    cargo build --release -p ocarina-cli --target "${CARGO_TARGET}"
else
    cargo build --release -p ocarina-cli
fi

CLI_BIN="${BUILD_DIR}/ocarina"
if [[ "${TARGET_TRIPLE}" == *windows* ]]; then
    CLI_BIN="${BUILD_DIR}/ocarina.exe"
fi
if [[ ! -x "${CLI_BIN}" ]]; then
    echo "expected CLI at ${CLI_BIN}, not found" >&2
    exit 1
fi

echo "==> Staging ${OUT}/"
rm -rf "${OUT}"
mkdir -p "${OUT}/bin" "${OUT}/engine/crates"

cp "${CLI_BIN}" "${OUT}/bin/"

# Runtime-relevant crates. Anything outside this list is editor-only
# (ocarina-editor, ocarina-viewport) or shipped as a binary
# (ocarina-cli) -- the host crate that compiles a user's game never
# pulls them.
RUNTIME_CRATES=(
    ocarina-bake
    ocarina-core
    ocarina-derive
    ocarina-gameplay-api
    ocarina-manifest
    ocarina-physics
    ocarina-render
    ocarina-rhi
    ocarina-rhi-null
    ocarina-rhi-wgpu
    ocarina-runtime
    ocarina-scene
    ocarina-schema
    ocarina-shader
)
for c in "${RUNTIME_CRATES[@]}"; do
    if [[ ! -d "crates/${c}" ]]; then
        echo "missing runtime crate dir: crates/${c}" >&2
        exit 1
    fi
    # `cp -r` excluding the per-crate `target/` if any was built in
    # tree (rare, but a stale `cargo build` inside a crate dir would
    # otherwise bloat the tarball).
    rsync -a --exclude='target/' --exclude='Cargo.lock' \
        "crates/${c}/" "${OUT}/engine/crates/${c}/"
done

cp scripts/sdk-Cargo.toml "${OUT}/engine/Cargo.toml"

echo "${VERSION}" > "${OUT}/VERSION"

cat > "${OUT}/README.md" <<EOF
# Ocarina SDK ${VERSION}

This bundle contains the \`ocarina\` CLI and the engine source it
needs to build user games.

## Install

\`\`\`bash
# Untar wherever you want -- conventional spots are ~/.ocarina/sdk/
# or /usr/local/share/ocarina-sdk/. The CLI finds the engine source
# relative to its own binary, so the path doesn't matter.
tar xzf ${OUT_NAME}.tar.gz -C ~/.ocarina/

# Add bin/ to PATH (or symlink ocarina into /usr/local/bin/).
export PATH="\$HOME/.ocarina/sdk/bin:\$PATH"
\`\`\`

## Requirements

- Rust 1.95+ (run \`rustup-init\` if you don't have it).

## Build a project

\`\`\`bash
cd path/to/your-game
ocarina build --release
\`\`\`

## Editor

The Ocarina editor is shipped separately as \`Ocarina.app\`. The
editor invokes this same CLI under the hood; on first launch it
looks for the SDK via, in order: \`\$OCARINA_SDK\`,
\`~/.ocarina/sdk/\`, then its own bundle's \`Resources/sdk/\`.
EOF

# Sanity: try invoking the staged CLI's --version. Catches obvious
# packaging bugs (binary missing libs, wrong arch).
if [[ -z "${CARGO_TARGET}" ]] || [[ "${CARGO_TARGET}" == "${HOST_TRIPLE:-}" ]]; then
    echo "==> Smoke-testing staged CLI"
    "${OUT}/bin/ocarina" --version
fi

if [[ "${TAR}" -eq 1 ]]; then
    ARCHIVE="dist/${OUT_NAME}.tar.gz"
    echo "==> Creating ${ARCHIVE}"
    tar czf "${ARCHIVE}" -C dist "${OUT_NAME}"
    echo "==> Done: ${ARCHIVE} ($(du -sh "${ARCHIVE}" | cut -f1))"
else
    echo "==> Done: ${OUT}/ ($(du -sh "${OUT}" | cut -f1))"
fi
