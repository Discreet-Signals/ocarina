#!/usr/bin/env bash
# Build the Ocarina editor as a macOS .app bundle.
#
# Output: dist/Ocarina.app
#
# The .app contains only the editor binary. The editor finds the
# SDK at runtime via $OCARINA_SDK, ~/.ocarina/sdk/, etc. (see
# `palette.rs::ocarina_binary_path`). For a self-contained user
# install, ship the .app + the SDK tarball and let the user untar
# the SDK to ~/.ocarina/sdk/.
#
# Usage:
#   ./scripts/make-app.sh             # release build, default
#   ./scripts/make-app.sh --debug     # debug build (faster iteration)
#
# Code-signing: not done here. For a notarized release the .app
# needs `codesign` + `xcrun notarytool submit`; that lives in a
# separate signing pipeline once you have a Developer ID cert.

set -euo pipefail

cd "$(dirname "$0")/.."

PROFILE="release"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)   PROFILE="dev" ;;
        --release) PROFILE="release" ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "make-app.sh produces a macOS .app bundle; run on macOS." >&2
    exit 1
fi

VERSION=$(grep -E '^version[[:space:]]*=' Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
if [[ -z "${VERSION}" ]]; then
    echo "could not parse version from Cargo.toml" >&2
    exit 1
fi

TARGET_DIR="target"
if [[ "${PROFILE}" == "dev" ]]; then
    BUILD_DIR="${TARGET_DIR}/debug"
    CARGO_PROFILE_FLAG=""
else
    BUILD_DIR="${TARGET_DIR}/release"
    CARGO_PROFILE_FLAG="--release"
fi

echo "==> Building ocarina-editor (${PROFILE})"
if [[ -n "${CARGO_PROFILE_FLAG}" ]]; then
    cargo build "${CARGO_PROFILE_FLAG}" -p ocarina-editor
else
    cargo build -p ocarina-editor
fi

EDITOR_BIN="${BUILD_DIR}/ocarina-editor"
if [[ ! -x "${EDITOR_BIN}" ]]; then
    echo "expected editor at ${EDITOR_BIN}, not found" >&2
    exit 1
fi

APP="dist/Ocarina.app"
echo "==> Staging ${APP}/"
rm -rf "${APP}"
mkdir -p "${APP}/Contents/MacOS" "${APP}/Contents/Resources"

# macOS convention: the executable in Contents/MacOS/ matches the
# CFBundleExecutable in Info.plist, which is the same string the
# Finder shows in the Dock. So `ocarina-editor` becomes `Ocarina`.
cp "${EDITOR_BIN}" "${APP}/Contents/MacOS/Ocarina"

cat > "${APP}/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>Ocarina</string>
    <key>CFBundleIdentifier</key>
    <string>com.discreetsignals.ocarina</string>
    <key>CFBundleName</key>
    <string>Ocarina</string>
    <key>CFBundleDisplayName</key>
    <string>Ocarina</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
    <key>CFBundleDocumentTypes</key>
    <array>
        <dict>
            <key>CFBundleTypeName</key>
            <string>Ocarina Project</string>
            <key>CFBundleTypeRole</key>
            <string>Editor</string>
            <key>LSItemContentTypes</key>
            <array>
                <string>com.discreetsignals.ocarina.project</string>
            </array>
        </dict>
    </array>
</dict>
</plist>
EOF

# Strip the binary (release builds already do this via Cargo.toml,
# but harmless on debug). Skipped if `strip` isn't available.
if [[ "${PROFILE}" == "release" ]] && command -v strip >/dev/null 2>&1; then
    strip "${APP}/Contents/MacOS/Ocarina" || true
fi

# Ad-hoc signature. Without it, macOS Gatekeeper still bounces the
# .app on first launch but the user can right-click -> Open to
# bypass. Drop the existing signature first if any (changes to the
# Mach-O after signing invalidate it).
if command -v codesign >/dev/null 2>&1; then
    codesign --remove-signature "${APP}" 2>/dev/null || true
    codesign --force --deep --sign - "${APP}" || true
fi

SIZE=$(du -sh "${APP}" | cut -f1)
echo "==> Done: ${APP} (${SIZE})"
echo
echo "Smoke-test:"
echo "  open ${APP}"
echo
echo "If macOS blocks the unsigned app:"
echo "  right-click ${APP} -> Open, confirm once."
