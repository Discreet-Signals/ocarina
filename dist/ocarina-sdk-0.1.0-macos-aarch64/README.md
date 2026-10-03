# Ocarina SDK 0.1.0

This bundle contains the `ocarina` CLI and the engine source it
needs to build user games.

## Install

```bash
# Untar wherever you want -- conventional spots are ~/.ocarina/sdk/
# or /usr/local/share/ocarina-sdk/. The CLI finds the engine source
# relative to its own binary, so the path doesn't matter.
tar xzf ocarina-sdk-0.1.0-macos-aarch64.tar.gz -C ~/.ocarina/

# Add bin/ to PATH (or symlink ocarina into /usr/local/bin/).
export PATH="$HOME/.ocarina/sdk/bin:$PATH"
```

## Requirements

- Rust 1.95+ (run `rustup-init` if you don't have it).

## Build a project

```bash
cd path/to/your-game
ocarina build --release
```

## Editor

The Ocarina editor is shipped separately as `Ocarina.app`. The
editor invokes this same CLI under the hood; on first launch it
looks for the SDK via, in order: `$OCARINA_SDK`,
`~/.ocarina/sdk/`, then its own bundle's `Resources/sdk/`.
