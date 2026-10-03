//! Auto-generated host binary. Do not edit by hand.
//!
//! `Ocarina.toml`'s `[project]` table named this crate; the
//! `[dependencies]` table populated `Cargo.toml`. This file just
//! routes through `ocarina_runtime::windowed::run`.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

// `build.rs` generates one `pub mod foo;` per .rs in the project's
// `scripts/` folder, with `#[path]` attrs pointing back at the real
// file. Including it makes each script's `#[derive(OcarinaScript)]`
// register at link time.
include!(concat!(env!("OUT_DIR"), "/scripts.rs"));

// Force-link every engine subsystem the baseline ships. Without
// these `use _ as _;` lines the linker drops the crates entirely
// (no symbols referenced) and their `inventory::submit!` component
// registrations never run -- `[collider]` / `[rigid_body]` etc.
// would then come back from the loader as "unknown components."
use ocarina_physics as _;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .init();

    // The CLI sets OCARINA_PROJECT to the project's absolute path
    // when invoking `cargo run`. Falling back to the parent of
    // `<project>/.ocarina/host/` matches `ocarina build`'s layout
    // when someone runs the produced binary directly.
    let project = std::env::var_os("OCARINA_PROJECT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(|p| p.parent())
                .map(std::path::PathBuf::from)
                .expect("ocarina-host located at <project>/.ocarina/host")
        });

    ocarina_runtime::windowed::run(&project)
}
