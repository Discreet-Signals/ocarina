//! Host binary for an Ocarina game.
//!
//! This file is engine plumbing -- the dev never touches it once the
//! project is scaffolded. All gameplay lives in:
//!
//! - `project/*.toml`    -- entities, components, `[[script]]` attaches.
//! - `project/scripts/*.rs` -- the scripts themselves. `build.rs` walks
//!   that folder and compiles each `.rs` into a top-level module of
//!   this binary, so `#[derive(OcarinaScript)]` registrations light up
//!   at link time without main.rs needing to know about them.
//! - `project/assets/...` -- meshes, textures, etc., referenced by
//!   `AssetPath` fields in components.
//!
//! Build:  `cargo build --release -p hello-game`
//! Run:    `cargo run -p hello-game`  (or double-click the built binary)
//!
//! `OCARINA_PROJECT=/path/to/another/project ./hello-game` overrides
//! the default project location.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

// Pull in every script the build script discovered. Each becomes a
// top-level module here so its `inventory::submit!` runs.
include!(concat!(env!("OUT_DIR"), "/scripts.rs"));

// Force the linker to retain `ocarina-physics`'s component
// submissions (Collider / RigidBody / Velocity) so `[collider]` /
// `[rigid_body]` TOML sections are recognised by the loader.
use ocarina_physics as _;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let project = std::env::var_os("OCARINA_PROJECT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("project")
        });

    ocarina_runtime::windowed::run(&project)
}
