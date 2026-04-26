//! Ocarina editor — Lapce-based IDE for the Ocarina game engine.
//!
//! This binary is intentionally tiny. The real editor is the Lapce
//! fork in `third_party/lapce/`, which we customize in-tree (Viewport
//! panel, wgpu device exposure, Ocarina default layout). This crate
//! is the workspace-member entry point so `cargo run -p ocarina-editor`
//! works from the project root, and so future Ocarina-specific shell
//! customizations have a place to live above the Lapce layer.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    lapce_app::app::launch();
}
