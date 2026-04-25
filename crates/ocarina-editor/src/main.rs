//! Ocarina editor entry point.

mod app;
mod camera;
mod gizmo;
mod mesh;
mod panes;
mod plugin;
mod project;
mod selection;
mod terminal;
mod transform_io;
mod viewport;

// Force the linker to retain `ocarina-core` so its `#[derive(Component)]`
// submissions (Transform, BasicMesh, Camera, Name) reach the `inventory`
// section of the final binary. Without this use the editor's scene
// loader sees zero registered components.
use ocarina_core as _;

use app::OcarinaApp;

fn main() -> eframe::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([800.0, 500.0])
            .with_title("Ocarina"),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    eframe::run_native(
        "Ocarina",
        options,
        Box::new(|cc| Ok(Box::new(OcarinaApp::new(cc)))),
    )
}
