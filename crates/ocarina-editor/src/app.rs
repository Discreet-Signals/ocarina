//! Top-level application: owns the dock state, the RHI, and the Viewport.

use std::sync::Arc;

use eframe::egui;
use egui::epaint::mutex::RwLock;
use egui_dock::{DockArea, DockState, NodeIndex, Style};

use ocarina_rhi::{Rhi, SwapchainHandle};
use ocarina_rhi_wgpu::WgpuRhi;

use crate::panes::{Pane, PaneViewer};
use crate::plugin::LoadedPlugin;
use crate::project::Project;
use crate::selection::Selection;
use crate::terminal::Terminal;
use crate::viewport::Viewport;

pub struct OcarinaApp {
    dock_state: DockState<Pane>,
    rhi: WgpuRhi,
    viewport: Viewport,
    terminal: Option<Terminal>,
    project: Option<Project>,
    plugin: Option<LoadedPlugin>,
    egui_renderer: Arc<RwLock<egui_wgpu::Renderer>>,
    selection: Selection,
    /// Path of the most recent file the file-browser routed into the
    /// shader-preview slot. Tracked so we don't re-compile when the
    /// user clicks the same shader twice.
    last_preview_target: Option<std::path::PathBuf>,
}

impl OcarinaApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Phase 1: we expect eframe's wgpu backend. The `renderer` setting in
        // `NativeOptions` pins it to wgpu, so `wgpu_render_state` is always
        // populated here.
        let render_state = cc
            .wgpu_render_state
            .as_ref()
            .expect("ocarina-editor is built against eframe's wgpu renderer");

        let rhi = WgpuRhi::from_existing(
            render_state.device.clone(),
            render_state.queue.clone(),
            &render_state.adapter,
        );

        let egui_renderer = render_state.renderer.clone();

        let mut rhi_mut = rhi;
        let viewport = Viewport::new(&mut rhi_mut).expect("Viewport::new failed");

        // Terminal is best-effort: if the PTY can't open we degrade to a
        // placeholder pane rather than refusing to launch.
        let terminal = match Terminal::new() {
            Ok(t) => Some(t),
            Err(e) => {
                log::warn!("terminal init failed: {e}");
                None
            }
        };

        // Open a project. Order of resolution:
        //   1. $OCARINA_PROJECT (absolute or relative to CWD)
        //   2. ./projects/example-scene (relative to workspace root / CWD)
        //   3. No project — file pane shows a message.
        let project = resolve_project_path().map(|p| {
            log::info!("opening project: {}", p.display());
            Project::open(p)
        });

        // Best-effort load of a compiled gameplay cdylib.
        let plugin = match crate::plugin::try_load_default() {
            Ok(p) => Some(p),
            Err(crate::plugin::LoadError::NotFound) => {
                log::info!(
                    "gameplay plugin: no dylib found (build `cargo build -p hello-gameplay` to enable)"
                );
                None
            }
            Err(e) => {
                log::warn!("gameplay plugin: {e}");
                None
            }
        };

        // Layout:
        //  +----------+----------------------+-----------+
        //  |  Files   |  Viewport / Editor   |  Terminal |
        //  |          |   (tabs)             |  (vert.)  |
        //  +----------+                      |           |
        //  |  Lens    |                      |           |
        //  +----------+----------------------+-----------+
        //
        // `split_*(parent, fraction, tabs)` returns `[old, new]`, where the
        // OLD node keeps `fraction` of the parent's area and the new node
        // takes the rest on the named side.
        let mut dock_state = DockState::new(vec![Pane::Viewport]);
        let surface = dock_state.main_surface_mut();

        // Note on `split_*` fractions: egui_dock's `fraction` is always the
        // share of the LEFT/TOP child after the split (see
        // `SplitNode.fraction`). For `split_right` / `split_below` the old
        // pane is left/top so fraction = old's share; for `split_left` /
        // `split_above` the NEW pane is left/top so fraction = new's share.
        // The per-fn docstrings in 0.19 are misleading on this point — go
        // by this convention instead.

        // Terminal on the right: center (left, old) keeps 70 %, terminal
        // (right, new) gets 30 %.
        let [center, _terminal] =
            surface.split_right(NodeIndex::root(), 0.70, vec![Pane::Terminal]);

        // Narrow files column on the left: files (new, left) gets 18 % of
        // center (≈ 13 % of window) — viewport + editor tabs share the
        // remaining ~82 % on the right.
        let [center, files] = surface.split_left(center, 0.25, vec![Pane::FileBrowser]);

        // Lens beneath Files (50 / 50 — files on top, lens bottom).
        let [_files, _lens] = surface.split_below(files, 0.5, vec![Pane::LensPanel]);

        // Editor as a tab alongside Viewport in the center node.
        // `append_tab` implicitly focuses the newly-added tab, so we reset
        // the active tab back to Viewport (index 0) afterwards — the
        // viewport should be what greets you on launch.
        surface[center].append_tab(Pane::TextEditor);
        if let egui_dock::Node::Leaf(leaf) = &mut surface[center] {
            let _ = leaf.set_active_tab(egui_dock::TabIndex(0));
        }

        Self {
            dock_state,
            rhi: rhi_mut,
            viewport,
            terminal,
            project,
            plugin,
            egui_renderer,
            selection: Selection::default(),
            last_preview_target: None,
        }
    }

    /// Swap in a new project rooted at `path`. The previous project's
    /// watcher is dropped (stopping filesystem subscriptions), the scene
    /// is replaced, and the file browser will pick up the new tree on
    /// the next frame.
    pub fn open_project(&mut self, path: std::path::PathBuf) {
        log::info!("opening project: {}", path.display());
        self.project = Some(Project::open(path));
        // A new project means the previous selection is meaningless.
        self.selection.clear();
        self.last_preview_target = None;
        self.viewport.clear_preview_shader(&mut self.rhi);
    }

    /// Drive the shader-preview slot off the current file-browser
    /// selection. Called once per frame after `pump`. If the selected
    /// file is `.hlsl` and changed since last frame, recompile + swap;
    /// if it became something else, revert to the built-in shader.
    fn sync_shader_preview(&mut self) {
        let selected: Option<std::path::PathBuf> = self
            .project
            .as_ref()
            .and_then(|p| p.selected.clone())
            .filter(|p| p.is_file());
        let is_hlsl = selected
            .as_ref()
            .and_then(|p| p.extension())
            .and_then(|s| s.to_str())
            == Some("hlsl");
        match (selected, is_hlsl) {
            (Some(path), true) if Some(&path) != self.last_preview_target.as_ref() => {
                if let Err(e) = self.viewport.try_load_preview_shader(&mut self.rhi, &path) {
                    log::warn!("shader preview: {e}");
                }
                self.last_preview_target = Some(path);
            }
            (_, false) if self.last_preview_target.is_some() => {
                self.viewport.clear_preview_shader(&mut self.rhi);
                self.last_preview_target = None;
            }
            _ => {}
        }
        // Hot-reload: if the previewed file changed on disk, recompile.
        self.viewport.poll_preview_shader(&mut self.rhi);
    }

    fn prompt_open_project(&mut self) {
        let start = self
            .project
            .as_ref()
            .map(|p| p.scene.root_dir.clone())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
        let picked = rfd::FileDialog::new()
            .set_title("Open Ocarina Project")
            .set_directory(&start)
            .pick_folder();
        if let Some(path) = picked {
            self.open_project(path);
        }
    }
}

/// Figure out which directory to open as the active project.
///
/// Resolution order:
///   1. `--project <path>` / `-p <path>` CLI arg
///   2. `$OCARINA_PROJECT` env var
///   3. `projects/example-scene` / `../projects/example-scene` (convenient
///      for `cargo run -p ocarina-editor` from the workspace root)
fn resolve_project_path() -> Option<std::path::PathBuf> {
    // 1. CLI args.
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--project" | "-p" => {
                if let Some(p) = args.next() {
                    let path = std::path::PathBuf::from(&p);
                    if path.is_dir() {
                        return Some(path);
                    }
                    log::warn!("--project {p} is not a directory; falling back");
                }
            }
            _ => {}
        }
    }

    // 2. Env var.
    if let Ok(env) = std::env::var("OCARINA_PROJECT") {
        let p = std::path::PathBuf::from(&env);
        if p.is_dir() {
            return Some(p);
        }
        log::warn!("OCARINA_PROJECT={env} is not a directory; falling back");
    }

    // 3. Common relative locations, tried in order.
    for rel in ["projects/example-scene", "../projects/example-scene"] {
        let p = std::path::PathBuf::from(rel);
        if p.is_dir() {
            return Some(p);
        }
    }
    None
}

impl eframe::App for OcarinaApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        // Pump filesystem events into the scene before drawing so the
        // file tree and lens panes render the latest state.
        if let Some(project) = self.project.as_mut() {
            let pump = project.pump();
            if pump.any() {
                log::debug!(
                    "project pump: {} changed, {} removed, {} rescans, {} errors",
                    pump.files_changed,
                    pump.files_removed,
                    pump.full_reloads,
                    pump.errors,
                );
            }
        }

        // Update shader preview based on whatever the file browser has
        // selected this frame.
        self.sync_shader_preview();

        // Stash a click target for "Open Project…" — the menu closure
        // only has an immutable-ish ui handle, but we need `&mut self`
        // to swap the project. Set a flag here, act after the menu.
        let mut requested_open = false;
        let current_project_label = self
            .project
            .as_ref()
            .map(|p| p.scene.root_dir.display().to_string())
            .unwrap_or_else(|| "<no project>".into());

        egui::Panel::top("menu").show_inside(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open Project…").clicked() {
                        requested_open = true;
                        ui.close();
                    }
                    if let Some(p) = &self.project {
                        ui.separator();
                        ui.label(
                            egui::RichText::new(format!("Current: {}", p.scene.root_dir.display()))
                                .small()
                                .color(egui::Color32::from_white_alpha(160)),
                        );
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.menu_button("View", |ui| {
                    ui.label("Reset layout — TODO");
                });
                ui.menu_button("Help", |ui| {
                    ui.label("Ocarina — a filesystem-first game engine.");
                    ui.label(format!("Version {}", env!("CARGO_PKG_VERSION")));
                });

                ui.separator();
                ui.label(format!("RHI: {}", self.rhi.backend_name()));
                ui.label(format!(
                    "({})",
                    self.rhi.capabilities().device_name.as_str()
                ));

                if let Some(p) = &self.plugin {
                    ui.separator();
                    ui.label(egui::RichText::new(format!(
                        "plugin: {} v{}",
                        p.name(),
                        p.version()
                    )));
                }

                if let Some(project) = &self.project {
                    ui.separator();
                    ui.label(format!(
                        "scene: {} entities / {} components",
                        project.scene.entity_count(),
                        project.scene.component_instance_count()
                    ));
                } else {
                    ui.separator();
                    ui.label(
                        egui::RichText::new("no project")
                            .color(egui::Color32::from_white_alpha(140)),
                    );
                }
                let _ = current_project_label;
            });
        });

        egui::CentralPanel::default_margins()
            .frame(egui::Frame::central_panel(ui.style()).inner_margin(0.0))
            .show_inside(ui, |ui| {
                let style = Style::from_egui(ui.style());
                let mut viewer = PaneViewer {
                    rhi: &mut self.rhi,
                    viewport: &mut self.viewport,
                    renderer: &self.egui_renderer,
                    terminal: self.terminal.as_mut(),
                    project: self.project.as_mut(),
                    selection: &mut self.selection,
                };
                DockArea::new(&mut self.dock_state)
                    .style(style)
                    .show_inside(ui, &mut viewer);
            });

        // File → Open Project… action. Deferred out of the menu closure
        // so we can borrow `self` mutably.
        if requested_open {
            self.prompt_open_project();
        }

        // End-of-frame: advance the RHI's deferred-destruction queue. We pass
        // `SwapchainHandle::INVALID` because the editor renders to offscreen
        // textures, not a swapchain — the wgpu backend uses this call purely
        // to rotate per-frame state.
        if let Err(e) = self.rhi.present(SwapchainHandle::INVALID) {
            log::warn!("rhi.present (frame rotation) failed: {e}");
        }

        // The Viewport pane draws each paint; request a repaint so edits
        // reflect promptly as we start wiring live systems.
        ctx.request_repaint();
    }
}
