//! Editor pane definitions. Each variant of [`Pane`] maps to one panel in
//! the docked layout. Most panes show placeholder content in Phase 0–1;
//! the Viewport pane is live as of Phase 1 and renders a triangle via the
//! RHI.

use std::sync::Arc;

use eframe::egui;
use egui::epaint::mutex::RwLock;
use egui_dock::TabViewer;

use ocarina_rhi_wgpu::WgpuRhi;

use crate::project::Project;
use crate::selection::Selection;
use crate::terminal::Terminal;
use crate::viewport::Viewport;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pane {
    FileBrowser,
    TextEditor,
    Viewport,
    LensPanel,
    Terminal,
}

impl Pane {
    pub const fn title(self) -> &'static str {
        match self {
            Self::FileBrowser => "Files",
            Self::TextEditor => "Editor",
            Self::Viewport => "Viewport",
            Self::LensPanel => "Lens",
            Self::Terminal => "Terminal",
        }
    }
}

/// Per-frame context threaded to the TabViewer. `rhi` and `viewport` are
/// borrowed mutably because the Viewport pane records and submits GPU work
/// during its draw; the other panes just read the backend name.
pub struct PaneViewer<'a> {
    pub rhi: &'a mut WgpuRhi,
    pub viewport: &'a mut Viewport,
    pub renderer: &'a Arc<RwLock<egui_wgpu::Renderer>>,
    pub terminal: Option<&'a mut Terminal>,
    pub project: Option<&'a mut Project>,
    pub selection: &'a mut Selection,
}

impl TabViewer for PaneViewer<'_> {
    type Tab = Pane;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        tab.title().into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        match tab {
            Pane::FileBrowser => file_browser(ui, self.project.as_deref_mut()),
            Pane::TextEditor => text_editor(ui),
            Pane::Viewport => {
                let scene = self.project.as_deref_mut().map(|p| &mut p.scene);
                self.viewport
                    .render_and_show(self.rhi, self.renderer, ui, scene, self.selection);
            }
            Pane::LensPanel => lens_panel(ui),
            Pane::Terminal => match self.terminal.as_deref_mut() {
                Some(t) => t.ui(ui),
                None => terminal_placeholder(ui),
            },
        }
    }

    fn closeable(&mut self, _tab: &mut Self::Tab) -> bool {
        false
    }
}

// ─────────────────────────── Placeholder panes ───────────────────────────

fn file_browser(ui: &mut egui::Ui, project: Option<&mut Project>) {
    ui.heading("Files");
    ui.add_space(2.0);

    let Some(project) = project else {
        ui.colored_label(egui::Color32::from_rgb(210, 150, 150), "No project open.");
        ui.add_space(6.0);
        ui.label(
            egui::RichText::new(
                "Use File → Open Project…, or pass --project <path> on the \
                 command line, or set OCARINA_PROJECT.",
            )
            .small()
            .color(egui::Color32::from_white_alpha(160)),
        );
        return;
    };

    ui.label(
        egui::RichText::new(project.tree.name.clone())
            .strong()
            .color(egui::Color32::from_rgb(200, 215, 240)),
    );
    ui.label(
        egui::RichText::new(format!(
            "{} entities · {} components · {} files loaded",
            project.scene.entity_count(),
            project.scene.component_instance_count(),
            project.last_report.files_parsed,
        ))
        .small()
        .color(egui::Color32::from_white_alpha(150)),
    );
    ui.add_space(4.0);
    ui.separator();
    ui.add_space(2.0);

    // Globally-stable id for the focus slot. We previously used
    // `ui.id().with("…")`, which mixes in the parent ui's auto-id, and
    // egui_dock's layout reshuffles parent stacks across frames so the
    // resulting id can drift — focus would be granted one frame and
    // then queried under a different id the next, silently losing it.
    let browser_id = egui::Id::new("ocarina-file-browser-pane");

    // Allocate a hit region covering the rest of the pane so clicks on
    // empty space (between rows) focus the pane. Sized to all remaining
    // space; rows render INSIDE the ScrollArea below this and intercept
    // their own clicks first because they're drawn on top.
    let pane_rect = ui.available_rect_before_wrap();
    let pane_response = ui.interact(pane_rect, browser_id, egui::Sense::click());
    let has_focus = pane_response.has_focus();

    // One-shot focus-change log so we can confirm behavior from stdout
    // when something feels wrong.
    static FOCUS_LAST: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    let prev_focus =
        FOCUS_LAST.swap(has_focus, std::sync::atomic::Ordering::Relaxed);
    if prev_focus != has_focus {
        log::info!("file-browser focus: {prev_focus} → {has_focus}");
    }

    if pane_response.clicked() {
        pane_response.request_focus();
    }

    if has_focus {
        ui.memory_mut(|m| {
            m.set_focus_lock_filter(
                browser_id,
                egui::EventFilter {
                    tab: false,
                    horizontal_arrows: true,
                    vertical_arrows: true,
                    escape: false,
                },
            );
        });
        let rows = flatten_visible(project);
        handle_key_nav(ui, project, &rows);
    }

    // Re-flatten after any keyboard mutation so this frame's render
    // includes the updated state.
    let rows = flatten_visible(project);

    egui::ScrollArea::vertical()
        .id_salt("project-tree")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            for row in &rows {
                draw_row(ui, project, row, browser_id, has_focus);
            }
        });

    if !project.errors.is_empty() {
        ui.add_space(4.0);
        ui.separator();
        ui.label(
            egui::RichText::new(format!("Load errors ({})", project.errors.len()))
                .small()
                .color(egui::Color32::LIGHT_RED),
        );
        egui::ScrollArea::vertical()
            .id_salt("project-errors")
            .max_height(80.0)
            .show(ui, |ui| {
                for e in project.errors.iter().rev().take(20) {
                    ui.label(
                        egui::RichText::new(e)
                            .small()
                            .color(egui::Color32::from_rgb(230, 150, 150)),
                    );
                }
            });
    }
}

// ───────────────────────── Tree flatten + rows ─────────────────────────

struct FlatRow {
    path: std::path::PathBuf,
    parent_path: std::path::PathBuf,
    name: String,
    depth: usize,
    is_dir: bool,
    has_children: bool,
}

fn flatten_visible(project: &Project) -> Vec<FlatRow> {
    let mut rows = Vec::new();
    for child in &project.tree.children {
        flatten(child, &project.expanded, 0, &mut rows);
    }
    rows
}

fn flatten(
    node: &crate::project::TreeNode,
    expanded: &std::collections::HashSet<std::path::PathBuf>,
    depth: usize,
    out: &mut Vec<FlatRow>,
) {
    let parent = node
        .path
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_default();
    out.push(FlatRow {
        path: node.path.clone(),
        parent_path: parent,
        name: node.name.clone(),
        depth,
        is_dir: node.is_dir,
        has_children: !node.children.is_empty(),
    });
    if node.is_dir && expanded.contains(&node.path) {
        let (mut dirs, mut files): (Vec<_>, Vec<_>) =
            node.children.iter().partition(|c| c.is_dir);
        dirs.sort_by(|a, b| a.name.cmp(&b.name));
        files.sort_by(|a, b| a.name.cmp(&b.name));
        for d in dirs {
            flatten(d, expanded, depth + 1, out);
        }
        for f in files {
            flatten(f, expanded, depth + 1, out);
        }
    }
}

/// Render one row VS-Code-style: full-width background (subtle on hover,
/// stronger on selection) with depth-indented chevron + name on top.
fn draw_row(
    ui: &mut egui::Ui,
    project: &mut Project,
    row: &FlatRow,
    browser_id: egui::Id,
    pane_focused: bool,
) {
    let is_selected = project.selected.as_deref() == Some(row.path.as_path());
    let is_expanded = project.expanded.contains(&row.path);

    // Allocate the full row, full available width. Sense::click so the
    // entire row is the hit target (not just the text).
    let row_height = 20.0;
    let avail_w = ui.available_width().max(80.0);
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(avail_w, row_height), egui::Sense::click());

    // Background colors. Selected (focused pane) uses egui's selection
    // accent; selected-but-unfocused fades to a flatter grey; hover is
    // a barely-there white wash.
    let bg = if is_selected {
        if pane_focused {
            egui::Color32::from_rgb(38, 79, 120) // VS Code "list.activeSelection"
        } else {
            egui::Color32::from_rgb(55, 60, 70)  // muted when not focused
        }
    } else if response.hovered() {
        egui::Color32::from_white_alpha(14)
    } else {
        egui::Color32::TRANSPARENT
    };
    if bg != egui::Color32::TRANSPARENT {
        ui.painter()
            .rect_filled(rect, egui::CornerRadius::ZERO, bg);
    }

    // Layout markers inside the row.
    let indent_px = row.depth as f32 * 14.0;
    let chevron_w = 14.0;
    let icon_w = 16.0;
    let pad_x = 6.0;

    let baseline = rect.center().y;
    let mut x = rect.left() + pad_x + indent_px;

    // Chevron (folders only). Always reserve the slot so file rows align.
    if row.is_dir {
        let chevron = if row.has_children {
            if is_expanded {
                "▾"
            } else {
                "▸"
            }
        } else {
            " "
        };
        ui.painter().text(
            egui::pos2(x + chevron_w * 0.5, baseline),
            egui::Align2::CENTER_CENTER,
            chevron,
            egui::FontId::monospace(12.0),
            egui::Color32::from_white_alpha(200),
        );
    }
    x += chevron_w;

    // File icon — files only. Folders skip and align to the chevron.
    if !row.is_dir {
        ui.painter().text(
            egui::pos2(x + icon_w * 0.5, baseline),
            egui::Align2::CENTER_CENTER,
            file_icon(&row.name),
            egui::FontId::proportional(13.0),
            file_color(&row.name),
        );
        x += icon_w;
    }

    // Name. Slightly brighter when the row is selected for legibility
    // against the accent fill.
    let name_color = if is_selected {
        egui::Color32::from_rgb(245, 248, 252)
    } else if row.is_dir {
        egui::Color32::from_rgb(215, 225, 240)
    } else {
        file_color(&row.name)
    };
    ui.painter().text(
        egui::pos2(x + 2.0, baseline),
        egui::Align2::LEFT_CENTER,
        &row.name,
        egui::FontId::proportional(13.0),
        name_color,
    );

    if response.clicked() {
        project.selected = Some(row.path.clone());
        project.scroll_to_selected = false;
        ui.memory_mut(|m| m.request_focus(browser_id));
        if row.is_dir {
            toggle_expanded(project, &row.path);
        }
    }

    if is_selected && project.scroll_to_selected {
        response.scroll_to_me(Some(egui::Align::Center));
        project.scroll_to_selected = false;
    }
}

fn toggle_expanded(project: &mut Project, path: &std::path::Path) {
    if project.expanded.contains(path) {
        project.expanded.remove(path);
    } else {
        project.expanded.insert(path.to_path_buf());
    }
}

// ───────────────────────── Keyboard nav ─────────────────────────

fn handle_key_nav(ui: &mut egui::Ui, project: &mut Project, rows: &[FlatRow]) {
    if rows.is_empty() {
        return;
    }

    // Consume arrow events if we want to act on them. Anything we don't
    // handle falls through.
    let (up, down, left, right, enter) = ui.input_mut(|i| {
        let up = i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp);
        let down = i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown);
        let left = i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowLeft);
        let right = i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight);
        let enter = i.consume_key(egui::Modifiers::NONE, egui::Key::Enter);
        (up, down, left, right, enter)
    });

    let cur_idx = project
        .selected
        .as_ref()
        .and_then(|p| rows.iter().position(|r| &r.path == p));

    if up {
        let new = match cur_idx {
            Some(i) if i > 0 => i - 1,
            Some(i) => i,       // already at top
            None => rows.len() - 1,
        };
        project.selected = Some(rows[new].path.clone());
        project.scroll_to_selected = true;
    }
    if down {
        let new = match cur_idx {
            Some(i) if i + 1 < rows.len() => i + 1,
            Some(i) => i,       // already at bottom
            None => 0,
        };
        project.selected = Some(rows[new].path.clone());
        project.scroll_to_selected = true;
    }
    if right {
        if let Some(i) = cur_idx {
            let row = &rows[i];
            if row.is_dir && row.has_children {
                if !project.expanded.contains(&row.path) {
                    project.expanded.insert(row.path.clone());
                } else if i + 1 < rows.len() {
                    // Already expanded → step into first child.
                    project.selected = Some(rows[i + 1].path.clone());
                    project.scroll_to_selected = true;
                }
            }
        }
    }
    if left {
        if let Some(i) = cur_idx {
            let row = &rows[i];
            if row.is_dir && project.expanded.contains(&row.path) {
                project.expanded.remove(&row.path);
            } else if !row.parent_path.as_os_str().is_empty() {
                // Jump up to the parent row if visible.
                if let Some(parent_idx) =
                    rows.iter().position(|r| r.path == row.parent_path)
                {
                    project.selected = Some(rows[parent_idx].path.clone());
                    project.scroll_to_selected = true;
                }
            }
        }
    }
    if enter {
        if let Some(i) = cur_idx {
            let row = &rows[i];
            if row.is_dir {
                toggle_expanded(project, &row.path);
            }
            // Non-dir: Phase 3 will open in the Editor pane. Nothing to
            // do here yet; `selected` is already set.
        }
    }
}

fn file_icon(name: &str) -> &'static str {
    if name.ends_with(".toml") {
        "⚙"
    } else if name.ends_with(".rs") {
        "🦀"
    } else if name.ends_with(".md") {
        "📄"
    } else if name.ends_with(".hlsl") || name.ends_with(".wgsl") || name.ends_with(".glsl") {
        "✦"
    } else if name.ends_with(".obj")
        || name.ends_with(".ply")
        || name.ends_with(".gltf")
        || name.ends_with(".glb")
    {
        "◆"
    } else {
        "·"
    }
}

fn file_color(name: &str) -> egui::Color32 {
    if name.ends_with(".toml") {
        egui::Color32::from_rgb(220, 180, 120) // TOML = amber
    } else if name.ends_with(".rs") {
        egui::Color32::from_rgb(220, 150, 120)
    } else if name.ends_with(".md") {
        egui::Color32::from_rgb(180, 200, 220)
    } else if name.ends_with(".hlsl") || name.ends_with(".wgsl") || name.ends_with(".glsl") {
        egui::Color32::from_rgb(180, 220, 180)
    } else {
        egui::Color32::from_rgb(210, 210, 215)
    }
}

fn text_editor(ui: &mut egui::Ui) {
    ui.heading("Editor");
    ui.add_space(6.0);
    ui.label("Text editor with tree-sitter TOML highlighting lands in Phase 3.");
    ui.add_space(4.0);
    ui.label(
        "The active file is shown here. Saving flushes to disk → `notify` fires → \
         the ECS world re-parses affected components. Buffer-vs-disk conflicts surface \
         a reload/keep/diff prompt.",
    );
}

fn lens_panel(ui: &mut egui::Ui) {
    ui.heading("Lens");
    ui.add_space(6.0);
    ui.label("Cursor-aware widget panel lands in Phase 3.");
    ui.add_space(4.0);
    ui.label(
        "Reads the cursor position in Editor, finds the AST node under it, and renders \
         a widget bound to that span. Widget edits splice text back into the buffer.",
    );
    ui.add_space(12.0);
    ui.separator();
    ui.add_space(6.0);
    ui.label("Preview widgets:");
    let mut color = egui::Color32::from_rgb(240, 170, 80);
    ui.color_edit_button_srgba(&mut color);
    let mut v = [0.0_f32, 0.0, 0.0];
    ui.horizontal(|ui| {
        ui.label("vec3");
        ui.add(egui::DragValue::new(&mut v[0]).speed(0.05));
        ui.add(egui::DragValue::new(&mut v[1]).speed(0.05));
        ui.add(egui::DragValue::new(&mut v[2]).speed(0.05));
    });
    let mut f = 0.5_f32;
    ui.horizontal(|ui| {
        ui.label("scalar");
        ui.add(egui::Slider::new(&mut f, 0.0..=1.0));
    });
}

fn terminal_placeholder(ui: &mut egui::Ui) {
    ui.heading("Terminal");
    ui.add_space(6.0);
    ui.colored_label(
        egui::Color32::LIGHT_RED,
        "PTY failed to start — terminal disabled for this session.",
    );
}
