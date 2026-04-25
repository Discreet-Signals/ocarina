//! Embedded terminal using `alacritty_terminal`.
//!
//! Owns:
//! - a `Term` behind `FairMutex` (shared with the alacritty reader thread),
//! - an `EventLoopSender` for writing input bytes to the PTY,
//! - a background thread (via `EventLoop::spawn`) that pumps PTY bytes.
//!
//! Per frame, [`Terminal::ui`] measures font metrics, resizes the grid + PTY
//! if the pane changed size, handles mouse + keyboard events (including
//! text selection and copy), snapshots the grid via `display_iter`, and
//! paints it as monospace text in egui with per-cell foreground +
//! background colors.

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Instant;

use alacritty_terminal::event::{Event, EventListener, WindowSize};
use alacritty_terminal::event_loop::{EventLoop, EventLoopSender, Msg};
use alacritty_terminal::grid::{Dimensions, Scroll};
use alacritty_terminal::sync::FairMutex;
use alacritty_terminal::term::cell::Flags;
use alacritty_terminal::term::{Config, Term};
use alacritty_terminal::tty;
use alacritty_terminal::vte::ansi::{Color as AnsiColor, NamedColor, Rgb};
use eframe::egui;

#[derive(Clone, Default)]
struct SilentListener;

impl EventListener for SilentListener {
    fn send_event(&self, _event: Event) {}
}

struct TermDimensions {
    columns: usize,
    screen_lines: usize,
}

impl Dimensions for TermDimensions {
    fn total_lines(&self) -> usize {
        self.screen_lines
    }
    fn screen_lines(&self) -> usize {
        self.screen_lines
    }
    fn columns(&self) -> usize {
        self.columns
    }
}

const MIN_COLS: usize = 8;
const MIN_ROWS: usize = 3;
const PADDING_X: f32 = 6.0;
const PADDING_Y: f32 = 4.0;

/// One cell's renderable state. Produced fresh each frame from the
/// alacritty grid; consumed by both the paint passes and the
/// selection-copy helper.
struct CellSnapshot {
    ch: char,
    fg: egui::Color32,
    bg: egui::Color32,
    flags: Flags,
}

// ───────── Selection ─────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct CellPos {
    row: usize,
    col: usize,
}

impl PartialOrd for CellPos {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for CellPos {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.row, self.col).cmp(&(other.row, other.col))
    }
}

#[derive(Clone, Copy, Debug)]
struct Selection {
    anchor: CellPos,
    current: CellPos,
}

impl Selection {
    fn normalized(&self) -> (CellPos, CellPos) {
        if self.anchor <= self.current {
            (self.anchor, self.current)
        } else {
            (self.current, self.anchor)
        }
    }

    fn is_single_cell(&self) -> bool {
        self.anchor == self.current
    }
}

pub struct Terminal {
    term: Arc<FairMutex<Term<SilentListener>>>,
    sender: EventLoopSender,
    cols: usize,
    rows: usize,
    font_size: f32,
    last_diag: Instant,
    selection: Option<Selection>,
    /// Temp directory containing our generated `.zshrc`. Cleaned up on Drop.
    zdotdir: Option<std::path::PathBuf>,
}

/// Zsh init that sources the user's real `~/.zshrc` (so PATH, aliases,
/// functions, etc. all apply) and then overrides `PROMPT` to something
/// modern. `OCARINA_NO_PROMPT_OVERRIDE=1` opts out if the user prefers
/// their own prompt (e.g. Starship) untouched.
const OCARINA_ZSHRC: &str = r#"# Ocarina editor override — runs instead of the default
# ZDOTDIR-less .zshrc when spawned inside the Ocarina terminal pane.

# Pull in the user's regular zsh config so PATH, aliases, and functions all work.
if [ -f "$HOME/.zshrc" ]; then
    source "$HOME/.zshrc"
fi

# Custom prompt — skip if the user opted out.
if [ -z "$OCARINA_NO_PROMPT_OVERRIDE" ]; then
    autoload -Uz colors 2>/dev/null && colors 2>/dev/null
    autoload -Uz vcs_info 2>/dev/null
    precmd_ocarina_vcs() { vcs_info 2>/dev/null }
    autoload -Uz add-zsh-hook 2>/dev/null && add-zsh-hook precmd precmd_ocarina_vcs 2>/dev/null
    zstyle ':vcs_info:git:*' formats ' %F{244}(%b)%f' 2>/dev/null
    setopt PROMPT_SUBST 2>/dev/null
    # Two-line prompt:
    #   ~/code/ocarina (git-branch)
    #   ❯
    PROMPT=$'%F{111}%~%f${vcs_info_msg_0_}\n%F{10}❯%f '
    RPROMPT=''
fi
"#;

impl Terminal {
    pub fn new() -> anyhow::Result<Self> {
        let cols: usize = 80;
        let rows: usize = 24;

        let window_size = WindowSize {
            num_cols: cols as u16,
            num_lines: rows as u16,
            cell_width: 8,
            cell_height: 16,
        };

        let mut options = tty::Options::default();
        options
            .env
            .insert("TERM".to_string(), "xterm-256color".to_string());
        options
            .env
            .insert("COLORTERM".to_string(), "truecolor".to_string());

        // Inject our zsh override so the user gets a modern prompt without
        // touching their real `~/.zshrc`. Best-effort — if the temp dir
        // can't be created we simply fall back to the default prompt.
        let zdotdir = setup_zdotdir().ok();
        if let Some(d) = &zdotdir {
            options
                .env
                .insert("ZDOTDIR".to_string(), d.to_string_lossy().into_owned());
            log::info!("terminal: ZDOTDIR={}", d.display());
        }

        let pty = tty::new(&options, window_size, 0)
            .map_err(|e| anyhow::anyhow!("pty spawn failed: {e}"))?;
        log::info!("terminal: PTY spawned (seed {cols}x{rows})");

        let term = Term::new(
            Config::default(),
            &TermDimensions {
                columns: cols,
                screen_lines: rows,
            },
            SilentListener,
        );
        let term = Arc::new(FairMutex::new(term));

        let event_loop = EventLoop::new(Arc::clone(&term), SilentListener, pty, false, false)
            .map_err(|e| anyhow::anyhow!("event loop init failed: {e}"))?;
        let sender = event_loop.channel();
        let _handle = event_loop.spawn();
        log::info!("terminal: event loop spawned");

        Ok(Self {
            term,
            sender,
            cols,
            rows,
            font_size: 12.0,
            last_diag: Instant::now(),
            selection: None,
            zdotdir,
        })
    }

    fn write(&self, data: Vec<u8>) {
        if !data.is_empty() {
            let _ = self.sender.send(Msg::Input(Cow::Owned(data)));
        }
    }

    fn resize_if_needed(&mut self, cols: usize, rows: usize, cell_w_px: f32, cell_h_px: f32) {
        let cols = cols.max(MIN_COLS);
        let rows = rows.max(MIN_ROWS);
        if cols == self.cols && rows == self.rows {
            return;
        }
        log::info!("terminal: resize {}x{} → {}x{}", self.cols, self.rows, cols, rows);
        self.cols = cols;
        self.rows = rows;
        // Resizing invalidates selection geometry.
        self.selection = None;

        {
            let mut term = self.term.lock();
            term.resize(TermDimensions {
                columns: cols,
                screen_lines: rows,
            });
        }

        let _ = self.sender.send(Msg::Resize(WindowSize {
            num_cols: cols as u16,
            num_lines: rows as u16,
            cell_width: cell_w_px.round().max(1.0) as u16,
            cell_height: cell_h_px.round().max(1.0) as u16,
        }));
    }

    /// Map a screen pixel position to a cell (row, col) in the visible grid.
    fn pos_to_cell(&self, pos: egui::Pos2, rect: egui::Rect, cell_w: f32, cell_h: f32) -> CellPos {
        let rx = (pos.x - rect.min.x - PADDING_X).max(0.0);
        let ry = (pos.y - rect.min.y - PADDING_Y).max(0.0);
        let col = (rx / cell_w).floor() as usize;
        let row = (ry / cell_h).floor() as usize;
        CellPos {
            row: row.min(self.rows.saturating_sub(1)),
            col: col.min(self.cols.saturating_sub(1)),
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let area = ui.available_size();

        // Stable id for the terminal pane's focus slot. Using
        // `ui.interact(rect, id, ...)` ensures `response.id == id` so
        // `response.has_focus()` and `memory.has_focus(id)` agree, and
        // `set_focus_lock_filter(id, …)` actually applies to the focused
        // widget.
        let id = ui.id().with("ocarina-terminal");
        let (rect, _alloc) = ui.allocate_exact_size(area, egui::Sense::hover());
        let response = ui
            .interact(rect, id, egui::Sense::click_and_drag())
            .on_hover_cursor(egui::CursorIcon::Text);

        if response.clicked() {
            response.request_focus();
            // A bare click clears any existing selection.
            self.selection = None;
        }

        let has_focus = response.has_focus();

        // Lock focus-navigation keys (arrows, Tab) to the terminal while it
        // has focus. Without this, egui forwards arrows to its focus
        // navigation system and our PTY never sees them. This only takes
        // effect starting the frame after focus was granted, so the first
        // arrow press after a click may still leak; subsequent presses
        // stay locked.
        if has_focus {
            ui.memory_mut(|m| {
                m.set_focus_lock_filter(
                    id,
                    egui::EventFilter {
                        tab: true,
                        horizontal_arrows: true,
                        vertical_arrows: true,
                        escape: false,
                    },
                );
            });
        }

        // Background.
        let default_bg = egui::Color32::from_rgb(14, 17, 22);
        ui.painter()
            .rect_filled(rect, egui::CornerRadius::same(0), default_bg);

        // Font metrics.
        let font = egui::FontId::monospace(self.font_size);
        let (cell_w, cell_h) = ui
            .ctx()
            .fonts_mut(|f| (f.glyph_width(&font, 'M'), f.row_height(&font)));

        // Resize grid + PTY to match pane.
        let usable_w = (rect.width() - PADDING_X * 2.0).max(0.0);
        let usable_h = (rect.height() - PADDING_Y * 2.0).max(0.0);
        let target_cols = (usable_w / cell_w).floor() as usize;
        let target_rows = (usable_h / cell_h).floor() as usize;
        self.resize_if_needed(target_cols, target_rows, cell_w, cell_h);

        // Mouse selection.
        if response.drag_started_by(egui::PointerButton::Primary) {
            if let Some(pos) = response.interact_pointer_pos() {
                let cell = self.pos_to_cell(pos, rect, cell_w, cell_h);
                self.selection = Some(Selection {
                    anchor: cell,
                    current: cell,
                });
                response.request_focus();
            }
        }
        if response.dragged_by(egui::PointerButton::Primary) {
            if let Some(pos) = response.interact_pointer_pos() {
                let cell = self.pos_to_cell(pos, rect, cell_w, cell_h);
                if let Some(sel) = self.selection.as_mut() {
                    sel.current = cell;
                }
            }
        }

        // Mouse-wheel scrollback; scrolling also clears selection because
        // cell coordinates no longer line up with the content.
        if response.hovered() {
            let scroll_y = ui.input(|i| i.smooth_scroll_delta.y);
            let lines = (scroll_y / cell_h).round() as i32;
            if lines != 0 {
                let mut term = self.term.lock();
                term.scroll_display(Scroll::Delta(lines));
                self.selection = None;
            }
        }

        let default_fg = egui::Color32::from_rgb(220, 222, 228);

        // Snapshot grid.
        let cursor_row;
        let cursor_col;
        let display_offset;
        let mut non_blank_lines = 0usize;

        let rendered: Vec<Vec<CellSnapshot>> = {
            let term = self.term.lock();
            let grid = term.grid();
            let rows_visible = grid.screen_lines();
            cursor_row = grid.cursor.point.line.0;
            cursor_col = grid.cursor.point.column.0;
            display_offset = grid.display_offset();

            let mut rows_buf: Vec<Vec<CellSnapshot>> = (0..rows_visible)
                .map(|_| {
                    (0..self.cols)
                        .map(|_| CellSnapshot {
                            ch: ' ',
                            fg: default_fg,
                            bg: default_bg,
                            flags: Flags::empty(),
                        })
                        .collect()
                })
                .collect();

            // When `display_offset > 0` (scrolled into history), alacritty's
            // `display_iter` yields rows with `line.0 ∈ [-offset,
            // screen_lines - offset - 1]`: negative for the top of the
            // viewport, positive for the bottom. Map back to 0-based
            // display row so the visible rect actually contains the
            // scrolled content.
            let offset = display_offset as i32;
            for indexed in grid.display_iter() {
                let line_idx = indexed.point.line.0;
                let col_idx = indexed.point.column.0;
                let display_row = line_idx + offset;
                if display_row < 0 || (display_row as usize) >= rows_visible {
                    continue;
                }
                if col_idx >= self.cols {
                    continue;
                }
                let cell = indexed.cell;
                let ch = cell.c;
                let ch = if ch == '\0' || ch == '\u{200B}' { ' ' } else { ch };

                let inverse = cell.flags.contains(Flags::INVERSE);
                let (fg_src, bg_src) = if inverse {
                    (cell.bg, cell.fg)
                } else {
                    (cell.fg, cell.bg)
                };

                let mut fg = resolve_color(fg_src, default_fg, default_bg, cell.flags);
                if cell.flags.contains(Flags::DIM) {
                    fg = blend(fg, default_bg, 0.5);
                }
                let bg = resolve_color(bg_src, default_fg, default_bg, cell.flags);

                rows_buf[display_row as usize][col_idx] = CellSnapshot {
                    ch,
                    fg,
                    bg,
                    flags: cell.flags,
                };
            }
            rows_buf
        };

        // Keyboard input (may trigger copy — needs access to `rendered`).
        if has_focus {
            let wrote = self.handle_input(ui, &rendered);
            if wrote {
                let mut term = self.term.lock();
                term.scroll_display(Scroll::Bottom);
            }
        }

        // ── Paint pass 1: per-cell backgrounds (non-default runs) ──
        for (i, row) in rendered.iter().enumerate() {
            let y = rect.min.y + PADDING_Y + i as f32 * cell_h;
            let mut run_start = 0usize;
            let mut run_color = row[0].bg;
            for col in 1..row.len() {
                if row[col].bg != run_color {
                    if run_color != default_bg {
                        let x0 = rect.min.x + PADDING_X + run_start as f32 * cell_w;
                        let x1 = rect.min.x + PADDING_X + col as f32 * cell_w;
                        ui.painter().rect_filled(
                            egui::Rect::from_min_max(
                                egui::pos2(x0, y),
                                egui::pos2(x1, y + cell_h),
                            ),
                            egui::CornerRadius::same(0),
                            run_color,
                        );
                    }
                    run_start = col;
                    run_color = row[col].bg;
                }
            }
            if run_color != default_bg {
                let x0 = rect.min.x + PADDING_X + run_start as f32 * cell_w;
                let x1 = rect.min.x + PADDING_X + row.len() as f32 * cell_w;
                ui.painter().rect_filled(
                    egui::Rect::from_min_max(egui::pos2(x0, y), egui::pos2(x1, y + cell_h)),
                    egui::CornerRadius::same(0),
                    run_color,
                );
            }
        }

        // ── Paint pass 2: selection overlay ──
        if let Some(sel) = self.selection {
            // Skip "selection" that's a single cell from a drag that never
            // moved — otherwise every click paints a phantom cell.
            if !sel.is_single_cell() {
                let overlay =
                    egui::Color32::from_rgba_unmultiplied(70, 120, 200, 110);
                for i in 0..rendered.len() {
                    let y = rect.min.y + PADDING_Y + i as f32 * cell_h;
                    // Find contiguous selected column run on this row.
                    let (s, e) = sel.normalized();
                    if i < s.row || i > e.row {
                        continue;
                    }
                    let col_start = if i == s.row { s.col } else { 0 };
                    let col_end = if i == e.row {
                        (e.col + 1).min(self.cols)
                    } else {
                        self.cols
                    };
                    if col_end > col_start {
                        let x0 = rect.min.x + PADDING_X + col_start as f32 * cell_w;
                        let x1 = rect.min.x + PADDING_X + col_end as f32 * cell_w;
                        ui.painter().rect_filled(
                            egui::Rect::from_min_max(
                                egui::pos2(x0, y),
                                egui::pos2(x1, y + cell_h),
                            ),
                            egui::CornerRadius::same(0),
                            overlay,
                        );
                    }
                }
            }
        }

        // ── Paint pass 3: foreground text ──
        for (i, row) in rendered.iter().enumerate() {
            let y = rect.min.y + PADDING_Y + i as f32 * cell_h;
            let mut col = 0usize;
            while col < row.len() {
                if row[col].ch == ' '
                    && row[col].flags & (Flags::UNDERLINE | Flags::BOLD) == Flags::empty()
                {
                    col += 1;
                    continue;
                }
                let fg = row[col].fg;
                let bold = row[col].flags.contains(Flags::BOLD);
                let start = col;
                let mut text = String::new();
                while col < row.len()
                    && row[col].fg == fg
                    && row[col].flags.contains(Flags::BOLD) == bold
                    && row[col].ch != '\0'
                {
                    text.push(row[col].ch);
                    col += 1;
                }
                if !text.trim().is_empty() {
                    non_blank_lines = non_blank_lines.saturating_add(1);
                    let x = rect.min.x + PADDING_X + start as f32 * cell_w;
                    ui.painter().text(
                        egui::pos2(x, y),
                        egui::Align2::LEFT_TOP,
                        &text,
                        if bold {
                            egui::FontId::new(self.font_size, egui::FontFamily::Monospace)
                        } else {
                            font.clone()
                        },
                        fg,
                    );
                    let flags = rendered[i][start].flags;
                    if flags.intersects(Flags::ALL_UNDERLINES) {
                        let uy = y + cell_h * 0.88;
                        ui.painter().line_segment(
                            [
                                egui::pos2(x, uy),
                                egui::pos2(x + text.chars().count() as f32 * cell_w, uy),
                            ],
                            egui::Stroke::new(1.0, fg),
                        );
                    }
                    if flags.contains(Flags::STRIKEOUT) {
                        let sy = y + cell_h * 0.55;
                        ui.painter().line_segment(
                            [
                                egui::pos2(x, sy),
                                egui::pos2(x + text.chars().count() as f32 * cell_w, sy),
                            ],
                            egui::Stroke::new(1.0, fg),
                        );
                    }
                } else if col == start {
                    col += 1;
                }
            }
        }

        // ── Paint pass 4: cursor (only when live, not scrolled back) ──
        if display_offset == 0 {
            let cx = rect.min.x + PADDING_X + (cursor_col as f32) * cell_w;
            let cy = rect.min.y + PADDING_Y + (cursor_row as f32) * cell_h;
            let cursor_rect = egui::Rect::from_min_size(
                egui::pos2(cx, cy),
                egui::vec2(cell_w, cell_h * 0.9),
            );
            let blink_on = ((Instant::now().elapsed().as_millis() as u64) / 500) % 2 == 0;
            if has_focus && blink_on {
                ui.painter().rect_filled(
                    cursor_rect,
                    egui::CornerRadius::same(0),
                    egui::Color32::from_rgb(90, 180, 140),
                );
            } else {
                ui.painter().rect_stroke(
                    cursor_rect,
                    egui::CornerRadius::same(0),
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(90, 180, 140)),
                    egui::StrokeKind::Inside,
                );
            }
        }

        // ── Corner status ──
        let selected_count = self
            .selection
            .as_ref()
            .filter(|s| !s.is_single_cell())
            .map(|s| {
                let (st, e) = s.normalized();
                let rows = e.row.saturating_sub(st.row) + 1;
                format!(" · {rows}L sel")
            })
            .unwrap_or_default();
        let scroll_tag = if display_offset > 0 {
            format!(" · scroll -{display_offset}")
        } else {
            String::new()
        };
        let status = format!(
            "{}x{}{}{}{}",
            self.cols,
            self.rows,
            scroll_tag,
            selected_count,
            if has_focus { " · focused" } else { "" },
        );
        ui.painter().text(
            rect.max - egui::vec2(PADDING_X, 2.0),
            egui::Align2::RIGHT_BOTTOM,
            status,
            egui::FontId::proportional(10.0),
            egui::Color32::from_white_alpha(120),
        );

        if self.last_diag.elapsed().as_secs() >= 10 {
            log::debug!(
                "terminal: {}x{}, cursor {}:{}, offset {}, focused={}, nonblank={}",
                self.cols,
                self.rows,
                cursor_row,
                cursor_col,
                display_offset,
                has_focus,
                non_blank_lines
            );
            self.last_diag = Instant::now();
        }
    }

    /// Pull keyboard events from egui. Copy shortcut (⌘C on mac,
    /// Ctrl+Shift+C elsewhere) is intercepted here and routed to the
    /// clipboard instead of the PTY. Ctrl+C alone still sends SIGINT.
    ///
    /// Returns true if any bytes were sent to the PTY.
    fn handle_input(
        &self,
        ui: &mut egui::Ui,
        rendered: &[Vec<CellSnapshot>],
    ) -> bool {
        let mut to_send: Vec<u8> = Vec::new();
        let mut copy_requested = false;

        ui.input_mut(|i| {
            i.events.retain(|ev| match ev {
                // Platform-neutral copy request (⌘C on mac, Ctrl+C on
                // Win/Linux). On Linux this would also fire for plain
                // Ctrl+C in a standard-widget context; terminal panes
                // won't have that because we consume Key(C) without
                // shift for SIGINT. Accepting `Event::Copy` is the mac
                // path; Ctrl+Shift+C on other platforms is handled by
                // the Key branch below.
                egui::Event::Copy => {
                    log::info!("terminal: Event::Copy received");
                    copy_requested = true;
                    false
                }
                // Paste: egui delivers the clipboard text already.
                egui::Event::Paste(t) => {
                    log::info!("terminal: Event::Paste received ({} chars)", t.chars().count());
                    to_send.extend_from_slice(t.as_bytes());
                    false
                }
                egui::Event::Text(t) => {
                    to_send.extend_from_slice(t.as_bytes());
                    false
                }
                egui::Event::Key {
                    key: egui::Key::C,
                    pressed: true,
                    modifiers,
                    ..
                } if is_copy_shortcut(*modifiers) => {
                    log::info!("terminal: copy key combo detected");
                    copy_requested = true;
                    false
                }
                egui::Event::Key {
                    key: egui::Key::V,
                    pressed: true,
                    modifiers,
                    ..
                } if is_paste_shortcut(*modifiers) => {
                    log::info!("terminal: paste key combo detected (reading clipboard)");
                    // egui doesn't expose a direct "get clipboard" call;
                    // our best shot is to consume the Key and ask egui
                    // to fire Event::Paste next frame by forwarding an
                    // OutputCommand. In practice egui will already have
                    // emitted Event::Paste alongside this Key event on
                    // ⌘V / Ctrl+V, so this branch is a fallback no-op.
                    false
                }
                egui::Event::Key {
                    key,
                    pressed: true,
                    modifiers,
                    ..
                } => {
                    if let Some(bytes) = key_to_bytes(*key, *modifiers) {
                        to_send.extend_from_slice(&bytes);
                        false
                    } else {
                        true
                    }
                }
                _ => true,
            });
        });

        if copy_requested {
            match self.extract_selected_text(rendered) {
                Some(text) => {
                    log::info!(
                        "terminal: copying {} chars to clipboard (first 40: {:?})",
                        text.len(),
                        text.chars().take(40).collect::<String>()
                    );
                    ui.ctx().copy_text(text);
                }
                None => {
                    log::info!("terminal: copy requested but selection is empty");
                }
            }
        }

        if to_send.is_empty() {
            false
        } else {
            self.write(to_send);
            true
        }
    }

    fn extract_selected_text(&self, rendered: &[Vec<CellSnapshot>]) -> Option<String> {
        let sel = self.selection?;
        if sel.is_single_cell() {
            return None;
        }
        let (start, end) = sel.normalized();
        let mut out = String::new();
        for row_idx in start.row..=end.row.min(rendered.len().saturating_sub(1)) {
            let row = &rendered[row_idx];
            let col_start = if row_idx == start.row { start.col } else { 0 };
            let col_end = if row_idx == end.row {
                (end.col + 1).min(row.len())
            } else {
                row.len()
            };
            let line: String = row[col_start..col_end].iter().map(|c| c.ch).collect();
            // Trim only the trailing padding spaces, not internal ones.
            out.push_str(line.trim_end());
            if row_idx < end.row {
                out.push('\n');
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }
}

impl Drop for Terminal {
    fn drop(&mut self) {
        let _ = self.sender.send(Msg::Shutdown);
        if let Some(d) = &self.zdotdir {
            let _ = std::fs::remove_dir_all(d);
        }
    }
}

fn is_copy_shortcut(m: egui::Modifiers) -> bool {
    // Mac: ⌘C (no ctrl, no shift, no alt).
    // Other platforms: Ctrl+Shift+C (plain Ctrl+C stays as SIGINT).
    let mac = m.mac_cmd && !m.ctrl && !m.shift && !m.alt;
    let other = m.ctrl && m.shift && !m.mac_cmd && !m.alt;
    mac || other
}

fn is_paste_shortcut(m: egui::Modifiers) -> bool {
    // Mac: ⌘V. Other platforms: Ctrl+Shift+V (plain Ctrl+V is readline
    // "verbatim next char"). egui emits `Event::Paste` for these combos
    // alongside the Key event; this predicate is only used to consume
    // the redundant Key event.
    let mac = m.mac_cmd && !m.ctrl && !m.shift && !m.alt;
    let other = m.ctrl && m.shift && !m.mac_cmd && !m.alt;
    mac || other
}

/// Write a one-off `.zshrc` into a PID-scoped temp dir and return the
/// path. The directory is passed to the spawned shell via `ZDOTDIR`, and
/// the `Terminal::drop` impl removes it when the pane goes away.
fn setup_zdotdir() -> std::io::Result<std::path::PathBuf> {
    let dir = std::env::temp_dir().join(format!("ocarina-zsh-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join(".zshrc"), OCARINA_ZSHRC)?;
    // Also drop a .zshenv so ZDOTDIR is respected even when the shell
    // runs as a login shell (login → .zshenv → .zprofile → .zshrc).
    std::fs::write(dir.join(".zshenv"), "")?;
    Ok(dir)
}

// ──────────────────────────── Color resolution ────────────────────────────

fn resolve_color(
    color: AnsiColor,
    default_fg: egui::Color32,
    default_bg: egui::Color32,
    flags: Flags,
) -> egui::Color32 {
    match color {
        AnsiColor::Named(named) => named_color(named, default_fg, default_bg, flags),
        AnsiColor::Spec(Rgb { r, g, b }) => egui::Color32::from_rgb(r, g, b),
        AnsiColor::Indexed(i) => indexed_color(i, default_fg, default_bg, flags),
    }
}

fn named_color(
    n: NamedColor,
    default_fg: egui::Color32,
    default_bg: egui::Color32,
    flags: Flags,
) -> egui::Color32 {
    use NamedColor::*;
    let base = match n {
        Black => (0x1e, 0x21, 0x2c),
        Red => (0xe0, 0x60, 0x5d),
        Green => (0x6f, 0xc2, 0x76),
        Yellow => (0xd9, 0xbd, 0x5c),
        Blue => (0x6c, 0x9e, 0xf0),
        Magenta => (0xc7, 0x84, 0xd6),
        Cyan => (0x6c, 0xdc, 0xd0),
        White => (0xd6, 0xd7, 0xde),
        BrightBlack => (0x4b, 0x4f, 0x5a),
        BrightRed => (0xff, 0x8a, 0x7a),
        BrightGreen => (0x8a, 0xe0, 0x90),
        BrightYellow => (0xff, 0xd9, 0x7a),
        BrightBlue => (0x90, 0xbc, 0xff),
        BrightMagenta => (0xe3, 0xa2, 0xf0),
        BrightCyan => (0x88, 0xf0, 0xe2),
        BrightWhite => (0xf6, 0xf6, 0xfb),
        Foreground => return default_fg,
        Background => return default_bg,
        BrightForeground => return default_fg,
        DimForeground => return default_fg,
        DimBlack | DimRed | DimGreen | DimYellow | DimBlue | DimMagenta | DimCyan | DimWhite => {
            let base_named = match n {
                DimBlack => Black,
                DimRed => Red,
                DimGreen => Green,
                DimYellow => Yellow,
                DimBlue => Blue,
                DimMagenta => Magenta,
                DimCyan => Cyan,
                DimWhite => White,
                _ => Foreground,
            };
            let c = named_color(base_named, default_fg, default_bg, flags);
            return blend(c, default_bg, 0.5);
        }
        Cursor => return egui::Color32::from_rgb(90, 180, 140),
    };
    let (r, g, b) = if flags.contains(Flags::BOLD) {
        brighten(base)
    } else {
        base
    };
    egui::Color32::from_rgb(r, g, b)
}

fn brighten((r, g, b): (u8, u8, u8)) -> (u8, u8, u8) {
    (
        ((r as u16 * 115 / 100).min(255)) as u8,
        ((g as u16 * 115 / 100).min(255)) as u8,
        ((b as u16 * 115 / 100).min(255)) as u8,
    )
}

fn indexed_color(
    i: u8,
    default_fg: egui::Color32,
    default_bg: egui::Color32,
    flags: Flags,
) -> egui::Color32 {
    if i < 16 {
        let named = match i {
            0 => NamedColor::Black,
            1 => NamedColor::Red,
            2 => NamedColor::Green,
            3 => NamedColor::Yellow,
            4 => NamedColor::Blue,
            5 => NamedColor::Magenta,
            6 => NamedColor::Cyan,
            7 => NamedColor::White,
            8 => NamedColor::BrightBlack,
            9 => NamedColor::BrightRed,
            10 => NamedColor::BrightGreen,
            11 => NamedColor::BrightYellow,
            12 => NamedColor::BrightBlue,
            13 => NamedColor::BrightMagenta,
            14 => NamedColor::BrightCyan,
            15 => NamedColor::BrightWhite,
            _ => NamedColor::Foreground,
        };
        named_color(named, default_fg, default_bg, flags)
    } else if i < 232 {
        let idx = i - 16;
        let r = idx / 36;
        let g = (idx % 36) / 6;
        let b = idx % 6;
        let ramp = [0u8, 95, 135, 175, 215, 255];
        egui::Color32::from_rgb(
            ramp[r as usize],
            ramp[g as usize],
            ramp[b as usize],
        )
    } else {
        let step = i - 232;
        let v = 8 + 10 * step;
        egui::Color32::from_rgb(v, v, v)
    }
}

fn blend(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    let inv = 1.0 - t;
    egui::Color32::from_rgb(
        (a.r() as f32 * inv + b.r() as f32 * t) as u8,
        (a.g() as f32 * inv + b.g() as f32 * t) as u8,
        (a.b() as f32 * inv + b.b() as f32 * t) as u8,
    )
}

// ──────────────────────────── Keyboard ────────────────────────────

fn key_to_bytes(key: egui::Key, modifiers: egui::Modifiers) -> Option<Vec<u8>> {
    use egui::Key;
    match key {
        Key::Enter => Some(b"\r".to_vec()),
        Key::Backspace => Some(vec![0x7f]),
        Key::Tab => Some(b"\t".to_vec()),
        Key::Escape => Some(vec![0x1b]),
        Key::ArrowUp => Some(b"\x1b[A".to_vec()),
        Key::ArrowDown => Some(b"\x1b[B".to_vec()),
        Key::ArrowRight => Some(b"\x1b[C".to_vec()),
        Key::ArrowLeft => Some(b"\x1b[D".to_vec()),
        Key::Home => Some(b"\x1b[H".to_vec()),
        Key::End => Some(b"\x1b[F".to_vec()),
        Key::Delete => Some(b"\x1b[3~".to_vec()),
        Key::PageUp => Some(b"\x1b[5~".to_vec()),
        Key::PageDown => Some(b"\x1b[6~".to_vec()),
        _ => {
            if modifiers.ctrl {
                let ch = format!("{key:?}");
                if ch.len() == 1 {
                    let c = ch.chars().next().unwrap();
                    if c.is_ascii_alphabetic() {
                        let code = (c.to_ascii_lowercase() as u8) - b'a' + 1;
                        return Some(vec![code]);
                    }
                }
            }
            None
        }
    }
}
