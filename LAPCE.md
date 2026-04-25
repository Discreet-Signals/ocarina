# Ocarina + Lapce: editor migration plan

## Why

The current `ocarina-editor` is built on `egui + egui_dock + alacritty_terminal` and works through Phase 2 + the Phase 4 viewport. To reach a competitive in-house editor we'd need to spend 4–8 months building a real text-editor widget (multi-cursor, virtualized rendering, LSP-grade diagnostics, code folding) that egui's `TextEdit` cannot host. Lapce already ships that — built on Floem (Apache-2.0, wgpu-based), with the editor *primitive* lifted out into `floem-editor-core` and the Lapce shell adding rust-analyzer LSP, multi-cursor, vim mode, fuzzy palette, integrated terminal, git diff, find-in-files, and a SSH-based remote-development proxy. Upstream velocity is ~1 commit/week, so the rebase tax is annual not weekly.

Cursor → VS Code is the precedent: take a great editor, make a different feature the headline. For us, the headline is **scene-as-files + lens panel + live ECS hot-reload**, replacing Cursor's LLM-as-headline.

## Architecture target

```
ocarina/                         (workspace root, currently exists)
├── crates/
│   ├── ocarina-rhi              (unchanged)
│   ├── ocarina-rhi-wgpu         (unchanged; editor uses this)
│   ├── ocarina-rhi-null         (unchanged)
│   ├── ocarina-shader           (unchanged)
│   ├── ocarina-schema           (unchanged)
│   ├── ocarina-derive           (unchanged)
│   ├── ocarina-core             (unchanged)
│   ├── ocarina-scene            (unchanged)
│   ├── ocarina-runtime          (unchanged; never linked to editor)
│   ├── ocarina-gameplay-api     (unchanged)
│   └── ocarina-editor           (REPLACED — see below)
├── third_party/
│   ├── lapce/                   (NEW — git subtree of forked Lapce)
│   └── floem/                   (NEW — git subtree of patched Floem)
└── ...
```

`ocarina-editor` becomes a thin shell that depends on the forked `lapce-app`, registers Ocarina-specific panel kinds (Viewport, Lens), wires `WgpuRhi` into Floem's shared device, and replaces Lapce's branding + Run-and-Debug panel with engine-aware equivalents.

`lapce-proxy`, `lapce-rpc`, `lapce-core`, and `floem-editor-core` are kept verbatim. They are the gold.

## Phase L0 — Fork bootstrap + WgpuView derisk  (2–5 days)

The single load-bearing technical question: can Floem composite a `wgpu::TextureView` we render into via our RHI? If yes, the rest is mechanical. Derisk this first — before touching Lapce.

1. **Init git in `ocarina/`.** Commit current state as `pre-lapce-migration`.
2. **Fork `lapce/lapce` and `lapce/floem`** to the project's GitHub. Pin to a known-good SHA (Lapce master as of the fork date; Floem as the rev Lapce pins).
3. **Add as git subtrees** under `third_party/lapce/` and `third_party/floem/`. Subtree (not submodule) so a fresh clone has everything; rebases land as merge commits.
4. **Patch Floem to expose its `wgpu::Device`.** In `floem-renderer/src/gpu_resources.rs`, add a public getter for the active `wgpu::Device`/`Queue`/`Adapter`. Estimate: <50 LOC.
5. **Patch Floem's `Renderer` trait** with `draw_texture(view: &wgpu::TextureView, rect: Rect)`. Implement in Floem's default backend (Vger). Estimate: 500–1500 LOC across `floem-renderer` + `vger` backend.
6. **Add a `WgpuView` widget** that takes a `Fn() -> wgpu::TextureView` callback and a sizing policy, calls into the new `draw_texture` each frame.
7. **Build a standalone `derisk_viewport` example** in the Floem fork: spawn a Floem window with a single `WgpuView`. Inside, construct `WgpuRhi::from_existing(floem_device, floem_queue, floem_adapter)`, render a triangle, display via `WgpuView`. Confirms the seam works end-to-end.

**Exit criteria:** the derisk example shows a triangle rendered through `ocarina-rhi-wgpu` inside a Floem window. If this fails, escalate: see *Risks and fallback paths* below before continuing.

## Phase L1 — Workspace integration, minimal shell  (3–5 days)

1. **Add `third_party/lapce` and `third_party/floem` to the root `Cargo.toml`** as workspace members. Floem becomes a path dep of `lapce-app`; both override their crates.io versions.
2. **Build the unmodified Lapce.** Confirm it launches, opens a folder, edits a file. This is the baseline.
3. **Strip from `lapce-app`** (delete or feature-gate, in this order):
   - About / branding strings → "Ocarina"
   - Plugin marketplace UI (keep the WASI host; just remove the marketplace browser)
   - DAP / debugger panel (re-add later if we want runtime debugging)
   - Source-control panel (keep `git2` integration in proxy; drop the dedicated panel — file-tree git decoration is enough)
   - Lapce-specific welcome screen
4. **Replace `lapce-app`'s `main.rs`** with `ocarina-editor`'s entry point. Project bootstrap (`$OCARINA_PROJECT` env var, default `./projects/example-scene`) moves here from current `ocarina-editor/src/main.rs`.
5. **Rename the binary** to `ocarina-editor` so `cargo run -p ocarina-editor` keeps working from existing tooling.

**Exit criteria:** an Ocarina-branded Lapce opens, edits TOML and Rust files, has the integrated terminal working. No viewport yet.

## Phase L2 — Viewport panel  (1–2 weeks)

The big migration. Port `crates/ocarina-editor/src/viewport.rs` (~1000 LOC) into a Lapce panel kind.

1. **Register a new panel kind** in `lapce-app/src/panel/`. Lapce's panel system already has Terminal, FileExplorer, Search, Plugins as variants; add `Viewport` and `Lens`.
2. **Construct `WgpuRhi` from Floem's shared device** at editor startup. `WgpuRhi::from_existing` already exists from Phase 1c.
3. **Port from `viewport.rs`:**
   - `Viewport` struct + `ViewportTarget` (color/depth `TextureHandle`s)
   - `OrbitCamera` (no UI deps; lifts directly)
   - `Selection` + `Mesh` + `Vertex` (no UI deps)
   - Cube + Triforce mesh data
   - HLSL → SPIR-V pipeline build (lifts directly)
   - Render pass recording + submit
4. **Port input handling.** This is the biggest concrete porting task: replace egui's `Response` / `Sense::click_and_drag` / `dragged_by` / `interact_pointer_pos` with Floem's pointer/keyboard events on a `WgpuView`. Concretely:
   - Click-pick → Floem `on_click` emitting cursor position
   - Gizmo drag → Floem `on_pointer_down` / `on_pointer_move` / `on_pointer_up` chain
   - Camera orbit/pan/zoom → same drag/scroll wheel events
5. **Port write-back.** `transform_io::write_transform` lifts directly; only the trigger (gizmo drag-end) changes shape.
6. **Port shader preview.** `try_load_preview_shader` + `poll_preview_shader` lift directly. The trigger (active file in file browser is a `.hlsl`) wires through Lapce's existing "active file" signal.
7. **Selection sync between panes.** Lapce already has shared state via reactive signals (Floem signals, Leptos-style); selection becomes one such signal that both viewport and (future) lens panel read.
8. **Status overlay.** Entity count, distance, shader name, shader-error overlay all paint via Floem text views inside the `WgpuView`'s parent stack.

**Exit criteria:** open `projects/example-scene/`, see the cubes rendered, orbit-camera, click-pick, drag a gizmo, see the TOML file change on disk, watch the file watcher reload it.

## Phase L3 — Lens panel + dylib hot-reload  (1–2 weeks)

The headline feature. Currently absent from the egui editor (Phase 3 in the README roadmap). Building it on Lapce is *easier* than building it on egui because Lapce already exposes cursor position + AST node info via tree-sitter and LSP semantic tokens.

1. **Register `Lens` as a panel kind** (per L2).
2. **Hook into Lapce's text-editor cursor signal.** When the active editor pane's cursor moves, fire a callback with `(buffer_uri, char_offset, line:col)`.
3. **Map cursor → AST node via tree-sitter.** Lapce already runs tree-sitter for highlighting; reuse the parse tree. For the active TOML buffer, walk to the smallest node containing the cursor.
4. **Dispatch widget by `#[ocarina(widget = "...")]` attribute.** Look up the field in `ocarina-schema` (already loaded). First widgets to ship: `vec3` scrubber, `color` picker, `float` scrubber. Per the README — ~90% of the value at ~10% of the cost vs inline widgets.
5. **Widget edits → text splice.** Use tree-sitter byte spans to splice the new value into the buffer, then trigger a save (which fires `notify`, which reloads ECS, which updates the viewport).
6. **Migrate dylib hot-reload** from `crates/ocarina-editor/src/plugin.rs`. The reload trigger (file save → rebuild → reload) integrates with Lapce's task system rather than the current bespoke filesystem-watch.

**Exit criteria:** click on `position = [1, 2, 3]` in a TOML file, see vec3 scrubbers in the lens panel, drag them, watch the file change and the viewport update.

## Phase L4 — Run/build pipeline  (3–5 days)

Repurpose what was Lapce's "Run and Debug" panel.

1. **Add an "Ocarina" toolbar group** with four buttons:
   - **Live Preview** — compile gameplay crate as `cdylib`, hot-reload into the embedded viewport. Already what the editor does today.
   - **Standalone** — `cargo build` + `ocarina bake` + spawn `ocarina-runtime` as a separate native window. Uses `WgpuRhi::new_for_window` (Phase 1c).
   - **Bake only** — produce the asset archive without running. For CI / shipping.
   - **Bake + Ship** — full release build per the Phase 6 plan in README. Stub for now; wire when Phase 6 lands.
2. **Wire to `ocarina-bake` CLI** via `Command::spawn`. Streams stdout/stderr into a Lapce output panel.
3. **Default `tasks.toml`** for new Ocarina projects so `Ctrl+Shift+B` (Lapce's task hotkey) runs the right thing.

**Exit criteria:** "Run Standalone" launches a separate window with the current scene playing.

## Phase L5 — Branding, polish, defaults  (1 week)

1. Replace Lapce icons / logo / app name everywhere.
2. Default theme tuned for Ocarina (the existing UI already has a color identity; port to a Lapce theme file).
3. **Project-aware mode.** When the user opens a folder containing `ocarina.toml`, auto-show Viewport + Lens + Files + Terminal. When opening a non-Ocarina folder, behave like vanilla Lapce. This is what makes the editor feel engine-native without locking out general-purpose editing.
4. Default keybindings for engine-specific actions (toggle viewport, frame-selected entity, run standalone).
5. Bundle `ocarina-lsp` so the editor speaks TOML completion/hover/diagnostics out of the box (also published as a Lapce plugin for external Lapce users).

**Exit criteria:** an outsider opening Ocarina cannot tell at a glance that it was built on Lapce. They open a project folder, see the engine UI, and start working.

## What we keep from Lapce

| | Why |
|---|---|
| `lapce-proxy` | Remote dev architecture — load-bearing for the team-development roadmap |
| `lapce-rpc` | Local↔proxy IPC |
| `lapce-core` | Language/syntax configs |
| `floem-editor-core` | Rope buffer, multi-cursor, modal/vim, transactional edits |
| `floem` text-editor view | The competitive code-editor primitive |
| Terminal panel | Already alacritty-backed (same crate as our current code) |
| Find-in-files | `ignore` + `grep-*` |
| Command palette + fuzzy finder | Standard editor furniture |
| Settings system | Reuse for Ocarina settings |
| Themes system | Reuse + ship Ocarina default theme |
| WASI plugin host | Future-proof for game-component plugins |
| LSP plumbing | Hosts `ocarina-lsp` + rust-analyzer simultaneously |
| Git plumbing | `git2` integration in proxy, file-tree decorations |

## What we strip from Lapce

| | Reason |
|---|---|
| Plugin marketplace UI | We curate. Keep the WASI host |
| About / welcome / branding | Replace with Ocarina equivalents |
| Lapce-specific docs links | Replace |
| Source-control panel (heavy) | File-tree git decoration is enough for v1 |
| DAP debugger panel | Defer; runtime profiling will be its own panel |

## Risks and fallback paths

**Risk 1 (P0): Floem `WgpuView` doesn't work.** Most likely failure: Floem's wgpu device isn't shareable in the way our RHI assumes (e.g., it's behind a `RefCell`/private state that resists cross-thread access).

- *Fallback A:* run Ocarina's RHI on its **own** wgpu device, render to a `wgpu::Texture`, then export to a CPU image (`wgpu::Buffer` readback) and have Floem `draw_img` it. **Bad** — GPU readback per frame at viewport resolution.
- *Fallback B:* skip Lapce, use `floem` + `floem-editor-core` directly as libraries (Path B from earlier research). Build only the panels we need on Floem. Loses Lapce's terminal/git/palette but keeps the competitive code editor primitive. ~3 months.
- *Fallback C (last resort):* stay on egui, wrap a custom code editor widget on `egui::TextEdit::multiline` with a custom layouter and tree-sitter highlighting. ~4–8 months for a credible Rust editor.

**Risk 2 (P1): Lapce upstream churn.** Mitigation: pin SHAs, rebase quarterly. ~1 commit/week upstream so this is a 1–2-hour chore per quarter.

**Risk 3 (P1): Floem API churn.** Mitigation: Lapce already pins Floem to a tested rev; we inherit that pin.

**Risk 4 (P2): WASI plugin host conflicts with our dylib gameplay model.** They don't actually conflict — WASI is for editor plugins (ours and third-party); native dylibs are for gameplay. But naming + UX needs to keep them distinct. Defer.

## Open questions / TBD

- Subtree vs submodule vs vendored copy for `third_party/lapce`. Subtree gives single-clone simplicity; submodule gives clearer upstream-tracking. Decide before L0.
- Whether to upstream the `WgpuView` patch to Floem. Probably yes — lowers our rebase risk and benefits Floem.
- Whether to publish `ocarina-lsp` to the Lapce plugin marketplace once we've stripped the marketplace UI from our own build (we can still publish *from* our repo to the marketplace catalog).
- DAP integration for runtime debugging — useful eventually but out of scope for the migration.

---

# Remote development roadmap

Standalone roadmap. R1 is inherent in the current architecture; R2 ships day-1 of the Lapce fork; R3+ are real engineering on top.

## R1 — Filesystem-as-truth + git-pull hot-reload  (DONE — inherent in Phase 2)

Already shipped. `notify` → re-parse → ECS update. `git pull` is just a batch of file events. Other engines cannot do this without inventing a custom protocol because their scenes are not files in any meaningful sense; we get it for free.

## R2 — Lapce remote-dev proxy  (FREE on day-1 of fork)

Inherited from Lapce. SSH into a remote machine, `lapce-proxy` runs there, all file/LSP/terminal/search/git operations execute remotely; the local editor renders the UI. Multiple devs can SSH into the same machine, each with their own proxy instance. They share the filesystem, so any save by one dev fires `notify` on the next dev's proxy → reload on their viewport.

This already gives us "team development" in a meaningful sense: edit your own files, save, everyone else sees the change in their viewport. Conflict UX is the same as VS Code/Zed today (file-changed-on-disk prompt).

**No work required.** Falls out of the fork.

## R3 — CRDT live text sync for whitelisted file types  (4–8 weeks)

Save-based sync (R2) → keystroke-based sync. Files of type `.toml` and `.rs` use a CRDT layer on top of the proxy's file watcher. Other file types (binary assets, shaders for now) keep R2's save-based behavior — users coordinate via Git.

- **CRDT lib:** prefer `loro` or `automerge` (mature, Rust-native). `xi-rope` is Lapce's existing rope and has CRDT-friendly primitives but predates modern alternatives.
- **Sync transport:** initially through `lapce-proxy` itself — devs connected to the same proxy have their CRDT ops broadcast via the existing RPC channel. Later: a dedicated relay if proxies become a bottleneck.
- **Whitelist enforcement:** by extension, configurable via project settings.
- **Conflict UX:** standard Google-Docs semantics. If two devs type into the same line, they get character-merge. If two devs change the same value (e.g., both edit `position = [...]`), the result is a CRDT merge that may produce semantically unintended values — by design, devs are expected to coordinate via the next layer (R4).

## R4 — Presence indicators  (1–2 weeks)

- **Cursor presence in files** — Figma/Google Docs style. Each user's cursor + selection painted in the buffer with their color.
- **Viewport object highlights** — when a teammate's cursor sits inside an entity's component, that entity gets a colored outline in the viewport. "Alice is editing the cave" becomes immediately visible.
- **No locking.** Soft signal only — devs can still edit the same object simultaneously. The highlight is a hint to "hop on a call before you both drag the same gizmo."

## R5 — Local port forwarding for game-server playtest  (1–2 weeks)

Devs co-developing a multiplayer game want to playtest together. The game server runs on the remote machine; each dev's local game client connects to it through a port forward.

- "Run Server" button on the Run panel: spawns the server binary on the remote, forwards a configurable local port through SSH.
- Each dev's local game connects to `localhost:<port>` thinking it's a LAN game. Server actually runs once on the remote, all clients connect to the same instance.
- Reuses Lapce's existing SSH session; no new transport.

## R6 — Remote screen-share for full co-op playtest  (research, way down the line)

For games where the testing experience requires shared input (couch co-op, controller-driven games):

- Stream the remote game's framebuffer to local windows, one per dev.
- Forward each dev's keyboard/mouse/controller input as if they were local to the remote.
- Effectively VNC + multi-input, scoped to the running game window.

Per the user: "way down the line." Documented here so the architecture stays compatible (no decision now should preclude this layer later).

## Roadmap dependencies

```
R1 (done) ─┬─> R2 (free)  ─┬─> R3 (CRDT)    ──> R4 (presence)
           │                 └─> R5 (port fwd)
           └─────────────────────> R6 (screen share, independent)
```

R3, R5, R6 can be developed in parallel once R2 is in place.
