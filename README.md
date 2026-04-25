# Ocarina

A filesystem-first, terminal-first, agent-friendly game engine for developers who want to write code, not click through node graphs.

## Vision

Ocarina inverts the Unity/Godot model. Instead of a scene being a proprietary tree owned by an editor, **a scene is a folder, a component is a file, and the filesystem is the single source of truth**. The engine is a thin runtime that loads those files into an ECS and renders them. The editor is a thin IDE that helps you author those files.

This means:

- A `git pull` instantly changes the running scene.
- A coding agent (Claude Code, etc.) edits scenes and components the same way it edits code — by editing text files.
- Humans edit the same text files, optionally through inline value widgets (color pickers, vec3 scrubbers, gizmos) that splice values back into the file.
- There is no separate scene database, no `.unity` blob, no import pipeline per asset. Scenes are diffable, mergeable text.

Ocarina is aimed at engineers who are comfortable writing shaders, DSP, and systems from scratch, and who want a lightweight engine they can actually ship fast binaries with.

## Two programs: editor and runtime

Ocarina is not one program, it is two:

- **The editor** is a developer IDE. It watches the filesystem, parses text files with tree-sitter, hot-reloads components into a live ECS, renders preview viewports, hosts an embedded terminal, and lets humans and agents edit source files. The editor is what you run during development.
- **The runtime** is the shippable engine that ends up in a player's hands. It has no filesystem watcher, no tree-sitter, no KDL parser, no editor UI, no terminal. It loads **baked assets** — a compiled scene blob, pre-compiled shader bytecode, GPU-ready vertex buffers, serialized DSP graphs — and runs the game.

They share the **RHI**, the **ECS component schema**, the **audio DSP node definitions**, and the **gameplay code**. They do not share the editor's machinery. A shipped game is a native binary bundling the runtime plus one baked asset archive; it is small and fast and has none of the editor's dependencies linked in.

The **bake step** is the bridge. It runs source files through parsers, shader compilers, mesh processors, and DSP graph serializers to produce the runtime's asset archive. Conceptually: editor consumes source → bake produces runtime assets → runtime consumes runtime assets. Source files never ship.

## Design principles

1. **Files are truth.** In-engine state is a read-derived cache of disk. Filesystem events drive the engine, never the reverse — except at explicit save/flush points.
2. **Text is the editing surface.** There is no parallel "scene editor state." Dragging a gizmo is a text splice into the transform file's position value. Widgets are lenses over spans of text.
3. **No undo system.** The text buffer's native undo covers keystrokes. Git covers history. We do not build a scene-graph undo stack.
4. **Coder-first.** Shaders in HLSL text files. Audio DSP in code. Components are small text files with minimal schema. No visual graph editors.
5. **Ship real binaries.** The runtime is a native binary, not a web app, not a VM. Performance matters.
6. **Abstraction only where it buys replaceability.** The RHI is abstracted because GPU backends will change. The ECS is not abstracted because we're picking one. Everything above the RHI talks to the RHI trait, not to the backend.

## Stack

| Layer | Choice | Scope | License |
|---|---|---|---|
| Language | Rust | both | — |
| GPU (initial backend) | `wgpu` → Vulkan / Metal / DX12 native | both | MIT / Apache-2.0 |
| GPU abstraction | Custom RHI trait; wgpu is backend #1 | both | — |
| ECS | `bevy_ecs` as a standalone crate (no Bevy-the-engine) | both | MIT / Apache-2.0 |
| Shader authoring / compiling | HLSL → DXC → DXIL / SPIR-V | editor + bake | MIT |
| Audio I/O | `cpal` | runtime | Apache-2.0 / MIT |
| Audio DSP graph | `fundsp` + custom nodes, exposed as ECS components | runtime | MIT / Apache-2.0 |
| Asset archive format | Custom binary (bincode-serialized ECS + blob table) | bake + runtime | — |
| Editor UI | `egui` + `egui_dock` | editor | MIT / Apache-2.0 |
| Terminal | `alacritty_terminal` | editor | Apache-2.0 |
| Filesystem watcher | `notify` | editor | CC0 / MIT |
| Parsing (source files) | `tree-sitter` with per-format grammars | editor + bake | MIT |
| Component file format | TOML (section headers = component type names) | editor + bake | — |

"Scope" indicates whether a dependency is linked into the runtime binary (what players ship), the editor binary (what developers use), the bake tool, or all of them. Keeping the runtime column short is a hard goal — every dependency there ships to players.

Rust is chosen for ecosystem fit on the editor side: the embeddable terminal (`alacritty_terminal`), in-process UI (`egui`), and ECS (`bevy_ecs`) we want are all Rust-first. The RHI design below is language-neutral — if the stack is ever moved to C++ the architecture does not change.

## Data model

### Folder = entity, files = component bundles

Entity identity comes from the folder. Components come from `.toml` files inside the folder. A file may declare one component or many; each top-level `[section]` in a file declares one component, where the section name matches a registered component type.

```
my_scene/
  _scene.toml              # scene-level: [skybox] [ambient_light] [fog]
  player/
    player.toml            # [transform] [character_controller] [camera] [input]
    hitbox/
      hitbox.toml          # [collider] [hitbox_tag]
  cave/
    cave.toml              # [transform] [mesh] [collider]
    ambience/
      ambience.toml        # [reverb] [audio_emitter]
```

Example file content:

```toml
# cave/cave.toml
[transform]
position = [0, 0, 0]

[mesh]
source = "meshes/cave.obj"

[collider]
shape = "mesh"
```

Rules:

- Folder hierarchy is entity hierarchy.
- Any `.toml` file in a folder contributes components to that folder's entity.
- Each top-level `[section]` declares one component whose type matches the section name (snake_case of the Rust struct).
- Missing fields use the component's defaults.
- **Duplicate component across files in the same folder is an error** — ECS does not permit two of the same component per entity. Surfaced as an LSP diagnostic.
- For multiple instances of a component type on a logical "thing" (a character with hitbox and hurtbox colliders), use **child folders** — each child folder is its own entity. This preserves ECS purity and makes the hierarchy inspectable in the file tree.
- Files starting with `_` are metadata about the folder, not entity components. `_scene.toml` is the scene's own components; `_prefab.toml` is reserved for future inheritance.
- Filenames are free-form human labels, not semantic identifiers. Author chooses the grouping.

**Prefabs are just files.** Since one file can contain many component sections, a "prefab" is a multi-component `.toml`. Copy the file into another folder to instantiate. Overrides happen by editing the local copy. No separate prefab system in v1; if copy-paste-with-overrides becomes painful we add an `[_inherit = "path/to/prefab.toml"]` directive with per-section override semantics later.

### Filesystem → ECS is one-way (editor only)

In the **editor**:

1. `notify` watches the project dir.
2. On any file event, the affected file is re-parsed.
3. The parse result replaces the component in the ECS world.
4. Unparseable files are surfaced as errors in the editor; the previous valid state remains in the ECS.

The ECS never writes to disk. There is no "save" from the ECS's perspective.

In the **runtime**:

1. At startup, the asset archive is loaded into memory.
2. The ECS world is deserialized directly from the archive — no parsing, no validation.
3. The world is then driven by game logic only. Components change as gameplay changes them, not because of any filesystem event.

The runtime has no concept of source files. The bake step is what bridges the two.

## Component authoring

A developer adds a new component by writing a Rust struct with `#[derive(Component)]`. From that single declaration the engine derives:

- The ECS registration (`bevy_ecs` Component impl).
- The TOML parser for files containing this section (`serde`).
- Schema metadata for the LSP (field names, types, defaults, docs, widget hints).
- Bake-time serialization into the runtime asset archive.

```rust
use ocarina::prelude::*;

/// Positions, rotates, and scales an entity in 3D space.
/// Transforms compose hierarchically through parent folders.
#[derive(Component, Serialize, Deserialize)]
pub struct Transform {
    /// World position in meters.
    #[ocarina(widget = "vec3")]
    pub position: Vec3,

    /// Euler rotation in degrees (XYZ order).
    #[ocarina(widget = "euler_rotation")]
    pub rotation: Vec3,

    /// Scale factor per axis.
    #[ocarina(widget = "vec3", default = "[1.0, 1.0, 1.0]")]
    pub scale: Vec3,
}
```

The TOML section name is the snake_case of the struct name (`CharacterController` → `[character_controller]`), overridable with `#[ocarina(name = "...")]`.

One `.rs` file can declare many components. `physics.rs` might declare `Collider`, `RigidBody`, and `Velocity` together — the macro picks each one up independently.

Built-in components ship with `ocarina-core`. Project components are declared in the user's gameplay crate. Both feed the same schema registry.

A scaffolding command creates the boilerplate:

```
ocarina new component CaveReverb
```

Generates `src/components/cave_reverb.rs` with a stub struct, adds it to the components module, and triggers a schema regeneration.

## Scripting and gameplay code

User gameplay code is native Rust, compiled into the shipped binary. No VM, no embedded scripting language.

To reconcile "native Rust" with "editor preview must iterate fast," the editor uses **dylib hot-reload**:

- The user's gameplay crate(s) compile as `cdylib` in editor mode.
- The editor loads the dylib via `libloading`, watches source files with `notify`, and rebuilds + reloads on change.
- Rust incremental compile on a single crate is typically 1–3 seconds for small changes.
- In ship mode, the same crate compiles as `staticlib` and statically links into the runtime binary. Zero runtime overhead, native speed.

```
ocarina-editor  ──loads──> game_logic.dylib   (editor mode, reloadable)
                            └── rebuilds on save

ocarina-runtime ──links──> libgame_logic.a    (ship mode, static)
```

Users with many gameplay systems can split into multiple crates (e.g., `game_player`, `game_enemies`, `game_ui`) for finer reload granularity — a change in AI code only reloads the enemies dylib.

ABI constraints for reloadable code:

- The dylib exports a C ABI entry point: `fn register_systems(world: *mut World, scheduler: *mut Scheduler)`.
- No Rust types (`Vec`, `String`, `HashMap`) cross the FFI boundary — only opaque pointers and FFI-safe types.
- Editor and dylib must be compiled with matching rustc versions.
- State stored inside the dylib is lost on reload. Put all state in the ECS world. This is healthy discipline anyway.

Libraries: `hot-lib-reloader` for the reload machinery, or `libloading` if we roll it ourselves. Handmade Hero (C) and Jonathan Blow's Jai both demonstrate production-grade versions of this pattern.

## Language server

Ocarina ships an LSP that provides syntax awareness, completion, diagnostics, and hover docs for `.toml` component files. Both the in-house editor and any external editor (VS Code, Zed, Helix, Neovim) use the same server binary.

What it provides:

- Section completion: typing `[` offers every registered component type.
- Field completion: inside a section, offers that component's fields with types and defaults.
- Hover documentation: shows the Rust doc comment from the component's source.
- Diagnostics: unknown component, unknown field, wrong value type, duplicate component in folder.
- Go-to-definition: on a section name, jump to the `.rs` file declaring the component.
- Semantic highlighting via LSP semantic tokens (known component vs typo).

Schema pipeline:

```
ocarina-core     #[derive(Component)]   built-in schemas
                         │
                         ▼
                 inventory/linkme table
                         │
                         ▼
user gameplay    #[derive(Component)]   project schemas
                         │
                         ▼
              cargo build → build script
                         │
                         ▼
        target/ocarina/schema.json
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
   in-process LSP             ocarina lsp --project .
   (Ocarina editor)           (VS Code / Zed / Helix / agents)
```

A build script collects all `#[derive(Component)]` types from the compiled crate (via `inventory` or `linkme`) and emits `target/ocarina/schema.json`. The LSP watches that file; rebuild of the gameplay dylib regenerates the schema, and completions update immediately.

Built-in components bake their schema into the LSP binary directly; project schemas load from the JSON. Both merge into one completion set.

Deployment modes:

- **In-process** (Ocarina editor): `ocarina-lsp` linked as a library, queried directly by the editor panes. No RPC.
- **Stdio** (external editors): `ocarina lsp --project .` speaks LSP over stdin/stdout. Agents like Claude Code can also query it to discover components before editing.

This is the same information source that backs the editor's lens panel in Phase 3 — widgets render LSP hover/completion responses as interactive controls.

### Editor → filesystem is through text splices

Editor interactions that look like they're editing objects (dragging a gizmo, picking a color, scrubbing a value) are all implemented as **text splices into the file buffer**, using byte spans obtained from `tree-sitter`. The text buffer flushes to disk on save (or debounced on drag-release). The disk write fires a filesystem event, which re-parses, which updates the ECS. Single dataflow.

### Conflict policy

Filesystem always wins. If the file on disk changes while the editor has an unsaved buffer for it, the editor surfaces a standard "file changed on disk, reload / keep / diff" prompt — the same UX as VS Code or Zed. This is not a scene-graph conflict; it is a text-buffer conflict. No special engine machinery.

## RHI abstraction

The renderer talks to a `Rhi` trait. wgpu is backend #1. The trait is designed so a future Vulkan, D3D12, or The Forge backend slots in without changing any engine code above it.

### Surface level

Modern explicit: device, queue, command buffer, render pass, compute pass, pipeline, buffer, texture, sampler, bind group (with explicit bind group layouts as first-class objects), shader. No materials, no meshes, no scene concepts — those live above the RHI.

### Key design decisions

- **Auto resource state tracking inside each backend.** Engine code never writes barriers. On wgpu this is free. On future VK/D3D12 backends we implement a tracker (~500 lines).
- **Opaque generational handles** (`BufferHandle { slot: u32, gen: u32 }`) into per-type pools. Copy-cheap, lifetime-safe, no backend types exposed.
- **Deferred destruction.** `destroy(handle)` enqueues; actual release happens after `MAX_FRAMES_IN_FLIGHT` frames have retired.
- **HLSL as canonical shader language.** DXC compiles offline or at runtime to SPIR-V (VK / wgpu) and DXIL (D3D12). SPIRV-Reflect gives bind group layouts. WGSL is an output target, not an input format.
- **Capability flags** (`mesh_shaders`, `ray_tracing`, `bindless_arrays`, `timestamp_queries`, `max_push_constant_size`, ...) for progressive feature enablement. Engine code queries capabilities; backends advertise what they support. Adding mesh shader support to a D3D12 backend does not change the trait — only adds a capability bit and a new entry point.

### Anti-leak guardrail

Before shipping the wgpu backend, we implement a **null backend** (returns fake handles, counts calls, used for unit tests). If the null backend can't implement the trait without doing anything wgpu-specific, the trait has wgpu-isms that need to be removed. This is the single most effective check against building an abstraction that only fits its first implementation.

### What the RHI does not handle (yet)

Deferred to later layers or later versions:

- Render graph / automatic barrier minimization (engine-layer concern; v1 uses a simple pass list).
- Transient / aliased memory.
- Multi-queue (async compute, copy queue). Single graphics queue for v1.
- Multi-GPU.

## Editor architecture

Three panes, all talking to the same underlying text buffers and ECS world.

```
+------------------+----------------------------+------------------+
|                  |                            |                  |
|   File browser   |      Preview viewport      |   Lens panel     |
|   (tree)         |      (wgpu render target)  |   (widgets for   |
|                  |                            |    value under   |
+------------------+                            |    cursor)       |
|                  |                            |                  |
|   Text editor    |                            |                  |
|   (active file,  |                            |                  |
|    tree-sitter   |                            |                  |
|    highlighted)  |                            |                  |
+------------------+----------------------------+------------------+
|                         Terminal                                 |
|                  (alacritty_terminal + PTY)                      |
+------------------------------------------------------------------+
```

### Preview viewport

A render target that displays **whatever is relevant to the currently selected file**:

- Selected a scene/entity → 3D scene rendered from the ECS, camera-controllable.
- Selected a shader file → preview primitives with that shader bound, in real time.
- Selected a mesh file → mesh viewer.
- Selected an audio DSP file → signal flow viz + scope / spectrogram of its output.

Gizmo interactions in the 3D viewport compute new values and splice them into the active text buffer at the correct span — the viewport is another "lens" writing to text, same contract as the lens panel.

### Lens panel

Reads the cursor position in the text editor, finds the AST node under the cursor via tree-sitter, and renders the appropriate widget (color picker, vec3 scrubber, bezier curve editor, etc.). Widget changes splice text into the buffer at the node's byte span. This is the ergonomic win of a scene inspector with none of the dual-state cost.

Inline widgets rendered directly in the text stream are deferred — they are the "v2" upgrade. The lens panel achieves ~90% of the ergonomics at ~10% of the implementation cost.

## Audio architecture

Audio gets first-class treatment, parallel to the shader pipeline. If an entity can have a custom fragment shader, an entity can have a custom DSP graph.

- The audio engine runs a DSP graph per frame, not a fixed mixer.
- DSP nodes are ECS components. `reverb.kdl`, `filter.kdl`, `distortion.kdl`, `custom_dsp.kdl` (pointing to a Rust or WASM DSP source file) attach to entities.
- Spatialization and buses are implicit from the scene hierarchy — a reverb on the `cave` entity is applied to audio emitted by children of `cave`, and routed in/out based on collider triggers.
- The audio buffer is as exposed as the render command stream. Users can write custom DSP nodes without touching the engine core, same as they write custom shaders.

`cpal` handles device I/O; `fundsp` provides the base DSP graph and primitives. Custom nodes are either native Rust (compiled in) or WASM (for hot-reloaded user DSP, later).

## Implementation roadmap

### Phase 0 — Foundations (days)

Cargo workspace, split along the editor/runtime boundary from day one:

- `ocarina-rhi` — trait, handles, Desc types. (shared)
- `ocarina-rhi-wgpu`, `ocarina-rhi-null` — backends. (shared)
- `ocarina-schema` — ECS component definitions, serde impls. (shared)
- `ocarina-runtime` — headless engine: loads asset archive, ticks ECS, renders, runs audio. No editor deps. (runtime-only)
- `ocarina-bake` — CLI + library: source files → asset archive. (tooling)
- `ocarina-editor` — IDE app. Depends on everything above. (editor-only)

Also in Phase 0:

- `Rhi` trait, all Desc structs, handle types, capability flags.
- Null backend. Unit tests against it.
- `winit` window + `egui` docked layout with placeholder panes, in `ocarina-editor`.

**Lint rule:** `ocarina-runtime`'s Cargo.toml must not depend on `egui`, `notify`, `tree-sitter`, `alacritty_terminal`, or any KDL parser. This is the discipline that keeps shipped games small.

### Phase 1 — Renderer and viewport (days) ✅

- wgpu backend implementing `Rhi` (resource pools with generational handles, auto-state-tracking via wgpu, deferred destruction queue, full command-list interpreter covering render/compute passes + copies + debug markers).
- Triangle rendered through the RHI into an offscreen texture, displayed in the Viewport pane via `egui-wgpu::Renderer::register_native_texture`.
- Editor binds `WgpuRhi::from_existing` to the `wgpu::Device`/`Queue`/`Adapter` eframe already owns, so the editor has one wgpu context (no resource-sharing dance).

### Phase 1b — Renderer polish + terminal (days) ✅

- **Procedural Triforce mesh** — three thin triangular prisms arranged as the classic Zelda motif, flat-shaded per face, rendered through the full RHI pipeline (vertex buffer + index buffer + uniform buffer + bind group + depth attachment). 54 vertices / 72 indices / 24 triangles. Keeps with the Ocarina theme; no external asset or licensing dependency.
- **Orbit camera** — yaw/pitch/distance driven by mouse drag and scroll inside the Viewport pane. View-projection matrix uploaded to a uniform buffer each frame.
- **Lit WGSL shader** — hemispheric ambient + Lambertian + mild fresnel rim for the metallic-gold feel, all in `shaders/lit.wgsl`.
- **Embedded terminal** — `alacritty_terminal` spawns a PTY-backed shell; its `EventLoop` runs on its own thread and feeds VT state into a shared `Term`. Per-frame snapshot rendered as monospace text in the Terminal pane; keyboard events (text, arrows, enter, backspace, ctrl-letter) dispatched back to the PTY via `Msg::Input`. Terminal is best-effort — PTY failure falls back to a placeholder pane without refusing to launch the editor.
- **SPIR-V passthrough** — the wgpu backend now accepts `ShaderBytecode::Spirv` directly (wgpu's `spirv` feature enabled). Any HLSL authoring flow that emits SPIR-V (DXC via `hassle-rs` when wired up) feeds in cleanly; the RHI doesn't need to know where the SPIR-V came from.

### Phase 1c — Renderer + terminal completion (days) ✅

- **HLSL authoring pipeline** — `ocarina-shader` crate with `compile_hlsl_to_spirv` on top of `hassle-rs`. The viewport attempts to compile `shaders/lit.hlsl` via DXC at startup; successful compile feeds SPIR-V into the wgpu backend via `ShaderBytecode::Spirv`. If DXC isn't on the system library path, we fall back to the equivalent `shaders/lit.wgsl` without a user-visible failure. `ocarina_shader::dxc_available()` is the probe.
- **Swapchain on `WgpuRhi`** — `WgpuRhi::new_for_window(window, w, h, present_mode)` owns its own `Instance`/`Adapter`/`Device`/`Queue`, creates a `wgpu::Surface<'static>` from an `Arc<W: HasWindowHandle + HasDisplayHandle>`, and stores it in the swapchain pool. `acquire_next_image` / `resize_swapchain` / `present` are real: they drive `get_current_texture`, `configure`, and `SurfaceTexture::present` respectively, and hand the backbuffer to user code as a regular `TextureHandle` (retired on `present`). A `standalone_triangle` example demonstrates the standalone-runtime path; `cargo run -p ocarina-rhi-wgpu --example standalone_triangle` pops a winit window with a triangle rendered via `WgpuRhi`.
- **Terminal polish** — per-cell foreground / background colors (xterm-256 palette: ANSI-16 + 6×6×6 color cube + 24-step grayscale), BOLD brightens to the high-intensity palette, DIM blends toward background, INVERSE swaps fg/bg, UNDERLINE / STRIKEOUT paint horizontal strokes. Mouse wheel over the pane drives `Term::scroll_display(Scroll::Delta)` for scrollback; typing a key snaps back to `Scroll::Bottom`. Background runs are painted as contiguous rects to keep paint calls low. PgUp / PgDn are wired to `\x1b[5~` / `\x1b[6~`.

### Phase 2 — File-first ECS (days) ✅

- **`ocarina-schema`** — `ComponentSchema` / `FieldSchema` / `FieldType` / `Widget` types; two `inventory` collectors (`ComponentRegistration` for the metadata, `ComponentLoader` for the TOML → ECS dispatch); helpers `iter_schemas`, `find_loader`, `collect_schemas`; `load_component::<T>()` that every derive-generated loader calls through.
- **`ocarina-derive`** — `#[derive(Component)]` parses the struct, its doc comment, and `#[ocarina(widget = "…", default = "…")]` field attrs, then emits: the `bevy_ecs::Component` impl, a lazy `ComponentSchema` builder, and the `ComponentLoader` entry. `#[ocarina(name = "custom_section")]` on the struct overrides the snake_case section name.
- **Built-in components** — `ocarina-core` now ships `Transform`, `BasicMesh`, `Camera`, and `Name` authored with the derive. Their schemas appear in the CLI's JSON output.
- **`ocarina-scene`** — folder walker (`reload_full`), single-file reloader (`reload_file`), `notify::recommended_watcher` wrapper emitting a normalized `SceneEvent::{Changed, Removed, Rescan}` stream. `Scene` holds a `bevy_ecs::World` + `dir_entity` / `file_entity` maps. Reserved `_`-prefixed sections are skipped.
- **Editor integration** — `OcarinaApp` opens a project at startup (`$OCARINA_PROJECT` or `./projects/example-scene`), pumps watcher events each frame, shows entity/component counts in the menu bar and a live file tree in the Files pane.
- **Example scene** — `projects/example-scene/` with `player/` (plus nested `hitbox/`), `cave/`, `camera/`, and a `_scene.toml` at the root; multi-component files, hierarchical folders, and component defaults all demonstrated.
- **`ocarina schema`** — CLI subcommand that serializes every registered schema to JSON. Optional `--output <path>`; pretty-printed by default, `--compact` for single-line. Consumed by Phase 2.5's LSP.
- **Gameplay dylib scaffolding** — `ocarina-gameplay-api` defines the FFI contract (`PluginInfo` + `ocarina_plugin_init` symbol + ABI version check); `examples/hello-gameplay` is a `cdylib` that implements it; editor `plugin::try_load_default` scans `target/{debug,release}` and loads via `libloading`. Successful loads surface in the menu bar.

**Linker-keepalive note** — Rust's linker drops crates whose symbols aren't referenced by the final binary. This kills `inventory::submit!` registrations silently. The fix is a `use ocarina_core as _;` in every binary that expects built-in components to be registered. The CLI and editor both do this.

### Phase 2.5 — Language server (days) ✅

- **`ocarina-lsp`** — `tower-lsp` 0.20 backend. `Backend::new` sources schemas from `inventory::iter` (everything `#[derive(Component)]`d in the linked binaries); `Backend::with_schema_json` is the offline alternative for editors that didn't link the user's components. Per-document state holds the source text + the most recent `toml::de::Error`, re-parsed on every `did_change`.
- **Completion** — typing `[tra…` offers component sections; typing inside a known section's body offers that component's fields. Each item carries the doc comment + Rust type as `MarkupContent`. Filters by prefix.
- **Hover** — cursor on a `[section]` shows the component's full markdown (description + field list with defaults); cursor on a `field = …` shows the field's doc + Rust type + declared default.
- **Diagnostics** — TOML parse errors (with byte-range from `toml::de::Error::span`), unknown component sections, unknown fields on known components (with the list of known field names suggested in the message). Re-published on every change.
- **`ocarina lsp`** — stdio server (`tokio` runtime, `Server::new(stdin, stdout, socket).serve(service)`). Wires straight into VS Code, Zed, Helix, Neovim. Smoke-tested with a Python harness that drives `initialize → didOpen → completion → hover`.
- **In-process API** — `analyze_completions` / `analyze_diagnostics` / `analyze_hover` are public functions over `LspDocument` + `LspSchemaIndex` so the editor can drive the same analyzer without going through stdio.
- **`ocarina new component <Name>`** — scaffolds `src/components/<snake_case>.rs` with a `#[derive(Component)]` stub, validates that `Name` is a single PascalCase identifier, and appends `pub mod <snake>;` to `src/components/mod.rs` if one already exists.

### Phase 2.5 follow-ups ✅

- **Go-to-definition** — `#[derive(Component)]` now stamps `source_file` (absolute path, captured by joining `std::env::current_dir()` at proc-macro expansion time with `file!()`) and `source_line` onto each `ComponentSchema`. The LSP advertises `definitionProvider` and resolves both section headers and field names to the `#[derive(Component)]` line in the declaring `.rs` file.
- **Wrong-value-type diagnostics** — every `field = value` is checked against the field's `FieldType`. Scalars match by TOML primitive (Bool/Integer/Float/String); Vec2/3/4 and Quat require numeric arrays of the right length; Color accepts a hex string or a 3-/4-element numeric array; Float accepts integer literals; `Other` is unchecked.
- **Duplicate component in folder** — `folder_conflict_diagnostics` reads sibling `.toml` files in the document's parent directory and warns when the same component section appears in two files (since folder = entity, both would attach to the same ECS entity). Skips `_`-prefixed files (folder metadata).

### Phase 3 — Text editor and lenses (week+)

- Text editor pane with tree-sitter TOML highlighting, semantic tokens from LSP.
- LSP client inside the editor.
- Lens panel: cursor → AST node → LSP query → widget dispatch. First widgets: color, vec3, float scrubber.
- Save flow: buffer → disk → notify → re-parse → ECS update.
- "File changed on disk" reload/keep/diff UX.

### Phase 4 — Viewport as editor ✅

- **Entities render in the viewport.** Every ECS entity with a `Transform` is drawn as a tinted unit cube (the `BasicMesh.tint` is honored when present). Per-instance data (model matrix + tint) lives in a single dynamic-offset UBO so a frame is N draws against one `set_bind_group(1, instance_bg, &[offset])`. The Triforce procedural mesh is kept in `mesh.rs` as reference but no longer drawn.
- **Transforms compose hierarchically through parent folders.** `Scene::world_transforms` walks `dir_entity` to build a parent map and folds each entity's local `Transform` (`Transform::matrix()`) with every ancestor's, with missing-Transform folders passing through as identity. The viewport draws, picks, and gizmo-anchors in world space. Gizmo write-back rotates the world-space drag delta back through `parent_world_inv` before persisting, so the file always stores parent-relative position/rotation/scale even when the dragged entity sits inside a translated/rotated parent.
- **Camera controls.** Left-drag orbits, scroll zooms, **middle-drag** or **Shift + left-drag** pans (`OrbitCamera::pan` moves the target laterally in view space, scaled by `distance` so feel is consistent at any zoom).
- **Entity picking.** Clicking the viewport shoots a world-space ray (`OrbitCamera::ray_through_uv`) and ray-AABB tests every entity's transformed unit-cube bounds. Closest hit → selection; click on empty space clears. Selection state lives on `OcarinaApp::selection` and is shared across panes.
- **Translate gizmo.** Three colored axis bars (X red, Y green, Z blue) render at the selected entity. Clicking and dragging a bar projects the cursor ray onto that axis line (`closest_point_on_axis`) and applies the delta to the entity's `Transform.position` in real time.
- **Write-back to disk.** On drag-end, `transform_io::write_transform` opens the entity's `.toml` (resolved via `Scene::file_entity` reverse-lookup, falling back to a directory scan for any file that has a `[transform]` section) and rewrites only `position`/`rotation`/`scale` via `toml_edit` — comments, ordering, and other sections survive untouched. The watcher then replays the change through the normal load path.
- **Shader preview.** Selecting a `.hlsl` file in the file browser compiles it via `ocarina-shader` (DXC → SPIR-V) and rebuilds the viewport pipeline using its `vs_main`/`fs_main`. Compile errors surface as a red overlay; the pipeline keeps the previous shader on failure. The previewed file is polled for `mtime` changes each frame, so saving in your editor hot-reloads without needing to re-click. Selecting a non-shader (or no file) reverts to the built-in lit shader.

### Phase 5 — Audio (week+)

- `cpal` output with a `fundsp` graph driven from ECS components.
- `reverb.kdl`, `gain.kdl`, emitter/listener components.
- Collider-triggered DSP routing (cave reverb example).
- Audio preview when a DSP component is selected.

### Phase 6 — Bake and ship pipeline (week+)

- `ocarina-bake` CLI: walk a project dir, parse every source file, compile shaders with DXC, process meshes into GPU-ready buffers, serialize the ECS world with `bincode`, emit a single asset archive.
- `ocarina-runtime` loads the archive, deserializes the ECS world, and ticks the game loop. No text parsing, no filesystem watching.
- `ocarina build` command: bake + statically link runtime + archive into a single native binary per target platform.
- Verify: a shipped binary is small (target: under 20 MB for an empty project) and has no editor deps resolvable in its symbol table.

### Phase 7 — Second backend pilot (parked)

- Write signatures (not implementations) for a D3D12 or VK backend against the trait. Verify no wgpu-isms leaked. Implement only when a real feature need justifies it (mesh shaders, true bindless, RT).

## Out of scope / explicit non-goals

- Visual scripting, node-based shader graph, node-based animation graph.
- Proprietary scene format, scene database, asset database.
- Dual-state editor model (in-memory edits divergent from disk).
- Built-in undo stack beyond text-editor native undo.
- Browser / web builds as a primary target (wgpu can, but we're optimizing for native).
- Multi-GPU, multi-queue (v1).
- Render graph abstraction (v1 — revisit when the pass count justifies it).

## Open questions

- **Prefab inheritance.** v1 ships without inheritance — prefabs are multi-component files you copy and edit. If copy-paste-with-overrides becomes painful in practice, add an `[_inherit = "path/to/prefab.toml"]` directive with per-section override semantics.
- **Asset archive versioning.** If we change a component schema, old archives should either migrate or fail loudly. Design a schema-hash check for v1 and real migration tooling later.
- **Project manifest.** Likely an `ocarina.toml` at project root listing gameplay crates (for multi-crate reload), build settings, and bake targets. Shape TBD.
