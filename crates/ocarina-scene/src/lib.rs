//! Filesystem-first scene loader.
//!
//! A *scene* is just a directory. Each subfolder is an entity; every
//! `.toml` file in a folder contributes components to that folder's
//! entity (one per `[section]` header). Child folders become child
//! entities, and the tree mirrors the filesystem.
//!
//! ```text
//! my_scene/
//!   _scene.toml          (scene-level settings, attached to the root entity)
//!   player/
//!     player.toml        [transform] [name]
//!     hitbox/
//!       hitbox.toml      [transform]
//!   cave/
//!     cave.toml          [transform] [mesh] [material]
//! ```
//!
//! [`Scene`] owns a `bevy_ecs::World` plus a map from filesystem paths
//! to entities. Reloading a single file edits only the components on that
//! path's entity; the rest of the world is untouched.
//!
//! [`SceneWatcher`] is a `notify`-based watcher that emits
//! [`SceneEvent`]s on a crossbeam-style channel. The editor drives the
//! frame loop: pump events, apply them to the `Scene`, repaint.

pub mod loader;
pub mod scene;
pub mod watcher;

pub use loader::LoadReport;
pub use scene::{Scene, SceneError};
pub use watcher::{SceneEvent, SceneWatcher};
