//! [`Scene`] — holds the ECS world + bookkeeping for which files own
//! which entities.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use ocarina_core::components::Transform;
use ocarina_core::math::Mat4;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SceneError {
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("TOML parse error in {path}: {message}")]
    Parse { path: PathBuf, message: String },
    #[error("unknown component `{section}` in {path}")]
    UnknownComponent { path: PathBuf, section: String },
    #[error("component `{section}` failed to load from {path}: {message}")]
    ComponentLoad {
        path: PathBuf,
        section: String,
        message: String,
    },
}

/// The scene. `root` is the entity for the project root directory;
/// every other directory-entity is a descendant.
pub struct Scene {
    /// The `bevy_ecs` world. Public so callers can query / iterate it.
    pub world: World,
    /// Filesystem root this scene was loaded from.
    pub root_dir: PathBuf,
    /// Entity corresponding to `root_dir`.
    pub root: Entity,
    /// directory path → entity.
    pub dir_entity: HashMap<PathBuf, Entity>,
    /// toml file path → entity it contributes components to (same as
    /// the entity of its containing directory). Used on file-change
    /// events to find the affected entity in O(1).
    pub file_entity: HashMap<PathBuf, Entity>,
}

impl Scene {
    /// Create an empty scene rooted at `root_dir`. Components aren't
    /// loaded until you call [`Scene::reload_full`].
    ///
    /// The path is canonicalized so every internal key (entries in
    /// `dir_entity` / `file_entity`) is absolute. This is required for
    /// consistency with `notify` watcher events, which always come back
    /// as absolute canonical paths — without this normalization, a
    /// project opened by a relative path would hot-reload into a
    /// parallel set of entities (one per path form) and you'd see
    /// duplicate meshes in the viewport.
    pub fn empty(root_dir: impl Into<PathBuf>) -> Self {
        let raw = root_dir.into();
        let root_dir = raw.canonicalize().unwrap_or(raw);
        let mut world = World::new();
        let root = world.spawn_empty().id();
        let mut dir_entity = HashMap::new();
        dir_entity.insert(root_dir.clone(), root);
        Self {
            world,
            root_dir,
            root,
            dir_entity,
            file_entity: HashMap::new(),
        }
    }

    /// How many entities this scene currently owns. Counted by summing
    /// archetype lengths.
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.world
            .archetypes()
            .iter()
            .map(|a| a.len() as usize)
            .sum()
    }

    /// How many (entity, component) pairs. Costs an archetype scan, so
    /// don't call per frame on a world with thousands of archetypes.
    #[must_use]
    pub fn component_instance_count(&self) -> usize {
        let mut total = 0usize;
        for archetype in self.world.archetypes().iter() {
            total += archetype.component_count() * archetype.len() as usize;
        }
        total
    }

    /// Walk the directory hierarchy and compose every `Transform` into
    /// a world-space matrix. Folders without a `Transform` of their own
    /// pass through as identity (purely organizational), so a transform
    /// on a deeply-nested folder still composes correctly through them.
    ///
    /// The result contains an entry for every entity that has a local
    /// `Transform` component. Entities without one are not included
    /// (they have no inherent placement to compose).
    ///
    /// Cost is O(entities + dir-depth-per-entity); fine to call once
    /// per frame at editor scales.
    #[must_use]
    pub fn world_transforms(&mut self) -> HashMap<Entity, Mat4> {
        let parents = self.entity_parent_map();
        let locals = self.collect_local_matrices();

        let mut cache: HashMap<Entity, Mat4> = HashMap::with_capacity(locals.len());
        let mut out: HashMap<Entity, Mat4> = HashMap::with_capacity(locals.len());
        for &entity in locals.keys() {
            let m = compose_world(entity, &locals, &parents, &mut cache);
            out.insert(entity, m);
        }
        out
    }

    /// World-space transform of `entity`'s *parent*. Identity if the
    /// entity has no parent in the directory hierarchy or its parents
    /// have no `Transform` components. Used by gizmo write-back to
    /// convert a world-space drag delta into a local-space change
    /// before persisting to disk.
    #[must_use]
    pub fn parent_world(&mut self, entity: Entity) -> Mat4 {
        let parents = self.entity_parent_map();
        let Some(&parent_entity) = parents.get(&entity) else {
            return Mat4::IDENTITY;
        };
        let locals = self.collect_local_matrices();
        let mut cache: HashMap<Entity, Mat4> = HashMap::new();
        compose_world(parent_entity, &locals, &parents, &mut cache)
    }

    /// Build `entity → parent_entity` from the directory layout. A
    /// folder's entity is the parent of every immediately-nested
    /// folder's entity. The implicit root entity has no parent.
    fn entity_parent_map(&self) -> HashMap<Entity, Entity> {
        let path_to_entity: HashMap<&Path, Entity> = self
            .dir_entity
            .iter()
            .map(|(p, e)| (p.as_path(), *e))
            .collect();
        let mut parents = HashMap::with_capacity(self.dir_entity.len());
        for (path, &entity) in &self.dir_entity {
            if let Some(parent_path) = path.parent() {
                if let Some(&parent_entity) = path_to_entity.get(parent_path) {
                    if parent_entity != entity {
                        parents.insert(entity, parent_entity);
                    }
                }
            }
        }
        parents
    }

    fn collect_local_matrices(&mut self) -> HashMap<Entity, Mat4> {
        let mut out = HashMap::new();
        let world = &mut self.world;
        let mut q = world.query::<(Entity, &Transform)>();
        for (e, t) in q.iter(world) {
            out.insert(e, t.matrix());
        }
        out
    }

    /// Find or create an entity for `dir`. Creates one lazily, walking
    /// up to the root if parent entities haven't been created yet.
    pub(crate) fn entity_for_dir(&mut self, dir: &Path) -> Entity {
        if let Some(&e) = self.dir_entity.get(dir) {
            return e;
        }
        // Resolve parent path before the recursive mutable call to avoid
        // holding an immutable borrow of `self.root_dir` across it.
        let parent_dir: PathBuf = dir
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| self.root_dir.clone());
        let _parent_entity = if parent_dir == dir {
            self.root
        } else {
            self.entity_for_dir(&parent_dir)
        };
        let entity = self.world.spawn_empty().id();
        self.dir_entity.insert(dir.to_path_buf(), entity);
        entity
    }

    /// Remove everything under `dir` (recursively). Used on folder-delete
    /// events and during full reload.
    pub(crate) fn drop_subtree(&mut self, dir: &Path) {
        let to_drop: Vec<PathBuf> = self
            .dir_entity
            .keys()
            .filter(|k| k.starts_with(dir))
            .cloned()
            .collect();
        for d in to_drop {
            if let Some(e) = self.dir_entity.remove(&d) {
                if e != self.root {
                    self.world.despawn(e);
                }
            }
        }
        let to_drop_files: Vec<PathBuf> = self
            .file_entity
            .keys()
            .filter(|k| k.starts_with(dir))
            .cloned()
            .collect();
        for f in to_drop_files {
            self.file_entity.remove(&f);
        }
    }
}

/// Recursive matrix accumulation with memoisation. Entities without a
/// local `Transform` are passed through as identity, so a folder with
/// only an organizational role doesn't break composition for its
/// descendants.
fn compose_world(
    entity: Entity,
    locals: &HashMap<Entity, Mat4>,
    parents: &HashMap<Entity, Entity>,
    cache: &mut HashMap<Entity, Mat4>,
) -> Mat4 {
    if let Some(&m) = cache.get(&entity) {
        return m;
    }
    let local = locals.get(&entity).copied().unwrap_or(Mat4::IDENTITY);
    let world = match parents.get(&entity) {
        Some(&parent) => compose_world(parent, locals, parents, cache) * local,
        None => local,
    };
    cache.insert(entity, world);
    world
}

#[cfg(test)]
mod tests {
    use super::*;
    use ocarina_core::math::Vec3;

    fn make_scene_with_two_levels() -> (Scene, Entity, Entity) {
        // Layout:
        //   /root
        //   /root/parent      [transform] position=[10,0,0]
        //   /root/parent/child  [transform] position=[1,0,0]
        let root = std::path::PathBuf::from("/root");
        let mut scene = Scene::empty(&root);
        let parent_entity = scene.entity_for_dir(&root.join("parent"));
        let child_entity = scene.entity_for_dir(&root.join("parent").join("child"));

        scene.world.entity_mut(parent_entity).insert(Transform {
            position: Vec3::new(10.0, 0.0, 0.0),
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
        });
        scene.world.entity_mut(child_entity).insert(Transform {
            position: Vec3::new(1.0, 0.0, 0.0),
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
        });

        (scene, parent_entity, child_entity)
    }

    #[test]
    fn world_transform_composes_translation() {
        let (mut scene, parent_entity, child_entity) = make_scene_with_two_levels();
        let xforms = scene.world_transforms();

        let parent_world = xforms[&parent_entity];
        let child_world = xforms[&child_entity];

        let parent_pos = parent_world.col(3).truncate();
        let child_pos = child_world.col(3).truncate();
        assert!((parent_pos - Vec3::new(10.0, 0.0, 0.0)).length() < 1e-4);
        // Child's local +1X composes with parent's +10X to land at +11X.
        assert!((child_pos - Vec3::new(11.0, 0.0, 0.0)).length() < 1e-4);
    }

    #[test]
    fn parent_world_returns_identity_for_root_level_entity() {
        let (mut scene, parent_entity, _) = make_scene_with_two_levels();
        let parent_of_parent = scene.parent_world(parent_entity);
        // parent of "/root/parent" is the root entity, which has no
        // Transform — so its world matrix is identity.
        assert_eq!(parent_of_parent, Mat4::IDENTITY);
    }

    #[test]
    fn parent_world_composes_through_organizational_folders() {
        // Skip the `parent` Transform: confirm the child still falls
        // back to identity-passthrough on the missing-Transform parent.
        let root = std::path::PathBuf::from("/root");
        let mut scene = Scene::empty(&root);
        let _empty = scene.entity_for_dir(&root.join("empty"));
        let leaf = scene.entity_for_dir(&root.join("empty").join("leaf"));
        scene.world.entity_mut(leaf).insert(Transform {
            position: Vec3::new(2.0, 3.0, 4.0),
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
        });

        let leaf_world = scene.world_transforms()[&leaf];
        assert!(
            (leaf_world.col(3).truncate() - Vec3::new(2.0, 3.0, 4.0)).length() < 1e-4
        );
    }
}
