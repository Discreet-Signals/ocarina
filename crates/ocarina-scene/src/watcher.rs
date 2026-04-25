//! Filesystem watcher.
//!
//! Thin wrapper around `notify::recommended_watcher`. Runs the watcher
//! in its own thread (notify spawns one internally) and forwards
//! coalesced events to a crossbeam-style channel the editor can drain
//! each frame.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};

/// One logical change to a tracked file or directory.
///
/// Notify emits low-level OS events (FsEvents on macOS, inotify on
/// Linux, ReadDirectoryChangesW on Windows) with platform-specific
/// quirks. We normalize to the three things the scene loader actually
/// cares about.
#[derive(Debug, Clone)]
pub enum SceneEvent {
    /// File was created or modified — re-parse it.
    Changed(PathBuf),
    /// File was deleted — strip any components it contributed.
    Removed(PathBuf),
    /// Subtree was renamed, moved, or had a bulk change. Triggers a
    /// full reload (cheaper than reasoning about what moved where).
    Rescan(PathBuf),
}

pub struct SceneWatcher {
    _watcher: RecommendedWatcher,
    rx: Receiver<SceneEvent>,
}

impl SceneWatcher {
    /// Start watching `root` recursively. The returned watcher stays
    /// active until it's dropped.
    pub fn start(root: &Path) -> notify::Result<Self> {
        let (tx, rx) = channel::<SceneEvent>();
        let handler_tx: Sender<SceneEvent> = tx;
        let mut watcher =
            notify::recommended_watcher(move |res: notify::Result<Event>| match res {
                Ok(event) => classify_and_send(&handler_tx, event),
                Err(e) => log::warn!("scene watcher error: {e}"),
            })?;
        watcher.watch(root, RecursiveMode::Recursive)?;
        log::info!("scene watcher: watching {}", root.display());
        Ok(Self {
            _watcher: watcher,
            rx,
        })
    }

    /// Drain pending events. Never blocks. Call each frame.
    pub fn drain(&self) -> Vec<SceneEvent> {
        let mut out = Vec::new();
        while let Ok(ev) = self.rx.try_recv() {
            out.push(ev);
        }
        out
    }
}

fn classify_and_send(tx: &Sender<SceneEvent>, event: Event) {
    let kind = event.kind;
    for path in event.paths {
        let ev = match kind {
            EventKind::Create(_) | EventKind::Modify(_) | EventKind::Access(_) => {
                // Access fires on metadata reads too; filter to changes
                // that would plausibly affect file content.
                if matches!(kind, EventKind::Access(_)) {
                    continue;
                }
                // Only care about TOML files (sidecar files like .swp
                // generate noise editors' save-and-rename patterns
                // produce).
                if is_toml(&path) {
                    SceneEvent::Changed(path)
                } else if path.is_dir() {
                    SceneEvent::Rescan(path)
                } else {
                    continue;
                }
            }
            EventKind::Remove(_) => {
                if is_toml(&path) {
                    SceneEvent::Removed(path)
                } else {
                    SceneEvent::Rescan(path)
                }
            }
            EventKind::Other | EventKind::Any => SceneEvent::Rescan(path),
        };
        let _ = tx.send(ev);
    }
}

fn is_toml(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("toml"))
}
