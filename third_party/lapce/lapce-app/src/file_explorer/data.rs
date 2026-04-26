use std::{
    borrow::Cow,
    collections::HashMap,
    ffi::OsStr,
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
};

use floem::{
    action::show_context_menu,
    event::EventPropagation,
    ext_event::create_ext_action,
    keyboard::Modifiers,
    menu::{Menu, MenuItem},
    reactive::{ReadSignal, RwSignal, Scope, SignalGet, SignalUpdate, SignalWith},
    views::editor::text::SystemClipboard,
};
use globset::Glob;
use lapce_core::{
    command::{EditCommand, FocusCommand, MoveCommand},
    mode::Mode,
    register::Clipboard,
};
use lapce_rpc::{
    file::{
        Duplicating, FileNodeItem, FileNodeViewKind, Naming, NamingState, NewNode,
        Renaming,
    },
    proxy::ProxyResponse,
};

use crate::{
    command::{CommandExecuted, CommandKind, InternalCommand, LapceCommand},
    config::LapceConfig,
    editor::EditorData,
    keypress::{KeyPressFocus, condition::Condition},
    main_split::Editors,
    window_tab::CommonData,
};

enum RenamedPath {
    NotRenaming,
    NameUnchanged,
    Renamed {
        current_path: PathBuf,
        new_path: PathBuf,
    },
}

#[derive(Clone, Debug)]
pub struct FileExplorerData {
    pub root: RwSignal<FileNodeItem>,
    pub naming: RwSignal<Naming>,
    pub naming_editor_data: EditorData,
    pub common: Rc<CommonData>,
    pub scroll_to_line: RwSignal<Option<f64>>,
    left_diff_path: RwSignal<Option<PathBuf>>,
    pub select: RwSignal<Option<FileNodeViewKind>>,
}

impl KeyPressFocus for FileExplorerData {
    fn get_mode(&self) -> Mode {
        Mode::Insert
    }

    fn check_condition(&self, condition: Condition) -> bool {
        if self.naming.with_untracked(Naming::is_accepting_input) {
            return condition == Condition::ModalFocus;
        }
        // When not in a naming editor, the file explorer behaves as a
        // navigable list — opt into the standard `list_focus` keybindings
        // so up/down/enter map to ListPrevious/ListNext/ListSelect.
        condition == Condition::ListFocus
    }

    fn run_command(
        &self,
        command: &LapceCommand,
        count: Option<usize>,
        mods: Modifiers,
    ) -> CommandExecuted {
        if self.naming.with_untracked(Naming::is_accepting_input) {
            match command.kind {
                CommandKind::Focus(FocusCommand::ModalClose) => {
                    self.cancel_naming();
                    CommandExecuted::Yes
                }
                CommandKind::Edit(EditCommand::InsertNewLine) => {
                    self.finish_naming();
                    CommandExecuted::Yes
                }
                CommandKind::Edit(_) => {
                    let command_executed =
                        self.naming_editor_data.run_command(command, count, mods);

                    if let Some(new_path) = self.naming_path() {
                        self.common
                            .internal_command
                            .send(InternalCommand::TestPathCreation { new_path });
                    }

                    command_executed
                }
                _ => self.naming_editor_data.run_command(command, count, mods),
            }
        } else {
            match &command.kind {
                CommandKind::Focus(FocusCommand::ListPrevious) => {
                    self.move_selection(-1);
                    CommandExecuted::Yes
                }
                CommandKind::Focus(FocusCommand::ListNext) => {
                    self.move_selection(1);
                    CommandExecuted::Yes
                }
                CommandKind::Focus(FocusCommand::ListSelect) => {
                    self.activate_selection();
                    CommandExecuted::Yes
                }
                CommandKind::Focus(FocusCommand::ListExpand)
                | CommandKind::Move(MoveCommand::Right) => {
                    self.expand_or_step_in();
                    CommandExecuted::Yes
                }
                CommandKind::Move(MoveCommand::Left) => {
                    self.collapse_or_step_out();
                    CommandExecuted::Yes
                }
                _ => CommandExecuted::No,
            }
        }
    }

    fn receive_char(&self, c: &str) {
        if self.naming.with_untracked(Naming::is_accepting_input) {
            self.naming_editor_data.receive_char(c);

            if let Some(new_path) = self.naming_path() {
                self.common
                    .internal_command
                    .send(InternalCommand::TestPathCreation { new_path });
            }
        }
    }
}

impl FileExplorerData {
    pub fn new(cx: Scope, editors: Editors, common: Rc<CommonData>) -> Self {
        let path = common.workspace.path.clone().unwrap_or_default();
        let root = cx.create_rw_signal(FileNodeItem {
            path: path.clone(),
            is_dir: true,
            read: false,
            open: false,
            children: HashMap::new(),
            children_open_count: 0,
        });
        let naming = cx.create_rw_signal(Naming::None);
        let naming_editor_data = editors.make_local(cx, common.clone());
        let data = Self {
            root,
            naming,
            naming_editor_data,
            common,
            scroll_to_line: cx.create_rw_signal(None),
            left_diff_path: cx.create_rw_signal(None),
            select: cx.create_rw_signal(None),
        };
        if data.common.workspace.path.is_some() {
            // only fill in the child files if there is open folder
            data.toggle_expand(&path);
        }

        // Reverse Ocarina selection sync: when the viewport pushes a
        // selection (a clicked entity → its owning folder), mirror it
        // into the file explorer's highlight + scroll-to. The forward
        // direction (file click → viewport) is handled in `click()`.
        let selection_signal = data.common.ocarina_selection;
        let effect_data = data.clone();
        cx.create_effect(move |_| {
            let Some(selection) = selection_signal.get() else {
                effect_data.select.set(None);
                return;
            };
            let folder = selection.folder;
            let already_on_folder = effect_data.select.with_untracked(|s| {
                matches!(s, Some(FileNodeViewKind::Path(p)) if p == &folder)
            });
            if already_on_folder {
                return;
            }
            let visible = effect_data.visible_paths();
            let idx = visible.iter().position(|p| p == &folder);
            effect_data.select_path(folder, idx);
        });

        data
    }

    /// Reload the file explorer data via reading the root directory.  
    /// Note that this will not update immediately.
    pub fn reload(&self) {
        let path = self.root.with_untracked(|root| root.path.clone());
        self.read_dir(&path);
    }

    /// Toggle whether the directory is expanded or not.  
    /// Does nothing if the path does not exist or is not a directory.
    pub fn toggle_expand(&self, path: &Path) {
        let Some(Some(read)) = self.root.try_update(|root| {
            let read = if let Some(node) = root.get_file_node_mut(path) {
                if !node.is_dir {
                    return None;
                }
                node.open = !node.open;
                Some(node.read)
            } else {
                None
            };

            if Some(true) == read {
                root.update_node_count_recursive(path);
            }
            read
        }) else {
            return;
        };

        // Read the directory's files if they haven't been read yet
        if !read {
            self.read_dir(path);
        }
    }

    pub fn read_dir(&self, path: &Path) {
        self.read_dir_cb(path, |_| {});
    }

    /// Read the directory's information and update the file explorer tree.  
    /// `done : FnOnce(was_read: bool)` is called when the operation is completed, whether success,
    /// failure, or ignored.
    pub fn read_dir_cb(&self, path: &Path, done: impl FnOnce(bool) + 'static) {
        let root = self.root;
        let data = self.clone();
        let config = self.common.config;
        let send = {
            let path = path.to_path_buf();
            create_ext_action(self.common.scope, move |result| {
                let Ok(ProxyResponse::ReadDirResponse { mut items }) = result else {
                    done(false);
                    return;
                };

                root.update(|root| {
                    // Get the node for this path, which should already exist if we're calling
                    // read_dir on it.
                    if let Some(node) = root.get_file_node_mut(&path) {
                        // TODO: do not recreate glob every time we read a directory
                        // Retain only items that are not excluded from view by the configuration
                        match Glob::new(&config.get().editor.files_exclude) {
                            Ok(glob) => {
                                let matcher = glob.compile_matcher();
                                items.retain(|i| !matcher.is_match(&i.path));
                            }
                            Err(e) => tracing::error!(
                                target:"files_exclude",
                                "Failed to compile glob: {}",
                                e
                            ),
                        }

                        node.read = true;

                        // Remove paths that no longer exist
                        let removed_paths: Vec<PathBuf> = node
                            .children
                            .keys()
                            .filter(|p| !items.iter().any(|i| &&i.path == p))
                            .map(PathBuf::from)
                            .collect();
                        for path in removed_paths {
                            node.children.remove(&path);
                        }

                        // Reread dirs that were already read and add new paths
                        for item in items {
                            if let Some(existing) = node.children.get(&item.path) {
                                if existing.read {
                                    data.read_dir(&existing.path);
                                }
                            } else {
                                node.children.insert(item.path.clone(), item);
                            }
                        }
                    }
                    root.update_node_count_recursive(&path);
                });

                done(true);
            })
        };

        // Ask the proxy for the directory information
        self.common.proxy.read_dir(path.to_path_buf(), send);
    }

    /// Returns `true` if `path` exists in the file explorer tree and is a directory, `false`
    /// otherwise.
    fn is_dir(&self, path: &Path) -> bool {
        self.root.with_untracked(|root| {
            root.get_file_node(path).is_some_and(|node| node.is_dir)
        })
    }

    /// The current path that we're renaming to / creating or duplicating a node at.  
    /// Note: returns `None` when renaming if the file name has not changed.
    fn naming_path(&self) -> Option<PathBuf> {
        self.naming.with_untracked(|naming| match naming {
            Naming::None => None,
            Naming::Renaming(_) => {
                if let RenamedPath::Renamed { new_path, .. } = self.renamed_path() {
                    Some(new_path)
                } else {
                    None
                }
            }
            Naming::NewNode(n) => {
                let relative_path = self.naming_editor_data.text();
                let relative_path: Cow<OsStr> = match relative_path.slice_to_cow(..)
                {
                    Cow::Borrowed(path) => Cow::Borrowed(path.as_ref()),
                    Cow::Owned(path) => Cow::Owned(path.into()),
                };

                Some(n.base_path.join(relative_path))
            }
            Naming::Duplicating(d) => {
                let relative_path = self.naming_editor_data.text();
                let relative_path: Cow<OsStr> = match relative_path.slice_to_cow(..)
                {
                    Cow::Borrowed(path) => Cow::Borrowed(path.as_ref()),
                    Cow::Owned(path) => Cow::Owned(path.into()),
                };

                let new_path =
                    d.path.parent().unwrap_or("".as_ref()).join(relative_path);

                Some(new_path)
            }
        })
    }

    /// If there is an in progress rename and the user has entered a path that differs from the
    /// current path, gets the current and new paths of the renamed node.
    fn renamed_path(&self) -> RenamedPath {
        self.naming.with_untracked(|naming| {
            if let Some(current_path) = naming.as_renaming().map(|x| &x.path) {
                let current_file_name = current_path.file_name().unwrap_or_default();
                // `new_relative_path` is the new path relative to the parent directory, unless the
                // user has entered an absolute path.
                let new_relative_path = self.naming_editor_data.text();

                let new_relative_path: Cow<OsStr> =
                    match new_relative_path.slice_to_cow(..) {
                        Cow::Borrowed(path) => Cow::Borrowed(path.as_ref()),
                        Cow::Owned(path) => Cow::Owned(path.into()),
                    };

                if new_relative_path == current_file_name {
                    RenamedPath::NameUnchanged
                } else {
                    let new_path = current_path
                        .parent()
                        .unwrap_or("".as_ref())
                        .join(new_relative_path);

                    RenamedPath::Renamed {
                        current_path: current_path.to_owned(),
                        new_path,
                    }
                }
            } else {
                RenamedPath::NotRenaming
            }
        })
    }

    /// If a naming is in progress and the user has entered a valid path, send the request to
    /// actually perform the change.
    pub fn finish_naming(&self) {
        match self.naming.get_untracked() {
            Naming::None => {}
            Naming::Renaming(_) => {
                let renamed_path = self.renamed_path();
                match renamed_path {
                    // Should not occur
                    RenamedPath::NotRenaming => {}
                    RenamedPath::NameUnchanged => {
                        self.cancel_naming();
                    }
                    RenamedPath::Renamed {
                        current_path,
                        new_path,
                    } => {
                        self.common.internal_command.send(
                            InternalCommand::FinishRenamePath {
                                current_path,
                                new_path,
                            },
                        );
                    }
                }
            }
            Naming::NewNode(n) => {
                let Some(path) = self.naming_path() else {
                    return;
                };

                self.common
                    .internal_command
                    .send(InternalCommand::FinishNewNode {
                        is_dir: n.is_dir,
                        path,
                    });
            }
            Naming::Duplicating(d) => {
                let Some(path) = self.naming_path() else {
                    return;
                };

                self.common.internal_command.send(
                    InternalCommand::FinishDuplicate {
                        source: d.path.to_owned(),
                        path,
                    },
                );
            }
        }
    }

    /// Closes the naming text box without applying the effect.
    pub fn cancel_naming(&self) {
        self.naming.set(Naming::None);
    }

    /// Flat list of currently visible file/directory paths in display
    /// order, mirroring what the virtual_stack would render. Skips
    /// in-progress naming/duplicating rows since they aren't navigable.
    fn visible_paths(&self) -> Vec<PathBuf> {
        let naming = self.naming.get_untracked();
        let mut items = Vec::new();
        self.root.with_untracked(|root| {
            root.append_view_slice(&mut items, &naming, 0, usize::MAX, 0, 1);
        });
        items
            .into_iter()
            .filter_map(|v| match v.kind {
                FileNodeViewKind::Path(p) => Some(p),
                _ => None,
            })
            .collect()
    }

    fn selected_path(&self) -> Option<PathBuf> {
        self.select.with_untracked(|s| {
            s.as_ref().and_then(|kind| match kind {
                FileNodeViewKind::Path(p) => Some(p.clone()),
                FileNodeViewKind::Renaming { path, .. } => Some(path.clone()),
                _ => None,
            })
        })
    }

    fn select_path(&self, path: PathBuf, visible_index: Option<usize>) {
        self.select.set(Some(FileNodeViewKind::Path(path)));
        if let Some(idx) = visible_index {
            self.scroll_to_line.set(Some(idx as f64));
        }
    }

    pub fn move_selection(&self, delta: isize) {
        let visible = self.visible_paths();
        if visible.is_empty() {
            return;
        }
        let len = visible.len() as isize;
        let new_idx = match self
            .selected_path()
            .and_then(|p| visible.iter().position(|v| v == &p))
        {
            Some(i) => (i as isize + delta).clamp(0, len - 1),
            None if delta >= 0 => 0,
            None => len - 1,
        } as usize;
        let path = visible[new_idx].clone();
        self.select_path(path, Some(new_idx));
    }

    /// Right-arrow / `o`: expand a closed directory, or step into the
    /// first child of an already-open one. No-op for files.
    pub fn expand_or_step_in(&self) {
        let Some(path) = self.selected_path() else {
            return;
        };
        let (is_dir, is_open) = self.root.with_untracked(|root| {
            root.get_file_node(&path)
                .map(|n| (n.is_dir, n.open))
                .unwrap_or((false, false))
        });
        if is_dir && !is_open {
            self.toggle_expand(&path);
        } else if is_dir && is_open {
            self.move_selection(1);
        }
    }

    /// Left-arrow: collapse the current directory if open, otherwise
    /// move selection up to the parent directory (without escaping the
    /// workspace root).
    pub fn collapse_or_step_out(&self) {
        let Some(path) = self.selected_path() else {
            return;
        };
        let (is_dir, is_open) = self.root.with_untracked(|root| {
            root.get_file_node(&path)
                .map(|n| (n.is_dir, n.open))
                .unwrap_or((false, false))
        });
        if is_dir && is_open {
            self.toggle_expand(&path);
            return;
        }
        let Some(parent) = path.parent().map(Path::to_path_buf) else {
            return;
        };
        let parent_in_tree = self
            .root
            .with_untracked(|root| root.get_file_node(&parent).is_some());
        if !parent_in_tree {
            return;
        }
        let visible = self.visible_paths();
        let idx = visible.iter().position(|p| p == &parent);
        self.select_path(parent, idx);
    }

    /// Enter / `list.select`: toggle a directory or open a file, just
    /// like a single click.
    pub fn activate_selection(&self) {
        let Some(path) = self.selected_path() else {
            return;
        };
        let config = self.common.config;
        self.click(&path, config);
    }

    pub fn click(&self, path: &Path, config: ReadSignal<Arc<LapceConfig>>) {
        // Mirror the click into the Ocarina selection signal: when
        // someone clicks a folder (or a file inside a folder) in the
        // explorer, the viewport should highlight the corresponding
        // entity. The reverse direction is handled in the viewport
        // panel — together they make file explorer ↔ viewport one
        // shared selection.
        self.sync_ocarina_selection(path);

        if self.is_dir(path) {
            self.toggle_expand(path);
        } else if !config.get_untracked().core.file_explorer_double_click {
            self.common
                .internal_command
                .send(InternalCommand::OpenFile {
                    path: path.to_path_buf(),
                })
        }
    }

    /// Resolve `path` to an Ocarina entity (via the scene's
    /// `dir_entity` map) and write it to the selection signal. A click
    /// on a file maps to the entity of its containing folder, since
    /// that's where Ocarina aggregates components. No-op if no scene
    /// is loaded or the path doesn't correspond to a known entity.
    fn sync_ocarina_selection(&self, path: &Path) {
        let Some(scene_rc) = self.common.ocarina_scene.get_untracked() else {
            return;
        };
        // Scene paths are canonicalized at load time; do the same on
        // the explorer side or the lookup will miss.
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let dir = if canonical.is_dir() {
            canonical
        } else {
            canonical
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or(canonical)
        };
        let entity = scene_rc.borrow().dir_entity.get(&dir).copied();
        if let Some(entity) = entity {
            self.common
                .ocarina_selection
                .set(Some(ocarina_viewport::ViewportSelection {
                    entity,
                    folder: dir,
                }));
        }
    }

    pub fn reveal_in_file_tree(&self, path: PathBuf) {
        let done = self
            .root
            .try_update(|root| {
                // the directories in which the file are located are all read and opened
                if root.get_file_node(&path).is_some() {
                    for current_path in path.ancestors() {
                        if let Some(file) = root.get_file_node_mut(current_path) {
                            if file.is_dir {
                                file.open = true;
                            }
                        }
                    }
                    root.update_node_count_recursive(&path);
                    true
                } else {
                    // read and open the directories in which the file are located
                    let mut read_dir = None;
                    // Whether the file is in the workspace
                    let mut exist = false;
                    for current_path in path.ancestors() {
                        if let Some(file) = root.get_file_node_mut(current_path) {
                            exist = true;
                            if file.is_dir {
                                file.open = true;
                                if !file.read {
                                    read_dir = Some(current_path.to_path_buf())
                                }
                            }
                            break;
                        }
                    }
                    if let (true, Some(dir)) = (exist, read_dir) {
                        let explorer = self.clone();
                        let select_path = path.clone();
                        self.read_dir_cb(&dir, move |_| {
                            explorer.reveal_in_file_tree(select_path);
                        })
                    }
                    false
                }
            })
            .unwrap_or(false);
        if done {
            let (found, line) =
                self.root.with_untracked(|x| x.find_file_at_line(&path));
            if found {
                self.scroll_to_line.set(Some(line));
                self.select.set(Some(FileNodeViewKind::Path(path)));
            }
        }
    }

    pub fn double_click(
        &self,
        path: &Path,
        config: ReadSignal<Arc<LapceConfig>>,
    ) -> EventPropagation {
        if self.is_dir(path) {
            EventPropagation::Continue
        } else if config.get_untracked().core.file_explorer_double_click {
            self.common.internal_command.send(
                InternalCommand::OpenAndConfirmedFile {
                    path: path.to_path_buf(),
                },
            );
            EventPropagation::Stop
        } else {
            self.common
                .internal_command
                .send(InternalCommand::MakeConfirmed);
            EventPropagation::Stop
        }
    }

    pub fn secondary_click(&self, path: &Path) {
        let common = self.common.clone();
        let path_a = path.to_owned();
        let left_diff_path = self.left_diff_path;
        // TODO: should we just pass is_dir into secondary click?
        let is_dir = self.is_dir(path);

        let Some(workspace_path) = self.common.workspace.path.as_ref() else {
            // There is no context menu if we are not in a workspace
            return;
        };

        let is_workspace = path == workspace_path;

        let base_path_a = if is_dir {
            Some(path_a.clone())
        } else {
            path_a.parent().map(ToOwned::to_owned)
        };
        let base_path_a = base_path_a.as_ref().unwrap_or(workspace_path);

        let mut menu = Menu::new("");

        let base_path = base_path_a.clone();
        let data = self.clone();
        let naming = self.naming;
        menu = menu.entry(MenuItem::new("New File").action(move || {
            let base_path_b = &base_path;
            let base_path = base_path.clone();
            data.read_dir_cb(base_path_b, move |was_read| {
                if !was_read {
                    tracing::warn!(
                        "Failed to read directory, avoiding creating node in: {:?}",
                        base_path
                    );
                    return;
                }

                naming.set(Naming::NewNode(NewNode {
                    state: NamingState::Naming,
                    base_path: base_path.clone(),
                    is_dir: false,
                    editor_needs_reset: true,
                }));
            });
        }));

        let base_path = base_path_a.clone();
        let data = self.clone();
        let naming = self.naming;
        menu = menu.entry(MenuItem::new("New Directory").action(move || {
            let base_path_b = &base_path;
            let base_path = base_path.clone();
            data.read_dir_cb(base_path_b, move |was_read| {
                if !was_read {
                    tracing::warn!(
                        "Failed to read directory, avoiding creating node in: {:?}",
                        base_path
                    );
                    return;
                }

                naming.set(Naming::NewNode(NewNode {
                    state: NamingState::Naming,
                    base_path: base_path.clone(),
                    is_dir: true,
                    editor_needs_reset: true,
                }));
            })
        }));

        menu = menu.separator();

        // TODO: there are situations where we can open the file explorer to remote files
        if !common.workspace.kind.is_remote() {
            let path = path_a.clone();
            #[cfg(not(target_os = "macos"))]
            let title = "Reveal in system file explorer";
            #[cfg(target_os = "macos")]
            let title = "Reveal in Finder";
            menu = menu.entry(MenuItem::new(title).action(move || {
                let path = path.parent().unwrap_or(&path);
                if !path.exists() {
                    return;
                }

                if let Err(err) = open::that(path) {
                    tracing::error!(
                        "Failed to reveal file in system file explorer: {}",
                        err
                    );
                }
            }));
        }

        if !is_workspace {
            let path = path_a.clone();
            menu = menu.entry(MenuItem::new("Rename").action(move || {
                naming.set(Naming::Renaming(Renaming {
                    state: NamingState::Naming,
                    path: path.clone(),
                    editor_needs_reset: true,
                }));
            }));

            let path = path_a.clone();
            menu = menu.entry(MenuItem::new("Duplicate").action(move || {
                naming.set(Naming::Duplicating(Duplicating {
                    state: NamingState::Naming,
                    path: path.clone(),
                    editor_needs_reset: true,
                }));
            }));

            // TODO: it is common for shift+right click to make 'Move file to trash' an actual
            // Delete, which can be useful for large files.
            let path = path_a.clone();
            let proxy = common.proxy.clone();
            let trash_text = if is_dir {
                "Move Directory to Trash"
            } else {
                "Move File to Trash"
            };
            menu = menu.entry(MenuItem::new(trash_text).action(move || {
                proxy.trash_path(path.clone(), |res| {
                    if let Err(err) = res {
                        tracing::warn!("Failed to trash path: {:?}", err);
                    }
                })
            }));
        }

        menu = menu.separator();

        let path = path_a.clone();
        menu = menu.entry(MenuItem::new("Copy Path").action(move || {
            let mut clipboard = SystemClipboard::new();
            clipboard.put_string(path.to_string_lossy());
        }));

        let path = path_a.clone();
        let workspace = common.workspace.clone();
        menu = menu.entry(MenuItem::new("Copy Relative Path").action(move || {
            let relative_path = if let Some(workspace_path) = &workspace.path {
                path.strip_prefix(workspace_path).unwrap_or(&path)
            } else {
                path.as_ref()
            };

            let mut clipboard = SystemClipboard::new();
            clipboard.put_string(relative_path.to_string_lossy());
        }));

        menu = menu.separator();

        let path = path_a.clone();
        menu = menu.entry(
            MenuItem::new("Select for Compare")
                .action(move || left_diff_path.set(Some(path.clone()))),
        );

        if let Some(left_path) = self.left_diff_path.get_untracked() {
            let common = self.common.clone();
            let right_path = path_a.to_owned();

            menu = menu.entry(MenuItem::new("Compare with Selected").action(
                move || {
                    common
                        .internal_command
                        .send(InternalCommand::OpenDiffFiles {
                            left_path: left_path.clone(),
                            right_path: right_path.clone(),
                        })
                },
            ))
        }

        menu = menu.separator();

        let internal_command = common.internal_command;
        menu = menu.entry(MenuItem::new("Refresh").action(move || {
            internal_command.send(InternalCommand::ReloadFileExplorer);
        }));

        show_context_menu(menu, None);
    }

    pub fn middle_click(&self, path: &Path) -> EventPropagation {
        if self.is_dir(path) {
            EventPropagation::Continue
        } else {
            self.common
                .internal_command
                .send(InternalCommand::OpenFileInNewTab {
                    path: path.to_path_buf(),
                });
            EventPropagation::Stop
        }
    }
}
