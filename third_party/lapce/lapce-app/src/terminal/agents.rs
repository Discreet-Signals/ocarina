use std::{collections::HashSet, path::PathBuf, sync::Arc};

use lapce_rpc::terminal::TerminalProfile;

use crate::{
    config::icon::LapceIcons, panel::kind::PanelKind, workspace::LapceWorkspace,
};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AgentKind {
    Claude,
    Codex,
    Opencode,
}

impl AgentKind {
    pub const ALL: &'static [AgentKind] =
        &[AgentKind::Claude, AgentKind::Codex, AgentKind::Opencode];

    pub fn panel_kind(self) -> PanelKind {
        match self {
            AgentKind::Claude => PanelKind::ClaudeAgent,
            AgentKind::Codex => PanelKind::CodexAgent,
            AgentKind::Opencode => PanelKind::OpencodeAgent,
        }
    }

    pub fn from_panel_kind(kind: PanelKind) -> Option<AgentKind> {
        match kind {
            PanelKind::ClaudeAgent => Some(AgentKind::Claude),
            PanelKind::CodexAgent => Some(AgentKind::Codex),
            PanelKind::OpencodeAgent => Some(AgentKind::Opencode),
            _ => None,
        }
    }

    pub fn binary_name(self) -> &'static str {
        match self {
            AgentKind::Claude => "claude",
            AgentKind::Codex => "codex",
            AgentKind::Opencode => "opencode",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            AgentKind::Claude => "Claude",
            AgentKind::Codex => "Codex",
            AgentKind::Opencode => "Opencode",
        }
    }

    pub fn icon_name(self) -> &'static str {
        match self {
            AgentKind::Claude => LapceIcons::AGENT_CLAUDE,
            AgentKind::Codex => LapceIcons::AGENT_OPENAI,
            AgentKind::Opencode => LapceIcons::AGENT_OPENCODE,
        }
    }

    pub fn new_tab_tooltip(self) -> &'static str {
        match self {
            AgentKind::Claude => "New Claude session",
            AgentKind::Codex => "New Codex session",
            AgentKind::Opencode => "New Opencode session",
        }
    }

    pub fn terminal_profile(
        self,
        workspace: &Arc<LapceWorkspace>,
    ) -> TerminalProfile {
        TerminalProfile {
            name: self.display_name().to_string(),
            command: Some(self.binary_name().to_string()),
            arguments: None,
            workdir: workspace
                .path
                .as_ref()
                .and_then(|p| url::Url::from_directory_path(p).ok()),
            environment: None,
        }
    }
}

/// Walk `PATH` and report which agent binaries are reachable. Cheap
/// (a few directory reads on startup); avoids a `which` crate dep.
pub fn detect_installed() -> HashSet<AgentKind> {
    let path_var = std::env::var_os("PATH").unwrap_or_default();
    let dirs: Vec<PathBuf> = std::env::split_paths(&path_var).collect();

    let mut found = HashSet::new();
    for agent in AgentKind::ALL {
        for dir in &dirs {
            let candidate = dir.join(agent.binary_name());
            if candidate.is_file() {
                found.insert(*agent);
                break;
            }
        }
    }
    found
}
