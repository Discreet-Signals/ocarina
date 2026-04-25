//! Schema lookup table for the LSP.
//!
//! Two construction paths:
//! - [`SchemaIndex::from_inventory`] — pulls every `#[derive(Component)]`
//!   schema linked into the current binary. Free, instantaneous, and the
//!   common case when the LSP is launched as `ocarina lsp` (the CLI links
//!   `ocarina-core` and any user gameplay crate).
//! - [`SchemaIndex::from_json`] — parses the JSON the `ocarina schema`
//!   subcommand emits. Used when the LSP is run by an external editor
//!   that didn't link in the user's components.

use std::collections::BTreeMap;

use ocarina_schema::ComponentSchema;

#[derive(Default)]
pub struct SchemaIndex {
    by_name: BTreeMap<String, ComponentSchema>,
}

impl SchemaIndex {
    /// Pull schemas from `inventory::iter`. Reflects whatever components
    /// the *current* process has linked.
    #[must_use]
    pub fn from_inventory() -> Self {
        let by_name = ocarina_schema::iter_schemas()
            .map(|s| (s.name.clone(), s))
            .collect();
        Self { by_name }
    }

    /// Parse the JSON produced by `ocarina schema`.
    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        let list: Vec<ComponentSchema> = serde_json::from_str(json)?;
        Ok(Self {
            by_name: list.into_iter().map(|s| (s.name.clone(), s)).collect(),
        })
    }

    pub fn get(&self, name: &str) -> Option<&ComponentSchema> {
        self.by_name.get(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = &ComponentSchema> {
        self.by_name.values()
    }

    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }
}
