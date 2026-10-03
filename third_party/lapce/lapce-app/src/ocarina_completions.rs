//! Schema-aware completions for Ocarina entity `.toml` files.
//!
//! These run *in-process* inside the editor: when the completion
//! machinery requests items for a `.toml` file, we synthesise
//! component sections and field suggestions directly from the
//! `ocarina-schema` registry. No LSP wire round-trip, no plugin
//! infrastructure.
//!
//! The standalone `ocarina-lsp` crate (currently parked due to a
//! `tower-lsp` / `lsp-types` version conflict with Lapce) covers the
//! same ground for external editors. When that comes back online the
//! shared logic should move into a neutral crate they both depend on,
//! but for now duplicating the schema → completion mapping here is
//! the cheapest path to a working autocomplete in the Ocarina editor.

use lsp_types::{
    CompletionItem, CompletionItemKind, Documentation, InsertTextFormat,
    MarkupContent, MarkupKind, Position,
};
use ocarina_schema::{ComponentSchema, FieldSchema, FieldType};

/// Build the completion list for the given document position.
///
/// `text` is the full buffer contents; `pos` is the LSP position of
/// the cursor. We scan backwards from `pos.line` to find the nearest
/// `[section]` header, and pick suggestions accordingly:
///
/// - If the cursor is inside a known component section, suggest that
///   component's fields.
/// - Otherwise (no enclosing section, or unknown component), suggest
///   every registered component as a `[name]` insertion.
pub fn completions_for_toml(text: &str, pos: Position) -> Vec<CompletionItem> {
    let schemas: Vec<ComponentSchema> = ocarina_schema::collect_schemas();
    if schemas.is_empty() {
        return Vec::new();
    }

    let line_idx = pos.line as usize;
    let lines: Vec<&str> = text.split('\n').collect();
    let enclosing_section = find_enclosing_section(&lines, line_idx);

    match enclosing_section.and_then(|name| {
        schemas.iter().find(|s| s.name == name).cloned()
    }) {
        Some(schema) => field_completions(&schema),
        None => component_completions(&schemas),
    }
}

/// Walk lines `0..=line_idx` (inclusive) backwards looking for the
/// nearest `[section_name]` header. Returns the section name if any.
/// Bare-bones parser: skips lines starting with `#` (comments) and
/// recognises `[name]` patterns. Doesn't handle nested arrays-of-tables
/// like `[[items]]` (those aren't part of the entity-component scheme).
fn find_enclosing_section(lines: &[&str], line_idx: usize) -> Option<String> {
    let upper = line_idx.min(lines.len().saturating_sub(1));
    for line in lines.iter().take(upper + 1).rev() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix('[') {
            // Reject `[[...]]` (array-of-tables) and bracketed values.
            if rest.starts_with('[') {
                continue;
            }
            if let Some(name) = rest.strip_suffix(']') {
                let name = name.trim();
                if !name.is_empty()
                    && name
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
                {
                    return Some(name.to_string());
                }
            }
        }
    }
    None
}

fn component_completions(schemas: &[ComponentSchema]) -> Vec<CompletionItem> {
    schemas
        .iter()
        .map(|schema| CompletionItem {
            label: format!("[{}]", schema.name),
            kind: Some(CompletionItemKind::CLASS),
            detail: Some(schema.rust_type.clone()),
            documentation: doc_markup(&schema.doc),
            insert_text: Some(format!("[{}]\n", schema.name)),
            insert_text_format: Some(InsertTextFormat::PLAIN_TEXT),
            filter_text: Some(schema.name.clone()),
            ..Default::default()
        })
        .collect()
}

fn field_completions(schema: &ComponentSchema) -> Vec<CompletionItem> {
    schema
        .fields
        .iter()
        .map(|field| CompletionItem {
            label: field.name.clone(),
            kind: Some(CompletionItemKind::FIELD),
            detail: Some(field.rust_type.clone()),
            documentation: doc_markup(&field.doc),
            insert_text: Some(format!("{} = {}", field.name, field_placeholder(field))),
            insert_text_format: Some(InsertTextFormat::SNIPPET),
            filter_text: Some(field.name.clone()),
            ..Default::default()
        })
        .collect()
}

/// Snippet body for the right-hand side of `name = ...`. Uses
/// `${1:default}` so the user can tab through and edit. Falls back to
/// the registered `default_toml` when present, otherwise a sensible
/// per-type stub (zero vector for `Vec3`, empty string for `String`,
/// etc.).
fn field_placeholder(field: &FieldSchema) -> String {
    if let Some(default) = field.default_toml.as_ref() {
        return format!("${{1:{}}}", default);
    }
    let stub = match field.ty {
        FieldType::Bool => "false",
        FieldType::Int | FieldType::Uint => "0",
        FieldType::Float => "0.0",
        FieldType::String | FieldType::AssetPath => "\"\"",
        FieldType::Vec2 => "[0.0, 0.0]",
        FieldType::Vec3 => "[0.0, 0.0, 0.0]",
        FieldType::Vec4 | FieldType::Quat => "[0.0, 0.0, 0.0, 0.0]",
        FieldType::Color => "[1.0, 1.0, 1.0, 1.0]",
        FieldType::Other => "",
    };
    format!("${{1:{}}}", stub)
}

fn doc_markup(doc: &str) -> Option<Documentation> {
    let trimmed = doc.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(Documentation::MarkupContent(MarkupContent {
        kind: MarkupKind::Markdown,
        value: trimmed.to_string(),
    }))
}
