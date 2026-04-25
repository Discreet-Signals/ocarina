//! Position → context analysis + diagnostic / completion / hover builders.
//!
//! The LSP works in `Position { line, character }` while the toml crate
//! gives byte offsets. We convert in both directions. UTF-16 column
//! semantics aren't strictly applied — fine for ASCII TOML; non-ASCII
//! identifiers in component / field names are vanishingly rare.

use tower_lsp::lsp_types::{
    CompletionItem, CompletionItemKind, Diagnostic, DiagnosticSeverity, Documentation, Hover,
    HoverContents, Location, MarkupContent, MarkupKind, Position, Range, Url,
};

use crate::document::Document;
use crate::schema::SchemaIndex;

// ────────────────────────── Diagnostics ──────────────────────────

pub fn diagnostics(schemas: &SchemaIndex, doc: &Document) -> Vec<Diagnostic> {
    let mut out = Vec::new();

    // Parse errors: take precedence and short-circuit semantic checks
    // because the parsed table is unavailable.
    if let Some(err) = &doc.parse_error {
        let range = err
            .span()
            .map(|span| byte_range_to_range(&doc.text, span))
            .unwrap_or_else(|| {
                Range::new(Position::new(0, 0), Position::new(0, 1))
            });
        out.push(Diagnostic {
            range,
            severity: Some(DiagnosticSeverity::ERROR),
            source: Some("ocarina".into()),
            message: format!("TOML parse error: {}", err.message()),
            ..Default::default()
        });
        return out;
    }

    let Some(toml::Value::Table(table)) = &doc.parsed else {
        return out;
    };

    // Walk sections. We rely on string search to find the section header
    // location in the source, since toml's parser doesn't give us spans
    // on Tables.
    for (section_name, value) in table {
        if section_name.starts_with('_') {
            continue;
        }

        let header_range = find_section_header_range(&doc.text, section_name);

        let Some(schema) = schemas.get(section_name) else {
            // Unknown component
            if let Some(range) = header_range {
                out.push(Diagnostic {
                    range,
                    severity: Some(DiagnosticSeverity::WARNING),
                    source: Some("ocarina".into()),
                    message: format!(
                        "unknown component `{section_name}` (no #[derive(Component)] type registered)"
                    ),
                    ..Default::default()
                });
            }
            continue;
        };

        // Field-name validation. Only fires when the section value is a
        // table — TOML allows `key = value` directly on a section but
        // we expect components to be `[section]`-headed tables.
        let toml::Value::Table(fields) = value else {
            continue;
        };
        for (field_name, field_value) in fields {
            let Some(field_schema) = schema.fields.iter().find(|f| f.name == *field_name) else {
                if let Some(range) =
                    find_field_range(&doc.text, section_name, field_name)
                {
                    let known: Vec<&str> =
                        schema.fields.iter().map(|f| f.name.as_str()).collect();
                    out.push(Diagnostic {
                        range,
                        severity: Some(DiagnosticSeverity::WARNING),
                        source: Some("ocarina".into()),
                        message: format!(
                            "unknown field `{field_name}` on `{section_name}`. Known: {}",
                            known.join(", ")
                        ),
                        ..Default::default()
                    });
                }
                continue;
            };

            if let Some(msg) = check_field_type(field_schema.ty, field_value) {
                if let Some(range) =
                    find_field_range(&doc.text, section_name, field_name)
                {
                    out.push(Diagnostic {
                        range,
                        severity: Some(DiagnosticSeverity::WARNING),
                        source: Some("ocarina".into()),
                        message: format!(
                            "field `{section_name}.{field_name}` expects {}; {}",
                            describe_field_type(field_schema.ty),
                            msg
                        ),
                        ..Default::default()
                    });
                }
            }
        }
    }

    out
}

// ────────────────────────── Completion ──────────────────────────

pub fn completions(
    schemas: &SchemaIndex,
    doc: &Document,
    pos: Position,
) -> Vec<CompletionItem> {
    // Inspect the line up to the cursor.
    let line = doc.line(pos.line as usize).unwrap_or("");
    let col = (pos.character as usize).min(line.len());
    let prefix = &line[..col];
    let trimmed = prefix.trim_start();

    // 1. Section-name completion: `[<cursor>` on this line.
    if trimmed.starts_with('[') && !trimmed.contains(']') {
        let typed = trimmed.trim_start_matches('[');
        return schemas
            .iter()
            .filter(|s| s.name.starts_with(typed))
            .map(|s| CompletionItem {
                label: s.name.clone(),
                kind: Some(CompletionItemKind::CLASS),
                detail: Some(s.rust_type.clone()),
                documentation: Some(Documentation::MarkupContent(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: s.doc.clone(),
                })),
                insert_text: Some(format!("{}]", s.name)),
                ..Default::default()
            })
            .collect();
    }

    // 2. Field-name completion: cursor is inside a section's body, on a
    //    line where a field name would go. Find the current section by
    //    walking backwards through the lines.
    let Some(section) = current_section(&doc.text, pos.line as usize) else {
        return Vec::new();
    };
    let Some(schema) = schemas.get(&section) else {
        return Vec::new();
    };

    // If the line already has an `=`, cursor's past it, so we'd be
    // completing a value, not a field name. We don't do value completions
    // in v1.
    if prefix.contains('=') {
        return Vec::new();
    }

    let typed = prefix.trim_start();
    schema
        .fields
        .iter()
        .filter(|f| f.name.starts_with(typed))
        .map(|f| {
            let detail = format!("{} ({:?})", f.rust_type, f.ty);
            let body = if f.doc.is_empty() {
                format!("`{}`", f.rust_type)
            } else {
                format!("`{}`\n\n{}", f.rust_type, f.doc)
            };
            let insert = match &f.default_toml {
                Some(d) => format!("{} = {}", f.name, d),
                None => format!("{} = ", f.name),
            };
            CompletionItem {
                label: f.name.clone(),
                kind: Some(CompletionItemKind::FIELD),
                detail: Some(detail),
                documentation: Some(Documentation::MarkupContent(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: body,
                })),
                insert_text: Some(insert),
                ..Default::default()
            }
        })
        .collect()
}

// ────────────────────────── Goto definition ──────────────────────────

/// Resolve the cursor to a source location in the Rust file declaring the
/// component. Triggers on the section header (`[name]`) or any field name
/// within that section. Both jump to the `#[derive(Component)]` line —
/// per-field source positions aren't tracked yet.
pub fn goto_definition(
    schemas: &SchemaIndex,
    doc: &Document,
    pos: Position,
) -> Option<Location> {
    let line = doc.line(pos.line as usize)?;
    let col = pos.character as usize;

    // Section header: `[<name>]`
    if let Some(range) = section_name_range(line) {
        if col >= range.0 && col <= range.1 {
            let name = &line[range.0..range.1];
            return schemas.get(name).and_then(schema_to_location);
        }
    }

    // Field on a `name = value` line — resolve via the enclosing section.
    if let Some(eq) = line.find('=') {
        if col <= eq {
            let leading = line.len() - line.trim_start().len();
            let field_name = line[leading..eq].trim();
            if !field_name.is_empty() {
                let section = current_section(&doc.text, pos.line as usize)?;
                let schema = schemas.get(&section)?;
                if schema.fields.iter().any(|f| f.name == field_name) {
                    return schema_to_location(schema);
                }
            }
        }
    }

    None
}

fn schema_to_location(schema: &ocarina_schema::ComponentSchema) -> Option<Location> {
    if schema.source_file.is_empty() {
        return None;
    }
    let uri = Url::from_file_path(&schema.source_file).ok()?;
    // LSP positions are 0-based; source_line is 1-based.
    let line = schema.source_line.saturating_sub(1);
    let range = Range::new(Position::new(line, 0), Position::new(line, 0));
    Some(Location { uri, range })
}

// ────────────────────────── Hover ──────────────────────────

pub fn hover(schemas: &SchemaIndex, doc: &Document, pos: Position) -> Option<Hover> {
    let line = doc.line(pos.line as usize)?;
    let col = pos.character as usize;

    // Section header? `[<name>]`
    if let Some(range) = section_name_range(line) {
        if col >= range.0 && col <= range.1 {
            let name = &line[range.0..range.1];
            if let Some(schema) = schemas.get(name) {
                return Some(component_hover(schema));
            }
        }
    }

    // Field name? `name = value` — left of the `=`.
    if let Some(eq) = line.find('=') {
        if col <= eq {
            let leading = line.len() - line.trim_start().len();
            let field_name = line[leading..eq].trim();
            if !field_name.is_empty() {
                if let Some(section) = current_section(&doc.text, pos.line as usize) {
                    if let Some(schema) = schemas.get(&section) {
                        if let Some(field) =
                            schema.fields.iter().find(|f| f.name == field_name)
                        {
                            return Some(field_hover(&section, field));
                        }
                    }
                }
            }
        }
    }

    None
}

fn component_hover(schema: &ocarina_schema::ComponentSchema) -> Hover {
    let mut value = format!(
        "**{}** &nbsp;·&nbsp; `{}`\n\n{}",
        schema.name, schema.rust_type, schema.doc
    );
    if !schema.fields.is_empty() {
        value.push_str("\n\n**Fields**\n");
        for f in &schema.fields {
            let default = f
                .default_toml
                .as_deref()
                .map(|d| format!(" *(default: `{d}`)*"))
                .unwrap_or_default();
            value.push_str(&format!("- `{}`: `{}`{}\n", f.name, f.rust_type, default));
        }
    }
    Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value,
        }),
        range: None,
    }
}

fn field_hover(section: &str, f: &ocarina_schema::FieldSchema) -> Hover {
    let mut value = format!(
        "**{}.{}** &nbsp;·&nbsp; `{}`",
        section, f.name, f.rust_type
    );
    if !f.doc.is_empty() {
        value.push_str("\n\n");
        value.push_str(&f.doc);
    }
    if let Some(d) = &f.default_toml {
        value.push_str(&format!("\n\n*default*: `{d}`"));
    }
    Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value,
        }),
        range: None,
    }
}

// ────────────────────────── Helpers ──────────────────────────

/// Walk backwards from `line_idx` to find the most recent `[section]`
/// header. Returns the section name (without brackets).
fn current_section(text: &str, line_idx: usize) -> Option<String> {
    let lines: Vec<&str> = text.lines().collect();
    for line in lines.iter().take(line_idx + 1).rev() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix('[') {
            // Skip array-of-tables `[[...]]` — same idea, easier to
            // strip the second bracket if present.
            let inner = rest.trim_start_matches('[');
            if let Some(end) = inner.find(']') {
                return Some(inner[..end].trim().to_string());
            }
        }
    }
    None
}

fn section_name_range(line: &str) -> Option<(usize, usize)> {
    let trimmed_start = line.len() - line.trim_start().len();
    let after = line.get(trimmed_start..)?;
    if !after.starts_with('[') {
        return None;
    }
    let inner_start = trimmed_start + 1;
    let close = line[inner_start..].find(']')?;
    Some((inner_start, inner_start + close))
}

/// Linear search for the line that begins with `[name]` and return the
/// LSP `Range` covering just `name`. Returns None if not found (which
/// can happen during incremental edits).
fn find_section_header_range(text: &str, name: &str) -> Option<Range> {
    let needle_open = format!("[{name}]");
    let needle_array = format!("[[{name}]]");
    let mut byte_offset = 0;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim_start();
        let leading = line.len() - trimmed.len();
        if trimmed.starts_with(&needle_open) || trimmed.starts_with(&needle_array) {
            let line_no = byte_pos_to_line(text, byte_offset);
            let col_open = leading + if trimmed.starts_with("[[") { 2 } else { 1 };
            return Some(Range::new(
                Position::new(line_no, col_open as u32),
                Position::new(line_no, (col_open + name.len()) as u32),
            ));
        }
        byte_offset += line.len();
    }
    None
}

/// Locate `field_name = …` inside the body of `[section]` and return the
/// LSP range for `field_name`.
fn find_field_range(text: &str, section: &str, field: &str) -> Option<Range> {
    let mut in_section = false;
    let mut byte_offset = 0;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if trimmed.starts_with('[') {
            // Section boundary
            let inner = trimmed.trim_start_matches('[');
            if let Some(end) = inner.find(']') {
                let name = inner[..end].trim();
                in_section = name == section;
                byte_offset += line.len();
                continue;
            }
        }
        if in_section {
            let leading = line.len() - trimmed.len();
            if let Some(eq) = trimmed.find('=') {
                let key = trimmed[..eq].trim();
                if key == field {
                    let line_no = byte_pos_to_line(text, byte_offset);
                    let col = leading;
                    return Some(Range::new(
                        Position::new(line_no, col as u32),
                        Position::new(line_no, (col + field.len()) as u32),
                    ));
                }
            }
        }
        byte_offset += line.len();
    }
    None
}

fn byte_pos_to_line(text: &str, byte: usize) -> u32 {
    text[..byte.min(text.len())].matches('\n').count() as u32
}

fn byte_range_to_range(text: &str, range: std::ops::Range<usize>) -> Range {
    Range::new(byte_to_position(text, range.start), byte_to_position(text, range.end))
}

fn byte_to_position(text: &str, byte: usize) -> Position {
    let prefix = &text[..byte.min(text.len())];
    let line = prefix.matches('\n').count() as u32;
    let last_nl = prefix.rfind('\n').map(|p| p + 1).unwrap_or(0);
    let col = (byte - last_nl) as u32;
    Position::new(line, col)
}

// ────────────────────────── Folder conflicts ──────────────────────────

/// Cross-file diagnostics: warn when a component section in this file
/// also appears in another `.toml` file in the same folder, since both
/// would attach the same component to the folder's entity.
///
/// Only checks sections whose names are registered components (so we
/// don't double-warn on things already flagged as unknown). Best-effort:
/// silently bails on filesystem errors or paths it can't resolve.
pub fn folder_conflict_diagnostics(
    schemas: &SchemaIndex,
    doc: &Document,
    uri: &Url,
) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    let Some(toml::Value::Table(table)) = &doc.parsed else {
        return out;
    };
    let Ok(this_path) = uri.to_file_path() else {
        return out;
    };
    let Some(parent) = this_path.parent() else {
        return out;
    };
    let Ok(entries) = std::fs::read_dir(parent) else {
        return out;
    };

    // Collect (section -> sibling file name) for every component section
    // in every sibling .toml that names a registered component.
    let mut sibling_sections: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path == this_path {
            continue;
        }
        if path.extension().and_then(|s| s.to_str()) != Some("toml") {
            continue;
        }
        // `_meta.toml` and similar underscore-prefixed files are folder
        // metadata, not entity components.
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if file_name.starts_with('_') {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(toml::Value::Table(sib_table)) = toml::from_str::<toml::Value>(&text) else {
            continue;
        };
        for sib_section in sib_table.keys() {
            if sib_section.starts_with('_') {
                continue;
            }
            if schemas.get(sib_section).is_none() {
                continue;
            }
            sibling_sections
                .entry(sib_section.clone())
                .or_default()
                .push(file_name.to_string());
        }
    }

    for section_name in table.keys() {
        if section_name.starts_with('_') {
            continue;
        }
        if schemas.get(section_name).is_none() {
            continue;
        }
        let Some(others) = sibling_sections.get(section_name) else {
            continue;
        };
        let Some(range) = find_section_header_range(&doc.text, section_name) else {
            continue;
        };
        out.push(Diagnostic {
            range,
            severity: Some(DiagnosticSeverity::WARNING),
            source: Some("ocarina".into()),
            message: format!(
                "duplicate component `{section_name}` in folder — also declared in {}. \
                 Each entity (folder) can have at most one of any component.",
                others.join(", ")
            ),
            ..Default::default()
        });
    }

    out
}

// ────────────────────────── Type checking ──────────────────────────

/// Validate a TOML value against a `FieldType`. Returns `None` if it's
/// fine, or a short "got X" sentence if it doesn't fit.
fn check_field_type(ty: ocarina_schema::FieldType, value: &toml::Value) -> Option<String> {
    use ocarina_schema::FieldType as F;
    match ty {
        F::Bool => match value {
            toml::Value::Boolean(_) => None,
            v => Some(format!("got {}", describe_value(v))),
        },
        F::Int => match value {
            toml::Value::Integer(_) => None,
            v => Some(format!("got {}", describe_value(v))),
        },
        F::Uint => match value {
            toml::Value::Integer(n) if *n >= 0 => None,
            toml::Value::Integer(n) => Some(format!("got negative integer `{n}`")),
            v => Some(format!("got {}", describe_value(v))),
        },
        F::Float => match value {
            // Accept integers too — TOML literals like `1` are fine for an f32 field.
            toml::Value::Float(_) | toml::Value::Integer(_) => None,
            v => Some(format!("got {}", describe_value(v))),
        },
        F::String => match value {
            toml::Value::String(_) => None,
            v => Some(format!("got {}", describe_value(v))),
        },
        F::Vec2 => check_numeric_array(value, 2),
        F::Vec3 => check_numeric_array(value, 3),
        F::Vec4 | F::Quat => check_numeric_array(value, 4),
        F::Color => match value {
            // Colors can be either a hex string or an array of 3 or 4 numbers.
            toml::Value::String(_) => None,
            toml::Value::Array(arr)
                if (arr.len() == 3 || arr.len() == 4) && arr.iter().all(is_numeric) =>
            {
                None
            }
            toml::Value::Array(arr) => Some(format!(
                "got array of length {} (expected 3 or 4 numbers, or hex string)",
                arr.len()
            )),
            v => Some(format!("got {}", describe_value(v))),
        },
        F::Other => None,
    }
}

fn check_numeric_array(value: &toml::Value, expected_len: usize) -> Option<String> {
    match value {
        toml::Value::Array(arr) if arr.len() == expected_len && arr.iter().all(is_numeric) => None,
        toml::Value::Array(arr) if arr.len() != expected_len => Some(format!(
            "got array of length {} (expected {expected_len})",
            arr.len()
        )),
        toml::Value::Array(_) => Some("got array with non-numeric elements".to_string()),
        v => Some(format!("got {}", describe_value(v))),
    }
}

fn is_numeric(v: &toml::Value) -> bool {
    matches!(v, toml::Value::Integer(_) | toml::Value::Float(_))
}

fn describe_value(v: &toml::Value) -> &'static str {
    match v {
        toml::Value::Boolean(_) => "boolean",
        toml::Value::Integer(_) => "integer",
        toml::Value::Float(_) => "float",
        toml::Value::String(_) => "string",
        toml::Value::Datetime(_) => "datetime",
        toml::Value::Array(_) => "array",
        toml::Value::Table(_) => "table",
    }
}

fn describe_field_type(ty: ocarina_schema::FieldType) -> &'static str {
    use ocarina_schema::FieldType as F;
    match ty {
        F::Bool => "a boolean",
        F::Int => "an integer",
        F::Uint => "a non-negative integer",
        F::Float => "a number",
        F::String => "a string",
        F::Vec2 => "a 2-element numeric array",
        F::Vec3 => "a 3-element numeric array",
        F::Vec4 => "a 4-element numeric array",
        F::Quat => "a 4-element numeric array (x, y, z, w)",
        F::Color => "a hex string or 3-/4-element numeric array",
        F::Other => "any value",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_section_walks_back() {
        let text = "[a]\nx = 1\n[b]\ny = 2\nz = 3\n";
        assert_eq!(current_section(text, 0).as_deref(), Some("a"));
        assert_eq!(current_section(text, 1).as_deref(), Some("a"));
        assert_eq!(current_section(text, 2).as_deref(), Some("b"));
        assert_eq!(current_section(text, 4).as_deref(), Some("b"));
    }

    #[test]
    fn find_section_header_basic() {
        let text = "[transform]\nposition = [0,0,0]\n";
        let r = find_section_header_range(text, "transform").unwrap();
        assert_eq!(r.start.line, 0);
        assert_eq!(r.start.character, 1);
        assert_eq!(r.end.character, 1 + "transform".len() as u32);
    }

    #[test]
    fn find_field_in_section() {
        let text = "[transform]\nposition = [0,0,0]\nrotation = [0,0,0]\n";
        let r = find_field_range(text, "transform", "rotation").unwrap();
        assert_eq!(r.start.line, 2);
        assert_eq!(r.start.character, 0);
    }
}
