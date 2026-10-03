//! Manual TOML loader for `[[script]]` array-of-tables.
//!
//! `Scripts` doesn't go through `#[derive(Component)]` because its
//! authored form is an *array* of tables, not a single table -- every
//! other component is one `[section]` keyed on a snake_case name and
//! deserialised straight into the struct. We register the loader
//! by hand here, scoped to the section name `"script"`, so the
//! existing scene loader's `find_loader` dispatch can locate it
//! alongside the derive-generated entries.

use ocarina_schema::{
    inventory, toml as schema_toml, ComponentLoader,
};

use crate::components::{ScriptRef, Scripts};

fn load_scripts(
    entity: &mut bevy_ecs::world::EntityWorldMut<'_>,
    value: &schema_toml::Value,
) -> Result<(), String> {
    // `[[script]]` parses as a top-level array of tables. The loader
    // also accepts a single table for ergonomics -- equivalent to one
    // `[[script]]` entry.
    let items: Vec<ScriptRef> = match value {
        schema_toml::Value::Array(arr) => arr
            .iter()
            .map(|elem| {
                elem.clone()
                    .try_into::<ScriptRef>()
                    .map_err(|e: schema_toml::de::Error| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?,
        schema_toml::Value::Table(_) => {
            let one: ScriptRef = value
                .clone()
                .try_into()
                .map_err(|e: schema_toml::de::Error| e.to_string())?;
            vec![one]
        }
        other => {
            return Err(format!(
                "expected `[[script]]` array of tables (or a single `[script]` table); got {}",
                describe(other)
            ));
        }
    };

    // Multiple `[[script]]` blocks in one TOML file produce one
    // `apply_toml_file` call per entry, so we merge into any existing
    // `Scripts` rather than replacing it.
    if let Some(mut existing) = entity.get_mut::<Scripts>() {
        existing.items.extend(items);
    } else {
        entity.insert(Scripts { items });
    }
    Ok(())
}

fn describe(v: &schema_toml::Value) -> &'static str {
    match v {
        schema_toml::Value::Boolean(_) => "boolean",
        schema_toml::Value::Integer(_) => "integer",
        schema_toml::Value::Float(_) => "float",
        schema_toml::Value::String(_) => "string",
        schema_toml::Value::Datetime(_) => "datetime",
        schema_toml::Value::Array(_) => "array",
        schema_toml::Value::Table(_) => "table",
    }
}

inventory::submit! {
    ComponentLoader {
        section_name: "script",
        rust_type: "ocarina_core::components::Scripts",
        load: load_scripts,
    }
}
