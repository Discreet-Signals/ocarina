//! `#[derive(Component)]` — one derive that produces:
//!
//! 1. An `impl bevy_ecs::component::Component` for the struct.
//! 2. A `ComponentRegistration` submitted to `ocarina_schema`'s inventory,
//!    carrying a schema descriptor (name, doc, fields, widget hints).
//! 3. A `ComponentLoader` submitted to `ocarina_schema`'s inventory,
//!    paired with the schema, that parses a TOML value into the struct
//!    and attaches it to an entity.
//!
//! The schema is the source of truth for the LSP, the editor's lens
//! panel, and the `ocarina schema` JSON emitter. Users don't write it —
//! it's generated from their struct definition + doc comments +
//! `#[ocarina(...)]` field attributes.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, spanned::Spanned, Data, DeriveInput, Fields, Lit, LitStr, Meta, Type,
};

// ─────────────────────── Entry point ───────────────────────

/// `#[derive(Component)]` with optional `#[ocarina(...)]` field attrs.
///
/// Accepted field-level attributes:
/// - `#[ocarina(widget = "vec3" | "color" | "euler_rotation" | "scalar" | "path")]`
/// - `#[ocarina(default = "[1.0, 1.0, 1.0]")]` — verbatim TOML literal
/// - `#[ocarina(name = "custom_section_name")]` on the struct to override
///   the derived snake_case section name.
#[proc_macro_derive(Component, attributes(ocarina))]
pub fn derive_component(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    match expand(&ast) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

// ─────────────────────── Implementation ────────────────────

fn expand(ast: &DeriveInput) -> syn::Result<TokenStream2> {
    let struct_name = &ast.ident;
    let struct_doc = doc_comment_of(&ast.attrs);
    let (section_override, _struct_storage) = parse_struct_attrs(&ast.attrs)?;
    let section_name = section_override.unwrap_or_else(|| to_snake_case(&struct_name.to_string()));

    let fields = match &ast.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(named) => &named.named,
            Fields::Unit => {
                // Marker component: empty field list.
                return Ok(emit_output(
                    struct_name,
                    &section_name,
                    &struct_doc,
                    vec![],
                ));
            }
            _ => {
                return Err(syn::Error::new(
                    ast.span(),
                    "#[derive(Component)] only supports structs with named fields or unit structs",
                ))
            }
        },
        _ => {
            return Err(syn::Error::new(
                ast.span(),
                "#[derive(Component)] only supports structs",
            ))
        }
    };

    let mut field_entries: Vec<FieldInfo> = Vec::new();
    for f in fields {
        let field_name = f.ident.as_ref().expect("named field").to_string();
        let field_doc = doc_comment_of(&f.attrs);
        let (widget, default_toml) = parse_field_attrs(&f.attrs)?;
        let rust_type_str = type_to_string(&f.ty);
        let field_type = infer_field_type(&f.ty);
        field_entries.push(FieldInfo {
            name: field_name,
            doc: field_doc,
            widget,
            default_toml,
            rust_type_str,
            field_type,
        });
    }

    Ok(emit_output(
        struct_name,
        &section_name,
        &struct_doc,
        field_entries,
    ))
}

struct FieldInfo {
    name: String,
    doc: String,
    widget: String,     // "Auto" | "Vec3" | "Color" | ...
    default_toml: Option<String>,
    rust_type_str: String,
    field_type: String, // "Bool" | "Int" | ...
}

fn emit_output(
    struct_name: &syn::Ident,
    section_name: &str,
    struct_doc: &str,
    fields: Vec<FieldInfo>,
) -> TokenStream2 {
    let rust_type = struct_name.to_string();
    let section_lit = LitStr::new(section_name, proc_macro2::Span::call_site());
    let rust_type_lit = LitStr::new(&rust_type, proc_macro2::Span::call_site());
    let struct_doc_lit = LitStr::new(struct_doc, proc_macro2::Span::call_site());

    // Capture rustc's working directory at proc-macro expansion time.
    // For Cargo builds this is the workspace root, which is the directory
    // `file!()` paths are relative to. Embedding it as a literal lets the
    // emitted `concat!(..., "/", file!())` produce an absolute path even
    // in multi-crate workspaces, where `CARGO_MANIFEST_DIR` would be
    // wrong (it's the inner crate's root, not the workspace's).
    let workspace_root = std::env::current_dir()
        .ok()
        .and_then(|p| p.to_str().map(str::to_owned))
        .unwrap_or_default();
    let workspace_root_lit = LitStr::new(&workspace_root, proc_macro2::Span::call_site());

    // Build the field vec construction for the schema.
    let field_tokens = fields.iter().map(|f| {
        let name_lit = LitStr::new(&f.name, proc_macro2::Span::call_site());
        let doc_lit = LitStr::new(&f.doc, proc_macro2::Span::call_site());
        let rust_ty_lit = LitStr::new(&f.rust_type_str, proc_macro2::Span::call_site());
        let widget_variant = format_ident!("{}", f.widget);
        let field_type_variant = format_ident!("{}", f.field_type);
        let default_tokens = match &f.default_toml {
            Some(d) => {
                let d_lit = LitStr::new(d, proc_macro2::Span::call_site());
                quote! { ::std::option::Option::Some(#d_lit.to_string()) }
            }
            None => quote! { ::std::option::Option::None },
        };
        quote! {
            ::ocarina_schema::FieldSchema {
                name: #name_lit.to_string(),
                ty: ::ocarina_schema::FieldType::#field_type_variant,
                doc: #doc_lit.to_string(),
                widget: ::ocarina_schema::Widget::#widget_variant,
                rust_type: #rust_ty_lit.to_string(),
                default_toml: #default_tokens,
            }
        }
    });

    // Build a fn name that won't collide across derives.
    let schema_fn = format_ident!("__ocarina_schema_{}", struct_name.to_string().to_lowercase());
    let loader_load_fn = format_ident!("__ocarina_load_{}", struct_name.to_string().to_lowercase());

    quote! {
        // bevy_ecs Component impl — a simple Table-storage mutable component.
        impl ::bevy_ecs::component::Component for #struct_name {
            const STORAGE_TYPE: ::bevy_ecs::component::StorageType =
                ::bevy_ecs::component::StorageType::Table;
            type Mutability = ::bevy_ecs::component::Mutable;
        }

        // Schema-building fn — invoked lazily by consumers (LSP, schema dump).
        // `file!()` and `line!()` evaluate at the macro call site, which is
        // wherever the user wrote `#[derive(Component)]`. Combined with the
        // captured workspace root we get an absolute path.
        #[allow(non_snake_case)]
        fn #schema_fn() -> ::ocarina_schema::ComponentSchema {
            ::ocarina_schema::ComponentSchema {
                name: #section_lit.to_string(),
                rust_type: #rust_type_lit.to_string(),
                doc: #struct_doc_lit.to_string(),
                source_file: ::std::concat!(
                    #workspace_root_lit,
                    "/",
                    ::std::file!()
                ).to_string(),
                source_line: ::std::line!(),
                fields: vec![
                    #(#field_tokens),*
                ],
            }
        }

        ::ocarina_schema::inventory::submit! {
            ::ocarina_schema::ComponentRegistration {
                build: #schema_fn,
            }
        }

        // Loader — type-erased TOML-value → entity insertion.
        #[allow(non_snake_case)]
        fn #loader_load_fn(
            entity: &mut ::bevy_ecs::world::EntityWorldMut<'_>,
            value: &::ocarina_schema::toml::Value,
        ) -> ::std::result::Result<(), ::std::string::String> {
            ::ocarina_schema::load_component::<#struct_name>(entity, value)
        }

        ::ocarina_schema::inventory::submit! {
            ::ocarina_schema::ComponentLoader {
                section_name: #section_lit,
                rust_type: #rust_type_lit,
                load: #loader_load_fn,
            }
        }
    }
}

// ─────────────────── Attribute / doc parsing ────────────────────

/// Pull every `/// doc` line off the attrs and join them into one string.
fn doc_comment_of(attrs: &[syn::Attribute]) -> String {
    let mut buf = String::new();
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        if let Meta::NameValue(nv) = &attr.meta {
            if let syn::Expr::Lit(expr_lit) = &nv.value {
                if let Lit::Str(s) = &expr_lit.lit {
                    if !buf.is_empty() {
                        buf.push('\n');
                    }
                    buf.push_str(s.value().trim_start_matches(' '));
                }
            }
        }
    }
    buf
}

fn parse_struct_attrs(attrs: &[syn::Attribute]) -> syn::Result<(Option<String>, ())> {
    let mut name: Option<String> = None;
    for attr in attrs {
        if !attr.path().is_ident("ocarina") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("name") {
                let s: LitStr = meta.value()?.parse()?;
                name = Some(s.value());
                Ok(())
            } else {
                Err(meta.error("unknown #[ocarina(...)] struct attribute"))
            }
        })?;
    }
    Ok((name, ()))
}

fn parse_field_attrs(attrs: &[syn::Attribute]) -> syn::Result<(String, Option<String>)> {
    let mut widget = "Auto".to_string();
    let mut default_toml: Option<String> = None;
    for attr in attrs {
        if !attr.path().is_ident("ocarina") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("widget") {
                let s: LitStr = meta.value()?.parse()?;
                widget = normalize_widget(&s.value()).ok_or_else(|| {
                    meta.error(
                        "unknown widget; expected: auto | scalar | vec3 | euler_rotation | color | path",
                    )
                })?;
                Ok(())
            } else if meta.path.is_ident("default") {
                let s: LitStr = meta.value()?.parse()?;
                default_toml = Some(s.value());
                Ok(())
            } else {
                Err(meta.error("unknown #[ocarina(...)] field attribute"))
            }
        })?;
    }
    Ok((widget, default_toml))
}

fn normalize_widget(raw: &str) -> Option<String> {
    Some(
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => "Auto",
            "scalar" => "Scalar",
            "vec3" => "Vec3",
            "euler_rotation" | "euler" => "EulerRotation",
            "color" => "Color",
            "path" => "Path",
            _ => return None,
        }
        .to_string(),
    )
}

// ─────────────────── Type inference ────────────────────

fn type_to_string(ty: &Type) -> String {
    quote!(#ty).to_string().replace(' ', "")
}

/// Map a Rust type to a `FieldType` variant name. Conservative — unknown
/// types fall through to `Other`, which is still fine for the schema.
fn infer_field_type(ty: &Type) -> String {
    let name = type_to_string(ty);
    // Last-path-segment match.
    let last = name.rsplit("::").next().unwrap_or(&name);
    match last {
        "bool" => "Bool",
        "i8" | "i16" | "i32" | "i64" | "isize" => "Int",
        "u8" | "u16" | "u32" | "u64" | "usize" => "Uint",
        "f32" | "f64" => "Float",
        "String" | "str" => "String",
        "Vec2" => "Vec2",
        "Vec3" => "Vec3",
        "Vec4" => "Vec4",
        "Quat" => "Quat",
        "Color" | "Color32" => "Color",
        _ => "Other",
    }
    .to_string()
}

fn to_snake_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    let mut prev_lower = false;
    for (i, ch) in s.chars().enumerate() {
        if ch.is_ascii_uppercase() {
            if i != 0 && prev_lower {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
            prev_lower = false;
        } else {
            out.push(ch);
            prev_lower = ch.is_ascii_lowercase() || ch.is_ascii_digit();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::to_snake_case;

    #[test]
    fn snake_case_basics() {
        assert_eq!(to_snake_case("Transform"), "transform");
        assert_eq!(to_snake_case("CharacterController"), "character_controller");
        assert_eq!(to_snake_case("BasicMesh"), "basic_mesh");
        assert_eq!(to_snake_case("Single"), "single");
        // Consecutive uppercase runs are lowercased without a split —
        // simple behavior matching what most codebases actually name
        // things. Users who want a custom TOML section name can pass
        // `#[ocarina(name = "...")]`.
        assert_eq!(to_snake_case("OSCComponent"), "osccomponent");
    }
}
