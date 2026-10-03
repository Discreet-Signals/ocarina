//! WGSL reflection for post-process shaders.
//!
//! Parses a user-supplied `.wgsl` file with `naga` and produces a
//! structured description of every `@group/@binding` declaration:
//! engine-managed slots (scene_color, sampler, self_previous,
//! chain_input, PostUniforms) are recognised by their well-known
//! group/binding pair; everything else is a user-declared resource
//! that the renderer must allocate and bind.
//!
//! Magic-comment metadata: a `// @attr(value)` line directly above a
//! struct field controls editor presentation + default values. Only
//! parsed for fields inside uniform structs (the bag of tweakable
//! game-dev parameters); ignored elsewhere. Supported attributes:
//!
//! - `@default(0.5)` — initial value when the camera config doesn't
//!   override the field. Type-matched against the field's WGSL type.
//! - `@range(0, 1)` — slider min / max (editor only; renderer ignores).
//! - `@label("Feedback")` — display name in the editor.
//! - `@color` — flag to render as a colour picker rather than three
//!   independent sliders. Implicit on field name `*_color`.
//!
//! The renderer is the source of truth for which slots are
//! engine-managed; `RESERVED_SLOTS` lists them. Any other binding is
//! treated as user-managed.

use std::collections::BTreeMap;

use naga::front::wgsl;

/// Reserved engine-managed bindings. The renderer hooks these up
/// before invoking the user shader. A post-process shader doesn't
/// know its order in the chain: `input_color` is the lit scene on
/// the first pass and the previous pass's output thereafter.
pub const SLOT_INPUT_COLOR: (u32, u32) = (0, 0);
pub const SLOT_PREV_COLOR: (u32, u32) = (0, 1);
pub const SLOT_SAMPLER: (u32, u32) = (0, 2);
pub const SLOT_POST_UNIFORMS: (u32, u32) = (1, 0);

const RESERVED_SLOTS: [(u32, u32); 4] = [
    SLOT_INPUT_COLOR,
    SLOT_PREV_COLOR,
    SLOT_SAMPLER,
    SLOT_POST_UNIFORMS,
];

/// Top-level reflection result for a parsed shader.
#[derive(Debug, Default, Clone)]
pub struct ShaderReflection {
    /// All resource bindings the shader declares. Sorted by (group, binding).
    pub bindings: Vec<BindingInfo>,
}

/// One reflected resource binding.
#[derive(Debug, Clone)]
pub struct BindingInfo {
    pub group: u32,
    pub binding: u32,
    /// WGSL identifier the shader uses for this resource. Helpful for
    /// debug labels and editor display.
    pub var_name: String,
    pub kind: BindingKind,
}

impl BindingInfo {
    /// Is this binding one the engine manages (scene_color, sampler,
    /// PostUniforms, etc.)? When true the renderer hooks the binding
    /// up automatically; when false the renderer expects the user
    /// shader's sidecar metadata / camera config to drive it.
    #[must_use]
    pub fn is_engine_managed(&self) -> bool {
        RESERVED_SLOTS.contains(&(self.group, self.binding))
    }
}

#[derive(Debug, Clone)]
pub enum BindingKind {
    /// `var<uniform> name: SomeStruct` -- a uniform buffer with a
    /// known struct layout. `fields` enumerates each leaf in order
    /// with offset + WGSL type, so callers can pack values from a
    /// `HashMap<String, toml::Value>` keyed by field name.
    Uniform {
        struct_name: String,
        fields: Vec<UniformField>,
        size_bytes: u32,
    },
    /// `var name: texture_2d<f32>` (or `texture_2d_array`, etc.).
    Texture {
        sample_type: TextureSampleType,
        dimension: TextureDimension,
        multisampled: bool,
    },
    /// `var name: texture_storage_2d<format, access>`. The renderer
    /// allocates a persistent texture matching `format` per shader
    /// instance and binds it -- read_write access lets the shader use
    /// it as a scratch buffer surviving across frames.
    StorageTexture {
        format: StorageFormat,
        dimension: TextureDimension,
        access: StorageAccess,
    },
    /// `var name: sampler` -- non-comparison sampler.
    Sampler,
    /// `var name: sampler_comparison` -- depth-compare sampler.
    SamplerComparison,
    /// Anything we don't yet understand. Logged once during reflection
    /// so the user can see they hit an unsupported case.
    Unsupported(String),
}

#[derive(Debug, Clone)]
pub struct UniformField {
    pub name: String,
    pub offset: u32,
    pub size: u32,
    pub kind: ScalarKind,
    pub metadata: FieldMetadata,
}

/// Editor / default-value metadata extracted from `// @attr(...)`
/// comments above each struct field.
#[derive(Debug, Default, Clone)]
pub struct FieldMetadata {
    pub default: Option<DefaultValue>,
    pub range: Option<(f32, f32)>,
    pub label: Option<String>,
    /// True when the field name ends in `_color` OR the user wrote
    /// `// @color` above it. Editor renders a colour picker.
    pub color: bool,
}

#[derive(Debug, Clone)]
pub enum DefaultValue {
    F32(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
}

/// Naga scalar / vector kinds we support in uniform fields. Anything
/// outside this set comes back as `BindingKind::Unsupported` for now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarKind {
    F32,
    Vec2,
    Vec3,
    Vec4,
    /// `i32` / `u32` -- packed but no editor UI yet.
    I32,
    U32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureSampleType {
    Float { filterable: bool },
    Depth,
    Sint,
    Uint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureDimension {
    D1,
    D2,
    D3,
    D2Array,
    Cube,
    CubeArray,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageFormat {
    Rgba8Unorm,
    Rgba8Snorm,
    Rgba16Float,
    Rgba32Float,
    R32Float,
    /// Future formats land here; renderer rejects unknown.
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageAccess {
    Load,
    Store,
    LoadStore,
}

#[derive(Debug, thiserror::Error)]
pub enum ReflectError {
    #[error("WGSL parse error: {0}")]
    Parse(String),
}

/// Parse + reflect a WGSL source. Magic-comment metadata is parsed
/// from the source text directly (naga discards comments).
pub fn reflect(source: &str) -> Result<ShaderReflection, ReflectError> {
    let module = wgsl::parse_str(source)
        .map_err(|e| ReflectError::Parse(e.emit_to_string(source)))?;

    let mut field_metadata = parse_field_metadata(source);

    let mut bindings: Vec<BindingInfo> = Vec::new();
    for (_h, var) in module.global_variables.iter() {
        let Some(binding) = var.binding.as_ref() else {
            continue;
        };
        let group = binding.group;
        let bind = binding.binding;
        let var_name = var.name.clone().unwrap_or_else(|| format!("var_g{group}b{bind}"));
        let kind = classify_binding(&module, var, &mut field_metadata);
        bindings.push(BindingInfo {
            group,
            binding: bind,
            var_name,
            kind,
        });
    }
    bindings.sort_by_key(|b| (b.group, b.binding));

    Ok(ShaderReflection { bindings })
}

fn classify_binding(
    module: &naga::Module,
    var: &naga::GlobalVariable,
    field_metadata: &mut BTreeMap<String, FieldMetadata>,
) -> BindingKind {
    let ty = &module.types[var.ty];
    match var.space {
        naga::AddressSpace::Uniform => uniform_kind(module, ty, field_metadata),
        naga::AddressSpace::Handle => handle_kind(ty),
        naga::AddressSpace::Storage { .. } => {
            BindingKind::Unsupported("storage buffer (not yet wired)".into())
        }
        other => BindingKind::Unsupported(format!("address space {other:?}")),
    }
}

fn uniform_kind(
    module: &naga::Module,
    ty: &naga::Type,
    field_metadata: &mut BTreeMap<String, FieldMetadata>,
) -> BindingKind {
    let struct_name = ty.name.clone().unwrap_or_else(|| "<anon>".into());
    let naga::TypeInner::Struct { ref members, span } = ty.inner else {
        return BindingKind::Unsupported(format!(
            "uniform `{}` is not a struct",
            struct_name
        ));
    };

    let mut fields: Vec<UniformField> = Vec::with_capacity(members.len());
    for m in members {
        let name = m
            .name
            .clone()
            .unwrap_or_else(|| format!("field_{}", m.offset));
        let inner = &module.types[m.ty];
        let Some(scalar) = wgsl_scalar_kind(&inner.inner) else {
            // Non-scalar / non-vector field; keep going but skip
            // editor metadata. Caller can still pack via offset+size.
            fields.push(UniformField {
                name: name.clone(),
                offset: m.offset,
                size: inner_size(&inner.inner).unwrap_or(0),
                kind: ScalarKind::F32, // best effort
                metadata: field_metadata.remove(&name).unwrap_or_default(),
            });
            continue;
        };
        let mut metadata = field_metadata.remove(&name).unwrap_or_default();
        if metadata.label.is_none() {
            metadata.label = Some(humanize_name(&name));
        }
        if !metadata.color && name.ends_with("_color") {
            metadata.color = true;
        }
        fields.push(UniformField {
            name,
            offset: m.offset,
            size: inner_size(&inner.inner).unwrap_or(0),
            kind: scalar,
            metadata,
        });
    }

    BindingKind::Uniform {
        struct_name,
        fields,
        size_bytes: span,
    }
}

fn handle_kind(ty: &naga::Type) -> BindingKind {
    match &ty.inner {
        naga::TypeInner::Image {
            dim,
            arrayed,
            class,
        } => {
            let dimension = naga_dim_to_dim(*dim, *arrayed);
            match class {
                naga::ImageClass::Sampled { kind, multi } => BindingKind::Texture {
                    sample_type: naga_sample_kind_to_sample_type(*kind),
                    dimension,
                    multisampled: *multi,
                },
                naga::ImageClass::Storage { format, access } => {
                    BindingKind::StorageTexture {
                        format: naga_storage_format_to_format(*format),
                        dimension,
                        access: naga_storage_access_to_access(*access),
                    }
                }
                naga::ImageClass::Depth { multi } => BindingKind::Texture {
                    sample_type: TextureSampleType::Depth,
                    dimension,
                    multisampled: *multi,
                },
            }
        }
        naga::TypeInner::Sampler { comparison } => {
            if *comparison {
                BindingKind::SamplerComparison
            } else {
                BindingKind::Sampler
            }
        }
        other => BindingKind::Unsupported(format!("handle type {other:?}")),
    }
}

fn wgsl_scalar_kind(inner: &naga::TypeInner) -> Option<ScalarKind> {
    match inner {
        naga::TypeInner::Scalar(s) => match (s.kind, s.width) {
            (naga::ScalarKind::Float, 4) => Some(ScalarKind::F32),
            (naga::ScalarKind::Sint, 4) => Some(ScalarKind::I32),
            (naga::ScalarKind::Uint, 4) => Some(ScalarKind::U32),
            _ => None,
        },
        naga::TypeInner::Vector { size, scalar } => {
            if scalar.kind != naga::ScalarKind::Float || scalar.width != 4 {
                return None;
            }
            Some(match size {
                naga::VectorSize::Bi => ScalarKind::Vec2,
                naga::VectorSize::Tri => ScalarKind::Vec3,
                naga::VectorSize::Quad => ScalarKind::Vec4,
            })
        }
        _ => None,
    }
}

fn inner_size(inner: &naga::TypeInner) -> Option<u32> {
    match inner {
        naga::TypeInner::Scalar(s) => Some(u32::from(s.width)),
        naga::TypeInner::Vector { size, scalar } => {
            let n = match size {
                naga::VectorSize::Bi => 2,
                naga::VectorSize::Tri => 3,
                naga::VectorSize::Quad => 4,
            };
            Some(n * u32::from(scalar.width))
        }
        _ => None,
    }
}

fn naga_dim_to_dim(dim: naga::ImageDimension, arrayed: bool) -> TextureDimension {
    match (dim, arrayed) {
        (naga::ImageDimension::D1, false) => TextureDimension::D1,
        (naga::ImageDimension::D2, false) => TextureDimension::D2,
        (naga::ImageDimension::D2, true) => TextureDimension::D2Array,
        (naga::ImageDimension::D3, false) => TextureDimension::D3,
        (naga::ImageDimension::Cube, false) => TextureDimension::Cube,
        (naga::ImageDimension::Cube, true) => TextureDimension::CubeArray,
        // 1D arrayed isn't a real wgpu type but naga lets it through.
        // Fall back to D1.
        (naga::ImageDimension::D1, true) => TextureDimension::D1,
        // 3D arrayed doesn't exist either.
        (naga::ImageDimension::D3, true) => TextureDimension::D3,
    }
}

fn naga_sample_kind_to_sample_type(kind: naga::ScalarKind) -> TextureSampleType {
    match kind {
        naga::ScalarKind::Float => TextureSampleType::Float { filterable: true },
        naga::ScalarKind::Sint => TextureSampleType::Sint,
        naga::ScalarKind::Uint => TextureSampleType::Uint,
        naga::ScalarKind::Bool | naga::ScalarKind::AbstractInt | naga::ScalarKind::AbstractFloat => {
            TextureSampleType::Float { filterable: true }
        }
    }
}

fn naga_storage_format_to_format(f: naga::StorageFormat) -> StorageFormat {
    match f {
        naga::StorageFormat::Rgba8Unorm => StorageFormat::Rgba8Unorm,
        naga::StorageFormat::Rgba8Snorm => StorageFormat::Rgba8Snorm,
        naga::StorageFormat::Rgba16Float => StorageFormat::Rgba16Float,
        naga::StorageFormat::Rgba32Float => StorageFormat::Rgba32Float,
        naga::StorageFormat::R32Float => StorageFormat::R32Float,
        _ => StorageFormat::Other,
    }
}

fn naga_storage_access_to_access(a: naga::StorageAccess) -> StorageAccess {
    let load = a.contains(naga::StorageAccess::LOAD);
    let store = a.contains(naga::StorageAccess::STORE);
    match (load, store) {
        (true, true) => StorageAccess::LoadStore,
        (true, false) => StorageAccess::Load,
        (false, true) => StorageAccess::Store,
        // Default to load if neither is set; shouldn't happen in
        // valid WGSL.
        (false, false) => StorageAccess::Load,
    }
}

// ─────────────────────────── magic-comment parser ───────────────────────────

/// Walk the source line-by-line collecting `// @attr(value)` comments
/// and associating them with the next struct field declaration.
/// Returns `field_name → FieldMetadata`.
///
/// Heuristic, not a full WGSL parser: looks for lines matching
/// `IDENT: TYPE,` after a contiguous run of `// @attr(...)` lines.
/// Good enough for the conventional WGSL formatting; users with
/// exotic layouts get default metadata, which is fine.
fn parse_field_metadata(source: &str) -> BTreeMap<String, FieldMetadata> {
    let mut out = BTreeMap::new();
    let mut pending = FieldMetadata::default();
    let mut have_pending = false;

    for raw in source.lines() {
        let line = raw.trim();
        if let Some(rest) = line.strip_prefix("//") {
            let body = rest.trim();
            if let Some(attrs) = body.strip_prefix('@') {
                // One attribute or several space-separated.
                for token in tokenize_attrs(attrs) {
                    apply_attr(&token, &mut pending);
                    have_pending = true;
                }
            }
            continue;
        }
        // Blank line resets pending metadata so it doesn't bleed into
        // a far-away later field.
        if line.is_empty() {
            pending = FieldMetadata::default();
            have_pending = false;
            continue;
        }
        if have_pending {
            if let Some(name) = first_field_ident(line) {
                out.insert(name, std::mem::take(&mut pending));
            }
            have_pending = false;
            pending = FieldMetadata::default();
        }
    }
    out
}

/// Split `default(0.5) range(0,1) label("Feedback")` into separate
/// attribute tokens, respecting parens.
fn tokenize_attrs(attrs: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in attrs.chars() {
        match c {
            '(' => {
                depth += 1;
                cur.push(c);
            }
            ')' => {
                depth -= 1;
                cur.push(c);
            }
            ' ' | '\t' if depth == 0 => {
                if !cur.is_empty() {
                    out.push(cur.clone());
                    cur.clear();
                }
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn apply_attr(token: &str, meta: &mut FieldMetadata) {
    // Tokens after the first arrive with their `@` prefix intact
    // (only the very first attr after `//` was stripped); strip it
    // here so name matching below sees `range`, not `@range`.
    let token = token.trim_start_matches('@');
    let (name, body) = match token.find('(') {
        Some(open) => {
            let name = &token[..open];
            let body = token[open + 1..].trim_end_matches(')');
            (name, Some(body.trim()))
        }
        None => (token, None),
    };
    match name {
        "default" => {
            if let Some(body) = body {
                meta.default = parse_default(body);
            }
        }
        "range" => {
            if let Some(body) = body {
                let parts: Vec<&str> = body.split(',').map(str::trim).collect();
                if parts.len() == 2 {
                    if let (Ok(lo), Ok(hi)) = (parts[0].parse::<f32>(), parts[1].parse::<f32>()) {
                        meta.range = Some((lo, hi));
                    }
                }
            }
        }
        "label" => {
            if let Some(body) = body {
                meta.label = Some(body.trim_matches('"').to_string());
            }
        }
        "color" => {
            meta.color = true;
        }
        _ => {} // unknown attributes are ignored, not errored
    }
}

fn parse_default(body: &str) -> Option<DefaultValue> {
    // Single number → F32. `[a, b]` / `[a, b, c]` / `[a, b, c, d]` →
    // Vec2 / Vec3 / Vec4. Anything else → None.
    let body = body.trim();
    if let Some(inside) = body.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        let parts: Vec<f32> = inside
            .split(',')
            .filter_map(|s| s.trim().parse::<f32>().ok())
            .collect();
        return match parts.len() {
            2 => Some(DefaultValue::Vec2([parts[0], parts[1]])),
            3 => Some(DefaultValue::Vec3([parts[0], parts[1], parts[2]])),
            4 => Some(DefaultValue::Vec4([parts[0], parts[1], parts[2], parts[3]])),
            _ => None,
        };
    }
    body.parse::<f32>().ok().map(DefaultValue::F32)
}

/// Try to extract `name` from `name: type,` or `name : type;`. Used
/// only when we already know we're inside a struct body (matched by
/// the metadata parser's pending-attr state).
fn first_field_ident(line: &str) -> Option<String> {
    let s = line.trim_start();
    let colon = s.find(':')?;
    let name = s[..colon].trim();
    if name.is_empty() {
        return None;
    }
    // Reject obvious non-identifier starts.
    if !name.chars().next().unwrap().is_ascii_alphabetic() {
        return None;
    }
    if name.chars().any(|c| !c.is_ascii_alphanumeric() && c != '_') {
        return None;
    }
    Some(name.to_string())
}

fn humanize_name(name: &str) -> String {
    // `feedback_amount` → "Feedback Amount". Cheap snake-to-Title.
    let mut out = String::with_capacity(name.len());
    for (i, part) in name.split('_').enumerate() {
        if i > 0 {
            out.push(' ');
        }
        let mut chars = part.chars();
        if let Some(first) = chars.next() {
            out.push(first.to_ascii_uppercase());
            out.extend(chars);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_uniform_struct() {
        let src = r#"
            struct Params {
                feedback: f32,
                tint: vec3<f32>,
            };
            @group(2) @binding(0) var<uniform> params: Params;
        "#;
        let r = reflect(src).unwrap();
        assert_eq!(r.bindings.len(), 1);
        let BindingKind::Uniform { fields, .. } = &r.bindings[0].kind else {
            panic!("expected uniform");
        };
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name, "feedback");
        assert_eq!(fields[0].kind, ScalarKind::F32);
        assert_eq!(fields[1].name, "tint");
        assert_eq!(fields[1].kind, ScalarKind::Vec3);
    }

    #[test]
    fn picks_up_magic_comment_metadata() {
        let src = r#"
            struct Params {
                // @default(0.5) @range(0, 1) @label("Feedback")
                feedback: f32,
                // @default([1.0, 0.5, 0.2]) @color
                tint: vec3<f32>,
            };
            @group(2) @binding(0) var<uniform> params: Params;
        "#;
        let r = reflect(src).unwrap();
        let BindingKind::Uniform { fields, .. } = &r.bindings[0].kind else {
            panic!("expected uniform");
        };
        let f = &fields[0];
        assert!(matches!(f.metadata.default, Some(DefaultValue::F32(_))));
        assert_eq!(f.metadata.range, Some((0.0, 1.0)));
        assert_eq!(f.metadata.label.as_deref(), Some("Feedback"));
        let t = &fields[1];
        assert!(matches!(t.metadata.default, Some(DefaultValue::Vec3(_))));
        assert!(t.metadata.color);
    }

    #[test]
    fn detects_engine_managed_slot() {
        let src = r#"
            @group(0) @binding(0) var scene_color: texture_2d<f32>;
        "#;
        let r = reflect(src).unwrap();
        assert_eq!(r.bindings.len(), 1);
        assert!(r.bindings[0].is_engine_managed());
    }
}
