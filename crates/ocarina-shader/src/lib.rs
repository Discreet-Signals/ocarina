//! Shader compilation.
//!
//! The canonical authoring language for Ocarina is HLSL. This crate wraps
//! DXC (via `hassle-rs`) to produce SPIR-V that wgpu can consume directly,
//! and will later grow DXIL and MSL outputs for backends that want them.
//!
//! Runtime compilation is best-effort: if `libdxcompiler` isn't present on
//! the system, [`compile_hlsl_to_spirv`] returns [`ShaderError::DxcMissing`]
//! so callers can fall back to pre-compiled bytecode or WGSL.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ShaderError {
    /// DXC dynamic library couldn't be loaded. Install DXC and ensure
    /// `libdxcompiler.{dylib,so,dll}` is on the library search path.
    #[error("DXC not available: {0}")]
    DxcMissing(String),

    /// The HLSL source compiled, but produced an error. The inner string is
    /// DXC's diagnostic output.
    #[error("HLSL compile failed:\n{0}")]
    CompileFailed(String),

    /// SPIR-V output was not 4-byte-aligned — should never happen for a
    /// successful DXC compile and indicates a toolchain bug.
    #[error("SPIR-V output has invalid size ({0} bytes; not 4-byte aligned)")]
    InvalidSpirv(usize),
}

/// Shader pipeline stage. Drives the DXC target profile selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Vertex,
    Fragment,
    Compute,
}

impl Stage {
    fn target_profile(self) -> &'static str {
        // Shader Model 6.0 is the floor for DXC's `-spirv` output.
        match self {
            Self::Vertex => "vs_6_0",
            Self::Fragment => "ps_6_0",
            Self::Compute => "cs_6_0",
        }
    }
}

/// Compile HLSL source to SPIR-V (a `Vec<u32>` of 4-byte words, which is
/// exactly the shape wgpu's `ShaderSource::SpirV` wants).
///
/// `source_name` is used in DXC diagnostics (e.g. `"lit.hlsl"`).
pub fn compile_hlsl_to_spirv(
    source_name: &str,
    source: &str,
    entry_point: &str,
    stage: Stage,
) -> Result<Vec<u32>, ShaderError> {
    // `-spirv` switches DXC's codegen to SPIR-V. `-fvk-use-dx-layout`
    // keeps HLSL-style tight packing so CBuffers match our uploaded
    // `repr(C)` uniforms.
    let bytes = hassle_rs::compile_hlsl(
        source_name,
        source,
        entry_point,
        stage.target_profile(),
        &["-spirv", "-fvk-use-dx-layout"],
        &[],
    )
    .map_err(translate_err)?;

    if bytes.len() % 4 != 0 {
        return Err(ShaderError::InvalidSpirv(bytes.len()));
    }
    let mut words = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(words)
}

/// Test whether DXC is loadable on this system without compiling anything.
/// Cheap probe suitable for gating UI features.
#[must_use]
pub fn dxc_available() -> bool {
    // `Dxc::new(None)` tries to locate and load the dynamic library. Dropping
    // it immediately is fine — we just want to know if the load succeeded.
    hassle_rs::Dxc::new(None).is_ok()
}

fn translate_err(e: hassle_rs::HassleError) -> ShaderError {
    match e {
        hassle_rs::HassleError::LibLoadingError(msg) => ShaderError::DxcMissing(msg.to_string()),
        hassle_rs::HassleError::LoadLibraryError { filename, inner } => {
            ShaderError::DxcMissing(format!("{}: {}", filename.display(), inner))
        }
        hassle_rs::HassleError::CompileError(msg) => ShaderError::CompileFailed(msg),
        other => ShaderError::CompileFailed(format!("{other:?}")),
    }
}
