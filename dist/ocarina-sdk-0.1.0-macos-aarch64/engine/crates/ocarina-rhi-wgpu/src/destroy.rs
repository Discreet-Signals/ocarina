//! Deferred resource destruction.
//!
//! When engine code calls `destroy_*`, the resource is not released immediately
//! — it is enqueued for release after `MAX_FRAMES_IN_FLIGHT` frames have
//! retired past the current submit. This makes "destroy + submit that uses
//! this resource" safe: the GPU is guaranteed to finish referencing the
//! resource before we release it.
//!
//! Every `submit` followed by a `present` advances the frame counter; older
//! destroy batches get drained then.

use std::collections::VecDeque;

/// How many CPU-submitted frames can be in-flight before we require a flush.
///
/// wgpu tracks real fence completion internally — this is a simple
/// conservative cap that keeps resources around "long enough." For the
/// v1 editor workload where we never queue more than 2 frames ahead, this
/// is plenty; the Phase 6 ship runtime may revisit.
pub const MAX_FRAMES_IN_FLIGHT: usize = 3;

/// A scheduled deletion, tagged with the slot/generation that identifies
/// which pool entry to drop. Resource kind is distinguished by which queue
/// it lives on inside [`DestroyQueue`].
#[derive(Debug, Clone, Copy)]
pub struct Pending {
    pub slot: u32,
    pub generation: u32,
}

#[derive(Default)]
pub struct DestroyQueue {
    frames: VecDeque<FrameBucket>,
    current: FrameBucket,
}

#[derive(Default, Debug)]
pub struct FrameBucket {
    pub buffers: Vec<Pending>,
    pub textures: Vec<Pending>,
    pub samplers: Vec<Pending>,
    pub shaders: Vec<Pending>,
    pub bgls: Vec<Pending>,
    pub bgs: Vec<Pending>,
    pub pipelines: Vec<Pending>,
    pub swapchains: Vec<Pending>,
}

impl DestroyQueue {
    #[must_use]
    pub fn new() -> Self {
        Self {
            frames: VecDeque::with_capacity(MAX_FRAMES_IN_FLIGHT + 1),
            current: FrameBucket::default(),
        }
    }

    pub fn push_buffer(&mut self, p: Pending) {
        self.current.buffers.push(p);
    }
    pub fn push_texture(&mut self, p: Pending) {
        self.current.textures.push(p);
    }
    pub fn push_sampler(&mut self, p: Pending) {
        self.current.samplers.push(p);
    }
    pub fn push_shader(&mut self, p: Pending) {
        self.current.shaders.push(p);
    }
    pub fn push_bgl(&mut self, p: Pending) {
        self.current.bgls.push(p);
    }
    pub fn push_bg(&mut self, p: Pending) {
        self.current.bgs.push(p);
    }
    pub fn push_pipeline(&mut self, p: Pending) {
        self.current.pipelines.push(p);
    }
    pub fn push_swapchain(&mut self, p: Pending) {
        self.current.swapchains.push(p);
    }

    /// Rotate: stash the current bucket, return any bucket that has now
    /// aged past the in-flight window (caller should actually release its
    /// entries).
    pub fn rotate(&mut self) -> Option<FrameBucket> {
        let this_frame = std::mem::take(&mut self.current);
        self.frames.push_back(this_frame);
        if self.frames.len() > MAX_FRAMES_IN_FLIGHT {
            self.frames.pop_front()
        } else {
            None
        }
    }

    /// Flush everything immediately (called from `wait_idle`).
    pub fn drain_all(&mut self) -> Vec<FrameBucket> {
        let mut out: Vec<FrameBucket> = self.frames.drain(..).collect();
        out.push(std::mem::take(&mut self.current));
        out
    }
}
