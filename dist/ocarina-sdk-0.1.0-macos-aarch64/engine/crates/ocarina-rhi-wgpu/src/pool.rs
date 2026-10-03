//! Generational resource pool.
//!
//! Each stored resource occupies a slot; slots carry a generation that is
//! incremented on free. A handle holds `(slot, generation)`; lookups validate
//! that the generation matches the pool's current value, so stale handles
//! are detectable rather than UB-aliasing a reused slot.

pub struct Pool<T> {
    entries: Vec<Option<Entry<T>>>,
    free: Vec<u32>,
}

struct Entry<T> {
    value: T,
    generation: u32,
}

impl<T> Default for Pool<T> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            free: Vec::new(),
        }
    }
}

#[allow(dead_code)] // some methods used only by future phases (get_mut for writes, iter_alive for diagnostics)
impl<T> Pool<T> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert `value` and return its `(slot, generation)`.
    pub fn insert(&mut self, value: T) -> (u32, u32) {
        if let Some(slot) = self.free.pop() {
            let idx = slot as usize;
            // Previous generation was even (= free); bump to make it odd (= alive).
            let current = self
                .entries
                .get(idx)
                .and_then(Option::as_ref)
                .map_or(0, |e| e.generation);
            let generation = current.wrapping_add(1);
            self.entries[idx] = Some(Entry { value, generation });
            (slot, generation)
        } else {
            let slot = self.entries.len() as u32;
            self.entries.push(Some(Entry {
                value,
                generation: 1,
            }));
            (slot, 1)
        }
    }

    /// Look up a resource by slot + generation.
    #[must_use]
    pub fn get(&self, slot: u32, generation: u32) -> Option<&T> {
        let entry = self.entries.get(slot as usize)?.as_ref()?;
        (entry.generation == generation).then_some(&entry.value)
    }

    /// Mutable lookup by slot + generation.
    #[must_use]
    pub fn get_mut(&mut self, slot: u32, generation: u32) -> Option<&mut T> {
        let entry = self.entries.get_mut(slot as usize)?.as_mut()?;
        (entry.generation == generation).then_some(&mut entry.value)
    }

    /// Remove a resource and return it, if the slot/generation match.
    pub fn remove(&mut self, slot: u32, generation: u32) -> Option<T> {
        let idx = slot as usize;
        let entry = self.entries.get_mut(idx)?;
        if entry.as_ref()?.generation != generation {
            return None;
        }
        let taken = entry.take()?;
        self.free.push(slot);
        Some(taken.value)
    }

    /// Remove the entry at `slot` regardless of generation. Useful for
    /// cleanup paths (like swapchain frame retire) where we know the slot
    /// is ours and don't need generation safety.
    pub fn remove_by_slot(&mut self, slot: u32) -> Option<T> {
        let idx = slot as usize;
        let entry = self.entries.get_mut(idx)?;
        let taken = entry.take()?;
        self.free.push(slot);
        Some(taken.value)
    }

    /// Iterate over all currently-alive entries (for diagnostics).
    pub fn iter_alive(&self) -> impl Iterator<Item = (u32, &T)> + '_ {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(slot, e)| e.as_ref().map(|entry| (slot as u32, &entry.value)))
    }

    #[must_use]
    pub fn alive_count(&self) -> usize {
        self.entries.iter().filter(|e| e.is_some()).count()
    }
}
