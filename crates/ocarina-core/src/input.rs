//! Per-frame input snapshot.
//!
//! Stored as a `bevy_ecs::Resource`. The host (editor in play mode, or
//! the shipped runtime's window event loop) populates it before each
//! tick via the methods on `Input`. Scripts read it through
//! `Context::input`.
//!
//! Snapshot semantics: `pressed` reports keys / buttons that are
//! currently held; `just_pressed` and `just_released` report changes
//! since the previous tick. The host calls `Input::end_frame()` after
//! the tick to roll the "just" sets forward.

use std::collections::HashSet;

use bevy_ecs::resource::Resource;
use glam::Vec2;

/// A keyboard key. Mirrors the subset of `winit::keyboard::KeyCode`
/// that gameplay code typically cares about. Hosts that integrate with
/// a different keymap source convert into this on the way in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
#[non_exhaustive]
pub enum Key {
    A, B, C, D, E, F, G, H, I, J, K, L, M,
    N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
    Up, Down, Left, Right,
    Space, Enter, Escape, Tab, Backspace,
    LShift, RShift, LCtrl, RCtrl, LAlt, RAlt,
    /// Catch-all for keys we haven't enumerated. Carries the platform's
    /// key code as a string (e.g. `"F1"`, `"BracketLeft"`) so scripts
    /// can still address them.
    Other(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    /// Aux button index (e.g. side buttons).
    Other(u16),
}

#[derive(Debug, Default, Resource)]
pub struct Input {
    keys_held: HashSet<Key>,
    keys_just_pressed: HashSet<Key>,
    keys_just_released: HashSet<Key>,

    mouse_held: HashSet<MouseButton>,
    mouse_just_pressed: HashSet<MouseButton>,
    mouse_just_released: HashSet<MouseButton>,

    mouse_pos: Vec2,
    mouse_delta: Vec2,
    scroll: Vec2,
}

impl Input {
    // ── Read API (used by scripts) ──

    #[must_use]
    pub fn key_held(&self, k: Key) -> bool {
        self.keys_held.contains(&k)
    }
    #[must_use]
    pub fn key_just_pressed(&self, k: Key) -> bool {
        self.keys_just_pressed.contains(&k)
    }
    #[must_use]
    pub fn key_just_released(&self, k: Key) -> bool {
        self.keys_just_released.contains(&k)
    }
    #[must_use]
    pub fn mouse_held(&self, b: MouseButton) -> bool {
        self.mouse_held.contains(&b)
    }
    #[must_use]
    pub fn mouse_just_pressed(&self, b: MouseButton) -> bool {
        self.mouse_just_pressed.contains(&b)
    }
    #[must_use]
    pub fn mouse_just_released(&self, b: MouseButton) -> bool {
        self.mouse_just_released.contains(&b)
    }
    #[must_use]
    pub fn mouse_pos(&self) -> Vec2 {
        self.mouse_pos
    }
    #[must_use]
    pub fn mouse_delta(&self) -> Vec2 {
        self.mouse_delta
    }
    /// Scroll delta accumulated this frame. Positive y = scroll up.
    #[must_use]
    pub fn scroll(&self) -> Vec2 {
        self.scroll
    }

    // ── Write API (used by the host before each tick) ──

    pub fn press_key(&mut self, k: Key) {
        if self.keys_held.insert(k) {
            self.keys_just_pressed.insert(k);
        }
    }
    pub fn release_key(&mut self, k: Key) {
        if self.keys_held.remove(&k) {
            self.keys_just_released.insert(k);
        }
    }
    pub fn press_mouse(&mut self, b: MouseButton) {
        if self.mouse_held.insert(b) {
            self.mouse_just_pressed.insert(b);
        }
    }
    pub fn release_mouse(&mut self, b: MouseButton) {
        if self.mouse_held.remove(&b) {
            self.mouse_just_released.insert(b);
        }
    }
    pub fn set_mouse_pos(&mut self, pos: Vec2) {
        self.mouse_delta = pos - self.mouse_pos;
        self.mouse_pos = pos;
    }
    pub fn add_scroll(&mut self, delta: Vec2) {
        self.scroll += delta;
    }

    /// Roll the "just pressed" / "just released" / scroll-accumulator
    /// state forward at the end of a frame. Held state and mouse_pos
    /// persist; deltas are zeroed.
    pub fn end_frame(&mut self) {
        self.keys_just_pressed.clear();
        self.keys_just_released.clear();
        self.mouse_just_pressed.clear();
        self.mouse_just_released.clear();
        self.mouse_delta = Vec2::ZERO;
        self.scroll = Vec2::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn just_pressed_clears_after_end_frame() {
        let mut input = Input::default();
        input.press_key(Key::Space);
        assert!(input.key_just_pressed(Key::Space));
        assert!(input.key_held(Key::Space));
        input.end_frame();
        assert!(!input.key_just_pressed(Key::Space));
        assert!(input.key_held(Key::Space));
    }

    #[test]
    fn release_records_just_released() {
        let mut input = Input::default();
        input.press_key(Key::A);
        input.end_frame();
        input.release_key(Key::A);
        assert!(input.key_just_released(Key::A));
        assert!(!input.key_held(Key::A));
    }

    #[test]
    fn mouse_delta_accumulates_then_clears() {
        let mut input = Input::default();
        input.set_mouse_pos(Vec2::new(10.0, 0.0));
        input.set_mouse_pos(Vec2::new(15.0, 0.0));
        assert_eq!(input.mouse_delta(), Vec2::new(5.0, 0.0));
        input.end_frame();
        assert_eq!(input.mouse_delta(), Vec2::ZERO);
        assert_eq!(input.mouse_pos(), Vec2::new(15.0, 0.0));
    }
}
