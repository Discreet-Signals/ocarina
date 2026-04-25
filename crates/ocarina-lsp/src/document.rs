//! Per-document state. We use FULL text-document sync (the editor sends
//! the whole file on each change), so the model is just "string + cached
//! parse." Re-parsing is O(file size) — fine for component files which
//! are typically <1 KB.

#[derive(Debug)]
pub struct Document {
    pub text: String,
    /// Successfully parsed root table, if the document is valid TOML.
    pub parsed: Option<toml::Value>,
    /// The most recent parse error, if any.
    pub parse_error: Option<toml::de::Error>,
}

impl Document {
    pub fn new(text: String) -> Self {
        let mut doc = Self {
            text,
            parsed: None,
            parse_error: None,
        };
        doc.reparse();
        doc
    }

    pub fn replace(&mut self, text: String) {
        self.text = text;
        self.reparse();
    }

    fn reparse(&mut self) {
        match toml::from_str::<toml::Value>(&self.text) {
            Ok(v) => {
                self.parsed = Some(v);
                self.parse_error = None;
            }
            Err(e) => {
                self.parsed = None;
                self.parse_error = Some(e);
            }
        }
    }

    /// Get line `n` (0-indexed) as a string slice.
    pub fn line(&self, n: usize) -> Option<&str> {
        self.text.lines().nth(n)
    }
}
