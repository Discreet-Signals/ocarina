//! Ocarina language server.
//!
//! The same crate ships in two shapes:
//!
//! - **stdio server** — `ocarina lsp` runs [`run_stdio`], which wires
//!   the server to the process's stdin/stdout for use with VS Code,
//!   Zed, Helix, Neovim, etc.
//! - **in-process** — the editor (and Phase 3+ tooling) construct a
//!   [`Backend`] directly and call its analysis helpers without going
//!   through the LSP wire protocol.

mod analysis;
mod document;
mod schema;

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::{
    CompletionOptions, CompletionParams, CompletionResponse, DidChangeTextDocumentParams,
    DidCloseTextDocumentParams, DidOpenTextDocumentParams, GotoDefinitionParams,
    GotoDefinitionResponse, Hover, HoverParams, HoverProviderCapability, InitializeParams,
    InitializeResult, InitializedParams, MessageType, OneOf, ServerCapabilities, ServerInfo,
    TextDocumentItem, TextDocumentSyncCapability, TextDocumentSyncKind, Url,
};
use tower_lsp::{async_trait, Client, LanguageServer, LspService, Server};

use crate::document::Document;
use crate::schema::SchemaIndex;

pub use crate::analysis::{
    completions as analyze_completions, diagnostics as analyze_diagnostics,
    folder_conflict_diagnostics as analyze_folder_conflicts,
    goto_definition as analyze_goto_definition, hover as analyze_hover,
};
pub use crate::document::Document as LspDocument;
pub use crate::schema::SchemaIndex as LspSchemaIndex;

/// LSP backend. Owns the schema index and per-document state.
pub struct Backend {
    client: Client,
    schemas: Arc<SchemaIndex>,
    docs: RwLock<HashMap<Url, Document>>,
}

impl Backend {
    pub fn new(client: Client) -> Self {
        let schemas = SchemaIndex::from_inventory();
        log::info!(
            "ocarina-lsp: built from inventory with {} component(s)",
            schemas.len()
        );
        Self {
            client,
            schemas: Arc::new(schemas),
            docs: RwLock::default(),
        }
    }

    pub fn with_schema_json(client: Client, json: &str) -> serde_json::Result<Self> {
        let schemas = SchemaIndex::from_json(json)?;
        log::info!(
            "ocarina-lsp: loaded {} component schema(s) from JSON",
            schemas.len()
        );
        Ok(Self {
            client,
            schemas: Arc::new(schemas),
            docs: RwLock::default(),
        })
    }

    async fn publish_diagnostics(&self, uri: &Url, doc: &Document) {
        let mut diags = analysis::diagnostics(&self.schemas, doc);
        diags.extend(analysis::folder_conflict_diagnostics(
            &self.schemas,
            doc,
            uri,
        ));
        self.client
            .publish_diagnostics(uri.clone(), diags, None)
            .await;
    }
}

#[async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                definition_provider: Some(OneOf::Left(true)),
                completion_provider: Some(CompletionOptions {
                    trigger_characters: Some(vec!["[".into(), ".".into()]),
                    resolve_provider: Some(false),
                    ..Default::default()
                }),
                ..Default::default()
            },
            server_info: Some(ServerInfo {
                name: "ocarina-lsp".into(),
                version: Some(env!("CARGO_PKG_VERSION").into()),
            }),
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        let msg = format!(
            "ocarina-lsp ready · {} component(s) registered",
            self.schemas.len()
        );
        self.client.log_message(MessageType::INFO, msg).await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let TextDocumentItem { uri, text, .. } = params.text_document;
        let doc = Document::new(text);
        self.publish_diagnostics(&uri, &doc).await;
        self.docs.write().await.insert(uri, doc);
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri;
        let mut docs = self.docs.write().await;
        if let Some(doc) = docs.get_mut(&uri) {
            // FULL sync: pick the last change; the server-capabilities
            // we advertised request a single full document each time.
            if let Some(change) = params.content_changes.into_iter().last() {
                doc.replace(change.text);
            }
            self.publish_diagnostics(&uri, doc).await;
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let uri = params.text_document.uri;
        self.docs.write().await.remove(&uri);
        self.client.publish_diagnostics(uri, vec![], None).await;
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;
        let docs = self.docs.read().await;
        Ok(docs.get(&uri).and_then(|doc| analysis::hover(&self.schemas, doc, pos)))
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;
        let docs = self.docs.read().await;
        Ok(docs
            .get(&uri)
            .and_then(|doc| analysis::goto_definition(&self.schemas, doc, pos))
            .map(GotoDefinitionResponse::Scalar))
    }

    async fn completion(
        &self,
        params: CompletionParams,
    ) -> Result<Option<CompletionResponse>> {
        let uri = params.text_document_position.text_document.uri;
        let pos = params.text_document_position.position;
        let docs = self.docs.read().await;
        let items = docs
            .get(&uri)
            .map(|doc| analysis::completions(&self.schemas, doc, pos))
            .unwrap_or_default();
        Ok(Some(CompletionResponse::Array(items)))
    }
}

/// Run the server bound to stdin/stdout. Blocks until the client
/// disconnects (or sends `exit`).
pub async fn run_stdio() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let (service, socket) = LspService::new(Backend::new);
    Server::new(stdin, stdout, socket).serve(service).await;
}
