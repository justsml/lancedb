// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

//! Read-only HTTP search bindings for LanceDB tables.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

#[cfg(target_arch = "wasm32")]
use arrow_array::RecordBatch;
#[cfg(target_arch = "wasm32")]
use arrow_schema::Schema as ArrowSchema;
use async_trait::async_trait;
use lance_io::object_store::WrappingObjectStore;
use object_store::path::Path;
use object_store::{
    DynObjectStore, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore as OSObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use serde::{Deserialize, Serialize};
use url::Url;

#[cfg(not(target_arch = "wasm32"))]
mod native;

#[cfg(target_arch = "wasm32")]
mod fetch_object_store;
#[cfg(target_arch = "wasm32")]
use fetch_object_store::FetchHttpStore;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(any(target_arch = "wasm32", test))]
mod browser;
#[cfg(any(target_arch = "wasm32", test))]
mod browser_expr;
#[cfg(target_arch = "wasm32")]
use browser::BrowserTable;

#[cfg(not(target_arch = "wasm32"))]
use lancedb::{Error, Result, Table};
#[cfg(target_arch = "wasm32")]
use local_error::{Error, Result};

#[cfg(any(target_arch = "wasm32", test))]
mod local_error {
    use std::fmt::{Display, Formatter};

    use arrow_schema::ArrowError;
    use lance_core::Error as LanceError;
    #[cfg(not(target_arch = "wasm32"))]
    use lancedb::Error as NativeError;

    pub type BoxError = Box<dyn std::error::Error + Send + Sync>;
    pub type Result<T> = std::result::Result<T, Error>;

    #[derive(Debug)]
    pub enum Error {
        InvalidInput {
            message: String,
        },
        Runtime {
            message: String,
        },
        NotSupported {
            message: String,
        },
        ObjectStore {
            source: object_store::Error,
        },
        Lance {
            source: LanceError,
        },
        Arrow {
            source: ArrowError,
        },
        Other {
            message: String,
            source: Option<BoxError>,
        },
    }

    impl Display for Error {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::InvalidInput { message } => write!(f, "Invalid input, {message}"),
                Self::Runtime { message } => write!(f, "Runtime error: {message}"),
                Self::NotSupported { message } => {
                    write!(f, "LanceDBError: not supported: {message}")
                }
                Self::ObjectStore { source } => write!(f, "object_store error: {source}"),
                Self::Lance { source } => write!(f, "lance error: {source}"),
                Self::Arrow { source } => write!(f, "Arrow error: {source}"),
                Self::Other { message, .. } => write!(f, "{message}"),
            }
        }
    }

    impl std::error::Error for Error {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            match self {
                Self::ObjectStore { source } => Some(source),
                Self::Lance { source } => Some(source),
                Self::Arrow { source } => Some(source),
                Self::Other {
                    source: Some(source),
                    ..
                } => Some(source.as_ref()),
                _ => None,
            }
        }
    }

    impl From<object_store::Error> for Error {
        fn from(source: object_store::Error) -> Self {
            Self::ObjectStore { source }
        }
    }

    impl From<object_store::path::Error> for Error {
        fn from(source: object_store::path::Error) -> Self {
            Self::ObjectStore {
                source: object_store::Error::InvalidPath { source },
            }
        }
    }

    impl From<LanceError> for Error {
        fn from(source: LanceError) -> Self {
            Self::Lance { source }
        }
    }

    impl From<ArrowError> for Error {
        fn from(source: ArrowError) -> Self {
            Self::Arrow { source }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    impl From<NativeError> for Error {
        fn from(source: NativeError) -> Self {
            Self::Other {
                message: source.to_string(),
                source: Some(Box::new(source)),
            }
        }
    }
}

// Current Lance latest-version resolution relies on listing `_versions/`.
// For generic static HTTP hosting we instead resolve through a deterministic
// copy of the latest manifest at this path.
pub(crate) const MANIFEST_PATH: &str = "_latest.manifest";
const LATEST_VERSION_PATH: &str = "_latest.version";
const WEB_METADATA_PATH: &str = "_web.json";
const SNAPSHOT_PATH: &str = "_snapshot.json";

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenTableOptions {
    #[serde(default)]
    pub headers: HashMap<String, String>,
    pub cache_bytes: Option<usize>,
    pub max_concurrent_ranges: Option<usize>,
    pub manifest_url: Option<String>,
    pub latest_version_url: Option<String>,
    pub web_metadata_url: Option<String>,
    pub snapshot_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchRequest {
    pub vector: Option<Vec<f32>>,
    pub text: Option<TextRequest>,
    pub distance_type: Option<SearchDistanceType>,
    pub filter: Option<String>,
    pub select: Option<SelectRequest>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub vector_column: Option<String>,
    pub prefilter: Option<bool>,
    pub with_row_id: Option<bool>,
    pub fast_search: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TextRequest {
    Query(String),
    Structured {
        query: String,
        columns: Option<Vec<String>>,
    },
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SearchDistanceType {
    #[default]
    L2,
    Cosine,
    Dot,
    Hamming,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SelectRequest {
    Columns(Vec<String>),
    Dynamic(BTreeMap<String, String>),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PublishedTableMetadata {
    version: u64,
    manifest_path: String,
    manifest_size_bytes: Option<u64>,
    manifest_naming_scheme: String,
    latest_manifest_path: String,
    latest_version_path: String,
    web_metadata_path: String,
    snapshot_path: String,
    default_vector_column: Option<String>,
    #[serde(default)]
    vector_columns: Vec<String>,
    #[serde(default)]
    fts_columns: Vec<String>,
    /// Arbitrary user-defined key-value metadata (e.g. embedding model, LLM URI).
    #[serde(default)]
    metadata: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PublishedSnapshot {
    version: u64,
    manifest_path: String,
    manifest_size_bytes: Option<u64>,
    manifest_naming_scheme: String,
    latest_manifest_path: String,
    latest_version_path: String,
    web_metadata_path: String,
    snapshot_path: String,
    default_vector_column: Option<String>,
    #[serde(default)]
    vector_columns: Vec<String>,
    #[serde(default)]
    fts_columns: Vec<String>,
    is_complete: bool,
    /// Arbitrary user-defined key-value metadata (e.g. embedding model, LLM URI).
    #[serde(default)]
    metadata: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub(crate) struct ResolvedPublishedState {
    pub(crate) current_version: Option<u64>,
    pub(crate) latest_version_url: String,
    pub(crate) manifest_url: String,
    pub(crate) snapshot: Option<PublishedSnapshot>,
    pub(crate) table_metadata: Option<PublishedTableMetadata>,
}

#[derive(Clone)]
pub struct RemoteSearchTable {
    table_url: String,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    table_name: String,
    options: OpenTableOptions,
    published: ResolvedPublishedState,
    #[cfg(not(target_arch = "wasm32"))]
    session: Arc<lance::session::Session>,
    #[cfg(not(target_arch = "wasm32"))]
    table: Option<Table>,
    #[cfg(target_arch = "wasm32")]
    table: Option<BrowserTable>,
}

impl RemoteSearchTable {
    pub async fn open(table_url: &str, options: OpenTableOptions) -> Result<Self> {
        let table_url = normalize_table_url(table_url)?;
        let table_name = table_name_from_url(&table_url)?;
        let published = resolve_published_state(&table_url, &options).await?;
        #[cfg(not(target_arch = "wasm32"))]
        let session = Arc::new(native::session_from_cache_bytes(options.cache_bytes));
        #[cfg(not(target_arch = "wasm32"))]
        let opened = native::open_table_with_options(
            &table_url,
            &table_name,
            &options,
            &published,
            session.clone(),
        )
        .await?;
        #[cfg(target_arch = "wasm32")]
        let opened = BrowserTable::open(&table_url, &options, &published.manifest_url).await?;

        Ok(Self {
            table_url,
            table_name,
            options,
            published,
            #[cfg(not(target_arch = "wasm32"))]
            session,
            #[cfg(not(target_arch = "wasm32"))]
            table: Some(opened),
            #[cfg(target_arch = "wasm32")]
            table: Some(opened),
        })
    }

    pub async fn schema(&self) -> Result<Vec<u8>> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let table = self.table_ref()?;
            let schema = table.schema().await?;
            lancedb::ipc::schema_to_ipc_file(schema.as_ref())
        }

        #[cfg(target_arch = "wasm32")]
        {
            let table = self.browser_table_ref()?;
            schema_to_ipc_file(&ArrowSchema::from(table.schema()))
        }
    }

    pub async fn search(&self, request: SearchRequest) -> Result<Vec<u8>> {
        let request = apply_published_request_defaults(request, &self.published)?;
        #[cfg(not(target_arch = "wasm32"))]
        {
            let table = self.table_ref()?;
            native::execute_table_search(table, request).await
        }

        #[cfg(target_arch = "wasm32")]
        {
            let table = self.browser_table_ref()?;
            execute_browser_search(table, request).await
        }
    }

    pub async fn search_json(&self, request_json: &str) -> Result<Vec<u8>> {
        let request: SearchRequest =
            serde_json::from_str(request_json).map_err(|source| Error::InvalidInput {
                message: format!("invalid search request: {source}"),
            })?;
        self.search(request).await
    }

    pub async fn refresh(&mut self) -> Result<bool> {
        if let (Some(current_version), latest_version_url) = (
            self.published.current_version,
            self.published.latest_version_url.as_str(),
        ) {
            let latest_version = fetch_latest_version(latest_version_url, &self.options).await?;
            if latest_version == current_version {
                return Ok(false);
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        let current_version = self.table_ref()?.version().await?;
        #[cfg(target_arch = "wasm32")]
        let current_version = self.browser_table_ref()?.version();
        let published = resolve_published_state(&self.table_url, &self.options).await?;
        #[cfg(not(target_arch = "wasm32"))]
        let reopened = native::open_table_with_options(
            &self.table_url,
            &self.table_name,
            &self.options,
            &published,
            self.session.clone(),
        )
        .await?;
        #[cfg(target_arch = "wasm32")]
        let reopened =
            BrowserTable::open(&self.table_url, &self.options, &published.manifest_url).await?;
        #[cfg(not(target_arch = "wasm32"))]
        let next_version = reopened.version().await?;
        #[cfg(target_arch = "wasm32")]
        let next_version = reopened.version();
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.table = Some(reopened);
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.table = Some(reopened);
        }
        self.published = published;
        Ok(next_version != current_version)
    }

    pub fn close(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.table = None;
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.table = None;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn table_ref(&self) -> Result<&Table> {
        self.table.as_ref().ok_or_else(|| Error::Runtime {
            message: "table is closed".to_string(),
        })
    }

    #[cfg(target_arch = "wasm32")]
    fn browser_table_ref(&self) -> Result<&BrowserTable> {
        self.table.as_ref().ok_or_else(|| Error::Runtime {
            message: "table is closed".to_string(),
        })
    }
}

fn normalize_table_url(table_url: &str) -> Result<String> {
    let mut parsed = Url::parse(table_url).map_err(|source| Error::InvalidInput {
        message: format!("invalid table URL '{table_url}': {source}"),
    })?;

    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(Error::InvalidInput {
            message: format!(
                "unsupported table URL scheme '{}'; expected http or https",
                parsed.scheme()
            ),
        });
    }

    if !parsed.path().ends_with('/') {
        let next_path = format!("{}/", parsed.path());
        parsed.set_path(&next_path);
    }

    Ok(parsed.to_string())
}

fn table_name_from_url(table_url: &str) -> Result<String> {
    let parsed = Url::parse(table_url).map_err(|source| Error::InvalidInput {
        message: format!("invalid table URL '{table_url}': {source}"),
    })?;
    let last_segment = parsed
        .path_segments()
        .and_then(|segments| segments.filter(|segment| !segment.is_empty()).next_back())
        .ok_or_else(|| Error::InvalidInput {
            message: format!("table URL '{table_url}' does not contain a table path"),
        })?;

    let table_name = std::path::Path::new(last_segment)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| Error::InvalidInput {
            message: format!("unable to derive a table name from URL '{table_url}'"),
        })?;

    Ok(table_name.to_string())
}

async fn resolve_published_state(
    table_url: &str,
    options: &OpenTableOptions,
) -> Result<ResolvedPublishedState> {
    let table_url = Url::parse(table_url).map_err(|source| Error::InvalidInput {
        message: format!("invalid table URL '{table_url}': {source}"),
    })?;
    let web_metadata_url = options
        .web_metadata_url
        .clone()
        .unwrap_or_else(|| resolve_url(&table_url, WEB_METADATA_PATH));
    let table_metadata =
        fetch_optional_json::<PublishedTableMetadata>(&web_metadata_url, &options.headers).await?;

    let snapshot_url = options.snapshot_url.clone().unwrap_or_else(|| {
        resolve_url(
            &table_url,
            table_metadata
                .as_ref()
                .map(|metadata| metadata.snapshot_path.as_str())
                .unwrap_or(SNAPSHOT_PATH),
        )
    });
    let snapshot = if table_metadata.is_none() {
        fetch_optional_json::<PublishedSnapshot>(&snapshot_url, &options.headers).await?
    } else {
        None
    };

    Ok(ResolvedPublishedState {
        current_version: table_metadata
            .as_ref()
            .map(|metadata| metadata.version)
            .or_else(|| snapshot.as_ref().map(|snapshot| snapshot.version)),
        latest_version_url: options.latest_version_url.clone().unwrap_or_else(|| {
            resolve_url(
                &table_url,
                table_metadata
                    .as_ref()
                    .map(|metadata| metadata.latest_version_path.as_str())
                    .or_else(|| {
                        snapshot.as_ref().map(|published_snapshot| {
                            published_snapshot.latest_version_path.as_str()
                        })
                    })
                    .unwrap_or(LATEST_VERSION_PATH),
            )
        }),
        manifest_url: options.manifest_url.clone().unwrap_or_else(|| {
            resolve_url(
                &table_url,
                table_metadata
                    .as_ref()
                    .map(|metadata| metadata.latest_manifest_path.as_str())
                    .or_else(|| {
                        snapshot.as_ref().map(|published_snapshot| {
                            published_snapshot.latest_manifest_path.as_str()
                        })
                    })
                    .unwrap_or(MANIFEST_PATH),
            )
        }),
        snapshot,
        table_metadata,
    })
}

async fn fetch_optional_json<T>(url: &str, headers: &HashMap<String, String>) -> Result<Option<T>>
where
    T: for<'de> Deserialize<'de>,
{
    let bytes = match fetch_optional_bytes(url, headers).await? {
        Some(bytes) => bytes,
        None => return Ok(None),
    };
    let parsed = serde_json::from_slice::<T>(&bytes).map_err(|source| Error::InvalidInput {
        message: format!("invalid published metadata at '{url}': {source}"),
    })?;
    Ok(Some(parsed))
}

async fn fetch_optional_bytes(
    url: &str,
    headers: &HashMap<String, String>,
) -> Result<Option<Vec<u8>>> {
    let url = Url::parse(url).map_err(|source| Error::InvalidInput {
        message: format!("invalid sidecar URL '{url}': {source}"),
    })?;
    let store = build_store(&url, headers)?;
    match store.get(&Path::default()).await {
        Ok(result) => Ok(Some(result.bytes().await?.to_vec())),
        Err(object_store::Error::NotFound { .. }) => Ok(None),
        Err(source) => Err(source.into()),
    }
}

async fn fetch_latest_version(url: &str, options: &OpenTableOptions) -> Result<u64> {
    let bytes = fetch_optional_bytes(url, &options.headers)
        .await?
        .ok_or_else(|| Error::Runtime {
            message: format!("latest version sidecar '{url}' was not found"),
        })?;
    let raw = String::from_utf8(bytes).map_err(|source| Error::InvalidInput {
        message: format!("latest version sidecar '{url}' is not valid UTF-8: {source}"),
    })?;
    raw.trim()
        .parse::<u64>()
        .map_err(|source| Error::InvalidInput {
            message: format!(
                "latest version sidecar '{url}' contained an invalid version: {source}"
            ),
        })
}

fn resolve_url(base_url: &Url, path_or_url: &str) -> String {
    Url::parse(path_or_url)
        .or_else(|_| base_url.join(path_or_url))
        .expect("joining published sidecar URL should succeed")
        .to_string()
}

fn apply_published_request_defaults(
    mut request: SearchRequest,
    published: &ResolvedPublishedState,
) -> Result<SearchRequest> {
    let published_metadata = published
        .table_metadata
        .as_ref()
        .map(PublishedMetadataView::Table)
        .or_else(|| {
            published
                .snapshot
                .as_ref()
                .map(PublishedMetadataView::Snapshot)
        });

    if let Some(text_request) = request.text.take() {
        request.text = Some(apply_published_text_defaults(
            text_request,
            published_metadata,
        )?);
    }

    if request.vector.is_some() && request.vector_column.is_none() {
        request.vector_column = published_metadata
            .and_then(|metadata| metadata.default_vector_column().map(ToOwned::to_owned));
    }

    Ok(request)
}

fn apply_published_text_defaults(
    request: TextRequest,
    published_metadata: Option<PublishedMetadataView<'_>>,
) -> Result<TextRequest> {
    let Some(published_metadata) = published_metadata else {
        return Ok(request);
    };

    let advertised_columns = published_metadata.fts_columns();
    if advertised_columns.is_empty() {
        return Ok(request);
    }

    let (query, columns) = match request {
        TextRequest::Query(query) => (query, advertised_columns.to_vec()),
        TextRequest::Structured { query, columns } => (
            query,
            columns.unwrap_or_else(|| advertised_columns.to_vec()),
        ),
    };

    for column in &columns {
        if !advertised_columns
            .iter()
            .any(|candidate| candidate == column)
        {
            return Err(Error::InvalidInput {
                message: format!(
                    "text search column '{column}' is not advertised in the published FTS metadata"
                ),
            });
        }
    }

    Ok(TextRequest::Structured {
        query,
        columns: Some(columns),
    })
}

#[derive(Clone, Copy)]
enum PublishedMetadataView<'a> {
    Table(&'a PublishedTableMetadata),
    Snapshot(&'a PublishedSnapshot),
}

impl PublishedMetadataView<'_> {
    fn default_vector_column(&self) -> Option<&str> {
        match self {
            Self::Table(metadata) => metadata.default_vector_column.as_deref(),
            Self::Snapshot(snapshot) => snapshot.default_vector_column.as_deref(),
        }
    }

    fn fts_columns(&self) -> &[String] {
        match self {
            Self::Table(metadata) => metadata.fts_columns.as_slice(),
            Self::Snapshot(snapshot) => snapshot.fts_columns.as_slice(),
        }
    }
}

pub(crate) fn build_open_store(
    table_url: &Url,
    table_path: &Path,
    manifest_url: &str,
    options: &OpenTableOptions,
) -> Result<(Arc<DynObjectStore>, Option<Arc<dyn WrappingObjectStore>>)> {
    let http_store = build_store(&object_store_root_url(table_url), &options.headers)?;

    let wrapper = options
        .manifest_url
        .as_deref()
        .or(Some(manifest_url))
        .filter(|resolved_manifest_url| {
            *resolved_manifest_url != resolve_url(table_url, MANIFEST_PATH).as_str()
        })
        .map(|resolved_manifest_url| {
            build_manifest_wrapper(
                table_url,
                table_path.child(MANIFEST_PATH),
                resolved_manifest_url,
                &options.headers,
            )
        })
        .transpose()?
        .map(|wrapper| Arc::new(wrapper) as Arc<dyn WrappingObjectStore>);

    Ok((http_store as Arc<DynObjectStore>, wrapper))
}

fn build_store(url: &Url, headers: &HashMap<String, String>) -> Result<Arc<DynObjectStore>> {
    #[cfg(target_arch = "wasm32")]
    {
        Ok(Arc::new(FetchHttpStore::new(url.clone(), headers.clone())) as Arc<DynObjectStore>)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        Ok(Arc::new(native::build_http_store(url, headers)?) as Arc<DynObjectStore>)
    }
}

fn build_manifest_wrapper(
    table_url: &Url,
    manifest_location: Path,
    manifest_url: &str,
    headers: &HashMap<String, String>,
) -> Result<ManifestRedirectWrapper> {
    let manifest_url = Url::parse(manifest_url).map_err(|source| Error::InvalidInput {
        message: format!("invalid manifest URL '{manifest_url}': {source}"),
    })?;

    if !matches!(manifest_url.scheme(), "http" | "https") {
        return Err(Error::InvalidInput {
            message: format!(
                "unsupported manifest URL scheme '{}'; expected http or https",
                manifest_url.scheme()
            ),
        });
    }

    let manifest_store = build_store(&manifest_url, headers)?;
    Ok(ManifestRedirectWrapper {
        table_url: table_url.to_string(),
        manifest_location,
        manifest_store,
    })
}

pub(crate) async fn preflight_manifest_check(
    store: Arc<DynObjectStore>,
    manifest_path: &Path,
) -> Result<()> {
    store
        .get_opts(
            manifest_path,
            GetOptions {
                range: Some((0_u64..1_u64).into()),
                ..Default::default()
            },
        )
        .await?
        .bytes()
        .await?;
    Ok(())
}

fn object_store_root_url(table_url: &Url) -> Url {
    let mut root = table_url.clone();
    root.set_path("/");
    root.set_query(None);
    root.set_fragment(None);
    root
}

pub(crate) fn object_store_table_path(table_url: &Url) -> Result<Path> {
    Path::from_url_path(table_url.path()).map_err(Error::from)
}

#[cfg(target_arch = "wasm32")]
async fn execute_browser_search(table: &BrowserTable, request: SearchRequest) -> Result<Vec<u8>> {
    let batches = table.search_batches(request).await?;
    batches_to_ipc_file(&batches)
}

#[cfg(target_arch = "wasm32")]
fn batches_to_ipc_file(batches: &[RecordBatch]) -> Result<Vec<u8>> {
    use arrow_ipc::writer::FileWriter;

    if batches.is_empty() {
        return Err(Error::Other {
            message: "No batches to write".to_string(),
            source: None,
        });
    }
    let schema = batches[0].schema();
    let mut writer = FileWriter::try_new(vec![], &schema)?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.finish()?;
    Ok(writer.into_inner()?)
}

#[cfg(target_arch = "wasm32")]
fn schema_to_ipc_file(schema: &ArrowSchema) -> Result<Vec<u8>> {
    use arrow_ipc::writer::FileWriter;

    let mut writer = FileWriter::try_new(vec![], schema)?;
    writer.finish()?;
    Ok(writer.into_inner()?)
}

#[derive(Debug)]
struct ManifestRedirectWrapper {
    table_url: String,
    manifest_location: Path,
    manifest_store: Arc<DynObjectStore>,
}

impl WrappingObjectStore for ManifestRedirectWrapper {
    fn wrap(
        &self,
        _store_prefix: &str,
        original: Arc<dyn OSObjectStore>,
    ) -> Arc<dyn OSObjectStore> {
        Arc::new(ManifestRedirectStore {
            original,
            manifest_store: self.manifest_store.clone(),
            manifest_location: self.manifest_location.clone(),
            table_url: self.table_url.clone(),
        })
    }
}

#[derive(Debug)]
struct ManifestRedirectStore {
    original: Arc<DynObjectStore>,
    manifest_store: Arc<DynObjectStore>,
    manifest_location: Path,
    table_url: String,
}

impl std::fmt::Display for ManifestRedirectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ManifestRedirectStore({})", self.table_url)
    }
}

impl ManifestRedirectStore {
    fn route<'a>(&'a self, location: &'a Path) -> (&'a Arc<DynObjectStore>, Cow<'a, Path>) {
        if location == &self.manifest_location {
            (&self.manifest_store, Cow::Owned(Path::default()))
        } else {
            (&self.original, Cow::Borrowed(location))
        }
    }
}

#[async_trait]
impl OSObjectStore for ManifestRedirectStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.original.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.original.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        let (store, mapped) = self.route(location);
        store.get_opts(mapped.as_ref(), options).await
    }

    async fn delete(&self, location: &Path) -> object_store::Result<()> {
        self.original.delete(location).await
    }

    fn list(
        &self,
        prefix: Option<&Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.original.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        self.original.list_with_delimiter(prefix).await
    }

    async fn copy(&self, from: &Path, to: &Path) -> object_store::Result<()> {
        self.original.copy(from, to).await
    }

    async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> object_store::Result<()> {
        self.original.copy_if_not_exists(from, to).await
    }
}

#[cfg(target_arch = "wasm32")]
fn wasm_err(source: Error) -> JsError {
    JsError::new(&source.to_string())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmRemoteSearchTable {
    inner: RemoteSearchTable,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmRemoteSearchTable {
    pub async fn schema(&self) -> std::result::Result<Vec<u8>, JsError> {
        self.inner.schema().await.map_err(wasm_err)
    }

    pub async fn search(&self, request_json: String) -> std::result::Result<Vec<u8>, JsError> {
        self.inner
            .search_json(&request_json)
            .await
            .map_err(wasm_err)
    }

    pub async fn refresh(&mut self) -> std::result::Result<bool, JsError> {
        self.inner.refresh().await.map_err(wasm_err)
    }

    pub fn close(&mut self) {
        self.inner.close();
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = open_table)]
pub async fn open_table(
    table_url: String,
    options_json: Option<String>,
) -> std::result::Result<WasmRemoteSearchTable, JsError> {
    let options = options_json
        .as_deref()
        .map(|json| {
            serde_json::from_str::<OpenTableOptions>(json).map_err(|source| Error::InvalidInput {
                message: format!("invalid open options: {source}"),
            })
        })
        .transpose()
        .map_err(wasm_err)?
        .unwrap_or_default();

    let inner = RemoteSearchTable::open(&table_url, options)
        .await
        .map_err(wasm_err)?;
    Ok(WasmRemoteSearchTable { inner })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;

    use super::*;

    fn empty_request() -> SearchRequest {
        SearchRequest {
            vector: None,
            text: None,
            distance_type: None,
            filter: None,
            select: None,
            limit: None,
            offset: None,
            vector_column: None,
            prefilter: None,
            with_row_id: None,
            fast_search: None,
        }
    }

    fn published_state(
        default_vector_column: Option<&str>,
        fts_columns: Vec<&str>,
    ) -> ResolvedPublishedState {
        let default_vector_column = default_vector_column.map(ToOwned::to_owned);
        let vector_columns = default_vector_column.iter().cloned().collect();

        ResolvedPublishedState {
            current_version: Some(7),
            latest_version_url: LATEST_VERSION_PATH.to_string(),
            manifest_url: MANIFEST_PATH.to_string(),
            snapshot: None,
            table_metadata: Some(PublishedTableMetadata {
                version: 7,
                manifest_path: "_versions/7.manifest".to_string(),
                manifest_size_bytes: Some(1024),
                manifest_naming_scheme: "v2".to_string(),
                latest_manifest_path: MANIFEST_PATH.to_string(),
                latest_version_path: LATEST_VERSION_PATH.to_string(),
                web_metadata_path: WEB_METADATA_PATH.to_string(),
                snapshot_path: SNAPSHOT_PATH.to_string(),
                default_vector_column,
                vector_columns,
                fts_columns: fts_columns.into_iter().map(ToOwned::to_owned).collect(),
                metadata: HashMap::new(),
            }),
        }
    }

    #[test]
    fn published_text_queries_pass_through_when_no_fts_columns_are_advertised() {
        let mut request = empty_request();
        request.text = Some(TextRequest::Query("apple".to_string()));

        let request = apply_published_request_defaults(request, &published_state(None, vec![]))
            .expect("published defaults should not reject pass-through text queries");

        match request.text {
            Some(TextRequest::Query(query)) => assert_eq!(query, "apple"),
            other => panic!("expected query text request to remain unchanged, got {other:?}"),
        }
    }

    #[test]
    fn published_text_defaults_fill_advertised_columns() {
        let mut request = empty_request();
        request.text = Some(TextRequest::Query("apple".to_string()));

        let request =
            apply_published_request_defaults(request, &published_state(None, vec!["doc"]))
                .expect("published defaults should inject advertised FTS columns");

        match request.text {
            Some(TextRequest::Structured { query, columns }) => {
                assert_eq!(query, "apple");
                assert_eq!(columns, Some(vec!["doc".to_string()]));
            }
            other => panic!("expected structured text request, got {other:?}"),
        }
    }

    #[test]
    fn published_text_defaults_still_validate_explicit_columns() {
        let mut request = empty_request();
        request.text = Some(TextRequest::Structured {
            query: "apple".to_string(),
            columns: Some(vec!["title".to_string()]),
        });

        let error = apply_published_request_defaults(request, &published_state(None, vec!["doc"]))
            .expect_err("published defaults should reject unadvertised text columns");

        match error {
            Error::InvalidInput { message } => {
                assert!(message.contains("not advertised"));
                assert!(message.contains("title"));
            }
            other => panic!("expected invalid input error, got {other:?}"),
        }
    }

    #[test]
    fn published_metadata_deserializes_user_metadata_map() {
        let metadata: PublishedTableMetadata = serde_json::from_value(json!({
            "version": 7,
            "manifestPath": "_versions/7.manifest",
            "manifestSizeBytes": 1024,
            "manifestNamingScheme": "v2",
            "latestManifestPath": "_latest.manifest",
            "latestVersionPath": "_latest.version",
            "webMetadataPath": "_web.json",
            "snapshotPath": "_snapshot.json",
            "vectorColumns": ["embedding"],
            "ftsColumns": ["doc"],
            "defaultVectorColumn": "embedding",
            "metadata": {
                "embedding_model": "text-embedding-3-large",
                "llm_uri": "openai://responses"
            }
        }))
        .expect("published metadata JSON should deserialize");

        assert_eq!(
            metadata.metadata.get("embedding_model"),
            Some(&"text-embedding-3-large".to_string())
        );
        assert_eq!(
            metadata.metadata.get("llm_uri"),
            Some(&"openai://responses".to_string())
        );
    }
}
