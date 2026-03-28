// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

//! Read-only HTTP search bindings for LanceDB tables.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use async_trait::async_trait;
use futures::TryStreamExt;
use http::header::HeaderName;
use lance::dataset::ReadParams;
use lance::session::Session;
use lance_index::scalar::FullTextSearchQuery;
use lance_io::object_store::WrappingObjectStore;
use lance_table::format::{IndexMetadata, Manifest, Transaction};
use lance_table::io::commit::{
    CommitError, CommitHandler, ManifestLocation, ManifestNamingScheme, ManifestWriter,
};
use lance_table::io::manifest::read_manifest;
use lancedb::ipc::{batches_to_ipc_file, schema_to_ipc_file};
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::table::{BaseTable, NativeTable};
use lancedb::{Error, Result, Table};
use object_store::http::HttpBuilder;
use object_store::path::Path;
use object_store::{
    DynObjectStore, GetOptions, GetResult, HeaderMap, HeaderValue, ListResult, MultipartUpload,
    ObjectMeta, ObjectStore as OSObjectStore, PutMultipartOptions, PutOptions, PutPayload,
    PutResult,
};
use serde::{Deserialize, Serialize};
use url::Url;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// Current Lance latest-version resolution relies on listing `_versions/`.
// For generic static HTTP hosting we instead resolve through a deterministic
// copy of the latest manifest at this path.
const MANIFEST_PATH: &str = "_latest.manifest";

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenTableOptions {
    #[serde(default)]
    pub headers: HashMap<String, String>,
    pub cache_bytes: Option<usize>,
    pub max_concurrent_ranges: Option<usize>,
    pub manifest_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchRequest {
    pub vector: Option<Vec<f32>>,
    pub text: Option<TextRequest>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SelectRequest {
    Columns(Vec<String>),
    Dynamic(BTreeMap<String, String>),
}

#[derive(Clone)]
pub struct RemoteSearchTable {
    table_url: String,
    table_name: String,
    options: OpenTableOptions,
    session: Arc<Session>,
    table: Option<Table>,
}

impl RemoteSearchTable {
    pub async fn open(table_url: &str, options: OpenTableOptions) -> Result<Self> {
        let table_url = normalize_table_url(table_url)?;
        let table_name = table_name_from_url(&table_url)?;
        let session = Arc::new(session_from_cache_bytes(options.cache_bytes));
        let table =
            open_table_with_options(&table_url, &table_name, &options, session.clone()).await?;

        Ok(Self {
            table_url,
            table_name,
            options,
            session,
            table: Some(table),
        })
    }

    pub async fn schema(&self) -> Result<Vec<u8>> {
        let table = self.table_ref()?;
        let schema = table.schema().await?;
        schema_to_ipc_file(schema.as_ref())
    }

    pub async fn search(&self, request: SearchRequest) -> Result<Vec<u8>> {
        let table = self.table_ref()?;
        match (request.vector.clone(), request.text.clone()) {
            (Some(vector), text) => {
                let mut query = table.query().nearest_to(vector)?;
                if let Some(text) = text {
                    query = query.full_text_search(build_fts_query(text)?);
                }
                query = apply_common_query(query, &request);
                if let Some(vector_column) = request.vector_column.as_deref() {
                    query = query.column(vector_column);
                }
                execute_query(query).await
            }
            (None, text) => {
                let mut query = table.query();
                if let Some(text) = text {
                    query = query.full_text_search(build_fts_query(text)?);
                }
                query = apply_common_query(query, &request);
                execute_query(query).await
            }
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
        let current_version = self.table_ref()?.version().await?;
        let reopened = open_table_with_options(
            &self.table_url,
            &self.table_name,
            &self.options,
            self.session.clone(),
        )
        .await?;
        let next_version = reopened.version().await?;
        self.table = Some(reopened);
        Ok(next_version != current_version)
    }

    pub fn close(&mut self) {
        self.table = None;
    }

    fn table_ref(&self) -> Result<&Table> {
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

fn session_from_cache_bytes(cache_bytes: Option<usize>) -> Session {
    match cache_bytes {
        Some(cache_bytes) => {
            let metadata_cache_size = cache_bytes / 4;
            let index_cache_size = cache_bytes.saturating_sub(metadata_cache_size);
            Session::new(index_cache_size, metadata_cache_size, Default::default())
        }
        None => Session::default(),
    }
}

async fn open_table_with_options(
    table_url: &str,
    table_name: &str,
    options: &OpenTableOptions,
    session: Arc<Session>,
) -> Result<Table> {
    let parsed_url = Url::parse(table_url).map_err(|source| Error::InvalidInput {
        message: format!("invalid table URL '{table_url}': {source}"),
    })?;
    let table_path = object_store_table_path(&parsed_url)?;

    let (http_store, wrapper) = build_open_store(&parsed_url, &table_path, options)?;
    let preflight_store = wrapper
        .as_ref()
        .map(|wrapper| wrapper.wrap("lancedb-wasm-preflight", http_store.clone()))
        .unwrap_or_else(|| http_store.clone());
    preflight_manifest_check(preflight_store, &table_path.child(MANIFEST_PATH)).await?;

    let read_params = ReadParams {
        session: Some(session),
        commit_handler: Some(Arc::new(LatestManifestCommitHandler)),
        store_options: Some(lance::io::ObjectStoreParams {
            object_store: Some((http_store, parsed_url)),
            object_store_wrapper: wrapper,
            ..Default::default()
        }),
        ..Default::default()
    };

    let native = NativeTable::open_with_params(
        table_url,
        table_name,
        vec![],
        None,
        Some(read_params),
        None,
        None,
        false,
        None,
    )
    .await?;

    Ok(Table::from(Arc::new(native) as Arc<dyn BaseTable>))
}

fn build_open_store(
    table_url: &Url,
    table_path: &Path,
    options: &OpenTableOptions,
) -> Result<(Arc<DynObjectStore>, Option<Arc<dyn WrappingObjectStore>>)> {
    let http_store = Arc::new(build_http_store(
        &object_store_root_url(table_url),
        &options.headers,
    )?);

    let wrapper = options
        .manifest_url
        .as_deref()
        .map(|manifest_url| {
            build_manifest_wrapper(
                table_url,
                table_path.child(MANIFEST_PATH),
                manifest_url,
                &options.headers,
            )
        })
        .transpose()?
        .map(|wrapper| Arc::new(wrapper) as Arc<dyn WrappingObjectStore>);

    Ok((http_store as Arc<DynObjectStore>, wrapper))
}

fn build_http_store(
    url: &Url,
    headers: &HashMap<String, String>,
) -> Result<object_store::http::HttpStore> {
    let mut client_options = object_store::ClientOptions::new();
    if url.scheme() == "http" {
        client_options = client_options.with_allow_http(true);
    }

    if !headers.is_empty() {
        let mut default_headers = HeaderMap::new();
        for (key, value) in headers {
            let name = key
                .parse::<HeaderName>()
                .map_err(|source| Error::InvalidInput {
                    message: format!("invalid header name '{key}': {source}"),
                })?;
            let value = HeaderValue::from_str(value).map_err(|source| Error::InvalidInput {
                message: format!("invalid header value for '{key}': {source}"),
            })?;
            default_headers.insert(name, value);
        }
        client_options = client_options.with_default_headers(default_headers);
    }

    HttpBuilder::new()
        .with_url(url.to_string())
        .with_client_options(client_options)
        .build()
        .map_err(Error::from)
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

    let manifest_store = Arc::new(build_http_store(&manifest_url, headers)?);
    Ok(ManifestRedirectWrapper {
        table_url: table_url.to_string(),
        manifest_location,
        manifest_store,
    })
}

async fn preflight_manifest_check(store: Arc<DynObjectStore>, manifest_path: &Path) -> Result<()> {
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

#[derive(Debug)]
struct LatestManifestCommitHandler;

#[async_trait]
impl CommitHandler for LatestManifestCommitHandler {
    async fn resolve_latest_location(
        &self,
        base_path: &Path,
        object_store: &lance_io::object_store::ObjectStore,
    ) -> lance_core::Result<ManifestLocation> {
        let path = base_path.child(MANIFEST_PATH);
        let meta = object_store.inner.head(&path).await?;
        let manifest = read_manifest(object_store, &path, Some(meta.size)).await?;
        Ok(manifest_location_from_latest_copy(path, meta, &manifest))
    }

    async fn commit(
        &self,
        _manifest: &mut Manifest,
        _indices: Option<Vec<IndexMetadata>>,
        _base_path: &Path,
        _object_store: &lance_io::object_store::ObjectStore,
        _manifest_writer: ManifestWriter,
        _naming_scheme: ManifestNamingScheme,
        _transaction: Option<Transaction>,
    ) -> std::result::Result<ManifestLocation, CommitError> {
        Err(CommitError::OtherError(lance_core::Error::not_supported(
            "lancedb-wasm is read-only",
        )))
    }
}

fn manifest_location_from_latest_copy(
    path: Path,
    meta: ObjectMeta,
    manifest: &Manifest,
) -> ManifestLocation {
    ManifestLocation {
        version: manifest.version,
        path,
        size: Some(meta.size),
        naming_scheme: ManifestNamingScheme::V2,
        e_tag: meta.e_tag,
    }
}

fn object_store_root_url(table_url: &Url) -> Url {
    let mut root = table_url.clone();
    root.set_path("/");
    root.set_query(None);
    root.set_fragment(None);
    root
}

fn object_store_table_path(table_url: &Url) -> Result<Path> {
    Path::from_url_path(table_url.path()).map_err(Error::from)
}

fn build_fts_query(request: TextRequest) -> Result<FullTextSearchQuery> {
    match request {
        TextRequest::Query(query) => Ok(FullTextSearchQuery::new(query)),
        TextRequest::Structured { query, columns } => {
            let query = FullTextSearchQuery::new(query);
            if let Some(columns) = columns {
                query.with_columns(&columns).map_err(Error::from)
            } else {
                Ok(query)
            }
        }
    }
}

fn apply_common_query<Q: QueryBase>(mut query: Q, request: &SearchRequest) -> Q {
    if let Some(limit) = request.limit {
        query = query.limit(limit);
    }
    if let Some(offset) = request.offset {
        query = query.offset(offset);
    }
    if let Some(filter) = request.filter.as_deref() {
        query = query.only_if(filter);
    }
    if let Some(select) = request.select.as_ref() {
        query = match select {
            SelectRequest::Columns(columns) => query.select(Select::Columns(columns.clone())),
            SelectRequest::Dynamic(columns) => query.select(Select::Dynamic(
                columns
                    .iter()
                    .map(|(name, expr)| (name.clone(), expr.clone()))
                    .collect(),
            )),
        };
    }
    if request.fast_search.unwrap_or(false) {
        query = query.fast_search();
    }
    if matches!(request.prefilter, Some(false)) {
        query = query.postfilter();
    }
    if request.with_row_id.unwrap_or(false) {
        query = query.with_row_id();
    }
    query
}

async fn execute_query<Q>(query: Q) -> Result<Vec<u8>>
where
    Q: ExecutableQuery,
{
    let stream = query.execute().await?;
    let schema = stream.schema();
    let batches = stream.try_collect::<Vec<_>>().await?;
    if batches.is_empty() {
        return schema_to_ipc_file(schema.as_ref());
    }
    batches_to_ipc_file(&batches)
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
