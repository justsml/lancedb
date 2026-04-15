// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

//! Native (non-WASM) helpers for opening and querying LanceDB tables over HTTP.

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use futures::TryStreamExt;
use lance::dataset::ReadParams;
use lance::session::Session;
use lance_index::scalar::FullTextSearchQuery;
use lance_table::format::{IndexMetadata, Manifest, Transaction};
use lance_table::io::commit::{
    CommitError, CommitHandler, ManifestLocation, ManifestNamingScheme, ManifestWriter,
};
use lance_table::io::manifest::read_manifest;
use lancedb::ipc::{batches_to_ipc_file, schema_to_ipc_file};
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::table::{BaseTable, NativeTable};
use lancedb::{Error, Result, Table};
use object_store::ObjectMeta;
use object_store::path::Path;

use http::header::HeaderName;
use object_store::http::HttpBuilder;
use object_store::{HeaderMap, HeaderValue};
use std::collections::HashMap;
use url::Url;

use crate::{
    MANIFEST_PATH, OpenTableOptions, ResolvedPublishedState, SearchDistanceType, SearchRequest,
    SelectRequest, TextRequest, build_open_store,
};

pub(crate) fn session_from_cache_bytes(cache_bytes: Option<usize>) -> Session {
    match cache_bytes {
        Some(cache_bytes) => {
            let metadata_cache_size = cache_bytes / 4;
            let index_cache_size = cache_bytes.saturating_sub(metadata_cache_size);
            Session::new(index_cache_size, metadata_cache_size, Default::default())
        }
        None => Session::default(),
    }
}

pub(crate) async fn open_table_with_options(
    table_url: &str,
    table_name: &str,
    options: &OpenTableOptions,
    published: &ResolvedPublishedState,
    session: Arc<Session>,
) -> Result<Table> {
    let parsed_url = Url::parse(table_url).map_err(|source| Error::InvalidInput {
        message: format!("invalid table URL '{table_url}': {source}"),
    })?;
    let table_path = crate::object_store_table_path(&parsed_url)?;

    let (http_store, wrapper) =
        build_open_store(&parsed_url, &table_path, &published.manifest_url, options)?;
    let preflight_store = wrapper
        .as_ref()
        .map(|wrapper| wrapper.wrap("lancedb-wasm-preflight", http_store.clone()))
        .unwrap_or_else(|| http_store.clone());
    crate::preflight_manifest_check(preflight_store, &table_path.child(MANIFEST_PATH)).await?;

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
        HashSet::new(),
        None,
    )
    .await?;

    Ok(Table::from(Arc::new(native) as Arc<dyn BaseTable>))
}

pub(crate) fn build_http_store(
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

pub(crate) fn build_fts_query(request: TextRequest) -> Result<FullTextSearchQuery> {
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

pub(crate) async fn execute_table_search(table: &Table, request: SearchRequest) -> Result<Vec<u8>> {
    match (request.vector.clone(), request.text.clone()) {
        (Some(vector), text) => {
            let mut query = table.query().nearest_to(vector)?;
            if let Some(text) = text {
                query = query.full_text_search(build_fts_query(text)?);
            }
            if let Some(vector_column) = request.vector_column.as_deref() {
                query = query.column(vector_column);
            }
            if let Some(distance_type) = request.distance_type {
                query = query.distance_type(distance_type.into());
            }
            query = apply_common_query(query, &request);
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

impl From<SearchDistanceType> for lancedb::DistanceType {
    fn from(value: SearchDistanceType) -> Self {
        match value {
            SearchDistanceType::L2 => Self::L2,
            SearchDistanceType::Cosine => Self::Cosine,
            SearchDistanceType::Dot => Self::Dot,
            SearchDistanceType::Hamming => Self::Hamming,
        }
    }
}
