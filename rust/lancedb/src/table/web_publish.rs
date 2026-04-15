// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;
use lance::dataset::{ReadParams, WriteParams};
use lance_table::format::{IndexMetadata, Manifest, RowIdMeta, Transaction};
use lance_table::io::commit::{
    CommitError, CommitHandler, ManifestLocation, ManifestNamingScheme, ManifestWriter,
    commit_handler_from_url,
};
use lance_table::io::deletion::relative_deletion_file_path;
use lance_table::io::manifest::read_manifest;
use lancedb_read_protocol::{
    LATEST_MANIFEST_PATH, LATEST_VERSION_PATH, PublishedBasePath, PublishedFile,
    PublishedSearchCapabilities, PublishedSnapshot, PublishedTableMetadata, SNAPSHOT_PATH,
    WEB_METADATA_PATH,
};
use log::warn;
use object_store::ObjectMeta;
use object_store::path::Path;

use crate::utils::supported_vector_data_type;
use crate::{Error, Result};

const WEB_METADATA_PREFIX: &str = "lancedb:web:";
const MANIFEST_NAMING_SCHEME_KEY: &str = "lancedb:web:manifest_naming_scheme";
const LATEST_MANIFEST_PATH_KEY: &str = "lancedb:web:latest_manifest_path";
const LATEST_VERSION_PATH_KEY: &str = "lancedb:web:latest_version_path";
const WEB_METADATA_PATH_KEY: &str = "lancedb:web:web_metadata_path";
const SNAPSHOT_PATH_KEY: &str = "lancedb:web:snapshot_path";
const DEFAULT_VECTOR_COLUMN_KEY: &str = "lancedb:web:default_vector_column";
const VECTOR_COLUMNS_KEY: &str = "lancedb:web:vector_columns";
const FTS_COLUMNS_KEY: &str = "lancedb:web:fts_columns";

#[derive(Debug)]
pub(crate) struct WebPublishCommitHandler {
    inner: Arc<dyn CommitHandler>,
}

impl WebPublishCommitHandler {
    pub(crate) fn new(inner: Arc<dyn CommitHandler>) -> Self {
        Self { inner }
    }
}

type WebSearchCapabilities = PublishedSearchCapabilities;
type WebTableMetadata = PublishedTableMetadata;

#[derive(Debug, Clone)]
struct SnapshotPlan {
    files: Vec<PublishedFile>,
    base_paths: Vec<PublishedBasePath>,
    is_complete: bool,
}

#[async_trait]
impl CommitHandler for WebPublishCommitHandler {
    async fn resolve_latest_location(
        &self,
        base_path: &Path,
        object_store: &lance_io::object_store::ObjectStore,
    ) -> lance_core::Result<ManifestLocation> {
        let latest_copy = resolve_latest_copy_location(base_path, object_store).await;
        let inner_latest = self
            .inner
            .resolve_latest_location(base_path, object_store)
            .await;

        match (latest_copy, inner_latest) {
            (Ok(Some(copy)), Ok(inner)) => Ok(if inner.version > copy.version {
                inner
            } else {
                copy
            }),
            (Ok(Some(copy)), Err(_)) => Ok(copy),
            (Ok(None), result) => result,
            (Err(source), Ok(inner)) => {
                warn!(
                    "Failed to read published latest manifest copy at {}: {}. Falling back to canonical manifest resolution.",
                    base_path.child(LATEST_MANIFEST_PATH),
                    source
                );
                Ok(inner)
            }
            (Err(source), Err(_)) => Err(source),
        }
    }

    async fn resolve_version_location(
        &self,
        base_path: &Path,
        version: u64,
        object_store: &dyn object_store::ObjectStore,
    ) -> lance_core::Result<ManifestLocation> {
        self.inner
            .resolve_version_location(base_path, version, object_store)
            .await
    }

    fn list_detached_manifest_locations<'a>(
        &self,
        base_path: &Path,
        object_store: &'a lance_io::object_store::ObjectStore,
    ) -> futures::stream::BoxStream<'a, lance_core::Result<ManifestLocation>> {
        self.inner
            .list_detached_manifest_locations(base_path, object_store)
    }

    fn list_manifest_locations<'a>(
        &self,
        base_path: &Path,
        object_store: &'a lance_io::object_store::ObjectStore,
        sorted_descending: bool,
    ) -> futures::stream::BoxStream<'a, lance_core::Result<ManifestLocation>> {
        self.inner
            .list_manifest_locations(base_path, object_store, sorted_descending)
    }

    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<IndexMetadata>>,
        base_path: &Path,
        object_store: &lance_io::object_store::ObjectStore,
        manifest_writer: ManifestWriter,
        naming_scheme: ManifestNamingScheme,
        transaction: Option<Transaction>,
    ) -> std::result::Result<ManifestLocation, CommitError> {
        let capabilities = collect_search_capabilities(manifest, indices.as_deref());
        let snapshot_plan = build_snapshot_plan(manifest, indices.as_deref());
        stamp_manifest_metadata(manifest, &capabilities, naming_scheme);

        let location = self
            .inner
            .commit(
                manifest,
                indices,
                base_path,
                object_store,
                manifest_writer,
                naming_scheme,
                transaction,
            )
            .await?;

        if let Err(source) = publish_sidecars(
            object_store,
            base_path,
            manifest,
            &location,
            &capabilities,
            &snapshot_plan,
        )
        .await
        {
            warn!(
                "Committed manifest {} but failed to update web publish sidecars: {}",
                location.path, source
            );
        }

        Ok(location)
    }

    async fn delete(&self, base_path: &Path) -> lance_core::Result<()> {
        self.inner.delete(base_path).await
    }
}

pub(crate) async fn patch_read_params(uri: &str, params: ReadParams) -> Result<ReadParams> {
    let commit_handler = match params.commit_handler.clone() {
        Some(commit_handler) => commit_handler,
        None => commit_handler_from_url(uri, &params.store_options).await?,
    };
    Ok(ReadParams {
        commit_handler: Some(wrap_commit_handler(commit_handler)),
        ..params
    })
}

pub(crate) async fn patch_write_params(uri: &str, mut params: WriteParams) -> Result<WriteParams> {
    #[allow(deprecated)]
    if params
        .store_params
        .as_ref()
        .map(|opts| opts.object_store.is_some())
        .unwrap_or_default()
        && params.commit_handler.is_none()
    {
        return Err(Error::InvalidInput {
            message: "when creating a dataset with a custom object store the commit_handler must also be specified".to_string(),
        });
    }

    let inner = match params.commit_handler.take() {
        Some(commit_handler) => {
            if uri.starts_with("s3+ddb") {
                return Err(Error::InvalidInput {
                    message: "`s3+ddb://` scheme and custom commit handler are mutually exclusive"
                        .to_string(),
                });
            }
            commit_handler
        }
        None => commit_handler_from_url(uri, &params.store_params).await?,
    };
    params.commit_handler = Some(wrap_commit_handler(inner));
    Ok(params)
}

pub(crate) fn wrap_commit_handler(inner: Arc<dyn CommitHandler>) -> Arc<dyn CommitHandler> {
    Arc::new(WebPublishCommitHandler::new(inner))
}

async fn resolve_latest_copy_location(
    base_path: &Path,
    object_store: &lance_io::object_store::ObjectStore,
) -> lance_core::Result<Option<ManifestLocation>> {
    let path = base_path.child(LATEST_MANIFEST_PATH);
    match object_store.inner.head(&path).await {
        Ok(meta) => {
            let manifest = read_manifest(object_store, &path, Some(meta.size)).await?;
            Ok(Some(manifest_location_from_latest_copy(
                path, meta, &manifest,
            )))
        }
        Err(object_store::Error::NotFound { .. }) => Ok(None),
        Err(source) => Err(source.into()),
    }
}

fn stamp_manifest_metadata(
    manifest: &mut Manifest,
    capabilities: &WebSearchCapabilities,
    naming_scheme: ManifestNamingScheme,
) {
    let metadata = manifest.table_metadata_mut();
    metadata.retain(|key, _| !key.starts_with(WEB_METADATA_PREFIX));
    metadata.insert(
        MANIFEST_NAMING_SCHEME_KEY.to_string(),
        manifest_naming_scheme_name(naming_scheme).to_string(),
    );
    metadata.insert(
        LATEST_MANIFEST_PATH_KEY.to_string(),
        LATEST_MANIFEST_PATH.to_string(),
    );
    metadata.insert(
        LATEST_VERSION_PATH_KEY.to_string(),
        LATEST_VERSION_PATH.to_string(),
    );
    metadata.insert(
        WEB_METADATA_PATH_KEY.to_string(),
        WEB_METADATA_PATH.to_string(),
    );
    metadata.insert(SNAPSHOT_PATH_KEY.to_string(), SNAPSHOT_PATH.to_string());
    metadata.insert(
        VECTOR_COLUMNS_KEY.to_string(),
        serde_json::to_string(&capabilities.vector_columns)
            .expect("serializing vector column metadata should succeed"),
    );
    metadata.insert(
        FTS_COLUMNS_KEY.to_string(),
        serde_json::to_string(&capabilities.fts_columns)
            .expect("serializing fts column metadata should succeed"),
    );
    if let Some(default_vector_column) = &capabilities.default_vector_column {
        metadata.insert(
            DEFAULT_VECTOR_COLUMN_KEY.to_string(),
            default_vector_column.clone(),
        );
    }
}

fn collect_search_capabilities(
    manifest: &Manifest,
    indices: Option<&[IndexMetadata]>,
) -> WebSearchCapabilities {
    let vector_columns = collect_indexed_columns(manifest, indices, "VectorIndexDetails");
    let fts_columns = collect_indexed_columns(manifest, indices, "InvertedIndexDetails");
    let schema_vector_columns = manifest
        .schema
        .fields
        .iter()
        .filter(|field| supported_vector_data_type(&field.data_type()))
        .map(|field| field.name.clone())
        .collect::<Vec<_>>();

    let default_vector_column = if vector_columns.len() == 1 {
        Some(vector_columns[0].clone())
    } else if vector_columns.is_empty() && schema_vector_columns.len() == 1 {
        Some(schema_vector_columns[0].clone())
    } else {
        None
    };

    WebSearchCapabilities {
        default_vector_column,
        vector_columns,
        fts_columns,
    }
}

fn collect_indexed_columns(
    manifest: &Manifest,
    indices: Option<&[IndexMetadata]>,
    type_suffix: &str,
) -> Vec<String> {
    let mut columns = BTreeSet::new();
    for index in indices.into_iter().flatten() {
        let matches_type = index
            .index_details
            .as_ref()
            .map(|details| details.type_url.ends_with(type_suffix))
            .unwrap_or(false);
        if !matches_type {
            continue;
        }
        for field_id in &index.fields {
            if let Some(field) = manifest.schema.field_by_id(*field_id) {
                columns.insert(field.name.clone());
            }
        }
    }
    columns.into_iter().collect()
}

fn build_snapshot_plan(manifest: &Manifest, indices: Option<&[IndexMetadata]>) -> SnapshotPlan {
    let mut files = Vec::new();
    for fragment in manifest.fragments.iter() {
        for data_file in &fragment.files {
            push_snapshot_file(
                &mut files,
                PublishedFile {
                    path: data_file.path.clone(),
                    kind: "data".to_string(),
                    size_bytes: data_file.file_size_bytes.get().map(|size| size.get()),
                    base_id: data_file.base_id,
                },
            );
        }
        if let Some(deletion_file) = &fragment.deletion_file {
            push_snapshot_file(
                &mut files,
                PublishedFile {
                    path: relative_deletion_file_path(fragment.id, deletion_file),
                    kind: "deletion".to_string(),
                    size_bytes: None,
                    base_id: deletion_file.base_id,
                },
            );
        }
        if let Some(RowIdMeta::External(external_file)) = &fragment.row_id_meta {
            push_snapshot_file(
                &mut files,
                PublishedFile {
                    path: external_file.path.clone(),
                    kind: "rowIds".to_string(),
                    size_bytes: Some(external_file.size),
                    base_id: None,
                },
            );
        }
    }

    let mut is_complete = true;
    for index in indices.into_iter().flatten() {
        let Some(index_files) = &index.files else {
            is_complete = false;
            continue;
        };
        for file in index_files {
            push_snapshot_file(
                &mut files,
                PublishedFile {
                    path: format!("_indices/{}/{}", index.uuid, file.path),
                    kind: "index".to_string(),
                    size_bytes: Some(file.size_bytes),
                    base_id: index.base_id,
                },
            );
        }
    }

    if let Some(transaction_file) = &manifest.transaction_file
        && !transaction_file.is_empty()
    {
        push_snapshot_file(
            &mut files,
            PublishedFile {
                path: transaction_file.clone(),
                kind: "transaction".to_string(),
                size_bytes: None,
                base_id: None,
            },
        );
    }

    files.sort_by(|left, right| {
        (&left.base_id, &left.kind, &left.path).cmp(&(&right.base_id, &right.kind, &right.path))
    });

    let mut base_paths = manifest
        .base_paths
        .values()
        .map(|base_path| PublishedBasePath {
            id: base_path.id,
            name: base_path.name.clone(),
            path: base_path.path.clone(),
            is_dataset_root: base_path.is_dataset_root,
        })
        .collect::<Vec<_>>();
    base_paths.sort_by_key(|base_path| base_path.id);

    SnapshotPlan {
        files,
        base_paths,
        is_complete,
    }
}

fn ensure_publishable_snapshot(snapshot_plan: &SnapshotPlan) -> Result<()> {
    if let Some(file) = snapshot_plan
        .files
        .iter()
        .find(|file| file.base_id.is_some())
    {
        return Err(Error::NotSupported {
            message: format!(
                "web publish sidecars do not support external base-path references yet; file '{}' ({}) uses base_id {}",
                file.path,
                file.kind,
                file.base_id.expect("checked above")
            ),
        });
    }

    if let Some(base_path) = snapshot_plan.base_paths.first() {
        return Err(Error::NotSupported {
            message: format!(
                "web publish sidecars do not support external base-path references yet; base path {} points to {}",
                base_path.id, base_path.path
            ),
        });
    }

    Ok(())
}

fn push_snapshot_file(files: &mut Vec<PublishedFile>, file: PublishedFile) {
    let already_present = files.iter().any(|candidate| {
        candidate.path == file.path
            && candidate.kind == file.kind
            && candidate.base_id == file.base_id
    });
    if !already_present {
        files.push(file);
    }
}

async fn publish_sidecars(
    object_store: &lance_io::object_store::ObjectStore,
    base_path: &Path,
    manifest: &Manifest,
    manifest_location: &ManifestLocation,
    capabilities: &WebSearchCapabilities,
    snapshot_plan: &SnapshotPlan,
) -> Result<()> {
    ensure_publishable_snapshot(snapshot_plan)?;

    let manifest_bytes = object_store
        .inner
        .get(&manifest_location.path)
        .await?
        .bytes()
        .await?;
    let published_metadata = manifest.config.clone();

    let latest_version_bytes = manifest.version.to_string().into_bytes();
    let metadata = WebTableMetadata {
        version: manifest.version,
        manifest_path: manifest_location.path.to_string(),
        manifest_size_bytes: manifest_location.size,
        manifest_naming_scheme: manifest_naming_scheme_name(manifest_location.naming_scheme)
            .to_string(),
        latest_manifest_path: LATEST_MANIFEST_PATH.to_string(),
        latest_version_path: LATEST_VERSION_PATH.to_string(),
        web_metadata_path: WEB_METADATA_PATH.to_string(),
        snapshot_path: SNAPSHOT_PATH.to_string(),
        capabilities: capabilities.clone(),
        metadata: published_metadata.clone(),
    };
    let metadata_json = serde_json::to_vec_pretty(&metadata).map_err(|source| Error::Runtime {
        message: format!("failed to serialize web table metadata: {source}"),
    })?;

    let mut snapshot_files = snapshot_plan.files.clone();
    push_snapshot_file(
        &mut snapshot_files,
        PublishedFile {
            path: manifest_location.path.to_string(),
            kind: "manifest".to_string(),
            size_bytes: manifest_location
                .size
                .or_else(|| u64::try_from(manifest_bytes.len()).ok()),
            base_id: None,
        },
    );
    push_snapshot_file(
        &mut snapshot_files,
        PublishedFile {
            path: LATEST_MANIFEST_PATH.to_string(),
            kind: "latestManifest".to_string(),
            size_bytes: u64::try_from(manifest_bytes.len()).ok(),
            base_id: None,
        },
    );
    push_snapshot_file(
        &mut snapshot_files,
        PublishedFile {
            path: LATEST_VERSION_PATH.to_string(),
            kind: "latestVersion".to_string(),
            size_bytes: u64::try_from(latest_version_bytes.len()).ok(),
            base_id: None,
        },
    );
    push_snapshot_file(
        &mut snapshot_files,
        PublishedFile {
            path: WEB_METADATA_PATH.to_string(),
            kind: "webMetadata".to_string(),
            size_bytes: u64::try_from(metadata_json.len()).ok(),
            base_id: None,
        },
    );
    snapshot_files.sort_by(|left, right| {
        (&left.base_id, &left.kind, &left.path).cmp(&(&right.base_id, &right.kind, &right.path))
    });

    let snapshot = PublishedSnapshot {
        version: manifest.version,
        manifest_path: manifest_location.path.to_string(),
        manifest_size_bytes: manifest_location.size,
        manifest_naming_scheme: manifest_naming_scheme_name(manifest_location.naming_scheme)
            .to_string(),
        latest_manifest_path: LATEST_MANIFEST_PATH.to_string(),
        latest_version_path: LATEST_VERSION_PATH.to_string(),
        web_metadata_path: WEB_METADATA_PATH.to_string(),
        snapshot_path: SNAPSHOT_PATH.to_string(),
        capabilities: capabilities.clone(),
        metadata: published_metadata,
        is_complete: snapshot_plan.is_complete,
        base_paths: snapshot_plan.base_paths.clone(),
        files: snapshot_files,
    };
    let snapshot_json = serde_json::to_vec_pretty(&snapshot).map_err(|source| Error::Runtime {
        message: format!("failed to serialize published snapshot manifest: {source}"),
    })?;

    object_store
        .inner
        .put(
            &base_path.child(LATEST_MANIFEST_PATH),
            manifest_bytes.into(),
        )
        .await?;
    object_store
        .inner
        .put(
            &base_path.child(LATEST_VERSION_PATH),
            latest_version_bytes.into(),
        )
        .await?;
    object_store
        .inner
        .put(&base_path.child(WEB_METADATA_PATH), metadata_json.into())
        .await?;
    object_store
        .inner
        .put(&base_path.child(SNAPSHOT_PATH), snapshot_json.into())
        .await?;

    Ok(())
}

fn manifest_naming_scheme_name(naming_scheme: ManifestNamingScheme) -> &'static str {
    match naming_scheme {
        ManifestNamingScheme::V1 => "v1",
        ManifestNamingScheme::V2 => "v2",
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

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;

    use arrow_array::{
        Array, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, StringArray,
    };
    use arrow_data::ArrayDataBuilder;
    use arrow_schema::{DataType, Field, Schema};
    use lance_table::format::BasePath;
    use tempfile::tempdir;

    use crate::connect;
    use crate::index::Index;
    use crate::table::{BaseTable, NativeTable};

    use super::{
        DEFAULT_VECTOR_COLUMN_KEY, FTS_COLUMNS_KEY, LATEST_MANIFEST_PATH, LATEST_VERSION_PATH,
        MANIFEST_NAMING_SCHEME_KEY, PublishedSnapshot, SNAPSHOT_PATH, VECTOR_COLUMNS_KEY,
        WEB_METADATA_PATH, WebTableMetadata, build_snapshot_plan, ensure_publishable_snapshot,
    };

    fn make_batch(start: i32, len: i32) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from_iter_values(start..(start + len)))],
        )
        .unwrap()
    }

    fn make_vectors(start: usize, rows: usize, dimension: i32) -> FixedSizeListArray {
        let values = Float32Array::from_iter_values(
            (start..(start + rows * dimension as usize)).map(|value| value as f32),
        );
        let list_type = DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dimension,
        );
        let data = ArrayDataBuilder::new(list_type)
            .len(rows)
            .add_child_data(values.into_data())
            .build()
            .unwrap();
        FixedSizeListArray::from(data)
    }

    fn make_search_batch(rows: usize) -> RecordBatch {
        let dimension = 8;
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("text", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dimension,
                ),
                false,
            ),
        ]));
        let text =
            StringArray::from_iter_values((0..rows).map(|row| ["cat", "dog", "fish"][row % 3]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from_iter_values(0..rows as i32)),
                Arc::new(text),
                Arc::new(make_vectors(0, rows, dimension)),
            ],
        )
        .unwrap()
    }

    fn table_root(base: &std::path::Path) -> PathBuf {
        base.join("published.lance")
    }

    #[tokio::test]
    async fn writes_web_sidecars_on_create_and_append() {
        let dir = tempdir().unwrap();
        let db = connect(dir.path().to_str().unwrap())
            .execute()
            .await
            .unwrap();

        let table = db
            .create_table("published", make_batch(0, 4))
            .execute()
            .await
            .unwrap();

        let root = table_root(dir.path());
        let latest_manifest = root.join(LATEST_MANIFEST_PATH);
        let latest_version = root.join(LATEST_VERSION_PATH);
        let web_metadata = root.join(WEB_METADATA_PATH);
        let snapshot_path = root.join(SNAPSHOT_PATH);

        assert!(latest_manifest.exists());
        assert!(latest_version.exists());
        assert!(web_metadata.exists());
        assert!(snapshot_path.exists());

        let initial_metadata: WebTableMetadata =
            serde_json::from_slice(&fs::read(&web_metadata).unwrap()).unwrap();
        assert_eq!(initial_metadata.version, 1);
        assert!(initial_metadata.manifest_path.contains("_versions/"));
        assert!(initial_metadata.manifest_size_bytes.is_some());
        assert_eq!(initial_metadata.manifest_naming_scheme, "v2");
        assert_eq!(initial_metadata.latest_manifest_path, LATEST_MANIFEST_PATH);
        assert_eq!(initial_metadata.latest_version_path, LATEST_VERSION_PATH);
        assert_eq!(fs::read_to_string(&latest_version).unwrap(), "1");

        table.add(make_batch(10, 2)).execute().await.unwrap();

        let updated_metadata: WebTableMetadata =
            serde_json::from_slice(&fs::read(&web_metadata).unwrap()).unwrap();
        assert_eq!(updated_metadata.version, 2);
        assert!(updated_metadata.manifest_path.contains("_versions/"));
        assert_eq!(fs::read_to_string(&latest_version).unwrap(), "2");
    }

    #[tokio::test]
    async fn writes_publish_metadata_for_searchable_table() {
        let dir = tempdir().unwrap();
        let db = connect(dir.path().to_str().unwrap())
            .execute()
            .await
            .unwrap();

        let table = db
            .create_table("published", make_search_batch(256))
            .execute()
            .await
            .unwrap();
        table
            .create_index(&["text"], Index::FTS(Default::default()))
            .execute()
            .await
            .unwrap();
        table
            .create_index(&["vector"], Index::Auto)
            .execute()
            .await
            .unwrap();
        table.delete("id = 0").await.unwrap();

        let root = table_root(dir.path());
        let metadata: WebTableMetadata =
            serde_json::from_slice(&fs::read(root.join(WEB_METADATA_PATH)).unwrap()).unwrap();
        let snapshot: PublishedSnapshot =
            serde_json::from_slice(&fs::read(root.join(SNAPSHOT_PATH)).unwrap()).unwrap();
        let manifest = table.as_native().unwrap().manifest().await.unwrap();

        assert_eq!(
            metadata.capabilities.default_vector_column.as_deref(),
            Some("vector")
        );
        assert_eq!(
            metadata.capabilities.vector_columns,
            vec!["vector".to_string()]
        );
        assert_eq!(metadata.capabilities.fts_columns, vec!["text".to_string()]);
        assert_eq!(metadata.metadata, manifest.config);
        assert_eq!(
            snapshot.capabilities.default_vector_column.as_deref(),
            Some("vector")
        );
        assert_eq!(
            snapshot.capabilities.vector_columns,
            vec!["vector".to_string()]
        );
        assert_eq!(snapshot.capabilities.fts_columns, vec!["text".to_string()]);
        assert_eq!(snapshot.metadata, manifest.config);
        assert!(snapshot.is_complete);
        assert!(
            snapshot
                .files
                .iter()
                .any(|file| file.kind == "manifest" && file.path == metadata.manifest_path)
        );
        assert!(
            snapshot
                .files
                .iter()
                .any(|file| file.kind == "latestManifest" && file.path == LATEST_MANIFEST_PATH)
        );
        assert!(
            snapshot
                .files
                .iter()
                .any(|file| file.kind == "latestVersion" && file.path == LATEST_VERSION_PATH)
        );
        assert!(
            snapshot
                .files
                .iter()
                .any(|file| file.kind == "webMetadata" && file.path == WEB_METADATA_PATH)
        );
        assert!(snapshot.files.iter().any(|file| file.kind == "data"));
        assert!(snapshot.files.iter().any(|file| file.kind == "index"));
        assert!(snapshot.files.iter().any(|file| file.kind == "deletion"));

        assert_eq!(
            manifest.table_metadata.get(MANIFEST_NAMING_SCHEME_KEY),
            Some(&"v2".to_string())
        );
        assert_eq!(
            manifest.table_metadata.get(DEFAULT_VECTOR_COLUMN_KEY),
            Some(&"vector".to_string())
        );
        assert_eq!(
            serde_json::from_str::<Vec<String>>(
                manifest.table_metadata.get(VECTOR_COLUMNS_KEY).unwrap()
            )
            .unwrap(),
            vec!["vector".to_string()]
        );
        assert_eq!(
            serde_json::from_str::<Vec<String>>(
                manifest.table_metadata.get(FTS_COLUMNS_KEY).unwrap()
            )
            .unwrap(),
            vec!["text".to_string()]
        );
    }

    #[tokio::test]
    async fn opens_from_latest_manifest_copy_without_versions_dir() {
        let dir = tempdir().unwrap();
        let db = connect(dir.path().to_str().unwrap())
            .execute()
            .await
            .unwrap();

        db.create_table("published", make_batch(0, 4))
            .execute()
            .await
            .unwrap();

        let root = table_root(dir.path());
        fs::rename(root.join("_versions"), root.join("_versions.hidden")).unwrap();

        let reopened = NativeTable::open(root.to_str().unwrap()).await.unwrap();
        assert_eq!(reopened.count_rows(None).await.unwrap(), 4);
    }

    #[tokio::test]
    async fn falls_back_to_canonical_manifest_when_latest_copy_is_corrupt() {
        let dir = tempdir().unwrap();
        let db = connect(dir.path().to_str().unwrap())
            .execute()
            .await
            .unwrap();

        db.create_table("published", make_batch(0, 4))
            .execute()
            .await
            .unwrap();

        let root = table_root(dir.path());
        fs::write(root.join(LATEST_MANIFEST_PATH), b"not a manifest").unwrap();

        let reopened = NativeTable::open(root.to_str().unwrap()).await.unwrap();
        assert_eq!(reopened.count_rows(None).await.unwrap(), 4);
    }

    #[tokio::test]
    async fn ignores_stale_latest_manifest_copy_when_newer_manifest_exists() {
        let dir = tempdir().unwrap();
        let db = connect(dir.path().to_str().unwrap())
            .execute()
            .await
            .unwrap();

        let table = db
            .create_table("published", make_batch(0, 4))
            .execute()
            .await
            .unwrap();

        let root = table_root(dir.path());
        let stale_manifest = fs::read(root.join(LATEST_MANIFEST_PATH)).unwrap();

        table.add(make_batch(10, 2)).execute().await.unwrap();

        fs::write(root.join(LATEST_MANIFEST_PATH), stale_manifest).unwrap();

        let reopened = NativeTable::open(root.to_str().unwrap()).await.unwrap();
        assert_eq!(reopened.count_rows(None).await.unwrap(), 6);
    }

    #[tokio::test]
    async fn publishes_user_metadata_in_sidecars() {
        let dir = tempdir().unwrap();
        let db = connect(dir.path().to_str().unwrap())
            .execute()
            .await
            .unwrap();

        let table = db
            .create_table("published", make_batch(0, 4))
            .execute()
            .await
            .unwrap();

        table
            .as_native()
            .unwrap()
            .update_config(vec![
                (
                    "embedding_model".to_string(),
                    "text-embedding-3-large".to_string(),
                ),
                ("llm_uri".to_string(), "openai://responses".to_string()),
            ])
            .await
            .unwrap();

        let root = table_root(dir.path());
        let manifest = table.as_native().unwrap().manifest().await.unwrap();
        let metadata: WebTableMetadata =
            serde_json::from_slice(&fs::read(root.join(WEB_METADATA_PATH)).unwrap()).unwrap();
        let snapshot: PublishedSnapshot =
            serde_json::from_slice(&fs::read(root.join(SNAPSHOT_PATH)).unwrap()).unwrap();

        assert_eq!(metadata.metadata, manifest.config);
        assert_eq!(
            metadata.metadata.get("embedding_model"),
            Some(&"text-embedding-3-large".to_string())
        );
        assert_eq!(
            metadata.metadata.get("llm_uri"),
            Some(&"openai://responses".to_string())
        );
        assert_eq!(snapshot.metadata, metadata.metadata);
    }

    #[tokio::test]
    async fn rejects_base_path_backed_tables_for_browser_publish_sidecars() {
        let dir = tempdir().unwrap();
        let db = connect(dir.path().to_str().unwrap())
            .execute()
            .await
            .unwrap();

        let table = db
            .create_table("published", make_batch(0, 4))
            .execute()
            .await
            .unwrap();

        let mut manifest = table.as_native().unwrap().manifest().await.unwrap();
        Arc::make_mut(&mut manifest.fragments)[0].files[0].base_id = Some(7);
        manifest.base_paths.insert(
            7,
            BasePath::new(
                7,
                "file:///tmp/external-root".to_string(),
                Some("external-root".to_string()),
                true,
            ),
        );

        let error = ensure_publishable_snapshot(&build_snapshot_plan(&manifest, None))
            .expect_err("browser publish sidecars should reject external base paths");

        match error {
            crate::Error::NotSupported { message } => {
                assert!(message.contains("base-path"));
                assert!(message.contains("base_id 7"));
            }
            other => panic!("expected not-supported error, got {other:?}"),
        }
    }
}
