// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

use std::sync::Arc;

use async_trait::async_trait;
use lance::dataset::{ReadParams, WriteParams};
use lance_table::format::{IndexMetadata, Manifest, Transaction};
use lance_table::io::commit::{
    CommitError, CommitHandler, ManifestLocation, ManifestNamingScheme, ManifestWriter,
    commit_handler_from_url,
};
use lance_table::io::manifest::read_manifest;
use log::warn;
use object_store::ObjectMeta;
use object_store::path::Path;
use serde::Serialize;

use crate::{Error, Result};

pub(crate) const LATEST_MANIFEST_PATH: &str = "_latest.manifest";
pub(crate) const WEB_METADATA_PATH: &str = "_web.json";

#[derive(Debug)]
pub(crate) struct WebPublishCommitHandler {
    inner: Arc<dyn CommitHandler>,
}

impl WebPublishCommitHandler {
    pub(crate) fn new(inner: Arc<dyn CommitHandler>) -> Self {
        Self { inner }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct WebTableMetadata {
    version: u64,
    manifest_path: String,
    manifest_size_bytes: Option<u64>,
    manifest_naming_scheme: &'static str,
}

#[async_trait]
impl CommitHandler for WebPublishCommitHandler {
    async fn resolve_latest_location(
        &self,
        base_path: &Path,
        object_store: &lance_io::object_store::ObjectStore,
    ) -> lance_core::Result<ManifestLocation> {
        let path = base_path.child(LATEST_MANIFEST_PATH);
        match object_store.inner.head(&path).await {
            Ok(meta) => {
                let manifest = read_manifest(object_store, &path, Some(meta.size)).await?;
                Ok(manifest_location_from_latest_copy(path, meta, &manifest))
            }
            Err(object_store::Error::NotFound { .. }) => {
                self.inner
                    .resolve_latest_location(base_path, object_store)
                    .await
            }
            Err(source) => Err(source.into()),
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

        if let Err(source) = publish_sidecars(object_store, base_path, manifest, &location).await {
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

async fn publish_sidecars(
    object_store: &lance_io::object_store::ObjectStore,
    base_path: &Path,
    manifest: &Manifest,
    manifest_location: &ManifestLocation,
) -> Result<()> {
    let manifest_bytes = object_store
        .inner
        .get(&manifest_location.path)
        .await?
        .bytes()
        .await?;

    object_store
        .inner
        .put(
            &base_path.child(LATEST_MANIFEST_PATH),
            manifest_bytes.into(),
        )
        .await?;

    let metadata = WebTableMetadata {
        version: manifest.version,
        manifest_path: manifest_location.path.to_string(),
        manifest_size_bytes: manifest_location.size,
        manifest_naming_scheme: manifest_naming_scheme_name(manifest_location.naming_scheme),
    };
    let metadata_json = serde_json::to_vec_pretty(&metadata).map_err(|source| Error::Runtime {
        message: format!("failed to serialize web table metadata: {source}"),
    })?;
    object_store
        .inner
        .put(&base_path.child(WEB_METADATA_PATH), metadata_json.into())
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

    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use tempfile::tempdir;

    use crate::connect;
    use crate::table::{BaseTable, NativeTable};

    use super::{LATEST_MANIFEST_PATH, WEB_METADATA_PATH};

    #[derive(Debug, serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct WebTableMetadata {
        version: u64,
        manifest_path: String,
        manifest_size_bytes: Option<u64>,
        manifest_naming_scheme: String,
    }

    fn make_batch(start: i32, len: i32) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from_iter_values(start..(start + len)))],
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
        let web_metadata = root.join(WEB_METADATA_PATH);

        assert!(latest_manifest.exists());
        assert!(web_metadata.exists());

        let initial_metadata: WebTableMetadata =
            serde_json::from_slice(&fs::read(&web_metadata).unwrap()).unwrap();
        assert_eq!(initial_metadata.version, 1);
        assert!(initial_metadata.manifest_path.contains("_versions/"));
        assert!(initial_metadata.manifest_size_bytes.is_some());
        assert_eq!(initial_metadata.manifest_naming_scheme, "v2");

        table.add(make_batch(10, 2)).execute().await.unwrap();

        let updated_metadata: WebTableMetadata =
            serde_json::from_slice(&fs::read(&web_metadata).unwrap()).unwrap();
        assert_eq!(updated_metadata.version, 2);
        assert!(updated_metadata.manifest_path.contains("_versions/"));
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
}
