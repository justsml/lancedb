// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use async_trait::async_trait;
use futures::future::BoxFuture;
use object_store::path::Path;
use std::fmt::Debug;

use lance_core::Result;
use lance_io::object_store::ObjectStore;

use crate::format::{IndexMetadata, Manifest, Transaction};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManifestNamingScheme {
    V1,
    V2,
}

#[derive(Debug, Clone)]
pub struct ManifestLocation {
    pub version: u64,
    pub path: Path,
    pub size: Option<u64>,
    pub naming_scheme: ManifestNamingScheme,
    pub e_tag: Option<String>,
}

pub type ManifestWriter = for<'a> fn(
    object_store: &'a ObjectStore,
    manifest: &'a mut Manifest,
    indices: Option<Vec<IndexMetadata>>,
    path: &'a Path,
    transaction: Option<Transaction>,
) -> BoxFuture<'a, Result<()>>;

#[derive(Debug)]
pub enum CommitError {
    OtherError(lance_core::Error),
}

impl From<lance_core::Error> for CommitError {
    fn from(value: lance_core::Error) -> Self {
        Self::OtherError(value)
    }
}

#[async_trait]
pub trait CommitHandler: Debug + Send + Sync {
    async fn resolve_latest_location(
        &self,
        base_path: &Path,
        object_store: &ObjectStore,
    ) -> Result<ManifestLocation>;

    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<IndexMetadata>>,
        base_path: &Path,
        object_store: &ObjectStore,
        manifest_writer: ManifestWriter,
        naming_scheme: ManifestNamingScheme,
        transaction: Option<Transaction>,
    ) -> std::result::Result<ManifestLocation, CommitError>;
}
