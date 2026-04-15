// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

//! Shared read/query protocol types for LanceDB remote and web runtimes.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

/// Deterministic copy of the latest manifest for static HTTP hosting.
pub const LATEST_MANIFEST_PATH: &str = "_latest.manifest";
/// Sidecar containing the latest published table version.
pub const LATEST_VERSION_PATH: &str = "_latest.version";
/// Published metadata sidecar.
pub const WEB_METADATA_PATH: &str = "_web.json";
/// Published snapshot sidecar.
pub const SNAPSHOT_PATH: &str = "_snapshot.json";

/// Read-only search request shared by the published web and remote runtimes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SearchRequest {
    /// Query vector for vector or hybrid search.
    pub vector: Option<Vec<f32>>,
    /// Optional full-text search query.
    pub text: Option<TextRequest>,
    /// Distance metric for vector search.
    pub distance_type: Option<SearchDistanceType>,
    /// Optional SQL filter.
    pub filter: Option<String>,
    /// Requested column projection.
    pub select: Option<SelectRequest>,
    /// Maximum number of rows to return.
    pub limit: Option<usize>,
    /// Row offset for pagination.
    pub offset: Option<usize>,
    /// Optional vector column override.
    pub vector_column: Option<String>,
    /// Whether to prefilter before vector search.
    pub prefilter: Option<bool>,
    /// Whether to include the `_rowid` meta column.
    pub with_row_id: Option<bool>,
    /// Whether to use serving-optimized execution when supported.
    pub fast_search: Option<bool>,
}

/// Full-text search request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum TextRequest {
    /// Simple text query across advertised default columns.
    Query(String),
    /// Text query with explicit column selection.
    Structured {
        /// Full-text search query.
        query: String,
        /// Optional subset of FTS-enabled columns.
        columns: Option<Vec<String>>,
    },
}

/// Distance metric for vector search.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SearchDistanceType {
    /// Euclidean distance.
    #[default]
    L2,
    /// Cosine distance.
    Cosine,
    /// Dot-product similarity.
    Dot,
    /// Hamming distance.
    Hamming,
}

/// Requested output columns for a search.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum SelectRequest {
    /// Plain projection by column name.
    Columns(Vec<String>),
    /// Dynamic projection where each key is an alias and each value is an expression.
    Dynamic(BTreeMap<String, String>),
}

/// Search capability hints advertised by published sidecars.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PublishedSearchCapabilities {
    /// Default vector column to use when the caller omits one.
    pub default_vector_column: Option<String>,
    /// Vector-searchable columns.
    #[serde(default)]
    pub vector_columns: Vec<String>,
    /// Full-text-searchable columns.
    #[serde(default)]
    pub fts_columns: Vec<String>,
}

impl PublishedSearchCapabilities {
    /// Returns the advertised default vector column, if any.
    pub fn default_vector_column(&self) -> Option<&str> {
        self.default_vector_column.as_deref()
    }

    /// Returns the advertised full-text-searchable columns.
    pub fn fts_columns(&self) -> &[String] {
        self.fts_columns.as_slice()
    }
}

/// Metadata published next to a table for read-only HTTP access.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PublishedTableMetadata {
    /// Current table version.
    pub version: u64,
    /// Canonical manifest path for the current version.
    pub manifest_path: String,
    /// Size of the canonical manifest, when known.
    pub manifest_size_bytes: Option<u64>,
    /// Manifest naming scheme used by the dataset.
    pub manifest_naming_scheme: String,
    /// Relative path to the deterministic latest manifest copy.
    pub latest_manifest_path: String,
    /// Relative path to the latest-version sidecar.
    pub latest_version_path: String,
    /// Relative path to the web metadata sidecar.
    pub web_metadata_path: String,
    /// Relative path to the published snapshot sidecar.
    pub snapshot_path: String,
    /// Advertised search capabilities.
    #[serde(flatten)]
    pub capabilities: PublishedSearchCapabilities,
    /// Arbitrary user-defined key-value metadata (for example embedding model).
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// Snapshot of the files and capability hints needed for read-only HTTP access.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PublishedSnapshot {
    /// Current table version.
    pub version: u64,
    /// Canonical manifest path for the current version.
    pub manifest_path: String,
    /// Size of the canonical manifest, when known.
    pub manifest_size_bytes: Option<u64>,
    /// Manifest naming scheme used by the dataset.
    pub manifest_naming_scheme: String,
    /// Relative path to the deterministic latest manifest copy.
    pub latest_manifest_path: String,
    /// Relative path to the latest-version sidecar.
    pub latest_version_path: String,
    /// Relative path to the web metadata sidecar.
    pub web_metadata_path: String,
    /// Relative path to the published snapshot sidecar.
    pub snapshot_path: String,
    /// Advertised search capabilities.
    #[serde(flatten)]
    pub capabilities: PublishedSearchCapabilities,
    /// Arbitrary user-defined key-value metadata (for example embedding model).
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    /// Whether the snapshot contains every required file path.
    pub is_complete: bool,
    /// Additional base paths referenced by the snapshot.
    #[serde(default)]
    pub base_paths: Vec<PublishedBasePath>,
    /// Files required to serve the published table.
    #[serde(default)]
    pub files: Vec<PublishedFile>,
}

/// External base path referenced by a published snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PublishedBasePath {
    /// Stable base-path identifier.
    pub id: u32,
    /// Optional human-readable name.
    pub name: Option<String>,
    /// URI or relative path backing the base path.
    pub path: String,
    /// Whether the base path points at the dataset root.
    pub is_dataset_root: bool,
}

/// File entry advertised by a published snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PublishedFile {
    /// Relative file path.
    pub path: String,
    /// File kind such as `data`, `index`, or `manifest`.
    pub kind: String,
    /// File size, when known.
    pub size_bytes: Option<u64>,
    /// Optional base-path identifier for externally rooted files.
    pub base_id: Option<u32>,
}

/// Shared accessors for published sidecar metadata types.
pub trait PublishedSearchMetadataExt {
    /// Returns the advertised search capabilities.
    fn capabilities(&self) -> &PublishedSearchCapabilities;

    /// Returns the default vector column advertised by the sidecar.
    fn default_vector_column(&self) -> Option<&str> {
        self.capabilities().default_vector_column()
    }

    /// Returns the full-text-searchable columns advertised by the sidecar.
    fn fts_columns(&self) -> &[String] {
        self.capabilities().fts_columns()
    }
}

impl PublishedSearchMetadataExt for PublishedTableMetadata {
    fn capabilities(&self) -> &PublishedSearchCapabilities {
        &self.capabilities
    }
}

impl PublishedSearchMetadataExt for PublishedSnapshot {
    fn capabilities(&self) -> &PublishedSearchCapabilities {
        &self.capabilities
    }
}
