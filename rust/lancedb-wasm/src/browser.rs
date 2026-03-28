// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashSet};
use std::sync::Arc;

use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchOptions};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use futures::{FutureExt, StreamExt, TryStreamExt};
use lance_core::ROW_ID;
use lance_core::cache::LanceCache;
use lance_core::datatypes::Schema as LanceSchema;
use lance_encoding::decoder::FilterExpression;
use lance_file::reader::{FileReader, FileReaderOptions, ReaderProjection};
use lance_io::ReadBatchParams;
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_table::format::{DataFile, Fragment, Manifest, RowIdMeta};
use lance_table::io::deletion::read_deletion_file;
use lance_table::io::manifest::read_manifest;
use lance_table::rowids::{RowIdSequence, read_row_ids};
use lance_table::utils::stream::{
    ReadBatchTask, ReadBatchTaskStream, RowIdAndDeletesConfig, merge_streams,
    wrap_with_row_id_and_delete,
};
use object_store::DynObjectStore;
use object_store::path::Path;
use url::Url;

use crate::local_error::{Error, Result};
use crate::{
    MANIFEST_PATH, OpenTableOptions, SearchRequest, SelectRequest, build_open_store,
    object_store_root_url, object_store_table_path, preflight_manifest_check,
};

const DEFAULT_BATCH_SIZE: u32 = 1024;
const DEFAULT_VECTOR_LIMIT: usize = 10;
const RESULT_DISTANCE_COLUMN: &str = "_distance";

#[derive(Clone)]
pub struct BrowserTable {
    manifest: Manifest,
    object_store: Arc<ObjectStore>,
    scan_scheduler: Arc<ScanScheduler>,
    table_path: Path,
    metadata_cache: LanceCache,
}

impl BrowserTable {
    pub async fn open(
        table_url: &str,
        options: &OpenTableOptions,
        manifest_url: &str,
    ) -> Result<Self> {
        let parsed_url = Url::parse(table_url).map_err(|source| Error::InvalidInput {
            message: format!("invalid table URL '{table_url}': {source}"),
        })?;
        let table_path = object_store_table_path(&parsed_url)?;
        let manifest_path = table_path.child(MANIFEST_PATH);

        let (http_store, wrapper) =
            build_open_store(&parsed_url, &table_path, manifest_url, options)?;
        let preflight_store = wrapper
            .as_ref()
            .map(|wrapper| wrapper.wrap("lancedb-wasm-preflight", http_store.clone()))
            .unwrap_or_else(|| http_store.clone());
        preflight_manifest_check(preflight_store, &manifest_path).await?;

        let object_store = Arc::new(ObjectStore::new(
            http_store as Arc<DynObjectStore>,
            object_store_root_url(&parsed_url),
            None,
            wrapper,
            false,
            true,
            lance_io::object_store::DEFAULT_CLOUD_IO_PARALLELISM,
            lance_io::object_store::DEFAULT_DOWNLOAD_RETRY_COUNT,
            None,
        ));
        let manifest_meta = object_store.inner.head(&manifest_path).await?;
        let manifest = read_manifest(
            object_store.as_ref(),
            &manifest_path,
            Some(manifest_meta.size),
        )
        .await
        .map_err(Error::from)?;
        let scan_scheduler = ScanScheduler::new(
            object_store.clone(),
            SchedulerConfig::max_bandwidth(&object_store),
        );
        let metadata_cache = match options.cache_bytes {
            Some(capacity) if capacity > 0 => LanceCache::with_capacity(capacity),
            _ => LanceCache::no_cache(),
        };

        Ok(Self {
            manifest,
            object_store,
            scan_scheduler,
            table_path,
            metadata_cache,
        })
    }

    pub fn schema(&self) -> &LanceSchema {
        &self.manifest.schema
    }

    pub fn version(&self) -> u64 {
        self.manifest.version
    }

    pub async fn search_batches(&self, request: SearchRequest) -> Result<Vec<RecordBatch>> {
        let vector = request.vector.ok_or_else(|| Error::NotSupported {
            message: "the browser path currently supports vector search only".to_string(),
        })?;
        if request.text.is_some() {
            return Err(Error::NotSupported {
                message: "full-text and hybrid search are not yet supported in the browser path"
                    .to_string(),
            });
        }
        if request.filter.is_some() {
            return Err(Error::NotSupported {
                message: "filter pushdown is not yet supported in the browser path".to_string(),
            });
        }
        if matches!(request.select, Some(SelectRequest::Dynamic(_))) {
            return Err(Error::NotSupported {
                message: "dynamic select expressions are not yet supported in the browser path"
                    .to_string(),
            });
        }

        let vector_column =
            request
                .vector_column
                .as_deref()
                .ok_or_else(|| Error::InvalidInput {
                    message: "vector searches require a vectorColumn".to_string(),
                })?;
        let projection = ProjectionPlan::new(
            &self.manifest.schema,
            request.select.as_ref(),
            vector_column,
            request.with_row_id.unwrap_or(false),
        )?;

        let offset = request.offset.unwrap_or(0);
        let limit = request.limit.unwrap_or(DEFAULT_VECTOR_LIMIT);
        let top_k = limit.saturating_add(offset);
        if top_k == 0 {
            return Ok(vec![]);
        }

        let mut heap = BinaryHeap::with_capacity(top_k);
        let mut ordinal = 0_u64;

        for fragment in self.manifest.fragments.iter() {
            let batches = self
                .read_fragment_batches(
                    fragment,
                    &projection.scan_projection,
                    projection.with_row_id,
                )
                .await?;
            for scan_batch in batches {
                let vector_index = scan_batch
                    .schema()
                    .index_of(vector_column)
                    .map_err(Error::from)?;
                let output_batch = scan_batch
                    .project(&projection.output_indices_for(scan_batch.schema().as_ref())?)
                    .map_err(Error::from)?;
                accumulate_batch_candidates(
                    &scan_batch,
                    &output_batch,
                    vector_index,
                    &vector,
                    &mut heap,
                    &mut ordinal,
                    top_k,
                )?;
            }
        }

        let mut rows = heap.into_vec();
        rows.sort_by(|left, right| {
            left.distance
                .total_cmp(&right.distance)
                .then_with(|| left.ordinal.cmp(&right.ordinal))
        });

        let output_rows = rows
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(|candidate| candidate.into_batch())
            .collect::<Result<Vec<_>>>()?;
        Ok(output_rows)
    }

    pub fn result_schema(&self, request: &SearchRequest) -> Result<ArrowSchema> {
        let vector_column =
            request
                .vector_column
                .as_deref()
                .ok_or_else(|| Error::InvalidInput {
                    message: "vector searches require a vectorColumn".to_string(),
                })?;
        let projection = ProjectionPlan::new(
            &self.manifest.schema,
            request.select.as_ref(),
            vector_column,
            request.with_row_id.unwrap_or(false),
        )?;
        Ok(with_distance_schema(&ArrowSchema::from(
            &projection.output_projection,
        )))
    }

    async fn read_fragment_batches(
        &self,
        fragment: &Fragment,
        projection: &LanceSchema,
        with_row_id: bool,
    ) -> Result<Vec<RecordBatch>> {
        let (streams, num_rows) = self.open_fragment_streams(fragment, projection).await?;
        if streams.is_empty() {
            return Ok(vec![]);
        }

        let deletion_vector = match fragment.deletion_file.as_ref() {
            Some(deletion_file) => Some(Arc::new(
                read_deletion_file(
                    fragment.id,
                    deletion_file,
                    &self.table_path,
                    self.object_store.as_ref(),
                )
                .await
                .map_err(Error::from)?,
            )),
            None => None,
        };
        let row_id_sequence = if with_row_id {
            load_row_id_sequence(fragment, self.object_store.as_ref(), &self.table_path).await?
        } else {
            None
        };

        let data = if streams.len() == 1 {
            streams.into_iter().next().unwrap()
        } else {
            merge_streams(streams)
        };
        let config = RowIdAndDeletesConfig {
            params: ReadBatchParams::RangeFull,
            with_row_id,
            with_row_addr: false,
            with_row_last_updated_at_version: false,
            with_row_created_at_version: false,
            deletion_vector,
            row_id_sequence,
            last_updated_at_sequence: None,
            created_at_sequence: None,
            make_deletions_null: false,
            total_num_rows: num_rows,
        };

        wrap_with_row_id_and_delete(data, fragment.id as u32, config)
            .buffered(4)
            .try_collect::<Vec<_>>()
            .await
            .map_err(Error::from)
    }

    async fn open_fragment_streams(
        &self,
        fragment: &Fragment,
        projection: &LanceSchema,
    ) -> Result<(Vec<ReadBatchTaskStream>, u32)> {
        let mut streams = Vec::new();
        let mut field_ids_in_files = HashSet::new();
        let mut num_rows = fragment.physical_rows.map(|rows| rows as u32);

        for data_file in &fragment.files {
            if data_file.base_id.is_some() {
                return Err(Error::NotSupported {
                    message: "external data file bases are not yet supported in the browser path"
                        .to_string(),
                });
            }
            if data_file.is_legacy_file() {
                return Err(Error::NotSupported {
                    message: "legacy Lance data files are not yet supported in the browser path"
                        .to_string(),
                });
            }

            let data_file_schema = data_file.schema(&self.manifest.schema);
            let schema_per_file = Arc::new(
                projection
                    .intersection_ignore_types(&data_file_schema)
                    .map_err(Error::from)?,
            );
            if schema_per_file.fields.is_empty() {
                continue;
            }
            field_ids_in_files.extend(
                schema_per_file
                    .field_ids()
                    .into_iter()
                    .filter(|field_id| *field_id >= 0),
            );

            let (stream, file_num_rows) = self
                .open_data_file_stream(data_file, schema_per_file)
                .await?;
            match num_rows {
                Some(existing) if existing != file_num_rows => {
                    return Err(Error::Runtime {
                        message: format!(
                            "fragment {} contained inconsistent file row counts ({existing} vs {file_num_rows})",
                            fragment.id
                        ),
                    });
                }
                None => num_rows = Some(file_num_rows),
                _ => {}
            }
            streams.push(stream);
        }

        let num_rows = num_rows.ok_or_else(|| Error::Runtime {
            message: format!(
                "fragment {} did not advertise a physical row count",
                fragment.id
            ),
        })?;

        let mut missing_fields = projection.field_ids();
        missing_fields.retain(|field_id| *field_id >= 0 && !field_ids_in_files.contains(field_id));
        if !missing_fields.is_empty() {
            let missing_projection = Arc::new(projection.project_by_ids(&missing_fields, true));
            streams.push(null_task_stream(
                missing_projection,
                num_rows,
                DEFAULT_BATCH_SIZE,
            ));
        }

        Ok((streams, num_rows))
    }

    async fn open_data_file_stream(
        &self,
        data_file: &DataFile,
        projection: Arc<LanceSchema>,
    ) -> Result<(ReadBatchTaskStream, u32)> {
        let path = self.table_path.child(data_file.path.as_str());
        let file_scheduler = self
            .scan_scheduler
            .open_file(&path, &data_file.file_size_bytes)
            .await
            .map_err(Error::from)?;
        let reader = FileReader::try_open(
            file_scheduler.clone(),
            None,
            Arc::default(),
            &self.metadata_cache,
            FileReaderOptions::default(),
        )
        .await
        .map_err(Error::from)?;
        let row_count = reader.metadata().num_rows as u32;

        let field_id_to_column_index = BTreeMap::from_iter(
            data_file
                .fields
                .iter()
                .copied()
                .zip(data_file.column_indices.iter().copied())
                .filter_map(|(field_id, column_index)| {
                    (column_index >= 0).then_some((field_id as u32, column_index as u32))
                }),
        );
        let reader_projection = ReaderProjection::from_field_ids(
            reader.metadata().version(),
            projection.as_ref(),
            &field_id_to_column_index,
        )
        .map_err(Error::from)?;
        let stream = reader
            .read_tasks(
                ReadBatchParams::RangeFull,
                DEFAULT_BATCH_SIZE,
                Some(reader_projection),
                FilterExpression::no_filter(),
            )
            .map_err(Error::from)?
            .map(|task| ReadBatchTask {
                task: task.task,
                num_rows: task.num_rows,
            })
            .boxed();

        Ok((stream, row_count))
    }
}

struct ProjectionPlan {
    output_projection: LanceSchema,
    scan_projection: LanceSchema,
    with_row_id: bool,
}

impl ProjectionPlan {
    fn new(
        full_schema: &LanceSchema,
        select: Option<&SelectRequest>,
        vector_column: &str,
        with_row_id: bool,
    ) -> Result<Self> {
        let requested_columns = match select {
            Some(SelectRequest::Columns(columns)) => columns.clone(),
            Some(SelectRequest::Dynamic(_)) => {
                return Err(Error::NotSupported {
                    message: "dynamic select expressions are not yet supported in the browser path"
                        .to_string(),
                });
            }
            None => full_schema
                .fields
                .iter()
                .map(|field| field.name.clone())
                .collect(),
        };

        let mut output_columns = requested_columns;
        if with_row_id && !output_columns.iter().any(|column| column == ROW_ID) {
            output_columns.push(ROW_ID.to_string());
        }

        let output_projection = full_schema
            .project_preserve_system_columns(&output_columns)
            .map_err(Error::from)?;
        let mut scan_columns = output_columns;
        if !scan_columns.iter().any(|column| column == vector_column) {
            scan_columns.push(vector_column.to_string());
        }
        let scan_projection = full_schema
            .project_preserve_system_columns(&scan_columns)
            .map_err(Error::from)?;

        Ok(Self {
            output_projection,
            scan_projection,
            with_row_id,
        })
    }

    fn output_indices_for(&self, schema: &ArrowSchema) -> Result<Vec<usize>> {
        self.output_projection
            .fields
            .iter()
            .map(|field| schema.index_of(&field.name).map_err(Error::from))
            .collect()
    }
}

struct CandidateRow {
    distance: f32,
    ordinal: u64,
    row: RecordBatch,
}

impl CandidateRow {
    fn into_batch(self) -> Result<RecordBatch> {
        let schema = with_distance_schema(self.row.schema().as_ref());
        let mut columns = self.row.columns().to_vec();
        columns.push(Arc::new(Float32Array::from(vec![self.distance])));
        RecordBatch::try_new(Arc::new(schema), columns).map_err(Error::from)
    }
}

impl PartialEq for CandidateRow {
    fn eq(&self, other: &Self) -> bool {
        self.distance.total_cmp(&other.distance) == Ordering::Equal && self.ordinal == other.ordinal
    }
}

impl Eq for CandidateRow {}

impl PartialOrd for CandidateRow {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CandidateRow {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.ordinal.cmp(&other.ordinal))
    }
}

fn accumulate_batch_candidates(
    scan_batch: &RecordBatch,
    output_batch: &RecordBatch,
    vector_index: usize,
    query_vector: &[f32],
    heap: &mut BinaryHeap<CandidateRow>,
    ordinal: &mut u64,
    top_k: usize,
) -> Result<()> {
    let vectors = scan_batch
        .column(vector_index)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| Error::NotSupported {
            message:
                "the browser path currently supports FixedSizeList<Float32> vector columns only"
                    .to_string(),
        })?;
    let values = vectors
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| Error::NotSupported {
            message: "the browser path currently supports Float32 vector columns only".to_string(),
        })?;
    let dimension = vectors.value_length() as usize;
    if dimension != query_vector.len() {
        return Err(Error::InvalidInput {
            message: format!(
                "query vector has dimension {}, but column '{}' has dimension {}",
                query_vector.len(),
                scan_batch.schema().field(vector_index).name(),
                dimension
            ),
        });
    }

    for row_index in 0..vectors.len() {
        if vectors.is_null(row_index) {
            continue;
        }
        let distance = squared_l2_distance(values, query_vector, row_index, dimension)?;
        let replace = heap.len() < top_k
            || heap
                .peek()
                .is_some_and(|current| current.distance.total_cmp(&distance) == Ordering::Greater);
        if !replace {
            *ordinal += 1;
            continue;
        }

        let row = output_batch.slice(row_index, 1);
        if heap.len() == top_k {
            heap.pop();
        }
        heap.push(CandidateRow {
            distance,
            ordinal: *ordinal,
            row,
        });
        *ordinal += 1;
    }

    Ok(())
}

fn squared_l2_distance(
    values: &Float32Array,
    query_vector: &[f32],
    row_index: usize,
    dimension: usize,
) -> Result<f32> {
    let base_offset = row_index * dimension;
    let mut sum = 0.0_f32;
    for (index, expected) in query_vector.iter().enumerate() {
        let value_index = base_offset + index;
        if values.is_null(value_index) {
            return Err(Error::NotSupported {
                message: "vector rows with null elements are not yet supported in the browser path"
                    .to_string(),
            });
        }
        let delta = values.value(value_index) - expected;
        sum += delta * delta;
    }
    Ok(sum)
}

fn with_distance_schema(schema: &ArrowSchema) -> ArrowSchema {
    let mut fields = schema.fields().iter().cloned().collect::<Vec<_>>();
    fields.push(Arc::new(ArrowField::new(
        RESULT_DISTANCE_COLUMN,
        DataType::Float32,
        false,
    )));
    ArrowSchema::new_with_metadata(fields, schema.metadata().clone())
}

fn null_task_stream(
    projection: Arc<LanceSchema>,
    num_rows: u32,
    batch_size: u32,
) -> ReadBatchTaskStream {
    let schema = Arc::new(ArrowSchema::from(projection.as_ref()));
    let mut remaining_rows = num_rows as usize;

    let tasks = std::iter::from_fn(move || {
        if remaining_rows == 0 {
            return None;
        }

        let this_batch_size = remaining_rows.min(batch_size as usize);
        remaining_rows -= this_batch_size;
        let columns = schema
            .fields()
            .iter()
            .map(|field| arrow_array::new_null_array(field.data_type(), this_batch_size))
            .collect::<Vec<_>>();
        let batch = RecordBatch::try_new_with_options(
            schema.clone(),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(this_batch_size)),
        )
        .expect("null batch construction should succeed");

        Some(ReadBatchTask {
            task: futures::future::ready(Ok(batch)).boxed(),
            num_rows: this_batch_size as u32,
        })
    })
    .collect::<Vec<_>>();
    futures::stream::iter(tasks).boxed()
}

async fn load_row_id_sequence(
    fragment: &Fragment,
    object_store: &ObjectStore,
    table_path: &Path,
) -> Result<Option<Arc<RowIdSequence>>> {
    match &fragment.row_id_meta {
        None => Ok(None),
        Some(RowIdMeta::Inline(data)) => {
            Ok(Some(Arc::new(read_row_ids(data).map_err(Error::from)?)))
        }
        Some(RowIdMeta::External(file_slice)) => {
            let path = table_path.child(file_slice.path.as_str());
            let range = file_slice.offset as usize..(file_slice.offset + file_slice.size) as usize;
            let bytes = object_store
                .open(&path)
                .await
                .map_err(Error::from)?
                .get_range(range)
                .await
                .map_err(Error::from)?;
            Ok(Some(Arc::new(read_row_ids(&bytes).map_err(Error::from)?)))
        }
    }
}
