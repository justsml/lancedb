// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::pin::Pin;

use async_trait::async_trait;
use futures::Future;

use crate::Result;

use super::CacheCodec;
use super::backend::{CacheBackend, CacheEntry, InternalCacheKey};

#[derive(Debug, Default)]
pub struct NoopCacheBackend;

#[async_trait]
impl CacheBackend for NoopCacheBackend {
    async fn get(&self, _key: &InternalCacheKey, _codec: Option<CacheCodec>) -> Option<CacheEntry> {
        None
    }

    async fn insert(
        &self,
        _key: &InternalCacheKey,
        _entry: CacheEntry,
        _size_bytes: usize,
        _codec: Option<CacheCodec>,
    ) {
    }

    async fn get_or_insert<'a>(
        &self,
        _key: &InternalCacheKey,
        loader: Pin<Box<dyn Future<Output = Result<(CacheEntry, usize)>> + Send + 'a>>,
        _codec: Option<CacheCodec>,
    ) -> Result<(CacheEntry, bool)> {
        let (entry, _size_bytes) = loader.await?;
        Ok((entry, false))
    }

    async fn invalidate_prefix(&self, _prefix: &str) {}

    async fn clear(&self) {}

    async fn num_entries(&self) -> usize {
        0
    }

    async fn size_bytes(&self) -> usize {
        0
    }
}
