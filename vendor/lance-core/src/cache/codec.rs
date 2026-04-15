// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Serialization codecs for cache entries.
//!
//! Implement [`CacheCodecImpl`] on concrete types, then use
//! [`CacheCodec::from_impl`] to produce a type-erased codec for the cache.

use std::sync::Arc;

use bytes::Bytes;

use crate::Result;

// ---------------------------------------------------------------------------
// CacheCodecImpl — trait for serializable cache entry types
// ---------------------------------------------------------------------------

/// Serialization trait for cache entries.
pub trait CacheCodecImpl: Send + Sync {
    fn serialize(&self, writer: &mut dyn std::io::Write) -> Result<()>;

    fn deserialize(data: &Bytes) -> Result<Self>
    where
        Self: Sized;
}

// ---------------------------------------------------------------------------
// CacheCodec — type-erased codec passed to backends
// ---------------------------------------------------------------------------

pub(crate) type ArcAny = Arc<dyn std::any::Any + Send + Sync>;

/// Type-erased codec for serializing and deserializing cache entries.
///
/// `CacheCodec` is two plain function pointers — it is `Copy` and has no
/// heap allocation.
#[derive(Copy, Clone)]
pub struct CacheCodec {
    pub(crate) serialize: fn(&ArcAny, &mut dyn std::io::Write) -> Result<()>,
    pub(crate) deserialize: fn(&Bytes) -> Result<ArcAny>,
}

impl std::fmt::Debug for CacheCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheCodec").finish_non_exhaustive()
    }
}

fn serialize_via_impl<T: CacheCodecImpl + 'static>(
    any: &ArcAny,
    writer: &mut dyn std::io::Write,
) -> Result<()> {
    let val = any
        .downcast_ref::<T>()
        .expect("CacheCodec::serialize called with wrong type (this is a bug in the cache layer)");
    val.serialize(writer)
}

fn deserialize_via_impl<T: CacheCodecImpl + 'static>(data: &Bytes) -> Result<ArcAny> {
    let val = T::deserialize(data)?;
    Ok(Arc::new(val) as ArcAny)
}

impl CacheCodec {
    /// Create a `CacheCodec` from plain function pointers.
    pub fn new(
        serialize: fn(&ArcAny, &mut dyn std::io::Write) -> Result<()>,
        deserialize: fn(&Bytes) -> Result<ArcAny>,
    ) -> Self {
        Self {
            serialize,
            deserialize,
        }
    }

    /// Create a `CacheCodec` from a [`CacheCodecImpl`] implementation.
    pub fn from_impl<T: CacheCodecImpl + 'static>() -> Self {
        Self {
            serialize: serialize_via_impl::<T>,
            deserialize: deserialize_via_impl::<T>,
        }
    }

    pub fn serialize(&self, value: &ArcAny, writer: &mut dyn std::io::Write) -> Result<()> {
        (self.serialize)(value, writer)
    }

    pub fn deserialize(&self, data: &Bytes) -> Result<ArcAny> {
        (self.deserialize)(data)
    }
}
