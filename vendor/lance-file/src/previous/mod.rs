// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Legacy Lance file v1 implementation kept for backwards compatibility.

pub mod format;
pub mod page_table;
pub mod reader;
#[cfg(not(target_arch = "wasm32"))]
pub mod writer;
