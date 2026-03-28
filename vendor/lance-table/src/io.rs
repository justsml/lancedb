// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#[cfg(not(target_arch = "wasm32"))]
pub mod commit;
#[cfg(target_arch = "wasm32")]
#[path = "io/commit_wasm.rs"]
pub mod commit;
pub mod deletion;
pub mod manifest;
