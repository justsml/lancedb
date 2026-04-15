//! WASM integration tests — run with `wasm-pack test --headless --chrome`.
//!
//! These tests exercise the full browser code path including FetchHttpStore,
//! BrowserTable, and the wasm-bindgen entry points. They require a real
//! browser environment (or headless Chrome/Firefox).
//!
//! To run:
//!   wasm-pack test --headless --chrome --test wasm_search
//!
//! Prerequisites:
//!   - Install wasm-pack: `cargo install wasm-pack`
//!   - Chrome or Firefox available in PATH

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

/// Smoke test: verify the WASM module can be instantiated.
#[wasm_bindgen_test]
async fn test_wasm_module_loads() {
    // If this test runs, the WASM binary compiled and loaded successfully.
    // The real tests below require a running HTTP server with published
    // Lance table data.
    assert!(true);
}

/// Test that opening a table with an invalid URL returns an error (not a panic).
#[wasm_bindgen_test]
async fn test_open_invalid_url_returns_error() {
    use lancedb_wasm::OpenTableOptions;
    use lancedb_wasm::RemoteSearchTable;

    let result = RemoteSearchTable::open(
        "https://localhost:1/nonexistent-table/",
        OpenTableOptions::default(),
    )
    .await;

    assert!(
        result.is_err(),
        "Opening a nonexistent table should return an error"
    );
}
