export interface WasmRemoteSearchHandle {
  schema(): Promise<Uint8Array>;
  search(requestJson: string): Promise<Uint8Array>;
  refresh(): Promise<boolean>;
  close(): void;
}

export interface WasmModule {
  open_table(
    tableUrl: string,
    optionsJson?: string,
  ): Promise<WasmRemoteSearchHandle>;
}

export async function open_table(): Promise<WasmRemoteSearchHandle> {
  throw new Error(
    "The generated lancedb_wasm artifact is missing. Build the Rust wasm package before using @lancedb/lancedb-web at runtime.",
  );
}
