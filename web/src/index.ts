import { Schema, Table as ArrowTable, tableFromIPC } from "apache-arrow";
import type {
  WasmModule,
  WasmRemoteSearchHandle,
} from "./generated/lancedb_wasm";

export interface OpenTableOptions {
  fetch?: typeof globalThis.fetch;
  headers?: HeaderProvider;
  cacheBytes?: number;
  maxConcurrentRanges?: number;
  manifestUrl?: string;
}

export type HeaderProvider =
  | Record<string, string>
  | (() => Promise<Record<string, string>> | Record<string, string>);

export type TextQuery = string | { query: string; columns?: string[] };

export type Selection = string[] | Record<string, string>;

export interface SearchRequest {
  vector?: Float32Array | number[];
  text?: TextQuery;
  filter?: string;
  select?: Selection;
  limit?: number;
  offset?: number;
  vectorColumn?: string;
  prefilter?: boolean;
  withRowId?: boolean;
  fastSearch?: boolean;
}

export interface RemoteSearchTable {
  schema(): Promise<Schema>;
  search(request: SearchRequest): Promise<ArrowTable>;
  refresh(): Promise<boolean>;
  close(): void;
}

type OpenOptionsPayload = Omit<OpenTableOptions, "fetch" | "headers"> & {
  headers?: Record<string, string>;
};

let wasmModuleLoader: () => Promise<WasmModule> = async () =>
  (await import("./generated/lancedb_wasm")) as WasmModule;

export function __setWasmModuleLoaderForTests(
  loader?: () => Promise<WasmModule>,
): void {
  wasmModuleLoader =
    loader ??
    (async () => (await import("./generated/lancedb_wasm")) as WasmModule);
}

export async function openTable(
  tableUrl: string,
  options: OpenTableOptions = {},
): Promise<RemoteSearchTable> {
  return RemoteSearchTableImpl.open(tableUrl, options);
}

class RemoteSearchTableImpl implements RemoteSearchTable {
  #tableUrl: string;
  #options: OpenTableOptions;
  #headersSignature: string;
  #handle: WasmRemoteSearchHandle | null;
  #wasmModule: WasmModule;

  private constructor(
    tableUrl: string,
    options: OpenTableOptions,
    headersSignature: string,
    handle: WasmRemoteSearchHandle,
    wasmModule: WasmModule,
  ) {
    this.#tableUrl = tableUrl;
    this.#options = options;
    this.#headersSignature = headersSignature;
    this.#handle = handle;
    this.#wasmModule = wasmModule;
  }

  static async open(
    tableUrl: string,
    options: OpenTableOptions,
  ): Promise<RemoteSearchTableImpl> {
    const normalizedTableUrl = normalizeHttpUrl(tableUrl, "table");
    const headers = await resolveHeaders(options.headers);
    await preflightOpen(normalizedTableUrl, headers, options);
    const wasmModule = await wasmModuleLoader();
    const handle = await wasmModule.open_table(
      normalizedTableUrl,
      JSON.stringify(makeOpenOptionsPayload(options, headers)),
    );

    return new RemoteSearchTableImpl(
      normalizedTableUrl,
      options,
      stableHeaderSignature(headers),
      handle,
      wasmModule,
    );
  }

  async schema(): Promise<Schema> {
    await this.#ensureHandle();
    return decodeSchema(await this.#handle!.schema());
  }

  async search(request: SearchRequest): Promise<ArrowTable> {
    await this.#ensureHandle();
    const bytes = await this.#handle!.search(
      JSON.stringify(normalizeSearchRequest(request)),
    );
    return tableFromIPC(bytes);
  }

  async refresh(): Promise<boolean> {
    await this.#ensureHandle();
    return this.#handle!.refresh();
  }

  close(): void {
    if (this.#handle === null) {
      return;
    }
    this.#handle.close();
    this.#handle = null;
  }

  async #ensureHandle(): Promise<void> {
    if (this.#handle === null) {
      throw new Error("RemoteSearchTable is closed");
    }

    const headers = await resolveHeaders(this.#options.headers);
    const nextSignature = stableHeaderSignature(headers);
    if (nextSignature === this.#headersSignature) {
      return;
    }

    await preflightOpen(this.#tableUrl, headers, this.#options);
    this.#handle.close();
    this.#handle = await this.#wasmModule.open_table(
      this.#tableUrl,
      JSON.stringify(makeOpenOptionsPayload(this.#options, headers)),
    );
    this.#headersSignature = nextSignature;
  }
}

function normalizeHttpUrl(value: string, label: string): string {
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch (error) {
    throw new Error(`Invalid ${label} URL "${value}": ${(error as Error).message}`);
  }

  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error(
      `Unsupported ${label} URL scheme "${parsed.protocol}". Expected http or https.`,
    );
  }

  if (!parsed.pathname.endsWith("/")) {
    parsed.pathname = `${parsed.pathname}/`;
  }

  return parsed.toString();
}

async function resolveHeaders(
  provider: HeaderProvider | undefined,
): Promise<Record<string, string>> {
  if (provider === undefined) {
    return {};
  }

  return typeof provider === "function" ? await provider() : provider;
}

function stableHeaderSignature(headers: Record<string, string>): string {
  return JSON.stringify(
    Object.entries(headers).sort(([left], [right]) => left.localeCompare(right)),
  );
}

function makeOpenOptionsPayload(
  options: OpenTableOptions,
  headers: Record<string, string>,
): OpenOptionsPayload {
  return {
    headers,
    cacheBytes: options.cacheBytes,
    maxConcurrentRanges: options.maxConcurrentRanges,
    manifestUrl: options.manifestUrl,
  };
}

async function preflightOpen(
  tableUrl: string,
  headers: Record<string, string>,
  options: OpenTableOptions,
): Promise<void> {
  const fetchFn = options.fetch ?? globalThis.fetch;
  if (fetchFn === undefined) {
    throw new Error(
      "No fetch implementation is available. Pass OpenTableOptions.fetch when running outside browsers and modern edge runtimes.",
    );
  }

  const manifestUrl =
    options.manifestUrl ?? new URL("_latest.manifest", tableUrl).toString();
  const response = await fetchFn(manifestUrl, {
    method: "GET",
    headers: {
      ...headers,
      Range: "bytes=0-0",
    },
  });

  if (response.status !== 206) {
    throw new Error(
      `Remote table open requires HTTP range support for ${manifestUrl}. Expected status 206 but received ${response.status}.`,
    );
  }

  const contentRange = response.headers.get("content-range");
  if (contentRange === null) {
    throw new Error(
      `Remote table open requires a Content-Range header for ${manifestUrl}.`,
    );
  }
}

function normalizeSearchRequest(request: SearchRequest): Record<string, unknown> {
  return {
    ...request,
    vector:
      request.vector instanceof Float32Array
        ? Array.from(request.vector)
        : request.vector,
  };
}

function decodeSchema(bytes: Uint8Array): Schema {
  return tableFromIPC(bytes).schema;
}
