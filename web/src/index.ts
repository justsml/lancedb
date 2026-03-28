import { Schema, Table as ArrowTable, tableFromIPC } from "apache-arrow";
import type {
  WasmModule,
  WasmRemoteSearchHandle,
} from "./generated/lancedb_wasm.js";

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
  latestVersionUrl?: string;
  snapshotUrl?: string;
  webMetadataUrl?: string;
};

interface PublishedTableMetadata {
  version: number;
  manifestPath: string;
  manifestSizeBytes?: number;
  manifestNamingScheme: string;
  latestManifestPath: string;
  latestVersionPath: string;
  webMetadataPath: string;
  snapshotPath: string;
  defaultVectorColumn?: string;
  vectorColumns: string[];
  ftsColumns: string[];
}

interface PublishedSnapshot {
  version: number;
  manifestPath: string;
  manifestSizeBytes?: number;
  manifestNamingScheme: string;
  latestManifestPath: string;
  latestVersionPath: string;
  webMetadataPath: string;
  snapshotPath: string;
  defaultVectorColumn?: string;
  vectorColumns: string[];
  ftsColumns: string[];
  isComplete: boolean;
}

interface ResolvedPublishedState {
  currentVersion: number | null;
  latestVersionUrl: string;
  manifestUrl: string;
  snapshot: PublishedSnapshot | null;
  snapshotUrl: string;
  tableMetadata: PublishedTableMetadata | null;
  webMetadataUrl: string;
}

let wasmModuleLoader: () => Promise<WasmModule> = async () =>
  (await import("./generated/lancedb_wasm.js")) as WasmModule;

export function __setWasmModuleLoaderForTests(
  loader?: () => Promise<WasmModule>,
): void {
  wasmModuleLoader =
    loader ??
    (async () => (await import("./generated/lancedb_wasm.js")) as WasmModule);
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
  #published: ResolvedPublishedState;
  #wasmModule: WasmModule;

  private constructor(
    tableUrl: string,
    options: OpenTableOptions,
    headersSignature: string,
    handle: WasmRemoteSearchHandle,
    published: ResolvedPublishedState,
    wasmModule: WasmModule,
  ) {
    this.#tableUrl = tableUrl;
    this.#options = options;
    this.#headersSignature = headersSignature;
    this.#handle = handle;
    this.#published = published;
    this.#wasmModule = wasmModule;
  }

  static async open(
    tableUrl: string,
    options: OpenTableOptions,
  ): Promise<RemoteSearchTableImpl> {
    const normalizedTableUrl = normalizeHttpUrl(tableUrl, "table");
    const headers = await resolveHeaders(options.headers);
    const published = await resolvePublishedState(
      normalizedTableUrl,
      headers,
      options,
    );
    await preflightOpen(published.manifestUrl, headers, options);
    const wasmModule = await wasmModuleLoader();
    const handle = await wasmModule.open_table(
      normalizedTableUrl,
      JSON.stringify(makeOpenOptionsPayload(options, headers, published)),
    );

    return new RemoteSearchTableImpl(
      normalizedTableUrl,
      options,
      stableHeaderSignature(headers),
      handle,
      published,
      wasmModule,
    );
  }

  async schema(): Promise<Schema> {
    await this.#ensureHandle();
    return decodeSchema(await this.#handle!.schema());
  }

  async search(request: SearchRequest): Promise<ArrowTable> {
    await this.#ensureHandle();
    const normalizedRequest = normalizeSearchRequest(request, this.#published);
    const bytes = await this.#handle!.search(
      JSON.stringify(normalizedRequest),
    );
    return tableFromIPC(bytes);
  }

  async refresh(): Promise<boolean> {
    await this.#ensureHandle();
    const fetchFn = this.#options.fetch ?? globalThis.fetch;
    if (
      fetchFn === undefined ||
      this.#published.currentVersion === null ||
      this.#published.latestVersionUrl.length === 0
    ) {
      const changed = await this.#handle!.refresh();
      if (changed && fetchFn !== undefined) {
        this.#published = await resolvePublishedState(
          this.#tableUrl,
          await resolveHeaders(this.#options.headers),
          this.#options,
        );
      }
      return changed;
    }

    const latestVersion = await fetchLatestVersion(
      this.#published.latestVersionUrl,
      await resolveHeaders(this.#options.headers),
      this.#options,
    );
    if (latestVersion === this.#published.currentVersion) {
      return false;
    }

    await this.#reopenHandle();
    return true;
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

    await this.#reopenHandle(headers, nextSignature);
  }

  async #reopenHandle(
    headers?: Record<string, string>,
    headersSignature?: string,
  ): Promise<void> {
    const resolvedHeaders = headers ?? (await resolveHeaders(this.#options.headers));
    const nextSignature =
      headersSignature ?? stableHeaderSignature(resolvedHeaders);
    const published = await resolvePublishedState(
      this.#tableUrl,
      resolvedHeaders,
      this.#options,
    );
    await preflightOpen(published.manifestUrl, resolvedHeaders, this.#options);
    this.#handle!.close();
    this.#handle = await this.#wasmModule.open_table(
      this.#tableUrl,
      JSON.stringify(
        makeOpenOptionsPayload(this.#options, resolvedHeaders, published),
      ),
    );
    this.#headersSignature = nextSignature;
    this.#published = published;
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
  published: ResolvedPublishedState,
): OpenOptionsPayload {
  return {
    headers,
    cacheBytes: options.cacheBytes,
    maxConcurrentRanges: options.maxConcurrentRanges,
    manifestUrl: published.manifestUrl,
    latestVersionUrl: published.latestVersionUrl,
    snapshotUrl: published.snapshotUrl,
    webMetadataUrl: published.webMetadataUrl,
  };
}

async function preflightOpen(
  manifestUrl: string,
  headers: Record<string, string>,
  options: OpenTableOptions,
): Promise<void> {
  const fetchFn = options.fetch ?? globalThis.fetch;
  if (fetchFn === undefined) {
    throw new Error(
      "No fetch implementation is available. Pass OpenTableOptions.fetch when running outside browsers and modern edge runtimes.",
    );
  }

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

async function resolvePublishedState(
  tableUrl: string,
  headers: Record<string, string>,
  options: OpenTableOptions,
): Promise<ResolvedPublishedState> {
  const webMetadataUrl = resolvePublishedUrl(tableUrl, "_web.json");
  const tableMetadata = await fetchJsonIfExists<PublishedTableMetadata>(
    webMetadataUrl,
    headers,
    options,
  );

  const snapshotUrl = resolvePublishedUrl(
    tableUrl,
    tableMetadata?.snapshotPath ?? "_snapshot.json",
  );
  const snapshot = await fetchJsonIfExists<PublishedSnapshot>(
    snapshotUrl,
    headers,
    options,
  );

  return {
    currentVersion: tableMetadata?.version ?? snapshot?.version ?? null,
    latestVersionUrl: resolvePublishedUrl(
      tableUrl,
      tableMetadata?.latestVersionPath ?? snapshot?.latestVersionPath ?? "_latest.version",
    ),
    manifestUrl:
      options.manifestUrl ??
      resolvePublishedUrl(
        tableUrl,
        tableMetadata?.latestManifestPath ??
          snapshot?.latestManifestPath ??
          "_latest.manifest",
      ),
    snapshot,
    snapshotUrl,
    tableMetadata,
    webMetadataUrl,
  };
}

async function fetchJsonIfExists<T>(
  url: string,
  headers: Record<string, string>,
  options: OpenTableOptions,
): Promise<T | null> {
  const fetchFn = options.fetch ?? globalThis.fetch;
  if (fetchFn === undefined) {
    return null;
  }

  const response = await fetchFn(url, {
    method: "GET",
    headers,
  });

  if (response.status === 404) {
    return null;
  }

  if (!response.ok) {
    throw new Error(
      `Failed to fetch published table metadata from ${url}. Received status ${response.status}.`,
    );
  }

  try {
    return (await response.json()) as T;
  } catch (error) {
    throw new Error(
      `Failed to parse published table metadata from ${url}: ${(error as Error).message}`,
    );
  }
}

async function fetchLatestVersion(
  latestVersionUrl: string,
  headers: Record<string, string>,
  options: OpenTableOptions,
): Promise<number> {
  const fetchFn = options.fetch ?? globalThis.fetch;
  if (fetchFn === undefined) {
    throw new Error("No fetch implementation is available for refresh.");
  }

  const response = await fetchFn(latestVersionUrl, {
    method: "GET",
    headers,
  });
  if (!response.ok) {
    throw new Error(
      `Failed to fetch latest table version from ${latestVersionUrl}. Received status ${response.status}.`,
    );
  }

  const rawVersion = (await response.text()).trim();
  const version = Number.parseInt(rawVersion, 10);
  if (!Number.isFinite(version)) {
    throw new Error(
      `Invalid latest table version "${rawVersion}" returned from ${latestVersionUrl}.`,
    );
  }
  return version;
}

function resolvePublishedUrl(tableUrl: string, pathOrUrl: string): string {
  try {
    return new URL(pathOrUrl).toString();
  } catch {
    return new URL(pathOrUrl, tableUrl).toString();
  }
}

function normalizeSearchRequest(
  request: SearchRequest,
  published: ResolvedPublishedState,
): Record<string, unknown> {
  const metadata = published.tableMetadata ?? published.snapshot;
  if (request.text !== undefined && metadata !== null && metadata.ftsColumns.length === 0) {
    throw new Error(
      "This table does not advertise any full-text search indexed columns in its published metadata.",
    );
  }

  const vectorColumn =
    request.vectorColumn ??
    (request.vector !== undefined ? metadata?.defaultVectorColumn : undefined);

  return {
    ...request,
    vectorColumn,
    vector:
      request.vector instanceof Float32Array
        ? Array.from(request.vector)
        : request.vector,
  };
}

function decodeSchema(bytes: Uint8Array): Schema {
  return tableFromIPC(bytes).schema;
}
