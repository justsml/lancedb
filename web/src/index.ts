import { Schema, Table as ArrowTable, tableFromIPC } from "apache-arrow";
import type {
  WasmModule,
  WasmRemoteSearchHandle,
} from "./generated/lancedb_wasm.js";
import type {
  WorkerRequest,
  WorkerResponse,
  WorkerSuccessResponse,
} from "./worker_protocol.js";

export interface OpenTableOptions {
  fetch?: typeof globalThis.fetch;
  headers?: HeaderProvider;
  cacheBytes?: number;
  maxConcurrentRanges?: number;
  manifestUrl?: string;
  noWorker?: boolean;
  workerInitTimeoutMs?: number;
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

type OpenOptionsPayload = Omit<
  OpenTableOptions,
  "fetch" | "headers" | "noWorker" | "workerInitTimeoutMs"
> & {
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

interface WorkerLike {
  addEventListener(
    type: "message",
    listener: (event: MessageEvent<WorkerResponse>) => void,
  ): void;
  addEventListener(type: "error", listener: (event: ErrorEvent) => void): void;
  removeEventListener(
    type: "message",
    listener: (event: MessageEvent<WorkerResponse>) => void,
  ): void;
  removeEventListener(type: "error", listener: (event: ErrorEvent) => void): void;
  postMessage(message: WorkerRequest): void;
  terminate(): void;
}

interface HandleBackend {
  schema(): Promise<Uint8Array>;
  search(requestJson: string): Promise<Uint8Array>;
  refresh(): Promise<boolean>;
  close(): void;
}

type WorkerFactory = () => Promise<WorkerLike> | WorkerLike;

const DEFAULT_WORKER_INIT_TIMEOUT_MS = 10_000;

let wasmModuleLoader: () => Promise<WasmModule> = async () =>
  (await import("./generated/lancedb_wasm.js")) as WasmModule;
let workerFactoryOverride: WorkerFactory | null = null;

class DirectHandleBackend implements HandleBackend {
  static async open(
    tableUrl: string,
    optionsJson?: string,
  ): Promise<DirectHandleBackend> {
    const wasmModule = await wasmModuleLoader();
    return new DirectHandleBackend(await wasmModule.open_table(tableUrl, optionsJson));
  }

  readonly #handle: WasmRemoteSearchHandle;

  private constructor(handle: WasmRemoteSearchHandle) {
    this.#handle = handle;
  }

  schema(): Promise<Uint8Array> {
    return this.#handle.schema();
  }

  search(requestJson: string): Promise<Uint8Array> {
    return this.#handle.search(requestJson);
  }

  refresh(): Promise<boolean> {
    return this.#handle.refresh();
  }

  close(): void {
    this.#handle.close();
  }
}

class WorkerHandleBackend implements HandleBackend {
  static async open(
    tableUrl: string,
    optionsJson: string | undefined,
    timeoutMs: number,
  ): Promise<WorkerHandleBackend> {
    const worker = await createWorker();
    if (worker === null) {
      throw new Error("Workers are not available in this runtime.");
    }

    const backend = new WorkerHandleBackend(worker);
    try {
      const response = await backend.#request(
        {
          id: 0,
          type: "open",
          tableUrl,
          optionsJson,
        },
        timeoutMs,
      );
      if (response.type !== "open") {
        throw new Error(`Unexpected worker response type "${response.type}"`);
      }
      return backend;
    } catch (error) {
      backend.close();
      throw error;
    }
  }

  readonly #worker: WorkerLike;
  readonly #pending = new Map<
    number,
    {
      timeout: ReturnType<typeof setTimeout>;
      resolve: (response: WorkerSuccessResponse) => void;
      reject: (reason?: unknown) => void;
    }
  >();
  readonly #onMessage: (event: MessageEvent<WorkerResponse>) => void;
  readonly #onError: (event: ErrorEvent) => void;
  #closed = false;
  #nextRequestId = 1;

  private constructor(worker: WorkerLike) {
    this.#worker = worker;
    this.#onMessage = (event) => {
      const response = event.data;
      const pending = this.#pending.get(response.id);
      if (pending === undefined) {
        return;
      }

      clearTimeout(pending.timeout);
      this.#pending.delete(response.id);
      if (!response.ok) {
        pending.reject(new Error(response.error));
        return;
      }
      pending.resolve(response);
    };
    this.#onError = (event) => {
      this.#rejectPending(
        new Error(
          event.message.length > 0
            ? event.message
            : "Remote search worker failed unexpectedly.",
        ),
      );
    };

    this.#worker.addEventListener("message", this.#onMessage);
    this.#worker.addEventListener("error", this.#onError);
  }

  async schema(): Promise<Uint8Array> {
    const response = await this.#request({
      id: this.#nextId(),
      type: "schema",
    });
    if (response.type !== "schema") {
      throw new Error(`Unexpected worker response type "${response.type}"`);
    }
    return new Uint8Array(response.bytes);
  }

  async search(requestJson: string): Promise<Uint8Array> {
    const response = await this.#request({
      id: this.#nextId(),
      type: "search",
      requestJson,
    });
    if (response.type !== "search") {
      throw new Error(`Unexpected worker response type "${response.type}"`);
    }
    return new Uint8Array(response.bytes);
  }

  async refresh(): Promise<boolean> {
    const response = await this.#request({
      id: this.#nextId(),
      type: "refresh",
    });
    if (response.type !== "refresh") {
      throw new Error(`Unexpected worker response type "${response.type}"`);
    }
    return response.changed;
  }

  close(): void {
    if (this.#closed) {
      return;
    }

    this.#closed = true;
    this.#worker.removeEventListener("message", this.#onMessage);
    this.#worker.removeEventListener("error", this.#onError);
    this.#rejectPending(new Error("Remote search worker was closed."));
    this.#worker.terminate();
  }

  #nextId(): number {
    return this.#nextRequestId++;
  }

  async #request(
    request: WorkerRequest,
    timeoutMs = DEFAULT_WORKER_INIT_TIMEOUT_MS,
  ): Promise<WorkerSuccessResponse> {
    if (this.#closed) {
      throw new Error("Remote search worker is closed");
    }

    return await new Promise<WorkerSuccessResponse>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.#pending.delete(request.id);
        reject(
          new Error(
            `Remote search worker request "${request.type}" timed out after ${timeoutMs}ms.`,
          ),
        );
      }, timeoutMs);

      this.#pending.set(request.id, { timeout, resolve, reject });
      this.#worker.postMessage(request);
    });
  }

  #rejectPending(error: Error): void {
    for (const { timeout, reject } of this.#pending.values()) {
      clearTimeout(timeout);
      reject(error);
    }
    this.#pending.clear();
  }
}

export function __setWasmModuleLoaderForTests(
  loader?: () => Promise<WasmModule>,
): void {
  wasmModuleLoader =
    loader ??
    (async () => (await import("./generated/lancedb_wasm.js")) as WasmModule);
}

export function __setWorkerFactoryForTests(
  factory?: WorkerFactory | null,
): void {
  workerFactoryOverride = factory ?? null;
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
  #handle: HandleBackend | null;
  #published: ResolvedPublishedState;

  private constructor(
    tableUrl: string,
    options: OpenTableOptions,
    headersSignature: string,
    handle: HandleBackend,
    published: ResolvedPublishedState,
  ) {
    this.#tableUrl = tableUrl;
    this.#options = options;
    this.#headersSignature = headersSignature;
    this.#handle = handle;
    this.#published = published;
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
    const handle = await openHandleBackend(
      normalizedTableUrl,
      JSON.stringify(makeOpenOptionsPayload(options, headers, published)),
      options,
    );

    return new RemoteSearchTableImpl(
      normalizedTableUrl,
      options,
      stableHeaderSignature(headers),
      handle,
      published,
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
    this.#handle = await openHandleBackend(
      this.#tableUrl,
      JSON.stringify(
        makeOpenOptionsPayload(this.#options, resolvedHeaders, published),
      ),
      this.#options,
    );
    this.#headersSignature = nextSignature;
    this.#published = published;
  }
}

async function openHandleBackend(
  tableUrl: string,
  optionsJson: string | undefined,
  options: OpenTableOptions,
): Promise<HandleBackend> {
  if (!options.noWorker) {
    try {
      return await WorkerHandleBackend.open(
        tableUrl,
        optionsJson,
        options.workerInitTimeoutMs ?? DEFAULT_WORKER_INIT_TIMEOUT_MS,
      );
    } catch {
      // Fall through to the direct runtime if the worker path is unavailable.
    }
  }

  return await DirectHandleBackend.open(tableUrl, optionsJson);
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

async function createWorker(): Promise<WorkerLike | null> {
  if (workerFactoryOverride !== null) {
    return await workerFactoryOverride();
  }

  if (typeof Worker !== "function") {
    return null;
  }

  const factoryModule = (await import("./worker_factory.js")) as {
    createDefaultWorker(): WorkerLike;
  };
  return factoryModule.createDefaultWorker();
}
