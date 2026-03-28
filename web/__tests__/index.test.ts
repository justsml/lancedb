import { tableFromArrays, tableToIPC } from "apache-arrow";
import {
  __setWasmModuleLoaderForTests,
  __setWorkerFactoryForTests,
  openTable,
  type OpenTableOptions,
} from "../src";
import type { WasmRemoteSearchHandle } from "../src/generated/lancedb_wasm";
import type { WorkerRequest, WorkerResponse } from "../src/worker_protocol";

function makeArrowIpc() {
  return tableToIPC(
    tableFromArrays({
      id: [1, 2],
      doc: ["apple pie", "banana split"],
    }),
  );
}

function makeHandle(): jest.Mocked<WasmRemoteSearchHandle> {
  const bytes = makeArrowIpc();
  return {
    schema: jest.fn(async () => bytes),
    search: jest.fn(async (_requestJson: string) => bytes),
    refresh: jest.fn(async () => true),
    close: jest.fn(),
  };
}

function publishedMetadata(overrides: Partial<Record<string, unknown>> = {}) {
  return {
    version: 7,
    manifestPath: "_versions/7.manifest",
    manifestNamingScheme: "v2",
    latestManifestPath: "_latest.manifest",
    latestVersionPath: "_latest.version",
    webMetadataPath: "_web.json",
    snapshotPath: "_snapshot.json",
    vectorColumns: ["embedding"],
    ftsColumns: ["doc"],
    defaultVectorColumn: "embedding",
    ...overrides,
  };
}

function publishedSnapshot(overrides: Partial<Record<string, unknown>> = {}) {
  return {
    ...publishedMetadata(),
    isComplete: true,
    ...overrides,
  };
}

describe("@lancedb/lancedb-web", () => {
  afterEach(() => {
    __setWasmModuleLoaderForTests();
    __setWorkerFactoryForTests();
    jest.restoreAllMocks();
  });

  it("uses published metadata to resolve sidecars and open the wasm handle", async () => {
    const handle = makeHandle();
    const openTableMock = jest.fn<
      Promise<WasmRemoteSearchHandle>,
      [string, string | undefined]
    >(async () => handle);
    __setWasmModuleLoaderForTests(async () => ({
      open_table: openTableMock,
    }));

    const fetchMock = jest.fn(async (input: RequestInfo | URL) => {
      const url = input.toString();
      if (url.endsWith("/_web.json")) {
        return Response.json(publishedMetadata());
      }
      if (url.endsWith("/_snapshot.json")) {
        return Response.json(publishedSnapshot());
      }
      if (url.endsWith("/_latest.manifest")) {
        return new Response(new Uint8Array([1]), {
          status: 206,
          headers: {
            "content-range": "bytes 0-0/1",
          },
        });
      }
      throw new Error(`unexpected fetch ${url}`);
    });

    await openTable("https://example.com/search_table.lance", {
      fetch: fetchMock as unknown as typeof globalThis.fetch,
      headers: { Authorization: "Bearer token" },
      cacheBytes: 4096,
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "https://example.com/search_table.lance/_web.json",
      expect.objectContaining({
        method: "GET",
        headers: expect.objectContaining({
          Authorization: "Bearer token",
        }),
      }),
    );
    expect(fetchMock).toHaveBeenCalledWith(
      "https://example.com/search_table.lance/_snapshot.json",
      expect.objectContaining({
        method: "GET",
        headers: expect.objectContaining({
          Authorization: "Bearer token",
        }),
      }),
    );
    expect(fetchMock).toHaveBeenCalledWith(
      "https://example.com/search_table.lance/_latest.manifest",
      expect.objectContaining({
        method: "GET",
        headers: expect.objectContaining({
          Authorization: "Bearer token",
          Range: "bytes=0-0",
        }),
      }),
    );
    expect(openTableMock).toHaveBeenCalledTimes(1);
    const firstOpenCall = openTableMock.mock.calls[0]!;
    expect(firstOpenCall[0]!).toBe(
      "https://example.com/search_table.lance/",
    );
    expect(JSON.parse(firstOpenCall[1]!)).toEqual({
      headers: { Authorization: "Bearer token" },
      cacheBytes: 4096,
      latestVersionUrl: "https://example.com/search_table.lance/_latest.version",
      manifestUrl: "https://example.com/search_table.lance/_latest.manifest",
      snapshotUrl: "https://example.com/search_table.lance/_snapshot.json",
      webMetadataUrl: "https://example.com/search_table.lance/_web.json",
    });
  });

  it("decodes schema and search results from Arrow IPC and applies published defaults", async () => {
    const handle = makeHandle();
    __setWasmModuleLoaderForTests(async () => ({
      open_table: async () => handle,
    }));

    const table = await openTable("https://example.com/search_table.lance", {
      fetch: successfulFetch(),
    });
    const schema = await table.schema();
    const results = await table.search({
      text: "apple",
      vector: new Float32Array([0, 1]),
    });

    expect(schema.fields.map((field) => field.name)).toEqual(["id", "doc"]);
    expect(results.numRows).toBe(2);
    expect(handle.search).toHaveBeenCalledWith(
      JSON.stringify({
        text: "apple",
        vector: [0, 1],
        vectorColumn: "embedding",
      }),
    );
  });

  it("uses a worker backend when available", async () => {
    const directOpenMock = jest.fn<
      Promise<WasmRemoteSearchHandle>,
      [string, string | undefined]
    >();
    __setWasmModuleLoaderForTests(async () => ({
      open_table: directOpenMock,
    }));

    const handle = makeHandle();
    const worker = makeWorker(handle);
    __setWorkerFactoryForTests(() => worker);

    const table = await openTable("https://example.com/search_table.lance", {
      fetch: successfulFetch(),
    });
    const schema = await table.schema();
    const results = await table.search({
      vector: new Float32Array([0, 1]),
    });

    expect(schema.fields.map((field) => field.name)).toEqual(["id", "doc"]);
    expect(results.numRows).toBe(2);
    expect(handle.schema).toHaveBeenCalledTimes(1);
    expect(handle.search).toHaveBeenCalledWith(
      JSON.stringify({
        vector: [0, 1],
        vectorColumn: "embedding",
      }),
    );
    expect(directOpenMock).not.toHaveBeenCalled();

    table.close();
    expect(worker.terminate).toHaveBeenCalledTimes(1);
  });

  it("falls back to the direct runtime when worker init fails", async () => {
    const handle = makeHandle();
    const openTableMock = jest.fn<
      Promise<WasmRemoteSearchHandle>,
      [string, string | undefined]
    >(async () => handle);
    __setWasmModuleLoaderForTests(async () => ({
      open_table: openTableMock,
    }));
    __setWorkerFactoryForTests(() => makeWorker(makeHandle(), { failOpen: true }));

    const table = await openTable("https://example.com/search_table.lance", {
      fetch: successfulFetch(),
    });
    await table.search({
      vector: new Float32Array([0, 1]),
    });

    expect(openTableMock).toHaveBeenCalledTimes(1);
    expect(handle.search).toHaveBeenCalledTimes(1);
  });

  it("reopens the wasm handle when dynamic headers change", async () => {
    const firstHandle = makeHandle();
    const secondHandle = makeHandle();
    const openTableMock = jest
      .fn<Promise<WasmRemoteSearchHandle>, [string, string | undefined]>()
      .mockResolvedValueOnce(firstHandle)
      .mockResolvedValueOnce(secondHandle);

    let headerValue = "token-a";
    const options: OpenTableOptions = {
      fetch: successfulFetch(),
      headers: async () => ({ Authorization: headerValue }),
    };

    __setWasmModuleLoaderForTests(async () => ({
      open_table: openTableMock,
    }));

    const table = await openTable("https://example.com/search_table.lance", options);
    headerValue = "token-b";
    await table.search({ vector: new Float32Array([0, 1]) });

    expect(firstHandle.close).toHaveBeenCalledTimes(1);
    const secondOpenCall = openTableMock.mock.calls[1]!;
    expect(secondOpenCall[0]!).toBe(
      "https://example.com/search_table.lance/",
    );
    expect(JSON.parse(secondOpenCall[1]!)).toEqual({
      headers: { Authorization: "token-b" },
      latestVersionUrl: "https://example.com/search_table.lance/_latest.version",
      manifestUrl: "https://example.com/search_table.lance/_latest.manifest",
      snapshotUrl: "https://example.com/search_table.lance/_snapshot.json",
      webMetadataUrl: "https://example.com/search_table.lance/_web.json",
    });
    table.close();
    expect(secondHandle.close).toHaveBeenCalledTimes(1);
  });

  it("uses _latest.version to skip reopening when the snapshot is unchanged", async () => {
    const handle = makeHandle();
    const openTableMock = jest.fn<
      Promise<WasmRemoteSearchHandle>,
      [string, string | undefined]
    >(async () => handle);
    __setWasmModuleLoaderForTests(async () => ({
      open_table: openTableMock,
    }));

    const fetchMock = sidecarFetch({
      latestVersion: "7",
    });

    const table = await openTable("https://example.com/search_table.lance", {
      fetch: fetchMock,
    });
    expect(await table.refresh()).toBe(false);
    expect(handle.refresh).not.toHaveBeenCalled();
    expect(openTableMock).toHaveBeenCalledTimes(1);
  });

  it("fails fast when published metadata advertises no FTS index", async () => {
    const handle = makeHandle();
    __setWasmModuleLoaderForTests(async () => ({
      open_table: async () => handle,
    }));

    const table = await openTable("https://example.com/search_table.lance", {
      fetch: sidecarFetch({
        metadata: publishedMetadata({ ftsColumns: [] }),
        snapshot: publishedSnapshot({ ftsColumns: [] }),
      }),
    });

    await expect(table.search({ text: "apple" })).rejects.toThrow(/full-text search indexed columns/);
    expect(handle.search).not.toHaveBeenCalled();
  });

  it("fails fast when range support is missing", async () => {
    __setWasmModuleLoaderForTests(async () => ({
      open_table: async () => makeHandle(),
    }));

    await expect(
      openTable("https://example.com/search_table.lance", {
        fetch: (async (input: RequestInfo | URL) => {
          const url = input.toString();
          if (url.endsWith("/_web.json") || url.endsWith("/_snapshot.json")) {
            return new Response(null, { status: 404 });
          }
          return new Response(null, { status: 200 });
        }) as unknown as typeof globalThis.fetch,
      }),
    ).rejects.toThrow(/Expected status 206/);
  });
});

function successfulFetch(): typeof globalThis.fetch {
  return sidecarFetch();
}

function sidecarFetch({
  latestVersion = "7",
  metadata = publishedMetadata(),
  snapshot = publishedSnapshot(),
}: {
  latestVersion?: string;
  metadata?: Record<string, unknown>;
  snapshot?: Record<string, unknown>;
} = {}): typeof globalThis.fetch {
  return (async (input: RequestInfo | URL) => {
    const url = input.toString();
    if (url.endsWith("/_web.json")) {
      return Response.json(metadata);
    }
    if (url.endsWith("/_snapshot.json")) {
      return Response.json(snapshot);
    }
    if (url.endsWith("/_latest.version")) {
      return new Response(latestVersion, { status: 200 });
    }
    if (url.endsWith("/_latest.manifest")) {
      return new Response(new Uint8Array([1]), {
        status: 206,
        headers: {
          "content-range": "bytes 0-0/1",
        },
      });
    }
    throw new Error(`unexpected fetch ${url}`);
  }) as unknown as typeof globalThis.fetch;
}

function makeWorker(
  handle: jest.Mocked<WasmRemoteSearchHandle>,
  options: { failOpen?: boolean } = {},
) {
  const messageListeners = new Set<
    (event: MessageEvent<WorkerResponse>) => void
  >();
  const errorListeners = new Set<(event: ErrorEvent) => void>();

  const emitMessage = (response: WorkerResponse) => {
    const event = { data: response } as MessageEvent<WorkerResponse>;
    for (const listener of messageListeners) {
      listener(event);
    }
  };

  const worker = {
    addEventListener: jest.fn(
      (
        type: "message" | "error",
        listener:
          | ((event: MessageEvent<WorkerResponse>) => void)
          | ((event: ErrorEvent) => void),
      ) => {
        if (type === "message") {
          messageListeners.add(
            listener as (event: MessageEvent<WorkerResponse>) => void,
          );
          return;
        }
        errorListeners.add(listener as (event: ErrorEvent) => void);
      },
    ),
    removeEventListener: jest.fn(
      (
        type: "message" | "error",
        listener:
          | ((event: MessageEvent<WorkerResponse>) => void)
          | ((event: ErrorEvent) => void),
      ) => {
        if (type === "message") {
          messageListeners.delete(
            listener as (event: MessageEvent<WorkerResponse>) => void,
          );
          return;
        }
        errorListeners.delete(listener as (event: ErrorEvent) => void);
      },
    ),
    postMessage: jest.fn((request: WorkerRequest) => {
      queueMicrotask(async () => {
        try {
          switch (request.type) {
            case "open":
              if (options.failOpen) {
                emitMessage({
                  id: request.id,
                  ok: false,
                  error: "worker open failed",
                });
              } else {
                emitMessage({
                  id: request.id,
                  ok: true,
                  type: "open",
                });
              }
              return;
            case "schema": {
              const bytes = await handle.schema();
              emitMessage({
                id: request.id,
                ok: true,
                type: "schema",
                bytes: copyBuffer(bytes),
              });
              return;
            }
            case "search": {
              const bytes = await handle.search(request.requestJson);
              emitMessage({
                id: request.id,
                ok: true,
                type: "search",
                bytes: copyBuffer(bytes),
              });
              return;
            }
            case "refresh":
              emitMessage({
                id: request.id,
                ok: true,
                type: "refresh",
                changed: await handle.refresh(),
              });
              return;
            default:
              request satisfies never;
          }
        } catch (error) {
          const event = {
            message: error instanceof Error ? error.message : String(error),
          } as ErrorEvent;
          for (const listener of errorListeners) {
            listener(event);
          }
        }
      });
    }),
    terminate: jest.fn(),
  };

  return worker;
}

function copyBuffer(bytes: Uint8Array): ArrayBuffer {
  const owned = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
    ? new Uint8Array(bytes)
    : bytes.slice();
  return owned.buffer.slice(0) as ArrayBuffer;
}
