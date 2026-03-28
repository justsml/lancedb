import { tableFromArrays, tableToIPC } from "apache-arrow";
import {
  __setWasmModuleLoaderForTests,
  openTable,
  type OpenTableOptions,
} from "../src";
import type { WasmRemoteSearchHandle } from "../src/generated/lancedb_wasm";

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

describe("@lancedb/lancedb-web", () => {
  afterEach(() => {
    __setWasmModuleLoaderForTests();
    jest.restoreAllMocks();
  });

  it("normalizes the table URL, checks range support, and opens the wasm handle", async () => {
    const handle = makeHandle();
    const openTableMock = jest.fn(async () => handle);
    __setWasmModuleLoaderForTests(async () => ({
      open_table: openTableMock,
    }));

    const fetchMock = jest.fn(async () => {
      return new Response(new Uint8Array([1]), {
        status: 206,
        headers: {
          "content-range": "bytes 0-0/1",
        },
      });
    });

    await openTable("https://example.com/search_table.lance", {
      fetch: fetchMock as unknown as typeof globalThis.fetch,
      headers: { Authorization: "Bearer token" },
      cacheBytes: 4096,
    });

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
    expect(openTableMock).toHaveBeenCalledWith(
      "https://example.com/search_table.lance/",
      JSON.stringify({
        headers: { Authorization: "Bearer token" },
        cacheBytes: 4096,
      }),
    );
  });

  it("decodes schema and search results from Arrow IPC", async () => {
    const handle = makeHandle();
    __setWasmModuleLoaderForTests(async () => ({
      open_table: async () => handle,
    }));

    const table = await openTable("https://example.com/search_table.lance", {
      fetch: successfulFetch(),
    });
    const schema = await table.schema();
    const results = await table.search({ text: "apple" });

    expect(schema.fields.map((field) => field.name)).toEqual(["id", "doc"]);
    expect(results.numRows).toBe(2);
    expect(handle.search).toHaveBeenCalledWith(JSON.stringify({ text: "apple" }));
  });

  it("reopens the wasm handle when dynamic headers change", async () => {
    const firstHandle = makeHandle();
    const secondHandle = makeHandle();
    const openTableMock = jest
      .fn()
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
    expect(openTableMock).toHaveBeenNthCalledWith(
      2,
      "https://example.com/search_table.lance/",
      JSON.stringify({
        headers: { Authorization: "token-b" },
      }),
    );
    table.close();
    expect(secondHandle.close).toHaveBeenCalledTimes(1);
  });

  it("fails fast when range support is missing", async () => {
    __setWasmModuleLoaderForTests(async () => ({
      open_table: async () => makeHandle(),
    }));

    await expect(
      openTable("https://example.com/search_table.lance", {
        fetch: (async () =>
          new Response(null, { status: 200 })) as unknown as typeof globalThis.fetch,
      }),
    ).rejects.toThrow(/Expected status 206/);
  });
});

function successfulFetch(): typeof globalThis.fetch {
  return (async () =>
    new Response(new Uint8Array([1]), {
      status: 206,
      headers: {
        "content-range": "bytes 0-0/1",
      },
    })) as unknown as typeof globalThis.fetch;
}
