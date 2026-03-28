// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

import * as apiArrow from "apache-arrow";
import * as arrow18 from "apache-arrow-18";

import {
  EmbeddingFunction,
  LanceSchema,
} from "../lancedb/embedding";
import {
  parseLlmUri,
  detectProvider,
  isLlmUri,
} from "../lancedb/embedding/llm_uri";
import { getRegistry, register } from "../lancedb/embedding/registry";

describe("parseLlmUri", () => {
  it("should parse a basic URI with host and model", () => {
    const config = parseLlmUri(
      "llm://api.openai.com/text-embedding-3-small",
    );
    expect(config.host).toBe("api.openai.com");
    expect(config.model).toBe("text-embedding-3-small");
    expect(config.label).toBeUndefined();
    expect(config.apiKey).toBeUndefined();
    expect(config.params).toEqual({});
    expect(config.raw).toBe(
      "llm://api.openai.com/text-embedding-3-small",
    );
  });

  it("should parse URI with query parameters", () => {
    const config = parseLlmUri(
      "llm://api.openai.com/text-embedding-3-large?dimensions=1024",
    );
    expect(config.host).toBe("api.openai.com");
    expect(config.model).toBe("text-embedding-3-large");
    expect(config.params).toEqual({ dimensions: "1024" });
  });

  it("should parse URI with multiple query parameters", () => {
    const config = parseLlmUri(
      "llm://api.openai.com/text-embedding-3-large?dimensions=1024&encoding_format=float",
    );
    expect(config.params).toEqual({
      dimensions: "1024",
      encoding_format: "float",
    });
  });

  it("should parse URI with label and API key", () => {
    const config = parseLlmUri(
      "llm://myapp:sk-test-key@api.openai.com/text-embedding-3-small",
    );
    expect(config.label).toBe("myapp");
    expect(config.apiKey).toBe("sk-test-key");
    expect(config.host).toBe("api.openai.com");
    expect(config.model).toBe("text-embedding-3-small");
  });

  it("should parse URI with label only (no apiKey)", () => {
    const config = parseLlmUri(
      "llm://myapp@api.openai.com/text-embedding-3-small",
    );
    expect(config.label).toBe("myapp");
    expect(config.apiKey).toBeUndefined();
  });

  it("should handle URL-encoded characters in apiKey", () => {
    const config = parseLlmUri(
      "llm://app:sk-key%2Fwith%3Dchars@api.openai.com/text-embedding-3-small",
    );
    expect(config.apiKey).toBe("sk-key/with=chars");
  });

  it("should parse model paths with slashes", () => {
    const config = parseLlmUri(
      "llm://generativelanguage.googleapis.com/models/text-embedding-004",
    );
    expect(config.host).toBe("generativelanguage.googleapis.com");
    expect(config.model).toBe("models/text-embedding-004");
  });

  it("should throw for non-llm:// scheme", () => {
    expect(() => parseLlmUri("https://api.openai.com/model")).toThrow(
      "Invalid llm:// URI",
    );
  });

  it("should throw for empty model", () => {
    expect(() => parseLlmUri("llm://api.openai.com/")).toThrow(
      "missing model",
    );
    expect(() => parseLlmUri("llm://api.openai.com")).toThrow(
      "missing model",
    );
  });
});

describe("detectProvider", () => {
  it("should detect OpenAI", () => {
    expect(detectProvider("api.openai.com")).toBe("openai");
  });

  it("should detect Anthropic", () => {
    expect(detectProvider("api.anthropic.com")).toBe("anthropic");
  });

  it("should detect Google", () => {
    expect(detectProvider("generativelanguage.googleapis.com")).toBe(
      "google",
    );
  });

  it("should detect Cohere", () => {
    expect(detectProvider("api.cohere.com")).toBe("cohere");
  });

  it("should detect Bedrock from regional hosts", () => {
    expect(
      detectProvider("bedrock-runtime.us-east-1.amazonaws.com"),
    ).toBe("bedrock");
    expect(
      detectProvider("bedrock-runtime.eu-west-1.amazonaws.com"),
    ).toBe("bedrock");
  });

  it("should return undefined for unknown hosts", () => {
    expect(detectProvider("custom.embeddings.com")).toBeUndefined();
  });
});

describe("isLlmUri", () => {
  it("should return true for llm:// URIs", () => {
    expect(isLlmUri("llm://api.openai.com/model")).toBe(true);
  });

  it("should return false for other strings", () => {
    expect(isLlmUri("openai")).toBe(false);
    expect(isLlmUri("https://api.openai.com")).toBe(false);
    expect(isLlmUri("text-embedding-3-small")).toBe(false);
  });
});

describe("registry.fromUri", () => {
  afterEach(() => {
    getRegistry().reset();
  });

  it("should create an embedding function from a URI via registered provider", () => {
    @register("openai")
    class MockOpenAI extends EmbeddingFunction<string> {
      model: string;
      apiKey?: string;
      // biome-ignore lint/suspicious/noExplicitAny: test
      constructor(options: any = {}) {
        super();
        this.model = options.model ?? "default";
        this.apiKey = options.apiKey;
      }
      ndims() {
        return 3;
      }
      embeddingDataType() {
        return new arrow18.Float32() as apiArrow.Float;
      }
      async computeSourceEmbeddings(data: string[]) {
        return data.map(() => [1, 2, 3]);
      }
    }

    const func = getRegistry().fromUri(
      "llm://api.openai.com/text-embedding-3-small",
    ) as MockOpenAI;

    expect(func).toBeInstanceOf(MockOpenAI);
    expect(func.model).toBe("text-embedding-3-small");
  });

  it("should pass API key from URI to embedding function", () => {
    @register("openai")
    class MockOpenAI extends EmbeddingFunction<string> {
      model: string;
      apiKey?: string;
      // biome-ignore lint/suspicious/noExplicitAny: test
      constructor(options: any = {}) {
        super();
        this.model = options.model ?? "default";
        this.apiKey = options.apiKey;
      }
      ndims() {
        return 3;
      }
      embeddingDataType() {
        return new arrow18.Float32() as apiArrow.Float;
      }
      async computeSourceEmbeddings(data: string[]) {
        return data.map(() => [1, 2, 3]);
      }
    }

    const func = getRegistry().fromUri(
      "llm://myapp:sk-test@api.openai.com/text-embedding-3-small",
    ) as MockOpenAI;

    expect(func.apiKey).toBe("sk-test");
    expect(func.model).toBe("text-embedding-3-small");
  });

  it("should pass query params as options", () => {
    @register("openai")
    class MockOpenAI extends EmbeddingFunction<string> {
      model: string;
      dimensions?: string;
      // biome-ignore lint/suspicious/noExplicitAny: test
      constructor(options: any = {}) {
        super();
        this.model = options.model ?? "default";
        this.dimensions = options.dimensions;
      }
      ndims() {
        return 3;
      }
      embeddingDataType() {
        return new arrow18.Float32() as apiArrow.Float;
      }
      async computeSourceEmbeddings(data: string[]) {
        return data.map(() => [1, 2, 3]);
      }
    }

    const func = getRegistry().fromUri(
      "llm://api.openai.com/text-embedding-3-large?dimensions=512",
    ) as MockOpenAI;

    expect(func.dimensions).toBe("512");
    expect(func.model).toBe("text-embedding-3-large");
  });

  it("should throw for unregistered provider", () => {
    expect(() =>
      getRegistry().fromUri(
        "llm://api.unknown-provider.com/some-model",
      ),
    ).toThrow("No embedding function registered");
  });

  it("should work end-to-end with LanceSchema", () => {
    @register("openai")
    class MockOpenAI extends EmbeddingFunction<string> {
      model: string;
      // biome-ignore lint/suspicious/noExplicitAny: test
      constructor(options: any = {}) {
        super();
        this.model = options.model ?? "default";
      }
      ndims() {
        return 3;
      }
      embeddingDataType() {
        return new arrow18.Float32() as apiArrow.Float;
      }
      async computeSourceEmbeddings(data: string[]) {
        return data.map(() => [1, 2, 3]);
      }
    }

    const func = getRegistry().fromUri(
      "llm://api.openai.com/text-embedding-3-small",
    );

    const schema = LanceSchema({
      id: new arrow18.Int32(),
      text: func.sourceField(new arrow18.Utf8() as apiArrow.DataType),
      vector: func.vectorField(),
    });

    // Verify schema has the embedding function metadata
    expect(schema.metadata.get("embedding_functions")).toBeDefined();
    const metadata = JSON.parse(schema.metadata.get("embedding_functions")!);
    expect(metadata[0].name).toBe("openai");
    expect(metadata[0].sourceColumn).toBe("text");
    expect(metadata[0].vectorColumn).toBe("vector");
  });
});
