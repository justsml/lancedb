import type { Schema, Table as ArrowTable } from "apache-arrow";
import {
  openTable,
  type OpenTableOptions,
  type RemoteSearchTable,
  type SearchRequest,
} from "./index.js";

export type PoolingStrategy = "mean" | "cls" | "last_token";

export interface QueryTransformContext {
  /** The Hugging Face model id being used to embed. */
  model: string;
}

// ---------------------------------------------------------------------------
// EmbeddingModel — the first-class abstraction
// ---------------------------------------------------------------------------

/**
 * A model that can embed text into vectors.
 *
 * Pass an `EmbeddingModel` as the second argument to `searchTable`, or to the
 * standalone `embed`/`embedMany` functions.  Use `transformersEmbedder()` to
 * create one backed by `@huggingface/transformers`, or bring your own
 * implementation.
 */
export interface EmbeddingModel {
  /** Embed a single text value. */
  embed(value: string, options?: { signal?: AbortSignal }): Promise<EmbedResult>;
  /** Embed multiple text values. */
  embedMany(values: string[], options?: { signal?: AbortSignal }): Promise<EmbedManyResult>;
  /**
   * Eagerly load the model weights and tokenizer so the first `embed()` call
   * doesn't pay the full download + init cost.  No-op if already loaded or if
   * the implementation doesn't support preloading.
   */
  preload?(): Promise<void>;
  /**
   * Release the loaded model and tokenizer from memory.  After calling
   * `dispose()` the model can still be used — it will simply re-download on
   * the next `embed()` call.
   */
  dispose?(): void;
}

/**
 * Result of a single embedding operation.  Mirrors the shape returned by
 * the Vercel AI SDK `embed()` function.
 */
export interface EmbedResult {
  /** The embedding vector for the input value. */
  embedding: number[];
}

/**
 * Result of a batch embedding operation.  Mirrors the shape returned by
 * the Vercel AI SDK `embedMany()` function.
 */
export interface EmbedManyResult {
  /** One embedding vector per input value, in the same order. */
  embeddings: number[][];
}

// ---------------------------------------------------------------------------
// Standalone embed / embedMany
// ---------------------------------------------------------------------------

export interface EmbedRequest {
  /** The text value to embed. */
  value: string;
  /**
   * An `EmbeddingModel`, or a Hugging Face model id string that will be
   * auto-promoted to one via `transformersEmbedder()`.
   */
  model?: string | EmbeddingModel;
  /** Abort signal for cancellation (e.g. typeahead debouncing). */
  signal?: AbortSignal;
}

export interface EmbedManyRequest {
  /** The text values to embed. */
  values: string[];
  /**
   * An `EmbeddingModel`, or a Hugging Face model id string that will be
   * auto-promoted to one via `transformersEmbedder()`.
   */
  model?: string | EmbeddingModel;
  /** Abort signal for cancellation. */
  signal?: AbortSignal;
}

/**
 * Embed a single text value.
 *
 * ```ts
 * import { embed } from "@lancedb/lancedb-web/transformers";
 *
 * const { embedding } = await embed({
 *   model: "BAAI/bge-small-en-v1.5",
 *   value: "best places to hike in colorado",
 * });
 * ```
 */
export async function embed(request: EmbedRequest): Promise<EmbedResult> {
  const model = resolveModel(request.model);
  return model.embed(request.value, { signal: request.signal });
}

/**
 * Embed multiple text values.
 *
 * ```ts
 * import { embedMany } from "@lancedb/lancedb-web/transformers";
 *
 * const { embeddings } = await embedMany({
 *   model: "BAAI/bge-small-en-v1.5",
 *   values: ["hiking trails", "mountain biking"],
 * });
 * ```
 */
export async function embedMany(
  request: EmbedManyRequest,
): Promise<EmbedManyResult> {
  const model = resolveModel(request.model);
  return model.embedMany(request.values, { signal: request.signal });
}

// ---------------------------------------------------------------------------
// searchTable
// ---------------------------------------------------------------------------

export interface SearchTableOptions extends OpenTableOptions {
  /**
   * An `EmbeddingModel`, or a Hugging Face model id to auto-promote.
   *
   * Defaults to `Xenova/all-MiniLM-L6-v2`.
   */
  model?: string | EmbeddingModel;
  /**
   * Optional tokenizer override (only used when `model` is a string).
   */
  tokenizer?: string;
  /**
   * Pooling strategy (only used when `model` is a string).
   */
  pooling?: PoolingStrategy;
  /**
   * Normalize the final embedding (only used when `model` is a string).
   *
   * Defaults to `true`.
   */
  normalize?: boolean;
  /**
   * Optional hook to rewrite query text before embedding (only used when
   * `model` is a string).
   */
  prepareQuery?: (query: string, context: QueryTransformContext) => string;
  /**
   * Options passed to `AutoModel.from_pretrained` (only used when `model` is
   * a string).
   */
  modelOptions?: Record<string, unknown>;
  /**
   * Options passed to the tokenizer (only used when `model` is a string).
   */
  tokenizerOptions?: {
    textPair?: string | string[];
    padding?: boolean | "max_length";
    addSpecialTokens?: boolean;
    truncation?: boolean;
    maxLength?: number;
  };
}

/** @deprecated Use `SearchTableOptions` instead. */
export type TransformersSearchTableOptions = SearchTableOptions;

export interface TextEmbeddingSearchRequest
  extends Omit<SearchRequest, "text" | "vector"> {
  /**
   * Natural language query text to embed on the client.
   */
  text: string;
  /**
   * When true, logs the resolved embedding configuration to `console.debug`.
   */
  debug?: boolean;
  /** Abort signal for cancellation (e.g. typeahead debouncing). */
  signal?: AbortSignal;
}

export interface TextEmbeddingSearchTable {
  /** The `EmbeddingModel` powering this table's search. */
  readonly model: EmbeddingModel;
  /** Arbitrary user-defined metadata from the published sidecar files. */
  readonly metadata: Record<string, string>;
  schema(): Promise<Schema>;
  search(request: TextEmbeddingSearchRequest): Promise<ArrowTable>;
  refresh(): Promise<boolean>;
  close(): void;
}

// ---------------------------------------------------------------------------
// Model resolver — pluggable so transformers.ts can register its factory
// ---------------------------------------------------------------------------

/**
 * A factory that builds an `EmbeddingModel` from a model id string.
 * Registered by `transformers.ts` at import time.
 */
export type EmbeddingModelFactory = (modelId?: string, options?: SearchTableOptions) => EmbeddingModel;

let _modelFactory: EmbeddingModelFactory | null = null;

/**
 * Register a default factory that turns a string model id into an
 * `EmbeddingModel`.  Called by `transformers.ts` at module scope.
 */
export function __registerModelFactory(factory: EmbeddingModelFactory): void {
  _modelFactory = factory;
}

function isEmbeddingModel(value: unknown): value is EmbeddingModel {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as EmbeddingModel).embed === "function" &&
    typeof (value as EmbeddingModel).embedMany === "function"
  );
}

function resolveModel(model: string | EmbeddingModel | undefined): EmbeddingModel {
  if (model === undefined || typeof model === "string") {
    if (_modelFactory === null) {
      throw new Error(
        "No EmbeddingModel factory is registered. " +
        "Import '@lancedb/lancedb-web/transformers' to use string model ids, " +
        "or pass an EmbeddingModel object directly.",
      );
    }
    return _modelFactory(model);
  }
  return model;
}

/**
 * Open a published Lance table and wrap it with client-side query embedding
 * generation.
 *
 * The second argument is an `EmbeddingModel`, a Hugging Face model id string
 * (auto-promoted via `transformersEmbedder()`), or an options object.
 *
 * ```ts
 * // String shorthand — auto-promoted to EmbeddingModel
 * const t = await searchTable(url, "BAAI/bge-small-en-v1.5");
 *
 * // Bring your own EmbeddingModel
 * const t = await searchTable(url, myEmbeddingModel);
 *
 * // Options object
 * const t = await searchTable(url, { model: "BAAI/bge-small-en-v1.5", normalize: false });
 * ```
 */
export async function searchTable(
  tableUrl: string,
  model?: string | EmbeddingModel,
  options?: OpenTableOptions,
): Promise<TextEmbeddingSearchTable>;
export async function searchTable(
  tableUrl: string,
  options?: SearchTableOptions,
): Promise<TextEmbeddingSearchTable>;
export async function searchTable(
  tableUrl: string,
  modelOrOptions:
    | string
    | EmbeddingModel
    | SearchTableOptions = {},
  options?: OpenTableOptions,
): Promise<TextEmbeddingSearchTable> {
  let embeddingModel: EmbeddingModel;
  let openOptions: OpenTableOptions;

  if (typeof modelOrOptions === "string") {
    embeddingModel = resolveModel(modelOrOptions);
    openOptions = options ?? {};
  } else if (isEmbeddingModel(modelOrOptions)) {
    embeddingModel = modelOrOptions;
    openOptions = options ?? {};
  } else {
    const {
      model,
      tokenizer,
      pooling,
      normalize,
      prepareQuery,
      modelOptions,
      tokenizerOptions,
      ...rest
    } = modelOrOptions;
    openOptions = rest;

    if (isEmbeddingModel(model)) {
      embeddingModel = model;
    } else if (_modelFactory !== null) {
      embeddingModel = _modelFactory(
        typeof model === "string" ? model : undefined,
        { tokenizer, pooling, normalize, prepareQuery, modelOptions, tokenizerOptions },
      );
    } else if (model === undefined) {
      throw new Error(
        "No EmbeddingModel factory is registered. " +
        "Import '@lancedb/lancedb-web/transformers' to use string model ids, " +
        "or pass an EmbeddingModel object directly.",
      );
    } else {
      throw new Error(
        "No EmbeddingModel factory is registered. " +
        "Import '@lancedb/lancedb-web/transformers' to use string model ids, " +
        "or pass an EmbeddingModel object directly.",
      );
    }
  }

  const table = await openTable(tableUrl, openOptions);
  return new TextEmbeddingSearchTableImpl(table, embeddingModel);
}

// ---------------------------------------------------------------------------
// TextEmbeddingSearchTableImpl
// ---------------------------------------------------------------------------

class TextEmbeddingSearchTableImpl implements TextEmbeddingSearchTable {
  readonly #table: RemoteSearchTable;
  readonly model: EmbeddingModel;

  get metadata(): Record<string, string> {
    return this.#table.metadata;
  }

  constructor(table: RemoteSearchTable, model: EmbeddingModel) {
    this.#table = table;
    this.model = model;
  }

  schema(): Promise<Schema> {
    return this.#table.schema();
  }

  async search(request: TextEmbeddingSearchRequest): Promise<ArrowTable> {
    const { text, debug, signal, ...vectorSearchRequest } = request;
    const { embedding } = await this.model.embed(text, { signal });

    if (debug && typeof console.debug === "function") {
      console.debug("[@lancedb/lancedb-web/transformers]", {
        query: text,
        dimensions: embedding.length,
      });
    }

    return await this.#table.search({
      ...vectorSearchRequest,
      vector: embedding,
    });
  }

  refresh(): Promise<boolean> {
    return this.#table.refresh();
  }

  close(): void {
    this.#table.close();
  }
}
