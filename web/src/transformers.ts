// ---------------------------------------------------------------------------
// Re-export everything from embedding.ts so that existing
// `import { ... } from "@lancedb/lancedb-web/transformers"` continues to work.
// ---------------------------------------------------------------------------
export {
  type EmbeddingModel,
  type EmbedResult,
  type EmbedManyResult,
  type EmbedRequest,
  type EmbedManyRequest,
  type PoolingStrategy,
  type QueryTransformContext,
  type SearchTableOptions,
  type TransformersSearchTableOptions,
  type TextEmbeddingSearchRequest,
  type TextEmbeddingSearchTable,
  embed,
  embedMany,
  searchTable,
  __registerModelFactory,
} from "./embedding.js";

import type {
  EmbeddingModel,
  EmbedResult,
  EmbedManyResult,
  PoolingStrategy,
  QueryTransformContext,
  SearchTableOptions,
} from "./embedding.js";
import { __registerModelFactory } from "./embedding.js";

// ---------------------------------------------------------------------------
// TransformersEmbedderOptions
// ---------------------------------------------------------------------------

export interface TransformersEmbedderOptions {
  /**
   * Hugging Face model id to load with `@huggingface/transformers`.
   *
   * Defaults to `Xenova/all-MiniLM-L6-v2`.
   */
  model?: string;
  /**
   * Optional tokenizer override. If omitted, the model id is reused.
   */
  tokenizer?: string;
  /**
   * Pooling strategy applied to the model hidden states.
   *
   * Defaults are inferred for known model families.
   */
  pooling?: PoolingStrategy;
  /**
   * Normalize the final embedding.
   *
   * Defaults to `true`.
   */
  normalize?: boolean;
  /**
   * Optional hook to rewrite query text before embedding.
   */
  prepareQuery?: (query: string, context: QueryTransformContext) => string;
  /**
   * Options passed to `AutoModel.from_pretrained`.
   */
  modelOptions?: Record<string, unknown>;
  /**
   * Options passed to the tokenizer call.
   */
  tokenizerOptions?: {
    textPair?: string | string[];
    padding?: boolean | "max_length";
    addSpecialTokens?: boolean;
    truncation?: boolean;
    maxLength?: number;
  };
}

// ---------------------------------------------------------------------------
// transformersEmbedder — factory for HuggingFace-backed EmbeddingModel
// ---------------------------------------------------------------------------

/**
 * Create an `EmbeddingModel` backed by `@huggingface/transformers`.
 *
 * ```ts
 * import { transformersEmbedder } from "@lancedb/lancedb-web/transformers";
 *
 * const model = transformersEmbedder("BAAI/bge-small-en-v1.5");
 * const { embedding } = await model.embed("best hikes in colorado");
 * ```
 *
 * The underlying ONNX session is cached at the module level so multiple
 * embedders that reference the same `model` (and `tokenizer`/`modelOptions`)
 * share the loaded weights.
 */
export function transformersEmbedder(
  model?: string,
  options?: Omit<TransformersEmbedderOptions, "model">,
): EmbeddingModel;
export function transformersEmbedder(
  options?: TransformersEmbedderOptions,
): EmbeddingModel;
export function transformersEmbedder(
  modelOrOptions?: string | TransformersEmbedderOptions,
  options?: Omit<TransformersEmbedderOptions, "model">,
): EmbeddingModel {
  const resolved =
    typeof modelOrOptions === "string"
      ? { ...options, model: modelOrOptions }
      : modelOrOptions ?? {};
  return buildEmbeddingModel(resolved);
}

// ---------------------------------------------------------------------------
// Register the transformers-backed factory so embedding.ts can resolve
// string model ids automatically.
// ---------------------------------------------------------------------------

__registerModelFactory(
  (modelId?: string, searchOptions?: SearchTableOptions): EmbeddingModel => {
    const opts: TransformersEmbedderOptions = {};
    if (modelId !== undefined) {
      opts.model = modelId;
    }
    if (searchOptions !== undefined) {
      if (searchOptions.tokenizer !== undefined) opts.tokenizer = searchOptions.tokenizer;
      if (searchOptions.pooling !== undefined) opts.pooling = searchOptions.pooling;
      if (searchOptions.normalize !== undefined) opts.normalize = searchOptions.normalize;
      if (searchOptions.prepareQuery !== undefined) opts.prepareQuery = searchOptions.prepareQuery;
      if (searchOptions.modelOptions !== undefined) opts.modelOptions = searchOptions.modelOptions;
      if (searchOptions.tokenizerOptions !== undefined) opts.tokenizerOptions = searchOptions.tokenizerOptions;
    }
    return buildEmbeddingModel(opts);
  },
);

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

type TokenizerOptions = {
  textPair?: string | string[];
  padding?: boolean | "max_length";
  addSpecialTokens?: boolean;
  truncation?: boolean;
  maxLength?: number;
};

type TransformersModule = {
  AutoModel: {
    from_pretrained(
      model: string,
      options?: Record<string, unknown>,
    ): Promise<ModelLike>;
  };
  AutoTokenizer: {
    from_pretrained(model: string): Promise<TokenizerLike>;
  };
};

type TokenizerLike = (
  input: string | string[],
  options?: TokenizerOptions,
) => Record<string, unknown> | Promise<Record<string, unknown>>;

type ModelLike = {
  forward(
    inputs: Record<string, unknown>,
  ): Promise<Record<string, unknown>>;
};

type TensorLike = {
  dims: number[];
  data: ArrayLike<number>;
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_MODEL = "Xenova/all-MiniLM-L6-v2";
const ENGLISH_BGE_RETRIEVAL_PREFIX =
  "Represent this sentence for searching relevant passages: ";
const CHINESE_BGE_RETRIEVAL_PREFIX =
  "为这个句子生成表示以用于检索相关文章：";
const MULTILINGUAL_GEMMA2_INSTRUCTION =
  "Given a web search query, retrieve relevant passages that answer the query.";
const DEFAULT_TOKENIZER_OPTIONS: TokenizerOptions = {
  padding: true,
};

// ---------------------------------------------------------------------------
// Module-level model/tokenizer cache
// ---------------------------------------------------------------------------

let transformersModuleLoader: () => Promise<TransformersModule> =
  defaultTransformersModuleLoader;

const resourcesCache = new Map<
  string,
  Promise<{ model: ModelLike; tokenizer: TokenizerLike }>
>();

function resourcesCacheKey(
  model: string,
  tokenizer: string,
  modelOptions?: Record<string, unknown>,
): string {
  return modelOptions
    ? `${model}|${tokenizer}|${JSON.stringify(modelOptions)}`
    : `${model}|${tokenizer}`;
}

async function loadSharedResources(
  modelId: string,
  tokenizerId: string,
  modelOptions?: Record<string, unknown>,
): Promise<{ model: ModelLike; tokenizer: TokenizerLike }> {
  const key = resourcesCacheKey(modelId, tokenizerId, modelOptions);
  let promise = resourcesCache.get(key);
  if (!promise) {
    promise = initializeResources(modelId, tokenizerId, modelOptions);
    resourcesCache.set(key, promise);
  }
  return promise;
}

async function initializeResources(
  modelId: string,
  tokenizerId: string,
  modelOptions?: Record<string, unknown>,
): Promise<{ model: ModelLike; tokenizer: TokenizerLike }> {
  const transformers = await transformersModuleLoader();
  try {
    const [model, tokenizer] = await Promise.all([
      transformers.AutoModel.from_pretrained(modelId, modelOptions),
      transformers.AutoTokenizer.from_pretrained(tokenizerId),
    ]);
    return { model, tokenizer };
  } catch (error) {
    resourcesCache.delete(
      resourcesCacheKey(modelId, tokenizerId, modelOptions),
    );
    throw new Error(
      `Failed to initialize transformers model "${modelId}": ${(error as Error).message}`,
    );
  }
}

/**
 * Override the Transformers.js module loader for tests.
 */
export function __setTransformersModuleLoaderForTests(
  loader?: () => Promise<TransformersModule>,
): void {
  transformersModuleLoader = loader ?? defaultTransformersModuleLoader;
  resourcesCache.clear();
}

// ---------------------------------------------------------------------------
// buildEmbeddingModel
// ---------------------------------------------------------------------------

function buildEmbeddingModel(
  options: TransformersEmbedderOptions,
): EmbeddingModel {
  const modelId = options.model ?? DEFAULT_MODEL;
  const tokenizerId = options.tokenizer ?? modelId;
  const pooling = options.pooling ?? inferDefaultPooling(modelId);
  const normalize = options.normalize ?? true;
  const prepareQuery = options.prepareQuery ?? defaultPrepareQuery;
  const modelOptions = options.modelOptions;
  const tokenizerOptions: TokenizerOptions = {
    ...DEFAULT_TOKENIZER_OPTIONS,
    ...options.tokenizerOptions,
  };

  async function embedOne(
    value: string,
    signal?: AbortSignal,
  ): Promise<number[]> {
    signal?.throwIfAborted();
    const prepared = prepareQuery(value, { model: modelId });
    const { tokenizer, model } = await loadSharedResources(
      modelId,
      tokenizerId,
      modelOptions,
    );
    signal?.throwIfAborted();
    const inputs = await tokenizer([prepared], tokenizerOptions);
    signal?.throwIfAborted();
    const attentionMask = extractAttentionMask(inputs);
    const outputs = await model.forward(inputs);
    signal?.throwIfAborted();
    let vector = poolTensor(firstTensor(outputs), pooling, attentionMask);
    if (normalize) {
      vector = normalizeVector(vector);
    }
    return vector;
  }

  const cacheKey = resourcesCacheKey(modelId, tokenizerId, modelOptions);

  return {
    async embed(
      value: string,
      opts?: { signal?: AbortSignal },
    ): Promise<EmbedResult> {
      return { embedding: await embedOne(value, opts?.signal) };
    },
    async embedMany(
      values: string[],
      opts?: { signal?: AbortSignal },
    ): Promise<EmbedManyResult> {
      if (values.length === 0) {
        return { embeddings: [] };
      }
      if (values.length === 1) {
        return { embeddings: [await embedOne(values[0], opts?.signal)] };
      }

      const signal = opts?.signal;
      signal?.throwIfAborted();

      const prepared = values.map((v) =>
        prepareQuery(v, { model: modelId }),
      );
      const { tokenizer, model } = await loadSharedResources(
        modelId,
        tokenizerId,
        modelOptions,
      );
      signal?.throwIfAborted();

      const inputs = await tokenizer(prepared, tokenizerOptions);
      signal?.throwIfAborted();

      const attentionMask = extractAttentionMask(inputs);
      const outputs = await model.forward(inputs);
      signal?.throwIfAborted();

      const tensor = firstTensor(outputs);
      const embeddings = poolBatchTensor(
        tensor,
        pooling,
        values.length,
        attentionMask,
      );
      if (normalize) {
        return { embeddings: embeddings.map(normalizeVector) };
      }
      return { embeddings };
    },
    async preload(): Promise<void> {
      await loadSharedResources(modelId, tokenizerId, modelOptions);
    },
    dispose(): void {
      resourcesCache.delete(cacheKey);
    },
  };
}

// ---------------------------------------------------------------------------
// Transformers.js loader
// ---------------------------------------------------------------------------

async function defaultTransformersModuleLoader(): Promise<TransformersModule> {
  // The variable indirection plus bundler-specific comments prevent bundlers
  // from statically resolving the import, keeping @huggingface/transformers
  const specifier = "@huggingface/transformers";
  try {
    return await import(/* webpackIgnore: true */ /* @vite-ignore */ specifier);
  } catch (error) {
    throw new Error(
      "Failed to load @huggingface/transformers. Install it to use `@lancedb/lancedb-web/transformers`.",
    );
  }
}

// ---------------------------------------------------------------------------
// Model-family defaults
// ---------------------------------------------------------------------------

function inferDefaultPooling(model: string): PoolingStrategy {
  if (model === "BAAI/bge-multilingual-gemma2") {
    return "last_token";
  }
  if (model.startsWith("BAAI/bge-")) {
    return "cls";
  }
  return "mean";
}

function defaultPrepareQuery(
  query: string,
  context: QueryTransformContext,
): string {
  if (context.model === "BAAI/bge-multilingual-gemma2") {
    return `<instruct>${MULTILINGUAL_GEMMA2_INSTRUCTION}\n<query>${query}`;
  }
  if (isEnglishBgeModel(context.model)) {
    return `${ENGLISH_BGE_RETRIEVAL_PREFIX}${query}`;
  }
  if (isChineseBgeModel(context.model)) {
    return `${CHINESE_BGE_RETRIEVAL_PREFIX}${query}`;
  }
  return query;
}

function isEnglishBgeModel(model: string): boolean {
  return /^BAAI\/bge-.*-en(?:-v1\.5)?$/.test(model);
}

function isChineseBgeModel(model: string): boolean {
  return /^BAAI\/bge-.*-zh(?:-v1\.5)?$/.test(model);
}

// ---------------------------------------------------------------------------
// Tensor utilities
// ---------------------------------------------------------------------------

/** Known output keys in priority order. */
const PREFERRED_OUTPUT_KEYS = ["last_hidden_state", "hidden_states", "embeddings"];

function firstTensor(outputs: Record<string, unknown>): TensorLike {
  // Try well-known keys first so we don't accidentally grab pooler_output or attentions.
  for (const key of PREFERRED_OUTPUT_KEYS) {
    const value = outputs[key];
    if (value !== undefined && isTensorLike(value)) {
      return value;
    }
  }
  // Fallback: first tensor-like value.
  for (const value of Object.values(outputs)) {
    if (isTensorLike(value)) {
      return value;
    }
  }
  throw new Error("Transformers model output did not contain an embedding tensor.");
}

function isTensorLike(value: unknown): value is TensorLike {
  if (typeof value !== "object" || value === null) {
    return false;
  }

  const candidate = value as Partial<TensorLike>;
  return Array.isArray(candidate.dims) && candidate.data !== undefined;
}

function extractAttentionMask(
  inputs: Record<string, unknown>,
): TensorLike | undefined {
  const mask = inputs.attention_mask;
  if (mask !== undefined && isTensorLike(mask)) {
    return mask;
  }
  return undefined;
}

function poolTensor(
  tensor: TensorLike,
  pooling: PoolingStrategy,
  attentionMask?: TensorLike,
): number[] {
  if (tensor.dims.length === 1) {
    return Array.from(tensor.data);
  }

  if (tensor.dims.length === 2) {
    const [tokenCount, hiddenSize] = tensor.dims;
    const data = tensor.data;
    // For 2D tensors the mask (if present) has shape [1, tokenCount] or [tokenCount].
    const mask1d = flattenMaskForBatchElement(attentionMask, 0, tokenCount);
    switch (pooling) {
      case "cls":
        return sliceToken(data, 0, hiddenSize);
      case "last_token":
        return sliceToken(data, tokenCount - 1, hiddenSize);
      case "mean":
        return meanPool(data, tokenCount, hiddenSize, mask1d);
    }
  }

  if (tensor.dims.length !== 3) {
    throw new Error(
      `Unsupported embedding tensor shape [${tensor.dims.join(", ")}]. Expected 1D, 2D, or 3D output.`,
    );
  }

  const [batchSize, tokenCount, hiddenSize] = tensor.dims;
  if (batchSize < 1) {
    throw new Error("Embedding tensor batch dimension must be at least 1.");
  }

  const data = tensor.data;
  const mask1d = flattenMaskForBatchElement(attentionMask, 0, tokenCount);
  switch (pooling) {
    case "cls":
      return sliceToken(data, 0, hiddenSize);
    case "last_token":
      return sliceToken(data, tokenCount - 1, hiddenSize);
    case "mean":
      return meanPool(data, tokenCount, hiddenSize, mask1d);
  }
}

/**
 * Convert a batched tensor into one embedding per batch element.
 *
 * Supports already pooled 2D tensors `[batchSize, hiddenSize]` and token-level
 * 3D tensors `[batchSize, tokenCount, hiddenSize]`.
 */
function poolBatchTensor(
  tensor: TensorLike,
  pooling: PoolingStrategy,
  batchSize: number,
  attentionMask?: TensorLike,
): number[][] {
  // Some models return already pooled embeddings shaped as [batchSize, hiddenSize].
  if (tensor.dims.length === 2) {
    const [rows, hiddenSize] = tensor.dims;
    if (rows !== batchSize) {
      throw new Error(
        `Unsupported batched 2D tensor shape [${tensor.dims.join(", ")}]. Expected first dimension to match batch size ${batchSize}.`,
      );
    }
    const data = tensor.data;
    const results: number[][] = [];
    for (let rowIndex = 0; rowIndex < rows; rowIndex += 1) {
      results.push(sliceTokenAt(data, rowIndex * hiddenSize, 0, hiddenSize));
    }
    return results;
  }

  if (tensor.dims.length < 2) {
    throw new Error(
      `Unsupported batched embedding tensor shape [${tensor.dims.join(", ")}]. Expected 2D or 3D output.`,
    );
  }

  const tokenCount = tensor.dims[1];
  const hiddenSize = tensor.dims[2];
  const batchStride = tokenCount * hiddenSize;
  const data = tensor.data;
  const results: number[][] = [];

  for (let b = 0; b < batchSize; b += 1) {
    const offset = b * batchStride;
    const mask1d = flattenMaskForBatchElement(attentionMask, b, tokenCount);

    switch (pooling) {
      case "cls":
        results.push(sliceTokenAt(data, offset, 0, hiddenSize));
        break;
      case "last_token":
        results.push(sliceTokenAt(data, offset, tokenCount - 1, hiddenSize));
        break;
      case "mean":
        results.push(meanPoolAt(data, offset, tokenCount, hiddenSize, mask1d));
        break;
    }
  }

  return results;
}

function sliceToken(
  data: ArrayLike<number>,
  tokenIndex: number,
  hiddenSize: number,
): number[] {
  return sliceTokenAt(data, 0, tokenIndex, hiddenSize);
}

function sliceTokenAt(
  data: ArrayLike<number>,
  baseOffset: number,
  tokenIndex: number,
  hiddenSize: number,
): number[] {
  const start = baseOffset + tokenIndex * hiddenSize;
  const result = new Array<number>(hiddenSize);
  for (let i = 0; i < hiddenSize; i += 1) {
    result[i] = data[start + i];
  }
  return result;
}

/**
 * Extract a 1D mask slice for a given batch element from the attention mask
 * tensor.  Returns `undefined` when no mask is available (all tokens are
 * treated as real).
 */
function flattenMaskForBatchElement(
  mask: TensorLike | undefined,
  batchIndex: number,
  tokenCount: number,
): ArrayLike<number> | undefined {
  if (mask === undefined) {
    return undefined;
  }
  // 1D mask — applies directly.
  if (mask.dims.length === 1) {
    return mask.data;
  }
  // 2D mask [batchSize, seqLen] — slice the row for this batch element.
  if (mask.dims.length === 2) {
    const seqLen = mask.dims[1];
    const start = batchIndex * seqLen;
    const out = new Array<number>(tokenCount);
    for (let i = 0; i < tokenCount; i += 1) {
      out[i] = mask.data[start + i];
    }
    return out;
  }
  return undefined;
}

function meanPool(
  data: ArrayLike<number>,
  tokenCount: number,
  hiddenSize: number,
  mask?: ArrayLike<number>,
): number[] {
  return meanPoolAt(data, 0, tokenCount, hiddenSize, mask);
}

function meanPoolAt(
  data: ArrayLike<number>,
  baseOffset: number,
  tokenCount: number,
  hiddenSize: number,
  mask?: ArrayLike<number>,
): number[] {
  const result = new Array<number>(hiddenSize).fill(0);
  let maskSum = 0;
  for (let tokenIndex = 0; tokenIndex < tokenCount; tokenIndex += 1) {
    const weight = mask ? mask[tokenIndex] : 1;
    if (weight === 0) continue;
    maskSum += weight;
    const offset = baseOffset + tokenIndex * hiddenSize;
    for (let i = 0; i < hiddenSize; i += 1) {
      result[i] += data[offset + i] * weight;
    }
  }
  const divisor = maskSum > 0 ? maskSum : 1;
  for (let i = 0; i < hiddenSize; i += 1) {
    result[i] /= divisor;
  }
  return result;
}

function normalizeVector(vector: number[]): number[] {
  let magnitudeSquared = 0;
  for (const value of vector) {
    magnitudeSquared += value * value;
  }

  if (magnitudeSquared === 0) {
    return vector;
  }

  const magnitude = Math.sqrt(magnitudeSquared);
  return vector.map((value) => value / magnitude);
}
