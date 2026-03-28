// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

/**
 * Parsed result from an `llm://` URI string.
 *
 * The URI format follows the llm-strings spec:
 *   llm://[label:apiKey@]host/model[?params]
 *
 * @example
 * ```ts
 * const config = parseLlmUri("llm://api.openai.com/text-embedding-3-small?dimensions=1536");
 * // { host: "api.openai.com", model: "text-embedding-3-small", params: { dimensions: "1536" } }
 * ```
 */
export interface LlmUriConfig {
  /** The original URI string */
  raw: string;
  /** Provider hostname (e.g. "api.openai.com") */
  host: string;
  /** Model identifier (e.g. "text-embedding-3-small") */
  model: string;
  /** Optional application label from the userinfo portion */
  label?: string;
  /** Optional API key from the userinfo portion */
  apiKey?: string;
  /** Query string parameters, values are always strings */
  params: Record<string, string>;
}

/**
 * Known provider hosts and their corresponding registry aliases.
 * When an `llm://` URI uses one of these hosts, the registry will
 * look up the embedding function by the mapped alias.
 */
const HOST_TO_PROVIDER: Record<string, string> = {
  "api.openai.com": "openai",
  "api.anthropic.com": "anthropic",
  "generativelanguage.googleapis.com": "google",
  "api.mistral.ai": "mistral",
  "api.cohere.com": "cohere",
  "openrouter.ai": "openrouter",
  "gateway.ai.vercel.sh": "vercel",
};

/**
 * Detect the provider alias from a hostname.
 *
 * Handles exact matches from the known provider map, plus pattern
 * matching for AWS Bedrock (bedrock-runtime.*.amazonaws.com).
 *
 * @param host - The hostname from the URI
 * @returns The provider alias, or undefined if unrecognized
 */
export function detectProvider(host: string): string | undefined {
  if (HOST_TO_PROVIDER[host]) {
    return HOST_TO_PROVIDER[host];
  }
  if (/^bedrock-runtime\..*\.amazonaws\.com$/.test(host)) {
    return "bedrock";
  }
  return undefined;
}

/**
 * Parse an `llm://` URI string into its component parts.
 *
 * Supports the full grammar:
 *   llm://[label:apiKey@]host/model[?key=value&...]
 *
 * @param uri - The URI string to parse. Must start with `llm://`.
 * @returns Parsed configuration
 * @throws Error if the URI scheme is not `llm://`
 * @throws Error if the URI is missing a host or model
 *
 * @example
 * ```ts
 * const config = parseLlmUri("llm://api.openai.com/text-embedding-3-small");
 * // { host: "api.openai.com", model: "text-embedding-3-small", params: {} }
 *
 * const withAuth = parseLlmUri("llm://myapp:sk-abc@api.openai.com/text-embedding-3-small");
 * // { label: "myapp", apiKey: "sk-abc", host: "api.openai.com", model: "text-embedding-3-small", params: {} }
 * ```
 */
export function parseLlmUri(uri: string): LlmUriConfig {
  if (!uri.startsWith("llm://")) {
    throw new Error(
      `Invalid llm:// URI: expected scheme "llm://" but got "${uri.slice(0, uri.indexOf("://") + 3 || 10)}"`,
    );
  }

  let url: URL;
  try {
    // Use a temporary https:// scheme for URL parsing since llm:// is not standard
    url = new URL(uri.replace("llm://", "https://"));
  } catch {
    throw new Error(`Invalid llm:// URI: could not parse "${uri}"`);
  }

  const host = url.hostname;
  if (!host) {
    throw new Error(`Invalid llm:// URI: missing host in "${uri}"`);
  }

  // The model is everything after the first slash in the pathname
  const model = url.pathname.replace(/^\//, "");
  if (!model) {
    throw new Error(`Invalid llm:// URI: missing model in "${uri}"`);
  }

  // Parse userinfo: label:apiKey@
  let label: string | undefined;
  let apiKey: string | undefined;
  if (url.username) {
    label = decodeURIComponent(url.username);
  }
  if (url.password) {
    apiKey = decodeURIComponent(url.password);
  }

  // Parse query parameters
  const params: Record<string, string> = {};
  url.searchParams.forEach((value, key) => {
    params[key] = value;
  });

  return { raw: uri, host, model, label, apiKey, params };
}

/**
 * Check if a string looks like an `llm://` URI.
 */
export function isLlmUri(value: string): boolean {
  return value.startsWith("llm://");
}
