import {
  __setTransformersModuleLoaderForTests,
  transformersEmbedder,
} from "@lancedb/lancedb-web/transformers";

const input = document.getElementById("search-input") as HTMLInputElement;
const btn = document.getElementById("search-btn") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLDivElement;
const resultsEl = document.getElementById("results") as HTMLDivElement;

const SEARCH_INDEX_URL = new URL(
  "./search/site-index.json",
  window.location.href,
).href;

type SearchIndexRow = {
  text: string;
  page: string;
  vector: number[];
};

type SearchIndex = {
  metadata?: { embeddingModel?: string };
  rows: SearchIndexRow[];
};

let model: ReturnType<typeof transformersEmbedder> | null = null;
let rows: SearchIndexRow[] = [];

// Parcel will not bundle the library's default indirect dynamic import.
// Load the published browser bundle directly so Parcel does not try to
// resolve the package's Node-only internals during development.
__setTransformersModuleLoaderForTests(
  async () => {
    const specifier =
      "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js";
    return await import(/* webpackIgnore: true */ /* @vite-ignore */ specifier);
  },
);

async function init() {
  statusEl.textContent = "Loading search index & model...";
  try {
    const response = await fetch(SEARCH_INDEX_URL);
    if (!response.ok) {
      throw new Error(
        `Failed to load search index: ${response.status} ${response.statusText}`,
      );
    }
    const searchIndex = (await response.json()) as SearchIndex;
    const modelId = searchIndex.metadata?.embeddingModel;
    if (!modelId) {
      throw new Error("No embeddingModel in table metadata");
    }
    if (!Array.isArray(searchIndex.rows) || searchIndex.rows.length === 0) {
      throw new Error("Search index did not contain any rows");
    }

    console.log(`Using embedding model: ${modelId}`);
    rows = searchIndex.rows;
    model = transformersEmbedder(modelId);
    await model.preload?.();
    statusEl.textContent = "Ready to think.";
  } catch (e) {
    statusEl.textContent = `Failed to load: ${e}`;
    console.error(e);
  }
}

async function doSearch() {
  if (!model || rows.length === 0) return;
  const query = input.value.trim();
  if (!query) return;

  btn.disabled = true;
  statusEl.textContent = "Thinking...";
  resultsEl.innerHTML = "";

  try {
    const { embedding } = await model.embed(query);
    const results = rows
      .map((row) => ({
        ...row,
        score: dotProduct(embedding, row.vector),
      }))
      .sort((left, right) => right.score - left.score)
      .slice(0, 5);

    if (results.length === 0) {
      statusEl.textContent = "No results. Try harder.";
      return;
    }

    statusEl.textContent = `${results.length} result(s)`;

    for (const row of results) {
      const card = document.createElement("div");
      card.className = "result-card";

      const page = row.page;
      const chunk = row.text;
      const score = row.score;

      card.innerHTML = `
        <h3><a href="${page}">${page}</a></h3>
        <p class="snippet">${chunk.slice(0, 200)}${chunk.length > 200 ? "..." : ""}</p>
        <p class="score">score: ${score.toFixed(4)}</p>
      `;
      resultsEl.appendChild(card);
    }
  } catch (e) {
    statusEl.textContent = `Search failed: ${e}`;
    console.error(e);
  } finally {
    btn.disabled = false;
  }
}

function dotProduct(left: number[], right: number[]): number {
  let total = 0;
  const dimensions = Math.min(left.length, right.length);
  for (let i = 0; i < dimensions; i += 1) {
    total += left[i] * right[i];
  }
  return total;
}

btn.addEventListener("click", doSearch);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") doSearch();
});

init();
