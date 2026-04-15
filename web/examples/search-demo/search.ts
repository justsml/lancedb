import { searchTable } from "@lancedb/lancedb-web/transformers";

const input = document.getElementById("search-input") as HTMLInputElement;
const btn = document.getElementById("search-btn") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLDivElement;
const resultsEl = document.getElementById("results") as HTMLDivElement;

const TABLE_URL = new URL("./search/site-index.lance", window.location.href).href;

let table: Awaited<ReturnType<typeof searchTable>> | null = null;

async function init() {
  statusEl.textContent = "Loading search index & model...";
  try {
    // Open the table first to read published metadata, then use the
    // embedded model name to configure the embedding model.
    const { openTable } = await import("@lancedb/lancedb-web");
    const raw = await openTable(TABLE_URL);
    const model = raw.metadata.embeddingModel;
    if (!model) {
      throw new Error("No embeddingModel in table metadata");
    }
    console.log(`Using embedding model: ${model}`);
    table = await searchTable(TABLE_URL, model);
    statusEl.textContent = "Ready to think.";
  } catch (e) {
    statusEl.textContent = `Failed to load: ${e}`;
    console.error(e);
  }
}

async function doSearch() {
  if (!table) return;
  const query = input.value.trim();
  if (!query) return;

  btn.disabled = true;
  statusEl.textContent = "Thinking...";
  resultsEl.innerHTML = "";

  try {
    const results = await table.search({ text: query, limit: 5 });
    if (results.length === 0) {
      statusEl.textContent = "No results. Try harder.";
      return;
    }

    statusEl.textContent = `${results.length} result(s)`;

    for (const row of results) {
      const card = document.createElement("div");
      card.className = "result-card";

      const page = row.page as string;
      const chunk = row.text as string;
      const score = row._distance as number;

      card.innerHTML = `
        <h3><a href="${page}">${page}</a></h3>
        <p class="snippet">${chunk.slice(0, 200)}${chunk.length > 200 ? "..." : ""}</p>
        <p class="score">distance: ${score.toFixed(4)}</p>
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

btn.addEventListener("click", doSearch);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") doSearch();
});

init();
