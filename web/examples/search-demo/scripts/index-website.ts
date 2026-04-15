/**
 * index-website.ts
 *
 * Quick-and-dirty indexer: reads the HTML pages in the parent directory,
 * strips tags to get text, chunks it, embeds with Xenova/bge-small-en-v1.5
 * via @huggingface/transformers, and writes a Lance table to
 * ../search/site-index.lance.
 *
 * Prerequisites:
 *   cd nodejs && npm run build   # build the native lancedb bindings
 *
 * Usage:
 *   bun run scripts/index-website.ts
 */

import { pipeline } from "@huggingface/transformers";
// Import directly from the monorepo's nodejs package so we always
// test against locally-built Rust code — no npm install needed.
import * as lancedb from "../../../../nodejs/lancedb/index";
import { readFileSync, readdirSync, writeFileSync } from "fs";
import { resolve, join } from "path";

const PAGES_DIR = resolve(import.meta.dirname!, "..");
const OUTPUT = resolve(PAGES_DIR, "search", "site-index.lance");
const MODEL = "Xenova/bge-small-en-v1.5";
const CHUNK_SIZE = 300; // characters per chunk (rough)

/**
 * Parse extra key=value pairs from CLI args into the manifest.
 * Usage: bun run scripts/index-website.ts llm=llm://openai/text-embedding-3-small foo=bar
 */
function parseManifestArgs(): Record<string, string> {
  const extras: Record<string, string> = {};
  for (const arg of process.argv.slice(2)) {
    const eq = arg.indexOf("=");
    if (eq > 0) {
      extras[arg.slice(0, eq)] = arg.slice(eq + 1);
    }
  }
  return extras;
}

// ---- helpers ----------------------------------------------------------------

function stripHtml(html: string): string {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, "")
    .replace(/<style[\s\S]*?<\/style>/gi, "")
    .replace(/<[^>]+>/g, " ")
    .replace(/&[a-z]+;/gi, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function chunk(text: string, size: number): string[] {
  const words = text.split(" ");
  const chunks: string[] = [];
  let buf = "";
  for (const w of words) {
    if (buf.length + w.length + 1 > size && buf.length > 0) {
      chunks.push(buf.trim());
      buf = "";
    }
    buf += w + " ";
  }
  if (buf.trim()) chunks.push(buf.trim());
  return chunks;
}

// ---- main -------------------------------------------------------------------

async function main() {
  console.log(`Loading embedding model: ${MODEL}`);
  const extractor = await pipeline("feature-extraction", MODEL);

  const htmlFiles = readdirSync(PAGES_DIR).filter((f) => f.endsWith(".html"));
  console.log(`Found ${htmlFiles.length} HTML files: ${htmlFiles.join(", ")}`);

  const rows: { text: string; page: string; vector: number[] }[] = [];

  for (const file of htmlFiles) {
    const html = readFileSync(join(PAGES_DIR, file), "utf-8");
    const text = stripHtml(html);
    const chunks = chunk(text, CHUNK_SIZE);
    console.log(`  ${file}: ${chunks.length} chunks`);

    for (const c of chunks) {
      const output = await extractor(c, { pooling: "mean", normalize: true });
      const vector = output.tolist()[0] as number[];
      rows.push({ text: c, page: file, vector });
    }
  }

  console.log(`Total rows: ${rows.length}`);
  console.log(`Writing Lance table to ${OUTPUT}...`);

  const searchDir = resolve(PAGES_DIR, "search");
  const db = await lancedb.connect(searchDir);
  await db.createTable("site-index", rows, { mode: "overwrite" });

  const metadata: Record<string, string> = {
    embeddingModel: MODEL,
    ...parseManifestArgs(),
  };
  const jsonIndexPath = join(searchDir, "site-index.json");
  writeFileSync(jsonIndexPath, JSON.stringify({ metadata, rows }, null, 2));
  console.log(`Wrote JSON search index to ${jsonIndexPath}`);

  // Patch metadata into the published sidecar files
  const tableDir = join(searchDir, "site-index.lance");
  for (const sidecar of ["_web.json", "_snapshot.json"]) {
    const path = join(tableDir, sidecar);
    try {
      const existing = JSON.parse(readFileSync(path, "utf-8"));
      existing.metadata = metadata;
      writeFileSync(path, JSON.stringify(existing, null, 2));
    } catch {
      // sidecar doesn't exist yet — skip
    }
  }
  console.log(`Wrote metadata to sidecars: ${JSON.stringify(metadata)}`);

  console.log("Done! Now serve the example directory and search away.");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
