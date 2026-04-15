const { createReadStream, statSync } = require("fs");
const { join } = require("path");

/**
 * Parcel dev-server proxy that serves the static `search/` directory
 * with support for Range requests (required by Lance).
 */
module.exports = function (app) {
  app.use((req, res, next) => {
    if (!req.url.startsWith("/search/")) return next();
    const relative = req.url.slice("/search/".length).split("?")[0];
    const filePath = join(__dirname, "search", relative);

    let stat;
    try {
      stat = statSync(filePath);
    } catch {
      return next();
    }
    if (!stat.isFile()) return next();

    const total = stat.size;

    // Enable CORS for the WASM fetch calls
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Headers", "Range");
    res.setHeader("Access-Control-Expose-Headers", "Content-Range, Content-Length");

    const contentType = filePath.endsWith(".json")
      ? "application/json"
      : "application/octet-stream";

    if (req.method === "OPTIONS") {
      res.writeHead(204);
      return res.end();
    }

    if (req.method === "HEAD") {
      res.writeHead(200, {
        "Content-Length": total,
        "Accept-Ranges": "bytes",
        "Content-Type": contentType,
      });
      return res.end();
    }

    const rangeHeader = req.headers.range;
    if (rangeHeader) {
      const match = rangeHeader.match(/bytes=(\d+)-(\d*)/);
      if (match) {
        const start = parseInt(match[1], 10);
        const end = match[2] ? parseInt(match[2], 10) : total - 1;
        res.writeHead(206, {
          "Content-Range": `bytes ${start}-${end}/${total}`,
          "Accept-Ranges": "bytes",
          "Content-Length": end - start + 1,
          "Content-Type": contentType,
        });
        createReadStream(filePath, { start, end }).pipe(res);
        return;
      }
    }

    res.writeHead(200, {
      "Content-Length": total,
      "Accept-Ranges": "bytes",
      "Content-Type": contentType,
    });
    createReadStream(filePath).pipe(res);
  });
};
