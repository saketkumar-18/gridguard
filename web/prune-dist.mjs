// After `vite build`: remove .wasm files emitted into dist/assets — at runtime
// the ORT loader fetches WASM from /ort/ (public/ort, set via wasmPaths), so
// the bundled copies are dead weight (the JSEP variant alone is ~28 MB).
import { readdirSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const dist = join(dirname(fileURLToPath(import.meta.url)), "dist", "assets");
let removed = 0;
for (const f of readdirSync(dist)) {
  if (f.endsWith(".wasm")) {
    rmSync(join(dist, f));
    removed++;
  }
}
console.log(`pruned ${removed} wasm files from dist/assets (served from /ort/ instead)`);
