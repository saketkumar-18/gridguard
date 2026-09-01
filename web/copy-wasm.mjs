// Copy onnxruntime-web WASM/MJS runtime files into public/ort/ so the app
// serves them same-origin (no CDN dependency; works on any static host).
// Slim: only the wasm-backend pair is needed (the app imports the ./wasm
// subpath; the 28 MB jsep variant would never be requested).
import { copyFileSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
// resolves to .../node_modules/onnxruntime-web/dist/ort.wasm.min.mjs
const wasmEntry = require.resolve("onnxruntime-web/wasm");
const ortDir = dirname(wasmEntry);
const webDir = dirname(fileURLToPath(import.meta.url));
const dest = join(webDir, "public", "ort");
mkdirSync(dest, { recursive: true });

const KEEP = [/^ort-wasm-simd-threaded\.wasm$/, /^ort-wasm-simd-threaded\.mjs$/];
let copied = 0;
for (const f of readdirSync(ortDir)) {
  if (KEEP.some((re) => re.test(f))) {
    copyFileSync(join(ortDir, f), join(dest, f));
    copied++;
  }
}
console.log(`copied ${copied} ort runtime files -> ${dest}`);
mkdirSync(join(webDir, "public", "model"), { recursive: true });
if (copied !== 2) throw new Error(`expected exactly 2 ort files, copied ${copied}`);
