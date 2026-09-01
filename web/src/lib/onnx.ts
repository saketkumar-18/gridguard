/** ONNX Runtime Web session management (WASM backend, single-thread). */
import * as ort from "onnxruntime-web/wasm";

let sessionP: Promise<ort.InferenceSession> | null = null;

export function getSession(): Promise<ort.InferenceSession> {
  if (!sessionP) {
    ort.env.wasm.numThreads = 1; // no COOP/COEP headers on static hosts
    ort.env.wasm.wasmPaths = "/ort/";
    sessionP = ort.InferenceSession.create("/model/model.onnx", {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  }
  return sessionP;
}

/** Score precomputed feature rows -> theft probabilities. */
export async function scoreMatrix(rows: number[][]): Promise<number[]> {
  const sess = await getSession();
  const flat = new Float32Array(rows.length * rows[0].length);
  rows.forEach((r, i) => r.forEach((v, j) => (flat[i * r.length + j] = v)));
  const input = new ort.Tensor("float32", flat, [rows.length, rows[0].length]);
  const out = await sess.run({ float_input: input });
  const key = Object.keys(out).find((k) => out[k].dims.length === 2 && out[k].dims[1] === 2)
    ?? Object.keys(out)[0];
  const probs = out[key].data as Float32Array;
  const step = probs.length / rows.length;
  const res: number[] = [];
  for (let i = 0; i < rows.length; i++) res.push(probs[i * step + 1]);
  return res;
}

export async function scoreOne(features: number[]): Promise<number> {
  return (await scoreMatrix([features]))[0];
}
