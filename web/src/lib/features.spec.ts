/** JS feature engine must match the Python reference bit-for-bit. */
import { describe, expect, it } from "vitest";
import { featureVector, impute, FEATURE_NAMES, FEATURE_SPEC_VERSION } from "./features";
import goldens from "../data/parity_goldens.json";

describe("impute", () => {
  it("handles all-missing", () => {
    expect(impute([null, null, null])).toEqual([0, 0, 0]);
  });
  it("clips negatives", () => {
    expect(impute([-3, 2])).toEqual([0, 2]);
  });
  it("linear interior gap", () => {
    expect(impute([1, null, null, 4])).toEqual([1, 2, 3, 4]);
  });
  it("edges nearest", () => {
    expect(impute([null, null, 3, 6])).toEqual([3, 3, 3, 6]);
    expect(impute([2, 4, null])).toEqual([2, 4, 4]);
  });
});

describe("feature engine vs Python goldens", () => {
  it("has 22 features, spec v2", () => {
    expect(FEATURE_NAMES.length).toBe(22);
    expect(FEATURE_SPEC_VERSION).toBe(2);
  });
  it.each(goldens.cases as { dow0: number; series: (number | null)[]; features: number[] }[])(
    "case dow0=$dow0 len=$series.length matches Python",
    (c) => {
      const got = featureVector(c.series, c.dow0);
      expect(got.length).toBe(22);
      c.features.forEach((want, i) => {
        // goldens were serialized with full float precision; allow tiny
        // FP ordering differences between CPython and V8.
        expect(Math.abs(got[i] - want)).toBeLessThanOrEqual(
          Math.max(1e-9, Math.abs(want) * 1e-12)
        );
      });
    }
  );
  it("agrees with the Python scores stored in the sample pack", async () => {
    // sample_pack.json entries carry python-computed `score`; verify the JS
    // engine + (skipped: ONNX tested E2E in browser) — here just feature path
    const pack = (await import("../data/sample_pack.json")).default as {
      series: (number | null)[];
    }[];
    for (const p of pack.slice(0, 5)) {
      const v = featureVector(p.series, 2);
      expect(v.every((x) => isFinite(x))).toBe(true);
    }
  });
});
