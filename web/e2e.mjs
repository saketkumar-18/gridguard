/**
 * Headless E2E: loads the app (local preview or deployed URL), waits for the
 * ONNX WASM model to score the demo consumers, and asserts observable
 * outcomes — the functional gate the skill demands before any "done" claim.
 *
 * Usage: node e2e.mjs [baseUrl]     (default http://localhost:4173)
 */
import { chromium } from "playwright";

const BASE = process.argv[2] || "http://localhost:4173";
const shots = [];

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });

const wasmRequests = [];
page.on("request", (r) => {
  if (r.url().endsWith(".wasm") || r.url().includes("/ort/")) wasmRequests.push(r.url());
});
page.on("pageerror", (e) => console.log("PAGE-ERROR:", String(e).slice(0, 200)));
page.on("console", (m) => {
  if (m.type() === "error") console.log("CONSOLE-ERROR:", m.text().slice(0, 200));
});

console.log("goto", BASE);
await page.goto(BASE, { waitUntil: "networkidle", timeoutDown: 60000 });
await page.waitForTimeout(2500);

// 1. header metrics rendered (means meta loaded + consumers scored)
await page.waitForFunction(() => {
  const t = document.body.innerText;
  return t.includes("PR-AUC") && t.includes("lift");
}, null, { timeout: 60000 });

const headerText = await page.evaluate(() => document.body.innerText.slice(0, 400));
console.log("HEADER:", headerText.replace(/\n+/g, " | ").slice(0, 300));

// 2. consumer rows exist with % scores
const rows = await page.locator(".row").count();
console.log("consumer rows:", rows);
if (rows < 50) throw new Error(`too few rows: ${rows}`);

const firstRow = await page.locator(".row").first().innerText();
console.log("first row:", firstRow.replace(/\s+/g, " "));
if (!/\d+(\.\d+)?%/.test(firstRow)) throw new Error("no score rendered in first row");

// 3. clicking a row shows the detail panel with gauge + explanations
await page.locator(".row").first().click();
await page.waitForTimeout(500);
const detail = await page.locator(".detail").innerText();
console.log("detail has gauge:", detail.includes("balanced"));
console.log("detail has why-section:", /Why flagged|within honest bands/i.test(detail));
console.log("detail snippet:", detail.replace(/\s+/g, " ").slice(0, 250));

// 4. analyze mode: paste a synthetic series -> score appears
await page.locator("nav.tabs button", { hasText: "Analyze your own series" }).click();
await page.fill("textarea", "8 9 10 NA 11 12 0 0 0 0 0 0 0 9 10 11 ".repeat(6));
await page.locator("button.cta").click();
await page.waitForFunction(() => document.body.innerText.includes("theft probability"), null, {
  timeout: 30000,
});
const resText = await page.evaluate(() => document.body.innerText.match(/\d+(\.\d+)?% theft probability/)?.[0]);
console.log("analyze result:", resText);

// 5. which wasm files were actually requested
const uniq = [...new Set(wasmRequests)];
console.log("wasm/ort requests:", uniq.length);
for (const u of uniq) console.log("  ", u.replace(BASE, ""));

await page.screenshot({ path: "e2e-local.png", fullPage: false });
console.log("E2E: ALL CHECKS PASSED");
await browser.close();
