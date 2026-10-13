// End-to-end screenshots of the four pages against a running stack.
//   BASE_URL=http://127.0.0.1:4173 OUT=../docs/media/web node scripts/screenshots.mjs
// RUN_JOB=1 drives the Demo page: pick the turnstile clip, press "Run tracking",
// wait for the CPU job, play a few seconds and capture the overlay.
import { mkdirSync } from "node:fs";
import { chromium } from "playwright-core";

const BASE = process.env.BASE_URL ?? "http://127.0.0.1:4173";
const OUT = process.env.OUT ?? "screens";
const CLIP = process.env.CLIP ?? "subway_turnstile";
const RUN_JOB = process.env.RUN_JOB !== "0";
const CHROME = process.env.CHROME ?? "/usr/bin/google-chrome";
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({ executablePath: CHROME, args: ["--autoplay-policy=no-user-gesture-required"] });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
const problems = [];
page.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
page.on("console", (m) => m.type() === "error" && problems.push(`console: ${m.text()}`));
const shot = async (name) => {
  await page.screenshot({ path: `${OUT}/${name}.png`, fullPage: true });
  console.log(`saved ${OUT}/${name}.png`);
};

const revealAll = async () => {
  for (let y = 0; y < 12000; y += 400) { await page.evaluate((v) => window.scrollTo(0, v), y); await page.waitForTimeout(60); }
  await page.evaluate(() => window.scrollTo(0, 0)); await page.waitForTimeout(500);
};
await page.goto(`${BASE}/`, { waitUntil: "networkidle" });
await page.waitForTimeout(1500);
await revealAll();
await shot("landing");
await page.setViewportSize({ width: 390, height: 844 });
await page.goto(`${BASE}/`, { waitUntil: "networkidle" });
await page.waitForTimeout(1000);
await revealAll();
await shot("landing-mobile");
await page.setViewportSize({ width: 1440, height: 900 });
await page.goto(`${BASE}/dashboard/scenarios`, { waitUntil: "networkidle" });
await page.getByText("Counting at a gate").click();
await page.waitForSelector("[data-testid=scenario-player] video");
await page.waitForFunction(() => (document.querySelector("video")?.readyState ?? 0) >= 2);
await page.getByTestId("seek-marker").first().click();
await page.waitForTimeout(2300);
await page.evaluate(() => document.querySelector("video")?.pause());
await shot("scenarios");
// Panshul's workstation screens, ported from dashboard/ (needs /api/viewer-clips from serve/)
for (const [name, path, sel] of [
  ["studio", "/dashboard/studio", "[data-testid=gate-ticker]"],
  ["compare", "/dashboard/compare", "[data-testid=jump-switch]"],
  ["analytics", "/dashboard/evidence", "[data-testid=papers]"],
  ["experiments", "/dashboard/experiments", "[data-testid=workstation]"],
]) {
  await page.goto(`${BASE}${path}`, { waitUntil: "networkidle" });
  await page.waitForSelector(sel);
  await page.waitForTimeout(800);
  await shot(name);
}
if (process.env.ONLY_NEW) { await browser.close(); process.exit(0); }
await page.goto(`${BASE}/dashboard`, { waitUntil: "networkidle" });
await page.waitForSelector("[data-testid=tile]");
await shot("overview");

await page.goto(`${BASE}/dashboard/results`, { waitUntil: "networkidle" });
await page.waitForSelector("table");
await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight)); // load the lazy PNG plots
await page.waitForFunction(() => [...document.images].every((i) => i.complete && i.naturalWidth > 0));
await page.evaluate(() => window.scrollTo(0, 0));
await shot("results");

await page.goto(`${BASE}/dashboard/architecture`, { waitUntil: "networkidle" });
await page.getByRole("tab").nth(3).click();
await page.waitForTimeout(500);
await shot("architecture");

await page.goto(`${BASE}/dashboard/demo`, { waitUntil: "networkidle" });
await page.getByLabel("Clip").selectOption(CLIP);
if (RUN_JOB) {
  const t0 = Date.now();
  await page.getByRole("button", { name: "Run tracking" }).click();
  await page.waitForSelector("[data-testid=progress]", { timeout: 30_000 });
  await page.waitForTimeout(4000);
  await shot("demo-progress");
  await page.waitForSelector("[data-testid=metrics]", { timeout: 15 * 60_000 });
  await page.waitForSelector("[data-testid=progress]", { state: "detached", timeout: 15 * 60_000 });
  console.log(`job finished in the UI after ${((Date.now() - t0) / 1000).toFixed(1)} s`);
}
await page.waitForSelector("[data-testid=metrics]", { timeout: 60_000 });
await page.waitForFunction(() => (document.querySelector("video")?.readyState ?? 0) >= 2);
await page.getByRole("button", { name: "Play" }).click();
await page.waitForTimeout(Number(process.env.PLAY_MS ?? 6000));
await page.getByRole("button", { name: "Pause" }).click();
await page.waitForTimeout(400);
await shot("demo");
const counters = await page.getByTestId("counters").innerText();
console.log("counters:", counters.replace(/\s+/g, " "));

await browser.close();
if (problems.length) {
  console.error(problems.join("\n"));
  process.exit(1);
}
