// Headless-Chrome screenshots of the built viewer (dist/index.html opened from file://).
// usage: node scripts/screenshots.mjs [outDir]   (CHROME=/path/to/chrome to override)
import { chromium } from "playwright-core";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const out = resolve(process.argv[2] ?? "../docs/media");
mkdirSync(out, { recursive: true });
const url = pathToFileURL(resolve("dist/index.html")).href;
const browser = await chromium.launch({
  executablePath: process.env.CHROME ?? "/usr/bin/google-chrome",
  args: ["--no-sandbox"],
});
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));

async function shot(name, { clip, left, right, frame }) {
  await page.goto(`${url}#f=${frame}`);
  await page.waitForSelector("canvas");
  if (clip) await page.selectOption("select >> nth=0", clip);
  if (left) await page.selectOption('select[aria-label="left"]', left);
  if (right) await page.selectOption('select[aria-label="right"]', right);
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${out}/${name}.png` });
  console.log(`${out}/${name}.png`);
}

await shot("viewer_mot17-04_bytetrack_vs_botsort", { clip: "MOT17-04", left: "bytetrack", right: "botsort", frame: 386 });
await shot("viewer_mot17-04_held_decisions", { clip: "MOT17-04", left: "bytetrack", right: "delayed_k5", frame: 3 });
await shot("viewer_mot17-09_gt_vs_bytetrack", { clip: "MOT17-09", left: "gt", right: "bytetrack", frame: 150 });
// timeline seek: clicking the first mark must jump 2 s before it
await page.goto(`${url}#f=1`);
await page.waitForSelector(".mark");
const title = await page.getAttribute(".mark >> nth=0", "title");
await page.click(".mark >> nth=0");
const frameText = await page.textContent(".frame b");
console.log(`clicked mark "${title}" -> frame ${frameText}`);
await browser.close();
if (errors.length) {
  console.error("page errors:", errors);
  process.exit(1);
}
