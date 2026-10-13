import { chromium } from "playwright";
import fs from "fs";
import path from "path";

const CHROME_PATH = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const BASE = "http://127.0.0.1:8000";
const out = path.join("C:\\Users\\Panshul\\AppData\\Local\\Temp", "zed-journey");
fs.mkdirSync(out, { recursive: true });

const errors = [];

async function main() {
  const browser = await chromium.launch({
    executablePath: fs.existsSync(CHROME_PATH) ? CHROME_PATH : undefined,
    headless: true,
  });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  page.on("pageerror", (err) => errors.push(err.message));

  const resp = await page.goto(BASE, { waitUntil: "networkidle", timeout: 20000 });
  console.log("HTTP", resp?.status(), "title", await page.title());

  await page.waitForSelector("text=Watch the failure", { timeout: 10000 });
  await page.waitForSelector("text=They walk out of view", { timeout: 10000 });
  await page.waitForSelector("text=What's happening here", { timeout: 8000 });
  await page.waitForSelector("text=They're still in the picture", { timeout: 8000 });
  await page.waitForSelector("[data-testid='privacy-note']", { timeout: 8000 });
  await page.waitForSelector("[data-testid='gate-ticker']", { timeout: 8000 });
  console.log("WATCH: heading + live beat + privacy ok");
  const canvasesWatch = await page.locator("canvas").count();
  console.log("WATCH canvases", canvasesWatch);
  if (canvasesWatch < 1) errors.push("watch has no canvas");
  const look = await page.locator("text=LOOK").count();
  console.log("WATCH LOOK labels", look);
  await page.screenshot({ path: path.join(out, "01-watch-desktop.png"), fullPage: true });

  await page.getByTestId("step-compare").click();
  await page.waitForSelector("text=Watch them come back on both sides", { timeout: 8000 });
  await page.waitForSelector("text=LOOK is still missing", { timeout: 8000 });
  await page.getByTestId("jump-switch").click();
  await page.waitForSelector("text=They're back", { timeout: 8000 });
  await page.waitForSelector("text=ID switch", { timeout: 8000 });
  const canvasesCompare = await page.locator("canvas").count();
  console.log("COMPARE canvases", canvasesCompare);
  if (canvasesCompare < 2) errors.push("compare needs two canvases");
  const switchChips = await page.locator("text=ID switch").count();
  console.log("COMPARE ID switch chips", switchChips);
  if (switchChips < 2) errors.push("compare should show ID switch on both trackers after the return");
  await page.screenshot({ path: path.join(out, "02-compare-desktop.png"), fullPage: true });

  await page.getByTestId("step-evidence").click();
  await page.waitForSelector("text=How often this happens", { timeout: 8000 });
  await page.waitForSelector("text=What the gate reports", { timeout: 8000 });
  await page.waitForSelector("text=Same person, same number", { timeout: 8000 });
  await page.waitForSelector("text=What the scores actually mean", { timeout: 8000 });
  await page.waitForSelector("text=Delayed assignment", { timeout: 8000 });
  await page.waitForSelector("[data-testid='papers']", { timeout: 8000 });
  const best = await page.locator("text=best").count();
  console.log("EVIDENCE best badges", best);
  await page.screenshot({ path: path.join(out, "03-evidence-desktop.png"), fullPage: true });

  await page.getByTestId("step-try").click();
  await page.waitForSelector("text=Drop your video", { timeout: 8000 });
  await page.waitForSelector("text=play it under the boxes from this demo", { timeout: 8000 });
  await page.screenshot({ path: path.join(out, "04-try-desktop.png"), fullPage: true });

  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByTestId("step-watch").click();
  await page.waitForSelector("text=They walk out of view", { timeout: 8000 });
  await page.screenshot({ path: path.join(out, "05-watch-mobile.png"), fullPage: true });
  await page.getByTestId("step-compare").click();
  await page.waitForSelector("text=Watch them come back on both sides", { timeout: 8000 });
  await page.screenshot({ path: path.join(out, "06-compare-mobile.png"), fullPage: true });

  await browser.close();
  console.log("errors", errors);
  if (errors.length) process.exit(1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
