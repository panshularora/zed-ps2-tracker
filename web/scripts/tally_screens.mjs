// Screenshots for the moment tally / strip / MOT17-09 opener.
//   BASE_URL=http://localhost:5173 OUT=../screens node scripts/tally_screens.mjs
import { mkdirSync } from "node:fs";
import { chromium } from "playwright-core";
const BASE = process.env.BASE_URL ?? "http://localhost:5173";
const OUT = process.env.OUT ?? "screens";
mkdirSync(OUT, { recursive: true });
const br = await chromium.launch({ executablePath: process.env.CHROME ?? "/usr/bin/google-chrome", args: ["--autoplay-policy=no-user-gesture-required"] });
const problems = [];
const page = async (vp, dpr = 1) => {
  const p = await br.newPage({ viewport: vp, deviceScaleFactor: dpr });
  p.on("pageerror", (e) => problems.push(e.message));
  p.on("console", (m) => m.type() === "error" && problems.push(m.text()));
  return p;
};
const clickContinue = (p) => p.evaluate(() => document.querySelector(".g-continue")?.click());

// opener: Scenarios tab lands on MOT17-09 (measured)
let p = await page({ width: 1440, height: 1100 });
await p.goto(`${BASE}/dashboard/scenarios`, { waitUntil: "networkidle" });
await p.waitForSelector("[data-testid=moment-tally]", { timeout: 20000 });
await p.waitForSelector(".g-continue", { timeout: 30000 });
await clickContinue(p);
await p.waitForSelector("[data-testid=guided-stills]", { timeout: 30000 });
await p.waitForTimeout(300);
await p.screenshot({ path: `${OUT}/mot17_09_opener.png` });
console.log("opener:", await p.getByTestId("clip-caption").textContent());
console.log("opener tally:", await p.getByTestId("moment-tally").textContent());

// desktop pause on the turnstile clip, stepped there with the arrow key
await p.getByText("Counting at a gate").click();
await p.waitForSelector("[data-testid=guided-player] video");
await p.waitForTimeout(800);
await p.keyboard.press("ArrowRight");
await p.keyboard.press("ArrowRight");
await p.waitForSelector(".g-continue", { timeout: 30000 });
await clickContinue(p);
await p.waitForSelector("[data-testid=guided-stills]", { timeout: 30000 });
await p.waitForTimeout(300);
await p.screenshot({ path: `${OUT}/tally_desktop_pause.png` });
console.log("turnstile:", await p.getByTestId("moment-tally").textContent(), "|", await p.getByTestId("moment-progress").textContent());

const m = await page({ width: 390, height: 844 }, 2);
await m.goto(`${BASE}/dashboard/scenarios`, { waitUntil: "networkidle" });
await m.getByText("Counting at a gate").click();
await m.waitForSelector(".g-continue", { timeout: 30000 });
await m.waitForTimeout(300);
const top = await m.evaluate(() => document.querySelector(".g-head").getBoundingClientRect().top + window.scrollY);
await m.screenshot({ path: `${OUT}/tally_mobile_pause.png`, fullPage: true, clip: { x: 0, y: top - 8, width: 390, height: 844 } });
await br.close();
console.log(problems.length ? `PROBLEMS:\n${problems.join("\n")}` : "no page errors");
