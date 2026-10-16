// Screenshots for the kept-vs-swap check and the comparisons (1366x768).
//   BASE_URL=http://localhost:4174 OUT=/workspace/zed-shots node scripts/keep_screens.mjs
import { mkdirSync } from "node:fs";
import { chromium } from "playwright-core";

const BASE = process.env.BASE_URL ?? "http://localhost:4174";
const OUT = process.env.OUT ?? "screens";
const CHROME = process.env.CHROME ?? "/usr/bin/google-chrome";
mkdirSync(OUT, { recursive: true });
const br = await chromium.launch({ executablePath: CHROME, args: ["--autoplay-policy=no-user-gesture-required"] });
const problems = [];
const page = async () => {
  const p = await br.newPage({ viewport: { width: 1366, height: 768 } });
  p.on("pageerror", (e) => problems.push(e.message));
  return p;
};
const shot = async (p, n) => { await p.screenshot({ path: `${OUT}/${n}.png` }); console.log(`saved ${OUT}/${n}.png`); };
const openClip = async (p, problem) => {
  await p.goto(`${BASE}/dashboard/scenarios`, { waitUntil: "networkidle" });
  await p.getByText(problem, { exact: true }).click();
  await p.waitForSelector("[data-testid=identity-player] video");
  await p.waitForFunction(() => document.querySelector("[data-testid=identity-player] video")?.readyState >= 2);
  await p.getByRole("button", { name: "Free play" }).click();
  await p.getByRole("button", { name: "Pause" }).click();
};
const at = async (p, t) => {
  await p.evaluate((t) => { const v = document.querySelector("[data-testid=identity-player] video"); v.currentTime = t; }, t);
  await p.waitForTimeout(700);
};
const scrollTo = (p, sel) => p.evaluate((s) => document.querySelector(s)?.scrollIntoView({ block: "start" }), sel);

// 1. turnstile #4: reappears at tracked frame 128 on a different person
{
  const p = await page();
  await openClip(p, "Counting at a gate");
  await p.getByTestId("swap-list").locator("summary").click();
  await at(p, 129.5 / 30);
  // follow #4 (click its box) so the flagged chip is large and the view zooms on it
  await p.evaluate(() => {
    const c = document.querySelector("[data-testid=identity-player] canvas"); const r = c.getBoundingClientRect();
    const x = r.left + ((855 + 98) / 1920) * r.width, y = r.top + ((183 + 145) / 1080) * r.height;
    c.dispatchEvent(new MouseEvent("click", { clientX: x, clientY: y, bubbles: true }));
  });
  await p.waitForTimeout(900);
  await scrollTo(p, "[data-testid=identity-stage]");
  await shot(p, "turnstile-4-flagged");
  await p.close();
}
// 2. verified kept examples (Pexels: seek the video; MOT17-09: boxes only, seek via the moment button)
const kept = [
  ["Counting at a gate", 86, 30, 0],
  ["Counting at a gate", 118, 30, 1],
  ["Recovering after occlusion", 424, 29.97, 0],
  ["Event entrance (Fast Lane)", 100, 30, 0, 0.62],
];
let n = 0;
for (const [problem, back, fps, idx, yFrac] of kept) {
  const p = await page();
  await openClip(p, problem);
  await at(p, (back + 3) / fps);
  await p.evaluate((i) => { const b = document.querySelectorAll("[data-testid=kept-moment]")[i]; b?.classList.add("on"); }, idx);
  if (yFrac) {
    // portrait clip: scroll the person into view, and pin the moment caption as a banner on top
    await p.evaluate((k) => {
      const st = document.querySelector("[data-testid=identity-stage]"); const r = st.getBoundingClientRect();
      window.scrollBy(0, r.top + r.height * k - 330);
      const cap = document.querySelector("[data-testid=story-caption]");
      const ban = cap.cloneNode(true); Object.assign(ban.style, { position: "fixed", left: "230px", right: "20px", top: "52px", zIndex: 50 }); document.body.appendChild(ban);
    }, yFrac);
    await p.waitForTimeout(300);
  } else await scrollTo(p, "[data-testid=watch-line]");
  await shot(p, `kept-examples-${++n}`);
  await p.close();
}
{
  const p = await page();
  await p.goto(`${process.env.BENCH_URL ?? BASE}/dashboard/scenarios`, { waitUntil: "networkidle" });
  await p.getByTestId("bench-card").first().click();
  await p.waitForSelector("[data-testid=kept-moment]");
  await p.getByRole("button", { name: "Free play" }).click();
  await p.getByTestId("kept-moment").first().click(); // seeks to just before the hide (frame 214 → back 235)
  await p.waitForTimeout(1450);
  await p.getByRole("button", { name: "Pause" }).click();
  await scrollTo(p, "[data-testid=watch-line]");
  await shot(p, `kept-examples-${++n}`);
  await p.close();
}
// 3. gate scene comparison strip + a replayed difference
{
  const p = await page();
  await p.goto(`${BASE}/dashboard/live`, { waitUntil: "networkidle" });
  await p.waitForSelector("[data-testid=compare-strip]");
  await p.waitForTimeout(2500);
  await p.getByTestId("show-diffs").click();
  await p.waitForTimeout(2600);
  await scrollTo(p, ".gs-caption");
  await shot(p, "gate-compare-v3");
  await p.close();
}
// 4. real-clip comparison on the turnstile, #4 moment
{
  const p = await page();
  await p.goto(`${BASE}/dashboard/scenarios/compare?clip=subway_turnstile`, { waitUntil: "networkidle" });
  await p.waitForSelector("[data-testid=rc-row]");
  await p.waitForFunction(() => document.querySelector("video")?.readyState >= 2, null, { timeout: 20000 });
  await p.getByRole("button", { name: "❚❚ Pause" }).click();
  await p.locator("[data-testid=rc-table-ours] [data-testid=rc-row]", { hasText: "#4" }).first().click();
  await p.evaluate(() => { document.querySelector("video").currentTime = 128.5 / 30; });
  await p.waitForTimeout(800);
  await scrollTo(p, ".real-compare");
  await shot(p, "real-compare");
  await p.close();
}
await br.close();
if (problems.length) { console.error("page errors:", problems); process.exit(1); }
