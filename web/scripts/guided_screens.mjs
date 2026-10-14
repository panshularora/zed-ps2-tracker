// After-screenshots + demo video of the guided replay against a running stack.
//   BASE_URL=http://localhost:5173 OUT=../screens node scripts/guided_screens.mjs
import { mkdirSync, renameSync } from "node:fs";
import { chromium } from "playwright-core";

const BASE = process.env.BASE_URL ?? "http://localhost:5173";
const OUT = process.env.OUT ?? "screens";
const CHROME = process.env.CHROME ?? "/usr/bin/google-chrome";
mkdirSync(OUT, { recursive: true });
const br = await chromium.launch({ executablePath: CHROME, args: ["--autoplay-policy=no-user-gesture-required"] });
const problems = [];
const watch = (p) => { p.on("pageerror", (e) => problems.push(e.message)); p.on("console", (m) => m.type() === "error" && problems.push(m.text())); };
const shot = async (p, n) => { await p.screenshot({ path: `${OUT}/${n}.png` }); console.log(`saved ${OUT}/${n}.png`); };
const caption = (p) => p.getByTestId("guided-caption").textContent();

const p = await br.newPage({ viewport: { width: 1440, height: 1100 } });
watch(p);
await p.goto(`${BASE}/dashboard/scenarios`, { waitUntil: "networkidle" });
await p.getByText("Counting at a gate").click();
await p.waitForSelector("[data-testid=guided-player] video");
await p.waitForTimeout(1200);
await shot(p, "after_scenarios_1_intro");
await p.waitForSelector("text=Continue", { timeout: 30000 });
await p.waitForTimeout(300);
await shot(p, "after_scenarios_2_question_paused");
await p.click("text=Continue");
await p.waitForSelector("[data-testid=guided-stills]", { timeout: 30000 });
await p.waitForTimeout(400);
await shot(p, "after_scenarios_3_answer_stills");
await p.click("text=Continue");
await p.waitForTimeout(700);
await shot(p, "after_scenarios_4_stake_line");
// moment 2 (counted twice) answer
await p.getByTestId("moment-dot").nth(1).click();
await p.waitForSelector("text=Continue", { timeout: 30000 }); await p.click("text=Continue");
await p.waitForSelector("[data-testid=guided-stills]", { timeout: 30000 });
await p.click("text=Continue");
await p.getByRole("button", { name: "ByteTrack (baseline)" }).click();
await p.waitForTimeout(900);
await shot(p, "after_scenarios_5_baseline_view");
await p.getByRole("button", { name: "Free play" }).click();
await p.waitForTimeout(1500);
await shot(p, "after_scenarios_6_free_play");
console.log("caption:", await caption(p));

await p.goto(`${BASE}/dashboard/studio`, { waitUntil: "networkidle" });
await p.waitForSelector("[data-testid=guided-player] video");
await p.waitForSelector("text=Continue", { timeout: 30000 });
await shot(p, "after_studio_1_turnstile_question");
await p.selectOption("[data-testid=studio-clip]", "bench:MOT17-09").catch((e) => problems.push(String(e)));
await p.waitForTimeout(800);
await shot(p, "after_studio_2_mot17_09");
await p.waitForSelector("text=Continue", { timeout: 40000 }).catch(() => problems.push("no MOT pause"));
await p.click("text=Continue").catch(() => {});
await p.waitForSelector("[data-testid=guided-stills]", { timeout: 30000 }).catch(() => problems.push("no MOT stills"));
await shot(p, "after_studio_3_mot17_09_answer");

const m = await br.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2 });
watch(m);
await m.goto(`${BASE}/dashboard/scenarios`, { waitUntil: "networkidle" });
await m.getByText("Counting at a gate").click();
await m.waitForSelector("text=Continue", { timeout: 30000 }); await m.click("text=Continue");
await m.waitForSelector("[data-testid=guided-stills]", { timeout: 30000 });
await m.locator("[data-testid=guided-stills]").scrollIntoViewIfNeeded();
await m.screenshot({ path: `${OUT}/after_mobile_stills.png` });

// demo video: guided turnstile replay, ~25 s
const ctx = await br.newContext({ viewport: { width: 1280, height: 1000 }, recordVideo: { dir: `${OUT}/_video`, size: { width: 1280, height: 1000 } } });
const v = await ctx.newPage();
watch(v);
await v.goto(`${BASE}/dashboard/scenarios`, { waitUntil: "networkidle" });
await v.getByText("Counting at a gate").click();
await v.waitForSelector("[data-testid=guided-player] video");
await v.evaluate(() => window.scrollTo(0, document.querySelector(".player-caption").getBoundingClientRect().top + window.scrollY - 60));
const t0 = Date.now();
for (let i = 0; i < 4 && Date.now() - t0 < 24000; i++) {
  await v.waitForSelector("text=Continue", { timeout: 30000 });
  await v.waitForTimeout(2600);
  await v.evaluate(() => document.querySelector(".g-continue")?.click()); // no auto-scroll
}
await v.waitForTimeout(Math.max(0, 27000 - (Date.now() - t0)));
const path = await v.video().path();
await ctx.close();
renameSync(path, `${OUT}/guided_turnstile.webm`);
console.log(`saved ${OUT}/guided_turnstile.webm`);
await br.close();
console.log(problems.length ? `PROBLEMS:\n${problems.join("\n")}` : "no page errors");
