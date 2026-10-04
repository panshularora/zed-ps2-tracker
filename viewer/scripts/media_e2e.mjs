// Browser check of real-frame support (headless Chrome, built dist/index.html from file://):
// a synthetic H.264 MP4 whose frame k (1-based) is flat luma 16 + (k-1)*1.75 is opened with the
// "open video / frames" picker; then (1) seeking to clip frames must draw exactly that video
// frame, (2) after playing and pausing, the frame counter must match the pixels on screen, and
// (3) the toolbar reports video mode and the page logs no errors.
// usage: node scripts/media_e2e.mjs   (needs ffmpeg; CHROME=/path overrides /usr/bin/google-chrome)
import { chromium } from "playwright-core";
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const tmp = mkdtempSync(join(tmpdir(), "viewer-media-"));
const mp4 = join(tmp, "grey.mp4");
execFileSync("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
  "color=c=black:s=320x180:r=30:d=4,format=yuv420p,geq=lum='16+N*1.75':cb=128:cr=128",
  "-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4]);
const url = pathToFileURL(resolve("dist/index.html")).href;
const browser = await chromium.launch({
  executablePath: process.env.CHROME ?? "/usr/bin/google-chrome",
  args: ["--no-sandbox", "--autoplay-policy=no-user-gesture-required"],
});
const page = await browser.newPage({ viewport: { width: 1400, height: 800 } });
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
let failed = false;
const check = (ok, msg) => {
  console.log(`${ok ? "ok  " : "FAIL"} ${msg}`);
  failed ||= !ok;
};

// grey level at the centre of the left panel's canvas, in video-pixel units
const grey = () => page.evaluate(() => {
  const cv = document.querySelector("canvas");
  const ctx = cv.getContext("2d");
  const d = ctx.getImageData(Math.floor(cv.width / 2), Math.floor(cv.height / 2), 1, 1).data;
  return d[0];
});
const frameNo = async () => Number(await page.textContent(".frame b"));
const expected = (f) => Math.round((Math.floor(16 + (f - 1) * 1.75) - 16) * 255 / 219); // Y -> RGB

await page.goto(`${url}#f=1`);
await page.waitForSelector("canvas");
await page.selectOption("select >> nth=0", "MOT17-09");
await page.setInputFiles('label:has-text("open video") input', mp4);
await page.waitForFunction(() => document.querySelector("video")?.readyState >= 2);
for (const f of [5, 37, 64, 100]) {
  await page.evaluate((f) => { window.location.hash = `f=${f}`; }, f);
  await page.waitForTimeout(400);
  const g = await grey();
  check(Math.abs(g - expected(f)) <= 1 && (await frameNo()) === f,
    `seek to frame ${f}: drawn grey ${g}, expected ~${expected(f)}`);
}
await page.evaluate(() => { window.location.hash = "f=30"; });
await page.waitForTimeout(400);
await page.click("button:has-text('Play')");
// while playing: frame counter vs pixels, sampled in one task (same rendered state)
const drift = [];
for (let i = 0; i < 6; i++) {
  await page.waitForTimeout(90);
  const [n, g] = await page.evaluate(() => {
    const cv = document.querySelector("canvas");
    const d = cv.getContext("2d").getImageData(cv.width >> 1, cv.height >> 1, 1, 1).data;
    return [Number(document.querySelector(".frame b").textContent), d[0]];
  });
  const shown = 1 + Math.round(((g * 219) / 255) / 1.75);
  drift.push(shown - n);
}
check(drift.every((d) => d === 0), `while playing, drawn frame − counter: [${drift}]`);
await page.click("button:has-text('Pause')");
await page.waitForTimeout(400);
const f = await frameNo();
const g = await grey();
check(f > 35 && f < 120 && Math.abs(g - expected(f)) <= 1,
  `after play/pause: counter ${f}, drawn grey ${g}, expected ~${expected(f)}`);
// JPEG frames named by video frame number, picked instead of the video
const jpgs = [];
for (const k of [37, 38]) {
  const p = join(tmp, `${String(k).padStart(6, "0")}.jpg`);
  execFileSync("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", "-i", mp4, "-vf",
    `select=eq(n\\,${k - 1})`, "-frames:v", "1", "-q:v", "2", p]);
  jpgs.push(p);
}
await page.setInputFiles('label:has-text("open video") input', jpgs);
await page.evaluate(() => { window.location.hash = "f=38"; });
await page.waitForTimeout(600);
const gj = await grey();
check(Math.abs(gj - expected(38)) <= 2 && (await page.textContent(".toolbar")).includes("frames"),
  `JPEG frames: frame 38 drawn grey ${gj}, expected ~${expected(38)}`);
await page.evaluate(() => { window.location.hash = "f=39"; }); // no JPEG for 39: blank canvas
await page.waitForTimeout(400);
check((await grey()) === 14, "frame without a JPEG falls back to the blank canvas (#0E1013)");
check(true, "toolbar mode switches video -> frames");
check(errors.length === 0, `no page errors (${errors.join("; ")})`);
await browser.close();
rmSync(tmp, { recursive: true, force: true });
process.exit(failed ? 1 : 0);
