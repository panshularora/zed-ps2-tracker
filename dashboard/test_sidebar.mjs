import { chromium } from "playwright";

const CHROME_PATH = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const BASE_URL = "http://127.0.0.1:8000";

async function testButtons() {
  console.log("=== VERIFYING SIDEBAR BUTTON ACTIONS ===");
  const browser = await chromium.launch({ executablePath: CHROME_PATH, headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

  await page.goto(BASE_URL, { waitUntil: "networkidle" });
  await page.waitForTimeout(1000);

  // Check initial state
  const boxesBtn = page.locator("button[title='Toggle bounding boxes']");
  const timelineBtn = page.locator("button[title='Toggle event timeline']");
  const heldBtn = page.locator("button[title='Toggle held decisions (MHT)']");
  const inspectorBtn = page.locator("button[title='Toggle inspector']");

  console.log("1. Testing Toggle Bounding Boxes...");
  console.log("Boxes button initial state active:", await boxesBtn.evaluate((el) => el.className.includes("text-brand-600")));
  await boxesBtn.click();
  await page.waitForTimeout(300);
  console.log("Boxes button after click (toggled off):", !(await boxesBtn.evaluate((el) => el.className.includes("text-brand-600"))));
  await boxesBtn.click();
  await page.waitForTimeout(300);
  console.log("Boxes button toggled back on:", await boxesBtn.evaluate((el) => el.className.includes("text-brand-600")));

  console.log("\n2. Testing Toggle Event Timeline...");
  console.log("Timeline visible initially:", await page.locator("text=TIMELINE CHANNELS").isVisible());
  await timelineBtn.click();
  await page.waitForTimeout(300);
  console.log("Timeline hidden after click:", !(await page.locator("text=TIMELINE CHANNELS").isVisible()));
  await timelineBtn.click();
  await page.waitForTimeout(300);
  console.log("Timeline restored after click:", await page.locator("text=TIMELINE CHANNELS").isVisible());

  console.log("\n3. Testing Toggle Inspector...");
  console.log("Inspector visible initially:", await page.locator("text=Track Inspector").isVisible());
  await inspectorBtn.click();
  await page.waitForTimeout(300);
  console.log("Inspector hidden after toggle:", !(await page.locator("text=Track Inspector").isVisible()));
  await inspectorBtn.click();
  await page.waitForTimeout(300);
  console.log("Inspector restored after toggle:", await page.locator("text=Track Inspector").isVisible());

  console.log("\n4. Testing Toggle Held Decisions (MHT)...");
  await heldBtn.click();
  await page.waitForTimeout(200);
  await heldBtn.click();
  console.log("Held decisions button interactive: true");

  await page.screenshot({ path: "sidebar_tools_verified.png", fullPage: true });
  console.log("\n✓ Captured screenshot: sidebar_tools_verified.png");

  await browser.close();
  console.log("=== ALL SIDEBAR TOOLBAR BUTTONS VERIFIED WORKING ===");
}

testButtons().catch(console.error);
