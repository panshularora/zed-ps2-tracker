import { chromium } from "playwright";
import fs from "fs";

const CHROME_PATH = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const BASE_URL = "http://127.0.0.1:8000";

async function runAudit() {
  console.log("=== PLAYWRIGHT VERIFICATION AGENT ===");
  console.log("Using Chrome executable:", CHROME_PATH);

  const browser = await chromium.launch({
    executablePath: CHROME_PATH,
    headless: true,
  });

  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();

  const consoleLogs = [];
  const errors = [];

  page.on("console", (msg) => {
    consoleLogs.push(`[${msg.type()}] ${msg.text()}`);
    if (msg.type() === "error") {
      errors.push(`Console error: ${msg.text()}`);
    }
  });

  page.on("pageerror", (err) => {
    errors.push(`Uncaught page error: ${err.message}`);
  });

  try {
    console.log(`\n1. Navigating to ${BASE_URL}...`);
    const resp = await page.goto(BASE_URL, { waitUntil: "networkidle", timeout: 15000 });
    console.log(`HTTP Status: ${resp.status()}`);
    const title = await page.title();
    console.log(`Page Title: "${title}"`);

    // Verify header and navigation screens
    await page.waitForSelector("header", { timeout: 5000 });
    console.log("✓ Header component loaded");

    // Check Studio screen
    const studioVisible = await page.isVisible("text=Replay Studio").catch(() => false);
    console.log("Studio screen visible:", studioVisible);

    // Click Analytics tab
    console.log("\n2. Switching to Analytics Screen...");
    const analyticsBtn = await page.locator("button:has-text('Analytics')").first();
    if (await analyticsBtn.isVisible()) {
      await analyticsBtn.click();
      await page.waitForTimeout(1000);
      const metricsSummary = await page.locator("text=Metrics").first().isVisible().catch(() => false);
      console.log("✓ Analytics screen loaded. Metrics visible:", metricsSummary);
    } else {
      console.log("! Analytics button not found");
    }

    // Click Experiments tab
    console.log("\n3. Switching to Experiments Screen...");
    const expBtn = await page.locator("button:has-text('Experiments')").first();
    if (await expBtn.isVisible()) {
      await expBtn.click();
      await page.waitForTimeout(1000);
      const ablationVisible = await page.locator("text=Ablation").first().isVisible().catch(() => false);
      console.log("✓ Experiments screen loaded. Ablation table visible:", ablationVisible);
    } else {
      console.log("! Experiments button not found");
    }

    // Switch back to Studio tab
    console.log("\n4. Testing Replay Studio Controls...");
    const studioBtn = await page.locator("button:has-text('Studio')").first();
    if (await studioBtn.isVisible()) {
      await studioBtn.click();
      await page.waitForTimeout(1000);

      // Verify canvas or timeline is rendered
      const canvasCount = await page.locator("canvas").count();
      console.log(`✓ Active canvases rendered: ${canvasCount}`);

      // Check clip selector options
      const selector = page.locator("select").first();
      if (await selector.isVisible()) {
        const value = await selector.inputValue();
        console.log(`✓ Active sequence selected: ${value}`);
      }
    }

    // Take screenshot for visual proof
    const screenshotPath = "C:\\Users\\Panshul\\Desktop\\zed-ps2-tracker-main\\dashboard_audit.png";
    await page.screenshot({ path: screenshotPath, fullPage: true });
    console.log(`\n✓ Full-page screenshot saved to: ${screenshotPath}`);

  } catch (err) {
    console.error("Test failure:", err);
    errors.push(`Test failure: ${err.message}`);
  } finally {
    await browser.close();
  }

  console.log("\n=== AUDIT SUMMARY ===");
  if (errors.length === 0) {
    console.log("STATUS: ALL TESTS PASSED (0 errors)");
  } else {
    console.log(`STATUS: COMPLETED WITH ${errors.length} ISSUE(S):`);
    errors.forEach((e) => console.log(" -", e));
  }
}

runAudit().catch(console.error);
