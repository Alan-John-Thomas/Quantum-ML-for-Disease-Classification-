/* Headless-Chrome capture of the NeuroQ-ASD dashboard.
 * Usage: node capture.js <url> <out_dir> [scan_file]
 */
const puppeteer = require("puppeteer-core");
const path = require("path");

const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const URL_ = process.argv[2] || "http://127.0.0.1:5000";
const OUT = process.argv[3] || "../reports/screenshots";
const SCAN = process.argv[4] || null;
const SUBJECT = process.argv[5] || null;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: true,
    args: ["--no-sandbox", "--hide-scrollbars", "--force-device-scale-factor=1"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 960 });

  await page.goto(URL_, { waitUntil: "networkidle0", timeout: 60000 });
  await page.waitForSelector("#subject-select option", { timeout: 30000 });
  await sleep(900); // reveal animations

  // 1) Landing page (hero only, before classification)
  await page.screenshot({ path: path.join(OUT, "ui_home.png") });
  console.log("saved ui_home.png");

  // 2) Pick a subject, reveal ground truth, classify
  const sid = await page.evaluate((want) => {
    const sel = document.getElementById("subject-select");
    const opt = (want && [...sel.options].find((o) => o.value === want)) ||
                [...sel.options].find((o) => o.text.includes("test")) || sel.options[0];
    sel.value = opt.value;
    sel.dispatchEvent(new Event("change"));
    return opt.value;
  }, SUBJECT);
  await page.click("#reveal-label");
  await page.click("#btn-classify");
  await page.waitForFunction(
    () => !document.getElementById("fig-grid").classList.contains("hidden") &&
          document.getElementById("fig-kernel_hist").src.startsWith("data:"),
    { timeout: 120000 }
  );
  await sleep(1500); // scroll + image decode
  await page.evaluate(() => window.scrollTo(0, 0));
  await sleep(400);

  // verdict viewport shot
  await page.screenshot({ path: path.join(OUT, "ui_result_verdict.png") });
  console.log("saved ui_result_verdict.png (subject", sid + ")");

  // full-page shot
  await page.screenshot({ path: path.join(OUT, "ui_result_full.png"), fullPage: true });
  console.log("saved ui_result_full.png");

  // figures section shot (scroll to grid)
  await page.evaluate(() =>
    document.getElementById("fig-grid").scrollIntoView({ block: "start" }));
  await sleep(700);
  await page.screenshot({ path: path.join(OUT, "ui_figures.png") });
  console.log("saved ui_figures.png");

  // 3) Optional: upload a real NIfTI scan -> MRI slices figure
  if (SCAN) {
    const input = await page.$("#file-input");
    await input.uploadFile(SCAN);
    await sleep(400);
    await page.click("#btn-classify");
    await page.waitForFunction(
      () => document.getElementById("fig-mri_slices") &&
            document.getElementById("fig-mri_slices").src.startsWith("data:"),
      { timeout: 180000 }
    );
    await sleep(1500);
    await page.evaluate(() => window.scrollTo(0, 0));
    await sleep(400);
    await page.screenshot({ path: path.join(OUT, "ui_scan_verdict.png") });
    await page.evaluate(() =>
      document.getElementById("card-mri").scrollIntoView({ block: "start" }));
    await sleep(700);
    await page.screenshot({ path: path.join(OUT, "ui_scan_mri.png") });
    console.log("saved ui_scan_verdict.png + ui_scan_mri.png");
  }

  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
