// Captures the interface screenshots used in the film. Needs Google Chrome.
const path = require("path");
const fs = require("fs");
const puppeteer = require("puppeteer-core");

const CHROME_PATH = process.env.CHROME_PATH || "C:/Program Files/Google/Chrome/Application/chrome.exe";
const PAGE_URL = "file:///" + path.resolve(__dirname, "..", "index.html").split(path.sep).join("/");
const SHOT_DIRECTORY = path.join(__dirname, "shots");

function wait(milliseconds) {
    return new Promise(function (resolve) { setTimeout(resolve, milliseconds); });
}

async function waitFor(page, expression, timeoutMilliseconds) {
    const deadline = Date.now() + timeoutMilliseconds;
    while (Date.now() < deadline) {
        if (await page.evaluate(expression)) { return; }
        await wait(500);
    }
    throw new Error("timed out waiting for: " + expression);
}

(async function capture() {
    fs.mkdirSync(SHOT_DIRECTORY, { recursive: true });
    const browser = await puppeteer.launch({ executablePath: CHROME_PATH, headless: "new" });
    const page = await browser.newPage();
    await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 });
    async function open(hash) {
        await page.goto("about:blank");
        await page.goto(PAGE_URL + "#" + hash);
        await wait(700);
    }
    async function shoot(name) {
        await page.screenshot({ path: path.join(SHOT_DIRECTORY, name + ".png") });
        console.log("captured", name);
    }

    await open("fresh=1&tab=samples&room=bedroom");
    await page.evaluate("sampleEditor.elementName = 'bed'; sampleEditor.hoverCell = [3, 2]; renderSamplesTab(); 1");
    await shoot("1-samples");

    await open("tab=tiles&room=bedroom");
    await shoot("2-tiles");

    // run the whole pipeline once, then photograph its results
    await open("fresh=1&run=pipeline&seed=5");
    await waitFor(page, "state.activeTab === 'combine' && state.apartments.length >= 30", 600000);
    await wait(8000);
    await page.evaluate("stopCombine(); 1");
    await wait(1500);

    await open("tab=generate&room=bedroom");
    await page.evaluate("document.querySelector('#tab-generate .side').scrollTop = 560; 1");
    await wait(300);
    await shoot("3-parameters");

    await open("tab=generate&room=kitchen");
    await shoot("4-generate");

    await open("tab=combine");
    await shoot("5-combine");
    await open("tab=collection");
    await wait(1200);
    await shoot("6-collection");
    await browser.close();
})();
