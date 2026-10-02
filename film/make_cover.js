// Draws the film's cover in landscape (1920 x 1080) and portrait (1080 x 1920), in Chinese and in English: node make_cover.js
// Needs renders/cover-landscape.png and renders/cover-portrait.png (blender -b -P render_models.py -- cover).
// The title is set in 文悦新青年体 when its font file is found in film/fonts/ or among the installed
// fonts; otherwise a bold stand-in is used and the script says so.
"use strict";

const fs = require("fs");
const path = require("path");
const { createCanvas, loadImage, GlobalFonts } = require("@napi-rs/canvas");

GlobalFonts.registerFromPath("C:/Windows/Fonts/msyh.ttc", "YaHei");
GlobalFonts.registerFromPath("C:/Windows/Fonts/msyhbd.ttc", "YaHeiBold");
GlobalFonts.registerFromPath("C:/Windows/Fonts/consola.ttf", "Consolas");

function findTitleFont() {
    const directories = [path.join(__dirname, "fonts"), "C:/Windows/Fonts", path.join(process.env.LOCALAPPDATA || "", "Microsoft/Windows/Fonts")];
    for (const directory of directories) {
        if (!fs.existsSync(directory)) { continue; }
        for (const fileName of fs.readdirSync(directory)) {
            const isFontFile = /\.(otf|ttf|ttc)$/i.test(fileName);
            const isNewYouth = /青年|qingnian|xinqingnian|wyue-?xqn|WenYue.*XQN/i.test(fileName);
            // any font file placed in film/fonts/ is taken to be the title font
            if (isFontFile && (isNewYouth || directory === directories[0])) { return path.join(directory, fileName); }
        }
    }
    return null;
}

const titleFontPath = findTitleFont();
let titleFamily = "YaHeiBold";
if (titleFontPath) {
    GlobalFonts.registerFromPath(titleFontPath, "CoverTitle");
    titleFamily = "CoverTitle";
    console.log("title font: " + titleFontPath);
} else {
    console.log("title font: 文悦新青年体 not found — using Microsoft YaHei Bold as a stand-in. Put the font file in film/fonts/ and run again.");
}

const COLOUR = { text: "#f2f5f9", muted: "#a3afbf", accent: "#f59e0b" };

function drawText(context, text, x, y, size, colour, family, letterSpacing) {
    context.font = size + "px " + family;
    context.fillStyle = colour;
    context.letterSpacing = (letterSpacing || 0) + "px";
    context.fillText(text, x, y);
    context.letterSpacing = "0px";
}

// A darker veil behind the text keeps it legible over the render.
function veil(context, x0, y0, x1, y1, fromAlpha, toAlpha) {
    const gradient = context.createLinearGradient(x0, y0, x1, y1);
    gradient.addColorStop(0, "rgba(14,17,22," + fromAlpha + ")");
    gradient.addColorStop(1, "rgba(14,17,22," + toAlpha + ")");
    context.fillStyle = gradient;
    context.fillRect(0, 0, context.canvas.width, context.canvas.height);
}

async function drawLandscape(isEnglish) {
    const canvas = createCanvas(1920, 1080);
    const context = canvas.getContext("2d");
    context.drawImage(await loadImage(path.join(__dirname, "renders", "cover-landscape.png")), 0, 0);
    veil(context, 0, 0, 1200, 0, 0.85, 0);
    context.fillStyle = COLOUR.accent;
    context.fillRect(96, 226, 10, 440);
    drawText(context, "INDEPENDENT STUDY · 2022", 140, 258, 34, COLOUR.accent, "Consolas", 2);
    if (isEnglish) {
        drawText(context, "Wave Function", 132, 420, 128, COLOUR.text, titleFamily, 0);
        drawText(context, "Collapse", 132, 566, 128, COLOUR.text, titleFamily, 0);
        drawText(context, "Generates Floor Plans", 136, 672, 78, COLOUR.text, titleFamily, 0);
        drawText(context, "An apartment generator grown cell by cell", 140, 762, 38, COLOUR.muted, "YaHei");
        drawText(context, "on a 55 cm grid", 140, 812, 38, COLOUR.muted, "YaHei");
    } else {
        drawText(context, "波函数坍缩", 132, 446, 160, COLOUR.text, titleFamily, 2);
        drawText(context, "生成住宅平面", 132, 640, 160, COLOUR.text, titleFamily, 2);
        drawText(context, "Floor Plan Generator Using", 140, 752, 38, COLOUR.muted, "YaHei");
        drawText(context, "the Wave Function Collapse Algorithm", 140, 802, 38, COLOUR.muted, "YaHei");
    }
    drawText(context, "Qian Li", 140, 920, 40, COLOUR.text, "YaHei");
    drawText(context, "WFC", 1690, 1000, 64, COLOUR.accent, "Consolas", 4);
    return canvas;
}

async function drawPortrait(isEnglish) {
    const canvas = createCanvas(1080, 1920);
    const context = canvas.getContext("2d");
    context.drawImage(await loadImage(path.join(__dirname, "renders", "cover-portrait.png")), 0, 0);
    veil(context, 0, 0, 0, 1000, 0.85, 0);
    context.fillStyle = COLOUR.accent;
    context.fillRect(84, 112, 10, 548);
    drawText(context, "INDEPENDENT STUDY · 2022", 128, 148, 36, COLOUR.accent, "Consolas", 2);
    if (isEnglish) {
        drawText(context, "Wave", 118, 316, 190, COLOUR.text, titleFamily, 0);
        drawText(context, "Function", 118, 490, 190, COLOUR.text, titleFamily, 0);
        drawText(context, "Collapse", 118, 664, 190, COLOUR.text, titleFamily, 0);
        drawText(context, "Generates Floor Plans", 124, 808, 78, COLOUR.text, titleFamily, 0);
        drawText(context, "An apartment generator grown cell by cell", 128, 884, 38, COLOUR.muted, "YaHei");
        drawText(context, "on a 55 cm grid", 128, 934, 38, COLOUR.muted, "YaHei");
    } else {
        drawText(context, "波函数", 118, 380, 222, COLOUR.text, titleFamily, 8);
        drawText(context, "坍缩", 118, 618, 222, COLOUR.text, titleFamily, 8);
        drawText(context, "生成住宅平面", 122, 800, 132, COLOUR.text, titleFamily, 6);
        drawText(context, "Floor Plan Generator Using", 128, 884, 38, COLOUR.muted, "YaHei");
        drawText(context, "the Wave Function Collapse Algorithm", 128, 934, 38, COLOUR.muted, "YaHei");
    }
    drawText(context, "Qian Li", 128, 1820, 44, COLOUR.text, "YaHei");
    drawText(context, "WFC", 860, 1826, 68, COLOUR.accent, "Consolas", 4);
    return canvas;
}

(async function main() {
    const outputDirectory = path.join(__dirname, "cover");
    fs.mkdirSync(outputDirectory, { recursive: true });
    fs.writeFileSync(path.join(outputDirectory, "cover_1920x1080.png"), (await drawLandscape(false)).toBuffer("image/png"));
    fs.writeFileSync(path.join(outputDirectory, "cover_1080x1920.png"), (await drawPortrait(false)).toBuffer("image/png"));
    fs.writeFileSync(path.join(outputDirectory, "cover_en_1920x1080.png"), (await drawLandscape(true)).toBuffer("image/png"));
    fs.writeFileSync(path.join(outputDirectory, "cover_en_1080x1920.png"), (await drawPortrait(true)).toBuffer("image/png"));
    console.log("wrote cover/cover_1920x1080.png, cover_1080x1920.png, cover_en_1920x1080.png and cover_en_1080x1920.png");
})();
