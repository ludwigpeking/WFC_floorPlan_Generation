// Renders the introduction film: node make_film.js
//   --preview 3.5,12,40   write single frames (PNG) at these times instead of the film
//   --output name.mp4     output file name (default: WFC_Floor_Plan_Introduction.mp4)
//   --refresh             generate the rooms and apartments again instead of reusing material_cache.json
//   --language en         the English film: clips from voice-en/, English-only text (default: zh, clips from voice/)
// Needs ffmpeg on the PATH, the narration clips in voice/ (or voice-en/), and the screenshots in shots/.
"use strict";

const fs = require("fs");
const path = require("path");
const childProcess = require("child_process");
const { createCanvas, loadImage, GlobalFonts } = require("@napi-rs/canvas");

const FRAME_WIDTH = 1920;
const FRAME_HEIGHT = 1080;
const FRAMES_PER_SECOND = 30;
const LINE_GAP_SECONDS = 0.45;
const SCENE_LEAD_SECONDS = 0.9;
const SCENE_TAIL_SECONDS = 1.0;
const PROJECT_DIRECTORY = path.resolve(__dirname, "..");
const LANGUAGE = process.argv.indexOf("--language") >= 0 ? process.argv[process.argv.indexOf("--language") + 1] : "zh";
const IS_ENGLISH = LANGUAGE === "en";

const COLOUR = {
    background: "#161b22", panel: "#1e252e", line: "#34404e", text: "#e6ebf1", muted: "#94a1b2",
    accent: "#f59e0b", blue: "#8ec5ff", bad: "#f87171", good: "#4ade80", paper: "#f4f1ea"
};

GlobalFonts.registerFromPath("C:/Windows/Fonts/msyh.ttc", "YaHei");
GlobalFonts.registerFromPath("C:/Windows/Fonts/msyhbd.ttc", "YaHeiBold");
GlobalFonts.registerFromPath("C:/Windows/Fonts/consola.ttf", "Consolas");

// ---------------------------------------------------------------------------
// The generator itself, taken straight out of index.html
// ---------------------------------------------------------------------------

const pageSource = fs.readFileSync(path.join(PROJECT_DIRECTORY, "index.html"), "utf8");
const engineSource = pageSource.split("/* ENGINE-BEGIN */")[1].split("/* ENGINE-END */")[0];
const ENGINE_NAMES = ["WALL_CODE", "WINDOW_CODE", "UNKNOWN_CODE", "CELL_SIZE_METRES", "ELEMENT_LIBRARY", "DEFAULT_PALETTES", "DRAWING_COLOURS",
    "elementCodes", "createDefaultRoomConfigurations", "createDefaultApartmentConfiguration", "codeColour", "createRandomSource", "createGrid",
    "gridFromRows", "gridToRows", "gridKey", "padGridWithWall", "extractTileLibrary", "generateAttempt", "evaluateRoomPlan", "evaluateCurve",
    "createRoomPlan", "searchApartments", "buildPlanDrawing", "renderPlanDrawing", "renderCodeGrid", "findAccessOpenings",
    "buildPlanScene", "renderPlanScene", "planSceneScale", "setCanvasFactory", "planSignature"];
const engine = new Function(engineSource + "\nreturn {" + ENGINE_NAMES.join(",") + "};")();
engine.setCanvasFactory(createCanvas);
const builtInSamples = JSON.parse(pageSource.split("id=\"builtInSamples\">")[1].split("</script>")[0]);

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function clamp01(value) { return Math.max(0, Math.min(1, value)); }
function ramp(time, startTime, durationSeconds) { return clamp01((time - startTime) / durationSeconds); }
function ease(value) { const x = clamp01(value); return x * x * (3 - 2 * x); }
function mix(from, to, amount) { return from + (to - from) * amount; }
function hashNoise(seed) {
    const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453;
    return x - Math.floor(x);
}

// On-screen text: the Chinese film shows both languages, the English film only English.
function bilingual(chinese, english) { return IS_ENGLISH ? english : chinese + "  " + english; }
function primary(chinese, english) { return IS_ENGLISH ? english : chinese; }
function secondary(english) { return IS_ENGLISH ? "" : english; }

// When, within its scene, the narration reaches a keyword (estimated from the keyword's position in the line).
const missingKeywords = new Set();
function cueTime(scene, lineIndex, keyword, englishKeyword) {
    const line = scene.lines[lineIndex];
    const narrationText = IS_ENGLISH ? line.en.toLowerCase() : line.zh;
    const searchedKeyword = IS_ENGLISH ? (englishKeyword || keyword).toLowerCase() : keyword;
    const foundPosition = narrationText.indexOf(searchedKeyword);
    if (foundPosition < 0 && !missingKeywords.has(searchedKeyword)) {
        missingKeywords.add(searchedKeyword);
        console.warn("cue keyword not in narration line " + line.name + ": " + searchedKeyword);
    }
    return scene.lineStart[lineIndex] + line.durationSeconds * Math.max(0, foundPosition) / narrationText.length;
}

function drawPhotoCard(context, image, centreX, centreY, maximumWidth, maximumHeight, alpha, zoom) {
    if (alpha <= 0) { return; }
    const scale = Math.min(maximumWidth / image.width, maximumHeight / image.height) * (zoom || 1);
    const width = image.width * scale, height = image.height * scale;
    context.save();
    context.globalAlpha *= alpha;
    context.shadowColor = "rgba(0,0,0,0.6)";
    context.shadowBlur = 36;
    context.fillStyle = "#f4f1ea";
    context.fillRect(centreX - width / 2 - 8, centreY - height / 2 - 8, width + 16, height + 16);
    context.shadowBlur = 0;
    context.drawImage(image, centreX - width / 2, centreY - height / 2, width, height);
    context.restore();
}

// Frames rendered by Blender are decoded one at a time: the frame loop loads the one
// the coming film frame needs before drawing it.
const renderedFrameCache = { index: -1, image: null };
function heroFrameIndex(time, material) {
    const scene = SCENES.filter(function (candidate) { return candidate.name === "model"; })[0];
    if (time < scene.start || time >= scene.end) { return -1; }
    return Math.max(0, Math.min(material.heroFrameCount - 1, Math.round((time - scene.start) * FRAMES_PER_SECOND)));
}
async function loadRenderedFrame(time, material) {
    const frameIndex = heroFrameIndex(time, material);
    if (frameIndex < 0 || frameIndex === renderedFrameCache.index) { return; }
    renderedFrameCache.image = await loadImage(path.join(__dirname, "renders", "hero", String(frameIndex).padStart(4, "0") + ".png"));
    renderedFrameCache.index = frameIndex;
}

// Every position that can collapse next, with its candidate tiles and their entropy.
function openPositions(grid, library) {
    const positions = [];
    for (let tileRow = 0; tileRow < grid.rowCount - 1; tileRow++) {
        for (let tileColumn = 0; tileColumn < grid.columnCount - 1; tileColumn++) {
            const baseIndex = tileRow * grid.columnCount + tileColumn;
            const known = [grid.cells[baseIndex], grid.cells[baseIndex + 1], grid.cells[baseIndex + grid.columnCount], grid.cells[baseIndex + grid.columnCount + 1]];
            let unknownCount = 0, activeCount = 0, hasPassage = false;
            for (const code of known) {
                if (code === engine.UNKNOWN_CODE) { unknownCount++; }
                else if (code !== engine.WALL_CODE && code !== engine.WINDOW_CODE) { activeCount++; }
                if (library.passageCodes.has(code)) { hasPassage = true; }
            }
            if (unknownCount === 0 || activeCount === 0) { continue; }
            const candidates = library.regularTileIndices.filter(function (tileIndex) {
                const codes = library.tiles[tileIndex].codes;
                for (let corner = 0; corner < 4; corner++) {
                    if (known[corner] !== engine.UNKNOWN_CODE && known[corner] !== codes[corner]) { return false; }
                    if (known[corner] === engine.UNKNOWN_CODE && library.passageCodes.has(codes[corner]) && !hasPassage) { return false; }
                }
                return true;
            });
            if (candidates.length === 0) { continue; }
            let frequencySum = 0;
            for (const tileIndex of candidates) { frequencySum += library.tiles[tileIndex].frequency; }
            let entropy = 0;
            for (const tileIndex of candidates) {
                const probability = library.tiles[tileIndex].frequency / frequencySum;
                entropy -= probability * Math.log(probability);
            }
            const frequencies = candidates.map(function (tileIndex) { return library.tiles[tileIndex].frequency; }).sort(function (first, second) { return second - first; });
            positions.push({ tileRow: tileRow, tileColumn: tileColumn, entropy: entropy, frequencies: frequencies });
        }
    }
    return positions;
}

function drawText(context, text, x, y, size, colour, options) {
    const settings = options || {};
    context.save();
    context.globalAlpha *= settings.alpha === undefined ? 1 : settings.alpha;
    context.font = size + "px " + (settings.family || "YaHei");
    context.textAlign = settings.align || "left";
    context.textBaseline = settings.baseline || "alphabetic";
    context.fillStyle = colour;
    context.fillText(text, x, y);
    context.restore();
}

function roundedRectangle(context, x, y, width, height, radius) {
    context.beginPath();
    context.moveTo(x + radius, y);
    context.arcTo(x + width, y, x + width, y + height, radius);
    context.arcTo(x + width, y + height, x, y + height, radius);
    context.arcTo(x, y + height, x, y, radius);
    context.arcTo(x, y, x + width, y, radius);
    context.closePath();
}

function strokeLine(context, x1, y1, x2, y2, colour, lineWidth) {
    context.strokeStyle = colour;
    context.lineWidth = lineWidth;
    context.beginPath();
    context.moveTo(x1, y1);
    context.lineTo(x2, y2);
    context.stroke();
}

// A plan rendered once into its own canvas, then reused every frame.
const spriteCache = new Map();
function planSprite(plan, cellPixelSize, viewMode) {
    if (!spriteCache.has(plan)) { spriteCache.set(plan, new Map()); }
    const sprites = spriteCache.get(plan);
    const key = cellPixelSize + ":" + viewMode;
    if (!sprites.has(key)) {
        const padding = Math.ceil(cellPixelSize * 0.6);
        const canvas = createCanvas(Math.ceil(plan.columnCount * cellPixelSize + 2 * padding), Math.ceil(plan.rowCount * cellPixelSize + 2 * padding));
        const context = canvas.getContext("2d");
        if (viewMode === "drawing") {
            engine.renderPlanDrawing(context, engine.buildPlanDrawing(plan), padding, padding, cellPixelSize, plan.ownerRoomTypes.length > 2);
        } else {
            for (let row = 0; row < plan.rowCount; row++) {
                for (let column = 0; column < plan.columnCount; column++) {
                    const cellIndex = row * plan.columnCount + column;
                    if (plan.owners[cellIndex] === 0) { continue; }
                    context.fillStyle = engine.codeColour(engine.DEFAULT_PALETTES[plan.ownerRoomTypes[plan.owners[cellIndex]]], plan.cells[cellIndex]);
                    context.fillRect(padding + column * cellPixelSize, padding + row * cellPixelSize, cellPixelSize - 1, cellPixelSize - 1);
                }
            }
        }
        sprites.set(key, canvas);
    }
    return sprites.get(key);
}

function drawPlan(context, plan, centreX, centreY, cellPixelSize, viewMode, alpha) {
    if (alpha <= 0) { return; }
    const sprite = planSprite(plan, cellPixelSize, viewMode);
    context.save();
    context.globalAlpha *= alpha;
    context.drawImage(sprite, Math.round(centreX - sprite.width / 2), Math.round(centreY - sprite.height / 2));
    context.restore();
}

// A plan as a 3D model, drawn through a scratch canvas so that it can fade as a whole.
const sceneCache = new Map();
const scratchCanvases = new Map();
function drawModel(context, plan, centreX, centreY, width, height, yawRadians, pitchRadians, alpha, heightScale) {
    if (alpha <= 0) { return; }
    if (!sceneCache.has(plan)) { sceneCache.set(plan, engine.buildPlanScene(plan)); }
    const scene = sceneCache.get(plan);
    const sizeKey = width + "x" + height;
    if (!scratchCanvases.has(sizeKey)) { scratchCanvases.set(sizeKey, createCanvas(width, height)); }
    const scratch = scratchCanvases.get(sizeKey);
    const scratchContext = scratch.getContext("2d");
    scratchContext.clearRect(0, 0, width, height);
    engine.renderPlanScene(scratchContext, scene, width / 2, height / 2, engine.planSceneScale(scene, width * 0.9, height * 0.74, pitchRadians), yawRadians, pitchRadians, heightScale);
    context.save();
    context.globalAlpha *= alpha;
    context.drawImage(scratch, Math.round(centreX - width / 2), Math.round(centreY - height / 2));
    context.restore();
}

function fitCellSize(plan, maximumWidth, maximumHeight) {
    return Math.floor(Math.min(maximumWidth / plan.columnCount, maximumHeight / plan.rowCount));
}

function drawImageCard(context, image, centreX, centreY, maximumWidth, maximumHeight, alpha, caption, pixelated) {
    if (alpha <= 0) { return; }
    const scale = Math.min(maximumWidth / image.width, maximumHeight / image.height);
    const width = image.width * scale, height = image.height * scale;
    context.save();
    context.globalAlpha *= alpha;
    context.shadowColor = "rgba(0,0,0,0.5)";
    context.shadowBlur = 30;
    context.fillStyle = COLOUR.paper;
    context.fillRect(centreX - width / 2 - 10, centreY - height / 2 - 10, width + 20, height + 20);
    context.shadowBlur = 0;
    context.imageSmoothingEnabled = !pixelated;
    context.drawImage(image, centreX - width / 2, centreY - height / 2, width, height);
    context.imageSmoothingEnabled = true;
    if (caption) { drawText(context, caption, centreX, centreY + height / 2 + 48, 24, COLOUR.muted, { align: "center" }); }
    context.restore();
}

// ---------------------------------------------------------------------------
// Material: rooms, apartments and collapse traces, generated with fixed seeds
// ---------------------------------------------------------------------------

function prepareMaterial(refresh) {
    const configurations = engine.createDefaultRoomConfigurations();
    const material = { configurations: configurations, libraries: {}, rooms: {}, sampleGrids: {} };
    const cachePath = path.join(__dirname, "material_cache.json");
    const cache = !refresh && fs.existsSync(cachePath) ? JSON.parse(fs.readFileSync(cachePath, "utf8")) : null;
    for (const roomType of Object.keys(configurations)) {
        material.sampleGrids[roomType] = builtInSamples[roomType].map(engine.gridFromRows);
        material.libraries[roomType] = engine.extractTileLibrary(material.sampleGrids[roomType], configurations[roomType], false);
        if (cache) {
            material.rooms[roomType] = cache.rooms[roomType].map(function (rows) {
                const evaluation = engine.evaluateRoomPlan(engine.gridFromRows(rows), configurations[roomType]);
                evaluation.plan = engine.createRoomPlan(evaluation.grid, roomType);
                return evaluation;
            });
            continue;
        }
        const random = engine.createRandomSource(5);
        const found = new Map();
        let lastNewAttempt = 0;
        for (let attempt = 0; attempt < 300000 && found.size < 150 && attempt - lastNewAttempt < 30000; attempt++) {
            const outcome = engine.generateAttempt(material.libraries[roomType], configurations[roomType], random, null, false);
            if (!outcome.success) { continue; }
            const evaluation = engine.evaluateRoomPlan(outcome.grid, configurations[roomType]);
            if (!evaluation.valid || evaluation.totalScore < configurations[roomType].scoreThreshold) { continue; }
            const key = engine.planSignature(engine.createRoomPlan(evaluation.grid, roomType));
            if (!found.has(key)) { found.set(key, evaluation); lastNewAttempt = attempt; }
        }
        material.rooms[roomType] = Array.from(found.values()).sort(function (first, second) { return second.totalScore - first.totalScore; });
        for (const room of material.rooms[roomType]) { room.plan = engine.createRoomPlan(room.grid, roomType); }
        console.log("  " + roomType + ": " + material.rooms[roomType].length + " schemes");
    }

    // apartments: only the best one of each living room, and only above the page's own threshold
    if (cache) {
        material.apartments = cache.apartments.map(function (stored) {
            return {
                plan: { rowCount: stored.rowCount, columnCount: stored.columnCount, cells: Int32Array.from(stored.cells), owners: Int16Array.from(stored.owners), ownerRoomTypes: stored.ownerRoomTypes },
                quantities: stored.quantities, totalScore: stored.totalScore, termScores: stored.termScores, roomScores: stored.roomScores
            };
        });
        console.log("  material read from material_cache.json (use --refresh to generate it again)");
    } else {
        const apartmentConfiguration = engine.createDefaultApartmentConfiguration();
        const random = engine.createRandomSource(3);
        const apartments = [];
        const apartmentSignatures = new Set();
        for (const livingRoom of material.rooms.living) {
            const candidates = { bedroom: material.rooms.bedroom, bathroom: material.rooms.bathroom, kitchen: material.rooms.kitchen };
            let best = null, stepCount = 0;
            for (const item of engine.searchApartments(livingRoom, candidates, apartmentConfiguration, random)) {
                if (item.apartment && (!best || item.apartment.totalScore > best.totalScore)) { best = item.apartment; }
                stepCount++;
                if (stepCount > 600) { break; }
            }
            if (best && !apartmentSignatures.has(engine.planSignature(best.plan))) {
                apartmentSignatures.add(engine.planSignature(best.plan));
                apartments.push(best);
            }
        }
        apartments.sort(function (first, second) { return second.totalScore - first.totalScore; });
        material.apartments = apartments.slice(0, 18);
        console.log("  apartments: " + apartments.length + " living rooms combined, best " + apartments[0].totalScore.toFixed(2) + ", 18th " + material.apartments[material.apartments.length - 1].totalScore.toFixed(2));
        const stored = { rooms: {}, apartments: [] };
        for (const roomType of Object.keys(configurations)) {
            material.rooms[roomType] = material.rooms[roomType].slice(0, 12);
            stored.rooms[roomType] = material.rooms[roomType].map(function (room) { return engine.gridToRows(room.grid); });
        }
        stored.apartments = material.apartments.map(function (apartment) {
            return {
                rowCount: apartment.plan.rowCount, columnCount: apartment.plan.columnCount, cells: Array.from(apartment.plan.cells), owners: Array.from(apartment.plan.owners),
                ownerRoomTypes: apartment.plan.ownerRoomTypes, quantities: apartment.quantities, totalScore: apartment.totalScore,
                termScores: apartment.termScores, roomScores: apartment.roomScores
            };
        });
        fs.writeFileSync(cachePath, JSON.stringify(stored));
    }

    // a collapse story for the bedroom: a long failure, a short failure, many more, then a success
    const bedroomConfiguration = configurations.bedroom, bedroomLibrary = material.libraries.bedroom;
    for (let seed = 1; seed < 400 && !material.collapseStory; seed++) {
        const storyRandom = engine.createRandomSource(seed);
        const attempts = [];
        for (let attemptIndex = 0; attemptIndex < 320; attemptIndex++) {
            const outcome = engine.generateAttempt(bedroomLibrary, bedroomConfiguration, storyRandom, null, true);
            attempts.push(outcome);
            if (!outcome.success) { continue; }
            const evaluation = engine.evaluateRoomPlan(outcome.grid, bedroomConfiguration);
            if (evaluation.valid && evaluation.totalScore >= 23.5) { outcome.evaluation = evaluation; break; }
        }
        const last = attempts[attempts.length - 1];
        function collapseStepCount(outcome) { return outcome.trace.filter(function (step) { return step.phase === "collapse"; }).length; }
        if (!last.evaluation || attempts.length < 40) { continue; }
        if (attempts[0].success || collapseStepCount(attempts[0]) < 11 || collapseStepCount(attempts[0]) > 22) { continue; }
        if (attempts[1].success || attempts[1].trace.length < 7) { continue; }
        last.evaluation.plan = engine.createRoomPlan(last.evaluation.grid, "bedroom");
        material.collapseStory = { attempts: attempts, seed: seed };
    }
    console.log("  collapse story: seed " + material.collapseStory.seed + ", " + material.collapseStory.attempts.length + " attempts");

    // one successful living-room growth for the opening explanation
    const livingRandom = engine.createRandomSource(21);
    for (let attemptIndex = 0; attemptIndex < 20000 && !material.livingGrowth; attemptIndex++) {
        const outcome = engine.generateAttempt(material.libraries.living, configurations.living, livingRandom, null, true);
        if (!outcome.success) { continue; }
        const evaluation = engine.evaluateRoomPlan(outcome.grid, configurations.living);
        if (evaluation.valid && evaluation.totalScore >= configurations.living.scoreThreshold) {
            evaluation.plan = engine.createRoomPlan(evaluation.grid, "living");
            material.livingGrowth = { outcome: outcome, evaluation: evaluation };
        }
    }
    return material;
}

function emptyWorkingGrid(configuration) {
    const grid = engine.createGrid(configuration.gridRowCount, configuration.gridColumnCount, engine.UNKNOWN_CODE);
    for (let row = 0; row < grid.rowCount; row++) {
        for (let column = 0; column < grid.columnCount; column++) {
            if (row === 0 || column === 0 || row === grid.rowCount - 1 || column === grid.columnCount - 1) { grid.cells[row * grid.columnCount + column] = engine.WALL_CODE; }
        }
    }
    return grid;
}

// The working grid after the first stepCount steps of an attempt.
function replayAttempt(outcome, stepCount, configuration, library) {
    const grid = emptyWorkingGrid(configuration);
    let lastStep = null;
    for (let stepIndex = 0; stepIndex < Math.min(stepCount, outcome.trace.length); stepIndex++) {
        const step = outcome.trace[stepIndex];
        lastStep = step;
        if (step.kind === "closed") { grid.cells.set(outcome.grid.cells); continue; }
        if (step.kind !== "place") { continue; }
        const codes = library.tiles[step.tileIndex].codes;
        const baseIndex = step.tileRow * grid.columnCount + step.tileColumn;
        const cellIndices = [baseIndex, baseIndex + 1, baseIndex + grid.columnCount, baseIndex + grid.columnCount + 1];
        for (let cornerIndex = 0; cornerIndex < 4; cornerIndex++) {
            if (grid.cells[cellIndices[cornerIndex]] === engine.UNKNOWN_CODE) { grid.cells[cellIndices[cornerIndex]] = codes[cornerIndex]; }
        }
    }
    return { grid: grid, lastStep: lastStep };
}

function drawWorkingGrid(context, grid, palette, originX, originY, cellPixelSize, time, shimmer) {
    engine.renderCodeGrid(context, grid, palette, originX, originY, cellPixelSize, true);
    if (!shimmer) { return; }
    // undecided cells flicker between everything they could still become
    const paletteCodes = [];
    for (const elementName of palette) {
        if (elementName !== "wall") { paletteCodes.push(engine.elementCodes(elementName)[0]); }
    }
    const tick = Math.floor(time * 5);
    for (let row = 0; row < grid.rowCount; row++) {
        for (let column = 0; column < grid.columnCount; column++) {
            if (grid.cells[row * grid.columnCount + column] !== engine.UNKNOWN_CODE) { continue; }
            for (let quadrant = 0; quadrant < 4; quadrant++) {
                const noise = hashNoise(row * 97 + column * 13 + quadrant * 7 + tick * 31);
                context.globalAlpha = 0.16 + 0.1 * hashNoise(row * 3 + column * 17 + tick);
                context.fillStyle = engine.codeColour(palette, paletteCodes[Math.floor(noise * paletteCodes.length)]);
                const half = cellPixelSize / 2;
                context.fillRect(originX + column * cellPixelSize + (quadrant % 2) * half + 2, originY + row * cellPixelSize + Math.floor(quadrant / 2) * half + 2, half - 3, half - 3);
            }
        }
    }
    context.globalAlpha = 1;
}

// ---------------------------------------------------------------------------
// Scenes. Each draw(context, time, scene, material) gets time in seconds from
// the start of its scene; scene.lineStart[i] is when narration line i begins.
// ---------------------------------------------------------------------------

const SCENES = [
    { name: "title", chapter: "", tailSeconds: 0.6, draw: drawTitleScene },
    { name: "algorithm", chapter: primary("A · 算法  The algorithm", "A · The algorithm"), tailSeconds: 0, draw: drawAlgorithmScene },
    { name: "grid", chapter: primary("B · 55 厘米网格  The 55 cm grid", "B · The 55 cm grid"), tailSeconds: 0, draw: drawGridScene },
    { name: "pixel", chapter: primary("C · 像素化  Pixelating a plan", "C · Pixelating a plan"), tailSeconds: 0, draw: drawPixelScene },
    { name: "library", chapter: primary("D · 连接库  The library of valid connections", "D · The library of valid connections"), tailSeconds: 0.6, draw: drawLibraryScene },
    { name: "collapse", chapter: primary("E · 坍缩  Collapse", "E · Collapse"), tailSeconds: 3.6, draw: drawCollapseScene },
    { name: "fitness", chapter: primary("F · 评分  Fitness", "F · Fitness"), tailSeconds: 2.2, draw: drawFitnessScene },
    { name: "whole", chapter: primary("G · 从局部到整体  From parts to the whole", "G · From parts to the whole"), tailSeconds: 1.6, draw: drawWholeScene },
    { name: "model", chapter: primary("H · 三维预览  3D preview", "H · 3D preview"), tailSeconds: 2.2, draw: drawModelScene },
    { name: "studio", chapter: primary("I · 一个界面  One interface", "I · One interface"), tailSeconds: 1.4, draw: drawStudioScene },
    { name: "ending", chapter: "", tailSeconds: 6.0, draw: drawEndingScene }
];

function drawTitleScene(context, time, scene, material) {
    const apartment = material.apartments[0];
    const cellPixelSize = fitCellSize(apartment.plan, 780, 800);
    const reveal = ease(ramp(time, 0.3, 5));
    context.save();
    context.beginPath();
    context.rect(960, 0, 960, mix(60, 1080, reveal));
    context.clip();
    drawPlan(context, apartment.plan, 1400, 500, cellPixelSize, "drawing", 0.95);
    context.restore();
    const first = ease(ramp(time, 0.4, 1.2));
    drawText(context, "INDEPENDENT STUDY · 2022", 150, 330 - 14 * (1 - first), 26, COLOUR.accent, { alpha: first, family: "Consolas" });
    const second = ease(ramp(time, 1.0, 1.2));
    drawText(context, primary("波函数坍缩", "Wave Function"), 146, 450, 104, COLOUR.text, { alpha: second, family: "YaHeiBold" });
    drawText(context, primary("住宅平面生成", "Collapse"), 146, 580, 104, COLOUR.text, { alpha: ease(ramp(time, 1.5, 1.2)), family: "YaHeiBold" });
    const third = ease(ramp(time, 2.4, 1.2));
    drawText(context, primary("Floor Plan Generator Using", "A generator of apartment floor plans,"), 150, 668, 38, COLOUR.muted, { alpha: third });
    drawText(context, primary("the Wave Function Collapse Algorithm", "grown cell by cell on a 55 cm grid"), 150, 718, 38, COLOUR.muted, { alpha: third });
    drawText(context, "Qian Li", 150, 800, 30, COLOUR.text, { alpha: ease(ramp(time, 3.2, 1.2)) });
    const abbreviation = ease(ramp(time, cueTime(scene, 1, "WFC") - 0.3, 0.8));
    drawText(context, "WFC", 420, 806, 54, COLOUR.accent, { alpha: abbreviation, family: "Consolas" });
    drawText(context, "Wave Function Collapse", 540, 800, 26, COLOUR.muted, { alpha: abbreviation });
}

function drawBullet(context, y, chinese, english, alpha) {
    if (alpha <= 0) { return; }
    context.save();
    context.globalAlpha *= alpha;
    context.fillStyle = COLOUR.accent;
    context.fillRect(150, y - 44 + 20 * (1 - alpha), 6, 62);
    drawText(context, primary(chinese, english), 180, y - 8 + 20 * (1 - alpha), IS_ENGLISH ? 38 : 46, COLOUR.text, { family: "YaHeiBold" });
    drawText(context, secondary(english), 180, y + 30 + 20 * (1 - alpha), 26, COLOUR.muted);
    context.restore();
}

function drawAlgorithmScene(context, time, scene, material) {
    const growth = material.livingGrowth;
    const configuration = material.configurations.living;
    const cellPixelSize = 34;
    const originX = 1360 - cellPixelSize * configuration.gridColumnCount / 2, originY = 490 - cellPixelSize * configuration.gridRowCount / 2;
    const growEnd = scene.duration - 3.2;
    const stepCount = Math.floor(ramp(time, 1.2, growEnd - 1.2) * growth.outcome.trace.length);
    const state = replayAttempt(growth.outcome, stepCount, configuration, material.libraries.living);
    const toDrawing = ease(ramp(time, growEnd + 0.4, 1.4));
    context.save();
    context.globalAlpha = 1 - toDrawing;
    drawWorkingGrid(context, state.grid, configuration.palette, originX, originY, cellPixelSize, time, true);
    if (state.lastStep && state.lastStep.kind === "place" && toDrawing === 0) {
        context.strokeStyle = "#ffffff";
        context.lineWidth = 3;
        context.strokeRect(originX + state.lastStep.tileColumn * cellPixelSize, originY + state.lastStep.tileRow * cellPixelSize, 2 * cellPixelSize, 2 * cellPixelSize);
    }
    context.restore();
    if (toDrawing > 0) {
        // the finished plan sits where its cells were in the working grid
        let minimumRow = 99, minimumColumn = 99;
        for (let row = 0; row < growth.outcome.grid.rowCount; row++) {
            for (let column = 0; column < growth.outcome.grid.columnCount; column++) {
                if (growth.outcome.grid.cells[row * growth.outcome.grid.columnCount + column] !== engine.WALL_CODE) {
                    minimumRow = Math.min(minimumRow, row); minimumColumn = Math.min(minimumColumn, column);
                }
            }
        }
        const plan = growth.evaluation.plan;
        drawPlan(context, plan, originX + (minimumColumn + plan.columnCount / 2) * cellPixelSize, originY + (minimumRow + plan.rowCount / 2) * cellPixelSize, cellPixelSize, "drawing", toDrawing);
    }
    drawText(context, "Maxim Gumin · 2016", 150, 250, 26, COLOUR.accent, { alpha: ease(ramp(time, cueTime(scene, 0, "马克西姆", "Maxim"), 1)), family: "Consolas" });
    drawBullet(context, 360, "随机", "Randomness", ease(ramp(time, cueTime(scene, 0, "随机", "randomness"), 0.8)));
    drawBullet(context, 480, "相邻单元的局部约束", "Local constraints between neighbours", ease(ramp(time, cueTime(scene, 0, "相邻", "local constraints"), 0.8)));
    drawBullet(context, 600, "不依赖经验，产生全新的设计", "Independent of experience, new designs", ease(ramp(time, cueTime(scene, 1, "不依赖", "does not depend"), 0.8)));
    drawBullet(context, 720, "家具与人体工程 · 自下而上", "Furniture and ergonomics, from the bottom up", ease(ramp(time, cueTime(scene, 2, "家具", "furniture"), 0.8)));
}

function drawSubdividedSquare(context, x, y, size, divisionCount, alpha, lineColour) {
    if (alpha <= 0) { return; }
    context.save();
    context.globalAlpha *= alpha;
    context.fillStyle = "#212830";
    context.fillRect(x, y, size, size);
    for (let index = 0; index <= divisionCount; index++) {
        const offset = size * index / divisionCount;
        strokeLine(context, x + offset, y, x + offset, y + size, lineColour, 1);
        strokeLine(context, x, y + offset, x + size, y + offset, lineColour, 1);
    }
    context.strokeStyle = "#ffffff";
    context.lineWidth = 4;
    context.strokeRect(x, y, size, size);
    context.restore();
}

function drawDimension(context, x1, x2, y, label, alpha) {
    if (alpha <= 0) { return; }
    context.save();
    context.globalAlpha *= alpha;
    strokeLine(context, x1, y, x2, y, COLOUR.accent, 2);
    strokeLine(context, x1, y - 9, x1, y + 9, COLOUR.accent, 2);
    strokeLine(context, x2, y - 9, x2, y + 9, COLOUR.accent, 2);
    drawText(context, label, (x1 + x2) / 2, y + 34, 26, COLOUR.accent, { align: "center", family: "Consolas" });
    context.restore();
}

function drawGridScene(context, time, scene, material) {
    const moduleSize = 150;
    const lineOne = scene.lineStart[1], lineTwo = scene.lineStart[2], lineThree = scene.lineStart[3];
    // line 0: the same room at two grid sizes
    const compare = ease(ramp(time, 0.3, 1)) * (1 - ease(ramp(time, lineTwo - 0.6, 0.8)));
    const fineFade = 1 - 0.75 * ease(ramp(time, cueTime(scene, 1, "而是", "I chose") - 0.4, 1));
    drawSubdividedSquare(context, 170, 250, 495, 16, compare * fineFade, "rgba(255,255,255,0.22)");
    drawText(context, "20 cm", 417, 800, 40, COLOUR.text, { align: "center", alpha: compare * fineFade, family: "Consolas" });
    drawText(context, primary("272 格 cells · 细节 detail · 慢 slow", "272 cells · detail · slow"), 417, 846, 26, COLOUR.muted, { align: "center", alpha: compare * fineFade });
    drawSubdividedSquare(context, 780, 250, 495, 6, compare, "rgba(255,255,255,0.3)");
    drawText(context, "55 cm", 1027, 800, 40, COLOUR.accent, { align: "center", alpha: compare, family: "Consolas" });
    drawText(context, primary("36 格 cells · 快 fast", "36 cells · fast"), 1027, 846, 26, COLOUR.muted, { align: "center", alpha: compare });
    drawText(context, primary("同一个 3.3 m × 3.3 m 的房间", "The same 3.3 m × 3.3 m room"), 1360, 440, 34, COLOUR.text, { alpha: compare });
    drawText(context, secondary("The same 3.3 m × 3.3 m room"), 1360, 486, 26, COLOUR.muted, { alpha: compare });
    const highlight = compare * ease(ramp(time, cueTime(scene, 1, "而是", "I chose"), 0.6));
    if (highlight > 0) {
        context.save();
        context.globalAlpha = highlight;
        context.fillStyle = "rgba(245,158,11,0.35)";
        context.fillRect(780 + 82.5 * 2, 250 + 82.5 * 2, 82.5, 82.5);
        context.strokeStyle = COLOUR.accent;
        context.lineWidth = 4;
        context.strokeRect(780 + 82.5 * 2, 250 + 82.5 * 2, 82.5, 82.5);
        context.restore();
    }

    // line 2: what 55 cm corresponds to
    const items = [
        { delay: cueTime(scene, 2, "站立", "That is") - lineTwo, chinese: "站立，含两臂", english: "Standing, arms included", draw: drawPersonIcon },
        { delay: cueTime(scene, 2, "衣柜", "wardrobe") - lineTwo, chinese: "衣柜进深", english: "Wardrobe depth", draw: drawWardrobeIcon },
        { delay: cueTime(scene, 2, "最窄", "narrowest") - lineTwo, chinese: "最窄的过道", english: "Narrowest passage", draw: drawPassageIcon },
        { delay: cueTime(scene, 2, "橱柜", "kitchen counter") - lineTwo, chinese: "台面深度", english: "Counter depth", draw: drawCounterIcon }
    ];
    const shrink = ease(ramp(time, lineThree - 0.3, 0.9));
    items.forEach(function (item, itemIndex) {
        const alpha = ease(ramp(time, lineTwo + item.delay, 0.7));
        if (alpha <= 0) { return; }
        const centreX = 330 + itemIndex * 420, top = mix(330, 190, shrink);
        context.save();
        context.globalAlpha = alpha;
        context.fillStyle = "#212830";
        context.fillRect(centreX - moduleSize / 2, top, moduleSize, moduleSize);
        context.strokeStyle = "rgba(255,255,255,0.35)";
        context.lineWidth = 2;
        context.strokeRect(centreX - moduleSize / 2, top, moduleSize, moduleSize);
        item.draw(context, centreX, top, moduleSize);
        drawDimension(context, centreX - moduleSize / 2, centreX + moduleSize / 2, top + moduleSize + 30, "55 cm", 1);
        drawText(context, primary(item.chinese, item.english), centreX, top + moduleSize + 124, IS_ENGLISH ? 28 : 36, COLOUR.text, { align: "center", family: "YaHeiBold" });
        drawText(context, secondary(item.english), centreX, top + moduleSize + 162, 24, COLOUR.muted, { align: "center" });
        context.restore();
    });

    // line 3: walls have no thickness, doors are two modules
    const wallCue = cueTime(scene, 3, "墙的厚度", "wall thickness"), doorCue = cueTime(scene, 3, "而门", "a door");
    const wallAlpha = ease(ramp(time, wallCue - 0.2, 0.8));
    if (wallAlpha > 0) {
        context.save();
        context.globalAlpha = wallAlpha;
        const thickness = mix(46, 6, ease(ramp(time, wallCue + 1.0, 1.8)));
        context.fillStyle = "#ffffff";
        context.fillRect(260, 730 - thickness / 2, 440, thickness);
        drawText(context, "10–24 cm ÷ 55 ≈ 0.3", 480, 660, 26, COLOUR.accent, { align: "center", family: "Consolas" });
        drawText(context, primary("墙厚 → 0 格", "Wall → 0 cells"), 480, 820, 36, COLOUR.text, { align: "center", family: "YaHeiBold" });
        drawText(context, bilingual("四舍五入", "Wall thickness rounds to zero cells"), 480, 856, 24, COLOUR.muted, { align: "center" });
        context.restore();
    }
    const doorAlpha = ease(ramp(time, doorCue - 0.2, 0.8));
    if (doorAlpha > 0) {
        context.save();
        context.globalAlpha = doorAlpha;
        const left = 1130, top = 600, cell = 90;
        context.fillStyle = "#212830";
        context.fillRect(left, top, cell * 4, cell * 2);
        for (let index = 0; index <= 4; index++) { strokeLine(context, left + index * cell, top, left + index * cell, top + 2 * cell, "rgba(255,255,255,0.25)", 1); }
        for (let index = 0; index <= 2; index++) { strokeLine(context, left, top + index * cell, left + 4 * cell, top + index * cell, "rgba(255,255,255,0.25)", 1); }
        // a wall stub and the frame stand inside the opening on both sides; the leaf is what is left
        const stub = 0.182 * cell, jamb = 0.282 * cell, leafLength = 1.482 * cell;
        strokeLine(context, left + cell + stub, top, left + cell + jamb, top, "#8d99a8", 8);
        strokeLine(context, left + cell + jamb + leafLength, top, left + 3 * cell - stub, top, "#8d99a8", 8);
        strokeLine(context, left, top, left + cell + stub, top, "#ffffff", 8);
        strokeLine(context, left + 3 * cell - stub, top, left + 4 * cell, top, "#ffffff", 8);
        const swing = ease(ramp(time, doorCue + 0.3, 1.4));
        context.strokeStyle = COLOUR.text;
        context.lineWidth = 3;
        context.beginPath();
        context.arc(left + cell + jamb, top, leafLength, 0, Math.PI / 2 * swing);
        context.stroke();
        const leafAngle = Math.PI / 2 * swing;
        strokeLine(context, left + cell + jamb, top, left + cell + jamb + Math.cos(leafAngle) * leafLength, top + Math.sin(leafAngle) * leafLength, COLOUR.text, 4);
        drawDimension(context, left + cell, left + 3 * cell, top - 30, "", 1);
        drawText(context, "90–100 cm ÷ 55 ≈ 1.7", left + 2 * cell, top - 48, 26, COLOUR.accent, { align: "center", family: "Consolas" });
        drawText(context, primary("门 → 2 格", "Door → 2 cells"), left + 2 * cell, 820, 36, COLOUR.text, { align: "center", family: "YaHeiBold" });
        drawText(context, bilingual("含墙垛与门框，门扇约 82 cm", "Stubs and frame included; the leaf is about 82 cm"), left + 2 * cell, 856, 22, COLOUR.muted, { align: "center" });
        context.restore();
    }
}

function drawPersonIcon(context, centreX, top, size) {
    const centreY = top + size / 2;
    context.strokeStyle = COLOUR.text;
    context.lineWidth = 3;
    context.beginPath(); context.ellipse(centreX, centreY, size * 0.42, size * 0.17, 0, 0, 2 * Math.PI); context.stroke();
    context.fillStyle = "#212830";
    context.beginPath(); context.arc(centreX, centreY, size * 0.15, 0, 2 * Math.PI); context.fill(); context.stroke();
    context.beginPath(); context.arc(centreX - size * 0.42, centreY + size * 0.12, size * 0.07, 0, 2 * Math.PI); context.stroke();
    context.beginPath(); context.arc(centreX + size * 0.42, centreY + size * 0.12, size * 0.07, 0, 2 * Math.PI); context.stroke();
}
function drawWardrobeIcon(context, centreX, top, size) {
    const left = centreX - size / 2;
    context.strokeStyle = COLOUR.text;
    context.lineWidth = 3;
    context.strokeRect(left + size * 0.07, top + size * 0.07, size * 0.86, size * 0.86);
    strokeLine(context, left + size * 0.07, top + size / 2, left + size * 0.93, top + size / 2, COLOUR.text, 3);
    for (const offset of [0.25, 0.5, 0.75]) { strokeLine(context, left + size * (offset - 0.04), top + size * 0.2, left + size * (offset + 0.04), top + size * 0.8, COLOUR.text, 3); }
}
function drawPassageIcon(context, centreX, top, size) {
    const left = centreX - size / 2;
    strokeLine(context, left, top, left, top + size, "#ffffff", 8);
    strokeLine(context, left + size, top, left + size, top + size, "#ffffff", 8);
    context.strokeStyle = COLOUR.text;
    context.lineWidth = 3;
    context.beginPath(); context.ellipse(centreX, top + size / 2, size * 0.34, size * 0.14, 0, 0, 2 * Math.PI); context.stroke();
    context.beginPath(); context.arc(centreX, top + size / 2, size * 0.12, 0, 2 * Math.PI); context.stroke();
    strokeLine(context, centreX, top + size * 0.12, centreX, top + size * 0.26, COLOUR.accent, 3);
    strokeLine(context, centreX - 8, top + size * 0.2, centreX, top + size * 0.12, COLOUR.accent, 3);
    strokeLine(context, centreX + 8, top + size * 0.2, centreX, top + size * 0.12, COLOUR.accent, 3);
}
function drawCounterIcon(context, centreX, top, size) {
    const left = centreX - size / 2;
    context.strokeStyle = COLOUR.text;
    context.lineWidth = 3;
    context.strokeRect(left + size * 0.05, top + size * 0.05, size * 0.9, size * 0.9);
    roundedRectangle(context, left + size * 0.2, top + size * 0.22, size * 0.6, size * 0.56, 12);
    context.stroke();
    context.beginPath(); context.arc(centreX, top + size * 0.5, size * 0.05, 0, 2 * Math.PI); context.stroke();
}

function drawCodeGridWithLabels(context, grid, palette, originX, originY, cellPixelSize, labelAlpha) {
    engine.renderCodeGrid(context, grid, palette, originX, originY, cellPixelSize, true);
    if (labelAlpha <= 0) { return; }
    for (let row = 0; row < grid.rowCount; row++) {
        for (let column = 0; column < grid.columnCount; column++) {
            const code = grid.cells[row * grid.columnCount + column];
            drawText(context, String(code).padStart(4, "0"), originX + (column + 0.5) * cellPixelSize, originY + (row + 0.5) * cellPixelSize + 5, 15,
                code === engine.WALL_CODE ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.72)", { align: "center", family: "Consolas", alpha: labelAlpha });
        }
    }
}

function drawPixelScene(context, time, scene, material) {
    const lineOne = scene.lineStart[1], lineTwo = scene.lineStart[2];
    const archive = ease(ramp(time, 0.2, 0.8)) * (1 - ease(ramp(time, lineOne - 0.5, 0.8)));
    drawImageCard(context, material.images.plan, 560, 480, 620, 640, archive, primary("2022 · 一个随机的卧室平面  A random bedroom plan", "2022 · A random bedroom plan"), false);
    const pixelated = archive * ease(ramp(time, 2.2, 1));
    drawImageCard(context, material.images.pixelated, 1340, 480, 560, 600, pixelated, bilingual("它的像素化", "Its pixelation"), true);
    if (pixelated > 0) {
        context.save();
        context.globalAlpha = pixelated;
        strokeLine(context, 900, 480, 1010, 480, COLOUR.accent, 4);
        strokeLine(context, 990, 462, 1012, 480, COLOUR.accent, 4);
        strokeLine(context, 990, 498, 1012, 480, COLOUR.accent, 4);
        context.restore();
    }

    const sampleGrid = material.sampleGrids.bedroom[0];
    const palette = material.configurations.bedroom.palette;
    const cellPixelSize = 62;
    const scan = ease(ramp(time, lineTwo - 0.4, 0.8));
    const show = ease(ramp(time, lineOne - 0.1, 0.9));
    if (show <= 0) { return; }
    context.save();
    context.globalAlpha = show;
    const gridLeft = mix(760, 330, scan), gridTop = 490 - cellPixelSize * sampleGrid.rowCount / 2;
    const samplePlan = material.samplePlan;
    drawPlan(context, samplePlan, 420, 490, cellPixelSize, "drawing", show * (1 - scan));
    drawCodeGridWithLabels(context, sampleGrid, palette, gridLeft, gridTop, cellPixelSize, ease(ramp(time, cueTime(scene, 1, "编码", "a code"), 0.8)));
    const legend = [["bed", "床", "Bed"], ["wardrobe", "衣柜", "Wardrobe"], ["passage", "过道", "Passage"], ["door", "门", "Door"], ["window", "窗", "Window"]].map(function (entry) {
        return entry.concat([cueTime(scene, 1, entry[1], entry[2]) - lineOne]);
    });
    legend.forEach(function (entry, entryIndex) {
        const alpha = ease(ramp(time, lineOne + entry[3], 0.5)) * (1 - scan);
        if (alpha <= 0) { return; }
        const codes = engine.elementCodes(entry[0]);
        const y = 270 + entryIndex * 92;
        context.save();
        context.globalAlpha = alpha * show;
        codes.slice(0, 4).forEach(function (code, codeIndex) {
            context.fillStyle = engine.codeColour(palette, codes[Math.floor(codeIndex * (codes.length - 1) / Math.max(1, Math.min(3, codes.length - 1)))]);
            context.fillRect(1420 + codeIndex * 30, y - 34, 26, 44);
        });
        drawText(context, primary(entry[1], entry[2]), 1570, y, IS_ENGLISH ? 36 : 40, COLOUR.text, { family: "YaHeiBold" });
        drawText(context, secondary(entry[2]), 1690, y, 26, COLOUR.muted);
        context.restore();
    });
    if (scan > 0) {
        // a 2 x 2 window sweeps over the sample; each stop is one tile
        const positionCount = (sampleGrid.rowCount - 1) * (sampleGrid.columnCount - 1);
        const interesting = [];
        for (let position = 0; position < positionCount; position++) {
            const row = Math.floor(position / (sampleGrid.columnCount - 1)), column = position % (sampleGrid.columnCount - 1);
            const codes = [0, 1, sampleGrid.columnCount, sampleGrid.columnCount + 1].map(function (offset) { return sampleGrid.cells[row * sampleGrid.columnCount + column + offset]; });
            if (codes.some(function (code) { return code !== engine.WALL_CODE; })) { interesting.push({ row: row, column: column, codes: codes }); }
        }
        const stop = interesting[Math.min(interesting.length - 1, Math.floor(ramp(time, lineTwo + 0.3, scene.duration - lineTwo - 1.2) * interesting.length))];
        context.globalAlpha = scan;
        context.strokeStyle = "#ffffff";
        context.lineWidth = 5;
        context.strokeRect(gridLeft + stop.column * cellPixelSize, gridTop + stop.row * cellPixelSize, 2 * cellPixelSize, 2 * cellPixelSize);
        const tileLeft = 1180, tileTop = 310, tileCell = 170;
        stop.codes.forEach(function (code, cornerIndex) {
            context.fillStyle = engine.codeColour(palette, code);
            context.fillRect(tileLeft + (cornerIndex % 2) * tileCell, tileTop + Math.floor(cornerIndex / 2) * tileCell, tileCell - 3, tileCell - 3);
            drawText(context, String(code).padStart(4, "0"), tileLeft + (cornerIndex % 2 + 0.5) * tileCell, tileTop + (Math.floor(cornerIndex / 2) + 0.5) * tileCell + 10, 30,
                code === engine.WALL_CODE ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.75)", { align: "center", family: "Consolas" });
        });
        strokeLine(context, gridLeft + (stop.column + 2) * cellPixelSize, gridTop + (stop.row + 1) * cellPixelSize, tileLeft - 14, tileTop + tileCell, "rgba(255,255,255,0.35)", 2);
        drawText(context, primary("一块拼图", "One tile"), 1570, 440, 44, COLOUR.text, { family: "YaHeiBold" });
        drawText(context, secondary("One tile"), 1570, 484, 28, COLOUR.muted);
        drawText(context, primary("四个角，四种颜色", "Four corners,"), 1570, IS_ENGLISH ? 510 : 560, 30, COLOUR.text);
        drawText(context, primary("Four corners, four colours", "four colours"), 1570, IS_ENGLISH ? 552 : 598, IS_ENGLISH ? 30 : 24, IS_ENGLISH ? COLOUR.text : COLOUR.muted);
    }
    context.restore();
}

function drawTile(context, tile, palette, x, y, size) {
    const half = size / 2;
    for (let cornerIndex = 0; cornerIndex < 4; cornerIndex++) {
        context.fillStyle = engine.codeColour(palette, tile.codes[cornerIndex]);
        context.fillRect(x + (cornerIndex % 2) * half, y + Math.floor(cornerIndex / 2) * half, half, half);
    }
}

function drawLibraryScene(context, time, scene, material) {
    const lineOne = scene.lineStart[1], lineTwo = scene.lineStart[2];
    const archive = ease(ramp(time, 0.2, 0.8)) * (1 - ease(ramp(time, lineOne - 0.5, 0.8)));
    drawImageCard(context, material.images.manualLibrary, 960, 470, 1150, 640, archive, primary("2022 · 手工定义的卧室连接  Bedroom connections, defined by hand", "2022 · Bedroom connections, defined by hand"), false);

    const show = ease(ramp(time, lineOne - 0.1, 0.9));
    if (show <= 0) { return; }
    const palette = material.configurations.bedroom.palette;
    const samples = material.sampleGrids.bedroom;
    const tiles = material.sortedBedroomTiles;
    context.save();
    context.globalAlpha = show;
    const tileProgress = ramp(time, lineOne + 1.4, scene.lineEnd[2] - lineOne - 2.6);
    const shownTileCount = Math.floor(ease(tileProgress) * tiles.length);
    samples.forEach(function (sampleGrid, sampleIndex) {
        const thumbnail = 138, columnIndex = sampleIndex % 5, rowIndex = Math.floor(sampleIndex / 5);
        const x = 130 + columnIndex * 152, y = 220 + rowIndex * 152;
        const appear = ease(ramp(time, lineOne + sampleIndex * 0.05, 0.5));
        context.globalAlpha = show * appear;
        context.fillStyle = "#0c1118";
        context.fillRect(x, y, thumbnail, thumbnail);
        const cellPixelSize = Math.min(thumbnail / sampleGrid.columnCount, thumbnail / sampleGrid.rowCount);
        engine.renderCodeGrid(context, sampleGrid, palette, x + (thumbnail - cellPixelSize * sampleGrid.columnCount) / 2, y + (thumbnail - cellPixelSize * sampleGrid.rowCount) / 2, cellPixelSize, false);
        // the sample being read right now
        if (Math.floor(tileProgress * samples.length) === sampleIndex && tileProgress > 0 && tileProgress < 1) {
            context.strokeStyle = COLOUR.accent;
            context.lineWidth = 4;
            context.strokeRect(x - 2, y - 2, thumbnail + 4, thumbnail + 4);
        }
    });
    context.globalAlpha = show;
    drawText(context, "19", 130, 870, 64, COLOUR.accent, { family: "Consolas" });
    drawText(context, bilingual("卧室样本", "bedroom samples"), 220, 866, 30, COLOUR.text);
    const tileSize = 42, tileGap = 10, tilesPerRow = 16;
    for (let tileIndex = 0; tileIndex < shownTileCount; tileIndex++) {
        const x = 960 + (tileIndex % tilesPerRow) * (tileSize + tileGap), y = 196 + Math.floor(tileIndex / tilesPerRow) * (tileSize + tileGap);
        drawTile(context, tiles[tileIndex], palette, x, y, tileSize);
    }
    const counted = ease(ramp(time, lineTwo, 0.6));
    drawText(context, String(shownTileCount), 960, 870, 64, COLOUR.accent, { family: "Consolas" });
    drawText(context, bilingual("种拼接", "tiles"), 1090, 866, 30, COLOUR.text);
    const limit = ease(ramp(time, cueTime(scene, 1, "不过", "However"), 0.8)) * (1 - ease(ramp(time, lineTwo - 0.4, 0.6)));
    if (limit > 0) {
        context.globalAlpha = show * limit;
        context.fillStyle = "rgba(22,27,34,0.88)";
        context.fillRect(940, 400, 900, 190);
        context.fillStyle = COLOUR.accent;
        context.fillRect(940, 400, 6, 190);
        drawText(context, primary("局限", "Limitation"), 980, 462, 44, COLOUR.accent, { family: "YaHeiBold" });
        drawText(context, primary("样本里没有出现过的连接，永远不会被生成", "A connection that never occurs in the samples"), 980, 516, 32, COLOUR.text);
        drawText(context, primary("A connection that never occurs in the samples can never be generated", "can never be generated"), 980, 556, IS_ENGLISH ? 32 : 24, IS_ENGLISH ? COLOUR.text : COLOUR.muted);
        context.globalAlpha = show;
    }
    if (counted > 0 && tiles.length > 0) {
        context.globalAlpha = show * counted;
        drawText(context, primary("最常见 × " + tiles[0].frequency + " · 最少见 × 1", "most common × " + tiles[0].frequency + " · rarest × 1"), 1800, 846, 22, COLOUR.muted, { align: "right" });
        drawText(context, bilingual("频率决定被选中的机会", "frequency sets the odds"), 1800, 876, 20, COLOUR.muted, { align: "right" });
    }
    context.restore();
}

function drawCollapseScene(context, time, scene, material) {
    const story = material.collapseStory;
    const configuration = material.configurations.bedroom, library = material.libraries.bedroom;
    const cellPixelSize = 70;
    const originX = 220, originY = 490 - cellPixelSize * configuration.gridRowCount / 2;
    // narration lines: 0 the door, 1 what entropy is, 2 one step, 3 dead end, 4 most attempts fail
    const entropyStart = scene.lineStart[1], stepStart = scene.lineStart[2], deadEndStart = scene.lineStart[3], failStart = scene.lineStart[4];
    const attempts = story.attempts;
    const firstAttempt = attempts[0];
    const doorStepCount = firstAttempt.trace.filter(function (step) { return step.phase === "door"; }).length;
    const montageStart = failStart + 0.2, montageEnd = failStart + 3.2;
    const finalAttempt = attempts[attempts.length - 1];
    const finalStart = montageEnd, finalEnd = scene.lineEnd[4] + 0.9;

    let attemptNumber = 1, state, failed = false, phaseLabel = "", candidateCount = null, showEntropies = false;
    if (time < deadEndStart + 1.5) {
        let stepCount;
        if (time < stepStart) { stepCount = Math.floor(ramp(time, cueTime(scene, 0, "一扇门", "a door"), entropyStart - cueTime(scene, 0, "一扇门", "a door") - 1.2) * doorStepCount); }
        else { stepCount = doorStepCount + Math.floor(ramp(time, stepStart + 0.4, deadEndStart - stepStart - 0.6) * (firstAttempt.trace.length - doorStepCount)); }
        state = replayAttempt(firstAttempt, stepCount, configuration, library);
        failed = stepCount >= firstAttempt.trace.length;
        phaseLabel = time < entropyStart ? bilingual("门", "door") : (time < stepStart ? bilingual("叠加态", "superposition") : bilingual("坍缩", "collapse"));
        if (time >= stepStart && state.lastStep && state.lastStep.kind === "place") { candidateCount = state.lastStep.candidateCount; }
        showEntropies = time >= cueTime(scene, 1, "计算", "compute") - 0.3 && !failed;
    } else if (time < montageStart) {
        attemptNumber = 2;
        const stepCount = Math.floor(ramp(time, deadEndStart + 1.5, montageStart - deadEndStart - 1.9) * attempts[1].trace.length);
        state = replayAttempt(attempts[1], stepCount, configuration, library);
        failed = stepCount >= attempts[1].trace.length;
        phaseLabel = bilingual("坍缩", "collapse");
    } else if (time < montageEnd) {
        attemptNumber = Math.min(attempts.length - 1, 3 + Math.floor(Math.pow(ramp(time, montageStart, montageEnd - montageStart), 1.6) * (attempts.length - 3)));
        const outcome = attempts[attemptNumber - 1];
        state = replayAttempt(outcome, outcome.trace.length, configuration, library);
        failed = !outcome.success;
        phaseLabel = "…";
    } else {
        attemptNumber = attempts.length;
        const stepCount = Math.floor(ramp(time, finalStart, finalEnd - finalStart) * finalAttempt.trace.length);
        state = replayAttempt(finalAttempt, stepCount, configuration, library);
        phaseLabel = stepCount >= finalAttempt.trace.length ? bilingual("闭合", "closed") : bilingual("坍缩", "collapse");
        if (state.lastStep && state.lastStep.kind === "place") { candidateCount = state.lastStep.candidateCount; }
    }
    const toDrawing = ease(ramp(time, finalEnd + 0.5, 1.3));
    context.save();
    context.globalAlpha = 1 - toDrawing;
    drawWorkingGrid(context, state.grid, configuration.palette, originX, originY, cellPixelSize, time, true);
    if (state.lastStep && toDrawing === 0 && time >= stepStart && (state.lastStep.kind === "place" || state.lastStep.kind === "failure")) {
        context.strokeStyle = state.lastStep.kind === "failure" ? COLOUR.bad : "#ffffff";
        context.lineWidth = 5;
        context.strokeRect(originX + state.lastStep.tileColumn * cellPixelSize, originY + state.lastStep.tileRow * cellPixelSize, 2 * cellPixelSize, 2 * cellPixelSize);
    }
    if (failed && state.lastStep && state.lastStep.kind === "failure") {
        const centreX = originX + (state.lastStep.tileColumn + 1) * cellPixelSize, centreY = originY + (state.lastStep.tileRow + 1) * cellPixelSize;
        strokeLine(context, centreX - 50, centreY - 50, centreX + 50, centreY + 50, COLOUR.bad, 12);
        strokeLine(context, centreX + 50, centreY - 50, centreX - 50, centreY + 50, COLOUR.bad, 12);
    }
    context.restore();

    // entropy of every open position, written where its four cells meet
    let positions = [];
    if (showEntropies) {
        positions = openPositions(state.grid, library);
        const labelAlpha = ease(ramp(time, cueTime(scene, 1, "计算", "compute") - 0.3, 0.8));
        let lowest = null, highest = null;
        for (const position of positions) {
            if (!lowest || position.entropy < lowest.entropy) { lowest = position; }
            if (!highest || position.entropy > highest.entropy) { highest = position; }
        }
        for (const position of positions) {
            const x = originX + (position.tileColumn + 1) * cellPixelSize, y = originY + (position.tileRow + 1) * cellPixelSize;
            const isLowest = position === lowest;
            context.globalAlpha = labelAlpha;
            context.fillStyle = isLowest ? COLOUR.accent : "rgba(13,17,23,0.82)";
            roundedRectangle(context, x - 25, y - 15, 50, 30, 8);
            context.fill();
            context.globalAlpha = 1;
            drawText(context, position.entropy.toFixed(1), x, y + 7, 20, isLowest ? "#1a1203" : COLOUR.text, { align: "center", family: "Consolas", alpha: labelAlpha });
        }
        // two positions side by side: few likely tiles against many
        const compareAlpha = ease(ramp(time, cueTime(scene, 1, "可能性", "more concentrated") - 0.4, 0.8)) * (1 - ease(ramp(time, stepStart - 0.3, 0.5)));
        if (compareAlpha > 0 && lowest && highest) {
            [[lowest, COLOUR.accent, primary("可能性集中 → 熵低", "Concentrated → low entropy"), "Concentrated possibilities, low entropy", 596], [highest, COLOUR.blue, primary("可能性分散 → 熵高", "Spread out → high entropy"), "Spread-out possibilities, high entropy", 732]].forEach(function (entry) {
                const position = entry[0];
                context.globalAlpha = compareAlpha;
                context.strokeStyle = entry[1];
                context.lineWidth = 4;
                context.strokeRect(originX + position.tileColumn * cellPixelSize + 3, originY + position.tileRow * cellPixelSize + 3, 2 * cellPixelSize - 6, 2 * cellPixelSize - 6);
                const frequencySum = position.frequencies.reduce(function (sum, value) { return sum + value; }, 0);
                position.frequencies.slice(0, 26).forEach(function (frequency, barIndex) {
                    const barHeight = Math.max(3, 60 * frequency / frequencySum / Math.max(0.25, position.frequencies[0] / frequencySum));
                    context.fillStyle = entry[1];
                    context.fillRect(1080 + barIndex * 16, entry[4] + 60 - barHeight, 11, barHeight);
                });
                context.globalAlpha = 1;
                drawText(context, entry[2] + "   H = " + position.entropy.toFixed(1), 1080, entry[4] + 98, 30, entry[1], { alpha: compareAlpha, family: "YaHeiBold" });
                drawText(context, (IS_ENGLISH ? "" : entry[3] + " · ") + position.frequencies.length + " candidate tiles", 1080, entry[4] + 128, 22, COLOUR.muted, { alpha: compareAlpha });
            });
        }
    }

    if (toDrawing > 0) {
        let minimumRow = 99, minimumColumn = 99;
        for (let row = 0; row < finalAttempt.grid.rowCount; row++) {
            for (let column = 0; column < finalAttempt.grid.columnCount; column++) {
                if (finalAttempt.grid.cells[row * finalAttempt.grid.columnCount + column] !== engine.WALL_CODE) {
                    minimumRow = Math.min(minimumRow, row); minimumColumn = Math.min(minimumColumn, column);
                }
            }
        }
        const plan = finalAttempt.evaluation.plan;
        drawPlan(context, plan, originX + (minimumColumn + plan.columnCount / 2) * cellPixelSize, originY + (minimumRow + plan.rowCount / 2) * cellPixelSize, cellPixelSize, "drawing", toDrawing);
    }

    // side panel
    const panelX = 1080;
    drawText(context, bilingual("尝试", "ATTEMPT"), panelX, 220, 24, COLOUR.muted);
    drawText(context, "#" + attemptNumber, panelX, 306, 90, failed ? COLOUR.bad : COLOUR.text, { family: "Consolas" });
    drawText(context, bilingual("阶段", "PHASE"), panelX, 380, 24, COLOUR.muted);
    drawText(context, toDrawing > 0.5 ? bilingual("一个房间", "a room") : phaseLabel, panelX, 432, 38, COLOUR.text, { family: "YaHeiBold" });
    const formulaAlpha = ease(ramp(time, cueTime(scene, 1, "熵", "entropy") - 0.2, 0.8)) * (1 - ease(ramp(time, deadEndStart - 0.3, 0.5)));
    drawText(context, primary("熵", "Entropy") + "  H = − Σ p · ln p", panelX, 506, 36, COLOUR.text, { alpha: formulaAlpha });
    drawText(context, primary("p：每块候选拼图在样本中的出现频率", "p: how often each candidate tile occurs in the samples"), panelX, 540, 20, COLOUR.muted, { alpha: formulaAlpha });
    drawText(context, secondary("p: how often each candidate tile occurs in the samples"), panelX, 566, 20, COLOUR.muted, { alpha: formulaAlpha });
    if (candidateCount !== null && toDrawing === 0 && time >= stepStart) {
        drawText(context, bilingual("这一格的候选拼图", "CANDIDATE TILES HERE"), panelX, 620, 24, COLOUR.muted);
        drawText(context, String(candidateCount), panelX, 692, 64, COLOUR.accent, { family: "Consolas" });
        for (let index = 0; index < Math.min(candidateCount, 24); index++) {
            context.fillStyle = index === 0 ? COLOUR.accent : COLOUR.line;
            context.fillRect(panelX + 110 + index * 22, 656, 16, 34);
        }
    }
    const tieAlpha = ease(ramp(time, cueTime(scene, 2, "如果", "If several") - 0.2, 0.7)) * (1 - ease(ramp(time, deadEndStart - 0.3, 0.5)));
    drawText(context, primary("熵相等 → 随机选一个", "Equal entropy → pick one at random"), panelX, 780, 36, COLOUR.accent, { alpha: tieAlpha, family: "YaHeiBold" });
    drawText(context, secondary("Equal entropy: one of them is picked at random"), panelX, 816, 24, COLOUR.muted, { alpha: tieAlpha });
    if (failed) {
        drawText(context, primary("矛盾：没有合法的拼图", "Contradiction: no tile fits"), panelX, 780, 40, COLOUR.bad, { family: "YaHeiBold" });
        drawText(context, primary("Contradiction: no learned tile fits", "None of the learned tiles matches its neighbours"), panelX, 818, 26, COLOUR.bad);
    }
    if (toDrawing > 0.3) {
        const evaluation = finalAttempt.evaluation;
        drawText(context, primary("第 " + attempts.length + " 次尝试", "Attempt " + attempts.length) + " · " + evaluation.quantities.areaSquareMetres.toFixed(1) + " m²", panelX, 780, 40, COLOUR.good, { alpha: toDrawing, family: "YaHeiBold" });
        drawText(context, primary("Attempt " + attempts.length + " closes into a valid bedroom", "It closes into a valid bedroom"), panelX, 818, 26, COLOUR.good, { alpha: toDrawing });
    }
}

function drawCurvePanel(context, x, y, width, height, title, subtitle, curves, maximumX, progress, marker) {
    context.save();
    context.fillStyle = COLOUR.panel;
    roundedRectangle(context, x, y, width, height, 12);
    context.fill();
    drawText(context, title, x + 22, y + 46, 30, COLOUR.text, { family: "YaHeiBold" });
    drawText(context, subtitle, x + 22, y + 78, 20, COLOUR.muted);
    const plotLeft = x + 30, plotRight = x + width - 26, plotTop = y + 108, plotBottom = y + height - 44;
    let maximumScore = 0.001;
    for (const curve of curves) {
        for (let sampleIndex = 0; sampleIndex <= 60; sampleIndex++) { maximumScore = Math.max(maximumScore, curve.evaluate(maximumX * sampleIndex / 60)); }
    }
    strokeLine(context, plotLeft, plotBottom, plotRight, plotBottom, COLOUR.line, 2);
    strokeLine(context, plotLeft, plotTop, plotLeft, plotBottom, COLOUR.line, 2);
    for (const curve of curves) {
        context.strokeStyle = curve.colour;
        context.lineWidth = 4;
        context.beginPath();
        const lastSample = Math.floor(60 * progress);
        for (let sampleIndex = 0; sampleIndex <= lastSample; sampleIndex++) {
            const quantityValue = maximumX * sampleIndex / 60;
            const score = Math.max(0, curve.evaluate(quantityValue));
            const px = mix(plotLeft, plotRight, sampleIndex / 60), py = mix(plotBottom, plotTop, score / maximumScore);
            if (sampleIndex === 0) { context.moveTo(px, py); } else { context.lineTo(px, py); }
        }
        context.stroke();
        if (curve.label && progress > 0.95) { drawText(context, curve.label, curve.labelX * (plotRight - plotLeft) + plotLeft, curve.labelY * (plotBottom - plotTop) + plotTop, 20, curve.colour); }
    }
    if (marker && progress >= 1) {
        const px = mix(plotLeft, plotRight, marker.quantityValue / maximumX), py = mix(plotBottom, plotTop, Math.max(0, curves[0].evaluate(marker.quantityValue)) / maximumScore);
        context.fillStyle = "#ffffff";
        context.beginPath(); context.arc(px, py, 8, 0, 2 * Math.PI); context.fill();
        strokeLine(context, px, py, px, plotBottom, "rgba(255,255,255,0.4)", 2);
    }
    drawText(context, "0", plotLeft, plotBottom + 26, 18, COLOUR.muted, { family: "Consolas" });
    drawText(context, marker ? marker.axisLabel : "", plotRight, plotBottom + 26, 18, COLOUR.muted, { align: "right" });
    context.restore();
}

function drawFitnessScene(context, time, scene, material) {
    const evaluation = material.collapseStory.attempts[material.collapseStory.attempts.length - 1].evaluation;
    const plan = evaluation.plan;
    const terms = material.configurations.bedroom.fitnessTerms;
    const cellPixelSize = fitCellSize(plan, 520, 520);
    drawPlan(context, plan, 400, 470, cellPixelSize, "drawing", ease(ramp(time, 0, 0.6)));
    const quantities = evaluation.quantities;
    drawText(context, quantities.areaSquareMetres.toFixed(1) + " m²  ·  " + primary("周长 perimeter ", "perimeter ") + (quantities.perimeterEdgeCount * engine.CELL_SIZE_METRES).toFixed(1) + " m  ·  " + primary("衣柜 wardrobe ", "wardrobe ") + quantities["count:wardrobe"] + primary(" 格", " cells"),
        400, 800, 24, COLOUR.muted, { align: "center", alpha: ease(ramp(time, 0.6, 0.8)) });

    const lineStarts = scene.lineStart;
    const panelWidth = 330, panelHeight = 330, panelTop = 190;
    const luxuryTerm = { curve: "peak", parameters: { targetValue: 14, penaltyCoefficient: 0.02, peakScore: 10 } };
    const panels = [
        { start: lineStarts[1], title: primary("① 紧凑度", "① Compactness"), subtitle: primary("Compactness: area / perimeter²", "area / perimeter²"), maximumX: 0.0625, termIndex: 2, axisLabel: primary("正方形 square", "square"),
            curves: [{ colour: COLOUR.accent, evaluate: function (x) { return engine.evaluateCurve(terms[2], x); } }] },
        { start: lineStarts[2], title: primary("② 面积", "② Area"), subtitle: primary("Area: rises, then falls", "rises, then falls"), maximumX: 25, termIndex: 0, axisLabel: "25 m²",
            curves: [
                { colour: COLOUR.accent, evaluate: function (x) { return engine.evaluateCurve(terms[0], x); } },
                { colour: COLOUR.blue, evaluate: function (x) { return engine.evaluateCurve(luxuryTerm, x); }, delay: cueTime(scene, 2, "高档", "Upscale") - lineStarts[2] }
            ] },
        { start: lineStarts[3], title: primary("③ 储物", "③ Storage"), subtitle: primary("Storage: diminishing returns", "diminishing returns"), maximumX: 12, termIndex: 1, axisLabel: primary("12 格 cells", "12 cells"),
            curves: [{ colour: COLOUR.accent, evaluate: function (x) { return engine.evaluateCurve(terms[1], x); } }] }
    ];
    panels.forEach(function (panel, panelIndex) {
        const alpha = ease(ramp(time, panel.start - 0.2, 0.6));
        if (alpha <= 0) { return; }
        context.save();
        context.globalAlpha = alpha;
        const termScore = evaluation.termScores[panel.termIndex];
        const curves = panel.curves.filter(function (curve) { return !curve.delay || time > panel.start + curve.delay; });
        const x = 830 + panelIndex * (panelWidth + 24);
        // each curve draws itself in; the first one finishes before a delayed one starts
        const progress = ramp(time, panel.start + 0.2, 1.6);
        drawCurvePanel(context, x, panelTop, panelWidth, panelHeight, panel.title, panel.subtitle, curves.slice(0, 1), panel.maximumX, progress,
            { quantityValue: termScore.quantityValue, axisLabel: panel.axisLabel });
        if (panelIndex === 1) { drawText(context, primary("紧凑 compact", "compact"), x + panelWidth - 22, panelTop + 32, 19, COLOUR.accent, { align: "right" }); }
        if (curves.length > 1) {
            context.save();
            context.beginPath();
            context.rect(x + 28, panelTop + 100, (panelWidth - 50) * ramp(time, panel.start + curves[1].delay, 1.6), panelHeight - 140);
            context.clip();
            const plotLeft = x + 30, plotRight = x + panelWidth - 26, plotTop = panelTop + 108, plotBottom = panelTop + panelHeight - 44;
            context.strokeStyle = curves[1].colour;
            context.lineWidth = 4;
            context.beginPath();
            for (let sampleIndex = 0; sampleIndex <= 60; sampleIndex++) {
                const score = Math.max(0, curves[1].evaluate(panel.maximumX * sampleIndex / 60));
                const px = mix(plotLeft, plotRight, sampleIndex / 60), py = mix(plotBottom, plotTop, score / 10);
                if (sampleIndex === 0) { context.moveTo(px, py); } else { context.lineTo(px, py); }
            }
            context.stroke();
            context.restore();
            drawText(context, primary("高档 upscale", "upscale"), x + panelWidth - 22, panelTop + 56, 19, COLOUR.blue, { align: "right" });
        }
        const scoreAlpha = ease(ramp(time, panel.start + 1.9, 0.6));
        drawText(context, termScore.score.toFixed(2), x + panelWidth / 2, panelTop + panelHeight + 96, 72, COLOUR.text, { align: "center", family: "Consolas", alpha: scoreAlpha });
        if (panelIndex > 0) { drawText(context, "+", x - 12, panelTop + panelHeight + 90, 56, COLOUR.muted, { align: "center", family: "Consolas", alpha: scoreAlpha }); }
        context.restore();
    });
    const totalAlpha = ease(ramp(time, scene.lineEnd[3] + 0.3, 0.8));
    if (totalAlpha > 0) {
        context.save();
        context.globalAlpha = totalAlpha;
        strokeLine(context, 830, 660, 1868, 660, COLOUR.line, 2);
        drawText(context, bilingual("加权总分", "Weighted total"), 830, 740, 34, COLOUR.text, { family: "YaHeiBold" });
        drawText(context, evaluation.totalScore.toFixed(2), 1868, 752, 84, COLOUR.accent, { align: "right", family: "Consolas" });
        drawText(context, primary("≥ " + material.configurations.bedroom.scoreThreshold + " 的方案被保留  ·  schemes", "Schemes") + " scoring at least " + material.configurations.bedroom.scoreThreshold + " are kept", 830, 800, 24, COLOUR.muted);
        context.restore();
    }
}

function drawWholeScene(context, time, scene, material) {
    const lineOne = scene.lineStart[1], lineTwo = scene.lineStart[2];
    const columns = [["bedroom", "卧室", "Bedroom"], ["bathroom", "卫生间", "Bathroom"], ["kitchen", "厨房", "Kitchen"], ["living", "客厅", "Living room"]];
    const galleryAlpha = 1 - ease(ramp(time, lineOne - 0.5, 0.8));
    const hub = ease(ramp(time, cueTime(scene, 0, "枢纽", "hub") - 0.4, 0.8));
    columns.forEach(function (column, columnIndex) {
        const alpha = ease(ramp(time, cueTime(scene, 0, column[1], column[2]), 0.7)) * galleryAlpha;
        if (alpha <= 0) { return; }
        const centreX = 280 + columnIndex * 455;
        context.save();
        context.globalAlpha = alpha;
        if (column[0] === "living" && hub > 0) {
            context.globalAlpha = alpha * hub;
            context.strokeStyle = COLOUR.accent;
            context.lineWidth = 4;
            roundedRectangle(context, centreX - 200, 150, 400, 690, 14);
            context.stroke();
            drawText(context, bilingual("枢纽", "hub"), centreX, 880, 30, COLOUR.accent, { align: "center", family: "YaHeiBold" });
            context.globalAlpha = alpha;
        }
        drawText(context, primary(column[1], column[2]), centreX, IS_ENGLISH ? 226 : 210, 44, COLOUR.text, { align: "center", family: "YaHeiBold" });
        drawText(context, secondary(column[2]), centreX, 246, 24, COLOUR.muted, { align: "center" });
        const rooms = material.rooms[column[0]];
        const picks = [rooms[0], rooms[Math.min(rooms.length - 1, 5)]];
        picks.forEach(function (room, pickIndex) {
            const centreY = 385 + pickIndex * 290;
            drawPlan(context, room.plan, centreX, centreY, fitCellSize(room.plan, 300, 190), "drawing", 1);
            drawText(context, room.totalScore.toFixed(2), centreX, centreY + 138, 24, COLOUR.accent, { align: "center", family: "Consolas" });
        });
        context.restore();
    });

    const assemble = ease(ramp(time, lineOne - 0.2, 0.8)) * (1 - ease(ramp(time, lineTwo - 0.5, 0.8)));
    if (assemble > 0) {
        const apartment = material.apartments[0];
        const plan = apartment.plan;
        const cellPixelSize = fitCellSize(plan, 760, 560);
        const centreX = 640, centreY = 520;
        const converge = ease(ramp(time, cueTime(scene, 1, "接到", "attached") - 0.3, 2.4));
        const merged = ease(ramp(time, cueTime(scene, 1, "拼成", "forming") + 0.6, 0.9));
        context.save();
        material.apartmentPieces.forEach(function (piece) {
            const spread = (1 - converge) * 130;
            const appear = piece.owner === 1 ? 1 : ease(ramp(time, lineOne + 0.4 + piece.owner * 0.25, 0.7));
            const sprite = planSprite(piece.plan, cellPixelSize, "drawing");
            context.globalAlpha = assemble * (1 - merged) * appear;
            context.drawImage(sprite, Math.round(centreX - sprite.width / 2 + piece.directionX * spread), Math.round(centreY - sprite.height / 2 + piece.directionY * spread));
        });
        context.restore();
        drawPlan(context, plan, centreX, centreY, cellPixelSize, "drawing", assemble * merged);
        context.save();
        context.globalAlpha = assemble;
        const scoring = ease(ramp(time, cueTime(scene, 1, "全套", "Its score") - 0.3, 0.8));
        drawText(context, primary("客厅上预留的开口", "Access openings"), 1280, 330, 40, COLOUR.accent, { family: "YaHeiBold", alpha: 1 - merged });
        drawText(context, primary("Access openings reserved in the living room", "reserved in the living room"), 1280, 372, 24, COLOUR.muted, { alpha: 1 - merged });
        drawText(context, apartment.quantities.areaSquareMetres.toFixed(1) + " m²", 1280, 250, 72, COLOUR.text, { family: "Consolas", alpha: merged });
        drawText(context, primary("两间卧室 · 卫生间 · 厨房 · 客厅", "Two bedrooms · bathroom · kitchen · living room"), 1280, 300, IS_ENGLISH ? 24 : 28, COLOUR.muted, { alpha: merged });
        // the apartment's score: its rooms' own scores, weighted together with its shape factor
        const roomLabels = { living: primary("客厅 Living room", "Living room"), bedroom: primary("卧室 Bedroom", "Bedroom"), bathroom: primary("卫生间 Bathroom", "Bathroom"), kitchen: primary("厨房 Kitchen", "Kitchen") };
        drawText(context, bilingual("各个房间得分", "Room scores"), 1280, 384, 26, COLOUR.accent, { alpha: scoring });
        apartment.roomScores.forEach(function (roomScore, roomIndex) {
            const rowAlpha = scoring * ease(ramp(time, cueTime(scene, 1, "全套", "Its score") + roomIndex * 0.25, 0.5));
            drawText(context, roomLabels[roomScore.roomType], 1280, 428 + roomIndex * 40, 26, COLOUR.text, { alpha: rowAlpha });
            drawText(context, roomScore.totalScore.toFixed(2), 1820, 428 + roomIndex * 40, 26, COLOUR.text, { alpha: rowAlpha, align: "right", family: "Consolas" });
        });
        const shape = scoring * ease(ramp(time, cueTime(scene, 1, "体形系数", "shape factor") - 0.2, 0.6));
        const shapeTerm = apartment.termScores.filter(function (term) { return term.label === "Shape factor"; })[0];
        drawText(context, bilingual("体形系数", "Shape factor"), 1280, 664, 26, COLOUR.accent, { alpha: shape });
        drawText(context, bilingual("面积 ÷ 周长²", "area / perimeter²"), 1280, 700, 22, COLOUR.muted, { alpha: shape });
        drawText(context, shapeTerm ? shapeTerm.score.toFixed(2) : "", 1820, 664, 26, COLOUR.text, { alpha: shape, align: "right", family: "Consolas" });
        const total = scoring * ease(ramp(time, scene.lineEnd[1] - 1.2, 0.6));
        context.globalAlpha = assemble * total;
        strokeLine(context, 1280, 730, 1820, 730, COLOUR.line, 2);
        context.globalAlpha = assemble;
        drawText(context, bilingual("加权总分", "Weighted total"), 1280, 790, 30, COLOUR.text, { alpha: total, family: "YaHeiBold" });
        drawText(context, apartment.totalScore.toFixed(2), 1820, 796, 56, COLOUR.accent, { alpha: total, align: "right", family: "Consolas" });
        context.restore();
    }
    const archive = ease(ramp(time, lineTwo - 0.1, 0.8));
    if (archive > 0) {
        material.images.results.forEach(function (image, imageIndex) {
            const alpha = archive * ease(ramp(time, lineTwo + imageIndex * 0.35, 0.7));
            if (alpha <= 0) { return; }
            const scale = Math.min(520 / image.width, 620 / image.height);
            const width = image.width * scale, height = image.height * scale, centreX = 370 + imageIndex * 590;
            context.save();
            context.globalAlpha = alpha;
            context.drawImage(image, centreX - width / 2, 520 - height / 2, width, height);
            context.strokeStyle = COLOUR.line;
            context.lineWidth = 2;
            context.strokeRect(centreX - width / 2, 520 - height / 2, width, height);
            context.restore();
        });
        drawText(context, primary("2022 · 原始输出（DXF）  Original output", "2022 · Original output (DXF)"), 1840, 92, 26, COLOUR.muted, { align: "right", alpha: archive });
    }
}

function drawModelScene(context, time, scene, material) {
    const lineOne = scene.lineStart[1];
    // the hero: the plan stands up and turns, rendered frame by frame in Blender Cycles
    const heroAlpha = ease(ramp(time, 0, 0.5)) * (1 - ease(ramp(time, lineOne - 0.6, 0.9)));
    if (heroAlpha > 0 && renderedFrameCache.image) {
        context.save();
        context.globalAlpha = heroAlpha;
        context.drawImage(renderedFrameCache.image, 0, 0, FRAME_WIDTH, FRAME_HEIGHT);
        context.restore();
    }
    const gridAlpha = ease(ramp(time, lineOne - 0.6, 0.9));
    if (gridAlpha > 0) {
        const zoom = 1 + 0.04 * ramp(time, lineOne - 0.6, scene.duration - lineOne + 0.6);
        context.save();
        context.globalAlpha = gridAlpha;
        context.drawImage(material.images.modelGrid, FRAME_WIDTH * (1 - zoom) / 2, FRAME_HEIGHT * (1 - zoom) / 2, FRAME_WIDTH * zoom, FRAME_HEIGHT * zoom);
        context.restore();
    }
    const captions = [
        [cueTime(scene, 0, "三维建模", "three dimensions") - 0.3, primary("三维预览", "3D preview"), secondary("3D preview"), 56],
        [cueTime(scene, 0, "以现在", "With today") - 0.2, "Blender · Cycles", primary("路径追踪：柔和的阴影，反弹的光", "Path tracing: soft shadows, bounced light"), 36, secondary("Path tracing: soft shadows, bounced light")]
    ];
    captions.forEach(function (caption, captionIndex) {
        const alpha = ease(ramp(time, caption[0], 0.8)) * heroAlpha;
        const y = 330 + captionIndex * 130;
        drawText(context, caption[1], 1330, y, caption[3], COLOUR.text, { alpha: alpha, family: captionIndex === 0 ? "YaHeiBold" : "Consolas" });
        drawText(context, caption[2], 1330, y + 40, captionIndex === 0 ? 26 : 22, COLOUR.muted, { alpha: alpha });
        if (caption[4]) { drawText(context, caption[4], 1330, y + 70, 22, COLOUR.muted, { alpha: alpha }); }
    });
}

function drawStudioScene(context, time, scene, material) {
    const lineOne = scene.lineStart[1];
    const before = ease(ramp(time, 0.1, 0.7)) * (1 - ease(ramp(time, lineOne - 0.5, 0.8)));
    if (before > 0) {
        context.save();
        context.globalAlpha = before;
        const files = ["design.py", "01 - room / jigsaw.py", "02 - toilet / jigsaw.py", "03 - kitchen / jigsaw.py", "04 - hall / jigsaw.py", "05 - combine / functions.py", "05 - combine / output_dxf_png.py"];
        drawText(context, "2022", 150, 230, 30, COLOUR.accent, { family: "Consolas" });
        files.forEach(function (fileName, fileIndex) {
            const alpha = ease(ramp(time, 0.3 + fileIndex * 0.22, 0.5));
            drawText(context, fileName, 150, 310 + fileIndex * 62, 34, COLOUR.text, { family: "Consolas", alpha: alpha });
        });
        const codeAlpha = ease(ramp(time, cueTime(scene, 0, "评分", "scoring") - 0.4, 0.7));
        context.globalAlpha = before * codeAlpha;
        context.fillStyle = "#0d1117";
        roundedRectangle(context, 880, 290, 920, 330, 14);
        context.fill();
        const codeLines = [
            ["AreaReality = AreaCell * ", "0.55", " ** 2"],
            ["Scoring = ", "-0.04", " * (AreaReality - ", "10", ") ** 2 + ", "10"],
            ["Storage_s = Storage ** ", "0.7"],
            ["Scoring_pe = PeriEfficiency * ", "180"],
            ["TotalScore = Scoring + Storage_s + Scoring_pe"]
        ];
        codeLines.forEach(function (parts, lineIndex) {
            let x = 920;
            context.font = "28px Consolas";
            parts.forEach(function (part, partIndex) {
                drawText(context, part, x, 356 + lineIndex * 54, 28, partIndex % 2 === 1 ? COLOUR.accent : "#c9d1d9", { family: "Consolas" });
                x += context.measureText(part).width;
            });
        });
        drawText(context, "01 - room / jigsaw.py · " + bilingual("系数写在代码里", "coefficients live in the code"), 880, 664, 22, COLOUR.muted);
        context.restore();
    }
    const shots = material.images.shots;
    const cues = [["现在", "Today"], ["看拼图库", "inspect"], ["改参数", "set parameters"], ["生成房间", "generate rooms"], ["组合户型", "combine apartments"], ["对它", "Tuning"]].map(function (keywords) {
        return cueTime(scene, 1, keywords[0], keywords[1]) - lineOne;
    });
    const labels = [["① 画样本", "Draw samples"], ["② 看拼图库", "Inspect the tile library"], ["③ 改参数", "Set the parameters"], ["③ 生成房间", "Generate rooms"], ["④ 组合户型", "Combine apartments"], ["⑤ 三维展示", "3D collection"]];
    const after = ease(ramp(time, lineOne - 0.1, 0.8));
    if (after <= 0) { return; }
    let shotIndex = 0;
    for (let cueIndex = 0; cueIndex < cues.length; cueIndex++) {
        if (time >= lineOne + cues[cueIndex]) { shotIndex = cueIndex; }
    }
    const frameWidth = 1240, frameHeight = frameWidth * 1080 / 1920, frameLeft = 960 - frameWidth / 2, frameTop = 128;
    context.save();
    context.globalAlpha = after;
    context.shadowColor = "rgba(0,0,0,0.6)";
    context.shadowBlur = 40;
    context.fillStyle = "#0d1117";
    roundedRectangle(context, frameLeft - 6, frameTop - 6, frameWidth + 12, frameHeight + 12, 12);
    context.fill();
    context.shadowBlur = 0;
    roundedRectangle(context, frameLeft, frameTop, frameWidth, frameHeight, 8);
    context.clip();
    for (let index = Math.max(0, shotIndex - 1); index <= shotIndex; index++) {
        const sinceCue = time - (lineOne + cues[index]);
        const zoom = 1 + 0.035 * clamp01(sinceCue / 4);
        context.globalAlpha = after * (index === shotIndex ? ease(ramp(sinceCue, 0, 0.45)) : 1);
        context.drawImage(shots[index], frameLeft - frameWidth * (zoom - 1) / 2, frameTop - frameHeight * (zoom - 1) / 2, frameWidth * zoom, frameHeight * zoom);
    }
    context.restore();
    context.save();
    context.globalAlpha = after;
    context.fillStyle = COLOUR.accent;
    const label = labels[shotIndex];
    const labelText = primary(label[0], label[0].slice(0, 2) + label[1]);
    context.font = "34px YaHeiBold";
    const labelWidth = context.measureText(labelText).width + 44;
    roundedRectangle(context, frameLeft + 24, frameTop + frameHeight - 78, labelWidth, 54, 8);
    context.fill();
    drawText(context, labelText, frameLeft + 46, frameTop + frameHeight - 40, 34, "#1a1203", { family: "YaHeiBold" });
    context.restore();
}

function drawEndingScene(context, time, scene, material) {
    const photos = material.images.photos;
    const lineOne = scene.lineStart[1], lineTwo = scene.lineStart[2];
    // line 0: the whiteboard in the study
    const boards = ["IMG_3156", "13", "15"];
    const boardsAlpha = 1 - ease(ramp(time, lineOne - 0.5, 0.8));
    const boardSpan = (lineOne - 0.2) / boards.length;
    boards.forEach(function (name, boardIndex) {
        const start = boardIndex * boardSpan;
        const alpha = ease(ramp(time, start, 0.7)) * (boardIndex === boards.length - 1 ? 1 : 1 - ease(ramp(time, start + boardSpan, 0.7)));
        drawPhotoCard(context, photos[name], 700, 470, 1160, 700, alpha * boardsAlpha, 1 + 0.03 * clamp01((time - start) / boardSpan));
    });
    drawPhotoCard(context, photos["14"], 1590, 470, 480, 700, ease(ramp(time, cueTime(scene, 0, "看着白板", "staring") - 0.3, 0.8)) * boardsAlpha, 1);
    drawText(context, primary("2022 · 书房的白板  The whiteboard in the study", "2022 · The whiteboard in the study"), 120, 92, 26, COLOUR.muted, { alpha: ease(ramp(time, 0.3, 0.8)) * boardsAlpha });

    // lines 1 and 2: the notebook pages drift past, then the worked notes
    const pages = ["01", "02", "03", "04", "05", "06", "07", "08", "09"];
    const notesCue = cueTime(scene, 2, "希望", "I hope");
    const pagesAlpha = ease(ramp(time, lineOne - 0.3, 0.8)) * (1 - ease(ramp(time, notesCue - 0.5, 0.8)));
    if (pagesAlpha > 0) {
        const drift = (time - lineOne) * 62;
        pages.forEach(function (name, pageIndex) {
            const centreX = 330 + pageIndex * 470 - drift;
            if (centreX < -300 || centreX > FRAME_WIDTH + 300) { return; }
            drawPhotoCard(context, photos[name], centreX, 480, 430, 640, pagesAlpha, 1);
        });
        drawText(context, primary("2021 – 2022 · 笔记本  Notebook pages", "2021 – 2022 · Notebook pages"), 120, 92, 26, COLOUR.muted, { alpha: pagesAlpha });
    }
    const endCard = ease(ramp(time, scene.lineEnd[2] + 0.5, 1.2));
    const notesAlpha = ease(ramp(time, notesCue - 0.2, 0.8));
    if (notesAlpha > 0) {
        ["10", "11", "12", "IMG_0361"].forEach(function (name, noteIndex) {
            const alpha = notesAlpha * ease(ramp(time, notesCue + noteIndex * 0.3, 0.7));
            drawPhotoCard(context, photos[name], 520 + (noteIndex % 2) * 880, 300 + Math.floor(noteIndex / 2) * 390, 800, 350, alpha, 1);
        });
    }
    if (endCard > 0) {
        context.fillStyle = "rgba(22,27,34," + (0.93 * endCard).toFixed(3) + ")";
        context.fillRect(0, 0, FRAME_WIDTH, FRAME_HEIGHT);
        context.save();
        context.globalAlpha = endCard;
        drawText(context, primary("波函数坍缩 · 住宅平面生成", "Floor Plan Generator"), 960, 420, 78, COLOUR.text, { align: "center", family: "YaHeiBold" });
        drawText(context, primary("Floor Plan Generator Using the Wave Function Collapse Algorithm", "Using the Wave Function Collapse Algorithm"), 960, 490, 32, COLOUR.muted, { align: "center" });
        context.fillStyle = COLOUR.accent;
        context.fillRect(900, 540, 120, 4);
        drawText(context, "Qian Li", 960, 620, 38, COLOUR.text, { align: "center" });
        drawText(context, primary("独立研究 Independent study, 2022  ·  单页界面 single-page studio, 2026", "Independent study, 2022  ·  Single-page studio, 2026"), 960, 672, 26, COLOUR.muted, { align: "center" });
        drawText(context, "github.com/ludwigpeking/WFC_floorPlan_Generation", 960, 760, 26, COLOUR.accent, { align: "center", family: "Consolas" });
        drawText(context, primary("画面中的平面均由该程序生成  ·  The plans in this film were generated by the program", "The plans in this film were generated by the program"), 960, 840, 22, COLOUR.muted, { align: "center" });
        context.restore();
    }
}

function drawBackground(context, time) {
    context.fillStyle = COLOUR.background;
    context.fillRect(0, 0, FRAME_WIDTH, FRAME_HEIGHT);
    context.strokeStyle = "rgba(255,255,255,0.028)";
    context.lineWidth = 1;
    const offset = (time * 4) % 55;
    context.beginPath();
    for (let x = -offset; x < FRAME_WIDTH; x += 55) { context.moveTo(x, 0); context.lineTo(x, FRAME_HEIGHT); }
    for (let y = 0; y < FRAME_HEIGHT; y += 55) { context.moveTo(0, y); context.lineTo(FRAME_WIDTH, y); }
    context.stroke();
}

function wrapSubtitle(context, text, maximumWidth, splitOnPunctuation) {
    if (context.measureText(text).width <= maximumWidth) { return [text]; }
    if (splitOnPunctuation) {
        let bestIndex = -1;
        for (let index = 1; index < text.length - 1; index++) {
            if ("，。：；、".indexOf(text[index]) >= 0 && (bestIndex < 0 || Math.abs(index - text.length / 2) < Math.abs(bestIndex - text.length / 2))) { bestIndex = index; }
        }
        if (bestIndex > 0) { return [text.slice(0, bestIndex + 1), text.slice(bestIndex + 1)]; }
    }
    const words = text.split(" ");
    let bestSplit = 1, bestDifference = Infinity;
    for (let index = 1; index < words.length; index++) {
        const difference = Math.abs(context.measureText(words.slice(0, index).join(" ")).width - context.measureText(words.slice(index).join(" ")).width);
        if (difference < bestDifference) { bestDifference = difference; bestSplit = index; }
    }
    return [words.slice(0, bestSplit).join(" "), words.slice(bestSplit).join(" ")];
}

function drawSubtitles(context, line, alpha) {
    if (!line || alpha <= 0) { return; }
    context.save();
    context.globalAlpha = alpha;
    context.shadowColor = "rgba(0,0,0,0.9)";
    context.shadowBlur = 12;
    if (IS_ENGLISH) {
        context.font = "38px YaHei";
        const onlyLines = wrapSubtitle(context, line.en, 1560, false);
        let lineY = 1034 - (onlyLines.length - 1) * 50;
        for (const text of onlyLines) { drawText(context, text, 960, lineY, 38, "#ffffff", { align: "center" }); lineY += 50; }
        context.restore();
        return;
    }
    context.font = "44px YaHei";
    const chineseLines = wrapSubtitle(context, line.zh, 1560, true);
    context.font = "27px YaHei";
    const englishLines = wrapSubtitle(context, line.en, 1500, false);
    let y = 1046 - (englishLines.length - 1) * 34 - 44 - (chineseLines.length - 1) * 56;
    for (const text of chineseLines) { drawText(context, text, 960, y, 44, "#ffffff", { align: "center" }); y += 56; }
    y -= 12;
    for (const text of englishLines) { drawText(context, text, 960, y, 27, "#c2ccd8", { align: "center" }); y += 34; }
    context.restore();
}

function buildTimeline() {
    const lines = JSON.parse(fs.readFileSync(path.join(__dirname, "narration.json"), "utf8"));
    for (const line of lines) {
        line.audioPath = path.join(__dirname, IS_ENGLISH ? "voice-en" : "voice", line.name + ".mp3");
        const probe = childProcess.execFileSync("ffprobe", ["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", line.audioPath]);
        line.durationSeconds = parseFloat(String(probe));
    }
    let cursor = 0;
    for (const scene of SCENES) {
        scene.start = cursor;
        scene.lines = lines.filter(function (line) { return line.scene === scene.name; });
        scene.lineStart = [];
        scene.lineEnd = [];
        let local = SCENE_LEAD_SECONDS;
        for (const line of scene.lines) {
            line.start = scene.start + local;
            line.end = line.start + line.durationSeconds;
            // a long line is shown as several subtitles, each for its share of the clip
            const lineParts = line.parts || [{ zh: line.zh, en: line.en }];
            let partStart = line.start;
            line.subtitles = lineParts.map(function (part) {
                const spokenField = IS_ENGLISH ? "en" : "zh";
                const partDuration = line.durationSeconds * part[spokenField].length / lineParts.reduce(function (sum, linePart) { return sum + linePart[spokenField].length; }, 0);
                const subtitle = { zh: part.zh, en: part.en, start: partStart, end: partStart + partDuration };
                partStart += partDuration;
                return subtitle;
            });
            scene.lineStart.push(local);
            scene.lineEnd.push(local + line.durationSeconds);
            local += line.durationSeconds + LINE_GAP_SECONDS;
        }
        scene.duration = local - LINE_GAP_SECONDS + SCENE_TAIL_SECONDS + scene.tailSeconds;
        cursor += scene.duration;
        scene.end = cursor;
    }
    return { lines: lines, totalSeconds: cursor };
}

function renderFrame(context, time, timeline, material) {
    let scene = SCENES[SCENES.length - 1];
    for (const candidate of SCENES) {
        if (time < candidate.end) { scene = candidate; break; }
    }
    const local = time - scene.start;
    drawBackground(context, time);
    context.save();
    scene.draw(context, local, scene, material);
    context.restore();
    if (scene.chapter) {
        const alpha = ease(ramp(local, 0.1, 0.6));
        context.fillStyle = COLOUR.accent;
        context.globalAlpha = alpha;
        context.fillRect(60, 64, 6, 34);
        context.globalAlpha = 1;
        drawText(context, scene.chapter, 82, 92, 28, COLOUR.text, { alpha: alpha });
    }
    context.fillStyle = COLOUR.line;
    context.fillRect(0, 0, FRAME_WIDTH, 4);
    context.fillStyle = COLOUR.accent;
    context.fillRect(0, 0, FRAME_WIDTH * time / timeline.totalSeconds, 4);
    let currentLine = null;
    for (const line of timeline.lines) {
        for (const subtitle of line.subtitles) {
            const isLast = subtitle === line.subtitles[line.subtitles.length - 1];
            if (time >= subtitle.start - (subtitle === line.subtitles[0] ? 0.1 : 0) && time <= subtitle.end + (isLast ? 0.3 : 0)) { currentLine = subtitle; }
        }
    }
    if (currentLine) {
        const gradient = context.createLinearGradient(0, 860, 0, 1080);
        gradient.addColorStop(0, "rgba(13,17,23,0)");
        gradient.addColorStop(1, "rgba(13,17,23,0.92)");
        context.fillStyle = gradient;
        context.fillRect(0, 860, FRAME_WIDTH, 220);
        drawSubtitles(context, currentLine, Math.min(ramp(time, currentLine.start - 0.1, 0.2), 1 - ramp(time, currentLine.end + 0.1, 0.2)));
    }
    // dip to black between scenes
    const edge = Math.min(local, scene.duration - local);
    const isFilmStart = scene === SCENES[0] && local < 1, isFilmEnd = scene === SCENES[SCENES.length - 1] && scene.duration - local < 1.2;
    const dip = isFilmStart || isFilmEnd ? 1 - clamp01(edge / 1.0) : 1 - clamp01(edge / 0.35);
    if (dip > 0) {
        context.fillStyle = "rgba(10,13,18," + (isFilmStart || isFilmEnd ? dip : dip * 0.85).toFixed(3) + ")";
        context.fillRect(0, 0, FRAME_WIDTH, FRAME_HEIGHT);
    }
}

function subtitleTimestamp(seconds) {
    const milliseconds = Math.round(seconds * 1000);
    const pad = function (value, length) { return String(value).padStart(length, "0"); };
    return pad(Math.floor(milliseconds / 3600000), 2) + ":" + pad(Math.floor(milliseconds / 60000) % 60, 2) + ":" + pad(Math.floor(milliseconds / 1000) % 60, 2) + "," + pad(milliseconds % 1000, 3);
}

async function main() {
    const argumentList = process.argv.slice(2);
    const previewIndex = argumentList.indexOf("--preview");
    const outputIndex = argumentList.indexOf("--output");
    const outputPath = path.join(__dirname, outputIndex >= 0 ? argumentList[outputIndex + 1] : (IS_ENGLISH ? "WFC_Floor_Plan_Introduction_EN.mp4" : "WFC_Floor_Plan_Introduction.mp4"));

    const timeline = buildTimeline();
    console.log("film length: " + timeline.totalSeconds.toFixed(1) + " s");
    for (const scene of SCENES) { console.log("  " + scene.name.padEnd(10) + scene.start.toFixed(1).padStart(7) + " → " + scene.end.toFixed(1)); }
    console.log("generating material…");
    const material = prepareMaterial(argumentList.indexOf("--refresh") >= 0);
    material.samplePlan = engine.createRoomPlan(material.sampleGrids.bedroom[0], "bedroom");
    material.sortedBedroomTiles = material.libraries.bedroom.tiles.slice().sort(function (first, second) { return second.frequency - first.frequency; });
    // the best apartment split into its rooms, for the assembly animation
    const plan = material.apartments[0].plan;
    material.apartmentPieces = [];
    let livingCentreX = 0, livingCentreY = 0, livingCellCount = 0;
    for (let cellIndex = 0; cellIndex < plan.cells.length; cellIndex++) {
        if (plan.owners[cellIndex] === 1) { livingCentreX += cellIndex % plan.columnCount; livingCentreY += Math.floor(cellIndex / plan.columnCount); livingCellCount++; }
    }
    livingCentreX /= livingCellCount; livingCentreY /= livingCellCount;
    for (let owner = 1; owner < plan.ownerRoomTypes.length; owner++) {
        const owners = new Int16Array(plan.owners.length), cells = new Int32Array(plan.cells.length);
        let centreX = 0, centreY = 0, cellCount = 0;
        for (let cellIndex = 0; cellIndex < plan.cells.length; cellIndex++) {
            if (plan.owners[cellIndex] !== owner) { continue; }
            // on its own, the living room still shows its access openings
            owners[cellIndex] = 1;
            cells[cellIndex] = plan.cells[cellIndex];
            centreX += cellIndex % plan.columnCount; centreY += Math.floor(cellIndex / plan.columnCount); cellCount++;
        }
        if (owner === 1) {
            for (let cellIndex = 0; cellIndex < plan.cells.length; cellIndex++) {
                const code = plan.cells[cellIndex];
                if (plan.owners[cellIndex] > 1 && (code === 9090 || code === 9390)) { owners[cellIndex] = 1; cells[cellIndex] = code === 9090 ? 9080 : 9086; }
            }
        }
        const offsetX = centreX / cellCount - livingCentreX, offsetY = centreY / cellCount - livingCentreY;
        const length = Math.hypot(offsetX, offsetY) || 1;
        material.apartmentPieces.push({
            owner: owner,
            plan: { rowCount: plan.rowCount, columnCount: plan.columnCount, cells: cells, owners: owners, ownerRoomTypes: [null, plan.ownerRoomTypes[owner]] },
            directionX: owner === 1 ? 0 : offsetX / length, directionY: owner === 1 ? 0 : offsetY / length
        });
    }
    const imageDirectory = path.join(PROJECT_DIRECTORY, "img");
    material.images = {
        plan: await loadImage(path.join(imageDirectory, "10.png")),
        pixelated: await loadImage(path.join(imageDirectory, "11.png")),
        manualLibrary: await loadImage(path.join(imageDirectory, "12.png")),
        results: [await loadImage(path.join(imageDirectory, "01.png")), await loadImage(path.join(imageDirectory, "02.png")), await loadImage(path.join(imageDirectory, "03.png"))],
        shots: []
    };
    material.images.photos = {};
    for (const fileName of fs.readdirSync(path.join(__dirname, "photos"))) {
        material.images.photos[fileName.replace(/\.jpg$/, "")] = await loadImage(path.join(__dirname, "photos", fileName));
    }
    const heroDirectory = path.join(__dirname, "renders", "hero");
    material.heroFrameCount = fs.existsSync(heroDirectory) ? fs.readdirSync(heroDirectory).filter(function (name) { return name.endsWith(".png"); }).length : 0;
    material.images.modelGrid = await loadImage(path.join(__dirname, "renders", "grid.png"));
    for (const shotName of ["1-samples", "2-tiles", "3-parameters", "4-generate", "5-combine", "6-collection"]) {
        material.images.shots.push(await loadImage(path.join(__dirname, "shots", shotName + ".png")));
    }

    const canvas = createCanvas(FRAME_WIDTH, FRAME_HEIGHT);
    const context = canvas.getContext("2d");

    if (previewIndex >= 0) {
        const previewDirectory = path.join(__dirname, "preview");
        fs.mkdirSync(previewDirectory, { recursive: true });
        for (const timeText of argumentList[previewIndex + 1].split(",")) {
            await loadRenderedFrame(parseFloat(timeText), material);
            renderFrame(context, parseFloat(timeText), timeline, material);
            const framePath = path.join(previewDirectory, (IS_ENGLISH ? "en-" : "") + "frame-" + parseFloat(timeText).toFixed(1).padStart(6, "0") + ".jpg");
            fs.writeFileSync(framePath, canvas.toBuffer("image/jpeg", 85));
            console.log("wrote " + framePath);
        }
        return;
    }

    // subtitles as a separate file too
    const allSubtitles = [];
    for (const line of timeline.lines) { allSubtitles.push.apply(allSubtitles, line.subtitles); }
    const subtitleBlocks = allSubtitles.map(function (subtitle, subtitleIndex) {
        return (subtitleIndex + 1) + "\n" + subtitleTimestamp(subtitle.start) + " --> " + subtitleTimestamp(subtitle.end) + "\n" + (IS_ENGLISH ? "" : subtitle.zh + "\n") + subtitle.en + "\n";
    });
    fs.writeFileSync(outputPath.replace(/\.mp4$/, ".srt"), subtitleBlocks.join("\n"), "utf8");

    const ffmpegArguments = ["-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", FRAME_WIDTH + "x" + FRAME_HEIGHT, "-r", String(FRAMES_PER_SECOND), "-i", "pipe:0"];
    const filterParts = [];
    timeline.lines.forEach(function (line, lineIndex) {
        ffmpegArguments.push("-i", line.audioPath);
        const delayMilliseconds = Math.round(line.start * 1000);
        filterParts.push("[" + (lineIndex + 1) + ":a]aresample=48000,adelay=" + delayMilliseconds + ":all=1[a" + lineIndex + "]");
    });
    const mixInputs = timeline.lines.map(function (line, lineIndex) { return "[a" + lineIndex + "]"; }).join("");
    filterParts.push(mixInputs + "amix=inputs=" + timeline.lines.length + ":normalize=0,apad=whole_dur=" + timeline.totalSeconds.toFixed(3) + ",loudnorm=I=-16:TP=-1.5:LRA=11[narration]");
    ffmpegArguments.push("-filter_complex", filterParts.join(";"), "-map", "0:v", "-map", "[narration]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-t", timeline.totalSeconds.toFixed(3), "-movflags", "+faststart", outputPath);
    const ffmpeg = childProcess.spawn("ffmpeg", ffmpegArguments, { stdio: ["pipe", "ignore", "inherit"] });
    const finished = new Promise(function (resolve, reject) {
        ffmpeg.on("close", function (exitCode) { if (exitCode === 0) { resolve(); } else { reject(new Error("ffmpeg exited with code " + exitCode)); } });
    });
    const frameCount = Math.ceil(timeline.totalSeconds * FRAMES_PER_SECOND);
    for (let frameIndex = 0; frameIndex < frameCount; frameIndex++) {
        await loadRenderedFrame(frameIndex / FRAMES_PER_SECOND, material);
        renderFrame(context, frameIndex / FRAMES_PER_SECOND, timeline, material);
        const pixels = context.getImageData(0, 0, FRAME_WIDTH, FRAME_HEIGHT).data;
        const canContinue = ffmpeg.stdin.write(Buffer.from(pixels.buffer, pixels.byteOffset, pixels.byteLength));
        if (!canContinue) { await new Promise(function (resolve) { ffmpeg.stdin.once("drain", resolve); }); }
        if (frameIndex % 300 === 0) { console.log("frame " + frameIndex + " / " + frameCount); }
    }
    ffmpeg.stdin.end();
    await finished;
    console.log("wrote " + outputPath);
}

main().catch(function (error) {
    console.error(error);
    process.exit(1);
});
