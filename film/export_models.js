// Writes the 3D models of the film's apartments to renders/models.json for render_models.py (Blender).
// Run after make_film.js has produced material_cache.json.
"use strict";

const fs = require("fs");
const path = require("path");

const pageSource = fs.readFileSync(path.resolve(__dirname, "..", "index.html"), "utf8");
const engineSource = pageSource.split("/* ENGINE-BEGIN */")[1].split("/* ENGINE-END */")[0];
const engine = new Function(engineSource + "\nreturn { buildPlanScene };")();
const cache = JSON.parse(fs.readFileSync(path.join(__dirname, "material_cache.json"), "utf8"));

const models = cache.apartments.slice(0, 7).map(function (stored, index) {
    const plan = {
        rowCount: stored.rowCount, columnCount: stored.columnCount, cells: Int32Array.from(stored.cells),
        owners: Int16Array.from(stored.owners), ownerRoomTypes: stored.ownerRoomTypes
    };
    const scene = engine.buildPlanScene(plan);
    return { name: "apartment-" + index, rowCount: scene.rowCount, columnCount: scene.columnCount, solids: scene.floors.concat(scene.solids) };
});
fs.mkdirSync(path.join(__dirname, "renders"), { recursive: true });
fs.writeFileSync(path.join(__dirname, "renders", "models.json"), JSON.stringify(models));
console.log("wrote " + models.length + " models, " + models[0].solids.length + " solids in the first");
