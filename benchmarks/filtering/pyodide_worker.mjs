// Run g1.measure() for one set inside Pyodide, under Node, and print its JSON.
//
//   node pyodide_worker.mjs <pyodide-dir> <set> [data-dir]
//
// <pyodide-dir> is an unpacked Pyodide distribution holding pyodide.mjs,
// pyodide-lock.json and the numpy wheel (the release tarball has them all);
// nothing is fetched from a CDN.  The filtering subpackage is copied into
// Pyodide's file system as the top-level package `zvfilter` -- it imports
// only numpy and itself -- and g1.py next to it, so the code measured is the
// code in this checkout.  run.py drives this once per set, in a fresh
// process, so each set's peak wasm memory is its own.

import { readdirSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const [pyodideDir, setName, dataDir] = process.argv.slice(2);
if (!pyodideDir || !setName) {
  console.error("usage: node pyodide_worker.mjs <pyodide-dir> <set> [data-dir]");
  process.exit(2);
}
const here = dirname(fileURLToPath(import.meta.url));
const packageDir = resolve(here, "../../zarr_vectors_tools/filtering");

const { loadPyodide } = await import(pathToFileURL(join(resolve(pyodideDir), "pyodide.mjs")).href);
const pyodide = await loadPyodide({ indexURL: resolve(pyodideDir) + "/" });
await pyodide.loadPackage("numpy", { messageCallback: () => {} });

const fs = pyodide.FS;
fs.mkdirTree("/bench/zvfilter");
for (const name of readdirSync(packageDir).filter((n) => n.endsWith(".py"))) {
  fs.writeFile(`/bench/zvfilter/${name}`, readFileSync(join(packageDir, name)));
}
fs.writeFile("/bench/g1.py", readFileSync(join(here, "g1.py")));
if (dataDir) {
  fs.mkdirTree("/data");
  pyodide.mountNodeFS("/data", resolve(dataDir));
}

const heapMb = () => pyodide._module.HEAPU8.buffer.byteLength / 1e6;
const startMb = heapMb();
const json = await pyodide.runPythonAsync(`
import json, platform, sys
sys.path.insert(0, "/bench")
import numpy as np
import g1, pyodide_js, zvfilter

def heap_mb():
    return pyodide_js._module.HEAPU8.buffer.byteLength / 1e6

tg = g1.make_set(zvfilter, ${JSON.stringify(setName)}, "/data")
result = g1.measure(zvfilter, tg, memory_mb=heap_mb)
result["runtime"] = {"name": "pyodide", "python": platform.python_version(),
                     "numpy": np.__version__, "pyodide": ${JSON.stringify(pyodide.version)}}
json.dumps(result)
`);
const result = JSON.parse(json);
// Wasm memory only grows, so its size at the end is the peak.
result.peak_mb = Math.round(heapMb());
result.startup_mb = Math.round(startMb);
console.log(JSON.stringify(result));
