// UI for the in-browser leakcheck. All Python work happens in worker.js; files never leave the browser.

const worker = new Worker("worker.js", { type: "module" });
const $ = (sel) => document.querySelector(sel);
const loaded = { train: null, test: null };
let pendingDemoTarget = null;

const DEMOS = {
  houses: { target: "price_nok", files: { train: "demo/houses_train.csv", test: "demo/houses_test.csv" } },
  patients: { target: "diagnosis", files: { train: "demo/patients_train.csv", test: "demo/patients_test.csv" } },
};
const ICONS = { fail: "✗", warn: "⚠", pass: "✓" };
const ORDER = { fail: 0, warn: 1, pass: 2 };

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function setEngine(text, cls = "") {
  const el = $("#engine");
  el.textContent = text;
  el.className = "pill " + cls;
}

// --- Loading files ---------------------------------------------------------------------------------------

async function sendFile(name, filename, buffer) {
  const drop = $(`#drop-${name}`);
  drop.classList.remove("done", "error");
  drop.querySelector(".drop-hint").textContent = `Reading ${filename}…`;
  loaded[name] = null;
  $("#config").hidden = true;
  $("#results").hidden = true;
  worker.postMessage({ type: "load", name, filename, bytes: buffer }, [buffer]);
}

for (const drop of document.querySelectorAll(".drop")) {
  const name = drop.dataset.name;
  const input = drop.querySelector("input");
  input.addEventListener("change", async () => {
    const file = input.files[0];
    if (file) sendFile(name, file.name, await file.arrayBuffer());
    input.value = "";
  });
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", async (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    const file = e.dataTransfer.files[0];
    if (file) sendFile(name, file.name, await file.arrayBuffer());
  });
}

for (const btn of document.querySelectorAll("[data-demo]")) {
  btn.addEventListener("click", async () => {
    const demo = DEMOS[btn.dataset.demo];
    pendingDemoTarget = demo.target;
    for (const [name, url] of Object.entries(demo.files)) {
      const res = await fetch(url);
      sendFile(name, url.split("/").pop(), await res.arrayBuffer());
    }
  });
}

// --- Configuration ---------------------------------------------------------------------------------------

function fillSelect(select, options, selected, noneLabel) {
  select.innerHTML = "";
  if (noneLabel) select.add(new Option(noneLabel, ""));
  for (const opt of options) select.add(new Option(opt, opt, false, opt === selected));
}

function onSuggested({ target, groups, times }) {
  const trainCols = loaded.train.columns;
  const shared = trainCols.filter((c) => loaded.test.columns.includes(c));
  const chosen = pendingDemoTarget && trainCols.includes(pendingDemoTarget) ? pendingDemoTarget : target;
  pendingDemoTarget = null;
  fillSelect($("#target"), trainCols, chosen, "— none —");
  // An empty value lets leakcheck auto-detect; picking a column forces that one.
  const auto = (found) => `Auto-detect (${found.length ? found.join(", ") : "none found"})`;
  fillSelect($("#group"), shared, null, auto(groups));
  fillSelect($("#time"), shared, null, auto(times));
  $("#config").hidden = false;
  if (chosen) runCheck();  // demos and obvious targets go straight to results
}

$("#similarity").addEventListener("input", (e) => {
  $("#sim-out").textContent = Math.round(e.target.value * 100) + "%";
});

function runCheck() {
  const btn = $("#run");
  btn.disabled = true;
  btn.textContent = "Checking…";
  worker.postMessage({
    type: "run",
    target: $("#target").value,
    group: $("#group").value,
    time: $("#time").value,
    similarity: parseFloat($("#similarity").value),
  });
}
$("#run").addEventListener("click", runCheck);

// --- Results ---------------------------------------------------------------------------------------------

function renderResults({ findings, seconds, train, test }) {
  findings.sort((a, b) => ORDER[a.status] - ORDER[b.status]);
  const fails = findings.filter((f) => f.status === "fail").length;
  const warns = findings.filter((f) => f.status === "warn").length;
  const [cls, headline] = fails
    ? ["fail", `${fails} problem${fails > 1 ? "s" : ""}, ${warns} warning${warns === 1 ? "" : "s"} — likely leakage`]
    : warns
      ? ["warn", `No problems, ${warns} warning${warns > 1 ? "s" : ""} worth a look`]
      : ["pass", "All clear — no leakage detected"];

  const items = findings.map((f) => {
    const details = Object.keys(f.details).length
      ? `<pre>${escapeHtml(JSON.stringify(f.details, null, 2))}</pre>` : "";
    return `<li class="finding ${f.status}"><details><summary>
        <span class="icon">${ICONS[f.status]}</span><span class="msg">${escapeHtml(f.message)}</span>
        <span class="tag">${escapeHtml(f.check)}</span></summary>${details}</details></li>`;
  }).join("");

  const el = $("#results");
  el.innerHTML = `<h2><span class="step">3</span> Results</h2>
    <div class="summary ${cls}">${headline}
      <small>train ${train[0].toLocaleString()} rows × ${train[1]} cols · test ${test[0].toLocaleString()} rows × ${test[1]} cols · checked in ${seconds}s</small></div>
    <ul class="findings">${items}</ul>`;
  el.hidden = false;
  el.scrollIntoView({ behavior: "smooth", block: "start" });
}

// --- Worker messages -------------------------------------------------------------------------------------

// Errors thrown while the worker script itself loads (e.g. the Pyodide CDN is unreachable) never reach
// onmessage, so without this the page would say "Starting Python…" forever.
worker.onerror = (e) => {
  setEngine("Couldn't load Python — check your connection and reload", "bad");
  console.error(e.message);
};

worker.onmessage = ({ data: msg }) => {
  switch (msg.type) {
    case "status":
      setEngine(msg.text);
      break;
    case "ready":
      setEngine(`Ready · leakcheck ${msg.versions.leakcheck}`, "ready");
      $("#versions").textContent = `leakcheck ${msg.versions.leakcheck} · pandas ${msg.versions.pandas} · Pyodide`;
      break;
    case "fatal":
      setEngine("Failed to start Python", "bad");
      console.error(msg.error);
      break;
    case "loaded": {
      loaded[msg.name] = msg;
      const drop = $(`#drop-${msg.name}`);
      drop.classList.add("done");
      drop.querySelector(".drop-hint").innerHTML =
        `${escapeHtml(msg.filename)}<br><small>${msg.rows.toLocaleString()} rows × ${msg.cols} columns</small>`;
      if (loaded.train && loaded.test) worker.postMessage({ type: "suggest" });
      break;
    }
    case "suggested":
      onSuggested(msg);
      break;
    case "result":
      $("#run").disabled = false;
      $("#run").textContent = "Check for leakage";
      renderResults(msg);
      break;
    case "error":
      if (msg.request === "load") {
        const drop = $(`#drop-${msg.name}`);
        drop.classList.add("error");
        drop.querySelector(".drop-hint").textContent = `Couldn't read this file: ${msg.error}`;
      } else {
        $("#run").disabled = false;
        $("#run").textContent = "Check for leakage";
        const el = $("#results");
        el.innerHTML = `<div class="error-box">Something went wrong: ${escapeHtml(msg.error)}</div>`;
        el.hidden = false;
      }
      break;
  }
};
