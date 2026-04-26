document.addEventListener("DOMContentLoaded", () => {
  void initStudio().catch((error) => showRunError(error));
});

async function initStudio() {
  const shared = window.MedusaShared;
  const { tasks, agents } = await shared.loadCatalog();
  shared.renderTopbarMeta(tasks, agents);
  bindControls(tasks, agents);
  renderTaskSelect(tasks);
  renderAgentSelect(
    agents.filter(
      (agent) =>
        agent.id === "random" ||
        agent.id === "heuristic" ||
        agent.id === "grpo_trained"
    )
  );
  await refreshPreview(tasks, agents);
}

function bindControls(tasks, agents) {
  const shared = window.MedusaShared;

  document.getElementById("task-select").addEventListener("change", async (event) => {
    shared.setTask(event.target.value);
    await refreshSelectedTask(tasks, agents).catch((error) => showRunError(error));
  });

  document.getElementById("agent-select").addEventListener("change", async (event) => {
    shared.setAgent(event.target.value);
    await refreshSelectedTask(tasks, agents).catch((error) => showRunError(error));
  });

  document.getElementById("rerun-agent").addEventListener("click", async () => {
    await runSelectedAgent(tasks, agents).catch((error) => showRunError(error));
  });

  document.getElementById("clean-df").addEventListener("click", async () => {
    await cleanUploadedDataframe().catch((error) => showRunError(error));
  });
}

async function refreshSelectedTask(tasks, agents) {
  await refreshPreview(tasks, agents);
}

function renderTaskSelect(tasks) {
  const shared = window.MedusaShared;
  const state = shared.getState();
  const select = document.getElementById("task-select");
  select.innerHTML = tasks
    .map(
      (task) => `
        <option value="${task.id}" ${task.id === state.taskId ? "selected" : ""}>
          ${task.name} · ${task.difficulty} · seed ${task.seed}
        </option>
      `
    )
    .join("");
}

function renderAgentSelect(agents) {
  const shared = window.MedusaShared;
  const state = shared.getState();
  const select = document.getElementById("agent-select");
  select.innerHTML = agents
    .map(
      (agent) => `
        <option value="${agent.id}" ${agent.id === state.agentId ? "selected" : ""}>
          ${agent.name} · ${agent.family}
        </option>
      `
    )
    .join("");
}

async function runSelectedAgent(tasks, agents) {
  const shared = window.MedusaShared;
  const state = shared.getState();
  const runButton = document.getElementById("rerun-agent");

  if (!state.taskId || !state.agentId) {
    throw new Error("Select both a task and an agent before running.");
  }

  shared.resetTrace();
  runButton.disabled = true;
  runButton.textContent = "Running...";
  document.getElementById("hero-status").innerHTML = `<span class="status-pill">Running selected agent...</span>`;
  document.getElementById("trace-list").innerHTML = `<li class="empty">Executing replay...</li>`;

  try {
    const preview = await shared.fetchJSON(`/api/run/autorun/${encodeURIComponent(state.taskId)}`, shared.agentPayload(), "POST");
    shared.setTrace(preview.actions);
    window.__medusaPreview = preview;
    await renderPreview(tasks, agents);
  } finally {
    runButton.disabled = false;
    runButton.textContent = "Run Selected Agent";
  }
}

function showRunError(error) {
  const message = error && error.message ? error.message : `${error}`;
  document.getElementById("hero-status").innerHTML = `<span class="status-pill is-bad">Run failed</span>`;
  document.getElementById("trace-list").innerHTML = `<li class="empty">${message}</li>`;
}

async function refreshPreview(tasks, agents) {
  const shared = window.MedusaShared;
  const preview = await shared.fetchJSON("/api/run/preview", shared.basePayload(), "POST");
  window.__medusaPreview = preview;
  await renderPreview(tasks, agents);
}

async function renderPreview(tasks = null, agents = null) {
  const shared = window.MedusaShared;
  const preview = window.__medusaPreview;
  if (!preview) {
    return;
  }

  const catalogPayload = await shared.loadCatalog();
  const catalog = tasks || catalogPayload.tasks;
  const agentCatalog = agents || catalogPayload.agents;
  const task = preview.task;
  const summary = preview.summary;
  const agent = preview.agent || shared.getCurrentAgent(agentCatalog);

  shared.renderTopbarMeta(catalog, agentCatalog);

  document.getElementById("hero-status").innerHTML = `
    <span class="status-pill ${summary.done ? "is-good" : ""}">
      ${task.name} · ${agent ? agent.name : "Agent run"} · ${summary.done ? "episode closed" : "trace live"}
    </span>
    <span class="status-pill">${preview.action_count} replayed step${preview.action_count === 1 ? "" : "s"}</span>
  `;

  document.getElementById("trace-count").textContent = `${preview.action_count} step${preview.action_count === 1 ? "" : "s"}`;
  document.getElementById("trace-list").innerHTML =
    preview.actions.length === 0
      ? `<li class="empty">No actions yet. Select an agent to run the task.</li>`
      : preview.actions
          .map((action, index) => `<li><strong>${index + 1}.</strong> ${action.action}</li>`)
          .join("");
}

async function cleanUploadedDataframe() {
  const shared = window.MedusaShared;
  const state = shared.getState();
  const fileInput = document.getElementById("df-upload");
  const file = fileInput.files && fileInput.files[0];
  if (!file) throw new Error("Please upload a CSV file first.");
  if (!state.agentId) throw new Error("Select an agent before cleaning.");

  const form = new FormData();
  form.append("agent_id", state.agentId);
  form.append("file", file);
  document.getElementById("clean-status").innerHTML = "<span class='status-pill'>Cleaning…</span>";
  document.getElementById("clean-trace-count").textContent = "0 steps";
  document.getElementById("clean-trace-list").innerHTML = `<li class="empty">Running cleaner…</li>`;
  document.getElementById("dq-section").style.display = "none";

  const response = await fetch("/api/run/clean-dataframe", { method: "POST", body: form });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail ? JSON.stringify(data.detail) : response.statusText);
  }

  const blob = new Blob([data.cleaned_csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const download = document.getElementById("download-cleaned");
  download.href = url;
  download.download = data.output_filename;
  download.style.display = "inline-flex";

  document.getElementById("clean-status").innerHTML = `
    <span class="status-pill is-good">Cleaned · ${escapeHTML(state.agentId)}</span>
    <span class="status-pill">${data.input_rows} → ${data.output_rows} rows</span>
  `;
  const trace = Array.isArray(data.action_trace) ? data.action_trace : [];
  document.getElementById("clean-trace-count").textContent = `${trace.length} step${trace.length === 1 ? "" : "s"}`;
  document.getElementById("clean-trace-list").innerHTML =
    trace.length === 0
      ? `<li class="empty">No cleaning trace emitted.</li>`
      : trace.map((item, i) => {
          const agentLabel = item.agent_id || state.agentId || "unknown";
          const actionLabel = item.action || "No action description";
          return `<li><strong>${i + 1}.</strong> ${escapeHTML(actionLabel)} <span class="tag">agent: ${escapeHTML(agentLabel)}</span></li>`;
        }).join("");

  // Score source vs cleaned and render DQ report right here on the page
  await showDqReport(file, data.cleaned_csv, data.output_filename);
}

async function showDqReport(sourceFile, cleanedCsv, cleanedFilename) {
  const section = document.getElementById("dq-section");
  const statusEl = document.getElementById("dq-status-studio");
  const gridEl = document.getElementById("dq-grid-studio");

  section.style.display = "";
  statusEl.innerHTML = `<span class="status-pill">Scoring…</span>`;
  gridEl.innerHTML = "";

  try {
    const scoreForm = new FormData();
    scoreForm.append("source", sourceFile, sourceFile.name);
    scoreForm.append("cleaned", new Blob([cleanedCsv], { type: "text/csv" }), cleanedFilename);

    const res = await fetch("/api/run/score-dataframes", { method: "POST", body: scoreForm });
    const scored = await res.json();
    if (!res.ok) throw new Error(scored.detail ? JSON.stringify(scored.detail) : res.statusText);

    statusEl.innerHTML = `
      <span class="status-pill is-good">DQ Scored</span>
      <span class="status-pill">${escapeHTML(sourceFile.name)}</span>
    `;
    renderDqGrid(scored.source, scored.cleaned, sourceFile.name, cleanedFilename, gridEl);
  } catch (err) {
    statusEl.innerHTML = `<span class="status-pill is-bad">DQ scoring failed: ${escapeHTML(String(err.message || err))}</span>`;
  }
}

function renderDqGrid(src, cln, sourceName, cleanedName, container) {
  const hasClean = cln != null;

  const METRICS = [
    { key: "score",                  label: "Overall Score",         fmt: pct,  lowerIsBetter: false },
    { key: "rows",                   label: "Total Rows",            fmt: num,  lowerIsBetter: null },
    { key: "columns",                label: "Total Columns",         fmt: num,  lowerIsBetter: null },
    { key: "missing_cells",          label: "NULL / Missing Values", fmt: num,  lowerIsBetter: true },
    { key: "null_values",            label: "Null Values (numeric)", fmt: num,  lowerIsBetter: true },
    { key: "nan_values",             label: "NaN Values (numeric)",  fmt: num,  lowerIsBetter: true },
    { key: "duplicate_rows",         label: "Duplicate Rows",        fmt: num,  lowerIsBetter: true },
    { key: "duplicate_column_names", label: "Duplicate Columns",     fmt: num,  lowerIsBetter: true },
    { key: "dirty_string_cells",     label: "Dirty String Cells",    fmt: num,  lowerIsBetter: true },
    { key: "bad_numeric_cells",      label: "Bad Numeric Cells",     fmt: num,  lowerIsBetter: true },
  ];

  const COMP = {
    readability: "Readability", completeness: "Completeness", uniqueness: "Uniqueness",
    type_consistency: "Type Consistency", date_format_sanity: "Date Format Sanity",
    column_quality: "Column Quality", string_cleanliness: "String Cleanliness",
    numeric_sanity: "Numeric Sanity",
  };

  function num(v) { return v == null ? "—" : Number(v).toLocaleString(); }
  function pct(v) { return v == null ? "—" : `${(Number(v) * 100).toFixed(1)}%`; }

  function tone(lowerIsBetter, sv, cv) {
    if (!hasClean || lowerIsBetter === null || sv == null || cv == null || cv === sv) return "";
    return (lowerIsBetter ? cv < sv : cv > sv) ? "dq-better" : "dq-worse";
  }

  function arrow(lowerIsBetter, sv, cv) {
    if (!hasClean || lowerIsBetter === null || sv == null || cv == null) return "";
    if (cv === sv) return `<span class="dq-arrow dq-arrow--same">→</span>`;
    return (lowerIsBetter ? cv < sv : cv > sv)
      ? `<span class="dq-arrow dq-arrow--better">▲</span>`
      : `<span class="dq-arrow dq-arrow--worse">▼</span>`;
  }

  const statRows = METRICS.map(({ key, label, fmt, lowerIsBetter }) => {
    const sv = src[key], cv = hasClean ? cln[key] : null;
    return `<div class="dq-stat-row">
      <div class="dq-stat-label">${escapeHTML(label)}</div>
      <div class="dq-stat-val">${escapeHTML(fmt(sv))}</div>
      ${hasClean ? `<div class="dq-stat-val ${tone(lowerIsBetter, sv, cv)}">${escapeHTML(fmt(cv))} ${arrow(lowerIsBetter, sv, cv)}</div>` : ""}
    </div>`;
  }).join("");

  const compRows = Object.entries(COMP).map(([key, label]) => {
    const sv = src.component_scores?.[key] ?? null;
    const cv = hasClean ? (cln.component_scores?.[key] ?? null) : null;
    const t = (hasClean && sv != null && cv != null) ? (cv > sv ? "dq-better" : cv < sv ? "dq-worse" : "") : "";
    const a = (hasClean && sv != null && cv != null)
      ? (cv > sv ? `<span class="dq-arrow dq-arrow--better">▲</span>`
          : cv < sv ? `<span class="dq-arrow dq-arrow--worse">▼</span>`
          : `<span class="dq-arrow dq-arrow--same">→</span>`)
      : "";
    return `<div class="dq-stat-row dq-stat-row--component">
      <div class="dq-stat-label">${escapeHTML(label)}</div>
      <div class="dq-stat-val">${sv != null ? pct(sv) : "—"}</div>
      ${hasClean ? `<div class="dq-stat-val ${t}">${cv != null ? pct(cv) : "—"} ${a}</div>` : ""}
    </div>`;
  }).join("");

  container.innerHTML = `
    <div class="dq-comparison-grid">
      <div class="dq-header-row">
        <div class="dq-col-metric"></div>
        <div class="dq-col-head">${escapeHTML(sourceName || "Source")}</div>
        ${hasClean ? `<div class="dq-col-head">${escapeHTML(cleanedName || "Cleaned")}</div>` : ""}
      </div>
      ${statRows}
      <div class="dq-section-divider">Component Scores</div>
      ${compRows}
    </div>`;
}

function escapeHTML(value) {
  return `${value ?? ""}`.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[c])
  );
}
