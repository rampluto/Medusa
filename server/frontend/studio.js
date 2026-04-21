document.addEventListener("DOMContentLoaded", () => {
  void initStudio();
});

async function initStudio() {
  const shared = window.MedusaShared;
  const { tasks, agents } = await shared.loadCatalog();
  shared.renderTopbarMeta(tasks, agents);
  bindControls(tasks, agents);
  renderTaskSelect(tasks);
  renderAgentSelect(agents);
  renderTableSelect();
  await runSelectedAgent(tasks, agents);
}

function bindControls(tasks, agents) {
  const shared = window.MedusaShared;

  document.getElementById("task-select").addEventListener("change", async (event) => {
    shared.setTask(event.target.value);
    await runSelectedAgent(tasks, agents);
  });

  document.getElementById("agent-select").addEventListener("change", async (event) => {
    shared.setAgent(event.target.value);
    await runSelectedAgent(tasks, agents);
  });

  document.getElementById("rerun-agent").addEventListener("click", async () => {
    await runSelectedAgent(tasks, agents);
  });

  document.getElementById("table-select").addEventListener("change", async (event) => {
    shared.setTable(event.target.value);
    await refreshTable();
  });

  document.getElementById("page-prev").addEventListener("click", async () => {
    const state = shared.getState();
    shared.setTablePage(Math.max(1, state.tablePage - 1));
    await refreshTable();
  });

  document.getElementById("page-next").addEventListener("click", async () => {
    const state = shared.getState();
    shared.setTablePage(state.tablePage + 1);
    await refreshTable();
  });
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

function renderTableSelect() {
  const shared = window.MedusaShared;
  const state = shared.getState();
  const tables = [
    "bronze_a",
    "bronze_a_prepped",
    "bronze_b",
    "bronze_b_prepped",
    "joined",
    "silver",
    "quarantine",
  ];
  const select = document.getElementById("table-select");
  select.innerHTML = tables
    .map(
      (table) => `
        <option value="${table}" ${table === state.selectedTable ? "selected" : ""}>
          ${table}
        </option>
      `
    )
    .join("");
}

async function runSelectedAgent(tasks, agents) {
  const shared = window.MedusaShared;
  const state = shared.getState();
  shared.resetTrace();
  document.getElementById("hero-status").innerHTML = `<span class="status-pill">Running selected agent...</span>`;
  document.getElementById("trace-list").innerHTML = `<li class="empty">Executing replay...</li>`;

  const preview = await shared.fetchJSON(`/api/run/autorun/${encodeURIComponent(state.taskId)}`, shared.agentPayload(), "POST");
  shared.setTrace(preview.actions);
  window.__medusaPreview = preview;
  await renderPreview(tasks, agents);
  await Promise.all([refreshAnalysis(), refreshTimeline(), refreshTable()]);
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
  const observation = preview.observation;
  const agent = preview.agent || shared.getCurrentAgent(agentCatalog);

  shared.renderTopbarMeta(catalog, agentCatalog);

  document.getElementById("hero-status").innerHTML = `
    <span class="status-pill ${summary.done ? "is-good" : ""}">
      ${task.name} · ${agent ? agent.name : "Agent run"} · ${summary.stage} · ${summary.done ? "episode closed" : "trace live"}
    </span>
    <span class="status-pill">${preview.action_count} replayed step${preview.action_count === 1 ? "" : "s"}</span>
  `;

  document.getElementById("task-meta").innerHTML = `
    <div class="stack stack--dense">
      <div class="info-pair"><span>Difficulty</span><strong class="${shared.difficultyTone(task.difficulty)}">${task.difficulty}</strong></div>
      <div class="info-pair"><span>Agent</span><strong>${agent ? agent.name : "No agent selected"}</strong></div>
      <div class="info-pair"><span>Seed</span><strong>${preview.seed}</strong></div>
      <div class="info-pair"><span>Scenario</span><strong>${preview.scenario.id || "Unknown"}</strong></div>
      <div class="info-pair"><span>Join key</span><strong>${preview.scenario.join_key || "n/a"}</strong></div>
      <div class="info-pair"><span>Description</span><strong>${task.description}</strong></div>
    </div>
    <div class="stack">
      ${task.success_criteria.map((criterion) => `<span class="tag">${criterion}</span>`).join("")}
    </div>
    <div class="action-row">
      <a class="button button--ghost" href="/medusa/audit">Open Audit Report</a>
    </div>
  `;

  document.getElementById("trace-count").textContent = `${preview.action_count} step${preview.action_count === 1 ? "" : "s"}`;
  document.getElementById("trace-list").innerHTML =
    preview.actions.length === 0
      ? `<li class="empty">No actions yet. Select an agent to run the task.</li>`
      : preview.actions
          .map((action, index) => `<li><strong>${index + 1}.</strong> ${action.action}</li>`)
          .join("");

  document.getElementById("observation-message").textContent = observation.message || "No observation yet.";

  document.getElementById("summary-cards").innerHTML = [
    shared.metricCard("Agent", agent ? agent.name : "n/a"),
    shared.metricCard("Stage", summary.stage),
    shared.metricCard("Reward", shared.formatNumber(summary.cumulative_reward)),
    shared.metricCard("Match Rate", shared.formatPercent(summary.match_rate)),
    shared.metricCard("Silver Rows", summary.silver_row_count),
    shared.metricCard("Quarantine", summary.quarantine_row_count),
    shared.metricCard("Join", summary.join_type || "pending"),
    shared.metricCard("Agent Steps", preview.auto_run ? preview.auto_run.agent_steps : preview.action_count),
  ].join("");

  document.getElementById("agent-brief").innerHTML = agent
    ? `
        <div class="info-pair"><span>Selected agent</span><strong>${agent.name}</strong></div>
        <div>${agent.description}</div>
        <div class="stack stack--inline">${agent.strengths.map((item) => `<span class="tag">${item}</span>`).join("")}</div>
      `
    : `<div>No agent selected.</div>`;
}

async function refreshAnalysis() {
  const shared = window.MedusaShared;
  const data = await shared.fetchJSON("/api/run/analysis", shared.basePayload(), "POST");
  const commit = data.analysis.commit;

  document.getElementById("analysis-commit").innerHTML = `
    <span class="metric-badge ${commit.ready ? "is-good" : "is-bad"}">
      ${commit.ready ? "Ready to commit" : "More work needed"}
    </span>
    ${
      commit.blockers.length
        ? commit.blockers.map((item) => `<div>${item}</div>`).join("")
        : `<div>No obvious blockers from the current trace.</div>`
    }
    <div><strong>Suggested next moves:</strong> ${commit.suggested_actions.join(", ")}</div>
  `;
}

async function refreshTimeline() {
  const shared = window.MedusaShared;
  const data = await shared.fetchJSON("/api/run/timeline", shared.basePayload(), "POST");
  const timeline = data.timeline;

  document.getElementById("timeline-list").innerHTML =
    timeline.length === 0
      ? `<div class="timeline__item empty">The governance log starts after the first action.</div>`
      : timeline
          .map(
            (entry) => `
              <article class="timeline__item">
                <div class="timeline__header">
                  <span>#${entry.step} · ${entry.action}</span>
                  <span class="${entry.reward >= 0 ? "is-good" : "is-bad"}">${shared.formatNumber(entry.reward)}</span>
                </div>
                <div class="timeline__meta">Cumulative reward: ${shared.formatNumber(entry.cumulative_reward)}</div>
                <div class="timeline__metrics">
                  ${Object.entries(entry.metrics || {})
                    .slice(0, 6)
                    .map(([key, value]) => `<span class="tag">${key}: ${shared.formatValue(value)}</span>`)
                    .join("")}
                </div>
              </article>
            `
          )
          .join("");
}

async function refreshTable() {
  const shared = window.MedusaShared;
  const state = shared.getState();
  const payload = {
    ...shared.basePayload(),
    table: state.selectedTable,
    page: state.tablePage,
    page_size: state.tablePageSize,
  };
  const data = await shared.fetchJSON("/api/run/tables", payload, "POST");
  shared.setTablePage(data.page);
  document.getElementById("table-pagination").textContent = `Page ${data.page} of ${data.total_pages} · ${data.total_rows} rows`;

  const table = document.getElementById("data-table");
  if (data.columns.length === 0) {
    table.innerHTML = `<tr><td class="empty">No columns yet for ${state.selectedTable}.</td></tr>`;
    return;
  }

  const head = `
    <thead>
      <tr>${data.columns.map((column) => `<th>${column}</th>`).join("")}</tr>
    </thead>
  `;
  const body = `
    <tbody>
      ${
        data.rows.length === 0
          ? `<tr><td colspan="${data.columns.length}" class="empty">No rows in this table yet.</td></tr>`
          : data.rows
              .map(
                (row) => `
                  <tr>
                    ${data.columns.map((column) => `<td>${shared.formatValue(row[column])}</td>`).join("")}
                  </tr>
                `
              )
              .join("")
      }
    </tbody>
  `;
  table.innerHTML = head + body;
}
