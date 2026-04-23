document.addEventListener("DOMContentLoaded", () => {
  void initAuditPage();
});

async function initAuditPage() {
  const shared = window.MedusaShared;
  const { tasks, agents } = await shared.loadCatalog();
  shared.renderTopbarMeta(tasks, agents);

  const payload = shared.basePayload();
  const [preview, graderData, evaluationData, featureData] = await Promise.all([
    shared.fetchJSON("/api/run/preview", payload, "POST"),
    shared.fetchJSON("/api/run/grader", payload, "POST"),
    shared.fetchJSON("/api/run/evaluate", payload, "POST"),
    shared.fetchJSON("/api/run/feature-vector", payload, "POST"),
  ]);

  const evaluation = graderData.evaluation || evaluationData.evaluation;
  renderStatus(tasks, agents, preview, graderData);
  renderSummary(tasks, agents, preview, graderData, evaluation);
  renderGrader(graderData);
  renderEvaluation(evaluation);
  renderTrace(tasks, agents, preview);
  renderFeatures(featureData.features);
}

function renderStatus(tasks, agents, preview, graderData) {
  const shared = window.MedusaShared;
  const task = preview.task || shared.getCurrentTask(tasks);
  const agent = shared.getCurrentAgent(agents);
  const tone = graderData.ready_for_commit ? "is-good" : "tone-warn";

  document.getElementById("audit-status").innerHTML = `
    <span class="status-pill ${tone}">${escapeHTML(task.name)} · ${escapeHTML(preview.summary.stage)}</span>
    <span class="status-pill">${escapeHTML(agent ? agent.name : "n/a")}</span>
    <span class="status-pill">${preview.action_count} step${preview.action_count === 1 ? "" : "s"}</span>
  `;
}

function renderSummary(tasks, agents, preview, graderData, evaluation) {
  const shared = window.MedusaShared;
  const task = preview.task || shared.getCurrentTask(tasks);
  const agent = shared.getCurrentAgent(agents);
  const score = typeof evaluation?.score === "number" ? evaluation.score : null;

  const metrics = [
    ["Task", task.name],
    ["Agent", agent ? agent.name : "n/a"],
    ["Difficulty", task.difficulty],
    ["Stage", preview.summary.stage],
    ["Steps", preview.action_count],
    ["Reward", shared.formatNumber(preview.summary.cumulative_reward)],
    ["Score", score === null ? "n/a" : shared.formatNumber(score)],
    ["Grade", evaluation?.grade || "n/a"],
    ["Ready", graderData.ready_for_commit ? "yes" : "no"],
    ["Silver Rows", preview.summary.silver_row_count],
    ["Quarantine", preview.summary.quarantine_row_count],
    ["Seed", preview.seed],
  ];

  const blockers = graderData.blockers.length
    ? graderData.blockers
    : ["No active blockers."];

  document.getElementById("audit-summary").innerHTML = `
    <div class="audit-metric-grid">
      ${metrics.map(([label, value]) => metricTile(label, value)).join("")}
    </div>
    <div class="audit-blocker-grid">
      ${blockers.map((blocker) => blockerTile(blocker, graderData.blockers.length === 0)).join("")}
    </div>
  `;
}

function renderGrader(graderData) {
  const lines = graderData.grader.lines.length
    ? graderData.grader.lines
    : ["No grader report yet. Commit the run to trigger the deterministic audit."];

  document.getElementById("grader-report").innerHTML = `
    <article class="audit-status-card ${graderData.grader.passed ? "is-good" : "tone-warn"}">
      <span>Grader</span>
      <strong>${graderData.grader.passed ? "Passed" : "Not passed yet"}</strong>
      <p>${graderData.ready_for_commit ? "Ready for commit." : "Resolve blockers before commit."}</p>
    </article>
    <div class="audit-line-grid">
      ${lines.map((line) => `<article class="audit-line ${lineTone(line)}">${escapeHTML(line)}</article>`).join("")}
    </div>
  `;
}

function renderEvaluation(evaluation) {
  const shared = window.MedusaShared;
  if (!evaluation) {
    document.getElementById("evaluation-breakdown").innerHTML = `<div class="empty">Evaluation is not available yet.</div>`;
    return;
  }

  const breakdown = Object.entries(evaluation.breakdown || {});
  document.getElementById("evaluation-breakdown").innerHTML = `
    <article class="audit-status-card ${evaluation.passed ? "is-good" : "tone-warn"}">
      <span>Overall</span>
      <strong>${shared.formatNumber(evaluation.score)} · ${escapeHTML(evaluation.grade)}</strong>
      <p>${evaluation.passed ? "Passing rubric result." : "Below pass threshold."}</p>
    </article>
    ${breakdown
      .map(
        ([key, value]) => `
          <article class="rubric-tile">
            <span>${escapeHTML(shared.titleize(key))}</span>
            <strong>${shared.formatNumber(value)}</strong>
            <div class="feature__bar"><span style="width:${shared.clampBar(value)}%"></span></div>
          </article>
        `
      )
      .join("")}
    ${
      evaluation.notes?.length
        ? `<article class="rubric-note"><span>Notes</span><strong>${evaluation.notes.map(escapeHTML).join(" | ")}</strong></article>`
        : ""
    }
  `;
}

function renderTrace(tasks, agents, preview) {
  const shared = window.MedusaShared;
  const state = shared.getState();
  const task = preview.task || shared.getCurrentTask(tasks);
  const agent = shared.getCurrentAgent(agents);

  document.getElementById("audit-trace").innerHTML = `
    <div class="audit-trace-meta">
      ${traceTile("Task", task.name)}
      ${traceTile("Agent", agent ? agent.name : "n/a")}
      ${traceTile("Stage", preview.summary.stage)}
      ${traceTile("Observation", preview.observation.message)}
    </div>
    <ol class="audit-action-list">
      ${
        state.actions.length
          ? state.actions
              .map(
                (action, index) => `
                  <li>
                    <span>${String(index + 1).padStart(2, "0")}</span>
                    <strong>${escapeHTML(action.action)}</strong>
                  </li>
                `
              )
              .join("")
          : `<li><span>00</span><strong>No actions recorded yet.</strong></li>`
      }
    </ol>
    <div class="action-row">
      <a class="button" href="/medusa/studio">Back To Studio</a>
      <a class="button button--ghost" href="/medusa/tasks">Browse Tasks</a>
    </div>
  `;
}

function renderFeatures(features) {
  const shared = window.MedusaShared;
  document.getElementById("feature-grid").innerHTML = features
    .map(
      (feature) => `
        <article class="feature">
          <div class="metric__label">${escapeHTML(feature.label)}</div>
          <div class="feature__value">${shared.formatNumber(feature.value)}</div>
          <div class="feature__bar"><span style="width:${shared.clampBar(feature.value)}%"></span></div>
          <div>${escapeHTML(feature.description)}</div>
        </article>
      `
    )
    .join("");
}

function metricTile(label, value) {
  return `
    <article class="metric metric--audit">
      <span class="metric__label">${escapeHTML(label)}</span>
      <span class="metric__value">${escapeHTML(`${value}`)}</span>
    </article>
  `;
}

function blockerTile(blocker, clear) {
  return `
    <article class="audit-blocker ${clear ? "audit-blocker--clear" : ""}">
      <span>${clear ? "Clear" : "Blocker"}</span>
      <strong>${escapeHTML(blocker)}</strong>
    </article>
  `;
}

function traceTile(label, value) {
  return `
    <article class="trace-tile">
      <span>${escapeHTML(label)}</span>
      <strong>${escapeHTML(`${value}`)}</strong>
    </article>
  `;
}

function lineTone(line) {
  const normalized = line.toLowerCase();
  if (normalized.includes("fail") || normalized.includes("crash") || normalized.includes("null_fail")) {
    return "is-bad";
  }
  if (normalized.includes("pass") || normalized.includes("ok")) {
    return "is-good";
  }
  return "";
}

function escapeHTML(value) {
  return `${value ?? ""}`.replace(/[&<>"']/g, (char) => {
    const replacements = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;",
    };
    return replacements[char];
  });
}
