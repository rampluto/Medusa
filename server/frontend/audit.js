document.addEventListener("DOMContentLoaded", () => {
  void initAuditPage();
});

async function initAuditPage() {
  const shared = window.MedusaShared;
  const { tasks, agents } = await shared.loadCatalog();
  shared.renderTopbarMeta(tasks, agents);
  await Promise.all([renderSummary(tasks, agents), renderGrader(), renderFeatures()]);
}

async function renderSummary(tasks, agents) {
  const shared = window.MedusaShared;
  const preview = await shared.fetchJSON("/api/run/preview", shared.basePayload(), "POST");
  const state = shared.getState();
  const task = preview.task || shared.getCurrentTask(tasks);
  const agent = shared.getCurrentAgent(agents);

  document.getElementById("audit-status").innerHTML = `
    <span class="status-pill ${preview.summary.done ? "is-good" : ""}">
      ${task.name} · ${preview.summary.done ? "committed or failed" : "still running"}
    </span>
    <span class="status-pill">${preview.action_count} replayed step${preview.action_count === 1 ? "" : "s"}</span>
  `;

  document.getElementById("audit-summary").innerHTML = [
    shared.metricCard("Task", task.name),
    shared.metricCard("Agent", agent ? agent.name : "n/a"),
    shared.metricCard("Difficulty", task.difficulty),
    shared.metricCard("Steps", preview.action_count),
    shared.metricCard("Reward", shared.formatNumber(preview.summary.cumulative_reward)),
  ].join("");

  document.getElementById("audit-trace").innerHTML = `
    <div class="info-pair"><span>Agent</span><strong>${agent ? agent.name : "n/a"}</strong></div>
    <div class="info-pair"><span>Description</span><strong>${task.description}</strong></div>
    <div class="info-pair"><span>Seed</span><strong>${preview.seed}</strong></div>
    <div class="info-pair"><span>Current stage</span><strong>${preview.summary.stage}</strong></div>
    <div class="info-pair"><span>Observation</span><strong>${preview.observation.message}</strong></div>
    <div class="info-pair"><span>Trace</span><strong>${
      state.actions.length
        ? state.actions.map((action, index) => `${index + 1}. ${action.action}`).join(" | ")
        : "No actions recorded yet."
    }</strong></div>
    <div class="action-row">
      <a class="button" href="/medusa/studio">Back To Studio</a>
      <a class="button button--ghost" href="/medusa/tasks">Browse Tasks</a>
    </div>
  `;
}

async function renderGrader() {
  const shared = window.MedusaShared;
  const [graderData, evaluationData] = await Promise.all([
    shared.fetchJSON("/api/run/grader", shared.basePayload(), "POST"),
    shared.fetchJSON("/api/run/evaluate", shared.basePayload(), "POST"),
  ]);

  const evaluation = graderData.evaluation || evaluationData.evaluation;
  document.getElementById("grader-report").innerHTML = `
    <span class="metric-badge ${graderData.grader.passed ? "is-good" : "is-bad"}">
      ${graderData.committed ? (graderData.grader.passed ? "Grader passed" : "Grader failed") : "Pre-commit audit"}
    </span>
    ${
      graderData.grader.lines.length
        ? graderData.grader.lines.map((line) => `<div>${line}</div>`).join("")
        : `<div>No grader report yet. Commit the run to trigger the deterministic audit.</div>`
    }
    ${
      graderData.blockers.length
        ? `<div><strong>Current blockers:</strong> ${graderData.blockers.join(" | ")}</div>`
        : ""
    }
  `;

  document.getElementById("evaluation-breakdown").innerHTML = evaluation
    ? `
        <div class="info-pair"><span>Score</span><strong>${evaluation.score}</strong></div>
        <div class="info-pair"><span>Grade</span><strong>${evaluation.grade}</strong></div>
        <div class="info-pair"><span>Passed</span><strong>${evaluation.passed ? "yes" : "no"}</strong></div>
        ${Object.entries(evaluation.breakdown || {})
          .map(
            ([key, value]) => `
              <div class="rubric-row">
                <span>${shared.titleize(key)}</span>
                <strong>${value}</strong>
              </div>
            `
          )
          .join("")}
        ${
          evaluation.notes?.length
            ? `<div class="info-pair"><span>Notes</span><strong>${evaluation.notes.join(" | ")}</strong></div>`
            : ""
        }
      `
    : `<div class="empty">Evaluation is not available yet.</div>`;
}

async function renderFeatures() {
  const shared = window.MedusaShared;
  const data = await shared.fetchJSON("/api/run/feature-vector", shared.basePayload(), "POST");
  document.getElementById("feature-grid").innerHTML = data.features
    .map(
      (feature) => `
        <article class="feature">
          <div class="metric__label">${feature.label}</div>
          <div class="feature__value">${shared.formatNumber(feature.value)}</div>
          <div class="feature__bar"><span style="width:${shared.clampBar(feature.value)}%"></span></div>
          <div>${feature.description}</div>
        </article>
      `
    )
    .join("");
}
