document.addEventListener("DOMContentLoaded", () => {
  void initTaskCatalog();
});

async function initTaskCatalog() {
  const shared = window.MedusaShared;
  const { tasks, agents } = await shared.loadCatalog();
  shared.renderTopbarMeta(tasks, agents);
  renderCurrentTrace(tasks);
  renderCatalogInsights(tasks, agents);
  renderTaskCards(tasks);
}

function renderCurrentTrace(tasks) {
  const shared = window.MedusaShared;
  const state = shared.getState();
  const task = shared.getCurrentTask(tasks);
  const host = document.getElementById("catalog-current");
  const status = document.getElementById("catalog-status");

  if (!task) {
    host.innerHTML = `<div class="empty">No task selected yet.</div>`;
    return;
  }

  status.innerHTML = `
    <span class="status-pill">${task.name} is ready</span>
    <span class="status-pill">${tasks.length} benchmark tasks</span>
  `;
  host.innerHTML = `
    <div class="info-pair"><span>Task</span><strong>${task.name}</strong></div>
    <div class="info-pair"><span>Difficulty</span><strong class="${shared.difficultyTone(task.difficulty)}">${task.difficulty}</strong></div>
    <div class="info-pair"><span>Seed</span><strong>${task.seed}</strong></div>
    <div class="info-pair"><span>Stored trace length</span><strong>${state.actions.length} step${state.actions.length === 1 ? "" : "s"}</strong></div>
    <div class="action-row">
      <a class="button" href="/medusa/studio">Open Studio</a>
      <a class="button button--ghost" href="/medusa/audit">Open Audit</a>
    </div>
  `;
}

function renderTaskCards(tasks) {
  const shared = window.MedusaShared;
  const currentState = shared.getState();
  const host = document.getElementById("task-cards");

  host.innerHTML = tasks
    .map(
      (task) => `
        <article class="task-card ${currentState.taskId === task.id ? "task-card--active" : ""}">
          <div class="task-card__top">
            <span class="metric-badge ${shared.difficultyTone(task.difficulty)}">${task.difficulty}</span>
            <span class="tag">seed ${task.seed}</span>
          </div>
          <h3>${task.name}</h3>
          <p>${task.description}</p>
          <div class="stack">
            ${task.success_criteria.map((criterion) => `<span class="tag">${criterion}</span>`).join("")}
          </div>
          <div class="task-card__rubric">
            ${Object.entries(task.scoring_rubric)
              .map(
                ([key, value]) => `
                  <div class="rubric-row">
                    <span>${shared.titleize(key)}</span>
                    <strong>${Math.round(value * 100)}%</strong>
                  </div>
                `
              )
              .join("")}
          </div>
          <div class="action-row">
            <button class="button" data-task="${task.id}">Use In Studio</button>
            <button class="button button--ghost" data-audit-task="${task.id}">Go To Audit</button>
          </div>
        </article>
      `
    )
    .join("");

  host.querySelectorAll("[data-task]").forEach((button) => {
    button.addEventListener("click", () => {
      shared.setTask(button.dataset.task);
      window.location.href = "/medusa/studio";
    });
  });

  host.querySelectorAll("[data-audit-task]").forEach((button) => {
    button.addEventListener("click", () => {
      shared.setTask(button.dataset.auditTask);
      window.location.href = "/medusa/audit";
    });
  });
}

function renderCatalogInsights(tasks, agents) {
  const shared = window.MedusaShared;
  const currentTask = shared.getCurrentTask(tasks);
  const currentAgent = shared.getCurrentAgent(agents);
  const totals = tasks.reduce(
    (accumulator, task) => {
      accumulator[task.difficulty] = (accumulator[task.difficulty] || 0) + 1;
      return accumulator;
    },
    { easy: 0, medium: 0, hard: 0 }
  );

  document.getElementById("catalog-insights").innerHTML = `
    <div class="summary-grid summary-grid--stacked">
      ${shared.metricCard("Tasks", tasks.length)}
      ${shared.metricCard("Easy", totals.easy)}
      ${shared.metricCard("Medium", totals.medium)}
      ${shared.metricCard("Hard", totals.hard)}
    </div>
    <article class="story-card">
      <h3>Launch Context</h3>
      <div class="stack stack--dense">
        <div class="info-pair"><span>Selected task</span><strong>${currentTask ? currentTask.name : "n/a"}</strong></div>
        <div class="info-pair"><span>Current agent</span><strong>${currentAgent ? currentAgent.name : "n/a"}</strong></div>
        <div class="info-pair"><span>Recommendation</span><strong>Open the Studio from a task card so the replay starts on the exact benchmark you mean to inspect.</strong></div>
      </div>
    </article>
  `;
}
