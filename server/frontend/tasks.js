document.addEventListener("DOMContentLoaded", () => {
  void initTaskCatalog();
});

async function initTaskCatalog() {
  const shared = window.MedusaShared;
  const { tasks, agents } = await shared.loadCatalog();
  shared.renderTopbarMeta(tasks, agents);
  renderTaskCards(tasks);
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
          <p class="task-card__description">${task.description}</p>
          <div class="action-row">
            <button class="button" data-task="${task.id}">Open Medusa</button>
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
}
