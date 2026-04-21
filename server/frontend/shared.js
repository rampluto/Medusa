(function () {
  const STORAGE_KEY = "medusa-live-playground-state";
  const defaultState = {
    taskId: null,
    agentId: null,
    actions: [],
    selectedTable: "bronze_a",
    tablePage: 1,
    tablePageSize: 15,
  };

  let cache = {
    tasks: null,
    actionSpace: null,
    agents: null,
    defaultAgentId: null,
  };

  function loadState() {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        return { ...defaultState };
      }
      const parsed = JSON.parse(raw);
      return { ...defaultState, ...parsed };
    } catch (error) {
      return { ...defaultState };
    }
  }

  function saveState(nextState) {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(nextState));
  }

  function getState() {
    return loadState();
  }

  function updateState(patch) {
    const nextState = { ...getState(), ...patch };
    saveState(nextState);
    return nextState;
  }

  function setTask(taskId) {
    return updateState({
      taskId,
      actions: [],
      tablePage: 1,
    });
  }

  function resetTrace() {
    return updateState({
      actions: [],
      tablePage: 1,
    });
  }

  function undoAction() {
    const state = getState();
    return updateState({
      actions: state.actions.slice(0, -1),
    });
  }

  function addAction(actionName) {
    const state = getState();
    return updateState({
      actions: [...state.actions, { action: actionName, params: {} }],
    });
  }

  function setTrace(actions) {
    return updateState({
      actions: actions.map((action) => ({
        action: action.action,
        params: action.params || {},
      })),
      tablePage: 1,
    });
  }

  function setAgent(agentId) {
    return updateState({
      agentId,
      actions: [],
      tablePage: 1,
    });
  }

  function setTable(table) {
    return updateState({
      selectedTable: table,
      tablePage: 1,
    });
  }

  function setTablePage(page) {
    return updateState({ tablePage: page });
  }

  function basePayload() {
    const state = getState();
    return {
      task_id: state.taskId,
      actions: state.actions,
    };
  }

  function agentPayload() {
    const state = getState();
    return {
      agent_id: state.agentId,
    };
  }

  async function fetchJSON(url, body = null, method = "GET") {
    const options = {
      method,
      headers: {
        "Content-Type": "application/json",
      },
    };

    if (body !== null) {
      options.body = JSON.stringify(body);
    }

    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) {
      const message = data.detail ? JSON.stringify(data.detail) : response.statusText;
      throw new Error(message);
    }
    return data;
  }

  async function loadCatalog() {
    if (cache.tasks && cache.actionSpace && cache.agents) {
      return cache;
    }
    const [tasksResponse, actionResponse, agentsResponse] = await Promise.all([
      fetchJSON("/api/tasks"),
      fetchJSON("/api/action-space"),
      fetchJSON("/api/agents"),
    ]);
    cache = {
      tasks: tasksResponse.tasks,
      actionSpace: actionResponse.actions,
      agents: agentsResponse.agents,
      defaultAgentId: agentsResponse.default_agent_id,
    };
    ensureTask(cache.tasks);
    ensureAgent(cache.agents, cache.defaultAgentId);
    return cache;
  }

  function ensureTask(tasks) {
    const state = getState();
    const taskIds = new Set(tasks.map((task) => task.id));
    const queryTask = new URLSearchParams(window.location.search).get("task_id");

    if (queryTask && taskIds.has(queryTask)) {
      setTask(queryTask);
      return queryTask;
    }

    if (state.taskId && taskIds.has(state.taskId)) {
      return state.taskId;
    }

    if (tasks[0]) {
      setTask(tasks[0].id);
      return tasks[0].id;
    }

    return null;
  }

  function getCurrentTask(tasks) {
    const state = getState();
    return tasks.find((task) => task.id === state.taskId) || tasks[0] || null;
  }

  function ensureAgent(agents, defaultAgentId = null) {
    const state = getState();
    const agentIds = new Set(agents.map((agent) => agent.id));
    const queryAgent = new URLSearchParams(window.location.search).get("agent_id");

    if (queryAgent && agentIds.has(queryAgent)) {
      setAgent(queryAgent);
      return queryAgent;
    }

    if (state.agentId && agentIds.has(state.agentId)) {
      return state.agentId;
    }

    const fallback = defaultAgentId || agents.find((agent) => agent.default)?.id || agents[0]?.id || null;
    if (fallback) {
      setAgent(fallback);
    }
    return fallback;
  }

  function getCurrentAgent(agents) {
    const state = getState();
    return agents.find((agent) => agent.id === state.agentId) || agents.find((agent) => agent.default) || agents[0] || null;
  }

  function titleize(value) {
    return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function formatPercent(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "n/a";
    }
    return `${(Number(value) * 100).toFixed(1)}%`;
  }

  function formatNumber(value) {
    if (value === null || value === undefined || value === "") {
      return "n/a";
    }
    if (typeof value === "number") {
      return Number.isInteger(value) ? `${value}` : value.toFixed(3);
    }
    return `${value}`;
  }

  function formatValue(value) {
    if (value === null || value === undefined || value === "") {
      return "—";
    }
    if (typeof value === "boolean") {
      return value ? "true" : "false";
    }
    if (Array.isArray(value)) {
      return value.join(", ");
    }
    if (typeof value === "object") {
      return JSON.stringify(value);
    }
    return `${value}`;
  }

  function clampBar(value) {
    const number = Number(value);
    if (Number.isNaN(number)) {
      return 0;
    }
    return Math.max(0, Math.min(100, number * 100));
  }

  function metricCard(label, value) {
    return `
      <article class="metric">
        <span class="metric__label">${label}</span>
        <span class="metric__value">${value}</span>
      </article>
    `;
  }

  function difficultyTone(difficulty) {
    if (difficulty === "easy") {
      return "tone-good";
    }
    if (difficulty === "hard") {
      return "tone-hot";
    }
    return "tone-warn";
  }

  function renderTopbarMeta(tasks, agents = []) {
    const host = document.getElementById("topbar-meta");
    if (!host) {
      return;
    }

    const state = getState();
    const task = getCurrentTask(tasks);
    const agent = agents.length ? getCurrentAgent(agents) : null;
    host.innerHTML = task
      ? `
          <span class="status-pill">
            ${task.name} · ${state.actions.length} step${state.actions.length === 1 ? "" : "s"}${agent ? ` · ${agent.name}` : ""}
          </span>
        `
      : `<span class="status-pill">No active task</span>`;
  }

  window.MedusaShared = {
    loadCatalog,
    fetchJSON,
    getState,
    updateState,
    setTask,
    setAgent,
    resetTrace,
    undoAction,
    addAction,
    setTrace,
    setTable,
    setTablePage,
    basePayload,
    agentPayload,
    ensureTask,
    ensureAgent,
    getCurrentTask,
    getCurrentAgent,
    titleize,
    formatPercent,
    formatNumber,
    formatValue,
    clampBar,
    metricCard,
    difficultyTone,
    renderTopbarMeta,
  };
})();
