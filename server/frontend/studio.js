document.addEventListener("DOMContentLoaded", () => {
  void initStudio().catch((error) => showRunError(error));
});

// ── Gauntlet config ─────────────────────────────────────────────────────────
const GAUNTLET_DAYS = 30;
const GAUNTLET_BOSS_DAYS = new Set([8, 14, 21, 28]);
const GAUNTLET_TRAP_BADGES = { 8: "🧪", 14: "💥", 21: "🌿", 28: "☢️" };
const GAUNTLET_TRAP_NAMES = {
  8: "Type Trap",
  14: "OOM Trap",
  21: "Schema Drift",
  28: "Null Nuke",
};
const GAUNTLET_STEP_DELAY_MS = 110;
const GAUNTLET_COMMIT_HOLD_MS = 320;

// Mutable state for the in-flight animation, used for cancellation.
let gauntletAnimationToken = 0;

async function initStudio() {
  const shared = window.MedusaShared;
  renderGauntletShell();
  setupDayDetail();
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
  resetGauntlet("Running agent…");

  try {
    const preview = await shared.fetchJSON(
      `/api/run/autorun/${encodeURIComponent(state.taskId)}`,
      shared.agentPayload(),
      "POST"
    );
    shared.setTrace(preview.actions);
    window.__medusaPreview = preview;
    await renderPreview(tasks, agents, { animateGauntlet: true });
  } finally {
    runButton.disabled = false;
    runButton.textContent = "Run Selected Agent";
  }
}

function showRunError(error) {
  const message = error && error.message ? error.message : `${error}`;
  document.getElementById("hero-status").innerHTML = `<span class="status-pill is-bad">Run failed</span>`;
  document.getElementById("trace-list").innerHTML = `<li class="empty">${message}</li>`;
  setGauntletStatus(`<span class="gauntlet-status-pill is-bad">Run failed: ${escapeHTML(message)}</span>`);
}

async function refreshPreview(tasks, agents) {
  const shared = window.MedusaShared;
  const preview = await shared.fetchJSON("/api/run/preview", shared.basePayload(), "POST");
  window.__medusaPreview = preview;
  await renderPreview(tasks, agents, { animateGauntlet: false });
}

async function renderPreview(tasks = null, agents = null, options = {}) {
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

  renderTraceList(preview);

  if (preview.action_count > 0) {
    let timeline = [];
    try {
      const timelinePayload = await shared.fetchJSON(
        "/api/run/timeline",
        shared.basePayload(),
        "POST"
      );
      timeline = Array.isArray(timelinePayload.timeline) ? timelinePayload.timeline : [];
    } catch (error) {
      console.warn("medusa: failed to load timeline", error);
    }
    if (options.animateGauntlet) {
      void animateGauntlet(timeline, preview);
    } else {
      paintGauntletFinal(timeline, preview);
    }
  } else {
    resetGauntlet("Select an agent and run to begin.");
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gauntlet calendar rendering
// ─────────────────────────────────────────────────────────────────────────────

function renderGauntletShell() {
  const row1 = document.getElementById("gauntlet-row-1");
  const row2 = document.getElementById("gauntlet-row-2");
  if (!row1 || !row2) return;
  const tileHTML = (day) => {
    const isBoss = GAUNTLET_BOSS_DAYS.has(day);
    return `
      <div
        class="gauntlet-tile is-hidden${isBoss ? " gauntlet-tile--boss" : ""}"
        role="listitem"
        tabindex="0"
        data-day="${day}"
        data-boss="${isBoss ? "true" : "false"}"
        title="Day ${day} — not yet reached"
      >
        <span class="gauntlet-tile__day">${String(day).padStart(2, "0")}</span>
        <span class="gauntlet-tile__badge" aria-hidden="true"></span>
      </div>
    `;
  };
  const phase1 = [];
  const phase2 = [];
  for (let day = 1; day <= GAUNTLET_DAYS; day += 1) {
    (day <= 15 ? phase1 : phase2).push(tileHTML(day));
  }
  row1.innerHTML = phase1.join("");
  row2.innerHTML = phase2.join("");

  document.querySelectorAll(".gauntlet-tile").forEach((tile) => {
    tile.addEventListener("click", () => {
      const day = Number(tile.dataset.day);
      scrollTraceToDay(day);
      pinDay(day, { source: "click", loadFull: true });
    });
    tile.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        const day = Number(tile.dataset.day);
        scrollTraceToDay(day);
        pinDay(day, { source: "keyboard", loadFull: true });
      }
    });
  });

  setProgressBar(0, 30);
  setKpi("days", "0 / 30");
  setKpi("reward", "0.0");
  setKpi("outcome", "—");
}

function getTile(day) {
  return document.querySelector(`.gauntlet-tile[data-day="${day}"]`);
}

function resetGauntlet(message = "") {
  gauntletAnimationToken += 1;
  document.querySelectorAll(".gauntlet-tile").forEach((tile) => {
    tile.classList.remove("is-active", "is-probed", "is-pass", "is-partial", "is-fail", "is-pinned");
    tile.classList.add("is-hidden");
    const badge = tile.querySelector(".gauntlet-tile__badge");
    if (badge) badge.textContent = "";
    const day = Number(tile.dataset.day);
    tile.title = `Day ${day} — not yet reached`;
    tile.dataset.dayState = "hidden";
  });
  setGauntletStatus(message);
  setProgressBar(0, 30);
  setKpi("days", "0 / 30");
  setKpi("reward", "0.0");
  setKpi("outcome", "—", null);
  resetDayDetail();
}

function setGauntletStatus(html) {
  const status = document.getElementById("gauntlet-status");
  if (!status) return;
  status.innerHTML = html || "";
}

function setProgressBar(committed, total = 30) {
  const bar = document.getElementById("gauntlet-progress-bar");
  if (!bar) return;
  const pct = Math.max(0, Math.min(100, (committed / total) * 100));
  bar.style.width = `${pct}%`;
}

function setKpi(name, value, tone = null) {
  const id = `gauntlet-kpi-${name}`;
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = value;
  const card = el.closest(".gauntlet-kpi");
  if (!card) return;
  card.classList.remove("gauntlet-kpi--good", "gauntlet-kpi--warn", "gauntlet-kpi--bad");
  if (tone) card.classList.add(`gauntlet-kpi--${tone}`);
}

function countCommittedDays() {
  return document.querySelectorAll(
    ".gauntlet-tile.is-pass, .gauntlet-tile.is-partial"
  ).length;
}

function applyTileState(day, kind, payload = {}) {
  const tile = getTile(day);
  if (!tile) return;
  const isBoss = GAUNTLET_BOSS_DAYS.has(day);
  tile.classList.remove("is-hidden", "is-active", "is-probed", "is-pass", "is-partial", "is-fail");
  const badge = tile.querySelector(".gauntlet-tile__badge");
  const setBadge = (text) => { if (badge) badge.textContent = text; };

  switch (kind) {
    case "active":
      tile.classList.add("is-active");
      setBadge("");
      tile.title = `Day ${day} — agent active${isBoss ? " · boss day" : ""}`;
      tile.dataset.dayState = "active";
      break;
    case "probed":
      tile.classList.add("is-probed");
      setBadge("⚠");
      tile.title = `Day ${day} — anomaly detected by profile_table`;
      tile.dataset.dayState = "probed";
      break;
    case "pass": {
      tile.classList.add("is-pass");
      setBadge(isBoss ? GAUNTLET_TRAP_BADGES[day] : "✓");
      tile.title = buildTileTitle(day, "passed", payload);
      tile.dataset.dayState = "pass";
      break;
    }
    case "partial": {
      tile.classList.add("is-partial");
      setBadge(isBoss ? GAUNTLET_TRAP_BADGES[day] : "◐");
      tile.title = buildTileTitle(day, "partial", payload);
      tile.dataset.dayState = "partial";
      break;
    }
    case "fail":
      tile.classList.add("is-fail");
      setBadge("✕");
      tile.title = buildTileTitle(day, "crash", payload);
      tile.dataset.dayState = "fail";
      break;
    default:
      tile.classList.add("is-hidden");
      setBadge("");
      tile.title = `Day ${day} — not yet reached`;
      tile.dataset.dayState = "hidden";
  }
}

function buildTileTitle(day, outcome, payload) {
  const parts = [`Day ${day} — ${outcome}`];
  if (GAUNTLET_BOSS_DAYS.has(day)) {
    parts.push(`Trap: ${GAUNTLET_TRAP_NAMES[day]}`);
  }
  if (payload.steps != null) {
    parts.push(`${payload.steps} step${payload.steps === 1 ? "" : "s"}`);
  }
  if (payload.dayReward != null) {
    parts.push(`reward ${formatSignedReward(payload.dayReward)}`);
  }
  if (payload.cumulativeReward != null) {
    parts.push(`cum ${formatSignedReward(payload.cumulativeReward)}`);
  }
  if (payload.note) {
    parts.push(payload.note);
  }
  return parts.join("\n");
}

function formatSignedReward(value) {
  const num = Number(value);
  if (Number.isNaN(num)) return `${value}`;
  return `${num >= 0 ? "+" : ""}${num.toFixed(1)}`;
}

// ─── Animation ─────────────────────────────────────────────────────────────

async function animateGauntlet(timeline, preview) {
  resetGauntlet("Replaying agent…");
  if (!Array.isArray(timeline) || timeline.length === 0) {
    setGauntletStatus(`<span class="gauntlet-status-pill">No timeline data.</span>`);
    return;
  }

  const myToken = ++gauntletAnimationToken;
  const dayStats = buildDayStats(timeline);
  const probedDays = new Set();
  let lastAnimatedDay = null;

  for (const entry of timeline) {
    if (myToken !== gauntletAnimationToken) return; // cancelled
    const day = Number(entry.day || entry.current_day || 0);
    const action = String(entry.action || "");
    if (!day || day < 1 || day > GAUNTLET_DAYS) {
      continue;
    }

    if (lastAnimatedDay !== day) {
      if (lastAnimatedDay && getTile(lastAnimatedDay)?.dataset.dayState === "active") {
        // Day passed without commit → leave as-is until commit/crash entries arrive.
      }
      applyTileState(day, "active");
      lastAnimatedDay = day;
      setGauntletStatus(
        `<span class="gauntlet-status-pill">Day ${day}/${GAUNTLET_DAYS} · agent active</span>`
      );
      await sleep(GAUNTLET_STEP_DELAY_MS);
      if (myToken !== gauntletAnimationToken) return;
    }

    if (action === "PROFILE_TABLE" && !probedDays.has(day)) {
      probedDays.add(day);
      applyTileState(day, "probed");
      setGauntletStatus(
        `<span class="gauntlet-status-pill">Day ${day} · ⚠ anomaly detected</span>`
      );
      await sleep(GAUNTLET_STEP_DELAY_MS);
      if (myToken !== gauntletAnimationToken) return;
      continue;
    }

    if (action === "COMMIT_DAY" || action === "COMMIT") {
      const stats = dayStats.get(day) || {};
      const outcome = entry.grader_passed === true ? "pass" : "partial";
      applyTileState(day, outcome, {
        steps: stats.steps,
        dayReward: stats.dayReward,
        cumulativeReward: entry.cumulative_reward,
        note: entry.grader_passed === false ? "grader did not fully pass" : undefined,
      });
      const committed = countCommittedDays();
      setProgressBar(committed, GAUNTLET_DAYS);
      setKpi("days", `${committed} / ${GAUNTLET_DAYS}`, outcome === "pass" ? "good" : "warn");
      if (entry.cumulative_reward != null) {
        setKpi("reward", formatSignedReward(entry.cumulative_reward));
      }
      setGauntletStatus(
        outcome === "pass"
          ? `<span class="gauntlet-status-pill is-good">Day ${day} committed · grader passed</span>`
          : `<span class="gauntlet-status-pill is-warn">Day ${day} committed · grader partial</span>`
      );
      // Visual-only pin during animation: paint the tile and populate the
      // Day Detail panel from timeline data we already have. We deliberately
      // skip the /api/run/day-detail fetch here — auto-pinning every commit
      // would trigger up to 30 full-replay requests for a single 30-day run
      // (≈155s of redundant server work). The final paint below does one
      // proper fetch for the latest day, and manual tile clicks do their own.
      pinDayVisual(day, {
        outcome,
        stats,
        cumulativeReward: entry.cumulative_reward,
        graderPassed: entry.grader_passed,
        graderReport: entry.grader_report,
      });
      await sleep(GAUNTLET_COMMIT_HOLD_MS);
      if (myToken !== gauntletAnimationToken) return;
      lastAnimatedDay = null;
      continue;
    }

    await sleep(GAUNTLET_STEP_DELAY_MS);
  }

  if (myToken !== gauntletAnimationToken) return;

  // Final state: if the run ended without a successful 30-day commit, mark
  // the last touched day as crash (red 💀).
  const summary = preview && preview.summary ? preview.summary : {};
  const finalState = preview && preview.state ? preview.state : {};
  const reachedDay = Number(finalState.current_day || 0);
  const stage = finalState.stage || summary.stage;
  const lastTimelineDay = Number(timeline[timeline.length - 1]?.day || reachedDay || 0);
  const lastDayCommitted = timeline
    .filter((e) => (e.action === "COMMIT_DAY" || e.action === "COMMIT") && Number(e.day) === lastTimelineDay)
    .length > 0;

  if (stage === "failed" || (!lastDayCommitted && lastTimelineDay > 0 && summary.done)) {
    const stats = dayStats.get(lastTimelineDay) || {};
    applyTileState(lastTimelineDay, "fail", {
      steps: stats.steps,
      dayReward: stats.dayReward,
      cumulativeReward: summary.cumulative_reward,
      note: "terminal crash",
    });
    setKpi("outcome", `Crashed · Day ${lastTimelineDay}`, "bad");
    if (summary.cumulative_reward != null) {
      setKpi("reward", formatSignedReward(summary.cumulative_reward), "bad");
    }
    setGauntletStatus(
      `<span class="gauntlet-status-pill is-bad">Crash on Day ${lastTimelineDay} · cumulative ${formatSignedReward(summary.cumulative_reward)}</span>`
    );
    // Single full fetch at the end — replaces the per-commit fetch storm.
    pinDay(lastTimelineDay, { source: "auto-final", loadFull: true });
  } else {
    const finalDay = lastDayCommitted ? lastTimelineDay : reachedDay || lastTimelineDay;
    const committed = countCommittedDays();
    const allGood = committed === GAUNTLET_DAYS;
    setKpi("outcome", allGood ? "Survived 30 days" : `Reached Day ${finalDay}`, allGood ? "good" : "warn");
    if (summary.cumulative_reward != null) {
      setKpi("reward", formatSignedReward(summary.cumulative_reward), allGood ? "good" : null);
    }
    setGauntletStatus(
      `<span class="gauntlet-status-pill is-good">Run complete · reached Day ${finalDay} · cumulative ${formatSignedReward(summary.cumulative_reward)}</span>`
    );
    if (finalDay >= 1) pinDay(finalDay, { source: "auto-final", loadFull: true });
  }
}

function paintGauntletFinal(timeline, preview) {
  resetGauntlet();
  if (!Array.isArray(timeline) || timeline.length === 0) {
    setGauntletStatus("");
    return;
  }
  gauntletAnimationToken += 1; // freeze any in-flight animation
  const dayStats = buildDayStats(timeline);
  const probedDays = new Set();
  const committedDays = new Map(); // day -> {grader_passed, cumulative_reward}

  for (const entry of timeline) {
    const day = Number(entry.day || 0);
    if (!day) continue;
    if (entry.action === "PROFILE_TABLE") probedDays.add(day);
    if (entry.action === "COMMIT_DAY" || entry.action === "COMMIT") {
      committedDays.set(day, {
        passed: entry.grader_passed === true,
        cum: entry.cumulative_reward,
      });
    }
  }

  for (const day of probedDays) {
    if (!committedDays.has(day)) {
      applyTileState(day, "probed");
    }
  }
  for (const [day, info] of committedDays.entries()) {
    const stats = dayStats.get(day) || {};
    applyTileState(day, info.passed ? "pass" : "partial", {
      steps: stats.steps,
      dayReward: stats.dayReward,
      cumulativeReward: info.cum,
    });
  }

  const summary = preview && preview.summary ? preview.summary : {};
  const finalState = preview && preview.state ? preview.state : {};
  const stage = finalState.stage || summary.stage;
  const lastTimelineDay = Number(timeline[timeline.length - 1]?.day || 0);
  const lastDayCommitted = committedDays.has(lastTimelineDay);
  const committedCount = countCommittedDays();
  setProgressBar(committedCount, GAUNTLET_DAYS);
  setKpi("days", `${committedCount} / ${GAUNTLET_DAYS}`, committedCount === GAUNTLET_DAYS ? "good" : null);
  if (summary.cumulative_reward != null) {
    setKpi("reward", formatSignedReward(summary.cumulative_reward));
  }

  if (stage === "failed" || (!lastDayCommitted && lastTimelineDay > 0 && summary.done)) {
    const stats = dayStats.get(lastTimelineDay) || {};
    applyTileState(lastTimelineDay, "fail", {
      steps: stats.steps,
      dayReward: stats.dayReward,
      cumulativeReward: summary.cumulative_reward,
      note: "terminal crash",
    });
    setKpi("outcome", `Crashed · Day ${lastTimelineDay}`, "bad");
    if (summary.cumulative_reward != null) {
      setKpi("reward", formatSignedReward(summary.cumulative_reward), "bad");
    }
    setGauntletStatus(
      `<span class="gauntlet-status-pill is-bad">Crash on Day ${lastTimelineDay} · cumulative ${formatSignedReward(summary.cumulative_reward)}</span>`
    );
    pinDay(lastTimelineDay, { source: "auto-final", loadFull: true });
  } else if (committedDays.size > 0) {
    const finalDay = Math.max(...committedDays.keys());
    const allGood = committedCount === GAUNTLET_DAYS;
    setKpi("outcome", allGood ? "Survived 30 days" : `Reached Day ${finalDay}`, allGood ? "good" : "warn");
    setGauntletStatus(
      `<span class="gauntlet-status-pill is-good">Last replay · reached Day ${finalDay} · cumulative ${formatSignedReward(summary.cumulative_reward)}</span>`
    );
    pinDay(finalDay, { source: "auto-final", loadFull: true });
  }
}

// ─── Day Detail panel ──────────────────────────────────────────────────────
//
// Pinned per-day audit (DQ before/after, cumulative-reward context, and a
// download for the cleaned silver snapshot). Driven by /api/run/day-detail
// and /api/run/day-snapshot/{day}.csv on the backend.

const dayDetailState = {
  day: null,
  tab: "overview",
  fetchToken: 0,
  downloadObjectUrl: null,
  // Tracks the in-flight day-detail fetch so a new pin can abort the old
  // request server-side (closing the HTTP connection). Without this, stale
  // requests still consume CPU on the server even though we'd ignore the
  // result — exactly the pathology that made auto-pin during animation
  // pile up minutes of redundant work.
  abortController: null,
  // Only days that have actually been server-fetched have full DQ data.
  // Visual-only pins (during animation) display lightweight overview info
  // from the timeline payload until the user clicks or animation ends.
  fullyLoadedDay: null,
};

const DAY_DETAIL_PANES = {
  overview: "day-detail-pane-overview",
  dq: "day-detail-pane-dq",
  download: "day-detail-pane-download",
};

function setupDayDetail() {
  const tabs = document.getElementById("day-detail-tabs");
  if (tabs) {
    tabs.addEventListener("click", (event) => {
      const button = event.target.closest(".day-detail__tab");
      if (!button) return;
      switchDayDetailTab(button.dataset.tab);
    });
  }
  const clear = document.getElementById("day-detail-clear");
  if (clear) clear.addEventListener("click", () => unpinDay());
  resetDayDetail();
}

function resetDayDetail() {
  dayDetailState.day = null;
  dayDetailState.tab = "overview";
  dayDetailState.fetchToken += 1;
  dayDetailState.fullyLoadedDay = null;
  abortInFlightDayDetailFetch();
  releaseDownloadObjectUrl();

  document.querySelectorAll(".gauntlet-tile.is-pinned").forEach((tile) => {
    tile.classList.remove("is-pinned");
  });

  const eyebrow = document.getElementById("day-detail-eyebrow");
  const title = document.getElementById("day-detail-title");
  const subtitle = document.getElementById("day-detail-subtitle");
  const pill = document.getElementById("day-detail-pill");
  const clear = document.getElementById("day-detail-clear");
  const tabs = document.getElementById("day-detail-tabs");
  const empty = document.getElementById("day-detail-empty");

  if (eyebrow) eyebrow.textContent = "Day Detail";
  if (title) title.textContent = "Click any committed day to inspect it";
  if (subtitle) {
    subtitle.textContent =
      "Hover for a quick read; click a tile to pin its full data-quality report and download the cleaned snapshot.";
  }
  if (pill) {
    pill.textContent = "Idle";
    pill.className = "day-detail__pill";
  }
  if (clear) clear.hidden = true;
  if (tabs) tabs.hidden = true;
  Object.values(DAY_DETAIL_PANES).forEach((id) => {
    const pane = document.getElementById(id);
    if (pane) pane.hidden = true;
  });
  if (empty) empty.hidden = false;
}

function switchDayDetailTab(tab) {
  if (!Object.prototype.hasOwnProperty.call(DAY_DETAIL_PANES, tab)) return;
  dayDetailState.tab = tab;
  document.querySelectorAll(".day-detail__tab").forEach((btn) => {
    const isActive = btn.dataset.tab === tab;
    btn.classList.toggle("is-active", isActive);
    btn.setAttribute("aria-selected", String(isActive));
  });
  Object.entries(DAY_DETAIL_PANES).forEach(([key, id]) => {
    const pane = document.getElementById(id);
    if (pane) pane.hidden = key !== tab;
  });
}

function pinDay(day, opts = {}) {
  const numeric = Number(day);
  if (!Number.isFinite(numeric) || numeric < 1 || numeric > GAUNTLET_DAYS) return;
  const tile = getTile(numeric);
  if (!tile) return;

  document.querySelectorAll(".gauntlet-tile.is-pinned").forEach((other) => {
    if (other !== tile) other.classList.remove("is-pinned");
  });
  tile.classList.add("is-pinned");

  // No-op fast path: if we're already pinned to this day AND it has been
  // fully fetched (or the caller explicitly only wanted a visual update),
  // there's nothing to do. The previous behaviour was to early-return on
  // any same-day pin, which dropped legitimate "promote this visual pin
  // into a full fetch" calls (e.g. animation finishes on a day the user
  // already manually clicked).
  const sameDay = dayDetailState.day === numeric;
  const wantsFull = opts.loadFull === true;
  const alreadyFullyLoaded = sameDay && dayDetailState.fullyLoadedDay === numeric;
  if (sameDay && (!wantsFull || alreadyFullyLoaded) && !opts.force) {
    return;
  }
  dayDetailState.day = numeric;

  const empty = document.getElementById("day-detail-empty");
  if (empty) empty.hidden = true;
  const tabs = document.getElementById("day-detail-tabs");
  if (tabs) tabs.hidden = false;
  const clear = document.getElementById("day-detail-clear");
  if (clear) clear.hidden = false;
  switchDayDetailTab(dayDetailState.tab || "overview");

  if (wantsFull) {
    void loadDayDetail(numeric);
  }
  // If !wantsFull, the caller is responsible for calling pinDayVisual to
  // populate the panel from already-known timeline data.
}

// Visual-only pin: paint the tile, populate the Day Detail panel from data
// the animation already has (timeline entry + dayStats), and crucially do
// not hit the server. This is what `animateGauntlet` calls after each
// `COMMIT_DAY` so a 30-day run doesn't fan out into 30 day-detail fetches.
function pinDayVisual(day, info = {}) {
  const numeric = Number(day);
  if (!Number.isFinite(numeric) || numeric < 1 || numeric > GAUNTLET_DAYS) return;
  pinDay(numeric, { source: "auto", loadFull: false });
  // pinDay early-returns when the day hasn't changed AND already-fully-loaded;
  // since we set fullyLoadedDay = null on visual pins, a later promote-to-full
  // call still goes through.
  dayDetailState.fullyLoadedDay = null;

  const status = info.outcome || (info.graderPassed === false ? "partial" : "pass");
  const isBoss = GAUNTLET_BOSS_DAYS.has(numeric);
  const trapName = isBoss ? GAUNTLET_TRAP_NAMES[numeric] : "";

  const eyebrowEl = document.getElementById("day-detail-eyebrow");
  const titleEl = document.getElementById("day-detail-title");
  const subtitleEl = document.getElementById("day-detail-subtitle");
  const pill = document.getElementById("day-detail-pill");
  const overviewPane = document.getElementById("day-detail-pane-overview");
  const dqPane = document.getElementById("day-detail-pane-dq");
  const downloadPane = document.getElementById("day-detail-pane-download");

  if (eyebrowEl) {
    eyebrowEl.textContent = isBoss
      ? `Boss Day · ${trapName}`
      : `Day ${String(numeric).padStart(2, "0")} audit`;
  }
  if (titleEl) {
    titleEl.textContent = isBoss
      ? `Day ${String(numeric).padStart(2, "0")} — ${trapName}`
      : `Day ${String(numeric).padStart(2, "0")}`;
  }
  if (subtitleEl) {
    subtitleEl.textContent =
      "Live preview from timeline. Click the tile after the run to fetch the full DQ report.";
  }
  if (pill) renderDayDetailPill(pill, status);

  const stats = info.stats || {};
  const cumulative = info.cumulativeReward;
  const metrics = [
    {
      label: "Day status",
      value: status === "pass" ? "Committed" : status === "partial" ? "Partial" : "—",
      sub: isBoss ? `${trapName} day` : "Daily commit",
      tone: status === "pass" ? "good" : "warn",
    },
    {
      label: "Steps used today",
      value: formatNumber(stats.steps),
      sub: stats.dayReward != null ? `day reward ${formatSignedReward(stats.dayReward)}` : "",
    },
  ];
  if (cumulative != null) {
    metrics.push({
      label: "Cumulative reward",
      value: formatSignedReward(cumulative),
      sub: "through this day",
    });
  }
  if (overviewPane) {
    overviewPane.innerHTML = `
      <div class="day-detail__metrics">
        ${metrics
          .map(
            (m) => `
              <div class="day-detail-metric">
                <span class="day-detail-metric__label">${escapeHTML(m.label)}</span>
                <span class="day-detail-metric__value${m.tone ? ` is-${m.tone}` : ""}">${escapeHTML(String(m.value))}</span>
                ${m.sub ? `<span class="day-detail-metric__sub">${escapeHTML(m.sub)}</span>` : ""}
              </div>
            `
          )
          .join("")}
      </div>
      <div class="day-detail__notes">
        <div class="day-detail-callout">
          <strong>Live replay:</strong> showing what the agent produced this day.
          The full data-quality report and silver download will load once the
          replay finishes (or click any tile to inspect it now).
        </div>
      </div>
    `;
  }
  if (dqPane) {
    dqPane.innerHTML = `<div class="day-detail__empty">Click the tile after the replay finishes to load the full DQ report.</div>`;
  }
  if (downloadPane) {
    downloadPane.innerHTML = `<div class="day-detail__empty">Download will be available after the replay finishes.</div>`;
  }
}

function unpinDay() {
  resetDayDetail();
}

function abortInFlightDayDetailFetch() {
  // AbortController has no effect once the response is fully received, but
  // it does cancel the underlying connection when the request is still
  // in-flight, so the server stops processing it. This matters whenever
  // the user pins a different day before the previous one finishes.
  const controller = dayDetailState.abortController;
  if (controller) {
    try {
      controller.abort();
    } catch (_) {
      /* ignore */
    }
  }
  dayDetailState.abortController = null;
}

async function loadDayDetail(day) {
  const shared = window.MedusaShared;
  const myToken = ++dayDetailState.fetchToken;
  abortInFlightDayDetailFetch();
  const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
  dayDetailState.abortController = controller;

  const titleEl = document.getElementById("day-detail-title");
  const subtitleEl = document.getElementById("day-detail-subtitle");
  const eyebrowEl = document.getElementById("day-detail-eyebrow");
  const pill = document.getElementById("day-detail-pill");
  const overviewPane = document.getElementById("day-detail-pane-overview");
  const dqPane = document.getElementById("day-detail-pane-dq");
  const downloadPane = document.getElementById("day-detail-pane-download");

  if (titleEl) titleEl.textContent = `Day ${String(day).padStart(2, "0")}`;
  if (subtitleEl) subtitleEl.textContent = "Loading day audit…";
  if (pill) {
    pill.className = "day-detail__pill";
    pill.textContent = "Loading…";
  }
  if (overviewPane) overviewPane.innerHTML = `<div class="day-detail__loading">Fetching day audit…</div>`;
  if (dqPane) dqPane.innerHTML = `<div class="day-detail__loading">Fetching DQ report…</div>`;
  if (downloadPane) downloadPane.innerHTML = `<div class="day-detail__loading">Preparing download…</div>`;

  let detail;
  try {
    const response = await fetch(`/api/run/day-detail/${day}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(shared.basePayload()),
      signal: controller ? controller.signal : undefined,
    });
    detail = await response.json();
    if (!response.ok) {
      throw new Error(
        typeof detail?.detail === "string" ? detail.detail : `HTTP ${response.status}`
      );
    }
  } catch (error) {
    if (error && error.name === "AbortError") return; // superseded by a newer pin
    if (myToken !== dayDetailState.fetchToken) return;
    if (overviewPane) {
      overviewPane.innerHTML = `<div class="day-detail__error">Could not load Day ${day} audit: ${escapeHTML(
        String(error.message || error)
      )}</div>`;
    }
    if (subtitleEl) subtitleEl.textContent = "Audit unavailable for this day.";
    if (pill) {
      pill.className = "day-detail__pill is-fail";
      pill.innerHTML = `<span class="day-detail__pill-dot"></span>Error`;
    }
    return;
  } finally {
    if (dayDetailState.abortController === controller) {
      dayDetailState.abortController = null;
    }
  }
  if (myToken !== dayDetailState.fetchToken) return;

  dayDetailState.fullyLoadedDay = day;
  renderDayDetail(detail);
}

function renderDayDetail(detail) {
  const day = Number(detail.day);
  const status = String(detail.status || "unknown");
  const isBoss = !!detail.is_boss_day;
  const trapName = detail.trap_name || "";

  const eyebrowEl = document.getElementById("day-detail-eyebrow");
  const titleEl = document.getElementById("day-detail-title");
  const subtitleEl = document.getElementById("day-detail-subtitle");
  const pill = document.getElementById("day-detail-pill");

  if (eyebrowEl) {
    eyebrowEl.textContent = isBoss
      ? `Boss Day · ${trapName}`
      : `Day ${String(day).padStart(2, "0")} audit`;
  }
  if (titleEl) {
    titleEl.textContent = isBoss
      ? `Day ${String(day).padStart(2, "0")} — ${trapName}`
      : `Day ${String(day).padStart(2, "0")}`;
  }
  if (subtitleEl) subtitleEl.textContent = subtitleForStatus(status, isBoss, trapName);
  if (pill) renderDayDetailPill(pill, status);

  renderDayDetailOverview(detail);
  renderDayDetailDQ(detail);
  renderDayDetailDownload(detail);
}

function subtitleForStatus(status, isBoss, trapName) {
  switch (status) {
    case "pass":
      return isBoss
        ? `Boss day cleared. The ${trapName} attack was contained and the silver delta committed.`
        : "Bronze ingested, anomalies cleaned, and the silver delta merged successfully.";
    case "partial":
      return "Day committed but the grader flagged residual issues — inspect the before/after report.";
    case "crash":
      return isBoss
        ? `Terminal crash on this boss day (${trapName}).`
        : "Terminal crash on this day — only the bronze inputs are available.";
    case "active":
      return "Currently in flight. The audit will populate once the agent commits this day.";
    case "unreached":
      return "The agent did not reach this day in this run.";
    default:
      return "No audit recorded for this day.";
  }
}

function renderDayDetailPill(pill, status) {
  pill.className = "day-detail__pill";
  let toneClass = "";
  let label = "";
  switch (status) {
    case "pass":
      toneClass = "is-pass";
      label = "Committed";
      break;
    case "partial":
      toneClass = "is-warn";
      label = "Partial";
      break;
    case "crash":
      toneClass = "is-fail";
      label = "Crashed";
      break;
    case "active":
      toneClass = "is-active";
      label = "Active";
      break;
    case "unreached":
      label = "Unreached";
      break;
    default:
      label = "Idle";
  }
  if (toneClass) pill.classList.add(toneClass);
  pill.innerHTML = `<span class="day-detail__pill-dot"></span>${escapeHTML(label)}`;
}

function renderDayDetailOverview(detail) {
  const pane = document.getElementById("day-detail-pane-overview");
  if (!pane) return;
  pane.innerHTML = `
    <div class="day-detail__metrics" id="day-detail-metrics"></div>
    <div class="day-detail__notes" id="day-detail-notes"></div>
  `;
  const metricsEl = pane.querySelector("#day-detail-metrics");
  const notesEl = pane.querySelector("#day-detail-notes");

  const summary = detail.summary || {};
  const dqBefore = detail.dq_before;
  const dqAfter = detail.dq_after;

  const beforeScore = dqBefore?.score;
  const afterScore = dqAfter?.score;
  const lift = beforeScore != null && afterScore != null ? afterScore - beforeScore : null;

  const metrics = [];
  metrics.push({
    label: "Bronze rows in",
    value: formatNumber(summary.raw_rows),
    sub: detail.is_boss_day ? `${detail.trap_name} day` : "Daily batch",
  });
  metrics.push({
    label: "Cleaned rows out",
    value: formatNumber(summary.cleaned_rows),
    sub: rowsDelta(summary.raw_rows, summary.cleaned_rows),
  });
  metrics.push({
    label: "Silver after commit",
    value: formatNumber(summary.silver_rows_after),
    sub: `+${formatNumber(summary.rows_added_to_silver)} this day`,
  });
  if (beforeScore != null) {
    metrics.push({
      label: "DQ — Bronze",
      value: formatPct(beforeScore),
      sub: dqBefore.passed ? "passes 0.80 floor" : "below 0.80 floor",
      tone: dqBefore.passed ? "good" : "warn",
    });
  }
  if (afterScore != null) {
    metrics.push({
      label: "DQ — Silver delta",
      value: formatPct(afterScore),
      sub: dqAfter.passed ? "passes 0.80 floor" : "below 0.80 floor",
      tone: dqAfter.passed ? "good" : "warn",
    });
  }
  if (lift != null) {
    metrics.push({
      label: "DQ lift",
      value: `${lift >= 0 ? "+" : ""}${(lift * 100).toFixed(1)}%`,
      sub: "after − before",
      tone: lift > 0.0001 ? "good" : lift < -0.0001 ? "bad" : null,
    });
  }
  if (detail.cumulative_reward != null) {
    metrics.push({
      label: "Cumulative reward",
      value: formatSignedReward(detail.cumulative_reward),
      sub: "through this day",
    });
  }

  metricsEl.innerHTML = metrics
    .map(
      (m) => `
      <div class="day-detail-metric">
        <span class="day-detail-metric__label">${escapeHTML(m.label)}</span>
        <span class="day-detail-metric__value${m.tone ? ` is-${m.tone}` : ""}">${escapeHTML(m.value)}</span>
        ${m.sub ? `<span class="day-detail-metric__sub">${escapeHTML(m.sub)}</span>` : ""}
      </div>
    `
    )
    .join("");

  const notes = [];
  if (detail.is_boss_day && detail.trap_name) {
    notes.push(
      `<div class="day-detail-callout"><strong>Boss day:</strong> ${escapeHTML(
        detail.trap_name
      )}. The agent had to recognise this trap from observable signals alone.</div>`
    );
  }
  if (detail.status === "pass" && detail.grader_passed) {
    notes.push(
      `<div class="day-detail-callout"><strong>Grader:</strong> deterministic Python checks passed. ${escapeHTML(
        detail.grader_report || ""
      )}</div>`
    );
  }
  if (detail.status === "partial") {
    notes.push(
      `<div class="day-detail-callout is-warn"><strong>Grader (partial):</strong> ${escapeHTML(
        detail.grader_report || "Some checks failed."
      )}</div>`
    );
  }
  if (detail.status === "crash") {
    notes.push(
      `<div class="day-detail-callout is-fail"><strong>Crash:</strong> ${escapeHTML(
        detail.failure_reason || detail.grader_report || "Episode terminated on this day."
      )}</div>`
    );
  }
  if (detail.status === "active") {
    notes.push(
      `<div class="day-detail-callout"><strong>Active:</strong> the agent is still working on this day. The audit will appear once it commits.</div>`
    );
  }
  if (detail.status === "unreached") {
    notes.push(
      `<div class="day-detail-callout"><strong>Unreached:</strong> the run ended before the agent got here.</div>`
    );
  }
  notesEl.innerHTML = notes.join("");
}

function renderDayDetailDQ(detail) {
  const pane = document.getElementById("day-detail-pane-dq");
  if (!pane) return;
  pane.innerHTML = `<div id="day-detail-dq"></div>`;
  const container = pane.querySelector("#day-detail-dq");
  if (!detail.dq_before && !detail.dq_after) {
    container.innerHTML = `<div class="day-detail__empty">No data-quality report available for this day.</div>`;
    return;
  }
  const beforeName = `Bronze · Day ${String(detail.day).padStart(2, "0")}`;
  const afterName = `Silver delta · Day ${String(detail.day).padStart(2, "0")}`;
  renderDqGrid(detail.dq_before || {}, detail.dq_after, beforeName, afterName, container);
}

function renderDayDetailDownload(detail) {
  const pane = document.getElementById("day-detail-pane-download");
  if (!pane) return;
  pane.innerHTML = "";

  if (!detail.download_available) {
    const reason =
      detail.status === "crash"
        ? "Terminal crash — no committed silver to download."
        : detail.status === "active"
        ? "The agent is still working on this day."
        : "No committed silver snapshot is available.";
    pane.innerHTML = `<div class="day-detail__empty">${escapeHTML(reason)}</div>`;
    return;
  }

  const summary = detail.summary || {};
  pane.innerHTML = `
    <div class="day-detail__download">
      <div class="day-detail__download-summary">
        <strong>Cleaned silver snapshot</strong> after Day ${String(detail.day).padStart(2, "0")} —
        ${formatNumber(summary.silver_rows_after)} rows cumulative,
        +${formatNumber(summary.rows_added_to_silver)} added this day.
      </div>
      <button type="button" class="button" id="day-detail-download-btn">Download silver_day_${String(detail.day).padStart(2, "0")}.csv</button>
      <div class="day-detail__download-meta" id="day-detail-download-meta">
        Click to fetch a CSV of the silver layer as of the end of Day ${String(detail.day).padStart(2, "0")}.
      </div>
    </div>
  `;
  const btn = pane.querySelector("#day-detail-download-btn");
  const meta = pane.querySelector("#day-detail-download-meta");
  btn.addEventListener("click", () => {
    void downloadDaySnapshot(detail.day, btn, meta);
  });
}

async function downloadDaySnapshot(day, btn, meta) {
  const shared = window.MedusaShared;
  btn.disabled = true;
  const originalLabel = btn.textContent;
  btn.textContent = "Preparing CSV…";
  if (meta) meta.textContent = `Building silver snapshot for Day ${day}…`;
  try {
    const response = await fetch(`/api/run/day-snapshot/${day}.csv`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(shared.basePayload()),
    });
    if (!response.ok) {
      let message = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        if (body?.detail) message = body.detail;
      } catch (_) {
        /* ignore */
      }
      throw new Error(message);
    }
    const blob = await response.blob();
    const filename =
      parseFilenameFromContentDisposition(response.headers.get("Content-Disposition")) ||
      `medusa_day${String(day).padStart(2, "0")}_silver.csv`;
    releaseDownloadObjectUrl();
    dayDetailState.downloadObjectUrl = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = dayDetailState.downloadObjectUrl;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    if (meta) meta.textContent = `Downloaded ${filename} (${formatBytes(blob.size)}).`;
  } catch (error) {
    if (meta) {
      meta.innerHTML = `<span style="color: var(--g-danger);">Download failed: ${escapeHTML(
        String(error.message || error)
      )}</span>`;
    }
  } finally {
    btn.disabled = false;
    btn.textContent = originalLabel;
  }
}

function parseFilenameFromContentDisposition(header) {
  if (!header) return null;
  const match = /filename="?([^";]+)"?/i.exec(header);
  return match ? match[1] : null;
}

function releaseDownloadObjectUrl() {
  if (dayDetailState.downloadObjectUrl) {
    URL.revokeObjectURL(dayDetailState.downloadObjectUrl);
    dayDetailState.downloadObjectUrl = null;
  }
}

function formatNumber(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toLocaleString();
}

function formatPct(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatBytes(bytes) {
  if (bytes == null || Number.isNaN(Number(bytes))) return "—";
  const num = Number(bytes);
  if (num < 1024) return `${num} B`;
  if (num < 1024 * 1024) return `${(num / 1024).toFixed(1)} KB`;
  return `${(num / (1024 * 1024)).toFixed(2)} MB`;
}

function rowsDelta(before, after) {
  if (before == null || after == null) return "";
  const delta = Number(after) - Number(before);
  if (delta === 0) return "no change";
  if (delta > 0) return `+${formatNumber(delta)} added`;
  return `${formatNumber(Math.abs(delta))} dropped`;
}

// ─── End Day Detail panel ──────────────────────────────────────────────────

function buildDayStats(timeline) {
  const stats = new Map();
  for (const entry of timeline) {
    const day = Number(entry.day || 0);
    if (!day) continue;
    if (!stats.has(day)) stats.set(day, { steps: 0, dayReward: 0 });
    const bucket = stats.get(day);
    bucket.steps += 1;
    bucket.dayReward += Number(entry.reward || 0);
  }
  return stats;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ─────────────────────────────────────────────────────────────────────────────
// Trace list rendering with day grouping (so the calendar can scroll into it)
// ─────────────────────────────────────────────────────────────────────────────

function renderTraceList(preview) {
  const traceListEl = document.getElementById("trace-list");
  const traceCountEl = document.getElementById("trace-count");
  const actions = Array.isArray(preview.actions) ? preview.actions : [];

  traceCountEl.textContent = `${actions.length} step${actions.length === 1 ? "" : "s"}`;

  if (actions.length === 0) {
    traceListEl.innerHTML = `<li class="empty">No actions yet. Select an agent to run the task.</li>`;
    return;
  }

  // We don't have per-action `day` from /api/run/preview — fall back to a flat
  // numbered list, but still tag each <li> with data-step so the calendar can
  // at least highlight steps.
  traceListEl.innerHTML = actions
    .map(
      (action, index) =>
        `<li data-step="${index + 1}"><strong>${index + 1}.</strong> ${escapeHTML(action.action)}</li>`
    )
    .join("");

  // After rendering, ask the timeline endpoint for day numbers and rewrite the
  // list with day-grouped banners. This is fire-and-forget and just enriches.
  const shared = window.MedusaShared;
  shared
    .fetchJSON("/api/run/timeline", shared.basePayload(), "POST")
    .then((payload) => {
      const timeline = Array.isArray(payload.timeline) ? payload.timeline : [];
      if (!timeline.length) return;
      rewriteTraceListWithDays(traceListEl, timeline);
    })
    .catch(() => {
      /* keep flat list */
    });
}

function rewriteTraceListWithDays(traceListEl, timeline) {
  const items = [];
  let prevDay = null;
  let stepCounter = 0;
  for (const entry of timeline) {
    const day = Number(entry.day || 0);
    if (!day) continue;
    if (day !== prevDay) {
      const isBoss = GAUNTLET_BOSS_DAYS.has(day);
      const trapHint = isBoss
        ? ` · ${GAUNTLET_TRAP_BADGES[day]} ${escapeHTML(GAUNTLET_TRAP_NAMES[day])}`
        : "";
      items.push(
        `<li data-day-banner="true" data-day="${day}">Day ${day}${trapHint}</li>`
      );
      prevDay = day;
    }
    stepCounter += 1;
    const action = escapeHTML(entry.action || "");
    const reward = entry.reward != null ? formatSignedReward(entry.reward) : "";
    const blocked = entry.blocked === true;
    const grader =
      entry.grader_passed === true
        ? ` <span class="tag" style="color: var(--success);">grader ✅</span>`
        : entry.grader_passed === false
        ? ` <span class="tag" style="color: var(--warn);">grader ⚠</span>`
        : "";
    items.push(
      `<li data-step="${stepCounter}" data-day="${day}">
        <strong>${stepCounter}.</strong> ${action}
        ${reward ? `<span class="tag">${reward}</span>` : ""}
        ${blocked ? `<span class="tag" style="color: var(--danger);">BLOCKED</span>` : ""}
        ${grader}
      </li>`
    );
  }
  if (items.length) {
    traceListEl.innerHTML = items.join("");
  }
}

function scrollTraceToDay(day) {
  const traceList = document.getElementById("trace-list");
  if (!traceList) return;
  const target =
    traceList.querySelector(`li[data-day-banner="true"][data-day="${day}"]`) ||
    traceList.querySelector(`li[data-day="${day}"]`);
  if (!target) return;

  traceList.querySelectorAll("li.is-day-highlight").forEach((el) => el.classList.remove("is-day-highlight"));
  traceList.querySelectorAll(`li[data-day="${day}"]`).forEach((el) => el.classList.add("is-day-highlight"));
  target.scrollIntoView({ behavior: "smooth", block: "center" });
  setTimeout(() => {
    traceList
      .querySelectorAll("li.is-day-highlight")
      .forEach((el) => el.classList.remove("is-day-highlight"));
  }, 2400);
}

// ─────────────────────────────────────────────────────────────────────────────
// Dataframe cleaner (unchanged)
// ─────────────────────────────────────────────────────────────────────────────

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
