document.addEventListener("DOMContentLoaded", () => {
  void initAuditPage();
});

async function initAuditPage() {
  const shared = window.MedusaShared;
  const { tasks, agents } = await shared.loadCatalog();
  shared.renderTopbarMeta(tasks, agents);

  try {
    const graderData = await shared.fetchJSON("/api/run/grader", shared.basePayload(), "POST");
    renderGrader(graderData);
  } catch (err) {
    document.getElementById("grader-report").innerHTML =
      `<article class="audit-status-card"><span>Grader</span>
       <strong>Unavailable</strong>
       <p>${escapeHTML(err.message || "Could not load grader data.")}</p></article>`;
  }
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
  return `${value ?? ""}`.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[c])
  );
}
