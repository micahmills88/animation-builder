const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function fmtBytes(n) {
  if (n == null) return "—";
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
  return (n / (1024 * 1024)).toFixed(1) + " MB";
}

function fmtTime(iso) {
  if (!iso) return "—";
  return new Date(iso).toLocaleString();
}

function truncate(s, n) {
  if (!s) return "";
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function loadModels() {
  const models = await fetch("/api/models").then(r => r.json());
  const select = $("#model-select");
  select.innerHTML = models.map(m => `<option value="${m.id}">${m.label}</option>`).join("");
}

async function loadJobs() {
  const jobs = await fetch("/api/animations").then(r => r.json());
  const tbody = $("#jobs-table tbody");
  if (jobs.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" class="muted">No animations yet. Submit the form to create one.</td></tr>`;
    return;
  }
  tbody.innerHTML = jobs.map(j => {
    const actions = [];
    if (j.status === "completed") {
      actions.push(`<a class="btn-view" href="/ui/view.html?job=${j.id}" title="Open the viewer to scrub + trim + regenerate">View / Edit</a>`);
      actions.push(`<a class="btn-download" href="/api/animations/${j.id}/clip" title="Download the FBX">Download</a>`);
    }
    if (j.status === "failed") {
      actions.push(`<span title="${escapeHtml(j.error || "")}" class="err">error</span>`);
    }
    if ((j.status === "completed" || j.status === "failed") && j.template_fbx) {
      actions.push(`<button type="button" class="btn-rerun" data-job="${j.id}" title="Re-run this animation with identical inputs">Rerun</button>`);
    }
    const actionsHtml = actions.join(" · ");
    const modelShort = j.model ? j.model.replace("Kimodo-SOMA-", "").replace("-v1.1", "") : "—";
    const promptCell = j.prompt
      ? `<span title="${escapeHtml(j.prompt)}">${escapeHtml(truncate(j.prompt, 70))}</span>`
      : `<span class="muted">—</span>`;
    return `
      <tr>
        <td>${fmtTime(j.created_at)}</td>
        <td class="prompt-cell">${promptCell}</td>
        <td title="${escapeHtml(j.model || "")}"><code>${escapeHtml(modelShort)}</code></td>
        <td>${j.seed ?? "—"}</td>
        <td>${j.duration_s != null ? j.duration_s.toFixed(1) + "s" : "—"}</td>
        <td><span class="status status-${j.status}">${j.status}</span></td>
        <td>${fmtBytes(j.fbx_size)}</td>
        <td>${actionsHtml}</td>
        <td><button type="button" class="btn-delete" data-job="${j.id}" title="Delete">🗑</button></td>
      </tr>
    `;
  }).join("");

  $$("#jobs-table .btn-delete").forEach(btn => {
    btn.addEventListener("click", async () => {
      if (!confirm("Delete this animation and its files?")) return;
      const id = btn.dataset.job;
      const resp = await fetch(`/api/animations/${id}`, { method: "DELETE" });
      if (!resp.ok) {
        alert("Delete failed: " + await resp.text());
        return;
      }
      await loadJobs();
    });
  });

  $$("#jobs-table .btn-rerun").forEach(btn => {
    btn.addEventListener("click", async () => {
      const id = btn.dataset.job;
      if (!confirm("Re-run this animation with identical inputs? A new animation entry will be created.")) return;
      btn.disabled = true;
      btn.textContent = "…";
      try {
        const resp = await fetch(`/api/animations/${id}/rerun`, { method: "POST" });
        if (!resp.ok) throw new Error(await resp.text());
        await loadJobs();
      } catch (err) {
        alert("Rerun failed: " + err.message);
        btn.disabled = false;
        btn.textContent = "Rerun";
      }
    });
  });
}

$("#refresh-jobs").addEventListener("click", loadJobs);

$("#generate-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const submit = e.target.querySelector("button[type=submit]");
  submit.disabled = true;
  submit.textContent = "Submitting…";
  try {
    const resp = await fetch("/api/animations", { method: "POST", body: fd });
    if (!resp.ok) throw new Error(await resp.text());
    await loadJobs();
  } catch (err) {
    alert("Generate failed: " + err.message);
  } finally {
    submit.disabled = false;
    submit.textContent = "Generate";
  }
});

loadModels();
loadJobs();
setInterval(loadJobs, 5000);
