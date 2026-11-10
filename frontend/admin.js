(() => {
  const state = { csrfToken: "", guides: [], offset: 0, limit: 20, total: 0, searchTimer: null };
  const byId = (id) => document.getElementById(id);

  function message(element, text, isError = false) {
    element.textContent = text;
    element.classList.toggle("error", isError);
  }

  async function api(path, options = {}) {
    const headers = new Headers(options.headers || {});
    if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
    if (options.method && !["GET", "HEAD"].includes(options.method.toUpperCase()) && state.csrfToken) {
      headers.set("X-CSRF-Token", state.csrfToken);
    }
    const response = await fetch(path, { ...options, headers, credentials: "same-origin" });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      if (response.status === 401) showLogin();
      throw new Error(data.detail || `Request failed (${response.status}).`);
    }
    return data;
  }

  function showLogin() {
    byId("loginPanel").hidden = false;
    byId("dashboard").hidden = true;
    byId("accountBar").hidden = true;
    state.csrfToken = "";
  }

  async function showDashboard(session) {
    state.csrfToken = session.csrf_token;
    byId("loginPanel").hidden = true;
    byId("dashboard").hidden = false;
    byId("accountBar").hidden = false;
    byId("accountName").textContent = session.username;
    await Promise.all([loadOverview(), loadGuides()]);
    await loadHistory();
  }

  async function loadOverview() {
    const overview = await api("/api/admin/overview");
    byId("totalScans").textContent = overview.total_scans;
    byId("scansToday").textContent = overview.scans_today;
    byId("diseasesSeen").textContent = overview.diseases_seen;
    byId("guideCount").textContent = overview.guide_count;
  }

  async function loadGuides() {
    const data = await api("/api/admin/guides");
    state.guides = data.guides;
    const select = byId("guideSelect");
    const filter = byId("classFilter");
    const selectedGuide = select.value;
    const selectedFilter = filter.value;
    select.replaceChildren();
    filter.replaceChildren(new Option("All classes", ""));
    for (const guide of state.guides) {
      const option = new Option(`${guide.disease_name} — ${guide.class_name}`, guide.class_name);
      select.add(option.cloneNode(true));
      filter.add(new Option(guide.class_name, guide.class_name));
    }
    if (state.guides.some((guide) => guide.class_name === selectedGuide)) select.value = selectedGuide;
    if (state.guides.some((guide) => guide.class_name === selectedFilter)) filter.value = selectedFilter;
    fillGuideForm();
  }

  function fillGuideForm() {
    const guide = state.guides.find((item) => item.class_name === byId("guideSelect").value) || state.guides[0];
    if (!guide) return;
    byId("guideSelect").value = guide.class_name;
    byId("guideName").value = guide.disease_name;
    byId("guideSymptoms").value = guide.symptoms;
    byId("guideOrganic").value = guide.organic_control;
    byId("guideChemical").value = guide.chemical_control;
    byId("guideSource").value = guide.source;
    message(byId("guideMessage"), "");
  }

  function renderHistory(items) {
    const list = byId("historyList");
    list.replaceChildren();
    if (items.length === 0) {
      const empty = document.createElement("p");
      empty.className = "empty-state";
      empty.textContent = "No scans match this search.";
      list.append(empty);
      return;
    }
    for (const item of items) {
      const card = document.createElement("article");
      card.className = "history-card";
      const image = document.createElement("img");
      image.className = "history-image";
      image.src = item.image_url;
      image.alt = `Uploaded image for ${item.disease_name}`;
      image.loading = "lazy";
      const details = document.createElement("div");
      details.className = "history-details";
      const title = document.createElement("h3");
      title.textContent = item.disease_name;
      const classification = document.createElement("p");
      classification.className = "muted";
      classification.textContent = item.class_name;
      const metadata = document.createElement("p");
      metadata.textContent = `${new Date(item.scanned_at).toLocaleString()} · ${Number(item.confidence).toFixed(1)}% confidence`;
      const filename = document.createElement("p");
      filename.className = "muted filename";
      filename.textContent = item.original_filename;
      details.append(title, classification, metadata, filename);
      const remove = document.createElement("button");
      remove.className = "button button-danger";
      remove.type = "button";
      remove.textContent = "Delete";
      remove.addEventListener("click", () => removeScan(item.id));
      card.append(image, details, remove);
      list.append(card);
    }
  }

  async function loadHistory() {
    const params = new URLSearchParams({ limit: String(state.limit), offset: String(state.offset) });
    const search = byId("historySearch").value.trim();
    const className = byId("classFilter").value;
    if (search) params.set("q", search);
    if (className) params.set("class_name", className);
    const data = await api(`/api/admin/history?${params}`);
    state.total = data.total;
    renderHistory(data.items);
    const first = data.total ? state.offset + 1 : 0;
    const last = Math.min(state.offset + data.items.length, data.total);
    byId("pageLabel").textContent = `${first}–${last} of ${data.total}`;
    byId("previousPage").disabled = state.offset === 0;
    byId("nextPage").disabled = state.offset + state.limit >= state.total;
    message(byId("historyMessage"), "");
  }

  async function removeScan(id) {
    if (!window.confirm("Delete this scan and its stored image?")) return;
    try {
      await api(`/api/admin/history/${id}`, { method: "DELETE" });
      await Promise.all([loadHistory(), loadOverview()]);
    } catch (error) {
      message(byId("historyMessage"), error.message, true);
    }
  }

  byId("loginForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    message(byId("loginMessage"), "Signing in…");
    try {
      const session = await api("/api/admin/login", {
        method: "POST",
        body: JSON.stringify({
          username: byId("username").value,
          password: byId("password").value,
        }),
      });
      byId("password").value = "";
      await showDashboard({ username: session.username, csrf_token: session.csrf_token });
    } catch (error) {
      message(byId("loginMessage"), error.message, true);
    }
  });

  byId("logoutButton").addEventListener("click", async () => {
    try {
      await api("/api/admin/logout", { method: "POST" });
      showLogin();
    } catch (error) {
      message(byId("loginMessage"), error.message, true);
    } finally {
      state.csrfToken = "";
    }
  });

  byId("historySearch").addEventListener("input", () => {
    window.clearTimeout(state.searchTimer);
    state.searchTimer = window.setTimeout(() => {
      state.offset = 0;
      loadHistory().catch((error) => message(byId("historyMessage"), error.message, true));
    }, 250);
  });
  byId("classFilter").addEventListener("change", () => {
    state.offset = 0;
    loadHistory().catch((error) => message(byId("historyMessage"), error.message, true));
  });
  byId("previousPage").addEventListener("click", () => {
    state.offset = Math.max(0, state.offset - state.limit);
    loadHistory().catch((error) => message(byId("historyMessage"), error.message, true));
  });
  byId("nextPage").addEventListener("click", () => {
    state.offset += state.limit;
    loadHistory().catch((error) => message(byId("historyMessage"), error.message, true));
  });
  byId("deleteAllButton").addEventListener("click", async () => {
    if (!window.confirm("Permanently delete every stored scan and image?")) return;
    try {
      await api("/api/admin/history", { method: "DELETE" });
      state.offset = 0;
      await Promise.all([loadHistory(), loadOverview()]);
    } catch (error) {
      message(byId("historyMessage"), error.message, true);
    }
  });

  byId("guideSelect").addEventListener("change", fillGuideForm);
  byId("guideForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const className = byId("guideSelect").value;
    const guide = {
      disease_name: byId("guideName").value,
      symptoms: byId("guideSymptoms").value,
      organic_control: byId("guideOrganic").value,
      chemical_control: byId("guideChemical").value,
      source: byId("guideSource").value,
    };
    try {
      const response = await api(`/api/admin/guides/${encodeURIComponent(className)}`, {
        method: "PUT",
        body: JSON.stringify(guide),
      });
      state.guides = state.guides.map((item) => item.class_name === className ? response.guide : item);
      message(byId("guideMessage"), "Guidance saved.");
    } catch (error) {
      message(byId("guideMessage"), error.message, true);
    }
  });

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((item) => item.classList.toggle("active", item === tab));
      byId("historyPanel").hidden = tab.dataset.panel !== "historyPanel";
      byId("guidesPanel").hidden = tab.dataset.panel !== "guidesPanel";
    });
  });

  api("/api/admin/session")
    .then(showDashboard)
    .catch((error) => {
      showLogin();
      if (error.message && error.message.includes("not configured")) {
        message(byId("loginMessage"), error.message, true);
      }
    });
})();
