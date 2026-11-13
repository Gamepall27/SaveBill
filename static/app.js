"use strict";

/* ================================================================ helpers */

const $ = (sel, root = document) => root.querySelector(sel);
const view = $("#view");

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const eur = new Intl.NumberFormat("de-DE", { style: "currency", currency: "EUR" });
const fmtAmount = (a) => (a == null ? "–" : eur.format(a));
const fmtDate = (iso) => {
  if (!iso) return "";
  const [y, m, d] = iso.split("-");
  return `${d}.${m}.${y}`;
};
const monthLabel = (iso) =>
  new Date(iso + "T12:00:00").toLocaleDateString("de-DE", { month: "long", year: "numeric" });
const todayIso = () => {
  const d = new Date();
  return new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
};
const amountInput = (a) => (a == null ? "" : a.toFixed(2).replace(".", ","));

function parseAmount(str) {
  let s = String(str ?? "").replace(/[€\s]/g, "");
  if (!s) return null;
  if (s.includes(",")) s = s.replace(/\./g, "").replace(",", ".");
  const n = Number(s);
  if (!Number.isFinite(n)) throw new Error("Betrag ist keine gültige Zahl");
  return Math.round(n * 100) / 100;
}

async function api(path, { method = "GET", json, body } = {}) {
  const opts = { method, headers: {} };
  if (json !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(json);
  } else if (body) {
    opts.body = body;
  }
  let r;
  try {
    r = await fetch(path, opts);
  } catch {
    throw new Error("Keine Verbindung zum Server");
  }
  if (!r.ok) {
    let msg = `Fehler ${r.status}`;
    try {
      const j = await r.json();
      if (typeof j.detail === "string") msg = j.detail;
      else if (Array.isArray(j.detail) && j.detail[0]?.msg) msg = j.detail[0].msg;
    } catch {}
    throw new Error(msg);
  }
  return r.json();
}

let toastTimer;
function toast(msg, isError = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.toggle("error", isError);
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), isError ? 4500 : 2500);
}

async function busy(text, fn) {
  const b = $("#busy");
  $("#busy-text").textContent = text;
  b.hidden = false;
  try {
    return await fn();
  } finally {
    b.hidden = true;
  }
}

/** Downscale + re-encode a camera photo in the browser (applies EXIF rotation). */
async function prepareImage(file) {
  try {
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
    const max = 2800;
    const s = Math.min(1, max / Math.max(bmp.width, bmp.height));
    const c = document.createElement("canvas");
    c.width = Math.round(bmp.width * s);
    c.height = Math.round(bmp.height * s);
    c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
    bmp.close?.();
    const blob = await new Promise((res) => c.toBlob(res, "image/jpeg", 0.9));
    return blob || file;
  } catch {
    return file; // server handles it (incl. EXIF)
  }
}

const ICONS = {
  search: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7" fill="none" stroke="currentColor" stroke-width="2"/><path d="m20 20-3.5-3.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
  camera: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 8a2 2 0 0 1 2-2h2l1.5-2h5L16 6h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2z" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linejoin="round"/><circle cx="12" cy="12.5" r="3.5" fill="none" stroke="currentColor" stroke-width="1.9"/></svg>',
  image: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2" fill="none" stroke="currentColor" stroke-width="1.9"/><circle cx="9" cy="10" r="2" fill="currentColor"/><path d="m4 18 5-5 4 4 3-3 4 4" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linejoin="round"/></svg>',
  receipt: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 2h12v20l-3-2-3 2-3-2-3 2z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M9 7h6M9 11h6M9 15h4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
  download: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v11m0 0-4-4m4 4 4-4M5 20h14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
};

/* ================================================================ corner editor */

function createCornerEditor(root, { src, width, height, corners }) {
  let pts = corners.map((p) => [...p]);
  root.innerHTML = `
    <div class="ce-stage">
      <img class="ce-img" alt="Foto der Quittung" draggable="false">
      <svg class="ce-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
        <path class="ce-shade"></path>
        <polygon class="ce-poly"></polygon>
      </svg>
      ${[0, 1, 2, 3].map((i) => `<div class="ce-h" data-i="${i}" role="slider" aria-label="Ecke ${i + 1}" tabindex="0"></div>`).join("")}
      <div class="ce-loupe"></div>
    </div>`;
  const stage = $(".ce-stage", root);
  const img = $(".ce-img", root);
  const poly = $(".ce-poly", root);
  const shade = $(".ce-shade", root);
  const loupe = $(".ce-loupe", root);
  const handles = [...root.querySelectorAll(".ce-h")];
  img.src = src;
  loupe.style.backgroundImage = `url("${src}")`;

  function draw() {
    poly.setAttribute("points", pts.map((p) => p.join(",")).join(" "));
    shade.setAttribute("d", `M0 0H${width}V${height}H0Z M${pts.map((p) => p.join(" ")).join(" L")}Z`);
    handles.forEach((h, i) => {
      h.style.left = `${(pts[i][0] / width) * 100}%`;
      h.style.top = `${(pts[i][1] / height) * 100}%`;
    });
  }

  const ZOOM = 2.5;
  function showLoupe(i) {
    const rect = img.getBoundingClientRect();
    const dx = (pts[i][0] / width) * rect.width;
    const dy = (pts[i][1] / height) * rect.height;
    const size = 120;
    loupe.style.display = "block";
    loupe.style.backgroundSize = `${rect.width * ZOOM}px ${rect.height * ZOOM}px`;
    loupe.style.backgroundPosition = `${-(dx * ZOOM - size / 2)}px ${-(dy * ZOOM - size / 2)}px`;
    loupe.classList.toggle("right", dx < rect.width / 2 && dy < rect.height / 3);
  }

  handles.forEach((h) => {
    const i = Number(h.dataset.i);
    h.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      h.setPointerCapture(e.pointerId);
      h.classList.add("drag");
      showLoupe(i);
    });
    h.addEventListener("pointermove", (e) => {
      if (!h.hasPointerCapture(e.pointerId)) return;
      const rect = img.getBoundingClientRect();
      const x = Math.min(Math.max(e.clientX - rect.left, 0), rect.width);
      const y = Math.min(Math.max(e.clientY - rect.top, 0), rect.height);
      pts[i] = [Math.round((x / rect.width) * width), Math.round((y / rect.height) * height)];
      draw();
      showLoupe(i);
    });
    const end = (e) => {
      if (h.hasPointerCapture(e.pointerId)) h.releasePointerCapture(e.pointerId);
      h.classList.remove("drag");
      loupe.style.display = "none";
    };
    h.addEventListener("pointerup", end);
    h.addEventListener("pointercancel", end);
    h.addEventListener("keydown", (e) => {
      const step = (e.shiftKey ? 20 : 4) * (width / Math.max(img.clientWidth, 1));
      const d = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[e.key];
      if (!d) return;
      e.preventDefault();
      pts[i] = [Math.min(Math.max(pts[i][0] + d[0], 0), width - 1), Math.min(Math.max(pts[i][1] + d[1], 0), height - 1)];
      draw();
    });
  });
  stage.addEventListener("contextmenu", (e) => e.preventDefault());
  draw();

  return {
    get corners() {
      return pts.map((p) => [...p]);
    },
    set(c) {
      pts = c.map((p) => [...p]);
      draw();
    },
  };
}

const fullFrame = (w, h) => [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]];

/* ================================================================ receipt form */

let categoryCache = null;
async function categories() {
  if (!categoryCache) {
    try {
      categoryCache = await api("/api/categories");
    } catch {
      categoryCache = [];
    }
  }
  return categoryCache;
}
const DEFAULT_CATEGORIES = ["Lebensmittel", "Drogerie", "Tanken", "Restaurant", "Baumarkt", "Elektronik", "Kleidung", "Apotheke", "Möbel"];

async function receiptFields(values) {
  const cats = [...new Set([...(await categories()), ...DEFAULT_CATEGORIES])];
  return `
    <label class="field">Name / Titel
      <input name="title" required maxlength="200" value="${esc(values.title)}" placeholder="z. B. Rewe Wocheneinkauf" autocomplete="off">
    </label>
    <div class="row">
      <label class="field">Datum
        <input name="date" type="date" required value="${esc(values.date || todayIso())}">
      </label>
      <label class="field">Betrag (€)
        <input name="amount" inputmode="decimal" value="${esc(amountInput(values.amount))}" placeholder="0,00" autocomplete="off">
      </label>
    </div>
    <label class="field">Kategorie
      <input name="category" list="cat-list" maxlength="100" value="${esc(values.category)}" placeholder="optional" autocomplete="off">
      <datalist id="cat-list">${cats.map((c) => `<option value="${esc(c)}">`).join("")}</datalist>
    </label>
    <label class="field">Notiz
      <textarea name="note" maxlength="5000" placeholder="optional">${esc(values.note)}</textarea>
    </label>`;
}

function readForm(form) {
  const f = new FormData(form);
  return {
    title: f.get("title").trim(),
    date: f.get("date"),
    amount: parseAmount(f.get("amount")),
    category: f.get("category").trim() || null,
    note: f.get("note").trim() || null,
  };
}

/* ================================================================ list view */

const filters = { q: "", from: "", to: "", category: "", period: "all" };

function periodRange(key) {
  const now = new Date();
  const iso = (d) => new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
  const y = now.getFullYear(), m = now.getMonth();
  switch (key) {
    case "month": return [iso(new Date(y, m, 1)), iso(new Date(y, m + 1, 0))];
    case "lastmonth": return [iso(new Date(y, m - 1, 1)), iso(new Date(y, m, 0))];
    case "year": return [iso(new Date(y, 0, 1)), iso(new Date(y, 11, 31))];
    case "lastyear": return [iso(new Date(y - 1, 0, 1)), iso(new Date(y - 1, 11, 31))];
    default: return ["", ""];
  }
}

function filterParams() {
  const p = new URLSearchParams();
  for (const k of ["q", "from", "to", "category"]) if (filters[k]) p.set(k, filters[k]);
  return p.toString();
}

async function renderList() {
  const cats = await categories();
  view.innerHTML = `
    <div class="filters">
      <div class="search">${ICONS.search}<input type="search" name="q" placeholder="Suchen (Name, Notiz, Text auf der Quittung)" value="${esc(filters.q)}" aria-label="Suchen"></div>
      <div class="chips" role="group" aria-label="Zeitraum">
        ${[["all", "Alle"], ["month", "Dieser Monat"], ["lastmonth", "Letzter Monat"], ["year", "Dieses Jahr"], ["lastyear", "Letztes Jahr"], ["custom", "Zeitraum…"]]
          .map(([k, l]) => `<button type="button" class="chip" data-period="${k}" aria-pressed="${filters.period === k}">${l}</button>`).join("")}
      </div>
      <div class="filter-more">
        <label class="field">Von<input type="date" name="from" value="${esc(filters.from)}"></label>
        <label class="field">Bis<input type="date" name="to" value="${esc(filters.to)}"></label>
        <label class="field">Kategorie
          <select name="category">
            <option value="">Alle Kategorien</option>
            ${cats.map((c) => `<option ${c === filters.category ? "selected" : ""}>${esc(c)}</option>`).join("")}
          </select>
        </label>
      </div>
    </div>
    <div class="summary">
      <div class="total muted" id="total">Lade…</div>
      <div class="actions">
        <a class="btn small" id="exp-zip" download>${ICONS.download} ZIP</a>
        <a class="btn small" id="exp-pdf" download>${ICONS.download} PDF</a>
      </div>
    </div>
    <div id="results"></div>`;

  const more = $(".filter-more");
  more.hidden = filters.period !== "custom" && !filters.category;

  let timer;
  $('input[name="q"]').addEventListener("input", (e) => {
    clearTimeout(timer);
    timer = setTimeout(() => { filters.q = e.target.value.trim(); load(); }, 250);
  });
  view.querySelectorAll(".chip").forEach((chip) =>
    chip.addEventListener("click", () => {
      filters.period = chip.dataset.period;
      view.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", c === chip));
      if (filters.period === "custom") {
        more.hidden = false;
      } else {
        [filters.from, filters.to] = periodRange(filters.period);
        $('input[name="from"]').value = filters.from;
        $('input[name="to"]').value = filters.to;
        more.hidden = !filters.category;
        load();
      }
    })
  );
  for (const name of ["from", "to"]) {
    $(`input[name="${name}"]`).addEventListener("change", (e) => { filters[name] = e.target.value; load(); });
  }
  $('select[name="category"]').addEventListener("change", (e) => { filters.category = e.target.value; load(); });

  await load();
}

async function load() {
  const qs = filterParams();
  $("#exp-zip").href = `/api/export.zip${qs ? "?" + qs : ""}`;
  $("#exp-pdf").href = `/api/export.pdf${qs ? "?" + qs : ""}`;
  let data;
  try {
    data = await api(`/api/receipts${qs ? "?" + qs : ""}`);
  } catch (e) {
    $("#results").innerHTML = `<p class="empty">${esc(e.message)}</p>`;
    return;
  }
  const filtered = Boolean(qs);
  $("#total").innerHTML = `${data.count} Quittung${data.count === 1 ? "" : "en"} · <strong>${fmtAmount(data.total)}</strong>`;
  $("#total").classList.remove("muted");
  $("#exp-zip").hidden = $("#exp-pdf").hidden = data.count === 0;

  if (!data.count) {
    $("#results").innerHTML = `<div class="empty">${ICONS.receipt}
      <p>${filtered ? "Keine Quittungen für diesen Filter." : "Noch keine Quittungen gespeichert."}</p>
      ${filtered ? "" : `<a class="btn primary" href="#/scan">${ICONS.camera} Erste Quittung scannen</a>`}</div>`;
    return;
  }
  const groups = new Map();
  for (const r of data.items) {
    const k = r.date.slice(0, 7);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(r);
  }
  $("#results").innerHTML = [...groups.values()].map((items) => {
    const sum = items.reduce((s, r) => s + (r.amount || 0), 0);
    return `<h2 class="month"><span>${esc(monthLabel(items[0].date))}</span><span>${fmtAmount(sum)}</span></h2>
      <div class="grid">${items.map(tile).join("")}</div>`;
  }).join("");
}

const tile = (r) => `
  <a class="tile card" href="#/r/${r.id}">
    <div class="thumb"><img src="${r.thumb_url}" alt="" loading="lazy"></div>
    <div class="meta">
      <span class="title">${esc(r.title)}</span>
      <span class="sub"><span>${fmtDate(r.date)}</span><span class="amount">${fmtAmount(r.amount)}</span></span>
      ${r.category ? `<span class="tag">${esc(r.category)}</span>` : ""}
    </div>
  </a>`;

/* ================================================================ scan view */

const steps = (n) => `<div class="steps" aria-hidden="true">${[1, 2, 3].map((i) => `<span class="${i <= n ? "on" : ""}"></span>`).join("")}</div>`;

function renderScan() {
  view.innerHTML = `
    ${steps(1)}
    <div class="pick">
      <h1>Quittung scannen</h1>
      <p class="muted">Leg die Quittung auf einen dunklen, ruhigen Untergrund und fotografiere sie von oben.</p>
      <label class="btn primary big">${ICONS.camera} Foto aufnehmen
        <input type="file" accept="image/*" capture="environment" hidden data-file>
      </label>
      <label class="btn big">${ICONS.image} Aus Galerie / Datei wählen
        <input type="file" accept="image/*" hidden data-file>
      </label>
    </div>`;
  view.querySelectorAll("[data-file]").forEach((inp) =>
    inp.addEventListener("change", async () => {
      const file = inp.files?.[0];
      if (!file) return;
      try {
        const scan = await busy("Erkenne Quittung…", async () => {
          const blob = await prepareImage(file);
          const fd = new FormData();
          fd.append("file", blob, "receipt.jpg");
          return api("/api/scans", { method: "POST", body: fd });
        });
        renderCorners(scan);
      } catch (e) {
        toast(e.message, true);
        inp.value = "";
      }
    })
  );
}

function renderCorners(scan) {
  view.innerHTML = `
    ${steps(2)}
    <div class="page-head"><h1>Zuschnitt prüfen</h1></div>
    <div class="ce">
      <div id="editor"></div>
      <p class="ce-hint muted">${scan.detected ? "Quittung erkannt. Ziehe die Ecken, falls nötig." : "Keine Quittung erkannt – bitte die Ecken auf die Quittung ziehen."}</p>
      <div class="actions">
        <button class="btn" data-act="retake">Neu aufnehmen</button>
        <button class="btn" data-act="auto">Auto</button>
        <button class="btn" data-act="full">Ganzes Bild</button>
        <button class="btn primary" data-act="next">Zuschneiden →</button>
      </div>
    </div>`;
  const ed = createCornerEditor($("#editor"), { src: scan.image_url, width: scan.width, height: scan.height, corners: scan.corners });
  view.querySelector('[data-act="retake"]').onclick = renderScan;
  view.querySelector('[data-act="auto"]').onclick = () => ed.set(scan.corners);
  view.querySelector('[data-act="full"]').onclick = () => ed.set(fullFrame(scan.width, scan.height));
  view.querySelector('[data-act="next"]').onclick = async () => {
    try {
      const res = await busy("Schneide zu und lese Text…", () =>
        api(`/api/scans/${scan.scan_id}/crop`, { method: "POST", json: { corners: ed.corners } })
      );
      scan.corners = ed.corners;
      renderDetailsStep(scan, res);
    } catch (e) {
      toast(e.message, true);
    }
  };
}

async function renderDetailsStep(scan, res) {
  const s = res.suggestions || {};
  view.innerHTML = `
    ${steps(3)}
    <div class="details-grid">
      <div class="preview"><img src="${res.crop_url}" alt="Zugeschnittene Quittung"></div>
      <form class="stack card pad" id="rf" novalidate>
        <h1>Details</h1>
        ${s.title || s.amount != null ? `<p class="muted" style="margin:0">Vorschläge aus der Texterkennung – bitte prüfen.</p>` : ""}
        ${await receiptFields({ title: s.title || "", date: s.date || todayIso(), amount: s.amount, category: s.category || "" })}
        ${res.ocr_text?.trim() ? `<details class="ocr"><summary>Erkannter Text</summary><pre>${esc(res.ocr_text)}</pre></details>` : ""}
        <div class="actions end">
          <button type="button" class="btn" data-act="back">← Ecken</button>
          <button class="btn primary">Speichern</button>
        </div>
      </form>
    </div>`;
  view.querySelector('[data-act="back"]').onclick = () => renderCorners(scan);
  const form = $("#rf");
  if (!s.title) form.title.focus();
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      const v = readForm(form);
      if (!v.title) throw new Error("Bitte einen Namen eingeben");
      if (!v.date) throw new Error("Bitte ein Datum wählen");
      const rec = await busy("Speichere…", () => api("/api/receipts", { method: "POST", json: { scan_id: scan.scan_id, ...v } }));
      categoryCache = null;
      toast("Quittung gespeichert");
      location.hash = `#/r/${rec.id}`;
    } catch (err) {
      toast(err.message, true);
    }
  });
}

/* ================================================================ detail view */

async function renderDetail(id) {
  let r;
  try {
    r = await api(`/api/receipts/${id}`);
  } catch (e) {
    view.innerHTML = `<a class="back" href="#/">← Alle Quittungen</a><p class="empty">${esc(e.message)}</p>`;
    return;
  }
  view.innerHTML = `
    <a class="back" href="#/">← Alle Quittungen</a>
    <div class="details-grid">
      <a class="preview" href="${r.image_url}" target="_blank" rel="noopener" title="In voller Größe öffnen">
        <img src="${r.image_url}" alt="Quittung ${esc(r.title)}">
      </a>
      <div class="stack">
        <form class="stack card pad" id="rf" novalidate>
          <h1>${esc(r.title)}</h1>
          ${await receiptFields(r)}
          <div class="actions end"><button class="btn primary">Änderungen speichern</button></div>
        </form>
        <div class="card pad stack">
          <div class="actions">
            <a class="btn" href="${r.image_url}&download=1">${ICONS.download} JPG</a>
            <a class="btn" href="/api/receipts/${r.id}/pdf">${ICONS.download} PDF</a>
            <a class="btn" href="#/r/${r.id}/crop">Neu zuschneiden</a>
            <button class="btn danger" data-act="del">Löschen</button>
          </div>
          ${r.ocr_text?.trim() ? `<details class="ocr"><summary>Erkannter Text</summary><pre>${esc(r.ocr_text)}</pre></details>` : ""}
        </div>
      </div>
    </div>`;
  const form = $("#rf");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      const v = readForm(form);
      if (!v.title) throw new Error("Bitte einen Namen eingeben");
      const upd = await api(`/api/receipts/${id}`, { method: "PATCH", json: v });
      categoryCache = null;
      $("h1", form).textContent = upd.title;
      toast("Gespeichert");
    } catch (err) {
      toast(err.message, true);
    }
  });
  view.querySelector('[data-act="del"]').onclick = async () => {
    if (!confirm(`„${r.title}“ wirklich löschen? Das kann nicht rückgängig gemacht werden.`)) return;
    try {
      await api(`/api/receipts/${id}`, { method: "DELETE" });
      categoryCache = null;
      toast("Gelöscht");
      location.hash = "#/";
    } catch (err) {
      toast(err.message, true);
    }
  };
}

async function renderRecrop(id) {
  let r;
  try {
    r = await api(`/api/receipts/${id}`);
  } catch (e) {
    view.innerHTML = `<p class="empty">${esc(e.message)}</p>`;
    return;
  }
  const img = new Image();
  img.src = r.original_url;
  try {
    await img.decode();
  } catch {
    view.innerHTML = `<a class="back" href="#/r/${id}">← Zurück</a><p class="empty">Originalbild nicht verfügbar.</p>`;
    return;
  }
  const w = img.naturalWidth, h = img.naturalHeight;
  view.innerHTML = `
    <a class="back" href="#/r/${id}">← Zurück</a>
    <div class="page-head"><h1>Neu zuschneiden</h1></div>
    <div class="ce">
      <div id="editor"></div>
      <div class="actions">
        <button class="btn" data-act="full">Ganzes Bild</button>
        <button class="btn primary" data-act="save">Übernehmen</button>
      </div>
    </div>`;
  const ed = createCornerEditor($("#editor"), { src: r.original_url, width: w, height: h, corners: r.corners || fullFrame(w, h) });
  view.querySelector('[data-act="full"]').onclick = () => ed.set(fullFrame(w, h));
  view.querySelector('[data-act="save"]').onclick = async () => {
    try {
      await busy("Schneide zu…", () => api(`/api/receipts/${id}/recrop`, { method: "POST", json: { corners: ed.corners } }));
      toast("Zuschnitt aktualisiert");
      location.hash = `#/r/${id}`;
    } catch (e) {
      toast(e.message, true);
    }
  };
}

/* ================================================================ router */

async function route() {
  const hash = location.hash || "#/";
  let m;
  const section = hash.startsWith("#/scan") ? "scan" : hash.startsWith("#/r/") ? "detail" : "list";
  document.querySelectorAll("[data-nav]").forEach((a) => {
    if (a.dataset.nav === section) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  document.body.classList.toggle("no-fab", section !== "list");
  window.scrollTo(0, 0);

  if (hash.startsWith("#/scan")) renderScan();
  else if ((m = hash.match(/^#\/r\/([0-9a-f]{32})\/crop$/))) await renderRecrop(m[1]);
  else if ((m = hash.match(/^#\/r\/([0-9a-f]{32})$/))) await renderDetail(m[1]);
  else await renderList();
}

window.addEventListener("hashchange", route);
route();
