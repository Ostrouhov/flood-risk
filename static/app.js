// Императивный мост HTMX → Leaflet (Этап 4). См. SRS §9.3.
//   - floodrisk.recalc() — собрать сценарий/модель/bbox и POST /ui/predict
//   - htmx:afterSwap на #prediction-data → пересоздать L.imageOverlay
//   - floodrisk.exportRun() — POST /ui/export; download-ready → скачивание

(function () {
  "use strict";

  const mapEl = document.getElementById("map");
  if (!mapEl) return;

  let coverage = null;
  try {
    coverage = JSON.parse(mapEl.dataset.coverage);
  } catch (e) {
    coverage = null;
  }

  const center = coverage
    ? [(coverage[1] + coverage[3]) / 2, (coverage[0] + coverage[2]) / 2]
    : [54.6, 100.7];
  const map = L.map(mapEl).setView(center, 9);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "© OpenStreetMap contributors",
  }).addTo(map);
  if (coverage) {
    map.fitBounds([
      [coverage[1], coverage[0]],
      [coverage[3], coverage[2]],
    ]);
  }

  const state = {
    map,
    currentOverlay: null,
    currentRunId: null,
    attributionOverlay: null,
    mode: "view",
  };

  state.setMode = function (m) {
    state.mode = m;
    window.dispatchEvent(new CustomEvent("mode-changed", { detail: m }));
    // В режиме «Объяснение» отключаем перетаскивание карты: тогда ЛКМ — это клик
    // (а не пан), и точка для объяснения регистрируется надёжно. Курсор — перекрестие.
    if (m === "explain") {
      map.dragging.disable();
      mapEl.classList.add("explain-mode");
    } else {
      map.dragging.enable();
      mapEl.classList.remove("explain-mode");
      if (state.attributionOverlay) {
        map.removeLayer(state.attributionOverlay);
        state.attributionOverlay = null;
      }
    }
  };

  function centerBbox() {
    const [w, s, e, n] = coverage;
    const cw = e - w;
    const ch = n - s;
    return [w + cw * 0.35, s + ch * 0.35, w + cw * 0.6, s + ch * 0.6];
  }
  function viewBbox() {
    const b = map.getBounds();
    return [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
  }
  function selectedBbox() {
    const mode = document.getElementById("bbox-mode").value;
    if (mode === "view" || !coverage) return viewBbox();
    return centerBbox();
  }
  function checkedValue(name) {
    const el = document.querySelector('input[name="' + name + '"]:checked');
    return el ? el.value : "";
  }

  state.recalc = function () {
    const scenario_id = checkedValue("scenario_id");
    const model_version = checkedValue("model_version");
    if (!scenario_id || !model_version) return;
    const bbox = selectedBbox().join(",");
    window.dispatchEvent(new Event("predict-start"));
    htmx
      .ajax("POST", "/ui/predict", {
        target: "#prediction-data",
        swap: "outerHTML",
        values: { scenario_id, model_version, bbox },
      })
      .finally(() => window.dispatchEvent(new Event("predict-end")));
  };

  state.exportRun = function () {
    if (!state.currentRunId) return;
    htmx.ajax("POST", "/ui/export", {
      target: "#export-result",
      swap: "innerHTML",
      values: { run_id: state.currentRunId },
    });
  };

  map.on("click", function (e) {
    if (state.mode !== "explain" || !state.currentRunId) return;
    // индикатор: explain вне Тулуна собирает признаки онлайн и может занять десятки секунд
    const panel = document.getElementById("explain-panel");
    if (panel) panel.innerHTML = '<p class="text-sm text-slate-500">Считаю важность признаков…</p>';
    htmx.ajax("POST", "/ui/explain", {
      target: "#explain-panel",
      swap: "innerHTML",
      values: { run_id: state.currentRunId, lat: e.latlng.lat, lon: e.latlng.lng },
    });
  });

  function bbox2bounds(b) {
    // [S, W, N, E] → Leaflet [[S,W],[N,E]]
    return [
      [b[0], b[1]],
      [b[2], b[3]],
    ];
  }

  document.body.addEventListener("htmx:afterSwap", function (evt) {
    const tid = evt.detail.target && evt.detail.target.id;

    if (tid === "prediction-data") {
      const el = document.getElementById("prediction-data");
      const png = el.dataset.pngUrl;
      const boundsStr = el.dataset.bounds;
      if (state.currentOverlay) {
        map.removeLayer(state.currentOverlay);
        state.currentOverlay = null;
      }
      if (png && boundsStr) {
        const bounds = bbox2bounds(JSON.parse(boundsStr));
        state.currentOverlay = L.imageOverlay(png, bounds, { opacity: 0.7 }).addTo(map);
        map.fitBounds(bounds);
      }
      state.currentRunId = el.dataset.runId || null;
      const btn = document.getElementById("export-btn");
      if (btn) btn.disabled = !state.currentRunId;
      document.getElementById("export-result").innerHTML = "";
    }

    if (tid === "explain-panel") {
      const data = document.getElementById("explanation-data");
      if (!data) return;
      if (state.attributionOverlay) {
        map.removeLayer(state.attributionOverlay);
        state.attributionOverlay = null;
      }
      let layers = [];
      try {
        layers = JSON.parse(data.dataset.attributionLayers || "[]");
      } catch (e) {
        layers = [];
      }
      if (layers.length) {
        const top = layers[0]; // слой важности топ-признака
        state.attributionOverlay = L.imageOverlay(top.png_url, bbox2bounds(top.bounds_wgs84), {
          opacity: 0.6,
        }).addTo(map);
      }
    }
  });

  document.body.addEventListener("download-ready", function (e) {
    const url = e.detail && e.detail.url;
    if (url) window.location = url;
  });

  window.floodrisk = state;
})();
