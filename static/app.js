// Императивный мост HTMX → Leaflet (Этап 4, расширен Кластером 1). См. SRS §9.3.
//   - выбранная зона = видимый редактируемый прямоугольник на карте (рисуется мышью)
//   - floodrisk.recalc() — собрать сценарий/модель/bbox(зоны) и POST /ui/predict
//   - инспектор точки: клик в режиме «Просмотр» → вероятность в точке (GET /api/runs/{id}/point)
//   - floodrisk.setOpacity() — прозрачность overlay; floodrisk.exportRun() — экспорт ZIP

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

  const MAX_BBOX_KM = 30; // онлайн-кап (см. online_features.MAX_BBOX_KM)

  const center = coverage
    ? [(coverage[1] + coverage[3]) / 2, (coverage[0] + coverage[2]) / 2]
    : [54.6, 100.7];
  const map = L.map(mapEl).setView(center, 9);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "© OpenStreetMap contributors",
  }).addTo(map);

  // Контур валидированного покрытия (A6): пунктир, не перехватывает клики.
  if (coverage) {
    const covBounds = [
      [coverage[1], coverage[0]],
      [coverage[3], coverage[2]],
    ];
    L.rectangle(covBounds, {
      color: "#0ea5e9",
      weight: 1,
      dashArray: "6 4",
      fill: false,
      interactive: false,
    })
      .addTo(map)
      .bindTooltip("Валидированная зона (Тулун-2019)", { sticky: true });
    map.fitBounds(covBounds);
  }

  const state = {
    map,
    currentOverlay: null,
    currentRunId: null,
    attributionOverlay: null,
    mode: "view",
    opacity: 0.7,
    zoneRect: null,
    drawing: false,
  };

  // ── Зона (выбранный bbox) ─────────────────────────────────────────────────
  function defaultZoneBounds() {
    if (!coverage) {
      const b = map.getBounds();
      return L.latLngBounds(
        [b.getSouth(), b.getWest()],
        [b.getNorth(), b.getEast()],
      );
    }
    const [w, s, e, n] = coverage;
    const cw = e - w;
    const ch = n - s;
    return L.latLngBounds([s + ch * 0.35, w + cw * 0.35], [s + ch * 0.6, w + cw * 0.6]);
  }

  function setZone(latLngBounds) {
    const b = L.latLngBounds(latLngBounds);
    if (state.zoneRect) {
      state.zoneRect.setBounds(b);
    } else {
      // interactive:false — клики проходят к карте (нужно для инспектора точки).
      state.zoneRect = L.rectangle(b, {
        color: "#1d4ed8",
        weight: 2,
        fillOpacity: 0.05,
        interactive: false,
      }).addTo(map);
    }
    updateZoneSize();
  }

  function zoneBbox() {
    const b = state.zoneRect.getBounds();
    return [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
  }

  function zoneSizeKm() {
    const [w, s, e, n] = zoneBbox();
    const latMid = (s + n) / 2;
    const hKm = (n - s) * 111.0;
    const wKm = (e - w) * 111.0 * Math.cos((latMid * Math.PI) / 180);
    return [Math.abs(wKm), Math.abs(hKm)];
  }

  function zoneInCoverage() {
    if (!coverage) return false;
    const [w, s, e, n] = zoneBbox();
    return w >= coverage[0] && s >= coverage[1] && e <= coverage[2] && n <= coverage[3];
  }

  // Обновляет подпись размера и гард (C1). Возвращает true, если зона допустима.
  function updateZoneSize() {
    if (!state.zoneRect) return false;
    const [wKm, hKm] = zoneSizeKm();
    const sizeEl = document.getElementById("zone-size");
    if (sizeEl) {
      const where = zoneInCoverage() ? "мозаика" : "онлайн";
      sizeEl.textContent = `≈ ${wKm.toFixed(1)} × ${hKm.toFixed(1)} км · ${where}`;
    }
    const tooBig = Math.max(wKm, hKm) > MAX_BBOX_KM && !zoneInCoverage();
    const warnEl = document.getElementById("zone-warning");
    if (warnEl) {
      if (tooBig) {
        warnEl.textContent = `Слишком большая зона для онлайна (>${MAX_BBOX_KM} км). Уменьшите выделение или выберите зону внутри Тулуна.`;
        warnEl.classList.remove("hidden");
      } else {
        warnEl.classList.add("hidden");
      }
    }
    const recalcBtn = document.getElementById("recalc-btn");
    if (recalcBtn) recalcBtn.disabled = tooBig;
    return !tooBig;
  }

  state.resetZone = function () {
    setZone(defaultZoneBounds());
    if (coverage) {
      map.fitBounds([
        [coverage[1], coverage[0]],
        [coverage[3], coverage[2]],
      ]);
    }
  };

  // ── Рисование зоны мышью (rubber-band, без плагинов) (A1) ──────────────────
  let drawStart = null;
  let drawTemp = null;

  state.armDraw = function () {
    state.drawing = true;
    map.dragging.disable();
    mapEl.classList.add("draw-mode");
    const btn = document.getElementById("draw-btn");
    if (btn) {
      btn.textContent = "Рисуйте…";
      btn.classList.add("bg-slate-200");
    }
  };

  function disarmDraw() {
    state.drawing = false;
    if (state.mode !== "explain") map.dragging.enable();
    mapEl.classList.remove("draw-mode");
    const btn = document.getElementById("draw-btn");
    if (btn) {
      btn.textContent = "Выделить на карте";
      btn.classList.remove("bg-slate-200");
    }
  }

  map.on("mousedown", function (e) {
    if (!state.drawing) return;
    drawStart = e.latlng;
    if (drawTemp) {
      map.removeLayer(drawTemp);
      drawTemp = null;
    }
  });

  map.on("mousemove", function (e) {
    if (!state.drawing || !drawStart) return;
    const b = L.latLngBounds(drawStart, e.latlng);
    if (drawTemp) {
      drawTemp.setBounds(b);
    } else {
      drawTemp = L.rectangle(b, {
        color: "#1d4ed8",
        weight: 2,
        dashArray: "4 3",
        fillOpacity: 0.05,
        interactive: false,
      }).addTo(map);
    }
  });

  map.on("mouseup", function (e) {
    if (!state.drawing || !drawStart) return;
    const b = L.latLngBounds(drawStart, e.latlng);
    drawStart = null;
    if (drawTemp) {
      map.removeLayer(drawTemp);
      drawTemp = null;
    }
    // Отбрасываем «клик без перетаскивания» (вырожденная зона).
    if (b.getNorth() - b.getSouth() < 1e-6 || b.getEast() - b.getWest() < 1e-6) {
      disarmDraw();
      return;
    }
    setZone(b);
    disarmDraw();
  });

  // ── Режим «Просмотр»/«Объяснение» ─────────────────────────────────────────
  state.setMode = function (m) {
    state.mode = m;
    window.dispatchEvent(new CustomEvent("mode-changed", { detail: m }));
    // В режиме «Объяснение» отключаем перетаскивание: тогда ЛКМ — клик (а не пан).
    if (m === "explain") {
      map.dragging.disable();
      mapEl.classList.add("explain-mode");
    } else {
      if (!state.drawing) map.dragging.enable();
      mapEl.classList.remove("explain-mode");
      if (state.attributionOverlay) {
        map.removeLayer(state.attributionOverlay);
        state.attributionOverlay = null;
      }
    }
  };

  state.setOpacity = function (v) {
    state.opacity = Number(v);
    if (state.currentOverlay) state.currentOverlay.setOpacity(state.opacity);
  };

  function checkedValue(name) {
    const el = document.querySelector('input[name="' + name + '"]:checked');
    return el ? el.value : "";
  }

  state.recalc = function () {
    const scenario_id = checkedValue("scenario_id");
    const model_version = checkedValue("model_version");
    if (!scenario_id || !model_version || !state.zoneRect) return;
    if (!updateZoneSize()) return; // гард: зона слишком велика для онлайна
    const bbox = zoneBbox().join(",");
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

  // ── Инспектор точки (A2) ──────────────────────────────────────────────────
  function inspectPoint(latlng) {
    const url = `/api/runs/${state.currentRunId}/point?lat=${latlng.lat}&lon=${latlng.lng}`;
    const popup = L.popup().setLatLng(latlng).setContent("…").openOn(map);
    fetch(url)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((d) => {
        popup.setContent(
          d.in_bounds
            ? `Вероятность затопления<br><span style="font-size:1.15em;font-weight:600">p = ${d.probability.toFixed(2)}</span>`
            : "Вне зоны расчёта",
        );
      })
      .catch(() => popup.setContent("Ошибка запроса"));
  }

  map.on("click", function (e) {
    if (state.drawing) return; // идёт рисование зоны — клик не трактуем
    if (state.mode === "explain") {
      if (!state.currentRunId) return;
      const panel = document.getElementById("explain-panel");
      if (panel) {
        panel.innerHTML =
          '<p class="text-sm text-slate-500">Считаю важность признаков…</p>';
      }
      htmx.ajax("POST", "/ui/explain", {
        target: "#explain-panel",
        swap: "innerHTML",
        values: { run_id: state.currentRunId, lat: e.latlng.lat, lon: e.latlng.lng },
      });
      return;
    }
    // Режим «Просмотр»: инспектор точки (нужен готовый расчёт).
    if (state.currentRunId) inspectPoint(e.latlng);
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
        state.currentOverlay = L.imageOverlay(png, bounds, {
          opacity: state.opacity,
        }).addTo(map);
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

  // Стартовая зона = центр покрытия (или текущий вид вне покрытия).
  setZone(defaultZoneBounds());

  window.floodrisk = state;
})();
