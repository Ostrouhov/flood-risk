// Императивный мост HTMX → Leaflet. См. SRS §9.3.
// На Этапе 0: инициализация карты с базовой OSM-подложкой.
// На Этапах 3–5 здесь появятся:
//   - htmx:afterSwap на #prediction-data → обновление imageOverlay
//   - htmx:afterSwap на #explanation-data → слои важности признаков
//   - map.on('click') при mode === 'explain' → htmx.ajax('/ui/explain')
//   - HX-Trigger: download-ready → window.location = ...

(function () {
  "use strict";

  const DEFAULT_CENTER = [55.75, 37.62]; // Москва-плейсхолдер; перепинить после OQ-1/OQ-9
  const DEFAULT_ZOOM = 10;

  const mapEl = document.getElementById("map");
  if (!mapEl) return;

  const map = L.map(mapEl).setView(DEFAULT_CENTER, DEFAULT_ZOOM);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "© OpenStreetMap contributors",
  }).addTo(map);

  // Экспортируем для последующих этапов и для отладки в DevTools.
  window.floodrisk = {
    map,
    currentOverlay: null,
    currentAttributionLayers: [],
  };
})();
