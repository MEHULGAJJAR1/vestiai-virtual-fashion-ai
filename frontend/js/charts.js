/* =====================================================================================
 * VestiAI — charts (re-exported for script-order clarity)
 * The implementation lives in state.js next to the store; this file exists so the HTML
 * can load charts.js independently and future chart types have a natural home.
 * ===================================================================================== */
(function () {
  'use strict';
  if (!window.VestiAI.charts) {
    console.warn('[VestiAI] charts.js loaded before state.js — charts unavailable.');
  }
})();
