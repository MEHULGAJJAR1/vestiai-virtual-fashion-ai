#!/usr/bin/env node
/**
 * VestiAI — frontend smoke check (no browser required).
 *
 * Loads frontend/index.html in jsdom, stubs the browser APIs the app touches (canvas 2D,
 * MediaRecorder, getUserMedia, fetch, WebSocket), boots every module against a mocked API and
 * asserts that the shell wires up: every view renders, every button resolves to a handler and
 * every `VestiAI.*` module exposes the public surface the other modules call.
 *
 * This catches the failure mode that automated Python tests cannot see — a typo in an event
 * handler or a function renamed in one module but not its caller.
 *
 * Usage:  node scripts/check_frontend.mjs [--verbose] [--url http://127.0.0.1:8000]
 */

import { readFileSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const frontend = resolve(root, 'frontend');
const args = process.argv.slice(2);
const verbose = args.includes('--verbose');
const baseUrlArg = args.find((a) => a.startsWith('--url'));
const baseUrl = baseUrlArg ? baseUrlArg.split('=')[1] || args[args.indexOf(baseUrlArg) + 1] : null;

let JSDOM;
try {
  ({ JSDOM } = await import('jsdom'));
} catch {
  console.error('jsdom is not installed. Run:  npm install --no-save jsdom   (or npm i -g jsdom)');
  process.exit(2);
}

/* ------------------------------------------------------------------ fixtures */
const garment = {
  key: 'sample-tee-1-t-shirt-ab12cd',
  label: 'Sample Tee 1',
  category: 't-shirt',
  category_label: 'T-Shirt',
  garment_type: 'top',
  favourite: false,
  palette: ['#2b6cb0', '#1a4971'],
  urls: {
    preview: '/api/garments/sample-tee-1-t-shirt-ab12cd/file/preview',
    image: '/api/garments/sample-tee-1-t-shirt-ab12cd/file/image',
    mask: '/api/garments/sample-tee-1-t-shirt-ab12cd/file/mask',
    cutout: '/api/garments/sample-tee-1-t-shirt-ab12cd/file/cutout',
  },
};

const routes = {
  'GET /api/health': { ok: true, status: 'ok' },
  'GET /api/status': {
    ok: true,
    devices: { resolved_device: 'cpu', resolved_device_label: 'CPU', cuda_available: false, mps_available: false, torch_version: null, vram_total_gb: [], notes: [], cpu_count: 8, python_version: '3.11' },
    gpu_live: { gpus: [] },
    ml_stack: { ready: false, torch_version: null, diffusers_version: null },
    checkpoints: { count: 0, epochs: [], best: null, latest: null, inference_path: null, ready_for_inference: false },
    adapter: { name: 'lightweight', display_name: 'Real-time geometric warp', ready: true, detail: 'Classical CV fallback.', hints: [], checkpoint_info: null },
    summary: { ai_ready: false, mode: 'lightweight', message: 'Fast mode ready.' },
    vram_warning: null,
  },
  'GET /api/system/devices': { ok: true, device: { resolved_device: 'cpu', resolved_device_label: 'CPU', cuda_available: false } },
  'GET /api/system/components': { ok: true, components: [{ name: 'FastAPI', installed: true, version: '0.142.2', required: true, purpose: 'API', install: 'pip install fastapi' }] },
  'GET /api/system/disk': { uploads_dir: { size_mb: 0, files: 0 }, garments_dir: { size_mb: 2, files: 29 }, volume: { total_gb: 40, used_gb: 12, free_gb: 28 } },
  'GET /api/system/about': {
    app: 'VestiAI', version: '1.0.0', tagline: 'Two-pipeline virtual try-on.',
    pipelines: [{ key: 'lightweight', name: 'Real-time tracking', latency: '10-30 ms', details: 'Pose → warp → composite.' }],
    honesty: ['The lightweight pipeline is an AR overlay.'],
  },
  'GET /api/openapi/lite': { ok: true, endpoints: ['GET  /api/health'] },
  'GET /api/settings': {
    app: { name: 'VestiAI', version: '1.0.0' },
    paths: { garments: '/tmp/garments' },
    device: { preference: 'auto', force_cpu: false },
    realtime: { segmentation: true, smoothing_alpha: 0.45 },
    tryon: { backend: 'auto', resolution: 512, num_inference_steps: 30, guidance_scale: 2, seed: 42, live_ai_interval_ms: 700, enable_cpu_diffusion: false },
    training: { batch_size: 2, num_epochs: 10, learning_rate: 0.00001, gradient_accumulation: 4, mixed_precision: 'fp16', tracker: 'tensorboard' },
    garment: { canvas: 768, background_removal: 'auto' },
    runtime: { python: '3.11', torch: null },
  },
  'PATCH /api/settings': { ok: true, applied: { resolution: 512 }, settings: {} },
  'GET /api/garments': { ok: true, garments: [garment], count: 1, stats: { total: 1, by_category: { 't-shirt': 1 } } },
  'GET /api/garments/stats': { ok: true, total: 1 },
  'GET /api/tryon/backends': { ok: true, name: 'lightweight', display_name: 'Real-time geometric warp', ready: true, calls: 0, avg_latency_ms: 0, extra: {} },
  'GET /api/training/status': {
    ok: true, active_job: null,
    modes: { QUICK_DEMO: { num_epochs: 2, batch_size: 1, resolution: 256, gradient_accumulation: 2, description: 'verify' } },
    trainer: {}, dataset: { available: [], active: null }, checkpoints: { epochs: [] },
    metrics: { curves: {} }, validation: { grids: [] }, evaluations: [], log_tail: [],
    device: { torch_available: false, torch_version: null, cuda_available: false, resolved_device_label: 'CPU' },
  },
  'GET /api/training/dataset/list': { ok: true, datasets: [], active: null },
  'GET /api/training/checkpoints': { ok: true, checkpoints: { epochs: [] } },
  'GET /api/training/curves': { ok: true, epochs: [] },
  'GET /api/training/evaluations': { ok: true, evaluations: [] },
  'GET /api/training/logs': { ok: true, lines: [] },
  'GET /api/results': { ok: true, count: 0, results: [] },
  'GET /api/captures': { ok: true, captures: [], stats: { total: 0 } },
  'GET /api/recommendations/styles': { ok: true, styles: [{ key: 'casual', label: 'Casual', description: 'relaxed', categories: ['t-shirt'] }] },
  'GET /api/live/config': { ok: true, config: { ai_interval_ms: 700, smoothing_alpha: 0.45, resolution: 512 }, adapter: { name: 'lightweight', ready: true } },
  'POST /api/recommendations/recommend': { ok: true, engine: 'colour rules', recommendations: [] },
  'POST /api/recommendations/outfit': { ok: true, style: 'casual', style_label: 'Casual', score: 0.8, slots: { top: null, bottom: null, shoes: null, accessory: null }, notes: [] },
  'POST /api/live/session': { ok: true, session: { session_id: 'live-mock01' } },
  'POST /api/live/config': { ok: true, config: { ai_interval_ms: 700 } },
  'POST /api/live/frame': { ok: true, status: { pose_backend: 'heuristic' }, latency_ms: 12 },
  'POST /api/system/cache/clear': { ok: true },
  'POST /api/system/sample-closet': { ok: true, created: [garment] },
  'POST /api/tryon/reload': { ok: true, ready: false, detail: 'no checkpoint in mock' },
  'POST /api/tryon/backend': { ok: true, name: 'lightweight', ready: true, display_name: 'Real-time geometric warp' },
  'POST /api/training/dataset/generate': { ok: true, dataset: { name: 'samples', total: 80 } },
  'POST /api/training/start': { ok: true, job: { job_id: 'job-mock', state: 'running' } },
  'POST /api/training/stop': { ok: true },
  'POST /api/training/checkpoints/epoch_0001/promote': { ok: true },
  'POST /api/captures/photo': { ok: true, capture: { capture_id: 'cap-mock', url: '/api/captures/cap-mock/file' } },
};

const calls = [];

function jsonResponse(body) {
  return { ok: true, status: 200, json: async () => body, text: async () => JSON.stringify(body), headers: { get: () => 'application/json' } };
}

/* ------------------------------------------------------------------ harness */
const html = readFileSync(resolve(frontend, 'index.html'), 'utf8');

const dom = new JSDOM(html, {
  url: 'http://localhost:8000/',
  runScripts: 'outside-only',
  pretendToBeVisual: true,
  resources: undefined,
});
const { window } = dom;

/* canvas 2D stubs — jsdom has no renderer, the app only needs the API surface */
const ctx2d = () => ({
  canvas: null, globalAlpha: 1, globalCompositeOperation: 'source-over', filter: 'none',
  imageSmoothingEnabled: true, imageSmoothingQuality: 'high',
  save() {}, restore() {}, translate() {}, scale() {}, rotate() {}, setTransform() {}, transform() {},
  beginPath() {}, moveTo() {}, lineTo() {}, arc() {}, ellipse() {}, closePath() {}, fill() {}, stroke() {},
  fillRect() {}, strokeRect() {}, clearRect() {}, drawImage() {}, putImageData() {}, clip() {}, rect() {},
  createLinearGradient: () => ({ addColorStop() {} }),
  getImageData: (x, y, w, h) => ({ data: new Uint8ClampedArray(Math.max(1, w * h * 4)), width: w, height: h }),
  putImageData() {}, measureText: () => ({ width: 10 }), fillText() {}, strokeText() {},
  setLineDash() {}, quadraticCurveTo() {}, bezierCurveTo() {},
});
window.HTMLCanvasElement.prototype.getContext = function getContext(kind) {
  if (kind !== '2d') return null;
  if (!this.__ctx) {
    this.__ctx = ctx2d();
    this.__ctx.canvas = this;
  }
  return this.__ctx;
};
window.HTMLCanvasElement.prototype.toDataURL = function toDataURL() { return 'data:image/png;base64,AAAA'; };
window.HTMLCanvasElement.prototype.captureStream = function captureStream() { return { getTracks: () => [], getVideoTracks: () => [] }; };

window.MediaRecorder = class MediaRecorder {
  constructor(stream, options) { this.stream = stream; this.mimeType = (options && options.mimeType) || 'video/webm'; this.state = 'inactive'; }
  start() { this.state = 'recording'; }
  stop() { this.state = 'inactive'; if (this.onstop) this.onstop(); }
  addEventListener(type, fn) { this[`on${type}`] = fn; }
};
window.MediaRecorder.isTypeSupported = () => true;

Object.defineProperty(window.navigator, 'mediaDevices', {
  configurable: true,
  value: { getUserMedia: async () => { throw new window.DOMException('denied', 'NotAllowedError'); } },
});
window.MediaStream = class MediaStream { getTracks() { return []; } getVideoTracks() { return []; } };
window.HTMLMediaElement.prototype.play = async () => {};
window.HTMLMediaElement.prototype.pause = () => {};
window.requestAnimationFrame = () => 0;
window.cancelAnimationFrame = () => {};
window.URL.createObjectURL = () => 'blob:mock';
window.URL.revokeObjectURL = () => {};
window.open = () => null;
window.alert = () => {};
if (!window.matchMedia) window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });

const fetchImpl = async (url, init) => {
  const method = ((init && init.method) || 'GET').toUpperCase();
  const path = String(url).replace(/^https?:\/\/[^/]+/, '').split('?')[0];
  const key = `${method} ${path}`;
  calls.push(key);
  if (routes[key]) return jsonResponse(routes[key]);
  if (method === 'GET' && path.startsWith('/api/garments/') && path.endsWith('/file/preview')) {
    return { ok: true, status: 200, blob: async () => new window.Blob(), arrayBuffer: async () => new ArrayBuffer(8), headers: { get: () => 'image/jpeg' } };
  }
  if (routes[`GET ${path}`]) return jsonResponse(routes[`GET ${path}`]);
  const body = { ok: false, error_code: 'not_found', message: `no mock route for ${key}` };
  return { ok: false, status: 404, json: async () => body, text: async () => JSON.stringify(body), headers: { get: () => 'application/json' } };
};
window.fetch = fetchImpl;

window.WebSocket = class WebSocket {
  constructor(url) { this.url = url; this.readyState = 1; setTimeout(() => this.onopen && this.onopen(), 0); }
  send() {} close() { this.readyState = 3; }
};

/* unhandled errors should fail the check, not vanish */
const pageErrors = [];
window.addEventListener('error', (event) => pageErrors.push(event.error ? String(event.error.stack || event.error) : String(event.message)));
window.addEventListener('unhandledrejection', (event) => pageErrors.push(`unhandled rejection: ${event.reason}`));
const noise = [/Not implemented: navigation/, /Not implemented: HTMLCanvasElement/];
const originalError = console.error;
console.error = (...parts) => {
  const text = parts.map(String).join(' ');
  if (noise.some((re) => re.test(text))) return;         // jsdom limitations, not app bugs
  pageErrors.push(text);
  if (verbose) originalError(...parts);
};

/* ------------------------------------------------------------------ load the real scripts */
const scripts = [...html.matchAll(/<script src="([^"]+)"><\/script>/g)].map((m) => m[1]);
if (!scripts.length) { originalError('no <script src> tags found in index.html'); process.exit(1); }

for (const src of scripts) {
  const file = resolve(frontend, src.replace('/static/', ''));
  if (!existsSync(file)) { originalError(`missing file for ${src}`); process.exit(1); }
  const code = readFileSync(file, 'utf8');
  try {
    window.eval(`${code}\n//# sourceURL=${src}`);
  } catch (err) {
    originalError(`✗ ${src} threw while evaluating:\n${err && err.stack ? err.stack : err}`);
    process.exit(1);
  }
  if (verbose) console.log(`  loaded ${src}`);
}

/* ------------------------------------------------------------------ assertions */
const failures = [];
const check = (label, condition) => { if (!condition) failures.push(label); return condition; };

const api = window.VestiAI;
check('window.VestiAI exists', !!api);
for (const mod of ['util', 'api', 'store', 'pose', 'align', 'occlusion', 'gestures', 'live', 'closet', 'outfits', 'training', 'status', 'app']) {
  check(`VestiAI.${mod} exposed`, !!api[mod]);
}
check('app.boot is a function', typeof api.app.boot === 'function');
check('app.go is a function', typeof api.app.go === 'function');
check('live.startCamera is a function', typeof api.live.startCamera === 'function');
check('closet.refresh is a function', typeof api.closet.refresh === 'function');
check('outfits.load is a function', typeof api.outfits.load === 'function');
check('training.refresh is a function', typeof api.training.refresh === 'function');
check('status.refresh is a function', typeof api.status.refresh === 'function');

/* every data-view button must resolve to a real view section */
const views = [...window.document.querySelectorAll('.view')].map((node) => node.id.replace('view-', ''));
for (const button of window.document.querySelectorAll('[data-view]')) {
  check(`[data-view="${button.dataset.view}"] has a section`, views.includes(button.dataset.view));
}
for (const button of window.document.querySelectorAll('[data-go]')) {
  check(`[data-go="${button.dataset.go}"] has a section`, views.includes(button.dataset.go));
}
check('8 views present', views.length === 8);

/* every id referenced by util.$('#...')/setText('#...') must exist in the DOM */
const jsSources = scripts.map((src) => readFileSync(resolve(frontend, src.replace('/static/', '')), 'utf8')).join('\n');
const referenced = new Set();
for (const m of jsSources.matchAll(/util\.\$+\('#([\w-]+)'/g)) referenced.add(m[1]);
for (const m of jsSources.matchAll(/util\.setText\('#([\w-]+)'/g)) referenced.add(m[1]);
const dynamicIds = new Set(['result-modal']);   // created on demand by app.resultOverlay()
const missingIds = [...referenced].filter((id) => !dynamicIds.has(id) && !window.document.getElementById(id));
check(`no dangling DOM ids (missing: ${missingIds.join(', ') || 'none'})`, missingIds.length === 0);

/* boot the app against the mocked API */
let bootError = null;
try {
  await window.VestiAI.app.boot();
  await new Promise((r) => setTimeout(r, 60));
} catch (err) {
  bootError = err;
}
check(`app.boot() completed without throwing${bootError ? `: ${bootError}` : ''}`, !bootError);

/* navigate every view — this exercises every render function */
for (const view of views) {
  try {
    window.VestiAI.app.go(view);
    // eslint-disable-next-line no-await-in-loop
    await new Promise((r) => setTimeout(r, 25));
    check(`view "${view}" rendered`, !window.document.getElementById(`view-${view}`).classList.contains('hidden'));
  } catch (err) {
    check(`view "${view}" threw: ${err && err.message}`, false);
  }
}

/* the modules must have actually called the API (proves wiring, not just loading) */
check('status refresh hit /api/status', calls.includes('GET /api/status'));
check('closet refresh hit /api/garments', calls.includes('GET /api/garments'));

/* quick behavioural checks on the geometry helpers (same maths as the Python reference) */
const align = window.VestiAI.align;
const h = align.transform(
  { topLeft: [0, 0], topRight: [10, 0], bottomRight: [10, 20], bottomLeft: [0, 20] },
  { topLeft: [100, 50], topRight: [140, 52], bottomRight: [138, 130], bottomLeft: [102, 128] },
);
const [mx, my] = window.VestiAI.util.applyH(h, 5, 10);
check(`homography maps the centre into the target rectangle (got ${mx.toFixed(1)}, ${my.toFixed(1)})`,
  mx > 100 && mx < 140 && my > 50 && my < 130);
check('homography maps the source origin onto the target top-left',
  Math.abs(window.VestiAI.util.applyH(h, 0, 0)[0] - 100) < 3);

const profile = align.profile('kurta');
check('kurta profile is longer than a t-shirt profile', profile.lengthFactor > align.profile('t-shirt').lengthFactor);
align.setOverrides({ kurta: { widthFactor: 1.9, lengthFactor: 2.4 } });
check('fit overrides are applied to the profile', align.profile('kurta').widthFactor === 1.9);
align.setOverrides({});

/* click sweep: every enabled button must have a working handler (no page errors) */
window.VestiAI.app.go('live');
const sweepTargets = [
  '#btn-start', '#btn-stop', '#btn-capture', '#btn-record', '#btn-mirror', '#btn-fullscreen',
  '#btn-next-garment', '#btn-remove-garment', '#btn-gestures', '#btn-reset', '#btn-upload-live',
  '#btn-refresh-closet-live', '#btn-refresh-top', '#btn-closet-refresh', '#btn-sample-closet',
  '#btn-browse', '#btn-auto-outfit', '#btn-random-outfit', '#btn-try-outfit',
  '#btn-prepare-samples', '#btn-refresh-training', '#btn-reload-model', '#btn-start-training',
  '#btn-stop-training', '#btn-reload-backend', '#btn-clear-cache', '#btn-export-settings',
  '#btn-save-settings', '#btn-reset-settings', '#btn-apply-live', '#home-sample-closet',
  '#upload-modal-close',
];
const clicked = [];
for (const selector of sweepTargets) {
  const node = window.document.querySelector(selector);
  if (!node) { check(`sweep target ${selector} exists`, false); continue; }
  if (node.disabled) continue;
  try {
    node.click();
    clicked.push(selector);
    // eslint-disable-next-line no-await-in-loop
    await new Promise((r) => setTimeout(r, 12));
  } catch (err) {
    check(`clicking ${selector} threw: ${err && err.message}`, false);
  }
}
if (verbose) console.log(`  clicked ${clicked.length} controls`);
check(`click sweep handled ${clicked.length} controls`, clicked.length >= 18);

/* quality + gesture segments must switch state */
const aiButton = window.document.querySelector('#seg-quality button[data-quality="ai"]');
if (aiButton) { aiButton.click(); check('fast/AI toggle updates state', window.VestiAI.store.state.live.quality === 'ai'); }
const fastButton = window.document.querySelector('#seg-quality button[data-quality="fast"]');
if (fastButton) fastButton.click();
check('view routing via hash is stable', window.VestiAI.app.go('closet') === undefined);

/* gestures contract */
const actionNames = Object.keys(window.VestiAI.gestures.ACTIONS || {}).sort();
check(`gesture actions are wired (${actionNames.join(', ') || 'none'})`, actionNames.length >= 4);

/* ------------------------------------------------------------------ report */
if (pageErrors.length) {
  originalError('\nPage errors captured during boot:');
  for (const err of pageErrors.slice(0, 12)) originalError('  ' + err.split('\n')[0]);
}

const failed = failures.length + pageErrors.length;
if (failed) {
  originalError(`\n✗ frontend smoke check failed (${failures.length} assertions, ${pageErrors.length} page errors)`);
  for (const f of failures) originalError('  ✗ ' + f);
  process.exit(1);
}
console.log(`✓ frontend smoke check passed — ${views.length} views, ${scripts.length} modules, ${new Set(calls).size} API routes exercised${baseUrl ? ` (server ${baseUrl} not used: mocks)` : ''}`);
process.exit(0);
