/* =====================================================================================
 * VestiAI — API client
 * Every backend call goes through here so error handling (including the backend's
 * machine-readable error_code payloads) is consistent everywhere in the UI.
 * ===================================================================================== */
(function () {
  'use strict';
  const { util } = window.VestiAI;

  const ERROR_HINTS = {
    invalid_image: 'The file could not be decoded as an image.',
    unsupported_garment: 'That image is not a usable garment product shot.',
    low_resolution: 'The image is too small — upload at least 160×160 px.',
    garment_not_found: 'That garment is no longer in the closet.',
    dataset_error: 'The dataset could not be prepared.',
    model_unavailable: 'No trained checkpoint is installed yet.',
    checkpoint_corrupt: 'The checkpoint file looks corrupted.',
    cuda_oom: 'The GPU ran out of memory.',
    gpu_unavailable: 'No CUDA GPU detected on this machine.',
    training_error: 'The training run failed to start.',
    training_busy: 'A training run is already active.',
    dependency_missing: 'An optional component is not installed.',
    inference_failed: 'Try-on inference failed.',
    validation_error: 'Some fields were invalid.',
    http_error: 'The server returned an error.',
    internal_error: 'Something went wrong inside VestiAI.',
  };

  class ApiError extends Error {
    constructor(payload, status) {
      const friendly = payload && (payload.user_message || ERROR_HINTS[payload.error_code]) ? null : null;
      super((payload && (payload.user_message || payload.message)) || `HTTP ${status}`);
      this.name = 'ApiError';
      this.status = status;
      this.code = payload && payload.error_code ? payload.error_code : 'unknown';
      this.detail = payload && payload.message ? payload.message : '';
      this.details = (payload && payload.details) || {};
      this.hint = ERROR_HINTS[this.code] || friendly;
      this.payload = payload;
    }
  }

  async function parse(response) {
    const text = await response.text();
    let payload = null;
    try { payload = text ? JSON.parse(text) : null; } catch (e) { payload = { message: text.slice(0, 500) }; }
    if (!response.ok) throw new ApiError(payload, response.status);
    return payload;
  }

  const api = {
    base: '',
    ApiError,

    async request(path, options) {
      const opts = Object.assign({ headers: {} }, options || {});
      if (opts.body && !(opts.body instanceof FormData)) {
        opts.headers['Content-Type'] = 'application/json';
        if (typeof opts.body !== 'string') opts.body = JSON.stringify(opts.body);
      }
      let response;
      try {
        response = await fetch(api.base + path, opts);
      } catch (err) {
        throw new ApiError({ error_code: 'network_error', message: `Network error: ${err.message}`,
          user_message: 'Could not reach the VestiAI backend. Is the server running?' }, 0);
      }
      return parse(response);
    },
    get(path) { return api.request(path, { method: 'GET' }); },
    post(path, body) { return api.request(path, { method: 'POST', body: body === undefined ? {} : body }); },
    patch(path, body) { return api.request(path, { method: 'PATCH', body }); },
    del(path) { return api.request(path, { method: 'DELETE' }); },
    upload(path, formData) { return api.request(path, { method: 'POST', body: formData }); },

    /* ---------------------------------------------------------------- system */
    health: () => api.get('/api/health'),
    status: () => api.get('/api/status'),
    devices: () => api.get('/api/system/devices'),
    components: () => api.get('/api/system/components'),
    disk: () => api.get('/api/system/disk'),
    about: () => api.get('/api/system/about'),
    clearCache: () => api.post('/api/system/cache/clear'),
    sampleCloset: (count, seed) => api.post('/api/system/sample-closet', { count: count || 10, seed: seed || 7 }),

    /* ---------------------------------------------------------------- closet */
    garments: (params) => {
      const q = new URLSearchParams(params || {}).toString();
      return api.get(`/api/garments${q ? `?${q}` : ''}`);
    },
    garment: (key) => api.get(`/api/garments/${key}`),
    garmentStats: () => api.get('/api/garments/stats'),
    uploadGarment(formData) { return api.upload('/api/garments/upload', formData); },
    previewGarment(formData) { return api.upload('/api/garments/preview', formData); },
    updateGarment(key, body) { return api.patch(`/api/garments/${key}`, body); },
    deleteGarment(key) { return api.del(`/api/garments/${key}`); },
    reprocessGarment(key) { return api.post(`/api/garments/${key}/reprocess`); },

    /* ---------------------------------------------------------------- try-on */
    tryOn: (body) => api.post('/api/tryon', body),
    backends: () => api.get('/api/tryon/backends'),
    setBackend: (name) => api.post(`/api/tryon/backend?name=${encodeURIComponent(name)}`),
    reloadModel: () => api.post('/api/tryon/reload'),

    /* ---------------------------------------------------------------- live */
    createSession: (body) => api.post('/api/live/session', body || {}),
    closeSession: (id) => api.del(`/api/live/session/${id}`),
    liveConfig: () => api.get('/api/live/config'),
    setLiveConfig: (body) => api.post('/api/live/config', body),
    liveFrame: (body) => api.post('/api/live/frame', body),

    /* ---------------------------------------------------------------- training */
    trainingStatus: () => api.get('/api/training/status'),
    startTraining: (body) => api.post('/api/training/start', body),
    stopTraining: (jobId) => api.post('/api/training/stop', { job_id: jobId || null }),
    trainingLogs: (lines) => api.get(`/api/training/logs?lines=${lines || 80}`),
    trainingMetrics: () => api.get('/api/training/metrics'),
    trainingValidation: () => api.get('/api/training/validation'),
    trainingCheckpoints: () => api.get('/api/training/checkpoints'),
    trainingCurves: () => api.get('/api/training/curves'),
    trainingEvaluations: () => api.get('/api/training/evaluations'),
    deleteCheckpoint: (name) => api.del(`/api/training/checkpoints/${encodeURIComponent(name)}`),
    promoteCheckpoint: (name) => api.post(`/api/training/checkpoints/${encodeURIComponent(name)}/promote`),
    datasetList: () => api.get('/api/training/dataset/list'),
    datasetGenerate: (body) => api.post('/api/training/dataset/generate', body),
    datasetConvert: (body) => api.post('/api/training/dataset/convert', body),
    datasetValidate: (name) => api.get(`/api/training/dataset/${encodeURIComponent(name)}/validate`),
    datasetStats: (name) => api.get(`/api/training/dataset/${encodeURIComponent(name)}/stats`),

    /* ---------------------------------------------------------------- results */
    results: (limit) => api.get(`/api/results?limit=${limit || 40}`),
    deleteResult: (id) => api.del(`/api/results/${id}`),
    captures: (kind) => api.get(`/api/captures${kind ? `?kind=${kind}` : ''}`),
    capturePhoto: (body) => api.post('/api/captures/photo', body),
    deleteCapture: (id) => api.del(`/api/captures/${id}`),
    saveVideo(formData) { return api.upload('/api/captures/video', formData); },

    /* ---------------------------------------------------------------- style */
    styles: () => api.get('/api/recommendations/styles'),
    recommend: (body) => api.post('/api/recommendations/recommend', body),
    buildOutfit: (body) => api.post('/api/recommendations/outfit', body),
    pairing: (key) => api.get(`/api/recommendations/pairing/${key}`),

    /* ---------------------------------------------------------------- settings */
    settings: () => api.get('/api/settings'),
    saveSettings: (body) => api.patch('/api/settings', body),
  };

  /* Convenience: show a toast for any error, with the backend's friendly message. */
  api.report = function report(err, context) {
    if (err instanceof ApiError) {
      const title = err.hint || 'Request failed';
      const detail = [err.message, err.detail && err.detail !== err.message ? err.detail : '', context ? `(${context})` : '']
        .filter(Boolean).join(' ');
      util.toast(title, detail, 'err');
      console.warn('[VestiAI]', err.code, err.message, err.details);
    } else {
      util.toast('Unexpected error', `${err && err.message ? err.message : err}`, 'err');
      console.error(err);
    }
  };

  window.VestiAI.api = api;
})();
