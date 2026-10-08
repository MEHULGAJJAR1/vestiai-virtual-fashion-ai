/* =====================================================================================
 * VestiAI — gesture recognition (mirrors backend/cv/gestures.py)
 *
 * MediaPipe hand landmark indices:
 *   0 wrist · 4 thumb tip · 8 index tip · 12 middle tip · 16 ring tip · 20 pinky tip
 *   chains: mcp = 5/9/13/17, pip = 6/10/14/18
 * ===================================================================================== */
(function () {
  'use strict';
  const { util } = window.VestiAI;

  const FINGERS = {
    index: [5, 6, 8], middle: [9, 10, 12], ring: [13, 14, 16], pinky: [17, 18, 20],
  };
  const ACTIONS = {
    open_palm: 'next_outfit',
    fist: 'previous_outfit',
    thumbs_up: 'capture',
    victory: 'toggle_quality',
    pinch: 'move_garment',
    swipe_left: 'previous_outfit',
    swipe_right: 'next_outfit',
    unknown: 'none',
  };

  function angle(a, b, c) {
    const ba = [a.x - b.x, a.y - b.y]; const bc = [c.x - b.x, c.y - b.y];
    const norm = Math.hypot(ba[0], ba[1]) * Math.hypot(bc[0], bc[1]);
    if (norm < 1e-6) return 180;
    const cosine = util.clamp((ba[0] * bc[0] + ba[1] * bc[1]) / norm, -1, 1);
    return (Math.acos(cosine) * 180) / Math.PI;
  }

  function handScale(lm) { return Math.hypot(lm[9].x - lm[0].x, lm[9].y - lm[0].y) + 1e-6; }

  function fingerExtended(lm, chain) {
    const [mcp, pip, tip] = chain;
    if (angle(lm[mcp], lm[pip], lm[tip]) < 150) return false;
    const wrist = lm[0];
    return Math.hypot(lm[tip].x - wrist.x, lm[tip].y - wrist.y)
      > Math.hypot(lm[pip].x - wrist.x, lm[pip].y - wrist.y) * 1.03;
  }

  function thumbExtended(lm) {
    if (angle(lm[1], lm[2], lm[3]) < 145) return false;
    const palm = [5, 9, 13, 17].reduce((acc, i) => ({ x: acc.x + lm[i].x / 4, y: acc.y + lm[i].y / 4 }), { x: 0, y: 0 });
    return Math.hypot(lm[4].x - palm.x, lm[4].y - palm.y) > Math.hypot(lm[3].x - palm.x, lm[3].y - palm.y);
  }

  function classify(lm) {
    if (!lm || lm.length < 21) return { gesture: 'unknown', confidence: 0, fingers: {} };
    const ext = {};
    Object.entries(FINGERS).forEach(([name, chain]) => { ext[name] = fingerExtended(lm, chain); });
    const thumb = thumbExtended(lm);
    ext.thumb = thumb;
    const count = Object.values(FINGERS).filter((_, i) => ext[Object.keys(FINGERS)[i]]).length;
    const scale = handScale(lm);
    const pinch = Math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y) / scale;

    let gesture = 'unknown'; let confidence = 0.5;
    if (pinch < 0.42 && ext.middle && ext.ring && ext.pinky && !ext.index) { gesture = 'pinch'; confidence = 0.7; }
    else if (thumb && count === 0) { gesture = lm[4].y < lm[0].y ? 'thumbs_up' : 'unknown'; confidence = 0.8; }
    else if (count === 4 && thumb) { gesture = 'open_palm'; confidence = 0.85; }
    else if (count === 0 && !thumb) { gesture = 'fist'; confidence = 0.85; }
    else if (ext.index && ext.middle && !ext.ring && !ext.pinky) { gesture = 'victory'; confidence = 0.8; }
    else if (count === 4 && !thumb) { gesture = 'open_palm'; confidence = 0.65; }
    return { gesture, action: ACTIONS[gesture], confidence, fingers: ext };
  }

  class Controller {
    constructor(options) {
      const opts = options || {};
      this.cooldownMs = opts.cooldownMs === undefined ? 1200 : opts.cooldownMs;
      this.confirmFrames = opts.confirmFrames === undefined ? 3 : opts.confirmFrames;
      this.swipeVelocity = opts.swipeVelocity === undefined ? 0.35 : opts.swipeVelocity;
      this.reset();
    }

    reset() {
      this.candidates = [];
      this.history = [];
      this.lastTrigger = 0;
      this.lastGesture = 'unknown';
      this.lastAction = 'none';
      this.moveDelta = 0;
    }

    /**
     * Feed one frame of hand landmarks.
     * @returns {{gesture:string, action:string, triggered:boolean, confidence:number}}
     */
    update(landmarks, frameWidth, frameHeight) {
      if (!landmarks) { this.candidates = []; this.history = []; return { gesture: 'unknown', action: 'none', triggered: false, confidence: 0 }; }
      const result = classify(landmarks);
      let gesture = result.gesture;

      /* swipe detection from the wrist's normalised x */
      const nx = landmarks[0].x / (frameWidth || 1);
      this.history.push(nx);
      if (this.history.length > 8) this.history.shift();
      if (this.history.length >= 6) {
        const delta = this.history[this.history.length - 1] - this.history[0];
        if (Math.abs(delta) > this.swipeVelocity && (gesture === 'open_palm' || gesture === 'unknown')) {
          gesture = delta > 0 ? 'swipe_right' : 'swipe_left';
          result.confidence = 0.6;
        }
      }

      /* pinch drag: expose the vertical delta for garment repositioning */
      if (gesture === 'pinch') {
        const y = landmarks[8].y / (frameHeight || 1);
        const previous = this.lastPinchY;
        this.moveDelta = previous === undefined ? 0 : (y - previous);
        this.lastPinchY = y;
      } else { this.lastPinchY = undefined; this.moveDelta = 0; }

      this.candidates.push(gesture);
      if (this.candidates.length > this.confirmFrames) this.candidates.shift();
      const stable = this.candidates.length >= this.confirmFrames
        && this.candidates.every((g) => g === this.candidates[0]);

      let triggered = false;
      const now = performance.now();
      if (stable && gesture !== 'unknown' && gesture !== 'none' && (now - this.lastTrigger) > this.cooldownMs) {
        this.lastTrigger = now;
        triggered = ACTIONS[gesture] !== 'none';
        this.lastAction = ACTIONS[gesture];
      }
      this.lastGesture = gesture;
      return {
        gesture,
        action: ACTIONS[gesture] || 'none',
        triggered,
        executed_action: this.lastAction,
        confidence: result.confidence,
        fingers: result.fingers,
        move_delta: this.moveDelta,
        supported: Object.keys(ACTIONS).filter((k) => k !== 'unknown'),
      };
    }
  }

  window.VestiAI.gestures = { classify, Controller, ACTIONS };
})();
