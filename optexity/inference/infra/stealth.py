STEALTH_SCRIPT = """
// Only patch the top-level frame. Cross-origin iframes (reCAPTCHA, Stripe, etc.)
// throw SecurityError on window.top access — we catch that and bail out.
// Without this guard our Object.defineProperty calls break captcha internals.
(function() {
try { if (window.self !== window.top) return; } catch (_) { return; }

// Remove webdriver flag
Object.defineProperty(navigator, 'webdriver', {
  get: () => undefined,
  configurable: true
});

// Add chrome runtime (absent in raw Playwright/CDP contexts)
if (!window.chrome) {
  window.chrome = {
    runtime: {
      connect: () => ({}),
      sendMessage: () => {}
    },
    loadTimes: function() { return {}; },
    csi: function() { return {}; },
    app: {
      isInstalled: false,
      InstallState: { DISABLED: 'disabled', INSTALLED: 'installed', NOT_INSTALLED: 'not_installed' },
      RunningState: { CANNOT_RUN: 'cannot_run', READY_TO_RUN: 'ready_to_run', RUNNING: 'running' }
    }
  };
}

// Make navigator.plugins non-empty (real browsers have 3+ plugins)
try {
  Object.defineProperty(navigator, 'plugins', {
    get: () => {
      const ps = [
        { name: 'Chrome PDF Plugin', description: 'Portable Document Format', filename: 'internal-pdf-viewer', length: 1 },
        { name: 'Chrome PDF Viewer', description: '', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', length: 1 },
        { name: 'Native Client', description: '', filename: 'internal-nacl-plugin', length: 2 }
      ];
      ps[Symbol.iterator] = Array.prototype[Symbol.iterator].bind(ps);
      return ps;
    },
    configurable: true
  });
} catch(_) {}

// Fix navigator.languages (empty in automation = instant flag)
Object.defineProperty(navigator, 'languages', {
  get: () => ['en-US', 'en'],
  configurable: true
});

// Fix Permissions API (broken in headless/CDP contexts)
try {
  const origQuery = window.navigator.permissions.query.bind(navigator.permissions);
  navigator.permissions.query = (params) =>
    params.name === 'notifications'
      ? Promise.resolve({ state: Notification.permission })
      : origQuery(params);
} catch(_) {}

})();
"""
