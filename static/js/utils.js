/**
 * static/js/utils.js
 *
 * Shared JavaScript utilities used across all pages.
 * Loaded via base.html so every template gets these for free.
 */

// ---------------------------------------------------------------------------
// Clock
// ---------------------------------------------------------------------------

/**
 * Update the #nav-clock element with the current local date and time.
 * Call once on load, then every second via setInterval.
 */
function updateClock() {
  const el = document.getElementById('nav-clock');
  if (!el) return;
  const now = new Date();
  el.textContent =
    now.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) +
    '  ' +
    now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

setInterval(updateClock, 1000);
updateClock();

// ---------------------------------------------------------------------------
// Number formatting
// ---------------------------------------------------------------------------

/**
 * Format a number as a dollar amount with 2 decimal places.
 * Returns '—' for null/undefined.
 *
 * @param {number|null} v
 * @returns {string}
 */
function fmt(v) {
  if (v == null) return '—';
  return '$' + Number(v).toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

/**
 * Format a number with 2 decimal places (no currency symbol).
 * Returns '—' for null/undefined.
 *
 * @param {number|null} v
 * @returns {string}
 */
function fmtN(v) {
  if (v == null) return '—';
  return Number(v).toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

/**
 * Format a number as a percentage with sign and 2 decimal places.
 * Returns '—' for null/undefined.
 *
 * @param {number|null} v  e.g. 0.05 for 5%
 * @returns {string}
 */
function pct(v) {
  if (v == null) return '—';
  return (v >= 0 ? '+' : '') + Number(v).toFixed(2) + '%';
}

/**
 * Return the CSS class name for a positive/negative P&L value.
 *
 * @param {number|null} v
 * @returns {'pos'|'neg'|''}
 */
function pnlCls(v) {
  if (v == null) return '';
  return v >= 0 ? 'pos' : 'neg';
}

// ---------------------------------------------------------------------------
// API fetch wrapper
// ---------------------------------------------------------------------------

/**
 * Fetch JSON from a URL, returning the parsed object or null on error.
 * Errors are logged to the console but not thrown, so callers don't need
 * try/catch for every dashboard widget.
 *
 * @param {string} url
 * @returns {Promise<object|null>}
 */
async function fetchJSON(url) {
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } catch (e) {
    console.error(`fetchJSON(${url}):`, e);
    return null;
  }
}

/**
 * POST JSON to a URL, returning the parsed response or null on error.
 *
 * @param {string} url
 * @param {object} body
 * @returns {Promise<object|null>}
 */
async function postJSON(url, body) {
  try {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } catch (e) {
    console.error(`postJSON(${url}):`, e);
    return null;
  }
}

// ---------------------------------------------------------------------------
// Chat message formatting
// ---------------------------------------------------------------------------

/**
 * Convert a bot response string to HTML.
 * Handles **bold**, newlines, and `inline code`.
 *
 * @param {string} text
 * @returns {string} HTML string
 */
function formatMessage(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>')
    .replace(
      /`(.*?)`/g,
      "<code style='background:var(--border);padding:1px 5px;border-radius:3px;font-family:var(--mono)'>$1</code>"
    );
}
