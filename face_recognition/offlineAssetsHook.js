const express = require('express');

// Atomic Vision must render immediately on a LAN with no WAN connection.
// Strip render-blocking public internet assets (Google Fonts / cdnjs FontAwesome)
// and use local/system fallbacks instead.
const originalSend = express.response.send;

const fallbackCss = `
<style id="atomic-offline-assets">
  html,body,button,input,select,textarea {
    font-family: Inter, "Segoe UI", Roboto, Ubuntu, Arial, sans-serif !important;
  }
  h1,h2,h3,h4,.logo-text {
    font-family: "Segoe UI", Inter, Roboto, Ubuntu, Arial, sans-serif !important;
  }
  /* Minimal offline icon fallback. The UI remains fully usable without FontAwesome. */
  i[class*="fa-"] { font-style: normal; display: inline-block; min-width: 1em; text-align: center; }
  i[class*="fa-"]::before { content: "•"; }
  .fa-video::before { content: "▶" !important; }
  .fa-video-slash::before { content: "▧" !important; }
  .fa-users-gear::before,.fa-user::before { content: "👤" !important; }
  .fa-folder-open::before { content: "▣" !important; }
  .fa-sliders::before { content: "☰" !important; }
  .fa-camera::before { content: "◉" !important; }
  .fa-trash-can::before { content: "×" !important; }
  .fa-square-plus::before,.fa-user-plus::before { content: "+" !important; }
  .fa-ellipsis-vertical::before { content: "⋮" !important; }
  .fa-circle-check::before { content: "✓" !important; }
  .fa-bell-slash::before { content: "○" !important; }
  .fa-chevron-left::before { content: "‹" !important; }
  .fa-chevron-right::before { content: "›" !important; }
  .fa-xmark::before { content: "×" !important; }
  .fa-magnifying-glass::before { content: "⌕" !important; }
</style>`;

express.response.send = function offlineAssetSend(body) {
  if (typeof body === 'string' && /<html[\s>]/i.test(body)) {
    body = body
      .replace(/\s*<link[^>]+href=["']https:\/\/fonts\.googleapis\.com[^>]*>/gi, '')
      .replace(/\s*<link[^>]+href=["']https:\/\/fonts\.gstatic\.com[^>]*>/gi, '')
      .replace(/\s*<link[^>]+rel=["']preconnect["'][^>]+fonts\.(googleapis|gstatic)\.com[^>]*>/gi, '')
      .replace(/\s*<link[^>]+href=["']https:\/\/cdnjs\.cloudflare\.com\/ajax\/libs\/font-awesome[^>]*>/gi, '');

    if (!body.includes('id="atomic-offline-assets"')) {
      body = body.replace('</head>', `${fallbackCss}\n</head>`);
    }
  }
  return originalSend.call(this, body);
};

console.log('[Offline] External fonts/icons disabled; local system fallbacks enabled.');
