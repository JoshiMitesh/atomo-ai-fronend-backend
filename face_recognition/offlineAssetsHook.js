const express = require('express');

// Atomic Vision must render immediately on a LAN with no WAN connection.
// Remove public internet font/icon styles and replace FontAwesome glyphs with
// self-contained Unicode symbols. Do not depend on any external font download.
const originalSend = express.response.send;

const fallbackCss = `
<style id="atomic-offline-assets">
  html,body,button,input,select,textarea {
    font-family: Inter, "Segoe UI", Roboto, Ubuntu, Arial, sans-serif !important;
  }
  h1,h2,h3,h4,.logo-text {
    font-family: "Segoe UI", Inter, Roboto, Ubuntu, Arial, sans-serif !important;
  }

  /* FontAwesome is intentionally not loaded offline. Force every fa icon to
     use a normal local symbol font so missing FontAwesome cannot make icons blank. */
  i[class*="fa-"] {
    font-family: "Segoe UI Symbol", "Noto Sans Symbols 2", "Noto Sans Symbols", Arial, sans-serif !important;
    font-style: normal !important;
    font-weight: 400 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-width: 1em !important;
    width: 1.15em !important;
    height: 1.15em !important;
    line-height: 1 !important;
    text-align: center !important;
    speak: never;
  }
  i[class*="fa-"]::before {
    font-family: "Segoe UI Symbol", "Noto Sans Symbols 2", "Noto Sans Symbols", Arial, sans-serif !important;
    font-weight: 400 !important;
    display: inline-block !important;
  }

  .fa-video::before { content: "▶" !important; }
  .fa-video-slash::before { content: "⊘" !important; }
  .fa-users-gear::before { content: "⚙" !important; }
  .fa-user::before,.fa-user-plus::before { content: "♙" !important; }
  .fa-folder-open::before { content: "▣" !important; }
  .fa-wand-magic-sparkles::before { content: "✦" !important; }
  .fa-sliders::before { content: "☷" !important; }
  .fa-camera::before { content: "◉" !important; }
  .fa-trash-can::before { content: "×" !important; }
  .fa-square-plus::before { content: "+" !important; }
  .fa-ellipsis-vertical::before { content: "⋮" !important; }
  .fa-circle-check::before { content: "✓" !important; }
  .fa-bell-slash::before { content: "○" !important; }
  .fa-chevron-left::before { content: "‹" !important; }
  .fa-chevron-right::before { content: "›" !important; }
  .fa-xmark::before { content: "×" !important; }
  .fa-magnifying-glass::before { content: "⌕" !important; }
  .fa-cloud-arrow-up::before { content: "↑" !important; }
  .fa-image::before { content: "▧" !important; }
  .fa-circle-info::before { content: "ⓘ" !important; }
  .fa-floppy-disk::before { content: "▣" !important; }
  .fa-network-wired::before { content: "⌁" !important; }
  .fa-file-video::before { content: "▷" !important; }
  .fa-brain-circuit::before { content: "⚛" !important; }
  .fa-circle::before { content: "●" !important; }
  .fa-check::before { content: "✓" !important; }
  .fa-plus::before { content: "+" !important; }
  .fa-minus::before { content: "−" !important; }
  .fa-play::before { content: "▶" !important; }
  .fa-pause::before { content: "Ⅱ" !important; }
  .fa-stop::before { content: "■" !important; }
  .fa-arrow-left::before { content: "←" !important; }
  .fa-arrow-right::before { content: "→" !important; }
  .fa-download::before { content: "↓" !important; }
  .fa-upload::before { content: "↑" !important; }
  .fa-refresh::before,.fa-arrows-rotate::before { content: "↻" !important; }
  .fa-house::before { content: "⌂" !important; }
  .fa-chart-line::before { content: "↗" !important; }
  .fa-database::before { content: "◉" !important; }
  .fa-gear::before,.fa-cog::before { content: "⚙" !important; }

  /* Ensure the pseudo-element itself remains visible even when the original
     stylesheet sets the icon color/font-family for FontAwesome. */
  .fa-solid::before,.fa-regular::before,.fa-brands::before {
    opacity: 1 !important;
    visibility: visible !important;
  }
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

console.log('[Offline] External fonts/icons disabled; local Unicode icon fallbacks enabled.');
