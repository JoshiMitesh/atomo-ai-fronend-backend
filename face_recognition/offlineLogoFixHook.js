const express = require('express');

const previousSend = express.response.send;

const atomicLogoSvg = '<svg class="atomic-logo-inline" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 736 756" role="img" aria-label="Atomic Vision logo"><g fill="none" stroke="currentColor" stroke-width="30"><ellipse cx="368" cy="407" rx="132" ry="334"/><ellipse cx="368" cy="407" rx="132" ry="334" transform="rotate(60 368 407)"/><ellipse cx="368" cy="407" rx="132" ry="334" transform="rotate(-60 368 407)"/><circle cx="368" cy="75" r="58"/><circle cx="80" cy="575" r="58"/><circle cx="656" cy="575" r="58"/></g></svg>';

const css = `<style id="atomic-offline-logo-fix">
.atomic-logo-wrap{display:flex!important;align-items:center!important;justify-content:center!important;background:transparent!important;box-shadow:none!important;width:42px!important;height:42px!important;flex:0 0 42px!important;color:#fff!important}
.atomic-logo-inline{display:block!important;width:42px!important;height:42px!important;max-width:42px!important;max-height:42px!important;color:#fff!important}
.atomic-brand{white-space:nowrap!important;font-size:1.05rem!important}
@media(max-width:760px){.atomic-logo-wrap{width:38px!important;height:38px!important;flex-basis:38px!important}.atomic-logo-inline{width:38px!important;height:38px!important;max-width:38px!important;max-height:38px!important}}
</style>`;

express.response.send = function atomicOfflineLogoSend(body) {
  if (typeof body === 'string' && /<html[\s>]/i.test(body)) {
    body = body
      .replace(/\s*<link[^>]+href=["']https:\/\/fonts\.googleapis\.com[^>]*>/gi, '')
      .replace(/\s*<link[^>]+href=["']https:\/\/fonts\.gstatic\.com[^>]*>/gi, '')
      .replace(/\s*<link[^>]+rel=["']preconnect["'][^>]+fonts\.(googleapis|gstatic)\.com[^>]*>/gi, '')
      .replace(/\s*<link[^>]+href=["']https:\/\/cdnjs\.cloudflare\.com\/ajax\/libs\/font-awesome[^>]*>/gi, '');

    const logoBlock = `<div class="logo-icon atomic-logo-wrap">${atomicLogoSvg}</div><span class="logo-text atomic-brand">Atomic Vision</span>`;

    body = body.replace(
      /<div class="logo-icon(?: atomic-logo-wrap)?">[\s\S]*?<\/div>\s*<span class="logo-text(?: atomic-brand)?">(?:AURA|Atomic Vision)<\/span>/i,
      logoBlock
    );

    if (!body.includes('id="atomic-offline-logo-fix"')) {
      body = body.replace('</head>', css + '</head>');
    }
  }

  return previousSend.call(this, body);
};

console.log('[Offline] Atomic Vision logo is now fully inline and WAN-independent.');
