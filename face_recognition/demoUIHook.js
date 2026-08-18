const express = require('express');

const originalSend = express.response.send;
express.response.send = function patchedDemoSend(body) {
  if (typeof body === 'string' && body.includes('</body>') && !body.includes('/js/demo.js')) {
    body = body.replace('</body>', '<script src="/js/demo.js?v=8"></script>\n<script src="/js/demoNavFix.js?v=1"></script>\n</body>');
  }
  return originalSend.call(this, body);
};

console.log('[Demo] Real-time bounding boxes + full-frame events + direct sidebar navigation enabled.');
