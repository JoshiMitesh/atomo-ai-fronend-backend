const express = require('express');

const originalSend = express.response.send;
express.response.send = function patchedDemoSend(body) {
  if (typeof body === 'string' && body.includes('</body>') && !body.includes('/js/demo.js')) {
    body = body.replace('</body>', '<script src="/js/demo.js?v=7"></script>\n</body>');
  }
  return originalSend.call(this, body);
};

console.log('[Demo] Real-time bounding boxes + full-frame fire event popup UI enabled.');
