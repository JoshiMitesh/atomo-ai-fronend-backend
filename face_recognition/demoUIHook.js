const express = require('express');

// Keep Demo browser code in public/js/demo.js instead of embedding a second
// template literal inside this Node preload hook. This avoids Node parsing CSS
// selectors from nested browser-script backticks.
const originalSend = express.response.send;

express.response.send = function patchedDemoSend(body) {
  if (
    typeof body === 'string' &&
    body.includes('</body>') &&
    !body.includes('/js/demo.js')
  ) {
    body = body.replace(
      '</body>',
      '<script src="/js/demo.js?v=3"></script>\n</body>'
    );
  }

  return originalSend.call(this, body);
};

console.log('[Demo] Centered Demo UI + reliable card navigation enabled.');
