const fs = require('fs');
const path = require('path');
const express = require('express');

const previousStatic = express.static;

function injectGallery(html) {
  if (!html.includes('data-tab="gallery"')) {
    html = html.replace(
      '<a href="#" class="nav-item" data-tab="settings">',
      `<a href="#" class="nav-item" data-tab="gallery">
          <i class="fa-solid fa-images"></i>
          <span>Event Gallery</span>
        </a>
        <a href="#" class="nav-item" data-tab="settings">`
    );
  }

  if (!html.includes('id="tab-content-gallery"')) {
    const section = `
        <!-- EVENT PHOTO GALLERY -->
        <section id="tab-content-gallery" class="tab-content">
          <div class="card">
            <div class="card-header" style="display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;">
              <div>
                <h2>Event Photo Gallery</h2>
                <p style="margin:6px 0 0;color:var(--text-muted);font-size:.86rem;">Browse captured face events by authorization status.</p>
              </div>
              <div id="gallery-filter" style="display:flex;gap:8px;flex-wrap:wrap;">
                <button class="btn btn-primary gallery-filter-btn" data-gallery-filter="all">All</button>
                <button class="btn btn-secondary gallery-filter-btn" data-gallery-filter="known"><i class="fa-solid fa-user-check"></i> Authorized</button>
                <button class="btn btn-secondary gallery-filter-btn" data-gallery-filter="unknown"><i class="fa-solid fa-user-xmark"></i> Unauthorized</button>
              </div>
            </div>
            <div class="card-body">
              <div id="gallery-summary" style="margin-bottom:16px;color:var(--text-muted);font-size:.88rem;"></div>
              <div id="event-gallery-grid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:16px;"></div>
            </div>
          </div>
        </section>
`;
    html = html.replace('        <!-- ============================================== -->\n        <!-- TAB 4: SETTINGS -->', section + '\n        <!-- ============================================== -->\n        <!-- TAB 4: SETTINGS -->');
  }

  const script = `
<script>
(function () {
  let galleryEvents = [];
  let galleryFilter = 'all';

  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"']/g, function(c) {
      return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
    });
  }

  function renderGallery() {
    const grid = document.getElementById('event-gallery-grid');
    const summary = document.getElementById('gallery-summary');
    if (!grid) return;

    const rows = galleryEvents.filter(function(ev) {
      if (galleryFilter === 'known') return ev.is_known === true;
      if (galleryFilter === 'unknown') return ev.is_known === false;
      return true;
    }).filter(function(ev) { return !!ev.crop_filename; });

    const knownCount = galleryEvents.filter(e => e.is_known === true && e.crop_filename).length;
    const unknownCount = galleryEvents.filter(e => e.is_known === false && e.crop_filename).length;
    if (summary) summary.textContent = rows.length + ' photos shown • ' + knownCount + ' authorized • ' + unknownCount + ' unauthorized';

    if (!rows.length) {
      grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1;"><i class="fa-solid fa-images"></i><p>No photos found for this filter.</p></div>';
      return;
    }

    grid.innerHTML = rows.map(function(ev) {
      const known = ev.is_known === true;
      const dt = ev.timestamp ? new Date(ev.timestamp) : null;
      const date = dt && !isNaN(dt) ? dt.toLocaleDateString() : '';
      const time = dt && !isNaN(dt) ? dt.toLocaleTimeString() : '';
      const name = known ? (ev.person_name || 'Known') : 'Unknown';
      const score = Math.round(Number(ev.score || 0) * 100);
      const photo = '/crops/' + encodeURIComponent(ev.crop_filename);
      return '<div style="background:var(--card-bg);border:1px solid var(--border-color);border-radius:12px;overflow:hidden;">' +
        '<div style="height:180px;background:#111;display:flex;align-items:center;justify-content:center;overflow:hidden;">' +
          '<img src="' + photo + '" alt="' + esc(name) + '" loading="lazy" style="width:100%;height:100%;object-fit:cover;">' +
        '</div>' +
        '<div style="padding:12px;">' +
          '<div style="display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:8px;">' +
            '<strong style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + esc(name) + '</strong>' +
            '<span style="font-size:.72rem;font-weight:700;color:' + (known ? '#22c55e' : '#ef4444') + ';">' + (known ? 'AUTHORIZED' : 'UNAUTHORIZED') + '</span>' +
          '</div>' +
          '<div style="color:var(--text-muted);font-size:.78rem;line-height:1.55;">' +
            '<div><i class="fa-regular fa-calendar"></i> ' + esc(date) + ' &nbsp; <i class="fa-regular fa-clock"></i> ' + esc(time) + '</div>' +
            '<div><i class="fa-solid fa-video"></i> ' + esc(ev.camera_name || 'Manual Upload') + '</div>' +
            '<div>Confidence: ' + score + '%</div>' +
          '</div>' +
        '</div>' +
      '</div>';
    }).join('');
  }

  async function loadGallery() {
    const grid = document.getElementById('event-gallery-grid');
    if (grid) grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1;"><div class="spinner"></div><p>Loading event photos...</p></div>';
    try {
      const res = await fetch('/api/events?limit=1000');
      if (!res.ok) throw new Error('HTTP ' + res.status);
      galleryEvents = await res.json();
      renderGallery();
    } catch (err) {
      if (grid) grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1;"><i class="fa-solid fa-triangle-exclamation"></i><p>Failed to load event photos.</p></div>';
    }
  }

  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.gallery-filter-btn').forEach(function(btn) {
      btn.addEventListener('click', function() {
        galleryFilter = btn.dataset.galleryFilter;
        document.querySelectorAll('.gallery-filter-btn').forEach(function(b) {
          b.classList.toggle('btn-primary', b === btn);
          b.classList.toggle('btn-secondary', b !== btn);
        });
        renderGallery();
      });
    });

    const nav = document.querySelector('[data-tab="gallery"]');
    if (nav) nav.addEventListener('click', function() { setTimeout(loadGallery, 0); });
  });
})();
</script>
`;
  if (!html.includes('function renderGallery()')) html = html.replace('</body>', script + '</body>');
  return html;
}

express.static = function galleryStatic(root, options) {
  const normalStatic = previousStatic(root, options);
  const isPublicRoot = path.basename(path.resolve(root)) === 'public';
  if (!isPublicRoot) return normalStatic;
  return function(req, res, next) {
    if (req.path === '/' || req.path === '/index.html') {
      try {
        const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
        res.type('html').send(injectGallery(html));
        return;
      } catch (err) {
        console.error('[Gallery] UI injection failed:', err.message);
      }
    }
    return normalStatic(req, res, next);
  };
};

console.log('[Gallery] Authorized/unauthorized event gallery enabled.');
