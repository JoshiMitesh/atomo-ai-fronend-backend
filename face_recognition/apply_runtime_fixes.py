#!/usr/bin/env python3
"""Idempotent source migration for the final_demo face-recognition service."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def replace_once(path, old, new, label):
    text = path.read_text()
    if new in text:
        print(f'[fixes] {label}: already applied')
        return
    if old not in text:
        raise SystemExit(f'[fixes] {label}: expected source pattern not found in {path.name}')
    path.write_text(text.replace(old, new, 1))
    print(f'[fixes] {label}: applied')


worker = ROOT / 'face_worker.py'
db = ROOT / 'db.js'

# 1) Improve CCTV face recall. Recognition still has its own >=0.60 quality gate.
replace_once(
    worker,
    'def __init__(self, modelPath: str, confThreshold: float = 0.90, try_npu: bool = True):',
    'def __init__(self, modelPath: str, confThreshold: float = 0.60, try_npu: bool = True):',
    'YuNet detection threshold 0.90 -> 0.60'
)

# 2) More robust multi-template scoring: strongest template plus top-3 consensus.
replace_once(
    worker,
    '''        # Use nearest-neighbor match (highest score among all enrolled templates of this person)\n        best_cand_score = scores[0]\n            \n        cand_scores.append({\n            "person_id": cand_id,\n            "name": cand_name,\n            "score": best_cand_score\n        })''',
    '''        # Robust multi-template score. Keep the strongest view while requiring\n        # support from up to the top 3 enrolled templates. This is less sensitive\n        # to one bad enrollment photo than pure nearest-neighbor matching.\n        top_scores = scores[:min(3, len(scores))]\n        nearest = top_scores[0]\n        consensus = float(np.mean(top_scores))\n        if dis_type == 0:\n            best_cand_score = (0.70 * nearest) + (0.30 * consensus)\n        else:\n            best_cand_score = (0.70 * nearest) + (0.30 * consensus)\n            \n        cand_scores.append({\n            "person_id": cand_id,\n            "name": cand_name,\n            "score": best_cand_score\n        })''',
    'multi-template recognition scoring'
)

# 3) Preserve settings during DB migration/restart.
replace_once(
    db,
    '''    clusters: [],\n    cluster_counter: 0\n  };''',
    '''    clusters: [],\n    cluster_counter: 0,\n    settings: { threshold: 0.50, dis_type: 0 }\n  };''',
    'default persisted settings'
)
replace_once(
    db,
    '''      dbData.clusters = existing.clusters || [];\n      dbData.cluster_counter = typeof existing.cluster_counter === 'number' ''',
    '''      dbData.clusters = existing.clusters || [];\n      dbData.settings = existing.settings || { threshold: 0.50, dis_type: 0 };\n      dbData.cluster_counter = typeof existing.cluster_counter === 'number' ''',
    'settings migration'
)

# 4) Give every cluster photo a stable id so individual promotion/move works.
replace_once(
    db,
    '''      bestCluster.photos.push({\n        filename: cropFilename,''',
    '''      bestCluster.photos.push({\n        id: Date.now().toString(36) + Math.random().toString(36).substr(2, 5),\n        filename: cropFilename,''',
    'existing cluster photo ids'
)
replace_once(
    db,
    '''        photos: [{\n          filename: cropFilename,''',
    '''        photos: [{\n          id: Date.now().toString(36) + Math.random().toString(36).substr(2, 5),\n          filename: cropFilename,''',
    'new cluster photo ids'
)

# 5) Improve unknown clustering: compare representative nearest templates instead
# of averaging every historical photo, which fragments a person as the cluster grows.
replace_once(
    db,
    '''      const avgScore = clusterScores.reduce((sum, s) => sum + s, 0) / clusterScores.length;\n      \n      if (disType === 0) { // Cosine: higher score is better\n        if (avgScore >= threshold && avgScore > bestScore) {\n          bestScore = avgScore;\n          bestCluster = cluster;\n        }\n      } else { // L2: lower score is better\n        if (avgScore <= threshold && avgScore < bestScore) {\n          bestScore = avgScore;\n          bestCluster = cluster;\n        }\n      }''',
    '''      clusterScores.sort((a, b) => disType === 0 ? b - a : a - b);\n      const representativeScores = clusterScores.slice(0, Math.min(3, clusterScores.length));\n      const representativeScore = representativeScores.reduce((sum, s) => sum + s, 0) / representativeScores.length;\n      \n      if (disType === 0) { // Cosine: higher score is better\n        if (representativeScore >= threshold && representativeScore > bestScore) {\n          bestScore = representativeScore;\n          bestCluster = cluster;\n        }\n      } else { // L2: lower score is better\n        if (representativeScore <= threshold && representativeScore < bestScore) {\n          bestScore = representativeScore;\n          bestCluster = cluster;\n        }\n      }''',
    'representative unknown clustering'
)

# 6) Google Sheets export hook. updateEvent is called when recognition finishes.
replace_once(
    db,
    "const path = require('path');",
    "const path = require('path');\nconst sheetsLogger = require('./googleSheetsLogger');",
    'Google Sheets logger import'
)
replace_once(
    db,
    '''      db.events[eventIndex] = { ...db.events[eventIndex], ...updates };\n      writeDB(db);\n      return db.events[eventIndex];''',
    '''      db.events[eventIndex] = { ...db.events[eventIndex], ...updates };\n      writeDB(db);\n      if (updates && updates.recognition_finalized === true) {\n        sheetsLogger.logEvent(db.events[eventIndex]).catch(err => {\n          console.error('[Google Sheets] Failed to export event:', err.message);\n        });\n      }\n      return db.events[eventIndex];''',
    'Google Sheets finalized-event export hook'
)

print('[fixes] all source migrations complete')
