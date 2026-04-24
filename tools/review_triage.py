#!/usr/bin/env python3
"""
tools/review_triage.py
──────────────────────
Flask grid UI for triaging extracted frames before sending them to CVAT.

Reads:
  - manifest:  <metadata>/frame_manifests/<split>/<batch_id>.csv
  - images:    <dataset>/<split_dir>/images/<batch_id>/*.png
Writes:
  - manifest in place with Triage_Status set to approved/rejected/pending
  - rejected files physically moved to <images>/<batch_id>/_rejected/

So after triage, ``images/<batch_id>/*.png`` (top-level only) is the set ready
for CVAT upload; rejects are quarantined under ``_rejected/`` for record.

Usage
-----
    python tools/review_triage.py \\
        --config configs/VR/wave3_vr.yaml \\
        --split  train \\
        --batch  VR_Wave3_VR_2026-04-24_batch01

    # Then open http://localhost:5000 (use --host 0.0.0.0 for SSH-forwarded review)

Keyboard shortcuts (in browser):
  a / →   approve focused
  x / ←   reject focused
  z       undo focused (back to pending)
  Space   advance focus to next pending
  Ctrl-S  save (also auto-saves on Approve / Reject)
"""

import argparse
import csv
import shutil
import sys
import webbrowser
from pathlib import Path
from threading import Lock

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).parent.parent))
from avistrack.config.loader import load_config


SPLIT_TO_DIR = {"train": "train", "val": "val_tuning", "test": "test_golden"}
CSV_FIELDS = [
    "Frame_Filename",
    "Source_Clip",
    "Original_Video_Path",
    "Frame_Idx",
    "Timestamp",
    "Triage_Status",
]
VALID_STATUS = {"pending", "approved", "rejected"}


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        print(f"❌ Manifest not found: {path}")
        sys.exit(1)
    with open(path, newline="") as f:
        return [
            {k.strip(): (v or "").strip() for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def write_manifest(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in CSV_FIELDS} for r in rows)


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Triage — {batch_id}</title>
<style>
  body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; padding: 12px;
         background: #111; color: #eee; }
  header { position: sticky; top: 0; z-index: 10; background: #111; padding: 8px 0 12px;
           border-bottom: 1px solid #333; display: flex; gap: 16px; align-items: center; }
  .stats { font-size: 14px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
          gap: 8px; margin-top: 12px; }
  .card { background: #1a1a1a; border: 2px solid #333; border-radius: 6px; overflow: hidden;
          cursor: pointer; user-select: none; transition: border-color 0.05s; position: relative; }
  .card.focus { border-color: #4ea1ff; }
  .card.approved { border-color: #4ade80; }
  .card.rejected { border-color: #f87171; opacity: 0.5; }
  .card img { width: 100%; display: block; aspect-ratio: 16/9; object-fit: cover; }
  .card .label { font-size: 11px; padding: 4px 6px; background: rgba(0,0,0,0.7);
                 position: absolute; top: 0; left: 0; right: 0; }
  .card .status { font-size: 12px; padding: 4px 6px; text-align: center; font-weight: bold; }
  .card.pending .status { color: #888; }
  .card.approved .status { color: #4ade80; }
  .card.rejected .status { color: #f87171; }
  button { background: #2a2a2a; color: #eee; border: 1px solid #444; padding: 6px 10px;
           border-radius: 4px; cursor: pointer; font-size: 13px; }
  button:hover { background: #3a3a3a; }
  .help { font-size: 12px; color: #888; }
</style></head>
<body>
<header>
  <div><b>{batch_id}</b></div>
  <div class="stats" id="stats">…</div>
  <button onclick="save(true)">Save & finalize (move rejects)</button>
  <span class="help">a/→ approve · x/← reject · z undo · space next-pending · Ctrl-S save</span>
</header>
<div id="grid" class="grid"></div>
<script>
const FRAMES = {frames_json};
let focusIdx = 0;

function render() {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  let approved = 0, rejected = 0, pending = 0;
  FRAMES.forEach((f, i) => {
    const card = document.createElement('div');
    card.className = 'card ' + f.status + (i === focusIdx ? ' focus' : '');
    card.dataset.idx = i;
    card.innerHTML =
      '<div class="label">' + f.name + '</div>' +
      '<img src="/img/' + encodeURIComponent(f.name) + '" loading="lazy">' +
      '<div class="status">' + f.status.toUpperCase() + '</div>';
    card.onclick = () => { focusIdx = i; render(); };
    grid.appendChild(card);
    if (f.status === 'approved') approved++;
    else if (f.status === 'rejected') rejected++;
    else pending++;
  });
  document.getElementById('stats').textContent =
    `${FRAMES.length} total · ${approved} approved · ${rejected} rejected · ${pending} pending`;
  const focused = grid.children[focusIdx];
  if (focused) focused.scrollIntoView({block: 'nearest'});
}

function setStatus(idx, status) {
  if (idx < 0 || idx >= FRAMES.length) return;
  FRAMES[idx].status = status;
  fetch('/set', { method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name: FRAMES[idx].name, status: status}) });
  render();
}

function nextPending() {
  for (let off = 1; off <= FRAMES.length; off++) {
    const i = (focusIdx + off) % FRAMES.length;
    if (FRAMES[i].status === 'pending') { focusIdx = i; render(); return; }
  }
}

function save(finalize) {
  fetch('/save?finalize=' + (finalize ? '1' : '0'), { method: 'POST' })
    .then(r => r.json())
    .then(j => alert(j.message || 'saved'));
}

document.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.key === 's') { e.preventDefault(); save(false); return; }
  if (e.key === 'a' || e.key === 'ArrowRight') { setStatus(focusIdx, 'approved'); nextPending(); }
  else if (e.key === 'x' || e.key === 'ArrowLeft') { setStatus(focusIdx, 'rejected'); nextPending(); }
  else if (e.key === 'z') { setStatus(focusIdx, 'pending'); }
  else if (e.key === ' ') { e.preventDefault(); nextPending(); }
});
render();
</script></body></html>
"""


def build_app(images_dir: Path, manifest_path: Path, batch_id: str):
    app = Flask(__name__)
    rows = load_manifest(manifest_path)
    by_name = {r["Frame_Filename"]: r for r in rows}
    lock = Lock()

    @app.route("/")
    def index():
        frames_json = (
            "[" + ",".join(
                '{"name":' + repr(r["Frame_Filename"]) + ',"status":' +
                repr(r.get("Triage_Status") or "pending") + "}"
                for r in rows
            ) + "]"
        ).replace("'", '"')
        return (HTML
                .replace("{batch_id}", batch_id)
                .replace("{frames_json}", frames_json))

    @app.route("/img/<path:fname>")
    def img(fname):
        return send_from_directory(images_dir, fname)

    @app.route("/set", methods=["POST"])
    def set_status():
        data = request.get_json()
        name, status = data.get("name"), data.get("status")
        if status not in VALID_STATUS or name not in by_name:
            return jsonify({"error": "bad input"}), 400
        with lock:
            by_name[name]["Triage_Status"] = status
            write_manifest(manifest_path, rows)
        return jsonify({"ok": True})

    @app.route("/save", methods=["POST"])
    def save():
        finalize = request.args.get("finalize") == "1"
        with lock:
            write_manifest(manifest_path, rows)
            if not finalize:
                return jsonify({"message": f"Manifest saved: {manifest_path}"})
            rejected_dir = images_dir / "_rejected"
            rejected_dir.mkdir(exist_ok=True)
            n_moved = 0
            for r in rows:
                if r.get("Triage_Status") == "rejected":
                    src = images_dir / r["Frame_Filename"]
                    if src.exists():
                        shutil.move(str(src), str(rejected_dir / src.name))
                        n_moved += 1
            n_app = sum(1 for r in rows if r.get("Triage_Status") == "approved")
            n_pend = sum(1 for r in rows if r.get("Triage_Status") == "pending")
            return jsonify({"message":
                f"Finalized: moved {n_moved} rejects → _rejected/. "
                f"{n_app} approved (ready for CVAT). {n_pend} still pending."})

    return app


def main():
    ap = argparse.ArgumentParser(description="Flask grid UI for triaging extracted frames.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--batch", required=True, help="batch_id (e.g. VR_Wave3_VR_2026-04-24_batch01)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset_root = cfg.drive.dataset
    metadata_root = cfg.drive.metadata
    images_dir = Path(dataset_root) / SPLIT_TO_DIR[args.split] / "images" / args.batch
    manifest_path = Path(metadata_root) / "frame_manifests" / args.split / f"{args.batch}.csv"

    if not images_dir.exists():
        print(f"❌ Images dir not found: {images_dir}")
        sys.exit(1)
    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        sys.exit(1)

    app = build_app(images_dir, manifest_path, args.batch)
    url = f"http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}"
    print(f"🌐 Triage UI: {url}")
    print(f"   Images: {images_dir}")
    print(f"   Manifest: {manifest_path}")
    if not args.no_browser and args.host == "127.0.0.1":
        webbrowser.open(url)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
