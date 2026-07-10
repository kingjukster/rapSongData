#!/usr/bin/env python3
"""Build a self-contained HTML app for manual ranking auto-judge candidates."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_REPORT_DIR = Path("reports/qwen3_4b_base_12line_v1_auto_quality_judge")
DISAGREEMENT_ID_RE = re.compile(r"^##\s+\d+\.\s+([A-Za-z0-9_-]+)\s*$", re.M)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--source",
        choices=["disagreements", "auto_keep", "needs_review", "auto_reject"],
        default="disagreements",
        help="Review set to build. Bucket sources are selected from judged_candidates.jsonl.",
    )
    parser.add_argument("--disagreements-md", type=Path, default=None)
    parser.add_argument("--judged-jsonl", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--export-prefix", default=None, help="Download filename prefix for browser exports.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(payload)
    return rows


def disagreement_ids(path: Path) -> list[str]:
    return DISAGREEMENT_ID_RE.findall(path.read_text(encoding="utf-8", errors="replace"))


def candidate_payload(row: dict[str, Any], rank: int) -> dict[str, Any]:
    judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
    structural = row.get("structural_metrics") if isinstance(row.get("structural_metrics"), dict) else {}
    dimensions = row.get("quality_dimensions") if isinstance(row.get("quality_dimensions"), dict) else {}
    line_stats = row.get("line_length_stats") if isinstance(row.get("line_length_stats"), dict) else {}
    rhyme = row.get("rhyme_metrics") if isinstance(row.get("rhyme_metrics"), dict) else {}
    return {
        "candidate_id": row.get("candidate_id") or row.get("row_id"),
        "initial_rank": rank,
        "prompt": row.get("prompt"),
        "lyrics": row.get("lyrics") or row.get("generated_text"),
        "theme": row.get("theme"),
        "prompt_family": row.get("prompt_family"),
        "quality_score": row.get("quality_score"),
        "heuristic_5": row.get("heuristic_5"),
        "combined_quality_score": row.get("combined_quality_score"),
        "confidence_bucket": row.get("confidence_bucket"),
        "quality_tags": row.get("quality_tags") or [],
        "judge": {
            "overall_quality": judge.get("overall_quality"),
            "usable_as_is": judge.get("usable_as_is"),
            "main_issue": judge.get("main_issue"),
            "short_reason": judge.get("short_reason"),
            "dimension_scores": judge.get("dimension_scores") or {},
        },
        "structural": {
            "line_count": structural.get("line_count"),
            "target_line_count": structural.get("target_line_count"),
            "slur_count": structural.get("slur_count"),
            "prompt_leakage": structural.get("prompt_leakage"),
            "incomplete_ending": structural.get("incomplete_ending"),
            "copy_similarity": structural.get("copy_similarity"),
            "structural_pass": structural.get("structural_pass"),
        },
        "dimensions": dimensions,
        "line_stats": {
            "avg_line_words": line_stats.get("avg_line_words"),
            "max_line_words": line_stats.get("max_line_words"),
            "line_word_counts": line_stats.get("line_word_counts") or [],
        },
        "rhyme": {
            "end_rhyme_rate": rhyme.get("end_rhyme_rate"),
            "internal_rhyme_rate": rhyme.get("internal_rhyme_rate"),
        },
    }


def spread_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit >= len(rows):
        return rows
    if limit == 1:
        return rows[:1]
    indexes = [round(index * (len(rows) - 1) / (limit - 1)) for index in range(limit)]
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index in indexes:
        if index not in seen:
            selected.append(rows[index])
            seen.add(index)
    return selected


def load_candidates(
    judge_dir: Path,
    source: str,
    disagreements_md: Path | None,
    judged_jsonl: Path | None,
    limit: int,
) -> list[dict[str, Any]]:
    disagreements_md = disagreements_md or judge_dir / "top_100_disagreements.md"
    judged_jsonl = judged_jsonl or judge_dir / "judged_candidates.jsonl"
    judged_rows = read_jsonl(judged_jsonl)
    rows_by_id = {str(row.get("candidate_id") or row.get("row_id")): row for row in judged_rows}
    if source == "disagreements":
        ids = disagreement_ids(disagreements_md)[:limit]
        missing = [candidate_id for candidate_id in ids if candidate_id not in rows_by_id]
        if missing:
            raise ValueError(f"Missing judged rows for candidate ids: {missing[:5]}")
        rows = [rows_by_id[candidate_id] for candidate_id in ids]
    else:
        rows = [row for row in judged_rows if row.get("confidence_bucket") == source]
        rows.sort(
            key=lambda row: (
                float(row.get("combined_quality_score") or 0.0),
                float(row.get("quality_score") or 0.0),
                str(row.get("candidate_id") or row.get("row_id") or ""),
            ),
            reverse=True,
        )
        rows = spread_rows(rows, limit)
    if not rows:
        raise ValueError(f"No candidates found for source {source!r}")
    return [candidate_payload(row, rank=index + 1) for index, row in enumerate(rows)]


def build_html(candidates: list[dict[str, Any]], *, review_set: str, export_prefix: str) -> str:
    data = {
        "title": "Qwen3 Manual Candidate Ranking",
        "source": "qwen3_4b_base_12line_v1_auto_quality_judge",
        "review_set": review_set,
        "export_prefix": export_prefix,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    data_json = html.escape(json.dumps(data, ensure_ascii=False), quote=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen3 Manual Candidate Ranking</title>
  <style>
    :root {{
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #181a1f;
      --muted: #626872;
      --line: #d9d9d2;
      --accent: #0f766e;
      --accent-2: #8b5a2b;
      --bad: #b42318;
      --good: #157347;
      --soft: #eef2ef;
      --chip: #ece8dd;
      --shadow: 0 1px 2px rgba(18, 22, 28, .08);
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      min-height: 100vh;
      overflow: hidden;
    }}
    button, input, select, textarea {{ font: inherit; }}
    button {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 6px;
      min-height: 30px;
      padding: 4px 8px;
      cursor: pointer;
    }}
    button:hover {{ border-color: var(--accent); }}
    button.primary {{
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }}
    button.danger {{ color: var(--bad); }}
    .app {{
      display: grid;
      grid-template-columns: 310px minmax(460px, 1fr) 330px;
      height: 100vh;
      min-width: 940px;
    }}
    header {{
      grid-column: 1 / 4;
      height: 50px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      box-shadow: var(--shadow);
      z-index: 3;
    }}
    h1 {{
      margin: 0;
      font-size: 16px;
      font-weight: 720;
      letter-spacing: 0;
    }}
    .header-sub {{
      color: var(--muted);
      font-size: 13px;
      margin-left: 10px;
    }}
    .header-actions {{ display: flex; gap: 8px; align-items: center; }}
    .sidebar, .detail, .review {{
      height: calc(100vh - 50px);
      overflow: auto;
    }}
    .sidebar {{
      border-right: 1px solid var(--line);
      background: #fbfbf8;
      padding: 8px;
    }}
    .search-row {{
      display: grid;
      grid-template-columns: 1fr 112px;
      gap: 8px;
      margin-bottom: 8px;
    }}
    input, select, textarea {{
      width: 100%;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 6px;
      padding: 6px 8px;
    }}
    .candidate-list {{ display: flex; flex-direction: column; gap: 6px; }}
    .candidate-card {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 8px;
      box-shadow: var(--shadow);
      cursor: pointer;
    }}
    .candidate-card.active {{ border-color: var(--accent); outline: 2px solid rgba(15, 118, 110, .16); }}
    .candidate-card.dragging {{ opacity: .55; }}
    .card-top {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 4px;
    }}
    .rank-pill {{
      width: 34px;
      height: 26px;
      border-radius: 999px;
      background: var(--soft);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-weight: 720;
      font-size: 13px;
    }}
    .score-line {{ font-size: 12px; color: var(--muted); white-space: nowrap; }}
    .candidate-id {{
      font-weight: 650;
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    .card-prompt {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.3;
      display: -webkit-box;
      -webkit-line-clamp: 1;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }}
    .chip-row {{ display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }}
    .chip {{
      display: inline-flex;
      align-items: center;
      min-height: 20px;
      padding: 2px 6px;
      border-radius: 999px;
      background: var(--chip);
      color: #333;
      font-size: 11px;
      line-height: 1.2;
    }}
    .chip.good {{ background: #ddf4e7; color: #0f5d35; }}
    .chip.bad {{ background: #fde5e2; color: #9a241b; }}
    .chip.neutral {{ background: #e6edf4; color: #284a66; }}
    .detail {{
      display: flex;
      flex-direction: column;
      min-height: 0;
      overflow: hidden;
      padding: 10px 14px;
    }}
    .detail-head {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: flex-start;
      flex: 0 0 auto;
      margin-bottom: 6px;
    }}
    .detail h2 {{
      margin: 0 0 3px;
      font-size: 16px;
      letter-spacing: 0;
    }}
    .prompt {{
      border-left: 3px solid var(--accent);
      padding-left: 9px;
      color: #2b3038;
      flex: 0 0 auto;
      font-size: 13px;
      line-height: 1.32;
      margin: 4px 0 8px;
      max-width: 900px;
      max-height: 54px;
      overflow: auto;
    }}
    .lyrics {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      flex: 1 1 auto;
      min-height: 0;
      overflow: auto;
      padding: 12px;
      white-space: pre-wrap;
      line-height: 1.43;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 13px;
      box-shadow: var(--shadow);
    }}
    .secondary-panel {{
      flex: 0 0 auto;
      margin-top: 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    .secondary-panel summary {{
      cursor: pointer;
      padding: 7px 10px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }}
    .secondary-panel[open] {{
      max-height: 220px;
      overflow: auto;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 6px;
      margin: 0;
      padding: 0 10px 8px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 6px;
      min-height: 42px;
    }}
    .metric .label {{ color: var(--muted); font-size: 11px; }}
    .metric .value {{ font-weight: 720; margin-top: 2px; font-size: 13px; }}
    .bars {{
      display: grid;
      grid-template-columns: repeat(2, minmax(180px, 1fr));
      gap: 6px 12px;
      margin: 0;
      padding: 0 10px 10px;
    }}
    .bar-row {{ display: grid; grid-template-columns: 136px 1fr 38px; gap: 6px; align-items: center; font-size: 11px; }}
    .bar-bg {{ height: 6px; border-radius: 999px; background: #e7e4da; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 999px; background: var(--accent); }}
    .review {{
      border-left: 1px solid var(--line);
      background: #fbfbf8;
      display: flex;
      flex-direction: column;
      gap: 8px;
      min-height: 0;
      overflow: hidden;
      padding: 10px;
    }}
    .review-section {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 9px;
      margin-bottom: 0;
      box-shadow: var(--shadow);
    }}
    .review-section h3 {{
      margin: 0 0 6px;
      font-size: 13px;
    }}
    .rating-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 5px; }}
    .rating-grid button {{ padding: 4px 0; font-weight: 720; }}
    .rating-grid button.selected {{ background: var(--accent); color: white; border-color: var(--accent); }}
    .decision-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px; }}
    .decision-grid button.selected {{ background: #263238; color: white; border-color: #263238; }}
    .move-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 5px; }}
    .nav-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 5px; margin-bottom: 8px; }}
    .review-notes {{
      flex: 1 1 auto;
      min-height: 110px;
      display: flex;
      flex-direction: column;
    }}
    textarea {{ min-height: 76px; flex: 1 1 auto; resize: vertical; line-height: 1.35; }}
    .review-reason {{
      flex: 0 1 auto;
      max-height: 118px;
      overflow: auto;
    }}
    .progress-wrap {{ display: flex; gap: 8px; align-items: center; color: var(--muted); font-size: 13px; }}
    .progress-bg {{ width: 140px; height: 8px; background: #e2dfd3; border-radius: 999px; overflow: hidden; }}
    .progress-fill {{ height: 100%; background: var(--accent-2); }}
    .small {{ color: var(--muted); font-size: 12px; line-height: 1.4; }}
    @media (max-width: 1080px) {{
      body {{ overflow: auto; }}
      .app {{ min-width: 0; height: auto; grid-template-columns: 1fr; }}
      header {{ grid-column: 1; position: sticky; top: 0; flex-wrap: wrap; height: auto; min-height: 58px; gap: 8px; padding: 10px; }}
      .sidebar, .detail, .review {{ height: auto; max-height: none; overflow: visible; }}
      .detail {{ display: block; }}
      .lyrics {{ max-height: none; overflow: visible; }}
      .sidebar, .review {{ border: 0; }}
      .metrics, .bars {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div>
        <h1>Manual Candidate Ranking <span class="header-sub" id="subtitle"></span></h1>
      </div>
      <div class="header-actions">
        <div class="progress-wrap">
          <span id="progressText">0 reviewed</span>
          <div class="progress-bg"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
        </div>
        <button id="exportJson">Export JSON</button>
        <button id="exportCsv">Export CSV</button>
        <button class="danger" id="resetState">Reset</button>
      </div>
    </header>
    <aside class="sidebar">
      <div class="search-row">
        <input id="search" placeholder="Search prompt, lyrics, issue">
        <select id="filter">
          <option value="all">All</option>
          <option value="unreviewed">Unreviewed</option>
          <option value="reviewed">Reviewed</option>
          <option value="keep">Keep</option>
          <option value="edit">Edit</option>
          <option value="drop">Drop</option>
        </select>
      </div>
      <div class="candidate-list" id="candidateList"></div>
    </aside>
    <main class="detail">
      <div class="detail-head">
        <div>
          <h2 id="detailTitle"></h2>
          <div class="small" id="detailMeta"></div>
        </div>
        <div class="chip-row" id="detailChips"></div>
      </div>
      <div class="prompt" id="promptText"></div>
      <div class="lyrics" id="lyricsText"></div>
      <details class="secondary-panel">
        <summary>Metrics</summary>
        <div class="metrics" id="metrics"></div>
        <div class="bars" id="dimensionBars"></div>
      </details>
    </main>
    <aside class="review">
      <div class="review-section">
        <h3>Manual Rating</h3>
        <div class="rating-grid" id="ratingButtons"></div>
      </div>
      <div class="review-section">
        <h3>Decision</h3>
        <div class="decision-grid" id="decisionButtons"></div>
      </div>
      <div class="review-section">
        <h3>Candidate</h3>
        <div class="nav-grid">
          <button id="prevCandidate">Previous</button>
          <button id="nextCandidate">Next</button>
        </div>
        <h3>Rank Position</h3>
        <div class="move-grid">
          <button id="moveTop">Top</button>
          <button id="moveUp">Up</button>
          <button id="moveDown">Down</button>
          <button id="moveBottom">Bottom</button>
        </div>
      </div>
      <div class="review-section review-notes">
        <h3>Notes</h3>
        <textarea id="notes" placeholder="Why this rank? What should change?"></textarea>
      </div>
      <div class="review-section review-reason">
        <h3>Current Candidate</h3>
        <p class="small" id="judgeReason"></p>
      </div>
    </aside>
  </div>
  <script id="candidateData" type="application/json">{data_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById('candidateData').textContent);
    const candidates = payload.candidates;
    const storageKey = 'qwen3-manual-rank-' + payload.source + '-' + payload.review_set + '-' + candidates.map(c => c.candidate_id).join('-').slice(0, 80);
    const defaultState = () => ({{
      order: candidates.map(c => c.candidate_id),
      reviews: Object.fromEntries(candidates.map(c => [c.candidate_id, {{ rating: null, decision: '', notes: '' }}]))
    }});
    let state = loadState();
    let selectedId = state.order[0];
    let draggedId = null;

    const byId = new Map(candidates.map(c => [c.candidate_id, c]));
    const el = id => document.getElementById(id);

    function loadState() {{
      try {{
        const saved = JSON.parse(localStorage.getItem(storageKey));
        if (saved && Array.isArray(saved.order) && saved.reviews) {{
          const known = new Set(candidates.map(c => c.candidate_id));
          saved.order = saved.order.filter(id => known.has(id));
          for (const c of candidates) if (!saved.order.includes(c.candidate_id)) saved.order.push(c.candidate_id);
          for (const c of candidates) saved.reviews[c.candidate_id] ||= {{ rating: null, decision: '', notes: '' }};
          return saved;
        }}
      }} catch {{}}
      return defaultState();
    }}
    function saveState() {{
      localStorage.setItem(storageKey, JSON.stringify(state));
      updateProgress();
    }}
    function review(id) {{ return state.reviews[id] ||= {{ rating: null, decision: '', notes: '' }}; }}
    function escapeHtml(value) {{
      return String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
    }}
    function chipClass(text) {{
      if (['auto_keep','yes'].includes(String(text))) return 'good';
      if (['auto_reject','no','awkward','generic','weak_payoff'].includes(String(text))) return 'bad';
      return 'neutral';
    }}
    function renderList() {{
      const q = el('search').value.trim().toLowerCase();
      const filter = el('filter').value;
      const list = el('candidateList');
      list.innerHTML = '';
      for (const [index, id] of state.order.entries()) {{
        const c = byId.get(id);
        const r = review(id);
        const haystack = [c.candidate_id, c.prompt, c.lyrics, c.judge.main_issue, ...(c.quality_tags || [])].join(' ').toLowerCase();
        if (q && !haystack.includes(q)) continue;
        if (filter === 'reviewed' && !r.rating) continue;
        if (filter === 'unreviewed' && r.rating) continue;
        if (['keep','edit','drop'].includes(filter) && r.decision !== filter) continue;
        const card = document.createElement('div');
        card.className = 'candidate-card' + (id === selectedId ? ' active' : '');
        card.draggable = true;
        card.dataset.id = id;
        card.innerHTML = `
          <div class="card-top">
            <span class="rank-pill">${{index + 1}}</span>
            <span class="score-line">J ${{escapeHtml(c.judge.overall_quality)}} | H ${{escapeHtml(c.quality_score)}} | C ${{escapeHtml(c.combined_quality_score)}}</span>
          </div>
          <div class="candidate-id">${{escapeHtml(c.candidate_id)}}</div>
          <div class="card-prompt">${{escapeHtml(c.prompt)}}</div>
          <div class="chip-row">
            <span class="chip ${{chipClass(c.judge.main_issue)}}">${{escapeHtml(c.judge.main_issue)}}</span>
            ${{r.rating ? `<span class="chip good">manual ${{r.rating}}</span>` : '<span class="chip">unrated</span>'}}
            ${{r.decision ? `<span class="chip neutral">${{escapeHtml(r.decision)}}</span>` : ''}}
          </div>`;
        card.addEventListener('click', () => selectCandidate(id));
        card.addEventListener('dragstart', () => {{ draggedId = id; card.classList.add('dragging'); }});
        card.addEventListener('dragend', () => {{ draggedId = null; card.classList.remove('dragging'); }});
        card.addEventListener('dragover', e => e.preventDefault());
        card.addEventListener('drop', e => {{
          e.preventDefault();
          const targetId = card.dataset.id;
          if (draggedId && targetId && draggedId !== targetId) moveBefore(draggedId, targetId);
        }});
        list.appendChild(card);
      }}
    }}
    function selectCandidate(id) {{
      selectedId = id;
      renderList();
      renderDetail();
    }}
    function renderDetail() {{
      const c = byId.get(selectedId);
      const r = review(selectedId);
      const rank = state.order.indexOf(selectedId) + 1;
      el('detailTitle').textContent = `${{rank}}. ${{c.candidate_id}}`;
      el('detailMeta').textContent = `${{c.prompt_family || 'unknown'}} | ${{c.confidence_bucket || 'needs_review'}} | initial rank ${{c.initial_rank}}`;
      el('promptText').textContent = c.prompt || '';
      el('lyricsText').textContent = c.lyrics || '';
      el('detailChips').innerHTML = [
        c.judge.main_issue,
        `judge ${{c.judge.overall_quality}}`,
        `usable ${{c.judge.usable_as_is}}`,
        ...(c.quality_tags || [])
      ].map(t => `<span class="chip ${{chipClass(t)}}">${{escapeHtml(t)}}</span>`).join('');
      const metrics = [
        ['Combined', c.combined_quality_score],
        ['Heuristic', c.quality_score],
        ['Judge', c.judge.overall_quality],
        ['Lines', `${{c.structural.line_count}}/${{c.structural.target_line_count}}`],
        ['Copy', c.structural.copy_similarity],
        ['End Rhyme', c.rhyme.end_rhyme_rate],
        ['Internal Rhyme', c.rhyme.internal_rhyme_rate],
        ['Avg Words', c.line_stats.avg_line_words],
      ];
      el('metrics').innerHTML = metrics.map(([label, value]) => `<div class="metric"><div class="label">${{escapeHtml(label)}}</div><div class="value">${{escapeHtml(value)}}</div></div>`).join('');
      renderBars(c);
      renderReviewControls(c, r);
      el('judgeReason').textContent = c.judge.short_reason || '';
    }}
    function renderBars(c) {{
      const entries = Object.entries(c.dimensions || {{}});
      el('dimensionBars').innerHTML = entries.map(([name, value]) => {{
        const pct = Math.max(0, Math.min(100, Number(value) * 100));
        return `<div class="bar-row"><span>${{escapeHtml(name.replaceAll('_',' '))}}</span><div class="bar-bg"><div class="bar-fill" style="width:${{pct}}%"></div></div><span>${{Number(value).toFixed(2)}}</span></div>`;
      }}).join('');
    }}
    function renderReviewControls(c, r) {{
      el('ratingButtons').innerHTML = [1,2,3,4,5].map(n => `<button class="${{r.rating === n ? 'selected' : ''}}" data-rating="${{n}}">${{n}}</button>`).join('');
      el('ratingButtons').querySelectorAll('button').forEach(btn => btn.addEventListener('click', () => {{
        review(selectedId).rating = Number(btn.dataset.rating);
        saveState(); renderList(); renderDetail();
      }}));
      const decisions = [['keep','Keep'], ['edit','Edit'], ['drop','Drop']];
      el('decisionButtons').innerHTML = decisions.map(([value,label]) => `<button class="${{r.decision === value ? 'selected' : ''}}" data-decision="${{value}}">${{label}}</button>`).join('');
      el('decisionButtons').querySelectorAll('button').forEach(btn => btn.addEventListener('click', () => {{
        review(selectedId).decision = btn.dataset.decision;
        saveState(); renderList(); renderDetail();
      }}));
      el('notes').value = r.notes || '';
    }}
    function updateProgress() {{
      const reviewed = candidates.filter(c => review(c.candidate_id).rating).length;
      const pct = Math.round(100 * reviewed / candidates.length);
      el('progressText').textContent = `${{reviewed}}/${{candidates.length}} reviewed`;
      el('progressFill').style.width = pct + '%';
      el('subtitle').textContent = `${{candidates.length}} ${{payload.review_set.replaceAll('_', ' ')}} candidates`;
    }}
    function moveBefore(sourceId, targetId) {{
      const next = state.order.filter(id => id !== sourceId);
      const targetIndex = next.indexOf(targetId);
      next.splice(targetIndex, 0, sourceId);
      state.order = next;
      selectedId = sourceId;
      saveState(); renderList(); renderDetail();
    }}
    function moveSelected(delta) {{
      const i = state.order.indexOf(selectedId);
      const j = Math.max(0, Math.min(state.order.length - 1, i + delta));
      if (i === j) return;
      const [id] = state.order.splice(i, 1);
      state.order.splice(j, 0, id);
      saveState(); renderList(); renderDetail();
    }}
    function moveSelectedTo(pos) {{
      const i = state.order.indexOf(selectedId);
      const [id] = state.order.splice(i, 1);
      state.order.splice(pos, 0, id);
      saveState(); renderList(); renderDetail();
    }}
    function selectRelative(delta) {{
      const i = state.order.indexOf(selectedId);
      const nextIndex = Math.max(0, Math.min(state.order.length - 1, i + delta));
      selectCandidate(state.order[nextIndex]);
    }}
    function exportRows() {{
      return state.order.map((id, index) => {{
        const c = byId.get(id);
        const r = review(id);
        return {{
          manual_rank: index + 1,
          candidate_id: id,
          manual_rating: r.rating,
          decision: r.decision,
          notes: r.notes,
          prompt: c.prompt,
          judge_quality: c.judge.overall_quality,
          judge_issue: c.judge.main_issue,
          judge_reason: c.judge.short_reason,
          heuristic_score: c.quality_score,
          combined_score: c.combined_quality_score,
          quality_tags: c.quality_tags,
          lyrics: c.lyrics
        }};
      }});
    }}
    function download(name, text, type) {{
      const blob = new Blob([text], {{ type }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = name; document.body.appendChild(a); a.click();
      a.remove(); URL.revokeObjectURL(url);
    }}
    function csvCell(value) {{
      return '"' + String(value ?? '').replace(/"/g, '""') + '"';
    }}
    el('notes').addEventListener('input', e => {{
      review(selectedId).notes = e.target.value;
      saveState(); renderList();
    }});
    el('search').addEventListener('input', renderList);
    el('filter').addEventListener('change', renderList);
    el('moveTop').addEventListener('click', () => moveSelectedTo(0));
    el('moveBottom').addEventListener('click', () => moveSelectedTo(state.order.length - 1));
    el('moveUp').addEventListener('click', () => moveSelected(-1));
    el('moveDown').addEventListener('click', () => moveSelected(1));
    el('prevCandidate').addEventListener('click', () => selectRelative(-1));
    el('nextCandidate').addEventListener('click', () => selectRelative(1));
    el('exportJson').addEventListener('click', () => download(`${{payload.export_prefix}}.json`, JSON.stringify(exportRows(), null, 2), 'application/json'));
    el('exportCsv').addEventListener('click', () => {{
      const rows = exportRows();
      const columns = ['manual_rank','candidate_id','manual_rating','decision','notes','judge_quality','judge_issue','heuristic_score','combined_score','prompt','lyrics'];
      const csv = [columns.join(',')].concat(rows.map(row => columns.map(col => csvCell(row[col])).join(','))).join('\\n');
      download(`${{payload.export_prefix}}.csv`, csv, 'text/csv');
    }});
    el('resetState').addEventListener('click', () => {{
      if (confirm('Clear all manual rankings, labels, and notes for this app?')) {{
        localStorage.removeItem(storageKey);
        state = defaultState();
        selectedId = state.order[0];
        renderList(); renderDetail(); updateProgress();
      }}
    }});
    window.addEventListener('keydown', e => {{
      if (e.target.matches('input, textarea, select')) return;
      if (e.key === 'ArrowUp') {{ e.preventDefault(); moveSelected(-1); }}
      if (e.key === 'ArrowDown') {{ e.preventDefault(); moveSelected(1); }}
      if (/^[1-5]$/.test(e.key)) {{
        review(selectedId).rating = Number(e.key);
        saveState(); renderList(); renderDetail();
      }}
    }});
    renderList();
    renderDetail();
    updateProgress();
  </script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be > 0")
    candidates = load_candidates(args.judge_dir, args.source, args.disagreements_md, args.judged_jsonl, args.limit)
    export_prefix = args.export_prefix or (
        "manual_rank_results" if args.source == "disagreements" else f"manual_rank_{args.source}_results"
    )
    out = args.out or args.judge_dir / (
        "manual_rank_30.html" if args.source == "disagreements" else f"manual_rank_{args.source}_{args.limit}.html"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(candidates, review_set=args.source, export_prefix=export_prefix), encoding="utf-8")
    print(json.dumps({"out": str(out), "source": args.source, "candidate_count": len(candidates)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
