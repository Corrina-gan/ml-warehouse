"""Both robots on the shared warehouse: pick a policy and watch a whole episode.

The episode is recorded once on the server. Its frames are then sent to the
browser, and the page animates them there, so playing, scrubbing and changing
speed never wait for the server. A second policy can be played next to the
first on the same map.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import streamlit as st

from environment.config import MAX_STEPS, N_ACTIONS, N_AGENTS, SEEDS
from environment.environment import make_warehouse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PPO_RESULTS = PROJECT_ROOT / "results" / "ppo"
REINFORCE_RESULTS = PROJECT_ROOT / "results" / "reinforce"
RULES_ONLY = "rules_only"
NOTHING = "nothing"


def show_warehouse():
    """Algorithm, comparison and seed controls, then the episode player."""
    choices = _algorithm_choices()
    labels = {key: label for key, label, _mtime in choices}
    mtimes = {key: mtime for key, _label, mtime in choices}
    st.session_state.setdefault("warehouse_seed", 0)
    if st.session_state.get("warehouse_algorithm") not in labels:
        st.session_state["warehouse_algorithm"] = choices[0][0]
    if st.session_state.get("warehouse_compare") not in labels:
        st.session_state["warehouse_compare"] = NOTHING

    algorithm_column, compare_column, seed_column = st.columns([3, 3, 1])
    with algorithm_column:
        st.selectbox(
            "Algorithm",
            list(labels),
            format_func=labels.get,
            key="warehouse_algorithm",
            help=(
                "The algorithm controls both robots. Random takes a uniform action. PPO and REINFORCE "
                "play the greedy policy from the 100,000-step run. Rules only plays shared/warehouse_rules.py "
                "by walking forward, with no network and no learning."
            ),
        )
    with compare_column:
        st.selectbox(
            "Side by side with",
            [NOTHING] + list(labels),
            format_func=lambda key: "Nothing" if key == NOTHING else labels[key],
            key="warehouse_compare",
            help="Plays a second algorithm on the same seed, in step with the first.",
        )
    with seed_column:
        st.number_input(
            "Seed",
            min_value=0,
            max_value=99,
            key="warehouse_seed",
            help="Same seeds as the analysis. Seed 0 is the sample episode.",
        )

    seed = int(st.session_state["warehouse_seed"])
    shown = [st.session_state["warehouse_algorithm"]]
    if st.session_state["warehouse_compare"] != NOTHING:
        shown.append(st.session_state["warehouse_compare"])
    try:
        episodes = [record_episode(seed, key, mtimes[key]) for key in shown]
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        st.error("PPO and REINFORCE need PyTorch in the environment that runs this app.")
        return

    page = _player_page(episodes, [labels[key] for key in shown], _theme())
    _show_page(page, fallback_height=900 if len(shown) > 1 else 860)


def _theme():
    """"light" or "dark", following the app's theme where Streamlit reports it."""
    theme = getattr(getattr(st, "context", None), "theme", None)
    kind = getattr(theme, "type", None) or st.get_option("theme.base")
    return "light" if kind == "light" else "dark"


def _show_page(page, fallback_height):
    """Put the player on the page.

    st.iframe makes the frame as tall as the player. A Streamlit version from
    before st.iframe gets the older components.html with a fixed height.
    """
    if hasattr(st, "iframe"):
        st.iframe(page, height="content")
    else:
        import streamlit.components.v1 as components

        components.html(page, height=fallback_height, scrolling=True)


def _algorithm_choices():
    """Random, each official 100k PPO and REINFORCE run whose weights are on disk, and the rules alone."""
    choices = [("random", "Random", 0)]
    for seed in SEEDS:
        path = PPO_RESULTS / f"seed_{seed}" / "ppo.pt"
        if path.is_file():
            choices.append((f"ppo:{path.resolve()}", f"PPO, seed {seed}", path.stat().st_mtime_ns))
    for seed in SEEDS:
        path = REINFORCE_RESULTS / f"seed_{seed}" / "reinforce.pt"
        if path.is_file():
            choices.append((f"reinforce:{path.resolve()}", f"REINFORCE, seed {seed}", path.stat().st_mtime_ns))
    choices.append((RULES_ONLY, "Rules only, walking forward, no learning", 0))
    return choices


@st.cache_resource(show_spinner="Loading the PPO policy")
def _load_ppo(path):
    from algorithms.PPO.ppo import load_agent

    return load_agent(path)


def _load_player(algorithm):
    """The object that chooses actions, or None for the uniform random policy."""
    if algorithm.startswith("ppo:"):
        return _load_ppo(algorithm.removeprefix("ppo:"))
    if algorithm.startswith("reinforce:"):
        # REINFORCE keeps a memory during an episode, so every recording loads its own copy.
        from algorithms.REINFORCE.reinforce import load_agent

        return load_agent(algorithm.removeprefix("reinforce:"))
    if algorithm == RULES_ONLY:
        from shared.warehouse_rules import RulesOnly

        return RulesOnly(forward=True)
    return None


@st.cache_data(show_spinner="Placing the robots in the warehouse")
def record_episode(seed, algorithm, weights_mtime):
    """One episode under the selected policy. Frames store positions only."""
    del weights_mtime
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"gymnasium\.utils\.passive_env_checker",
    )
    env = make_warehouse()
    try:
        observations, _info = env.reset(seed=int(seed))
        warehouse = env.unwrapped
        layout = {
            "rows": int(warehouse.grid_size[0]),
            "cols": int(warehouse.grid_size[1]),
            "highways": warehouse.highways.astype(bool).tolist(),
            "goals": [(int(x), int(y)) for x, y in warehouse.goals],
        }
        frames = [_snapshot(warehouse, None)]
        rng = np.random.default_rng(int(seed))
        agent = _load_player(algorithm)
        # REINFORCE keeps a memory during an episode. PPO has nothing to reset.
        if hasattr(agent, "reset_episode"):
            agent.reset_episode()
        for _step in range(MAX_STEPS):
            if agent is None:
                actions = tuple(int(action) for action in rng.integers(0, N_ACTIONS, size=N_AGENTS))
            elif hasattr(agent, "act_greedy"):
                actions = agent.act_greedy(observations)
            else:
                actions = agent.choose_actions(observations)
            observations, rewards, terminated, truncated, _info = env.step(actions)
            frames.append(_snapshot(warehouse, actions, rewards))
            if terminated or truncated:
                break
        return {"layout": layout, "frames": frames}
    finally:
        env.close()


def _snapshot(warehouse, actions, rewards=None):
    requested = set(warehouse.request_queue)
    return {
        "agents": [
            {
                "x": int(agent.x),
                "y": int(agent.y),
                "dir": int(agent.dir.value),
                "carrying": agent.carrying_shelf is not None,
            }
            for agent in warehouse.agents
        ],
        "shelves": [
            {
                "x": int(shelf.x),
                "y": int(shelf.y),
                "requested": shelf in requested,
            }
            for shelf in warehouse.shelfs
        ],
        "actions": None if actions is None else tuple(int(action) for action in actions),
        "rewards": None if rewards is None else tuple(float(reward) for reward in rewards),
    }


def _player_page(episodes, names, theme):
    """The self-contained page the browser animates: every frame of each episode, plus the player."""
    layout = episodes[0]["layout"]
    data = {
        "rows": layout["rows"],
        "cols": layout["cols"],
        "highways": [[int(cell) for cell in row] for row in layout["highways"]],
        "goals": [list(goal) for goal in layout["goals"]],
        "policies": [
            {
                "name": name,
                # Per step and robot: column, row, facing, carrying.
                "robots": [
                    [[agent["x"], agent["y"], agent["dir"], int(agent["carrying"])] for agent in frame["agents"]]
                    for frame in episode["frames"]
                ],
                # Per step and shelf: column, row, requested. A shelf keeps its place in the list.
                "shelves": [
                    [[shelf["x"], shelf["y"], int(shelf["requested"])] for shelf in frame["shelves"]]
                    for frame in episode["frames"]
                ],
                "rewards": [
                    [0] * len(frame["agents"]) if frame["rewards"] is None else [int(reward > 0) for reward in frame["rewards"]]
                    for frame in episode["frames"]
                ],
            }
            for episode, name in zip(episodes, names)
        ],
    }
    packed = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return PLAYER.replace("__THEME__", theme).replace("__DATA__", packed)


PLAYER = r"""<!doctype html>
<html><head><meta charset="utf-8"><style>
.app { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); font-size: 14px;
       background: var(--page); border-radius: 12px; padding: 12px 14px 14px; }
.app.dark  { --page:#0e1117; --ink:#e9ecf2; --soft:#98a2b3; --card:#161b25; --line:#273040; --floor:#0f141c; --aisle:#1c2330;
             --shelf:#5550b8; --req:#2dd4bf; --drop:#e9ecf2; --dropink:#0f141c; --r1:#ff9f43; --r2:#4aa3ff; --good:#34d399; --accent:#4aa3ff; --other:#7c8798; }
.app.light { --page:#ffffff; --ink:#1b1f27; --soft:#5b6472; --card:#ffffff; --line:#e1e5ec; --floor:#f3f5f8; --aisle:#e4e8ee;
             --shelf:#5a54b0; --req:#0f9d94; --drop:#2b303b; --dropink:#ffffff; --r1:#f5870a; --r2:#1e88e5; --good:#12a150; --accent:#257de0; --other:#8a93a3; }
* { box-sizing: border-box; }
body { margin: 0; background: transparent; }
.bar { display: flex; align-items: center; gap: 10px; padding: 4px 2px 12px; flex-wrap: wrap; }
button { font: inherit; color: var(--ink); background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 7px 12px; cursor: pointer; }
button:hover { border-color: var(--soft); }
button.main { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; min-width: 84px; }
.speed { display: flex; margin-left: 4px; }
.speed button { border-radius: 0; margin-left: -1px; padding: 7px 10px; color: var(--soft); }
.speed button:first-child { border-radius: 8px 0 0 8px; margin-left: 0; }
.speed button:last-child { border-radius: 0 8px 8px 0; }
.speed button.on { color: var(--ink); background: var(--line); }
input[type=range] { flex: 1; min-width: 160px; accent-color: var(--accent); }
.stepno { color: var(--soft); min-width: 104px; text-align: right; font-variant-numeric: tabular-nums; }
.stage { display: grid; gap: 22px; align-items: start; grid-template-columns: minmax(0, var(--board, 560px)) minmax(300px, 1fr); }
.stage.two { grid-template-columns: repeat(2, minmax(0, var(--board, 420px))) minmax(320px, 1fr); }
@media (max-width: 1000px) { .stage.two { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); } .stage.two > div:last-child { grid-column: 1 / -1; } }
@media (max-width: 700px) { .stage:not(.two) { grid-template-columns: minmax(0, 1fr); } }
.head { display: flex; justify-content: space-between; align-items: baseline; margin: 0 2px 8px; }
.head .name { font-weight: 600; }
.head .count { color: var(--soft); font-variant-numeric: tabular-nums; }
canvas { display: block; width: 100%; }
.cards { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.stage.two .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; }
.card .label { color: var(--soft); font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.card .value { font-size: 28px; font-weight: 650; line-height: 1.15; font-variant-numeric: tabular-nums; }
.section { color: var(--soft); font-size: 12px; margin: 16px 2px 6px; }
.robot { display: flex; align-items: center; gap: 8px; padding: 4px 2px; }
.dot { width: 12px; height: 12px; border-radius: 3px; flex: none; }
.robot .who { font-weight: 600; min-width: 58px; }
.robot .job { color: var(--soft); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.log { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 6px 12px; height: 170px; overflow: hidden; font-size: 13px; }
.log div { display: flex; gap: 8px; padding: 3px 0; align-items: center; white-space: nowrap; }
.log .chip { width: 18px; height: 18px; border-radius: 4px; flex: none; color: #fff; font-size: 11px; font-weight: 700; display: inline-flex; align-items: center; justify-content: center; }
.log .at { color: var(--soft); min-width: 58px; font-variant-numeric: tabular-nums; }
.log .what { overflow: hidden; text-overflow: ellipsis; }
.log .won { color: var(--good); font-weight: 600; }
.log .none { color: var(--soft); }
.key { display: flex; flex-wrap: wrap; gap: 6px 16px; color: var(--soft); font-size: 12px; margin-top: 14px; }
.key span { display: inline-flex; align-items: center; gap: 6px; }
.key i { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
.key i.ring { background: transparent; border: 3px solid; }
.legend { display: flex; gap: 14px; color: var(--soft); font-size: 12px; margin: 0 2px 4px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
</style></head>
<body><div class="app __THEME__" id="app">
  <div class="bar">
    <button class="main" id="play">Play</button>
    <button id="back" title="One step back">Back</button>
    <button id="next" title="One step forward">Step</button>
    <button id="restart">Restart</button>
    <div class="speed" id="speed">
      <button data-rate="5">Slow</button><button data-rate="14" class="on">Normal</button><button data-rate="40">Fast</button><button data-rate="120">Turbo</button>
    </div>
    <input type="range" id="scrub" min="0" value="0" step="1">
    <span class="stepno" id="stepno"></span>
  </div>
  <div class="stage" id="stage"></div>
  <div class="key">
    <span><i style="background:var(--shelf)"></i>shelf in storage</span>
    <span><i style="background:var(--req)"></i>requested shelf</span>
    <span><i style="background:var(--drop)"></i>drop-off tile</span>
    <span><i style="background:var(--r1)"></i>robot 1</span>
    <span><i style="background:var(--r2)"></i>robot 2</span>
    <span><i class="ring" style="border-color:var(--req)"></i>carrying a shelf that still has to be delivered</span>
    <span><i class="ring" style="border-color:var(--shelf)"></i>carrying a delivered shelf back to storage</span>
    <span>an empty robot drives under the shelves</span>
  </div>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const app = document.getElementById('app');
// Use dark or light colours to match the page around the player. If that page cannot be read, keep the server's choice.
try {
  const host = window.parent.document;
  const [red, green, blue, alpha = 1] = getComputedStyle(host.querySelector('.stApp') || host.body).backgroundColor.match(/[\d.]+/g).map(Number);
  if (alpha > 0) { const dark = red * 0.299 + green * 0.587 + blue * 0.114 < 128; app.classList.toggle('dark', dark); app.classList.toggle('light', !dark); }
} catch (error) {}
const C = {};
for (const k of ['soft','card','line','floor','aisle','shelf','req','drop','dropink','r1','r2','good','accent','other'])
  C[k] = getComputedStyle(app).getPropertyValue('--' + k).trim();
const P = D.policies, T = P[0].robots.length - 1, R = P[0].robots[0].length;
const JOBS = ['looking for a requested shelf', 'carrying a shelf to the drop-off', 'taking the shelf back to storage'];
const ROBOT = [C.r1, C.r2];

// What happened in each episode: which shelf a robot carries, the delivery count after every step, the events.
for (const p of P) {
  p.carry = p.robots.map((robots, t) => robots.map(a => a[3] ? p.shelves[t].findIndex(s => s[0] === a[0] && s[1] === a[1]) : -1));
  p.job = (t, r) => { const j = p.carry[t][r]; return j < 0 ? 0 : (p.shelves[t][j][2] ? 1 : 2); };
  p.totals = [new Array(R).fill(0)]; p.events = [];
  for (let t = 1; t <= T; t++) {
    const counts = p.totals[t - 1].slice();
    for (let r = 0; r < R; r++) {
      const was = p.job(t - 1, r), now = p.job(t, r);
      if (p.rewards[t][r]) { counts[r]++; p.events.push([t, r, 'delivered a shelf', true]); }
      else if (was === 0 && now === 1) p.events.push([t, r, 'picked up a requested shelf']);
      else if (was === 0 && now === 2) p.events.push([t, r, 'picked up an unrequested shelf']);
      else if (was !== 0 && now === 0) p.events.push([t, r, 'put the shelf back']);
    }
    p.totals.push(counts);
  }
  p.team = p.totals.map(c => c.reduce((a, b) => a + b, 0));
}

// Page skeleton: one board per policy, then the panel.
const stage = document.getElementById('stage');
if (P.length > 1) stage.classList.add('two');
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
const boards = P.map(p => {
  const wrap = el('div'), head = el('div', 'head'), count = el('span', 'count'), canvas = el('canvas');
  canvas.style.aspectRatio = D.cols + ' / ' + D.rows;
  head.append(el('span', 'name', p.name), count); wrap.append(head, canvas); stage.append(wrap);
  return {canvas, count};
});
const panel = el('div'); stage.append(panel);
const cards = el('div', 'cards'); panel.append(cards);
const cardSpecs = P.length > 1 ? P.map(p => p.name) : ['Deliveries', 'Robot 1', 'Robot 2'];
const cardValues = cardSpecs.map(label => { const c = el('div', 'card'), v = el('div', 'value', '0'); c.append(el('div', 'label', label), v); cards.append(c); return v; });
panel.append(el('div', 'section', P.length > 1 ? 'What the robots of ' + P[0].name + ' are doing' : 'What each robot is doing'));
const jobText = [];
for (let r = 0; r < R; r++) {
  const row = el('div', 'robot'), dot = el('span', 'dot'), job = el('span', 'job'); dot.style.background = ROBOT[r % 2];
  row.append(dot, el('span', 'who', 'Robot ' + (r + 1)), job); panel.append(row); jobText.push(job);
}
panel.append(el('div', 'section', 'Deliveries so far'));
if (P.length > 1) {
  const legend = el('div', 'legend');
  P.forEach((p, i) => { const s = el('span'), line = el('i'); line.style.background = i ? C.other : C.accent; s.append(line, document.createTextNode(p.name)); legend.append(s); });
  panel.append(legend);
}
const chart = el('canvas'); chart.style.height = '132px'; panel.append(chart);
panel.append(el('div', 'section', P.length > 1 ? 'Latest events, ' + P[0].name : 'Latest events'));
const log = el('div', 'log'); panel.append(log);

// A board is never taller than the browser window, so the whole player fits on one screen.
function sizeBoards() {
  let tall = 900;
  try { tall = window.parent.innerHeight || tall; } catch (error) {}
  const widest = P.length > 1 ? 420 : 560;
  stage.style.setProperty('--board', Math.max(280, Math.min(widest, Math.round((tall - 170) * D.cols / D.rows))) + 'px');
}
sizeBoards();

function rounded(g, x, y, w, h, r, fill, stroke, width) {
  g.beginPath();
  if (g.roundRect) g.roundRect(x, y, w, h, r); else g.rect(x, y, w, h);
  if (fill) { g.fillStyle = fill; g.fill(); }
  if (stroke) { g.strokeStyle = stroke; g.lineWidth = width; g.stroke(); }
}
function fit(canvas, height) {
  const ratio = window.devicePixelRatio || 1, w = canvas.clientWidth;
  if (canvas.width !== Math.round(w * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(w * ratio); canvas.height = Math.round(height * ratio); canvas.style.height = height + 'px';
  }
  const g = canvas.getContext('2d'); g.setTransform(ratio, 0, 0, ratio, 0, 0); g.clearRect(0, 0, w, height); return [g, w];
}
const ANGLE = [-90, 90, 180, 0];   // facing up, down, left, right

function drawBoard(canvas, p, pos) {
  const t = Math.min(T, Math.floor(pos)), part = Math.min(1, pos - t), t2 = Math.min(T, t + 1);
  const cell = canvas.clientWidth / D.cols, [g, w] = fit(canvas, cell * D.rows);
  rounded(g, 0, 0, w, cell * D.rows, 12, C.floor);
  for (let y = 0; y < D.rows; y++) for (let x = 0; x < D.cols; x++)
    if (D.highways[y][x]) rounded(g, x * cell + 1.5, y * cell + 1.5, cell - 3, cell - 3, 5, C.aisle);
  g.textAlign = 'center'; g.textBaseline = 'middle';
  for (const [x, y] of D.goals) {
    rounded(g, x * cell + 3, y * cell + 3, cell - 6, cell - 6, 7, C.drop);
    g.fillStyle = C.dropink; g.font = '700 ' + Math.round(cell * 0.22) + 'px system-ui, sans-serif'; g.fillText('DROP', (x + .5) * cell, (y + .52) * cell);
  }
  const carried = new Set(p.carry[t].filter(j => j >= 0)), under = new Set();
  p.shelves[t].forEach((s, j) => {
    if (carried.has(j)) return;
    under.add(s[0] + ',' + s[1]);
    const pad = cell * 0.13;
    rounded(g, s[0] * cell + pad, s[1] * cell + pad, cell - 2 * pad, cell - 2 * pad, cell * 0.15, s[2] ? C.req : C.shelf);
  });
  for (let r = 0; r < R; r++) {
    const a = p.robots[t][r], b = p.robots[t2][r];
    const x = a[0] + (b[0] - a[0]) * part, y = a[1] + (b[1] - a[1]) * part;
    const j = p.carry[t][r];
    if (j >= 0) {   // the shelf it carries, drawn as a frame around the robot
      const pad = cell * 0.09;
      rounded(g, x * cell + pad, y * cell + pad, cell - 2 * pad, cell - 2 * pad, cell * 0.2, C.floor, p.shelves[t][j][2] ? C.req : C.shelf, cell * 0.11);
    }
    g.globalAlpha = (j < 0 && under.has(a[0] + ',' + a[1])) ? 0.85 : 1;
    const turn = ((ANGLE[b[2]] - ANGLE[a[2]] + 540) % 360) - 180, angle = (ANGLE[a[2]] + turn * part) * Math.PI / 180;
    const cx = (x + .5) * cell, cy = (y + .5) * cell, half = cell * 0.25;
    g.save(); g.translate(cx, cy); g.rotate(angle);
    g.beginPath(); g.moveTo(half + cell * 0.14, 0); g.lineTo(half - 1, -cell * 0.12); g.lineTo(half - 1, cell * 0.12); g.closePath();
    g.fillStyle = ROBOT[r % 2]; g.fill(); g.restore();
    rounded(g, cx - half, cy - half, 2 * half, 2 * half, cell * 0.12, ROBOT[r % 2]);
    g.globalAlpha = 1;
    g.fillStyle = '#ffffff'; g.font = '700 ' + Math.round(cell * 0.3) + 'px system-ui, sans-serif'; g.fillText(String(r + 1), cx, cy + 1);
  }
  for (let age = 0; age < 12 && t - age >= 1; age++) {   // a delivery glows on its tile for a moment
    const past = t - age;
    for (let r = 0; r < R; r++) if (p.rewards[past][r]) {
      const a = p.robots[past][r], fade = Math.max(0, 1 - (age + part) / 12);
      g.globalAlpha = fade;
      rounded(g, a[0] * cell + 2, a[1] * cell + 2, cell - 4, cell - 4, 8, null, C.good, 4);
      g.fillStyle = C.good; g.font = '700 ' + Math.round(cell * 0.42) + 'px system-ui, sans-serif';
      g.fillText('+1', (a[0] + .5) * cell, (a[1] - .35 - 0.04 * (age + part)) * cell);
      g.globalAlpha = 1;
    }
  }
}

function drawChart(pos) {
  const [g, w] = fit(chart, 132), left = 26, right = 10, top = 8, bottom = 20, pw = w - left - right, ph = 132 - top - bottom;
  const best = Math.max(1, ...P.map(p => p.team[T]));
  const X = s => left + pw * s / T, Y = v => top + ph * (1 - v / best);
  g.font = '11px system-ui, sans-serif'; g.fillStyle = C.soft; g.strokeStyle = C.line; g.lineWidth = 1;
  g.textAlign = 'right'; g.textBaseline = 'middle';
  for (const v of [0, best]) { g.beginPath(); g.moveTo(left, Y(v)); g.lineTo(left + pw, Y(v)); g.stroke(); g.fillText(String(v), left - 6, Y(v)); }
  g.textBaseline = 'alphabetic';
  [[0, 'left'], [Math.round(T / 2), 'center'], [T, 'right']].forEach(([s, align]) => { g.textAlign = align; g.fillText('step ' + s, X(s), 128); });
  const t = Math.min(T, Math.floor(pos));
  P.slice().reverse().forEach((p, k) => {
    const main = k === P.length - 1, colour = main ? C.accent : C.other;
    if (main) {   // shade what has been played so far
      g.beginPath(); g.moveTo(X(0), Y(0));
      for (let s = 1; s <= t; s++) { g.lineTo(X(s), Y(p.team[s - 1])); g.lineTo(X(s), Y(p.team[s])); }
      g.lineTo(X(pos), Y(p.team[t])); g.lineTo(X(pos), Y(0)); g.closePath(); g.globalAlpha = 0.14; g.fillStyle = colour; g.fill(); g.globalAlpha = 1;
    }
    g.beginPath(); g.moveTo(X(0), Y(0));
    for (let s = 1; s <= T; s++) { g.lineTo(X(s), Y(p.team[s - 1])); g.lineTo(X(s), Y(p.team[s])); }
    g.strokeStyle = colour; g.lineWidth = 2; g.lineJoin = 'round'; g.stroke();
    g.beginPath(); g.arc(X(pos), Y(p.team[t]), 4.5, 0, 2 * Math.PI); g.fillStyle = colour; g.fill(); g.strokeStyle = C.card; g.lineWidth = 1.5; g.stroke();
  });
  g.beginPath(); g.moveTo(X(pos), top); g.lineTo(X(pos), top + ph); g.strokeStyle = C.soft; g.globalAlpha = 0.6; g.lineWidth = 1; g.stroke(); g.globalAlpha = 1;
}

const scrub = document.getElementById('scrub'), stepno = document.getElementById('stepno'), playButton = document.getElementById('play');
scrub.max = T;
let pos = 0, playing = false, rate = 14, last = null, dirty = true, shown = -1;

function showStep(t) {
  const main = P[0];
  if (P.length > 1) P.forEach((p, i) => { cardValues[i].textContent = p.team[t]; });
  else { cardValues[0].textContent = main.team[t]; for (let r = 0; r < Math.min(R, 2); r++) cardValues[r + 1].textContent = main.totals[t][r]; }
  P.forEach((p, i) => { boards[i].count.textContent = p.team[t] + (p.team[t] === 1 ? ' delivery' : ' deliveries'); });
  for (let r = 0; r < R; r++) jobText[r].textContent = JOBS[main.job(t, r)];
  stepno.textContent = 'step ' + t + ' / ' + T;
  const recent = main.events.filter(e => e[0] <= t).slice(-6).reverse();
  log.replaceChildren(...(recent.length ? recent.map(e => {
    const row = el('div'), chip = el('span', 'chip', String(e[1] + 1)); chip.style.background = ROBOT[e[1] % 2];
    row.append(chip, el('span', 'at', 'step ' + e[0]), el('span', e[3] ? 'what won' : 'what', e[2])); return row;
  }) : [el('div', 'none', 'Nothing has happened yet.')]));
}
function setPlaying(on) { playing = on; last = null; playButton.textContent = on ? 'Pause' : (pos >= T ? 'Replay' : 'Play'); }
function jump(to) { pos = Math.max(0, Math.min(T, to)); dirty = true; }
function tick(now) {
  if (playing) {
    if (last !== null) pos = Math.min(T, pos + (now - last) / 1000 * rate);
    last = now; dirty = true;
    if (pos >= T) setPlaying(false);
  }
  if (dirty) {
    dirty = false;
    P.forEach((p, i) => drawBoard(boards[i].canvas, p, pos));
    drawChart(pos);
    const t = Math.min(T, Math.floor(pos));
    if (t !== shown) { shown = t; showStep(t); scrub.value = t; }
  }
  requestAnimationFrame(tick);
}
playButton.onclick = () => { if (!playing && pos >= T) pos = 0; setPlaying(!playing); dirty = true; };
document.getElementById('back').onclick = () => { setPlaying(false); jump(Math.ceil(pos) - 1); };
document.getElementById('next').onclick = () => { setPlaying(false); jump(Math.floor(pos) + 1); };
document.getElementById('restart').onclick = () => { setPlaying(false); jump(0); playButton.textContent = 'Play'; };
scrub.oninput = () => { setPlaying(false); jump(Number(scrub.value)); };
document.getElementById('speed').onclick = event => {
  if (!event.target.dataset.rate) return;
  rate = Number(event.target.dataset.rate);
  for (const b of document.querySelectorAll('#speed button')) b.classList.toggle('on', b === event.target);
};
document.addEventListener('keydown', event => {
  if (event.code === 'Space') { event.preventDefault(); playButton.click(); }
  if (event.code === 'ArrowRight') document.getElementById('next').click();
  if (event.code === 'ArrowLeft') document.getElementById('back').click();
});
window.addEventListener('resize', () => { sizeBoards(); dirty = true; });
requestAnimationFrame(tick);
</script></body></html>
"""
