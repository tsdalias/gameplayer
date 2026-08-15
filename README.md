# GamePlayer

Teaching an AI to play a video game with Deep Reinforcement Learning, built up
from zero AI knowledge.

**New here? Read [`PLAN.md`](PLAN.md) first** — it explains the fundamentals
(neural networks, reinforcement learning, DQN) in plain English and lays out the
whole roadmap. This README is just how to run things.

## Setup (one time)

From the project root (`C:\Projects\gameplayer`):

```powershell
# 1. Create an isolated Python environment so we don't touch your system Python.
python -m venv .venv

# 2. Activate it (PowerShell).
.\.venv\Scripts\Activate.ps1

# 3. Install the tools.
pip install -r requirements.txt
```

> If step 2 gives a "running scripts is disabled" error, run PowerShell once as:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again.

You'll know the environment is active when your prompt starts with `(.venv)`.

## Phase 1 — run it

**Step 1 — the random (non-learning) baseline.** Opens a window; the agent
presses random buttons and usually drops the pole in ~20 steps.

```powershell
python src/random_agent.py
```

**Step 2 — the learning agent (DQN).** Trains for ~50k steps (a couple of
minutes on CPU), saves the model, and reports its score. CartPole is "solved"
at ~500.

```powershell
python src/train_sb3.py
```

**Watch the trained agent play:**

```powershell
python src/train_sb3.py --play
```

**See the learning curves:**

```powershell
tensorboard --logdir runs
```

Then open the printed `http://localhost:6006` link in your browser.

## Where things are

| Path | What it is |
|------|------------|
| `PLAN.md` | The guide + full roadmap. Start here. |
| `src/random_agent.py` | Phase 1: no-learning baseline. |
| `src/train_sb3.py` | Phase 1: DQN via Stable-Baselines3. |
| `requirements.txt` | The tools to install. |
| `runs/` | TensorBoard training logs (created on first run). |
| `models/` | Saved trained agents (created on first run). |

## What's next

Phases 2–4 (build DQN from scratch, then move to Atari Pong) are described in
`PLAN.md`. The `src/dqn/` and `src/wrappers/` folders will be created then.
