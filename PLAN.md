# GamePlayer — Teaching an AI to Play a Video Game (from zero)

This document is both a **plan** and a **beginner's guide**. It assumes you have
**no prior AI knowledge**. We build up the ideas from scratch — what a neural
network even is, what "learning" means — and only then get into the project
itself. Read it top to bottom; each section relies on the one before it.

The goal: build an AI that learns to play an Atari video game (starting with
Pong) by trial and error, the same *kind* of system DeepMind became famous for
in 2013–2015.

- **Target environment:** Atari (starting with Pong)
- **Your goal:** learn the concepts hands-on
- **Your hardware:** laptop / CPU only

---

# Part 1 — The Fundamentals (no code yet)

## 1.1 What is "AI" here, really?

Forget robots and sci-fi. In this project, "AI" means one concrete thing:

> A **function** that takes some input (what the game screen looks like) and
> produces an output (which button to press), and that **improves itself** by
> practicing.

That's it. The whole field is about building functions that get better with
experience instead of being programmed by hand. We never write the rule
"if the ball is here, move up." Instead we build a system that *figures that out
on its own*.

## 1.2 Three families of Machine Learning

**Machine Learning (ML)** = getting computers to learn patterns from data
instead of following hand-written rules. There are three main styles:

1. **Supervised learning** — learn from labeled examples.
   *Example:* 10,000 photos labeled "cat" or "dog"; the system learns to tell
   them apart. You need the correct answer for every example.

2. **Unsupervised learning** — find structure in data with no labels.
   *Example:* group customers into segments without being told the groups.

3. **Reinforcement learning (RL)** — learn by *doing* and receiving *rewards*.
   *Example:* an agent plays a game, wins or loses, and gradually learns which
   actions lead to winning. **This is what our project uses.**

RL is special: there's no teacher giving the right answer. There's only a
**reward signal** (score went up = good), and the agent must discover a
strategy on its own. This is why it feels the most like "an entity that learns."

## 1.3 What is a neural network?

A **neural network** is just a particular kind of adjustable function. Picture it
as a machine with lots of **knobs** (millions of them, called *weights*):

```
   input  ──►  [ big function with millions of adjustable knobs ]  ──►  output
 (game pixels)                                                     (which button)
```

- The **input** is numbers (a game screen is just a grid of pixel brightness values).
- The **output** is numbers too (e.g. a score for each possible button).
- In between are the **weights** — the knobs. Different knob settings make the
  network compute different things.

"**Deep** learning" just means the network has **many layers** stacked up
(input → layer → layer → … → output). More layers let it learn more complex
patterns. A **CNN** (Convolutional Neural Network) is a type of deep network
specialized for images — it's what lets our agent understand a game screen.

**Key mental model:** a neural network starts out useless (random knobs). Learning
= automatically adjusting the knobs until the outputs are good.

## 1.4 What does "learning" / "training" actually mean?

Training is a loop that tunes the knobs:

1. **Predict** — feed an input through the network, get an output.
2. **Measure error** — compare the output to what we wanted, using a number
   called the **loss** (high loss = very wrong, low loss = nearly right).
3. **Adjust** — nudge every knob a tiny bit in the direction that *reduces* the
   loss. The math trick that computes "which way to nudge each knob" is called
   **gradient descent** (via *backpropagation*). You don't need the math yet —
   just know the library (PyTorch) does this for you.
4. **Repeat** millions of times.

Over many repetitions the loss goes down and the network gets good. That's all
"training a model" means: repeatedly nudging knobs to reduce error.

The catch in RL: nobody hands us "what we wanted" (step 2). We have to *construct*
a target from the rewards the agent experiences. That construction is the clever
part, explained next.

---

# Part 2 — How an Agent Learns to Play (the RL core)

## 2.1 The vocabulary (learn these five words)

RL is a loop between an **agent** and an **environment** (the game):

```
        ┌─────────────────────────────────────────────┐
        │                                             │
        ▼                                             │
   ┌─────────┐   action (press a button)        ┌───────────┐
   │  AGENT  │ ───────────────────────────────► │    GAME   │
   │ (our AI)│                                   │  (Pong)   │
   │         │ ◄─────────────────────────────── │           │
   └─────────┘   new state (screen) + reward     └───────────┘
```

- **State** — the current situation (the game screen right now).
- **Action** — what the agent can do (move up, move down, do nothing).
- **Reward** — a number the game gives back (+1 for scoring, -1 for being scored on).
- **Policy** — the agent's strategy: given a state, which action to take. *This is
  what we're ultimately training.*
- **Discount (γ, "gamma")** — how much the agent cares about future rewards vs.
  immediate ones (e.g. 0.99 = cares a lot about the future). This lets it learn
  that a move now can pay off several steps later.

The whole setup — states, actions, rewards, transitions — is formally called a
**Markov Decision Process (MDP)**. That's just the textbook name for this loop.

## 2.2 The core idea: value of an action (Q-values)

Suppose that for any screen, the agent could estimate:

> "If I press **UP** now, how much total reward will I eventually collect?"
> "If I press **DOWN** now, how much?"

Call these estimates **Q-values** (Q for "quality"). If the agent had accurate
Q-values, playing well is trivial: **in every state, pick the action with the
highest Q-value.** Done.

So the entire problem reduces to: **learn accurate Q-values.** This approach is
called **Q-learning**.

## 2.3 Deep Q-Learning (DQN) — the DeepMind breakthrough

For a tiny game you could store Q-values in a table. But a game screen has
astronomically many possible states — no table could hold them all. DeepMind's
insight: **use a neural network to estimate Q-values** instead of a table.

```
   game screen ──►  [ neural network ]  ──►  Q-value for UP
                                            Q-value for DOWN
                                            Q-value for DO-NOTHING
```

This is a **Deep Q-Network (DQN)**. The network's job: look at the screen, output
a Q-value for each button. Training = making those Q-value estimates accurate.

**How do we get a "target" to train toward (remember Part 1.4 needed one)?**
We use the reward we actually received plus the network's own estimate of the
next state's value:

> target Q  =  reward we just got  +  γ × (best Q-value in the next state)

The network is nudged so its prediction moves toward this target. Over millions
of steps, the estimates become self-consistent and accurate. (This equation is
the **Bellman equation** — again, just the textbook name.)

## 2.4 The two tricks that made DQN actually work

Naively training the above is unstable and usually fails. DeepMind added two
fixes that are the heart of the 2015 Nature paper:

1. **Experience replay.** Instead of learning from each moment as it happens
   (consecutive frames are almost identical, which confuses the network), store
   every experience `(state, action, reward, next state)` in a big memory buffer.
   Then train on **random samples** from that memory. This breaks up correlations
   and reuses old experiences efficiently — a bit like studying from shuffled
   flashcards instead of always in order.

2. **Target network.** The training target in 2.3 uses the network's *own*
   estimates — so the target moves every time we update, like chasing your own
   shadow. Fix: keep a **frozen copy** of the network to compute targets, and only
   update that copy occasionally. This gives a stable target to aim at.

3. **Exploration vs. exploitation (ε-greedy).** If the agent always does what it
   currently thinks is best, it never discovers better moves. So early on it acts
   **randomly** a lot (explore), and over time shifts to acting on what it's
   learned (exploit). The knob controlling "chance of a random action" is **ε
   ("epsilon")**, and we shrink it as training progresses.

If you understand these three ideas, you understand DQN. Everything below is
putting them into practice.

---

# Part 3 — Reality Check: CPU-only Atari

DeepMind's DQN trained on **~200 million frames**. On a single laptop CPU that is
**weeks** of nonstop computation — not practical. So this plan is deliberately
staged:

1. Learn every concept on **cheap, fast toy problems** first (minutes to train).
2. Then run a **scaled-down** Atari agent where learning is **visible in hours,
   not weeks**.

The same code that solves the toy problem is ~90% of the code that plays Pong.
When you reach the heavy pixel training, a **free Colab or Kaggle GPU** turns
days into hours at no cost — and your code doesn't change.

| Expectation | CPU reality |
|---|---|
| Toy task (CartPole) solved | Minutes ✅ |
| Pong showing real learning | Hours ✅ |
| Full DQN benchmark on Pong | Days–weeks ⚠️ (use a free cloud GPU) |
| Many Atari games at benchmark scores | Not practical on CPU ❌ |

> **What is CartPole?** A minimal practice game built into the RL toolkit: balance
> a pole on a moving cart by pushing left or right. It has the exact same
> agent/environment/reward structure as Atari but is tiny, so it trains in
> minutes — perfect for learning the pipeline before we touch pixels.

---

# Part 4 — The Build Plan (phase by phase)

Each phase has a concrete **deliverable** so you always know if it worked.

## Phase 0 — Foundations
You just did this by reading Parts 1–2.
**Deliverable:** you can explain, in plain English, the agent/environment loop
and the three DQN tricks (replay, target network, ε-greedy).

## Phase 1 — Set up tools + solve a toy game (~day 1)
- **Install the toolkit** (see Part 5): Python, PyTorch, Gymnasium, Stable-Baselines3.
- **Run a random agent** on CartPole — it just presses random buttons. Purpose:
  see the loop (`reset` the game, `step` with an action, read the reward) and get
  a baseline for how bad "no learning" is.
- **Train a DQN on CartPole** using a ready-made library (Stable-Baselines3).
  Solves in ~1–2 minutes on CPU. Purpose: prove your setup works and watch a
  reward curve climb.

**Deliverable:** a plot of reward going up — your first learning agent.

## Phase 2 — Build DQN yourself (~days 2–4)
Now re-create DQN from scratch in PyTorch, still on CartPole so each experiment
runs in minutes. You'll write the pieces from Part 2 with your own hands:
- the **Q-network** (a small neural net),
- the **replay buffer** (the memory),
- the **target network**,
- **ε-greedy** action selection,
- the **training step** that nudges predictions toward the Bellman target.

This is the phase where everything clicks. It's roughly 200 lines of code.

**Deliverable:** your own DQN matching the library's CartPole result.

## Phase 3 — Go to pixels: Atari Pong (~days 5–8)
Swap the toy game for a real one. The only *new* idea is handling images:
- **Preprocess frames** — convert to grayscale, shrink to 84×84, and stack the
  last 4 frames together so the network can perceive motion (one still frame
  can't tell you which way the ball is moving).
- **Add a CNN** to the front of your Q-network so it can "see." Everything else
  from Phase 2 stays the same.
- **Train on Pong** (`PongNoFrameskip-v4`) with reduced scope. Expect to *see*
  the agent start returning the ball within a few hours, even if it doesn't reach
  benchmark scores on CPU.
- Run Stable-Baselines3's version alongside as a correctness reference.
- **Recommended:** do the heavy training run on a free Colab/Kaggle GPU.

**Deliverable:** a video of your agent playing Pong noticeably better than random.

## Phase 4 — Understand & improve (ongoing)
- **Visualize training** with TensorBoard (watch loss, reward, and ε over time).
- **Add one upgrade** DeepMind later published — **Double DQN** or **Dueling DQN**
  (small code changes, big "aha" value) — and compare before/after.
- **Optional:** try **PPO**, a different modern algorithm (the *policy-gradient*
  family) that often trains faster and more stably than DQN.

---

# Part 5 — Tech Stack (what each tool is for)

- **Python 3.10+** — the programming language everything is written in.
- **PyTorch** — the deep-learning library. It builds neural networks and does the
  "nudge the knobs" math (gradient descent) for you. Install the **CPU build**
  locally; the GPU build on cloud.
- **Gymnasium** — a standard collection of RL environments (games) with a common
  interface (`reset`, `step`). CartPole and the Atari games come from here.
- **ale-py** — the Arcade Learning Environment; provides the actual Atari games.
- **Stable-Baselines3** — ready-made, trustworthy implementations of DQN/PPO. We
  use it as a reference and sanity check against our from-scratch code.
- **TensorBoard** — a dashboard that plots training curves so you can *see* learning.

---

# Part 6 — Proposed Project Structure

Nothing here exists yet — this is the target layout we'll create in Phase 1.

```
gameplayer/
├── PLAN.md                  # this file
├── requirements.txt         # the list of tools to install
├── README.md
├── src/
│   ├── random_agent.py      # Phase 1: the no-learning baseline
│   ├── train_sb3.py         # Phase 1: DQN via Stable-Baselines3
│   ├── dqn/
│   │   ├── network.py       # the Q-network (MLP → later a CNN)
│   │   ├── replay_buffer.py # the experience memory
│   │   ├── agent.py         # ε-greedy + target network + learn step
│   │   └── train.py         # Phase 2/3: the from-scratch training loop
│   └── wrappers/
│       └── atari.py         # image preprocessing: grayscale/resize/stack
├── runs/                    # TensorBoard logs
└── videos/                  # recorded gameplay
```

---

# Part 7 — Milestones (definition of done)

- [ ] Random agent runs and reports episode scores.
- [ ] Library DQN solves CartPole (score ~500).
- [ ] Your from-scratch DQN solves CartPole.
- [ ] Your from-scratch DQN + CNN shows learning on Pong.
- [ ] One improvement (Double or Dueling DQN) implemented and compared.

---

# Part 8 — Glossary (quick reference)

- **Agent** — the AI that makes decisions.
- **Environment** — the game the agent plays.
- **State** — the current situation (a game screen).
- **Action** — a move the agent can make (a button).
- **Reward** — feedback number from the environment (+ good, − bad).
- **Policy** — the agent's strategy (state → action).
- **Q-value** — estimated total future reward of an action in a state.
- **Neural network** — an adjustable function with many tunable weights.
- **Weights / parameters** — the "knobs" that get tuned during training.
- **CNN** — a neural network specialized for images.
- **Loss** — a number measuring how wrong a prediction is.
- **Gradient descent** — the method that nudges weights to reduce loss.
- **Discount (γ)** — how much future rewards count vs. immediate ones.
- **Epsilon (ε)** — probability of taking a random action (exploration).
- **Experience replay** — training from a memory of shuffled past experiences.
- **Target network** — a frozen network copy used to compute stable targets.
- **MDP** — Markov Decision Process; the formal name for the RL loop.
- **DQN** — Deep Q-Network; a neural net that estimates Q-values.

---

# Part 9 — References (to go deeper later)

- Mnih et al., *Playing Atari with Deep Reinforcement Learning* (2013) — the first paper.
- Mnih et al., *Human-level control through deep reinforcement learning*, Nature (2015) — the famous one.
- Sutton & Barto, *Reinforcement Learning: An Introduction* — the free, standard textbook.
- Gymnasium docs — https://gymnasium.farama.org
- Stable-Baselines3 docs — https://stable-baselines3.readthedocs.io
