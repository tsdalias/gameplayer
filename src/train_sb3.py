"""
Phase 1, Step 2: A LEARNING agent, using a ready-made DQN.

Here we use Stable-Baselines3's DQN implementation. We are NOT writing the
learning algorithm ourselves yet (that's Phase 2) -- the point of this step is:
  1. Prove your whole setup works end to end.
  2. Watch a reward curve climb, so you SEE learning happen.
  3. Get a trusted reference result to compare our from-scratch DQN against later.

Run training with:
    python src/train_sb3.py

Then watch the trained agent play with:
    python src/train_sb3.py --play

See the learning curves with:
    tensorboard --logdir runs
    (then open the printed http://localhost:6006 link in a browser)
"""

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

ENV_NAME = "CartPole-v1"
MODEL_PATH = Path("models") / "dqn_cartpole"
LOG_DIR = "runs"

# ~50k steps is plenty to solve CartPole and only takes a couple of minutes on CPU.
TOTAL_TIMESTEPS = 50_000


def train() -> None:
    env = gym.make(ENV_NAME)

    # "MlpPolicy" = the Q-network is a small multi-layer perceptron (a plain
    # neural net), appropriate because CartPole's state is 4 numbers, not pixels.
    # In Phase 3 we'll switch to a CNN policy for image input.
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,               # print progress
        tensorboard_log=LOG_DIR, # write curves we can view in TensorBoard
    )

    print(f"Training DQN on {ENV_NAME} for {TOTAL_TIMESTEPS:,} steps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}.zip")

    # Quick numeric check: average reward over 10 fresh episodes.
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Evaluation: mean reward {mean_reward:.1f} +/- {std_reward:.1f}")
    print("(CartPole is 'solved' at ~500. Random agents score ~20.)")


def play() -> None:
    # Load the saved model and watch it play in a window.
    env = gym.make(ENV_NAME, render_mode="human")
    model = DQN.load(MODEL_PATH)

    for episode in range(1, 6):
        state, info = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            # deterministic=True -> always take the action the agent thinks is best.
            action, _ = model.predict(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {episode} | total reward: {total_reward:.0f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or watch a DQN on CartPole.")
    parser.add_argument(
        "--play",
        action="store_true",
        help="Load the saved model and watch it play (instead of training).",
    )
    args = parser.parse_args()

    if args.play:
        play()
    else:
        train()
