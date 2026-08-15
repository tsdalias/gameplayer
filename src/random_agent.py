"""
Phase 1, Step 1: A RANDOM agent.

This agent does NOT learn anything. On every step it presses a random button.
Its whole purpose is to (a) show you the agent<->environment loop from PLAN.md
Part 2, and (b) give us a baseline: "how good is no learning at all?"

Run it with:
    python src/random_agent.py

You'll see each episode's total reward printed. For CartPole, reward = how many
steps the pole stayed balanced. A random agent usually manages only ~20 steps
before it drops the pole. In Phase 1 Step 2 a *learning* agent will reach ~500.
"""

import gymnasium as gym


def run_random_agent(env_name: str = "CartPole-v1", episodes: int = 10) -> None:
    # render_mode="human" opens a window so you can WATCH it play.
    # If you're on a headless machine (no display), change it to None.
    env = gym.make(env_name, render_mode="human")

    for episode in range(1, episodes + 1):
        # reset() starts a fresh game and returns the first observation (state).
        state, info = env.reset()

        total_reward = 0.0
        done = False

        while not done:
            # The agent's "policy": pick a completely random valid action.
            # (No brain here — this is the point.)
            action = env.action_space.sample()

            # step() applies the action and returns what happened next:
            #   next_state  – the new observation
            #   reward      – the reward for THIS step
            #   terminated  – True if the episode ended naturally (pole fell)
            #   truncated   – True if it ended due to a time limit
            #   info        – extra debug info (ignored here)
            next_state, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            state = next_state
            done = terminated or truncated

        print(f"Episode {episode:2d} | total reward: {total_reward:.0f}")

    env.close()


if __name__ == "__main__":
    run_random_agent()
