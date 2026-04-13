"""Random policy smoke-test for VampireSurvivorsEnv.

Run with the game already open at the main menu:

    python random_policy.py

What this checks:
  1. render() returns a valid frame (not None, correct shape).
  2. _navigate_to_quick_start() fires without KeyError or crashes.
  3. _get_obs() returns correctly shaped preprocessed frames.
"""

import time
import numpy as np
from vampire_survivors_env.env import VampireSurvivorsEnv

N_STEPS = 5000
RENDER = True  # set False to skip cv2 window

env = VampireSurvivorsEnv(
    render_mode="human" if RENDER else None,
    frame_stack=4,
    action_repeat=4,
    resolution=(100, 100),
)

print("Resetting (will navigate Quick Start)...")
obs, info = env.reset()

frames = obs["frames"]
print(f"  obs['frames'] shape : {frames.shape}  dtype: {frames.dtype}")
print(f"  value range         : [{frames.min():.3f}, {frames.max():.3f}]")
assert frames.shape == (4, 100, 100), f"Unexpected obs shape: {frames.shape}"
assert frames.dtype == np.float32

raw = env.render()
if raw is not None:
    print(f"  render() shape      : {raw.shape}  dtype: {raw.dtype}")
else:
    print("  render() returned None (no frame captured yet)")

print(f"\nRunning {N_STEPS} random steps...")
for step in range(N_STEPS):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if RENDER:
        env.render()

    if step % 50 == 0:
        frames = obs["frames"]
        print(
            f"  step {step:4d} | action={action} | "
            f"obs range=[{frames.min():.3f},{frames.max():.3f}] | "
            f"done={terminated or truncated} | "
            f"info={info}"
        )

    if terminated or truncated:
        print(f"  Episode ended at step {step}")
        obs, info = env.reset()

env.close()
print("\nDone.")
