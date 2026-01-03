import gymnasium as gym
from colorama import init
from kbhit import KBHit

init(autoreset=True)

env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=True, render_mode='ansi')
env.reset()
print(env.render())

key = KBHit()

while True:
    action = key.getarrow()
    if action not in [0, 1, 2, 3]:
        print("Game aborted!")
        break

    state, reward, terminated, truncated, info = env.step(action)
    print(env.render())
    print("State:", state, "Action:", action, "Reward:", reward, "Info:", info)

    if terminated or truncated:
        print("Finished with reward", reward)
        break