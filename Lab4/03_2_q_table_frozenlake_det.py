import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False)

# Initialize table with 0
Q = np.zeros([env.observation_space.n, env.action_space.n])
# learning parameters
dis = .99
num_episodes = 2000

# create lists(total rewards and steps per episode)
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    state, info = env.reset()
    rAll = 0
    done = False

    e = 1. / ((i // 100) + 1)

    # The Q-Table learning algorithm
    while not done:
        # Choose an action by e-greedy
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        # Get new state and reward from environment
        new_state, reward, terminated, truncated, info = env.step(action) 
        done = terminated or truncated

        # Update Q-Table with new knowledge using decay rate
        Q[state, action] = reward + dis * np.max(Q[new_state, :])

        rAll += reward
        state = new_state
    rList.append(rAll)

print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color='b', alpha=0.4)
plt.show()