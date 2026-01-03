import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random as pr

def rargmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False)

# Initialize table with 0
Q = np.zeros([env.observation_space.n, env.action_space.n])

# repeate 2000 times
num_episodes = 2000

# create lists to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    state, info = env.reset()
    rAll = 0
    done = False

    # The Q-Table learning algorithm
    while not done:
        action = rargmax(Q[state, :])

        # Get new state and reward from environment
        new_state, reward, terminated, truncated, info = env.step(action) 
        done = terminated or truncated

        # Update Q-Table with new knowledge
        Q[state, action] = reward + np.max(Q[new_state, :])

        rAll += reward
        state = new_state
    rList.append(rAll)

print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)

plt.bar(range(len(rList)), rList, color="blue")
plt.show()