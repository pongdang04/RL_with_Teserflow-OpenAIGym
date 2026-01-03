import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim

env = gym.make('FrozenLake-v1', is_slippery=True)

input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1

# Simple linear model (Q-network)
model = nn.Linear(input_size, output_size, bias=False)
nn.init.uniform_(model.weight, 0, 0.01)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Set Q-learning parameters
dis = .99
num_episodes = 2000

rList = []

one_hot_matrix = torch.eye(input_size)

def one_hot(x):
    return one_hot_matrix[x:x+1]

start_time = time.time()

for i in range(num_episodes):
    s, info = env.reset()
    e = 1. / ((i / 50) + 10)
    rAll = 0
    done = False

    while not done:
        # Get Q values
        with torch.no_grad():
            Qs = model(one_hot(s)).clone()

        # e-greedy action selection
        if np.random.rand() < e:
            a = env.action_space.sample()
        else:
            a = int(torch.argmax(Qs))

        # Get new state and reward
        s1, reward, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        # Update target Q value
        if done:
            Qs[0, a] = reward
        else:
            with torch.no_grad():
                Qs1 = model(one_hot(s1))
            Qs[0, a] = reward + dis * torch.max(Qs1).item()

        # Train
        optimizer.zero_grad()
        Q_pred = model(one_hot(s))
        loss = torch.sum((Qs - Q_pred) ** 2)
        loss.backward()
        optimizer.step()

        rAll += reward
        s = s1

    rList.append(rAll)
    
    # current state print
    if (i + 1) % 100 == 0:
        print(f"Episode {i+1}/{num_episodes}, Recent success rate: {sum(rList[-100:])/100:.2%}")

print("--- %s seconds ---" % (time.time() - start_time))
print("Success rate: " + str(sum(rList) / num_episodes))
plt.bar(range(len(rList)), rList, color='b', alpha=0.4)
plt.show()