import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim

env = gym.make('CartPole-v1')

# Constants defining our neural network
learning_rate = 1e-1
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

# Simple Q-network
model = nn.Linear(input_size, output_size)
nn.init.xavier_uniform_(model.weight)
nn.init.zeros_(model.bias)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Values
dis = .99
num_episodes = 2000
rList = []

start_time = time.time()

for i in range(num_episodes):
    e = 1. / ((i / 10) + 1)
    rAll = 0
    step_count = 0
    s, info = env.reset()
    done = False

    while not done:
        step_count += 1
        x = torch.FloatTensor(s).unsqueeze(0)
        
        # Get Q values
        with torch.no_grad():
            Qs = model(x).numpy().copy()
        
        # e-greedy action selection
        if np.random.rand() < e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)

        # Get new state and reward
        s1, reward, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        
        if done:
            Qs[0, a] = -100
        else:
            x1 = torch.FloatTensor(s1).unsqueeze(0)
            with torch.no_grad():
                Qs1 = model(x1).numpy()
            Qs[0, a] = reward + dis * np.max(Qs1)

        # Train
        optimizer.zero_grad()
        Q_pred = model(x)
        target = torch.FloatTensor(Qs)
        loss = torch.sum((target - Q_pred) ** 2)
        loss.backward()
        optimizer.step()
        
        s = s1

    rList.append(step_count)
    print(f"Episode: {i} steps: {step_count}")
    
    # If last 10's avg steps are 500, It's good enough
    if len(rList) > 10 and np.mean(rList[-10:]) > 500:
        break

print("--- %s seconds ---" % (time.time() - start_time))

# See our trained network in action
env_render = gym.make('CartPole-v1', render_mode='human')
observation, info = env_render.reset()
reward_sum = 0

while True:
    env_render.render()
    
    x = torch.FloatTensor(observation).unsqueeze(0)
    with torch.no_grad():
        Qs = model(x).numpy()
    a = np.argmax(Qs)

    observation, reward, terminated, truncated, info = env_render.step(a)
    reward_sum += reward
    
    if terminated or truncated:
        print(f"Total score: {reward_sum}")
        break

env_render.close()