import numpy as np
import random
from collections import deque
import gymnasium as gym
from dqn import DQN

env = gym.make('CartPole-v1')

# Constants defining our neural network
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

dis = 0.9
REPLAY_MEMORY = 50000

def ddqn_replay_train(mainDQN, targetDQN, train_batch):
    '''
    Double DQN implementation
    '''
    x_stack = np.empty(0).reshape(0, mainDQN.input_size)
    y_stack = np.empty(0).reshape(0, mainDQN.output_size)

    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

        if done:
            Q[0, action] = reward
        else:
            # Double DQN: y = r + gamma * targetDQN(s')[a] where a = argmax(mainDQN(s'))
            Q[0, action] = reward + dis * targetDQN.predict(next_state)[0, np.argmax(mainDQN.predict(next_state))]

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

    return mainDQN.update(x_stack, y_stack)

def bot_play(mainDQN, env):
    state, info = env.reset()
    reward_sum = 0
    while True:
        env.render()
        action = np.argmax(mainDQN.predict(state))
        state, reward, terminated, truncated, info = env.step(action)
        reward_sum += reward
        if terminated or truncated:
            print(f"Total score: {reward_sum}")
            break

def main():
    max_episodes = 5000
    replay_buffer = deque()

    mainDQN = DQN(input_size, output_size, name="main")
    targetDQN = DQN(input_size, output_size, name="target")
    
    # Initial copy main -> target
    targetDQN.copy_from(mainDQN)

    for episode in range(max_episodes):
        e = 1. / ((episode / 10) + 1)
        done = False
        step_count = 0
        state, info = env.reset()

        while not done:
            if np.random.rand() < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(mainDQN.predict(state))

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if done:
                reward = -100

            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > REPLAY_MEMORY:
                replay_buffer.popleft()

            state = next_state
            step_count += 1
            if step_count > 10000:
                break

        print(f"Episode: {episode} steps: {step_count}")
        
        if step_count > 10000:
            pass

        if episode % 10 == 1 and len(replay_buffer) > 10:
            for _ in range(50):
                minibatch = random.sample(replay_buffer, min(10, len(replay_buffer)))
                loss, _ = ddqn_replay_train(mainDQN, targetDQN, minibatch)

            print(f"Loss: {loss}")
            targetDQN.copy_from(mainDQN)

    # Test trained bot
    env_render = gym.make('CartPole-v1', render_mode='human')
    for i in range(5):
        bot_play(mainDQN, env_render)
    env_render.close()

if __name__ == "__main__":
    main()