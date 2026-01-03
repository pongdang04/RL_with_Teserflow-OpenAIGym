import gymnasium as gym

env = gym.make('CartPole-v1', render_mode='human') 
observation, info = env.reset()

random_episodes = 0
reward_sum = 0

while random_episodes < 10:
    env.render()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    print(observation, reward, done)
    reward_sum += reward
    
    if done:
        random_episodes += 1
        print("Reward for this episode was:", reward_sum)
        reward_sum = 0
        observation, info = env.reset()

env.close()