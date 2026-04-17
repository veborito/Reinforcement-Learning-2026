import gymnasium as gym
from stable_baselines3 import A2C

def random_event():
  # Initialise the environment
  env = gym.make("CartPole-v1", render_mode="human")

  rewards = 0
  # Reset the environment to generate the first observation
  observation, info = env.reset(seed=42)
  episode_over = False
  while not episode_over:
      # this is where you would insert your policy
      action = env.action_space.sample()

      # step (transition) through the environment with the action
      # receiving the next observation, reward and if the episode has terminated or truncated
      observation, reward, terminated, truncated, info = env.step(action)
      
      rewards += reward
      # If the episode has ended then we can reset to start a new episode
      episode_over = terminated or truncated
  observation.reset()
  print(f"Total baseline rewards: {rewards}")

  env.close()

def train():
  env = gym.make("CartPole-v1", render_mode=None)

  model = A2C("MlpPolicy", env, verbose=1)
  model.learn(total_timesteps=100_000)
  model.save("CartPole_A2C")

def test():
  env = gym.make("CartPole-v1", render_mode="human")
  observation, info = env.reset(seed=42)
  
  model = A2C.load("CartPole_A2C")
  rewards = 0
  # Reset the environment to generate the first observation
  episode_over = False
  while not episode_over:
      # this is where you would insert your policy
      action, _states = model.predict(observation)

      # step (transition) through the environment with the action
      # receiving the next observation, reward and if the episode has terminated or truncated
      observation, reward, terminated, truncated, info = env.step(action)
      
      rewards += reward
      # If the episode has ended then we can reset to start a new episode
      episode_over = terminated or truncated
  observation, info = env.reset()
  print(f"Total rewards: {rewards}")

  env.close()


train()
#test()
