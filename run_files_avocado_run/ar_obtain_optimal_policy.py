from agents import ArValueIterationAgent
import gymnasium as gym
import avocado_run


env = gym.make(id="AvocadoRun-v0", render_mode="human",
               num_avocados=1, num_enemies=1, aggressive_enemies=False)

agent = ArValueIterationAgent(env)

agent.find_optimal_policy(
    gamma=0.99,
    theta=1e-6,
    max_iterations=1000,
    file_path="optimal_policy/optimal_policy.npy",
)
