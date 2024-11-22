from agents import ArValueIterationAgent
import gymnasium as gym
import avocado_run


env = gym.make(id="AvocadoRun-v0", render_mode="human",
               num_avocados=1, num_enemies=1, aggressive_enemies=False)

agent = ArValueIterationAgent(env)

agent.test_policy(
    policy_file_path="optimal_policy/optimal_policy.npy",
    episodes=1_000_000,
    render=False,
    verify_never_caught=True
)
