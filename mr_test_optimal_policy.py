from agents import MangoRunOptimalAgent
import gymnasium as gym
import mango_run


env = gym.make(id="MangoRun-v0", render_mode="human",
               agent_spawn_all_legal_locations=False,
               spawn_unripe_mangoes=True)

agent = MangoRunOptimalAgent(env)

agent.test_policy(
    episodes=1_000,
    render=True,
)
