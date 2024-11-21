from gymnasium.envs.registration import register

register(
    id='CorridorRun-v0',
    entry_point='corridor_run.envs:CorridorRun',
    max_episode_steps=200,
)
