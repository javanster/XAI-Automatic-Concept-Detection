from gymnasium.envs.registration import register

register(
    id='AvocadoRun-v0',
    entry_point='avocado_run.envs:AvocadoRunEnv',
    max_episode_steps=200,
)
