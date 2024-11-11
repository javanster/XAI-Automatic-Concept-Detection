from gymnasium.envs.registration import register

register(
    id='MangoRun-v0',
    entry_point='mango_run.envs:MangoRun',
    max_episode_steps=200,
)
