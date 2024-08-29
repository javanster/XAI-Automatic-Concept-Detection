from AvocadoRunEnv import AvocadoRunEnv
from QLAgent import QLAgent
from DQLAgent import DQLAgent

ql_env = AvocadoRunEnv(moving_enemy=True)
dql_env = AvocadoRunEnv(observations_as_images=True,
                        moving_enemy=True, num_enemies=2)

ql_agent = QLAgent(ql_env)
dql_agent = DQLAgent(dql_env)

# ql_agent.train(episodes=1_000_000)
# ql_agent.test("qtables/qtable-1724926991.pickle")
# dql_agent.train(500)
dql_agent.test(
    "models/256x2__500.00max__494.10avg__470.00min__1724893683.keras")
