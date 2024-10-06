# Positions for entities in observations (x, y), for use in XAI methods

# Index 0 in each list represents observations with no enemies and a single avocado, where the agent got stuck
# Index 1 in each list introduces 1 enemy to the environment
# Index 2 in each list introduces an additional avocado to the environment, in the same location as the enemy

ood_1_avo_stuck_0 = {
    "agent_position_list": [(2, 7), (2, 7), (2, 7)],
    "avocado_positions_list": [[(6, 5)], [(6, 5)], [(6, 5), (3, 9)]],
    "enemy_positions_list": [[], [(3, 9)], []]
}

ood_1_avo_stuck_1 = {
    "agent_position_list": [(6, 3), (6, 3), (6, 3)],
    "avocado_positions_list": [[(6, 5)], [(6, 5)], [(6, 5), (7, 3)]],
    "enemy_positions_list": [[], [(7, 3)], []]
}

ood_1_avo_stuck_2 = {
    "agent_position_list": [(6, 9), (6, 9), (6, 9)],
    "avocado_positions_list": [[(0, 6)], [(0, 6)], [(0, 6), (7, 7)]],
    "enemy_positions_list": [[], [(7, 7)], []]
}


ood_1_avo_stuck_positions = {
    "ood_1_avo_stuck_0": ood_1_avo_stuck_0,
    "ood_1_avo_stuck_1": ood_1_avo_stuck_1,
    "ood_1_avo_stuck_2": ood_1_avo_stuck_2,
}
