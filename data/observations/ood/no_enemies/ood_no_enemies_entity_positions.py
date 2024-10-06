# Positions for entities in observations (x, y), for use in XAI methods
# The first index of each position lists together form an out of distribution observation where there are no enemies
# The second index forms observations where two enemies are present to compare with the first ones

ood_1_no_enemies = {
    "agent_position_list": [(2, 7), (2, 7)],
    "avocado_positions_list": [[(6, 5)], [(6, 5)]],
    "enemy_positions_list": [[], [(0, 6), (2, 9)]]
}

ood_2_no_enemies = {
    "agent_position_list": [(6, 3), (6, 3)],
    "avocado_positions_list": [[(6, 5)], [(6, 5)]],
    "enemy_positions_list": [[], [(6, 2), (8, 3)]]
}

ood_3_no_enemies = {
    "agent_position_list": [(6, 9), (6, 9)],
    "avocado_positions_list": [[(0, 6)], [(0, 6)]],
    "enemy_positions_list": [[], [(7, 9), (7, 7)]]
}


ood_no_enemies_positions = {
    "ood_1_no_enemies": ood_1_no_enemies,
    "ood_2_no_enemies": ood_2_no_enemies,
    "ood_3_no_enemies": ood_3_no_enemies,
}
