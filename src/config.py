BASE_ENV_CONFIG = {
    "observation": {
        "type": "Kinematics"
    },
    "action": {
        "type": "DiscreteMetaAction"
    },
    "simulation_frequency": 5,
    "policy_frequency": 1,

    "lanes_count": 3,
    "vehicles_count": 20,
    "controlled_vehicles": 1,
    "duration": 30,

    "collision_reward": -1.0,
    "right_lane_reward": 0.1,
    "high_speed_reward": 0.4,
    "lane_change_reward": 0.0,
    "reward_speed_range": [20, 30],

    "normalize_reward": True,
    "offroad_terminal": False
}

NO_RIGHT_LANE_CONFIG = {
    "observation": {
        "type": "Kinematics"
    },
    "action": {
        "type": "DiscreteMetaAction"
    },
    "simulation_frequency": 5,
    "policy_frequency": 1,

    "lanes_count": 3,
    "vehicles_count": 20,
    "controlled_vehicles": 1,
    "duration": 30,

    "collision_reward": -1.0,
    "right_lane_reward": 0.0,
    "high_speed_reward": 0.4,
    "lane_change_reward": 0.0,
    "reward_speed_range": [20, 30],

    "normalize_reward": True,
    "offroad_terminal": False
}

HIGH_SPEED_LANE_CONFIG = {
    "observation": {"type": "Kinematics"},
    "action": {"type": "DiscreteMetaAction"},

    "simulation_frequency": 10,
    "policy_frequency": 1,

    "lanes_count": 3,
    "vehicles_count": 20,
    "controlled_vehicles": 1,
    "duration": 30,

    "collision_reward": -2.0,
    "right_lane_reward": 0.0,
    "high_speed_reward": 0.1,
    "lane_change_reward": -0.05,
    "reward_speed_range": [25, 30],

    "normalize_reward": False,
    "offroad_terminal": False
}

OBS_3_VEHICLES_CONFIG = {
    **HIGH_SPEED_LANE_CONFIG,
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 3,
        "normalize": True
    }
}

OBS_7_VEHICLES_CONFIG = {
    **HIGH_SPEED_LANE_CONFIG,
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 7,
        "normalize": True
    }
}
