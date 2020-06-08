from gym.envs.registration import register
register(
    id='KukaGraspTask-v0',
    entry_point='env.KukaGymEnv:KukaDiverseObjectEnv'
)