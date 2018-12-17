from gym.envs.registration import register

register(
    id='hotforest-v0',
    entry_point='gym_hotforest.envs:HotforestEnv',
)

register(
    id='hotforest-l2-v0',
    entry_point='gym_hotforest.envs:HotforestEnv',
    kwargs={'length': 2}
)

register(
    id='hotforest-l3-v0',
    entry_point='gym_hotforest.envs:HotforestEnv',
    kwargs={'length': 3}
)

register(
    id='hotforest-l4-v0',
    entry_point='gym_hotforest.envs:HotforestEnv',
    kwargs={'length': 4}
)

register(
    id='hotforest-l5-v0',
    entry_point='gym_hotforest.envs:HotforestEnv',
    kwargs={'length': 5}
)

register(
    id='hotforest-l8-v0',
    entry_point='gym_hotforest.envs:HotforestEnv',
    kwargs={'length': 8}
)

register(
    id='hotforest-l16-v0',
    entry_point='gym_hotforest.envs:HotforestEnv',
    kwargs={'length': 16}
)

register(
    id='hotforest-l32-v0',
    entry_point='gym_hotforest.envs:HotforestEnv',
    kwargs={'length': 32}
)

