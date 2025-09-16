from gym.utils import seeding
from airlift.envs.generators.cargo_generators import DynamicCargoGenerator
from airlift.tests.util import generate_environment
from gym import logger

logger.set_level(logger.WARN)


def test_dynamic_cargo(render):
    seed = 55551
    np_random, seed = seeding.np_random(seed)
    num_of_tasks = 1
    max_dynamic_cargo = 3
    dynamic_cargo_generator = DynamicCargoGenerator(cargo_creation_rate=1000,
                                                    num_initial_tasks=num_of_tasks,
                                                    max_cargo_to_create=3,
                                                    soft_deadline_multiplier=50,
                                                    hard_deadline_multiplier=100)
    env = generate_environment(num_of_airports=3, num_of_agents=1, cargo_generator=dynamic_cargo_generator)
    env.reset(seed)
    env.step(None)
    assert len(env.cargo) == num_of_tasks + 1
    env.step(None)
    assert len(env.cargo) == num_of_tasks + 2
    env.step(None)
    assert len(env.cargo) == num_of_tasks + max_dynamic_cargo

    # Should not generate more than 3 (Max Dynamic Cargo)
    env.step(None)
    assert len(env.cargo) == num_of_tasks + max_dynamic_cargo

    # Cargo IDs should be incremental from initialized tasks
    sort_list = sorted(env.cargo, key=lambda value: value.id, reverse=False)
    for index, cargo in enumerate(sort_list):
        assert index == cargo.id



