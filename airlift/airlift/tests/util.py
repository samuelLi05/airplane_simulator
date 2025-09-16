from airlift.envs import AirliftEnv, AirliftWorldGenerator, PlaneType, FlatRenderer
from airlift.envs import StaticCargoGenerator
from airlift.envs import NoEventIntervalGen
from airlift.envs import AirplaneGenerator
from airlift.envs import RandomAirportGenerator
from airlift.envs import RouteByDistanceGenerator


def generate_environment(num_of_airports: int = 5,
                         num_of_agents: int = 1,
                         processing_time: int = 10,
                         working_capacity: int = 1,
                         make_drop_off_area=False,
                         make_pick_up_area=False,
                         malfunction_generator=NoEventIntervalGen(),
                         poisson_lambda=0,
                         cargo_generator=StaticCargoGenerator(1),
                         plane_types=[PlaneType(id=0, model='A0', max_range=2.0, speed=0.1, max_weight=5)],
                         max_cycles=10 ** 4):
    return AirliftEnv(
        world_generator=AirliftWorldGenerator(
            plane_types=plane_types,
            airport_generator=RandomAirportGenerator(max_airports=num_of_airports,
                                                     processing_time=processing_time,
                                                     working_capacity=working_capacity,
                                                     make_drop_off_area=make_drop_off_area,
                                                     make_pick_up_area=make_pick_up_area,
                                                     num_drop_off_airports=1,
                                                     num_pick_up_airports=1),
            route_generator=RouteByDistanceGenerator(malfunction_generator=malfunction_generator, poisson_lambda=poisson_lambda),
            cargo_generator=cargo_generator,
            airplane_generator=AirplaneGenerator(num_of_agents),
            max_cycles=max_cycles
        ),
        renderer=FlatRenderer(show_routes=False)
    )
