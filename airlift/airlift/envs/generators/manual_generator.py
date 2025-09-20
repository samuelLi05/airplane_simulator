
from typing import Tuple, List, Set, Dict
import json

#from airlift.envs import AirliftEnv

from airlift.envs.agents import EnvAgent
from airlift.envs.airport import Airport
from airlift.envs.cargo import Cargo
from airlift.envs.generators.world_generators import WorldGenerator
from airlift.envs.generators.cargo_generators import StaticCargoGenerator
from airlift.envs.generators.airplane_generators import AirplaneGenerator
from airlift.envs.generators.airport_generators import RandomAirportGenerator
from airlift.envs.generators.route_generators import RouteByDistanceGenerator
from airlift.envs.plane_types import PlaneType
from airlift.envs.route_map import RouteMap
from airlift.envs.world_map import FlatCoordinate, FlatLandOnlyMap

class DummyAirportGenerator():
    def __init__(self, processing_time):
        self.processing_time = processing_time


class ManualWorldGenerator(WorldGenerator):
    """
    Current in progress to mimic behavior of the World Generator. However we can use the generate method to extract
    necessary data from the provided JSON files. 
    """
    def __init__(self, json_file_path: str):
        super().__init__()
        self.json_file_path = json_file_path
        with open(self.json_file_path, 'r') as f:
            self.global_state = json.load(f)
        #self.airport_generator = DummyAirportGenerator(processing_time=10)
        

    def generate(self, old_routemap:RouteMap) -> Tuple[RouteMap, List[EnvAgent], Set[Cargo]]:
        """
        Generates airplane, routemap, and cargo data from provided JSON file. 

        :param old_routemap: The routemap initally loaded in from the pkl file to build off of
        """
        # Create Airports
        airports = []
        for node_id, node_data in self.global_state['route_map']['json://0']['_node'].items():
            airport_id = int(node_id.split('json://')[-1])
            if isinstance(node_data['pos'], dict):
                pos_data = node_data['pos']['py/seq']
            else:
                pos_data = node_data['pos']
            #pos_data = node_data['pos']['py/seq']
            position = FlatCoordinate(pos_data)
            working_capacity = node_data.get('working_capacity', 1) # Default to 1 if missing
            airports.append(Airport(id=airport_id, pos=position, working_capacity=working_capacity))

        routemap = RouteMap(map=old_routemap.map, airports=airports, plane_types=self.plane_types, drop_off_area=old_routemap.drop_off_area, pick_up_area=old_routemap.pick_up_area)
        routemap.poisson_dist = old_routemap.poisson_dist
        for start_node, end_nodes in self.global_state['route_map']['json://0']['_adj'].items():
            start_airport_id = int(start_node.split('json://')[-1])
            for end_node, edge_data in end_nodes.items():
                end_airport_id = int(end_node.split('json://')[-1])
                
                start_airport = routemap.airports_by_id[start_airport_id]
                end_airport = routemap.airports_by_id[end_airport_id]
                
                # Assuming all routes are for the first plane type for simplicity
                plane_type = routemap.plane_types_by_id[0]

                routemap.add_route(
                    plane=plane_type,
                    start=start_airport,
                    end=end_airport,
                    time=edge_data.get('time', 1),
                    cost=edge_data.get('cost', 1),
                    malfunction_generator=None,
                    bidirectional=False
                )

        # Create Airplanes
        airplanes = []
        for agent_id, agent_data in self.global_state['agents'].items():
            start_airport = routemap.airports_by_id[agent_data['current_airport']]
            plane_type = routemap.plane_types_by_id[agent_data['plane_type']]
            
            agent = EnvAgent(
                start_airport=start_airport,
                routemap=routemap,
                plane_type=plane_type,
                max_loaded_weight=agent_data.get('max_weight', 10) # Default to 10 if missing
            )
            airplanes.append(agent)

        # Create Cargo
        cargo = set()
        for cargo_data in self.global_state['active_cargo']:
            cargo_item = Cargo(
                id=cargo_data.get('id', -1),
                source_airport=routemap.airports_by_id[cargo_data['location']],
                end_airport=routemap.airports_by_id[cargo_data['destination']],
                weight=cargo_data.get('weight', 1),
                earliest_pickup_time=0,#cargo_data.get('earliest_pickup_time', 0),
                soft_deadline=cargo_data.get('soft_deadline', 100),
                hard_deadline=cargo_data.get('hard_deadline', 200)
            )
            cargo.add(cargo_item)
            routemap.airports_by_id[cargo_data['location']].add_cargo(cargo_item)

        return routemap, airplanes, cargo
    
     # Various properties which do not change (these can be accessed before seeding/generation). 

     # TODO to replace the world generator, but need to add additional generators here

    @property
    def num_agents(self):
        return len(self.global_state['agents']) # FROM JSON
    
    @property
    def plane_types(self):
        planes = []
        for pt in self.global_state['plane_types']:
            if isinstance(pt, dict):
                plane_data = pt['py/seq']
                planes.append(PlaneType(id=plane_data[0], max_weight=plane_data[1]))
            else:
                planes.append(PlaneType(id=pt[0], max_weight=pt[1]))
        return planes
    # not taking into account singular non list object loading
    #[PlaneType(id=pt['py/seq'][0], max_weight=pt['py/seq'][1]) for pt in self.global_state['plane_types']] #FROM JSON
    
    @property
    def max_airports(self):
        return len(self.global_state['route_map']['json://0']['_node']) # From JSON

    @property
    def max_cargo_per_episode(self):
        return 140

    @property
    def soft_deadline_multiplier(self):
        return 40

    @property
    def hard_deadline_multiplier(self):
        return 120

    @property
    def route_malfunction_rate(self):
        return 0.2

    @property
    def route_malfunction_max_duration(self):
        return 20

    @property
    def route_malfunction_min_duration(self):
        return 10
    

if __name__ == "__main__":
    routes, airplanes, cargo =ManualWorldGenerator("./database/example.json").generate()
    print (routes)
    print (airplanes)
    print (cargo)
