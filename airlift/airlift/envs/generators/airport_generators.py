import math
import warnings
from typing import Tuple, Collection, List, Optional
import numpy as np
from gym.utils import seeding
from airlift.envs.airport import Airport
from airlift.envs.generators.map_generators import PlainMapGenerator, FlatMapGenerator
from airlift.envs.world_map import FlatCoordinate, FlatMap, FlatArea, FlatCircle, FlatRectangle, EmptyArea
from airlift.utils.seeds import generate_seed


class CannotPlaceZoneOnLand(Exception):
    pass


class AirportGenerator:
    """
    Generates Airports in the environment. Controls where pick up and drop of zones are located as well as their
    respective sizes and distances from each other. Controls which map generator is also utilized.
    """

    def __init__(self,
                 max_airports,
                 processing_time,
                 working_capacity,
                 make_drop_off_area,
                 drop_off_border_fraction,
                 drop_off_area_radius,
                 make_pick_up_area,
                 pick_up_border_fraction,
                 pick_up_area_size,
                 min_land_coverage_fraction,
                 mapgen=PlainMapGenerator()):

        self.max_airports = max_airports
        self._mapgen: FlatMapGenerator = mapgen
        self.processing_time = processing_time
        self.working_capacity = working_capacity
        self.make_drop_off_area = make_drop_off_area
        self.drop_off_border_fraction = drop_off_border_fraction
        self.drop_off_area_radius = drop_off_area_radius
        self.make_pick_up_area = make_pick_up_area
        self.pick_up_border_fraction = pick_up_border_fraction
        self.pick_up_area_size = pick_up_area_size
        self.min_land_coverage_fraction = min_land_coverage_fraction

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        self._mapgen.seed(generate_seed(self._np_random))

    def generate(self) -> Tuple[Collection[Airport], FlatMap, Optional[FlatArea], Optional[FlatArea]]:
        raise NotImplementedError

    def _generate_airports_from_coords(self, coords: Collection[FlatCoordinate], dropoff_area, pick_up_area) -> List[
        Airport]:
        """
        Generates airports from given coordinates

        :Parameters:
        ------------
        `coord` : A list of FlatCoordinates where the airports will be placed.
        `dropoff_area` : a designated zone that contains cargo dropoff airports
        `pick_up_area` : a designated zone that contains cargo pickup airports

        :Returns:
        ---------
        `airports` : List[Airport]
        """
        airports = [Airport(coord, i + 1, processing_time=self.processing_time, working_capacity=self.working_capacity)
                    for i, coord in enumerate(coords)]

        for airport in airports:
            if dropoff_area.is_in_area(airport.position):
                airport.in_drop_off_area = True

            if pick_up_area.is_in_area(airport.position):
                airport.in_pick_up_area = True

        return airports

    def _generate_zones(self, map: FlatMap) -> Tuple[Optional[FlatArea], Optional[FlatArea]]:
        """
        Generates the pickup and drop off areas. Keeps trying for a certain period of time until it can find the
        available land mass or area.

        :Parameters:
        ----------
        `map` : An already generated map

        :Returns:
        -------
        `dropoff_area` : An area that designates all the airports within it as dropoff areas
        `pick_up_area` : An area that designates all the airports within it as pickup areas
        """

        # Note that we keep the areas within the map and also try to keep them over land to some degree
        # Dropoff area
        dropoff_area = EmptyArea()
        if self.make_drop_off_area:
            try_count = 0
            while True:
                upper_left_bounds = ((1 - self.drop_off_border_fraction) * map.width,
                                     self.drop_off_area_radius)
                lower_right_bounds = (map.width - self.drop_off_area_radius,
                                      map.height - self.drop_off_area_radius)
                raw_center = self._np_random.random(size=(2))
                center_x = upper_left_bounds[0] + raw_center[0] * (lower_right_bounds[0] - upper_left_bounds[0])
                center_y = upper_left_bounds[1] + raw_center[1] * (lower_right_bounds[1] - upper_left_bounds[1])
                center = FlatCoordinate((center_x, center_y))
                dropoff_area = FlatCircle(center, radius=self.drop_off_area_radius)

                if map.land_coverage(dropoff_area) > self.min_land_coverage_fraction:
                    break

                try_count += 1
                if try_count > 100:
                    raise CannotPlaceZoneOnLand()

        # Pickup area
        pick_up_area = EmptyArea()
        if self.make_pick_up_area:
            try_count = 0
            while True:
                upper_left_bounds = (self.pick_up_area_size[0] / 2,
                                     self.pick_up_area_size[1] / 2)
                lower_right_bounds = (self.pick_up_border_fraction * map.width,
                                      map.height - self.pick_up_area_size[1] / 2)
                raw_center = self._np_random.random(size=(2))
                center_x = upper_left_bounds[0] + raw_center[0] * (lower_right_bounds[0] - upper_left_bounds[0])
                center_y = upper_left_bounds[1] + raw_center[1] * (lower_right_bounds[1] - upper_left_bounds[1])
                center = FlatCoordinate((center_x, center_y))
                pick_up_area = FlatRectangle(center,
                                             width=self.pick_up_area_size[0],
                                             height=self.pick_up_area_size[1])

                if map.land_coverage(pick_up_area) > self.min_land_coverage_fraction:
                    break

                try_count += 1
                if try_count > 100:
                    raise CannotPlaceZoneOnLand()

        return dropoff_area, pick_up_area


class GridAirportGenerator(AirportGenerator):
    """
    Generates Airports in a Grid utilizing a plain map
    """

    def __init__(self,
                 rows=2,
                 columns=2,
                 airport_radius: float = 1,
                 processing_time=0,
                 working_capacity=2 ** 32,
                 make_drop_off_area=True,
                 drop_off_border_fraction=0.2,
                 drop_off_area_radius=0.16,
                 make_pick_up_area=True,
                 pick_up_border_fraction=0.2,
                 pick_up_area_size=(0.25, 0.25),
                 min_land_coverage_fraction=0.7,
                 mapgen=PlainMapGenerator(), ):
        super().__init__(rows * columns,
                         processing_time,
                         working_capacity,
                         make_drop_off_area,
                         drop_off_border_fraction,
                         drop_off_area_radius,
                         make_pick_up_area,
                         pick_up_border_fraction,
                         pick_up_area_size,
                         min_land_coverage_fraction,
                         mapgen, )
        self.airport_radius = airport_radius
        self.rows = rows
        self.columns = columns

    def generate(self) -> Tuple[Collection[Airport], FlatMap, Optional[FlatArea], Optional[FlatArea]]:
        """
        Generates the airport coordinates/locations, the map, as well as the designated drop-off and pickup areas.

        :Returns:
        -------
        `airports` :  A list of airports
        `map` : A generated world map that was created using the map generator
        `dropoff_area` : An area containing the designated cargo dropoff zone
        `pick_up_area` : An area containing the designated cargo pickup zone
        """
        # edge_pad = airport_radius
        edge_pad = 0.02  # Fraction of height/width with which to pad

        # height = 2 * edge_pad + (columns * 2 * airport_radius)
        height = (1 + 2 * edge_pad) * ((self.rows * 2) * self.airport_radius)
        width = (1 + 2 * edge_pad) * ((self.columns * 2) * self.airport_radius)

        good_zones = False
        while not good_zones:
            map = self._mapgen.generate(height, width)
            try:
                dropoff_area, pick_up_area = self._generate_zones(map)
                good_zones = True
            except CannotPlaceZoneOnLand:
                pass

        airport_coords = []
        # Generate x and y coordinates - space out the airports so that there is exactly 1 per unit square area
        for col in range(self.columns):
            for row in range(self.rows):
                world_coord = FlatCoordinate((width * edge_pad + ((1 + 2 * col) * self.airport_radius),
                                              height * edge_pad + ((1 + 2 * row) * self.airport_radius)))

                if map.is_land(world_coord) and world_coord not in airport_coords:
                    airport_coords.append(world_coord)

        airports = self._generate_airports_from_coords(airport_coords, dropoff_area, pick_up_area)

        if len(airports) < self.max_airports:
            warnings.warn(f"Could not set all required airports! Created {len(airports)}/{self.max_airports}")

        return airports, map, dropoff_area, pick_up_area


class RandomAirportGenerator(AirportGenerator):
    """

    Generates Airports uniformly at random

    """

    def __init__(self,
                 max_airports=20,
                 airports_per_unit_area: float = 1,
                 num_drop_off_airports: float = 1,
                 num_pick_up_airports: float = 1,
                 processing_time=0,
                 working_capacity=2 ** 32,
                 aspect_ratio=16 / 9,
                 make_drop_off_area=True,
                 drop_off_border_fraction=0.2,
                 drop_off_area_radius=0.1,
                 make_pick_up_area=True,
                 pick_up_border_fraction=0.2,
                 pick_up_area_size=(0.25, 0.25),
                 min_land_coverage_fraction=0.7,
                 mapgen=PlainMapGenerator()):
        super().__init__(max_airports,
                         processing_time,
                         working_capacity,
                         make_drop_off_area,
                         drop_off_border_fraction,
                         drop_off_area_radius,
                         make_pick_up_area,
                         pick_up_border_fraction,
                         pick_up_area_size,
                         min_land_coverage_fraction,
                         mapgen)

        self.airports_per_unit_area = airports_per_unit_area

        if make_drop_off_area:
            self.num_drop_off_airports = num_drop_off_airports
        else:
            self.num_drop_off_airports = 0

        if make_pick_up_area:
            self.num_pick_up_airports = num_pick_up_airports
        else:
            self.num_pick_up_airports = 0

        self.aspect_ratio = aspect_ratio
        self.mapgen = mapgen

    def _grid_to_coord(self, map: FlatMap, gridxy: Tuple[int, int]) -> FlatCoordinate:
        x = gridxy[0] * map.width / map.grid_size[0]
        y = gridxy[1] * map.height / map.grid_size[1]
        return FlatCoordinate((x, y))

    def generate(self) -> Tuple[Collection[Airport], FlatMap, Optional[FlatArea], Optional[FlatArea]]:
        """
        Generates the airport coordinates/locations, the map, as well as the designated drop-off and pickup areas.

        :return: A tuple containing,
            `airports` :  A list of airports
            `map` : A generated world map that was created using the map generator
            `dropoff_area` : An area containing the designated cargo dropoff zone
            `pick_up_area` : An area containing the designated cargo pickup zone

        """
        assert self.max_airports > self.num_drop_off_airports + self.num_pick_up_airports
        total_area = (
                             self.max_airports - self.num_drop_off_airports - self.num_pick_up_airports) / self.airports_per_unit_area \
                     + 2 * math.pi * self.drop_off_area_radius \
                     + self.pick_up_area_size[0] * self.pick_up_area_size[1]
        width = math.sqrt(total_area / self.aspect_ratio)
        height = width / self.aspect_ratio

        good_zones = False
        while not good_zones:
            map = self._mapgen.generate(height, width)
            try:
                dropoff_area, pick_up_area = self._generate_zones(map)
                good_zones = True
            except CannotPlaceZoneOnLand:
                pass

        # Mark zones - 0 = general area, 1 = drop off zone, 2 = pick up zone, 3 = keep out (airport placed, water, etc..)
        zone_grid = np.zeros((map.grid_size[1], map.grid_size[0]), dtype=np.uint8)
        if dropoff_area is not None:
            dropoff_mask = dropoff_area.get_grid_mask(map.height, map.width, map.grid_size)
            zone_grid[np.where(dropoff_mask == 1.0)] = 1
        if pick_up_area is not None:
            pick_up_mask = pick_up_area.get_grid_mask(map.height, map.width, map.grid_size)
            zone_grid[np.where(pick_up_mask == 1.0)] = 2

        zones = np.array([0, 1, 2])
        num_airports_by_zone = np.array([self.max_airports - self.num_drop_off_airports - self.num_pick_up_airports,
                                         self.num_drop_off_airports,
                                         self.num_pick_up_airports])
        # zone_probs = num_airports_by_zone / np.sum(num_airports_by_zone)

        trycounter = 0
        airport_coords = []
        for zone in zones:
            successcount = 0
            while successcount < num_airports_by_zone[zone]:
                allowed_indexes = np.where(zone_grid == zone)
                num_allowed_points = len(allowed_indexes[0])
                if num_allowed_points == 0:
                    break
                point_index = self._np_random.integers(num_allowed_points)
                x = int(allowed_indexes[1][point_index])
                y = int(allowed_indexes[0][point_index])

                zone_grid[y, x] = 0

                world_coord = FlatCoordinate((width * (x / map.grid_size[0]),
                                              height * (y / map.grid_size[1])))

                if map.is_land(world_coord) and world_coord not in airport_coords:
                    airport_coords.append(world_coord)
                    successcount += 1

                trycounter += 1
                if trycounter > 10000:
                    break
        airports = self._generate_airports_from_coords(airport_coords, dropoff_area, pick_up_area)

        if len(airports) < self.max_airports:
            warnings.warn(f"Could not set all required airports! Created {len(airports)}/{self.max_airports}")
        return airports, map, dropoff_area, pick_up_area


class GridAirportGenerator(AirportGenerator):
    """
    Generates Airports in a Grid utilizing a plain map
    """

    def __init__(self,
                 rows=2,
                 columns=2,
                 airport_radius: float = 1,
                 processing_time=0,
                 working_capacity=2 ** 32,
                 make_drop_off_area=True,
                 drop_off_border_fraction=0.2,
                 drop_off_area_radius=0.16,
                 make_pick_up_area=True,
                 pick_up_border_fraction=0.2,
                 pick_up_area_size=(0.25, 0.25),
                 min_land_coverage_fraction=0.7,
                 mapgen=PlainMapGenerator(), ):
        super().__init__(rows * columns,
                         processing_time,
                         working_capacity,
                         make_drop_off_area,
                         drop_off_border_fraction,
                         drop_off_area_radius,
                         make_pick_up_area,
                         pick_up_border_fraction,
                         pick_up_area_size,
                         min_land_coverage_fraction,
                         mapgen, )
        self.airport_radius = airport_radius
        self.rows = rows
        self.columns = columns

    def generate(self) -> Tuple[Collection[Airport], FlatMap, Optional[FlatArea], Optional[FlatArea]]:
        """
        Generates the airport coordinates/locations, the map, as well as the designated drop-off and pickup areas.

        :return: A tuple containing,
            `airports` :  A list of airports
            `map` : A generated world map that was created using the map generator
            `dropoff_area` : An area containing the designated cargo dropoff zone
            `pick_up_area` : An area containing the designated cargo pickup zone

        """
        # edge_pad = airport_radius
        edge_pad = 0.02  # Fraction of height/width with which to pad

        # height = 2 * edge_pad + (columns * 2 * airport_radius)
        height = (1 + 2 * edge_pad) * ((self.rows * 2) * self.airport_radius)
        width = (1 + 2 * edge_pad) * ((self.columns * 2) * self.airport_radius)

        map = self._mapgen.generate(height, width)
        dropoff_area, pick_up_area = self._generate_zones(map)

        airport_coords = []
        # Generate x and y coordinates - space out the airports so that there is exactly 1 per unit square area
        for col in range(self.columns):
            for row in range(self.rows):
                world_coord = FlatCoordinate((width * edge_pad + ((1 + 2 * col) * self.airport_radius),
                                              height * edge_pad + ((1 + 2 * row) * self.airport_radius)))

                if map.is_land(world_coord) and world_coord not in airport_coords:
                    airport_coords.append(world_coord)

        airports = self._generate_airports_from_coords(airport_coords, dropoff_area, pick_up_area)

        if len(airports) < self.max_airports:
            warnings.warn(f"Could not set all required airports! Created {len(airports)}/{self.max_airports}")

        return airports, map, dropoff_area, pick_up_area


class HardcodedAirportGenerator(AirportGenerator):
    def __init__(self,
                 airports: List[Airport],
                 height: float = 1,
                 width: float = 1,
                 drop_off_area=None,
                 pick_up_area=None,
                 mapgen=PlainMapGenerator()):
        super().__init__(len(airports),
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         mapgen)
        self.height = height
        self.width = width
        self.airports = airports
        self.drop_off_area = drop_off_area
        self.pick_up_area = pick_up_area

    def generate(self) -> Tuple[Collection[Airport], FlatMap, Optional[FlatArea], Optional[FlatArea]]:
        """
        Generates the airport coordinates/locations, the map, as well as the designated drop-off and pickup areas.

        :return: A tuple containing,
            `airports` :  A list of airports
            `map` : A generated world map that was created using the map generator
            `dropoff_area` : An area containing the designated cargo dropoff zone
            `pick_up_area` : An area containing the designated cargo pickup zone

        """

        map = self._mapgen.generate(self.height, self.width)

        for airport in self.airports:
            if self.drop_off_area.is_in_area(airport.position):
                airport.in_drop_off_area = True

            if self.pick_up_area.is_in_area(airport.position):
                airport.in_pick_up_area = True

        return self.airports, map, self.drop_off_area, self.pick_up_area
