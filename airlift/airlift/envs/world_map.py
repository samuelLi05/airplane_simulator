import math
from functools import lru_cache
from typing import Tuple, Optional, NamedTuple

import PIL
import numpy as np
import numpy.typing as npt
from PIL.Image import Image


class Coordinate:
    pass


class FlatCoordinate(Coordinate, Tuple[float, float]):
    def __add__(self, other):
        return self.__class__((self[0] + other[0],
                               self[1] + other[1]))

    def __sub__(self, other):
        return self.__class__((self[0] - other[0],
                               self[1] - other[1]))

    def __mul__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        return self.__class__((self[0] * other,
                               self[1] * other))

    def __truediv__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        return self.__mul__(1 / other)

    __rmul__ = __mul__


class FlatArea:
    def __init__(self):
        pass

    @property
    def size(self) -> Tuple[float, float]:
        raise NotImplementedError

    def is_in_area(self, coord: FlatCoordinate):
        raise NotImplementedError

    def get_area(self):
        raise NotImplementedError

    def get_grid_mask(self, height, width, grid_size):
        raise NotImplementedError

    def _get_grid_bound_box(self, height, width, grid_size, box_width, box_height):
        row_start = max(int((self.center[1] - box_height / 2) * grid_size[1] / height), 0)
        row_end = min(int((self.center[1] + box_height / 2) * grid_size[1] / height), grid_size[1])
        col_start = max(int((self.center[0] - box_width / 2) * grid_size[0] / width), 0)
        col_end = min(int((self.center[0] + box_width / 2) * grid_size[0] / width), grid_size[0])

        return row_start, row_end, col_start, col_end


class FlatRectangle(FlatArea):
    def __init__(self, center, height, width):
        super().__init__()
        self.center: FlatCoordinate = center
        self.height: float = height
        self.width: float = width

    @property
    def size(self) -> Tuple[float, float]:
        return (self.width, self.height)

    def is_in_area(self, coord: FlatCoordinate):
        return (self.center[0] - self.width / 2) <= coord[0] <= (self.center[0] + self.width / 2) and \
               (self.center[1] - self.height / 2) <= coord[1] <= (self.center[1] + self.height / 2)

    def get_area(self):
        return self.height * self.width

    @lru_cache
    def get_grid_mask(self, height, width, grid_size):
        row_start, row_end, col_start, col_end = self._get_grid_bound_box(height, width, grid_size, self.width,
                                                                          self.height)
        mask = np.zeros((grid_size[1], grid_size[0]))
        mask[row_start: row_end, col_start: col_end] = 1.0

        return mask


class FlatCircle(FlatArea):
    def __init__(self, center, radius):
        super().__init__()
        self.center: FlatCoordinate = center
        self.radius: float = radius

    @property
    def size(self) -> Tuple[float, float]:
        return (2 * self.radius, 2 * self.radius)

    def is_in_area(self, coord: FlatCoordinate):
        return FlatMap.distance(self.center, coord) <= self.radius

    def get_area(self):
        return math.pi * self.radius ** 2

    @lru_cache
    def get_grid_mask(self, height, width, grid_size):
        row_start, row_end, col_start, col_end = self._get_grid_bound_box(height, width, grid_size, 2 * self.radius,
                                                                          2 * self.radius)

        mask = np.zeros((grid_size[1], grid_size[0]))
        for i in range(row_start, row_end):
            for j in range(col_start, col_end):
                x = j * width / grid_size[0]
                y = i * height / grid_size[1]
                if self.is_in_area((x, y)):
                    mask[i][j] = 1.0

        return mask


class EmptyArea(FlatArea):
    def __init__(self):
        super().__init__()

    @property
    def size(self) -> Tuple[float, float]:
        return (0, 0)

    def is_in_area(self, coord: FlatCoordinate):
        return False

    def get_area(self):
        return 0

    @lru_cache
    def get_grid_mask(self, height, width, grid_size):
        return np.zeros((grid_size[1], grid_size[0]))


class Map:
    def __init__(self):
        pass

    @classmethod
    def distance(cls, obj1, obj2):
        raise NotImplementedError

    @classmethod
    def airplane_position(cls, start_position: Tuple[float, float], destination_position: Tuple[float, float],
                          total_flight_time: int, elapsed_flight_time: int) -> Tuple[float, float]:
        raise NotImplementedError

    @classmethod
    def airplane_direction(cls, start_position: Tuple[float, float],
                           destination_position: Tuple[float, float]) -> FlatCoordinate:
        raise NotImplementedError


class FlatMap(Map):
    def __init__(self, height, width, image=None, grid_size=None):
        super().__init__()

        self.height = height
        self.width = width
        self.image: Optional[Image] = image
        if grid_size is None:
            self.grid_size = image.size
        else:
            self.grid_size = grid_size

    def _coord_to_pixel(self, coord: FlatCoordinate) -> Tuple[int, int]:
        xpixel = round(self.grid_size[0] * coord[0] / self.width)
        ypixel = round(self.grid_size[1] * coord[1] / self.height)
        return (xpixel, ypixel)

    def _pixel_to_coord(self, pixel_location: Tuple[int, int]) -> FlatCoordinate:
        x = pixel_location[0] * self.width / self.grid_size[0]
        y = pixel_location[1] * self.height / self.grid_size[1]
        return FlatCoordinate((x, y))

    def is_land(self, coord: FlatCoordinate):
        raise NotImplementedError

    def land_coverage(self, area: FlatArea) -> float:
        raise NotImplementedError

    @classmethod
    def distance(cls, obj1, obj2):
        return math.sqrt((obj1[0] - obj2[0]) ** 2 + (obj1[1] - obj2[1]) ** 2)
        # return np.linalg.norm(np.subtract(obj1, obj2))

    @classmethod
    def airplane_position(cls, start_position: Tuple[float, float], destination_position: Tuple[float, float],
                          total_flight_time: int, elapsed_flight_time: int) -> Tuple[float, float]:
        """

        Parameters
        ----------
        start_position - A tuple containing the start position as (x,y) coordinates
        destination_position - A tuple containing the destination as (x,y) coordinates
        total_flight_time - int that represents total flight time between the source and destination
        elapsed_flight_time - int that represented elapsed flight time between source and destination

        Returns
        -------
        Tuple that contains (x,y) coordinates of current position
        """
        assert total_flight_time != 0
        return start_position + elapsed_flight_time * (destination_position - start_position) / total_flight_time

    @classmethod
    def airplane_direction(cls, start_position: Tuple[float, float],
                           destination_position: Tuple[float, float]) -> FlatCoordinate:
        """

        Parameters
        ----------
        start_position - A tuple containing the starting position coordinates
        destination_position - A tuple containing the destination position coordinates

        Returns
        -------
        Returns the distance of two points on a cartesian plane
        """
        unnorm_direction = destination_position - start_position
        norm_direction = unnorm_direction / np.linalg.norm(unnorm_direction)
        return FlatCoordinate((norm_direction[0], norm_direction[1]))


class FlatLandOnlyMap(FlatMap):
    def __init__(self, height, width, grid_size):
        super().__init__(height,
                         width,
                         grid_size=grid_size)

    def is_land(self, coord: FlatCoordinate):
        return True

    def land_coverage(self, area: FlatArea) -> float:
        return 1.0


class FlatLandWaterMap(FlatMap):
    WATER_COLOR = [0, 0, 139]  # Dark Blue
    LAND_COLOR = [34, 139, 34]  # Dark Green

    def __init__(self, height, width, land_water_map):
        super().__init__(height,
                         width,
                         image=PIL.Image.fromarray(self._add_color(land_water_map).astype('uint8')))
        self.land_water_map = land_water_map

    def is_land(self, coord: FlatCoordinate):
        return self.image.getpixel(self._coord_to_pixel(coord)) == tuple(self.LAND_COLOR)

    def land_coverage(self, area: FlatArea) -> float:
        # allowed_indexes = np.where(zone_grid == new_zone)
        mask = area.get_grid_mask(self.height, self.width, self.grid_size)
        #
        # area_total = 0
        # land_total = 0
        # for xpixel in range(self.image.size[0]):
        #     for ypixel in range(self.image.size[1]):
        #         if area.is_in_area(self._pixel_to_coord((xpixel, ypixel))):
        #             area_total += 1
        #             if self.image.getpixel((xpixel, ypixel)) == tuple(self.LAND_COLOR):
        #                 land_total += 1
        return sum(sum((mask == 1.0) & (self.land_water_map == 1.0))) / sum(sum(mask == 1.0))

    def _add_color(self, land_water_map: npt.ArrayLike) -> npt.ArrayLike:
        """

        Parameters
        ----------
        world - Numpy Array containing the world map values generated by perlin noise

        Returns
        -------
        color_world - Numpy Array containing RGB color values for the world
        """
        color_world = np.zeros(land_water_map.shape + (3,))
        for i in range(land_water_map.shape[0]):
            for j in range(land_water_map.shape[1]):
                if land_water_map[i][j]:
                    color_world[i][j] = self.LAND_COLOR
                else:
                    color_world[i][j] = self.WATER_COLOR
        return color_world


class FlatRealLandWaterMap(FlatMap):
    WATER_COLOR = [0, 0, 139]  # Dark Blue
    LAND_COLOR = [34, 139, 34]  # Dark Green

    def __init__(self, height, width, land_water_map):
        self.land_water_map = land_water_map

        super().__init__(height,
                         width,
                         image=PIL.Image.fromarray(land_water_map))

    def is_land(self, coord: FlatCoordinate):
        return self.image.getpixel(self._coord_to_pixel(coord)) == tuple(self.LAND_COLOR)

    def land_coverage(self, area: FlatArea) -> float:
        return 1.0

