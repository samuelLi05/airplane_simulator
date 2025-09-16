import numpy.typing as npt
import noise
import numpy as np
import math
from gym.utils import seeding
from airlift.envs.world_map import FlatMap, FlatLandWaterMap, FlatLandOnlyMap


class FlatMapGenerator:
    """
    Base class for generating world maps.
    """
    def __init__(self):
        self._np_random = None

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)

    def generate(self, height, width) -> FlatMap:
        raise NotImplementedError


class PlainMapGenerator(FlatMapGenerator):
    """
    Creates a map containing only land (all green screen at the moment!)
    """

    def __init__(self, grid_size_y=600):
        super().__init__()
        self.grid_size_y = grid_size_y

    def generate(self, height, width) -> FlatMap:
        return FlatLandOnlyMap(height, width, (int(self.grid_size_y * width / height), self.grid_size_y))


# This PerlinMapGenerator was made with the assistance of the code located at
# https://stackoverflow.com/questions/50453437/python-perlin-noise-procedural-map-add-road-river
class PerlinMapGenerator(FlatMapGenerator):
    """
    Creates randomly generated maps based on the perlin noise algorithm
    """

    def __init__(self,
                 grid_size_y=600,
                 color_range=10,
                 color_perlin_scale=.25,
                 scale=350,
                 octaves=6,
                 persistance=0.55,
                 lacunarity=2.0):
        super().__init__()
        self.grid_size_y = grid_size_y

        self.scale = scale
        self.octaves = octaves
        self.persistance = persistance
        self.lacunarity = lacunarity
        self.threshold = -0.014

        self.randomColorRange = color_range
        self.colorPerlinScale = color_perlin_scale

    def generate(self, height, width) -> FlatMap:
        map_size = (int(self.grid_size_y * width / height), self.grid_size_y)
        return FlatLandWaterMap(height, width, self._generate_map("world_map", map_size))

    def _generate_map(self, map_type: str, map_size) -> npt.ArrayLike:
        """
        Generates a world map. Creates a vector that contains RGB values (blue or green) for the world.

        :Parameters:
        ----------
        `map_type` : A string that contains the map type, we could have multiple map types in the future

        :Returns:
        -------
        `land_water_map` : A numpy array containing a colored world based on perlin noise and thresholds
        """

        height_map = np.zeros((map_size[1], map_size[0]))

        random_nr = self._np_random.integers(0, math.ceil(map_size[0]) ** 2)
        y_starting_pos = self._np_random.integers(0, math.ceil(map_size[1]))
        x_starting_pos = self._np_random.integers(0, math.ceil(map_size[0]))

        for i in range(height_map.shape[0]):
            for j in range(height_map.shape[1]):
                new_i = i + y_starting_pos
                new_j = j + x_starting_pos

                height_map[i][j] = noise.pnoise3(new_i / self.scale, new_j / self.scale, random_nr,
                                                      octaves=self.octaves,
                                                      persistence=self.persistance, lacunarity=self.lacunarity,
                                                      repeatx=10000000, repeaty=10000000, base=0)

        if map_type == "world_map":
            gradient = self._create_circular_gradient(height_map)
            land_water_map = self._generate_land_water(gradient)
        else:
            land_water_map = self._generate_land_water(height_map)
        return land_water_map

    # Water = 0.0
    # Land = 1.0
    def _generate_land_water(self, world: npt.ArrayLike) -> npt.ArrayLike:
        land_water_map = np.zeros(world.shape)
        for i in range(world.shape[0]):
            for j in range(world.shape[1]):
                if world[i][j] >= self.threshold + 0.02:
                    land_water_map[i][j] = 1.0
        return land_water_map

    def _create_circular_gradient(self, world: npt.ArrayLike) -> npt.ArrayLike:
        """
        Creates a circular gradient

        :Parameters:
        ----------
        `world` : Numpy Array containing the map values

        :Returns
        -------
        `grad_world` : Numpy Array containing gradients
        """
        center_x, center_y = world.shape[0] // 2, world.shape[1] // 2
        circle_grad = np.zeros_like(world)

        for y in range(world.shape[0]):
            for x in range(world.shape[1]):
                distx = abs(x - center_x)
                disty = abs(y - center_y)
                dist = math.sqrt(distx * distx + disty * disty)
                circle_grad[y][x] = dist

        # get it between -1 and 1
        max_grad = np.max(circle_grad)
        circle_grad = circle_grad / max_grad
        circle_grad -= 2
        circle_grad *= 3
        circle_grad = -circle_grad

        # shrink gradient
        for y in range(world.shape[0]):
            for x in range(world.shape[1]):
                if circle_grad[y][x] > 2:
                    circle_grad[y][x] *= 20

        # get it between 0 and 1
        max_grad = np.max(circle_grad)
        circle_grad = circle_grad / max_grad
        grad_world = self._apply_gradient_noise(world, circle_grad)
        return grad_world

    def _apply_gradient_noise(self, world: npt.ArrayLike, c_grad: npt.ArrayLike) -> npt.ArrayLike:
        """
        Applies gradient noise values to the world.

        :Parameters:
        ----------
        `world` : Numpy Array containing the world map values

        `c_grad` : Numpy Array containing the circular gradient values

        :Returns:
        -------
        `world_noise` : Numpy Array containing gradient noise values
        """
        world_noise = np.zeros_like(world)

        for i in range(world.shape[0]):
            for j in range(world.shape[1]):
                world_noise[i][j] = (world[i][j] * c_grad[i][j])
                if world_noise[i][j] > 0:
                    world_noise[i][j] *= 1120

        # get it between 0 and 1
        max_grad = np.max(world_noise)
        world_noise = world_noise / max_grad
        return world_noise