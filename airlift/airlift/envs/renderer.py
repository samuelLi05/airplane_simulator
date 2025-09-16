import itertools
import os
from datetime import datetime
from typing import List

import PIL.ImageDraw
import numpy as np
import pyglet as pgl
from PIL import Image
from PIL.Image import Resampling

from airlift.envs.agents import PlaneState, EnvAgent
from airlift.envs.route_map import RouteMap
from airlift.envs.world_map import EmptyArea
from airlift.utils.definitions import ROOT_DIR

TRANSPARENT = 0
OPAQUE = 255


class EnvRenderer:
    def render_to_window(self):
        raise NotImplementedError

    def close_window(self):
        raise NotImplementedError

    def render_to_rgb_array(self) -> np.ndarray:
        raise NotImplementedError

    def render_to_file(self, filename):
        raise NotImplementedError

    def render_to_image(self) -> Image:
        raise NotImplementedError

    def render_to_video(self):
        raise NotImplementedError


default_height_in_pixels = 600


class FlatRenderer(EnvRenderer):
    """Handles any type of rendering of the environment. Includes rendering to window, video, images or files"""
    def __init__(self,
                 width_in_pixels=None,
                 height_in_pixels=default_height_in_pixels,
                 show_routes=True,
                 color_planes_by_type=None):

        self.out = None
        self.fourcc = None
        self.frame_size = None
        self.window: pgl.window.Window = None  # If window is not open this should be None
        self.close_requested = False  # user has clicked

        self.lower_left_coord = None
        self.upper_right_coord = None

        self.routemap = None
        self.airplanes = None

        self.routewidth = 1

        sColors = "ff6d00#2962ff#0091ea#00b8d4#d50000#c51162#aa00ff#6200ea#304ffe#00bfa5#00c853" + \
                  "#64dd17#aeea00#ffd600#ffab00#ff6d00#ff3d00#5d4037#455a64"

        self.agent_colors = [self._rgb_s2i(sColor) for sColor in sColors.split("#")]

        self.line_colors = [(255, 255, 255),  # default airport
                            (0, 0, 255),
                            (0, 255, 0),  # pick up area / green
                            (255, 202, 24), # drop off area / yellow
                            (255, 255, 255), # routes 1
                            (0, 0, 255)] # routes 2

        # self.airport_line_length = 30  # How large do we want the airports to appear.
        self.airport_line_length = 0.04
        self.airplane_length = 0.04
        self.cargo_length = 0.04

        self.color_planes_by_type = color_planes_by_type

        self._initialized: bool = False
        self.widthPx: int = width_in_pixels
        self.heightPx: int = height_in_pixels
        self.show_routes: bool = show_routes

    def reset(self,
              routemap: RouteMap,
              airplanes: List[EnvAgent]):
        self.routemap = routemap
        self.airplanes = airplanes
        self.lower_left_coord = (0, 0)
        self.upper_right_coord = (routemap.map.width, routemap.map.height)

        if self.color_planes_by_type is None:
            # If not specified, color by type if there is more than one plane type
            color_planes_by_type = len(routemap.plane_types) > 1
        else:
            color_planes_by_type = self.color_planes_by_type

        if color_planes_by_type:
            self.airplane_color = {a: self.agent_colors[a.plane_type.id] for a in
                                   airplanes}  # Assume type id is sequential 0, 1, 2, ...
        else:
            self.airplane_color = {a: color for a, color in zip(airplanes, itertools.cycle(self.agent_colors))}

    def _update_size_in_pixels(self):
        if self.widthPx is None:
            self.widthPx = int(self.heightPx * self.routemap.map.width / self.routemap.map.height)

    def render_to_window(self):
        if not self.is_window_open():
            self._open_window()

        if self.close_requested:
            if self.is_window_open():
                self.close_window()
            self.close_requested = False
            return

        self._processEvents()

        pil_img = self.render_to_image()

        # convert our PIL image to pyglet:
        bytes_image = pil_img.tobytes()
        pgl_image = pgl.image.ImageData(pil_img.width, pil_img.height,
                                        'RGBA',
                                        bytes_image, pitch=-pil_img.width * 4)

        pgl_image.blit(0, 0)

    def _open_window(self):
        # Can't have None or 0 value for WidthPx or HeightPx
        self._update_size_in_pixels()
        self.window = pgl.window.Window(width=self.widthPx, height=self.heightPx, caption="Airlift", resizable=True,
                                        vsync=False)
        icon = pgl.image.load(ROOT_DIR + "/airlift/envs/png/plane.png")
        self.window.set_icon(icon)

        @self.window.event
        def on_draw():
            self.window.clear()
            self.render_to_window()

        @self.window.event
        def on_resize(width, height):
            self.widthPx = width
            self.heightPx = height
            self._initialize()
            self.render_to_window()
            self.window.dispatch_event("on_draw")

        @self.window.event
        def on_close():
            self.close_requested = True

    def is_window_open(self):
        return self.window is not None

    def close_window(self):
        if self.is_window_open():
            self.window.close()
            self.window = None

    def set_render_options(self,
                           width_in_pixels=None,
                           height_in_pixels=None,
                           show_routes=None):
        # This will probably break video creation if we are in the middle of recording a video

        self._initialized = False

        if width_in_pixels is not None:
            self.widthPx = width_in_pixels
        if height_in_pixels is not None:
            self.heightPx = height_in_pixels
        self._update_size_in_pixels()

        if show_routes is not None:
            self.show_routes = show_routes

        self.close_window()

    def render_to_file(self, filename):
        """
        Renders the current scene into image file

        :parameter filename: filename where to store the rendering output_generator.

        """
        img = self.render_to_image()
        img.save(filename)

    def render_to_rgb_array(self) -> np.ndarray:
        img = self.render_to_image()
        return np.array(img)[:, :, :3]

    def render_to_image(self, scale: float = 1.0):
        if self.widthPx is None:
            self._update_size_in_pixels()

        # Do lazy initialization
        if not self._initialized:
            self._initialize()

        airport_layer = self._create_layer()
        if self.show_routes:
            self._place_routes(airport_layer)
        self._place_airports(airport_layer)

        agent_layer = self._create_layer()
        for agent in self.airplanes:
            if agent.state == PlaneState.MOVING:
                agent_position = self.routemap.map.airplane_position(agent.flight_start_position,
                                                                     agent.destination_airport.position,
                                                                     agent.total_flight_time,
                                                                     agent.elapsed_flight_time)
                direction = self.routemap.map.airplane_direction(agent.flight_start_position,
                                                                 agent.destination_airport.position)
                agent.last_direction = direction
            else:
                agent_position = agent.current_airport.position
            self._set_agent_at(agent_layer, agent, agent_position)
        img = self._alpha_composite_layers([self._background_layer, airport_layer, agent_layer])
        if scale != 1.0:
            img = img.resize((int(self.widthPx * scale), int(self.heightPx * scale)))
        return img

    def render_to_video(self):
        import cv2

        if self.widthPx is None:
            self._update_size_in_pixels()
        if not self._initialized:
            self._initialize()
            # Video Recording
            now = datetime.now()
            dt_string = now.strftime("Date_%m-%d-%Y_Time_%H-%M-%S")
            self.frame_size = (self.widthPx, self.heightPx)
            self.fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            #self.fourcc = cv2.VideoWriter_fourcc('L', 'A', 'G', 'S')
            path = "./" + dt_string + ".wmv"

            # # The following code created a folder for the wmv file as well, which caused issues
            # path = "./recordings/" + dt_string + ".wmv"
            # path_exists = os.path.exists(path)
            # if not path_exists:
            #     os.makedirs(path)

            self.out = cv2.VideoWriter(path, self.fourcc, 12.0,
                                       self.frame_size)
        airport_layer = self._create_layer()
        self._place_airports(airport_layer)

        img = self.render_to_image()

        pil_img_resized = img.resize(self.frame_size, resample=Resampling.NEAREST)
        frame = np.array(pil_img_resized)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.out.write(frame)

    def _initialize(self):
        self._background_layer = self._create_layer(color=(0, 0, 0), opacity=OPAQUE)
        if self.routemap.map.image is not None:
            self._background_layer.paste(
                self.routemap.map.image.resize(self._background_layer.size),
                (0, 0)
            )

        self.pil_airplane = Image.open(ROOT_DIR + "/airlift/envs/png/plane.png").resize(
            self._vector_to_pixel((self.airplane_length, self.airplane_length)))

        # Regular box icon
        self.pil_img_cargo_at_airport = Image.open(ROOT_DIR + "/airlift/envs/png/box.png").resize(
            self._vector_to_pixel((self.cargo_length, self.cargo_length)))

        # Late delivery icon
        self.pil_img_cargo_at_airport_late = Image.open(ROOT_DIR + "/airlift/envs/png/yellow_late.png").resize(
            self._vector_to_pixel((self.cargo_length, self.cargo_length)))

        # Missed delivery icon
        self.pil_img_cargo_at_airport_missed = Image.open(ROOT_DIR + "/airlift/envs/png/red_missed.png").resize(
            self._vector_to_pixel((self.cargo_length, self.cargo_length)))

        # Delivered on time icon
        self.pil_img_cargo_at_airport_delivered = Image.open(ROOT_DIR + "/airlift/envs/png/green_delivered.png").resize(
            self._vector_to_pixel((self.cargo_length, self.cargo_length)))
        self._initialized = True

    def _place_airports(self, layer):
        step_counter = 0
        step_counter += 1
        draw = PIL.ImageDraw.Draw(layer)
        if not isinstance(self.routemap.drop_off_area, EmptyArea):
            length = self._coord_to_pixel(self.routemap.drop_off_area.size)
            xy = self._coord_to_pixel(self.routemap.drop_off_area.center)
            drop_off_area_length = length[0]
            top_right = (xy[0] + 1 / 2 * drop_off_area_length, xy[1] + 1 / 2 * drop_off_area_length)
            bottom_left = (xy[0] - 1 / 2 * drop_off_area_length, xy[1] - 1 / 2 * drop_off_area_length)
            shape = [bottom_left, top_right]
            draw.arc(shape, start=0, end=360, fill=self.line_colors[3], width=3)

        if not isinstance(self.routemap.pick_up_area, EmptyArea):
            length = self._coord_to_pixel(self.routemap.pick_up_area.size)
            xy = self._coord_to_pixel(self.routemap.pick_up_area.center)
            self._draw_rectangle(draw, length, xy, color=self.line_colors[2], width=3)

        for airport in self.routemap.airports:
            xy = self._coord_to_pixel(airport.position)
            if not airport.in_drop_off_area and not airport.in_pick_up_area:
                airport_color = self.line_colors[0]
            elif airport.in_pick_up_area:
                airport_color = self.line_colors[2]
            else:
                airport_color = self.line_colors[3]

            pixsize = self._vector_to_pixel((self.airport_line_length, self.airport_line_length))

            # Paste a piece of the base map so that route lines don't go through airports
            region = (int(xy[0]-pixsize[0]/2),
                      int(xy[1]-pixsize[1]/2),
                      int(xy[0]+pixsize[0]/2),
                      int(xy[1]+pixsize[1]/2))
            submap = self._background_layer.crop(region)
            layer.paste(submap, region)

            # Draw airport box
            self._draw_rectangle(draw, pixsize, xy, color=airport_color, width=1)

            if airport.cargo:
                self._draw_image(layer, self.pil_img_cargo_at_airport, airport.position)

            for cargo in airport.cargo:
                # Cargo hasn't been delivered yet and missed the soft deadline (green)
                if cargo.delivery_time is not None and not cargo.missed_softdeadline:
                    self._draw_image(layer, self.pil_img_cargo_at_airport_delivered, airport.position)

                # Cargo missed the soft deadline but not the hard deadline (yellow)
                elif cargo.missed_softdeadline and not cargo.missed_hardeadline:
                    self._draw_image(layer, self.pil_img_cargo_at_airport_late, airport.position)

                # Cargo missed the hard deadline (red)
                elif cargo.missed_hardeadline:
                    self._draw_image(layer, self.pil_img_cargo_at_airport_missed, airport.position)

    def _draw_rectangle(self, draw, length, xy, color=(0, 0, 0), width=1):
        top_right = (xy[0] + 1 / 2 * length[0],
                     xy[1] + 1 / 2 * length[1])
        bottom_left = (xy[0] - 1 / 2 * length[0],
                       xy[1] - 1 / 2 * length[1])
        draw.rectangle(top_right + bottom_left, outline=color, width=width)

    def _place_routes(self, layer):

        draw = PIL.ImageDraw.Draw(layer)
        for i, plane_type in enumerate(self.routemap.plane_types):
            for u, v, d in self.routemap.graph[plane_type].edges(data=True):
                if not d["mal"].in_malfunction:
                    coord1 = self.routemap.airports_by_id[u].position
                    coord2 = self.routemap.airports_by_id[v].position
                    draw.line(xy=self._coord_to_pixel(coord1) + self._coord_to_pixel(coord2), fill=self.line_colors[4+i],
                              width=self.routewidth)

    def _rgb_s2i(self, sRGB):
        """ convert a hex RGB string like 0091ea to 3-tuple of ints """
        return tuple(int(sRGB[iRGB * 2:iRGB * 2 + 2], 16) for iRGB in [0, 1, 2])

    def _draw_image(self, layer, pil_img, xyCenter, xySize=None):
        xyPixCenter = self._coord_to_pixel(xyCenter)

        if xySize is None:
            xyPixSize = pil_img.size
        else:
            xyPixSize = self._vector_to_pixel(2 * xySize)
            pil_img = pil_img.resize(xyPixSize)

        if pil_img.mode == "RGBA":
            pil_mask = pil_img
        else:
            pil_mask = None

        layer.paste(pil_img, (xyPixCenter[0] - xyPixSize[0] // 2, xyPixCenter[1] - xyPixSize[1] // 2), pil_mask)

    @staticmethod
    def _alpha_composite_layers(layers):
        img = layers[0]
        for img2 in layers[1:]:
            img = Image.alpha_composite(img, img2)
        return img

    def _create_layer(self, color=(255, 255, 255), opacity=TRANSPARENT):
        img = Image.new("RGBA", (self.widthPx, self.heightPx), color + (opacity,))
        return img

    def _set_agent_at(self, layer, agent, coord):
        if agent.last_direction is None:
            rotation = 0
        else:
            # Note arctan2 gives angle between [1,0] and the given vector.
            angle = -np.arctan2(agent.last_direction[0], -agent.last_direction[1])
            rotation = np.rad2deg(angle)

        rotated_pil = self.pil_airplane.rotate(rotation)
        rotated_pil = self._recolor_image(rotated_pil, [0, 0, 0], self.airplane_color[agent], False)
        self._draw_image(layer, rotated_pil, coord)

    def _recolor_image(self, pil, a3BaseColor, ltColors, invert=False):

        data = np.array(pil)  # "data" is a height x width x 4 numpy array
        red, green, blue, alpha = data.T  # Temporarily unpack the bands for readability

        # Replace white with red... (leaves alpha values alone...)
        black_areas = (red == 0) & (blue == 0) & (green == 0)
        data[..., :-1][black_areas.T] = ltColors  # Transpose back needed

        pil2 = Image.fromarray(data)

        return pil2

    def _coord_to_pixel(self, coord):
        world_width = self.upper_right_coord[0] - self.lower_left_coord[0]
        world_height = self.upper_right_coord[1] - self.lower_left_coord[1]

        # Normalize the coordinates between 0 and 1
        normalized_coord = (
            (coord[0] - self.lower_left_coord[0]) / world_width,
            (coord[1] - self.lower_left_coord[1]) / world_height
        )

        return (
            round(normalized_coord[0] * self.widthPx),
            round(normalized_coord[1] * self.heightPx)
        )

    # Scale a vector
    def _vector_to_pixel(self, vector):
        length = self._coord_to_pixel(vector)
        origin = self._coord_to_pixel((0, 0))
        return (length[0] - origin[0],
                length[1] - origin[1])

    def _processEvents(self):
        """ This is the replacement for a custom event loop for Pyglet.
            The lines below are typical of Pyglet examples.
            Manually resizing the window is still very clunky.
        """
        pgl.clock.tick()
        if self.is_window_open():
            self.window.switch_to()
            self.window.dispatch_events()
            self.window.flip()
        # print(" events done")
