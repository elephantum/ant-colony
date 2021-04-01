from typing import Tuple, Any, Optional, List
from numpy.random import randint, random
import numpy as np
import skimage.draw
import skimage.transform
from scipy.signal import convolve2d

import esper

import pygame as pg
from pygame import gfxdraw
from pygame.math import Vector2

N_SCENTS = 2
SCENT_TO_HOME = 0
SCENT_TO_FOOD = 1

SCENT_DECAY = 0.995


ANT_VELOCITY = 1.
ANT_ROTATION_VEL = 60
CONTACT_DISTANCE = 5
SNIFF_DISTANCE = 30.


STATE_SEARCHING_FOOD = 'search_food'
STATE_GOING_HOME = 'going_home'
STATE_AGE = 1000


FOOD_SIZE = 50
N_FOODS = 20

N_ANTS = 100

WORLD_SIZE = Vector2(500, 500)


class Movable:
    def __init__(self, position: Vector2, velocity: Vector2):
        self.position = position
        self.velocity = velocity


class Renderable:
    def __init__(self, size, color):
        self.size = size
        self.color = color


class Stinky:
    def __init__(self, size: int, scent: int = None):
        self.size = size
        self.scent = scent


class Home:
    pass


class Food:
    def __init__(self, size):
        self.size = size


class Ant:
    def __init__(self):
        self.state = STATE_SEARCHING_FOOD
        self.state_age = 0


class MovementProcessor(esper.Processor):
    def __init__(self, world_size: Vector2):
        super().__init__()
        self.world_size = world_size
    
    def process(self):
        for ent, mov in self.world.get_component(Movable):
            mov.position = mov.position + mov.velocity

            mov.position.x = min(max(0, mov.position.x), self.world_size.x - 1)
            mov.position.y = min(max(0, mov.position.y), self.world_size.y - 1)


class FoodProcessor(esper.Processor):
    def __init__(self, world_size) -> None:
        super().__init__()

        self.world_size = world_size

        self.recache = True
        self.ents = np.zeros((int(world_size.x), int(world_size.y)), dtype=int)
        self.ents[:,:] = -1

    def cache(self):
        if self.recache:
            self.ents[:,:] = -1

            for ent, (food, mov) in self.world.get_components(Food, Movable):
                cooords = skimage.draw.circle(int(mov.position.x), int(mov.position.y), CONTACT_DISTANCE, (int(self.world_size.x), int(self.world_size.y)))

                self.ents[cooords[0], cooords[1]] = ent
            
            self.recache = False

    def get_food(self, pos: Vector2) -> Optional[Food]:
        self.cache()

        ent = self.ents[int(pos.x), int(pos.y)]

        if ent != -1:
            return self.world.component_for_entity(ent, Food)
        
        return None

    def process(self):
        self.cache()

        for ent, (food, ren, stinky) in self.world.get_components(Food, Renderable, Stinky):
            if food.size > 0:
                stinky.size = int(food.size / 10)
                ren.size = int(food.size / 10)
            else:
                self.world.delete_entity(ent)
                self.recache = True


class AntProcessor(esper.Processor):
    def __init__(self, food_processor, scent_processor) -> None:
        super().__init__()
        self.food_processor = food_processor
        self.scent_processor = scent_processor

    def is_home(self, pos: Vector2) -> bool:
        for ent, (home, mov) in self.world.get_components(Home, Movable):
            if pos.distance_to(mov.position) < CONTACT_DISTANCE:
                return True
        return False

    def sniff(self, mov: Movable, scent: int) -> Tuple[Any, Any]:
        center_dir = mov.position + (mov.velocity * SNIFF_DISTANCE)
        left_dir = mov.position + (mov.velocity.rotate(-60) * SNIFF_DISTANCE)
        right_dir = mov.position + (mov.velocity.rotate(60) * SNIFF_DISTANCE)


        left = self.scent_processor.sniff(
            [mov.position, center_dir, left_dir],
            scent
        )

        right = self.scent_processor.sniff(
            [mov.position, center_dir, right_dir],
            scent
        )

        return left, right

    def process(self):
        for ent, (ant, mov, stinky) in self.world.get_components(Ant, Movable, Stinky):
            ant.state_age += 1

            if ant.state_age < STATE_AGE:
                if ant.state == STATE_GOING_HOME:
                    stinky.scent = SCENT_TO_FOOD
                elif ant.state == STATE_SEARCHING_FOOD:
                    stinky.scent = SCENT_TO_HOME
            else:
                stinky.scent = None

            food = self.food_processor.get_food(mov.position)
            is_home = self.is_home(mov.position)

            if ant.state == STATE_SEARCHING_FOOD:
                if is_home:
                    ant.state_age = 0

                if food is not None:
                    food.size -= 1
                    ant.state=STATE_GOING_HOME

                    mov.velocity = mov.velocity.rotate(180)
                    ant.state_age = 0

            elif ant.state == STATE_GOING_HOME:
                if food is not None:
                    ant.state_age = 0

                if is_home:
                    ant.state = STATE_SEARCHING_FOOD
                    mov.velocity = mov.velocity.rotate(180)
                    ant.state_age = 0


            if ant.state == STATE_SEARCHING_FOOD:
                sniff_for = SCENT_TO_FOOD
            elif ant.state == STATE_GOING_HOME:
                sniff_for = SCENT_TO_HOME


            l, r = self.sniff(mov, sniff_for)

            mov.velocity.rotate_ip((randint(-ANT_ROTATION_VEL, 0) * (l+1) + randint(0, ANT_ROTATION_VEL) * (r+1)) / (l+r+2))
            mov.velocity.rotate_ip(randint(-ANT_ROTATION_VEL/4, ANT_ROTATION_VEL/4))


class ScentProcessor(esper.Processor):
    def __init__(self, world_size, k=1) -> None:
        super().__init__()

        self.k = k

        self.world_size = world_size
        self.scents = np.zeros((int(world_size.x / self.k), int(world_size.y / self.k), N_SCENTS))

        d = 0.001 / (k*k)
        self.diff_kernel = np.array([
            [0.00, d,  0.00,],
            [d,  0.992,  d   ],
            [0.00, d,  0.00,]
        ])

    def leave_scent(self, position, size, scent) -> None:
        cooords = skimage.draw.circle(int(position.x / self.k), int(position.y / self.k), size / self.k, (int(self.world_size.x / self.k), int(self.world_size.y / self.k)))

        self.scents[cooords[0], cooords[1], scent] = 1
    
    def sniff(self, polygon: List[Vector2], scent: int) -> float:
        mask = skimage.draw.polygon(
            [int(i.x / self.k) for i in polygon],
            [int(i.y / self.k) for i in polygon],
            (int(self.world_size.x / self.k), int(self.world_size.y / self.k))
        )

        scent_val = self.scents[mask[0], mask[1], scent].sum()

        return scent_val

    def scents_for_vis(self):
        return skimage.transform.resize(self.scents, (int(self.world_size.x), int(self.world_size.y), N_SCENTS))

    def process(self):
        self.scents[:,:, SCENT_TO_FOOD] = convolve2d(self.scents[:,:, SCENT_TO_FOOD], self.diff_kernel, mode='same')
        self.scents[:,:, SCENT_TO_HOME] = convolve2d(self.scents[:,:, SCENT_TO_HOME], self.diff_kernel, mode='same')

        for ent, (mov, stinky) in self.world.get_components(Movable, Stinky):
            if stinky.scent is not None:
                self.leave_scent(mov.position, stinky.size, stinky.scent)


class RenderProcessor(esper.Processor):
    def __init__(self, world_size, screen, scent_processor, clear_color=(0, 0, 0)):
        super().__init__()
        self.world_size = world_size
        self.screen = screen
        self.clear_color = clear_color

        self.scent_processor = scent_processor

    def process(self):
        scents = self.scent_processor.scents_for_vis()

        scents_arr = np.zeros((int(self.world_size.x), int(self.world_size.y), 3))
        scents_arr += scents[:,:,[SCENT_TO_HOME]] * np.array([0, 200, 0])
        scents_arr += scents[:,:,[SCENT_TO_FOOD]] * np.array([0, 0, 200])

        surface = pg.surfarray.make_surface(scents_arr)

        for ent, (mov, ren) in self.world.get_components(Movable, Renderable):
            gfxdraw.filled_circle(surface, int(mov.position.x), int(mov.position.y), ren.size + 1, pg.Color(0, 0, 0))
            gfxdraw.filled_circle(surface, int(mov.position.x), int(mov.position.y), ren.size, ren.color)

        self.screen.fill((30, 30, 30))
        self.screen.blit(surface, (0, 0))

        pg.display.flip()


def main() -> None:
    pg.init()
    screen = pg.display.set_mode((int(WORLD_SIZE.x), int(WORLD_SIZE.y)))
    clock = pg.time.Clock()

    running = True

    world = esper.World()

    home = world.create_entity()
    home_pos = WORLD_SIZE.elementwise() * Vector2(random(), random())
    world.add_component(home, Home())
    world.add_component(home, Stinky(10, SCENT_TO_HOME))
    world.add_component(home, Movable(home_pos, Vector2()))
    world.add_component(home, Renderable(10, pg.Color(0, 200, 0)))

    for _ in range(N_FOODS):
        food = world.create_entity()
        world.add_component(food, Food(FOOD_SIZE))
        world.add_component(food, Stinky(1, SCENT_TO_FOOD))
        world.add_component(food, Movable(WORLD_SIZE.elementwise() * Vector2(random(), random()), Vector2()))
        world.add_component(food, Renderable(1, pg.Color(0, 0, 200)))

    for _ in range(N_ANTS):
        ant = world.create_entity()
        world.add_component(ant, Ant())
        world.add_component(ant, Stinky(1, None))
        world.add_component(ant, Movable(home_pos, Vector2(1, 0).rotate(randint(0, 360))))
        world.add_component(ant, Renderable(3, pg.Color(200, 0, 0)))

    scent_processor = ScentProcessor(WORLD_SIZE, 2)
    food_processor = FoodProcessor(WORLD_SIZE)
    ant_processor = AntProcessor(food_processor, scent_processor)
    movement_processor = MovementProcessor(WORLD_SIZE)
    render_processor = RenderProcessor(WORLD_SIZE, screen, scent_processor)

    world.add_processor(ant_processor)
    world.add_processor(scent_processor)
    world.add_processor(movement_processor)
    world.add_processor(render_processor)
    world.add_processor(food_processor)

    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        world.process()

        clock.tick(60)


if __name__ == '__main__':
    main()