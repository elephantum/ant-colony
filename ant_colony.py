from typing import Tuple, Any, Optional
from numpy.random import randint, random
import numpy as np
import math
import skimage.draw
from scipy.signal import convolve2d

import pygame as pg
from pygame import gfxdraw
from pygame.math import Vector2

N_SCENTS = 2
IDX_SCENT_TO_HOME = 0
IDX_SCENT_TO_FOOD = 1

SCENT_DECAY = 0.995


ANT_VELOCITY = 1.
ANT_ROTATION_VEL = 60
CONTACT_DISTANCE = 20
SNIFF_DISTANCE = 30.


STATE_SEARCHING_FOOD = 'search_food'
STATE_GOING_HOME = 'going_home'


FOOD_SIZE = 50
N_FOODS = 20


WORLD_SIZE = Vector2(500, 500)


class Ant:
    def __init__(self, position: Vector2, velocity: Vector2):
        self.position = position
        self.velocity = velocity

        self.state = STATE_SEARCHING_FOOD
        self.state_age = 0
    
    def sniff(self, world: 'World') -> Tuple[Any, Any]:
        center_dir = self.position + (self.velocity * SNIFF_DISTANCE)
        left_dir = self.position + (self.velocity.rotate(-60) * SNIFF_DISTANCE)
        right_dir = self.position + (self.velocity.rotate(60) * SNIFF_DISTANCE)


        left_mask = skimage.draw.polygon(
            [int(self.position.x), int(center_dir.x), int(left_dir.x)],
            [int(self.position.y), int(center_dir.y), int(left_dir.y)],
            (int(world.size.x), int(world.size.y))
        )

        right_mask = skimage.draw.polygon(
            [int(self.position.x), int(center_dir.x), int(right_dir.x)],
            [int(self.position.y), int(center_dir.y), int(right_dir.y)],
            (int(world.size.x), int(world.size.y))
        )

        left = world.scents[left_mask[0], left_mask[1], :].sum(0)
        right = world.scents[right_mask[0], right_mask[1], :].sum(0)

        return left, right

    def tick(self, world: 'World') -> None:
        self.state_age += 1

        is_home = world.is_home(self.position)
        food = world.get_food(self.position)

        if self.state == STATE_SEARCHING_FOOD:
            if is_home:
                self.state_age = 0

            if food is not None:
                food.size -= 1
                self.state=STATE_GOING_HOME

                self.velocity = self.velocity.rotate(180)
                self.state_age = 0

        elif self.state == STATE_GOING_HOME:
            if food is not None:
                self.state_age = 0

            if is_home:
                self.state = STATE_SEARCHING_FOOD
                self.velocity = self.velocity.rotate(180)
                self.state_age = 0


        if self.state == STATE_SEARCHING_FOOD:
            if self.state_age < 1000:
                world.leave_scent(self.position, IDX_SCENT_TO_HOME)
            sniff_for = IDX_SCENT_TO_FOOD
        elif self.state == STATE_GOING_HOME:
            if self.state_age < 1000:
                world.leave_scent(self.position, IDX_SCENT_TO_FOOD)
            sniff_for = IDX_SCENT_TO_HOME


        l, r = self.sniff(world)
        l = l[sniff_for]
        r = r[sniff_for]

        self.velocity.rotate_ip((randint(-ANT_ROTATION_VEL, 0) * (l+1) + randint(0, ANT_ROTATION_VEL) * (r+1)) / (l+r+2) + randint(-ANT_ROTATION_VEL/4, ANT_ROTATION_VEL/4))

        self.position += self.velocity

        self.position.x = min(max(0, self.position.x), world.size.x - 1)
        self.position.y = min(max(0, self.position.y), world.size.y - 1)


class Food:
    def __init__(self, position, size):
        self.position = position
        self.size = size


class World:
    def __init__(self, size: Vector2, n_ants, n_foods) -> None:
        self.size = size

        self.home_pos = size.elementwise() * Vector2(random(), random())

        self.foods = [
            Food(size.elementwise() * Vector2(random(), random()), FOOD_SIZE)
            for i in range(n_foods)
        ]

        self.ants = [
            Ant(Vector2(self.home_pos), Vector2(1,0).rotate(randint(0, 360)))
            for i in range(n_ants)
        ]

        self.scents = np.zeros((int(size.x), int(size.y), N_SCENTS))

    def draw(self) -> Any:
        scents_arr = np.zeros((int(self.size.x), int(self.size.y), 3))
        scents_arr += self.scents[:,:,[IDX_SCENT_TO_HOME]] * np.array([0, 200, 0])
        scents_arr += self.scents[:,:,[IDX_SCENT_TO_FOOD]] * np.array([0, 0, 200])


        # for ant in self.ants:
        #     l, r = ant.sniff(self)
        #     scents_arr[l[0], l[1], :] += np.array([80, 0, 0])
        #     scents_arr[r[0], r[1], :] += np.array([120, 0, 0])


        surface = pg.surfarray.make_surface(scents_arr)

        gfxdraw.filled_circle(surface, int(self.home_pos.x), int(self.home_pos.y), 10, pg.Color(0, 0, 0))
        gfxdraw.filled_circle(surface, int(self.home_pos.x), int(self.home_pos.y), 8, pg.Color(0, 200, 0))

        for food in self.foods:
            gfxdraw.filled_circle(surface, int(food.position.x), int(food.position.y), int(food.size / 10), pg.Color(0, 0, 200))
            # gfxdraw.filled_circle(surface, food.x, food.y, 4, pg.Color(0, 0, 200))

        for ant in self.ants:
            gfxdraw.filled_circle(surface, int(ant.position.x), int(ant.position.y), 5, pg.Color(0, 0, 0))
            gfxdraw.filled_circle(surface, int(ant.position.x), int(ant.position.y), 4, pg.Color(200, 0, 0))

        return surface

    def get_food(self, pos: Vector2) -> Optional[Food]:
        for food in self.foods:
            if pos.distance_to(food.position) < CONTACT_DISTANCE and food.size > 0:
                return food
        return None

    def is_home(self, pos: Vector2) -> bool:
        if pos.distance_to(self.home_pos) < CONTACT_DISTANCE:
            return True
        else:
            return False

    def tick(self) -> None:
        decay_mat = np.array([
            [0.00, 0.0005,  0.00,],
            [0.0005,  0.992,  0.0005   ],
            [0.00, 0.0005,  0.00,]
        ])
        self.scents[:,:, IDX_SCENT_TO_FOOD] = convolve2d(self.scents[:,:, IDX_SCENT_TO_FOOD], decay_mat, mode='same')
        self.scents[:,:, IDX_SCENT_TO_HOME] = convolve2d(self.scents[:,:, IDX_SCENT_TO_HOME], decay_mat, mode='same')

        for ant in self.ants:
            ant.tick(self)

    def leave_scent(self, position, scent) -> None:
        self.scents[int(position.x), int(position.y), scent] += 1


def main() -> None:
    world = World(WORLD_SIZE, 100, N_FOODS)


    pg.init()
    screen = pg.display.set_mode((int(WORLD_SIZE.x), int(WORLD_SIZE.y)))
    clock = pg.time.Clock()

    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        world.tick()

        surface = world.draw()

        screen.fill((30, 30, 30))
        screen.blit(surface, (0, 0))
        pg.display.flip()
        clock.tick(60)


if __name__ == '__main__':
    main()