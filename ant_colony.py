from numpy.random import randint
import pygame as pg
from pygame import gfxdraw
import numpy as np
import math
import skimage.draw


N_SCENTS = 2
IDX_SCENT_TO_HOME = 0
IDX_SCENT_TO_FOOD = 1

SCENT_DECAY = 0.995


ANT_VELOCITY = 1.
ANT_ROTATION_VEL = 60
FOOD_DETECTION_RAD = 20
DETECTION_DISTANCE = 20


STATE_SEARCHING_FOOD = 'search_food'
STATE_GOING_HOME = 'going_home'


FOOD_SIZE = 50
N_FOODS = 20


WORLD_W = 500
WORLD_H = 500


def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2) ** 2)


class Ant:
    def __init__(self, x, y, dir):
        self.x = x
        self.y = y
        self.dir = dir
        self.state = STATE_SEARCHING_FOOD
        self.state_age = 0
    
    def sniff(self, world):
        left_mask = skimage.draw.polygon(
            [int(self.x), int(self.x + math.cos(self.dir / 360. * 2. * math.pi)*DETECTION_DISTANCE), int(self.x + math.cos((self.dir - 90) / 360. * 2. * math.pi)*DETECTION_DISTANCE)],
            [int(self.y), int(self.y + math.sin(self.dir / 360. * 2. * math.pi)*DETECTION_DISTANCE), int(self.y + math.sin((self.dir - 90) / 360. * 2. * math.pi)*DETECTION_DISTANCE)],
            (world.w, world.h)
        )

        right_mask = skimage.draw.polygon(
            [int(self.x), int(self.x + math.cos(self.dir / 360. * 2. * math.pi)*DETECTION_DISTANCE), int(self.x + math.cos((self.dir + 90) / 360. * 2. * math.pi)*DETECTION_DISTANCE)],
            [int(self.y), int(self.y + math.sin(self.dir / 360. * 2. * math.pi)*DETECTION_DISTANCE), int(self.y + math.sin((self.dir + 90) / 360. * 2. * math.pi)*DETECTION_DISTANCE)],
            (world.w, world.h)
        )

        left = world.scents[left_mask[0], left_mask[1], :].sum(0)
        right = world.scents[right_mask[0], right_mask[1], :].sum(0)

        return left, right

    def tick(self, world: 'World') -> None:
        self.state_age += 1

        is_home = world.is_home(self.x, self.y)
        food = world.get_food(self.x, self.y)

        if self.state == STATE_SEARCHING_FOOD:
            if is_home:
                self.state_age = 0

            if food is not None:
                food.size -= 1
                self.state=STATE_GOING_HOME
                self.dir += 180
                self.state_age = 0

        elif self.state == STATE_GOING_HOME:
            if food is not None:
                self.state_age = 0

            if is_home:
                self.state = STATE_SEARCHING_FOOD
                self.dir += 180
                self.state_age = 0


        if self.state == STATE_SEARCHING_FOOD:
            if self.state_age < 1000:
                world.scents[int(self.x), int(self.y), IDX_SCENT_TO_HOME] += 1
            sniff_for = IDX_SCENT_TO_FOOD
        elif self.state == STATE_GOING_HOME:
            if self.state_age < 1000:
                world.scents[int(self.x), int(self.y), IDX_SCENT_TO_FOOD] += 1
            sniff_for = IDX_SCENT_TO_HOME


        l, r = self.sniff(world)
        l = l[sniff_for]
        r = r[sniff_for]

        self.dir += (randint(-ANT_ROTATION_VEL, 0) * (l+1) + randint(0, ANT_ROTATION_VEL) * (r+1)) / (l+r+2)
        self.dir %= 360


        self.x += ANT_VELOCITY * math.cos(self.dir / 360. * 2. * math.pi)
        self.y += ANT_VELOCITY * math.sin(self.dir / 360. * 2. * math.pi)

        self.x = min(max(0, self.x), world.w-1)
        self.y = min(max(0, self.y), world.h-1)


class Food:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size


class World:
    def __init__(self, w, h, n_ants, n_foods) -> None:
        self.w = w
        self.h = h

        self.home_xy = (randint(0, w), randint(0, h))

        self.foods = [
            Food(randint(0, w), randint(0, h), FOOD_SIZE)
            for i in range(n_foods)
        ]

        self.ants = [
            Ant(self.home_xy[0], self.home_xy[1], randint(0, 360))
            for i in range(n_ants)
        ]

        self.scents = np.zeros((w, h, N_SCENTS))

    def draw(self):
        scents_arr = np.zeros((self.w, self.h, 3))
        scents_arr += self.scents[:,:,[IDX_SCENT_TO_HOME]] * np.array([0, 200, 0])
        scents_arr += self.scents[:,:,[IDX_SCENT_TO_FOOD]] * np.array([0, 0, 200])


        # for ant in self.ants:
        #     l, r = ant.sniff(self)
        #     scents_arr[l[0], l[1], :] += np.array([80, 0, 0])
        #     scents_arr[r[0], r[1], :] += np.array([120, 0, 0])


        surface = pg.surfarray.make_surface(scents_arr)

        gfxdraw.filled_circle(surface, self.home_xy[0], self.home_xy[1], 10, pg.Color(0, 0, 0))
        gfxdraw.filled_circle(surface, self.home_xy[0], self.home_xy[1], 8, pg.Color(0, 200, 0))

        for food in self.foods:
            gfxdraw.filled_circle(surface, food.x, food.y, int(food.size / 10), pg.Color(0, 0, 200))
            # gfxdraw.filled_circle(surface, food.x, food.y, 4, pg.Color(0, 0, 200))

        for ant in self.ants:
            gfxdraw.filled_circle(surface, int(ant.x), int(ant.y), 5, pg.Color(0, 0, 0))
            gfxdraw.filled_circle(surface, int(ant.x), int(ant.y), 4, pg.Color(200, 0, 0))

        return surface

    def get_food(self, x, y):
        for food in self.foods:
            if distance(x, y, food.x, food.y) < FOOD_DETECTION_RAD and food.size > 0:
                return food

    def is_home(self, x, y):
        if distance(x, y, self.home_xy[0], self.home_xy[1]) < FOOD_DETECTION_RAD:
            return True
        else:
            return False

    def tick(self):
        self.scents *= SCENT_DECAY

        for ant in self.ants:
            ant.tick(self)


def main():
    world = World(WORLD_W, WORLD_H, 100, N_FOODS)


    pg.init()
    screen = pg.display.set_mode((WORLD_W, WORLD_H))
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