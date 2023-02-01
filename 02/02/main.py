from __future__ import annotations
from telnetlib import GA
import numpy as np
import os
import matplotlib
import matplotlib.backend_bases

if os.name == "darwin":
    matplotlib.use("MacOSX")  # for mac
else:
    matplotlib.use("TkAgg")  # for unix/windows

import matplotlib.pyplot as plt

SPACE_SIZE = (9, 9)
plt.rcParams["figure.figsize"] = (15, 15)
plt.ion()  # interactive mode
plt.style.use("dark_background")


def rotation_mat(degrees):
    theta = np.radians(degrees)
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([
        [cos, -sin, 0],
        [sin, cos, 0],
        [0, 0, 1]
    ])


def translation_mat(dx, dy):
    T = np.identity(3)  # TODO
    return T


def scale_mat(sx, sy):
    S = np.identity(3)  # TODO
    return S


def scew_mat(x, y):
    W = np.identity(3)  # TODO
    return W


def dot(X, Y):

    is_transposed = False

    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    if X.shape[1] != Y.shape[0]:
        is_transposed = True
        Y = np.transpose(Y)

    X_rows = X.shape[0]
    Y_rows = Y.shape[0]
    Y_columns = Y.shape[1]

    product = np.zeros((X_rows, Y_columns))
    for i in range(X_rows):
        for j in range(Y_columns):
            product[i, j] = np.sum(X[i, :] * Y[:, j])

    if is_transposed:
        product = np.transpose(product)

    if product.shape[0] == 1:
        product = product.flatten()

    return product


def vec2d_to_vec3d(vec2):
    I = np.array([
        [1, 0],
        [0, 1],
        [0, 0]
    ])
    vec3 = np.dot(I, vec2) + np.array([0, 0, 1])
    return vec3


def vec3d_to_vec2d(vec3):
    I = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])
    vec2 = np.dot(I, vec3)
    return vec2


class StaticObject:
    def __init__(self, vec_pos):
        self.vec_pos: np.ndarray = vec_pos.astype(np.float)
        self.vec_dir_init = np.array([0.0, 1.0])
        self.vec_dir = np.array(self.vec_dir_init)
        self.geometry: list[np.ndarray] = []
        self.__angle: float = 0
        self.color = 'r'

        self.C = np.identity(3)
        self.R = np.identity(3)
        self.S = np.identity(3)
        self.T = np.identity(3)

        self.update_transformation()

    def set_angle(self, angle):
        self.__angle = angle
        self.R = rotation_mat(angle)

        vec3d = vec2d_to_vec3d(self.vec_dir_init)
        vec3d = dot(self.R, vec3d)
        self.vec_dir = vec3d_to_vec2d(vec3d)

        self.update_transformation()

    def get_angle(self) -> float:
        return self.__angle

    def update_movement(self, delta_time: float):
        pass

    def update_transformation(self):
        self.T = translation_mat(self.vec_pos[0], self.vec_pos[1])

        self.C = np.identity(3)
        self.C = self.R @ self.C
        self.C = self.T @ self.C
        # TODO chain together R, T, S in C using correct order!

    def draw(self):
        x_values = []
        y_values = []

        for vec2d in self.geometry:
            vec3d = vec2d_to_vec3d(vec2d)
            vec3d = dot(self.C, vec3d)
            vec2d = vec3d_to_vec2d(vec3d)

            x_values.append(vec2d[0])
            y_values.append(vec2d[1])

        plt.plot(x_values, y_values, c=self.color)


class Planet(StaticObject):

    def __init__(self, vec_pos: np.ndarray = None):
        super().__init__(vec_pos)

        self.color = 'b'
        self.vec_F = np.zeros(2, )  # panet gravity force

        step = 2 * np.pi / 20
        self.radius = np.random.random() * 0.5 + 1.0
        self.geometry = []
        theta = 0
        while theta < 2 * np.pi:
            self.geometry.append(np.array([
                np.cos(theta) * self.radius,
                np.sin(theta) * self.radius
            ]))
            theta += step
        self.geometry.append(np.array(self.geometry[0]))
        self.update_transformation()

    def update_movement(self, delta_time: float):
        # TODO implement gravity calculation with regards to player
        self.vec_F = np.zeros(2, )


class MovableObject(StaticObject):
    def __init__(self, vec_pos):
        super().__init__(vec_pos)
        self.speed = 0

    def update_movement(self, delta_time: float):
        self.vec_pos += self.vec_dir * self.speed * delta_time
        self.update_transformation()

        if abs(self.vec_pos[0]) > SPACE_SIZE[0]:
            self.vec_pos[0] = -self.vec_pos[0] / abs(self.vec_pos[0]) * SPACE_SIZE[0]
        if abs(self.vec_pos[1]) > SPACE_SIZE[1]:
            self.vec_pos[1] = -self.vec_pos[1] / abs(self.vec_pos[1]) * SPACE_SIZE[1]


class Asteroid(MovableObject):
    def __init__(self, vec_pos):
        super().__init__(vec_pos)

        self.color = 'g'

        step = 2 * np.pi / 20
        self.radius = np.random.random() * 0.2 + 0.2
        self.geometry = []
        theta = 0
        while theta < 2 * np.pi:
            radius = self.radius + np.random.random() * 0.1 - 0.05
            self.geometry.append(np.array([
                np.cos(theta) * radius,
                np.sin(theta) * radius
            ]))
            theta += step
        self.geometry.append(np.array(self.geometry[0]))

        self.S = scew_mat(x=0.2 * np.random.random(), y=0.2 * np.random.random())

        self.speed = np.random.random() * 20 + 10
        self.set_angle(np.random.random() * 360)
        self.update_transformation()

    def update_movement(self, delta_time: float):
        super().update_movement(delta_time)


class Rocket(MovableObject):
    def __init__(self, vec_pos):
        super().__init__(vec_pos)

        self.color = 'y'

        self.geometry = np.array([
            [0, -0.1],
            [0, 0.1],
        ])

        self.pos = np.array(Player.get_instance().vec_pos)
        self.speed = 60 + Player.get_instance().speed
        self.set_angle(Player.get_instance().get_angle())
        self.update_transformation()

    def update_movement(self, delta_time: float):
        super().update_movement(delta_time)


class Player(MovableObject):
    _instance: Player = None

    def __init__(self, vec_pos):
        super().__init__(vec_pos)
        self.geometry = np.array([
            [-1, 0],
            [1, 0],
            [0, 1],
            [-1, 0]
        ])

        T_centered = translation_mat(dx=0, dy=-0.5)
        self.S = np.dot(scale_mat(sx=0.2, sy=0.5), T_centered)
        self.speed = 0
        self.update_transformation()

        if not Player._instance:
            Player._instance = self
        else:
            raise Exception("Cannot construct singleton twice")

    def activate_thrusters(self):
        self.speed += 50.0

    def fire_rocket(self):
        rocket = Rocket(self.vec_pos)
        Game.get_instance().actors.append(rocket)

    def update_movement(self, delta_time: float):
        self.speed -= delta_time * 30.0
        self.speed = max(0, self.speed)

        for actor in Game.get_instance().actors:
            if isinstance(actor, Planet):
                self.vec_pos += actor.vec_F * delta_time

        super().update_movement(delta_time)

    @staticmethod
    def get_instance() -> Player:
        if not Player._instance:
            Player()
        return Player._instance


class Game:
    _instance: Game = None

    def __init__(self):
        super(Game, self).__init__()
        self.is_running = True
        self.score = 0
        self.lives = 0

        self.actors: list[StaticObject] = [
            Player(vec_pos=np.array([0, 0])),
            Planet(vec_pos=np.array([-7, -3])),
            Planet(vec_pos=np.array([8, -4])),
        ]

        for _ in range(5):
            asteroid = Asteroid(vec_pos=np.array(
                [
                    np.random.randint(-SPACE_SIZE[0], SPACE_SIZE[0]),
                    np.random.randint(-SPACE_SIZE[1], SPACE_SIZE[1]),
                ]
            ))
            self.actors.append(asteroid)

        if not Game._instance:
            Game._instance = self
        else:
            raise Exception("Cannot construct singleton twice")

    def press(self: Game, event: matplotlib.backend_bases.Event):
        player = Player.get_instance()
        print("press", event.key)
        if event.key == "escape":
            self.is_running = False
        elif event.key == "right":
            player.set_angle(player.get_angle() - 5)
        elif event.key == "left":
            player.set_angle(player.get_angle() + 5)
        elif event.key == "up":
            player.activate_thrusters()
        elif event.key == " ":
            player.fire_rocket()

    def on_close(self: Game, event: matplotlib.backend_bases.Event):
        self.is_running = False

    def main(self: Game):

        fig, _ = plt.subplots()
        fig.canvas.mpl_connect("key_press_event", self.press)
        fig.canvas.mpl_connect("close_event", self.on_close)
        dt = 1e-3

        while self.is_running:
            plt.clf()
            plt.axis("off")

            plt.title(
                f"score: {self.score} speed: {round(Player.get_instance().speed, 1)} pos:  {Player.get_instance().vec_pos}")
            plt.tight_layout(pad=0)

            plt.xlim(-10, 10)
            plt.ylim(-10, 10)

            for actor in self.actors:  # polymorhism
                actor.update_movement(dt)
                actor.draw()

            plt.draw()
            plt.pause(dt)

    @staticmethod
    def get_instance() -> Game:
        if not Game._instance:
            Game()
        return Game._instance


game = Game.get_instance()
game.main()
