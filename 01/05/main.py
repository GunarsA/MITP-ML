import numpy as np
import os
import matplotlib

if os.name == "darwin":
    matplotlib.use("MacOSX")  # for mac
else:
    matplotlib.use("TkAgg")  # for unix/windows

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (15, 15)  # size of window
plt.ion()  # interactive mode
plt.style.use('dark_background')


# TODO add class structures

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class StaticObject:  # https://refactoring.guru/design-patterns/singleton/python/example
    def __init__(self):
        self.vec_pos = np.array([0., 0.])
        self.vec_dir_init = np.array([0., 0.])
        self.vec_dir = np.array([0., 0.])
        self.geometry = [
            np.array([-1, -1]),
            np.array([-1, 1]),
            np.array([1, 1]),
            np.array([1, -1]),
            np.array([-1, -1]),
        ]
        self.__angle = 0.

    def set_angle(self, angle) -> None:
        pass

    def get_angle(self) -> float:
        pass

    def update_movement(self, delta_time: float) -> None:
        pass

    def draw(self) -> None:
        pass


class Planet(StaticObject):
    def __init__(self):
        super().__init__()

    def update_movement(self, delta_time: float) -> None:
        pass


class MovableObject(StaticObject):
    def __init__(self):
        super().__init__()
        self.speed = 0.

    def update_movement(self, delta_time: float) -> None:
        pass


class Asteroid(MovableObject):
    def __init__(self):
        super().__init__()

    def update_movement(self, delta_time: float) -> None:
        pass


class Player(MovableObject, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__()
        self.rockets = [Rocket()]

    def activate_thrusters(self) -> None:
        pass

    def fire_rocket(self) -> None:
        pass

    def update_movement(self, delta_time) -> None:
        pass


class Rocket(MovableObject):
    def update_movement(self, delta_time) -> None:
        pass


class Dummy():
    def __init__(self):
        super().__init__()
        self.geometry = [
            np.array([-1, -1]),
            np.array([-1, 1]),
            np.array([1, 1]),
            np.array([1, -1]),
            np.array([-1, -1]),
        ]

    def update_movment(self, dt):
        pass

    def draw(self):
        x_data = []  # temporary variable use instead self.geometry
        y_data = []
        for vec2 in self.geometry:
            x_data.append(vec2[0])
            y_data.append(vec2[1])
        plt.plot(x_data, y_data)


class Game(metaclass=SingletonMeta):
    def __init__(self):
        super(Game, self).__init__()
        self.is_running = True
        self.score = 0
        self.lives = 0

        self.actors = [Player(), Planet(), Asteroid()]  # TODO add Player, Planets and Asteroids

    def press(self, event):
        player = Player()  # TODO get player
        print('press', event.key)
        if event.key == 'escape':
            self.is_running = False  # quits app
        elif event.key == 'right':
            player.set_angle(player.get_angle() - 5)
        elif event.key == 'left':
            player.set_angle(player.get_angle() + 5)
        elif event.key == ' ':
            player.activate_thrusters()

    def on_close(self, event):
        self.is_running = False

    def main(self):

        fig, _ = plt.subplots()
        fig.canvas.mpl_connect('key_press_event', self.press)
        fig.canvas.mpl_connect('close_event', self.on_close)
        dt = 1e-3

        while self.is_running:
            plt.clf()
            plt.axis('off')
            plt.tight_layout(pad=0)

            plt.xlim(-10, 10)
            plt.ylim(-10, 10)

            for actor in self.actors:  # polymorhism
                actor.update_movement(dt)
                actor.draw()

            plt.draw()
            plt.pause(dt)


game = Game()
game.main()
