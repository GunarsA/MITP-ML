import numpy as np
import os
import matplotlib

if os.name == "darwin":
    matplotlib.use("MacOSX")  # for mac
else:
    matplotlib.use("TkAgg") # for unix/windows

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 15) # size of window
plt.ion() # interactive mode
plt.style.use('dark_background')


# TODO add class structures

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

class Game:
    def __init__(self):
        super(Game, self).__init__()
        self.is_running = True
        self.score = 0
        self.lives = 0

        self.actors = [Dummy()] # TODO add Player, Planets and Asteroids

    def press(self, event):
        player = None  # TODO get player
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
                actor.update_movment(dt)
                actor.draw()

            plt.draw()
            plt.pause(dt)

game = Game()
game.main()