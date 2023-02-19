import numpy as np
import sys
import matplotlib

if sys.platform == 'darwin':
    matplotlib.use("MacOSX")  # for mac
else:
    matplotlib.use("TkAgg")  # for unix/windows

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (7, 7)  # size of window
plt.ion()
plt.style.use('dark_background')

target_point = np.array([-3.0, 0])
anchor_point = np.array([0, 0])

is_running = True


def button_press_event(event):
    global target_point
    target_point = np.array([event.xdata, event.ydata])


def press(event):
    global is_running
    print('press', event.key)
    if event.key == 'escape':
        is_running = False  # quits app


def on_close():
    global is_running
    is_running = False


fig, _ = plt.subplots()
fig.canvas.mpl_connect('close_event', on_close)
fig.canvas.mpl_connect('button_press_event', button_press_event)
fig.canvas.mpl_connect('key_press_event', press)

length_joint = 2.0
theta_1 = np.deg2rad(-10)
theta_2 = np.deg2rad(-10)


def rotation(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([
        [c, -s],
        [s, c]]
    )
    return R


while is_running:
    plt.clf()

    segment = np.array([0.0, 1.0]) * length_joint
    joints = []

    R1 = rotation(theta_1)
    R2 = rotation(theta_2)

    joints.append(anchor_point)

    point_1 = np.dot(R1, segment)
    joints.append(point_1)

    point_2 = R1 @ (R2 @ segment) + point_1
    joints.append(point_2)

    np_joints = np.array(joints)

    loss = np.sum((point_2 - target_point) ** 2)

    if len(np_joints):
        plt.plot(np_joints[:, 0], np_joints[:, 1])
    plt.scatter(target_point[0], target_point[1], s=50, c='r')

    plt.title(
        f'theta_1: {round(np.rad2deg(theta_1))} '
        f'theta_2: {round(np.rad2deg(theta_2))} '
        f'loss: {loss}'
    )

    plt.xlim(-5, 5)
    plt.ylim(0, 10)
    plt.draw()
    plt.pause(1e-3)
exit()
