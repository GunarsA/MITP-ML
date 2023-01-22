import math
import numpy as np
import matplotlib.pyplot as plt


def f(x) -> float:
    return math.sin(2 * x) + 2 * math.e ** (3 * x)


def main():
    x = np.linspace(-2, 2, 100)

    # the function, which is y = x^3 here
    y = [f(i) for i in x]

    # setting the axes at the centre
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # plot the function
    plt.plot(x, y, 'b')

    # show the plot
    plt.show()


if __name__ == '__main__':
    main()
