from drift.robots.ackermann import Ackermann
import numpy as np


def load_ack_model():
    ack = Ackermann()
    start = False

    while not start:
        start = input("Is the gazebo model running?(y/n) >> ")
        if start == 'y':
            start = True
        else:
            start = False

    return ack


def simulate_func(model):
    t = 0
    i = 0
    s = 0
    intervals = 20, 5  # left, right turn intervals
    model.sequence.clear()
    model.reset()
    while t <= 20:
        if i <= intervals[0]:  # left turn
            s = np.radians(10)
        elif intervals[0] < i < np.sum(intervals):  # right turn
            s = 0
        else:  # reset
            i = 0
        d = 3.5
        model.sequence(model(s, d))
        i += 1
        t += model.dt


if __name__ == "__main__":
    ack = load_ack_model()
    simulate_func(ack)
