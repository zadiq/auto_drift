import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import os
import json
from matplotlib.animation import FuncAnimation
from matplotlib.transforms import Affine2D
from collections import OrderedDict, deque
from functools import wraps


def is_iter(obj):
    """
    Check if an object is iterable
    """

    try:
        iter(obj)
        return True
    except TypeError:
        return False


class Vector:
    """
    An ordered and named vector
    """

    def __init__(self, values, *args, keys, name=None):

        self.init = True
        if type(values) in [int, float]:
            values = (values,)
        vec = [*values, *args]
        assert len(vec) == len(keys), ValueError("numbers of values do not match that of keys")
        self.keys = list(keys)
        self.vector = OrderedDict()
        self.update(vec)
        self.name = name
        self.init = False
        self.tolist = self.to_list

    def __getitem__(self, item):
        if type(item) == tuple:
            """Return a Vector instance if requesting for more than one item"""
            return Vector([self.vector[i] for i in item], keys=item)
        return self.vector[item]

    def __setitem__(self, key, value):
        self.vector[key] = value
        if key not in self.keys:
            self.keys.append(key)

    def __iter__(self):
        yield from self.vector.values()

    def __len__(self):
        return len(self.vector)

    def __sub__(self, other):
        return self.to_array() - other.to_array()

    def items(self):
        for k, v in zip(self.keys, self.to_list()):
            yield k, v

    def append(self, k, v):
        self.keys.append(k)
        self.vector[k] = v

    def update(self, vec):
        if not self.init:
            assert len(vec) == len(self), ValueError("Length of provided iterable "
                                                     "must match vector")
        for i, k in enumerate(self.keys):
            self[k] = vec[i]

    def to_list(self):
        return list(self.vector.values())

    def to_array(self):
        return np.array(self.to_list())

    def __call__(self, vector=None):
        """
        Update if vector is provided and
        return Vector values as an array.
        :param vector: a iterable containing values to be updated. Must be in order of
                       keys and have same length as vector
        """
        if vector is not None:
            self.update(vector)
        return self.to_array()

    def __repr__(self):
        return "Vector<{}>".format(self.to_list())


class GroupVector:
    """
    Group multiple vectors together
    """

    def __init__(self, *vectors):
        self.vectors = vectors
        self.keys = []
        for i, v in enumerate(vectors):
            self.keys.extend([f"{i}_{k}" for k in v.keys])

    def __iter__(self):
        yield from self.keys

    def __getitem__(self, item):
        keys = item.split('_')
        i = keys[0]
        k = "_".join(keys[1:])
        return self.vectors[int(i)][k]

    def items(self):
        for k in self:
            yield k, self[k]


class Sequence:
    """
    A class used to store and update multiple items' values in time sequence.
    Format
    ------
    {
        'item_1': [t1, t2, ..., tn],
        'item_2': [t1, t2, ..., tn],
        'item_3': [t1, t2, ..., tn],
    }
    - Now supports dynamic lengths for each items (all items do not have to have
      the numbers of time steps)
    """
    def __init__(self, max_len=None):
        self.max_len = max_len  # maximum length of item data
        self.sequence = {}
        self.can_strip_key = False

    def clear(self):
        self.sequence.clear()

    def __call__(self, variables):

        if type(variables) == GroupVector:
            self.can_strip_key = True

        for k, v in variables.items():
            if k not in self.sequence:
                self.sequence[k] = deque(maxlen=self.max_len)
            self.sequence[k].append(v)

    def __getitem__(self, item):
        return self.sequence[item]

    def __len__(self):
        if not self.sequence:
            return 0
        # return length of one of the items
        return len(self.sequence[list(self.keys())[0]])

    def __iter__(self):
        yield from self.sequence

    def data_as_array(self):
        s = Sequence()
        for k in self.sequence:
            s.sequence[k] = np.array(self[k])
        return s

    def dump(self, path, prefix="", append_time=False):
        """Save sequence to file"""
        ap_time = f"-{int(time.time() * 1e6)}" if append_time else ""
        full_path = os.path.join(path, f"{prefix}{ap_time}.sequence")
        with open(full_path, 'wb') as fp:
            pickle.dump(self, fp)

        print(f"Successfully dumped sequence to {full_path}")

    @classmethod
    def load(cls, path):
        """Load sequence from file"""
        with open(path, 'rb') as fp:
            obj: cls = pickle.load(fp)
            return obj

    def get_batch(self, size, stack_len=1, to_stack=("current_state", "new_state"),
                  has_terminal=True):
        """
        TODO requires improvement, looks confusing
        TODO need to implement terminal

        :param has_terminal: indicate whether sequence has an ite named terminal
        :param size: size of batch
        :param to_stack: keys of data to stack, others will single index
        :param stack_len: length of previous sequential data to prepend with indexed data
        :return:
        """
        batch = Sequence(max_len=size)
        batch_index = np.random.choice(range(len(self)), size)

        def update_data(keys=self, stacking=False):
            for k in keys:
                _bool = stacking and has_terminal and self['terminal'][oi-1]
                if i < 0 or _bool:
                    o_data = self[k][oi]
                    data = [0] * len(o_data) if is_iter(o_data) else 0
                else:
                    data = self[k][i]

                if is_iter(data):
                    stack[k].extend(data)
                else:
                    stack[k].append(data)

        for i in batch_index:
            stack = Vector([[] for _ in range(len(self.keys()))], keys=self.keys())
            oi = i  # original index
            update_data()

            for _ in range(stack_len - 1):
                i -= 1
                update_data(to_stack, True)  # stack previous sequence

            batch(stack)

        return batch.data_as_array(), batch_index

    def keys(self):
        return self.sequence.keys()

    def plot(self, fig_size=(20, 50), path=False, sharex="all", plot_ind=False):

        def strip_key(key):
            """extract name from key"""
            if self.can_strip_key:
                n = key.split('_')
                return "_".join(n[1:])
            return key

        rows = (len(self.sequence) // 2) + (len(self.sequence) % 2)
        fig, axes = plt.subplots(rows, 2, sharex=sharex, figsize=fig_size)
        axes = axes.ravel()

        if plot_ind:
            plt.figure()
            plt.title("Plot of vehicle's trajectory in X-axis")
            plt.plot(self['0_X'])
            plt.xlabel("time (s)")
            plt.ylabel("Distance (m)")

            plt.figure()
            plt.title("Plot of vehicle's trajectory in Y-axis")
            plt.plot(self['0_Y'])
            plt.xlabel("time (s)")
            plt.ylabel("Distance (m)")

            plt.figure()
            plt.title("Plot of vehicle's yaw")
            plt.plot(self['0_yaw'])
            plt.xlabel("time (s)")
            plt.ylabel("Angle (degrees)")

        for i, k in enumerate(self):
            axes[i].set_title(strip_key(k))
            axes[i].plot(self[k], label=strip_key(k))
            axes[i].grid()
            axes[i].legend()
        if path:
            plt.figure()
            plt.title("Path")
            plt.plot(self['0_X'], self['0_Y'], label="{}".format(max(self['0_X'])))
            plt.legend()
        plt.show()


class Scale:
    """
    A simple (unnecessary) class for scaling model
    parameters.
    """

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, value):
        return value / self.scale


class Animation:
    """
    Used to illustrate the path and movement of models/agents based on
    provided sequence
    """

    def __init__(self, sequence: Sequence, imp="normal"):
        self.yaw = np.array(sequence['0_yaw']) % (2 * np.pi)
        self.x_array = np.array(sequence['0_X'])
        self.y_array = np.array(sequence['0_Y'])
        self.sequence = sequence

        r = {  # robot patch parameters
            'w': 6,  # width
            'h': 3,  # height
            'hw': 1,  # head width
        }
        scale_factor = max(abs(self.x_array).max(), abs(self.y_array).max())
        scale = 150 / scale_factor  # scale down
        robot_shape = np.array([r['w'], r['h']]) / scale
        self.robot = plt.Rectangle((0, 0), *robot_shape, fc="r")
        head_shape = np.array([r['hw'], r['h']]) / scale
        self.robot_head = plt.Rectangle((0, 0), *head_shape, fc="black")

        self.set_pos((0, 0))

        # plot
        self.fig = plt.figure()
        self.ax = plt.axes()
        plt.title(f"2D motion path of {imp} profile")
        plt.xlabel("X")
        plt.ylabel("Y")
        self.fig.set_dpi(100)
        self.fig.set_size_inches(7, 6.5)
        plt.plot(self.x_array, self.y_array)
        plt.plot([self.x_array[0]], [self.y_array[0]], marker='d', color="g", label="Start Point")
        plt.plot([self.x_array[-1]], [self.y_array[-1]], marker='d', color="r", label="End Point")
        self.ax.add_patch(self.robot)
        self.ax.add_patch(self.robot_head)
        plt.axis('scaled')
        plt.legend()

    @property
    def params(self):
        return dict(
            fig=self.fig, func=self.animate,
            frames=len(self.yaw), blit=True,
            interval=10,
        )

    def set_pos(self, pos):
        # set robot position
        w, h = self.robot.get_width(), self.robot.get_height()
        x, y = pos[0] - (w / 2), pos[1] - (h / 2)
        self.robot.set_xy((x, y))

        # set robot head position
        self.robot_head.set_xy((x + w, y))

    def transform(self, t):
        self.robot.set_transform(t)
        self.robot_head.set_transform(t)

    def plot_at(self, i):
        """
        :param i: index
        :return:
        """
        x = self.x_array[i]
        y = self.y_array[i]
        t = Affine2D().rotate_around(x, y, self.yaw[i]) + self.ax.transData
        self.transform(t)
        self.set_pos((x, y))

    def show(self):
        # necessary to assign to FuncAnimation a variable for it to be displayed
        ani = FuncAnimation(**self.params)
        plt.show()
        return ani

    def save(self, *args, **kwargs):
        animation = FuncAnimation(**self.params)
        animation.save(*args, **kwargs)

    def animate(self, i):
        self.plot_at(i)
        return self.robot, self.robot_head


class Agent:
    """
    A wrapper class to wrap models (simulation and real-world
    models).
    """

    def __init__(self, v_model):
        """
        :param v_model: vehicle model. could be simulation or real world model.
        """
        self.v_model = v_model

    def __call__(self, *args, **kwargs):
        return self.v_model(*args, **kwargs)

    def __getattr__(self, item):
        """
        Fallback to model's attributes if not implemented
        here.
        """
        return getattr(self.v_model, item)


def extract_returns(call_func):
    """
    A decorator function used to wrap vehicle model __call__ function.
    It extracts necessary parameters before and after performing an action
    and returns them.
    """
    keys = ['current_state', 'action', 'reward', 'new_state', 'terminal']
    call_return = Vector([0] * len(keys), keys=keys)

    @wraps(call_func)
    def wrapper(self, *args, **kwargs):
        current_state = self.get_state()  # get current state before performing action
        call_func(self, *args, **kwargs)  # perform action
        self.new_state = new_state = self.get_state()
        action = self.action
        call_return([
            current_state, action,
            self.get_reward(), new_state,
            False,  # terminal value will be updated in main loop
        ])
        return call_return

    return wrapper


class OrnsteinUhlenbeckActionNoise:
    """
    Taken from: https://github.com/pemami4911/deep-rl/blob/master/ddpg/ddpg.py
    # Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
    # based on htp://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    """
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class ParamsBase:
    """
    A base class for Param classes, can save/load params
    to/from json.
    """
    excludes = [  # list of attributes to exclude when saving
        "to_json",
        "from_json"
    ]

    def to_json(self, save_path):
        save_path = os.path.join(save_path, "params.json")

        with open(save_path, 'w') as fp:
            data = {}
            for a in dir(self):
                if not a.startswith("__"):
                    data[a] = getattr(self, a)

            [data.pop(x) for x in self.excludes]

            json.dump(data, fp, indent=4)

    @classmethod
    def from_json(cls, save_path):
        save_path = os.path.join(save_path, "params.json")
        obj = cls()
        with open(save_path, 'r') as fp:
            data = json.load(fp)
            for k, v in data.items():
                setattr(obj, k, v)
        return obj
