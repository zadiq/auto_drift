import numpy as np
from drift.commons import (
    Vector, Sequence,
    GroupVector, Scale,
    extract_returns
)
from drift.metrics import get_drift_reward


class ETHParameters:
    """
    A class used to set ETH model parameters with different implementations.
    Different implementations exhibit different behaviours.
    """
    keys = [
        'B', 'C', 'D',
        'Cm1', 'Cm2', 'Cr', 'Cd'
    ]

    def __init__(self, variable, which):
        getattr(self, '_' + which)()
        [setattr(variable, k, getattr(self, k)) for k in self.keys]

    def __iter__(self):
        yield from self.keys

    def _normal(self):
        """
        - travels at 7.2 m/s on a straight line
        - doesn't drift when driving at angle
        """
        self.B = 10
        self.C = 0.8
        self.D = 2700

        self.Cm1 = 1250
        self.Cm2 = 1
        self.Cr = 100
        self.Cd = 45

    def _drifty(self):
        """
        - hard to control
        - madness
        - B:C{comments}
            - 1:6{moves weird @15deg}
            - 7:9.5{kinda okay}
        """
        self.B = 8
        self.C = 9.5  #
        self.D = 2500

        self.Cm1 = 1250
        self.Cm2 = 3  # this must be an odd number
        self.Cr = 100
        self.Cd = 45


class ETHVariables:
    """
    Variables Description
    ---------------------
    vx_dot:         linear acceleration in longitudinal direction
    vy_dot:         linear acceleration in latitude direction
    omega_dot:      angular acceleration in reference to the global frame
    X_dot:          linear X velocity in reference to the global frame
    Y_dot:          linear Y velocity in reference to the global frame
    yaw_dot|omega:  angular velocity in reference to the global frame
    yaw:            angular rotation around the z axis
    vx:             linear velocity in longitudinal direction
    vy:             linear velocity in latitude direction
    delta:          steering applied to front wheel
    duty_cycle:     duty cycle use to apply velocity to the vehicle
    v:              velocity of vehicle
    """

    def __init__(self, imp="normal", dt=0.01, scale=43):
        """
        :param imp: parameter implementation
        :param dt: delta time
        :param scale: value for scaling down model's physical attributes
        """

        variables = [
            'vx_dot', 'vy_dot', 'omega_dot',
            'vx', 'vy', 'omega', 'yaw',
            'X_dot', 'Y_dot',
            'F_fy', 'F_ry', 'F_rx',
            'X', 'Y', 'v', 'radius'
        ]

        self.space_variables = Vector([0] * len(variables), keys=variables)
        self.control_variables = Vector(0, 1, keys=['delta', 'duty_cycle'])

        self.group = GroupVector(self.space_variables, self.control_variables)
        _ = Scale(scale)

        self.mass = _(2500)
        self.Iz = _(300.9)
        self.lf = _(1.1)
        self.lr = _(1.59)

        self.B = None
        self.C = None
        self.D = None
        self.Cm1 = None
        self.Cm2 = None
        self.Cr = None
        self.Cd = None

        # can base dynamic variables directly as dict or a string for
        # choosing a dynamic profile
        if type(imp) == dict:
            self.__dict__.update(imp)
        else:
            ETHParameters(self, imp)

        self.dt = dt

    def __getitem__(self, item):
        return self.space_variables[item]

    def __setitem__(self, key, value):
        self.space_variables[key] = value

    @staticmethod
    def assert_which(which):
        assert which in ['f', 'r'], ValueError("Invalid which provided")

    @property
    def front_wheel_y_slip(self):  # alpha_f
        """equation (2a)."""
        delta = self.control_variables['delta']
        omega_dot = self.space_variables['omega_dot']
        vy = self.space_variables['vy']
        vx = self.space_variables['vx']

        return delta - np.arctan2((omega_dot*self.lf) + vy, vx)

    @property
    def rear_wheel_y_slip(self):  # alpha_r
        """equation (2b)."""
        omega_dot = self.space_variables['omega_dot']
        vy = self.space_variables['vy']
        vx = self.space_variables['vx']

        return np.arctan2((omega_dot*self.lr) - vy, vx)

    def wheel_y_force(self, which):
        """
        Lateral forces of front and rear wheels
        equation (2a, 2b)
        """
        self.assert_which(which)
        if which == 'f':
            slip, var_key = self.front_wheel_y_slip, 'F_fy'
        else:
            slip, var_key = self.rear_wheel_y_slip, 'F_ry'

        force = self.D * np.sin(self.C * np.arctan(self.B * slip))
        self.space_variables[var_key] = force

        return force

    @property
    def front_wheel_y_force(self):  # F_fy
        return self.wheel_y_force('f')

    @property
    def rear_wheel_y_force(self):  # F_ry
        return self.wheel_y_force('r')

    @property
    def rear_wheel_x_force(self):
        """
        Longitudinal forces on the rear wheel
        equation (2c).
        """
        d = self.control_variables['duty_cycle']
        vx = self.space_variables['vx']

        force = ((self.Cm1 - (self.Cm2 * vx)) * d) - self.Cr - (self.Cd * (vx**2))
        self.space_variables['F_rx'] = force
        return force


class ETHDesiredStates:

    def __init__(self):
        self.keys = ['vx', 'vy', 'omega']

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def one(self):
        """
        Desired average velocity of 5, tilted around 23.96 deg
        :return:
        """
        return Vector(4.5, 2, 3.125, keys=self.keys)

    @property
    def two(self):
        """
        Desired average velocity of 5, tilted around 36.87 deg
        :return:
        """
        return Vector(4, 3, 3.125, keys=self.keys)

    @property
    def three(self):
        """[NEW]
        Desired average velocity of 4.6, tilted around 27.12 deg
        :return:
        """
        return Vector(4.1, -2.1, 2.3, keys=self.keys)


class ETHModel:
    """
    Based On: Optimization-based autonomous racing of 1:43 scale RC cars
    """

    def __init__(self, ds="one", reward_params=None, angle_mode="r",
                 reward_version="v2", **kwargs):
        """
        :param angle_mode: (r, radian | d, degree) to specify the angle mode
            of steering values to be provided when using __call__.
        :param reward_params: other parameters passed to reward function
        :param ds: a string mapped to one of ETHDesiredStates properties.
                   Used to choose a desired state
        :param kwargs: other arguments passed to ETHVariables
        """
        mode_choices = ['r', 'radian', 'd', 'degree']
        assert angle_mode in mode_choices, ValueError(f'Invalid angle '
                                                      f'mode choose from {mode_choices}')
        self.angle_mode = angle_mode
        self.reward_params = reward_params or {}
        self.desired_state = ETHDesiredStates()[ds]
        self.reward_func = get_drift_reward(reward_version)
        self.kwargs = kwargs
        self._build()

    def _build(self):
        self.variables = ETHVariables(**self.kwargs)
        self.sequence = Sequence()
        self.action = None

    def update_sequence(self):
        self.sequence(self.variables.group)

    def reset(self):
        self._build()

    def save(self, path):
        """
        TODO implement this, reset before saving
        :param path:
        :return:
        """

    def resolve_angle(self, angle):
        if self.angle_mode in ['r', 'radian']:
            return angle
        return np.radians(angle)

    @extract_returns
    def __call__(self, steering, duty_cycle=None):
        self.action = steering = self.resolve_angle(steering)
        self.variables.control_variables['delta'] = steering

        if duty_cycle is not None:
            self.variables.control_variables['duty_cycle'] = duty_cycle

        self.solve_space_vector()
        self.update_space_variables()

    def simulate(self, func=None, *args, **kwargs):
        if func is not None:
            """Use custom simulation function"""
            return func(self, *args, **kwargs)
        else:
            t = 0
            while t <= 100:
                self.sequence(self.variables.group)
                s = 0.01 if t < 50 else 0
                self(s)
                t += self.variables.dt

    def solve_space_vector(self):
        delta = self.variables.control_variables['delta']

        vx = self.variables['vx']
        vy = self.variables['vy']
        yaw = self.variables['yaw']
        omega = self.variables['omega']
        m = self.variables.mass
        iz = self.variables.Iz
        lf = self.variables.lf
        lr = self.variables.lr
        f_fy = self.variables.front_wheel_y_force
        f_rx = self.variables.rear_wheel_x_force
        f_ry = self.variables.rear_wheel_y_force

        self.variables['X_dot'] = (vx * np.cos(yaw)) - (vy * np.sin(yaw))
        self.variables['Y_dot'] = (vx * np.sin(yaw)) + (vy * np.cos(yaw))

        self.variables['vx_dot'] = (f_rx - (f_fy * np.sin(delta)) + (m * vy * omega)) / m
        self.variables['vy_dot'] = (f_ry + (f_fy * np.cos(delta)) - (m * vx * omega)) / m
        self.variables['omega_dot'] = ((f_fy * lf * np.cos(delta)) - (f_ry * lr)) / iz

    def update_space_variables(self):
        dt = self.variables.dt

        self.variables['vx'] += (dt * self.variables['vx_dot'])
        self.variables['vy'] += (dt * self.variables['vy_dot'])
        self.variables['v'] = np.sqrt((self.variables['vx']**2) + (self.variables['vy']**2))
        self.variables['omega'] += (dt * self.variables['omega_dot'])
        self.variables['radius'] = self.variables['v'] / self.variables['omega']
        self.variables['yaw'] += (dt * self.variables['omega'])

        self.variables['X'] += (dt * self.variables['X_dot'])
        self.variables['Y'] += (dt * self.variables['Y_dot'])

    def get_state(self):
        """
        Returns a Vector of variables used to represent the
        state of the vehicle.
        """
        return self.variables['vx', 'vy', 'omega']

    @staticmethod
    def sample_steering(bound=30):
        """
        Returns a random steering value within a range.
        """
        return np.radians(np.random.uniform(-bound, bound))

    def get_reward(self):
        new_state = self.get_state()  # get new_state after performing an action
        return self.reward_func(self.desired_state, new_state, **self.reward_params)

    @property
    def info(self):
        """
        dimension of action and state
        :return:
        """
        return Vector([1, 3], keys=['action', 'state'])
