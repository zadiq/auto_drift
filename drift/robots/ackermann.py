"""
An interface with the Gazebo Ackermann model
"""
from threading import Thread
import socket
import struct
import socketserver
import json
import time
import numpy as np
from drift.commons import Vector, extract_returns, Sequence
from drift.metrics import get_drift_reward

"""
TODO:
- The reset in controller requires a timer between pausing and unpausing [+]
- Issues with services dying, temp resolved using inf retries but it is slowing down simulation [+p]
- get_state is being called three times [+]
- Try to reduce simulation time [+]
- How to record/log gazebo data [+]
- 

Notes:
-----
- Gravity can be tuned between -7.5 and -9.8
- Throttle between 4 and 4.5
- Steering bounded at .20

Analysis:
--------
- Increasing Z gravity (-9.8) increase vx but little effect on vy 
"""


class ServerHandler(socketserver.BaseRequestHandler):

    def handle(self):
        raw_data = self.receive_data()
        data = json.loads(raw_data.decode('ascii'))
        # print(data)
        self.server.processor(data)

    def receive_data(self):
        raw_data_len = self.receive_all(4)
        if not raw_data_len:
            return None
        data_len = struct.unpack('>I', raw_data_len)[0]
        return self.receive_all(data_len)

    def receive_all(self, n):
        data = b''
        while len(data) < n:
            packet = self.request.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


class Ackermann:
    """
    Communicates with the ros package controlling the Ackermann model
    on Gazebo.
    """

    def __init__(self, host="localhost", port=9190,
                 action_dim=1, reward_v="v1", desired_state=0,
                 reward_params=None, thr_lim=0):

        # create server
        self.server = ThreadedTCPServer((host, port), ServerHandler)
        ip, host = self.server.server_address
        self.server.processor = self.process_data
        server_thread = Thread(target=self.server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        print(f"Serving Ackermann interface at ({ip}, {host})")

        # controller server address
        self.con_server_address = ("localhost", 9091)

        self.states = None
        self.can_send_actions = False
        self.new_state = None
        self.coord = None
        self.action_dim = action_dim
        self.reward_func = get_drift_reward(reward_v)
        self.reward_params = reward_params or {}
        self.desired_state = self.get_desired_state[desired_state]
        self.sequence = Sequence()
        self.dt = 0.01
        self.throttle_lim = thr_lim

    @property
    def get_desired_state(self):
        keys = ['vx', 'vy', 'omega']
        return [                                    # ix    mu1     mu2
            Vector(2.1, -1.17, 1.6, keys=keys),     # 0
            Vector(3.5, -2, 2, keys=keys),          # 1
            Vector(4.1, -2.1, 2.3, keys=keys),      # 2
            Vector(3.6, -2.2, 2.1, keys=keys),      # 3
            Vector(4.1, -1.9, 2.1, keys=keys),      # 4
            Vector(2.1, -3.8, 2, keys=keys),        # 5
            Vector(1, -2, 2, keys=keys),            # 6
            Vector(1.2, -2.6, 2, keys=keys),        # 7     0.2     0.4
            Vector(2, -2, 2, keys=keys),            # 8     0.4     0.4
            Vector(1, -1, 2, keys=keys),            # 9     0.4     0.4
        ]

    @extract_returns
    def __call__(self, steering=0.15, throttle=4):
        """Apply actions"""
        if self.action_dim > 1:
            throttle += self.throttle_lim
            self.action = [steering, throttle]
        else:
            self.action = steering
        # self.action = steering if self.action_dim == 1 else [steering, throttle]
        while not self.can_send_actions:
            """Wait until controller is ready to receive action"""
            time.sleep(.0001)

        # disable sending action until the controller is ready
        self.can_send_actions = False
        data = {
            'action': {
                "throttle": throttle,
                "steering": steering
            }
        }
        self.send_data(data)

        while not self.can_send_actions:
            """wait until controller receives data"""
            time.sleep(.0001)

    def send_data(self, data):
        ros_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ros_client.settimeout(3)
        ros_client.connect(self.con_server_address)
        data = bytes(json.dumps(data), 'ascii')
        data = struct.pack('>I', len(data)) + data
        ros_client.sendall(data)
        ros_client.close()

    def get_state(self):
        if self.states is None:
            # print("hit here")
            self.send_data({
                'request_states': True
            })
            while not self.states:
                time.sleep(.0001)

        states = self.states
        self.states = None
        # self.coord = None
        return states

    def reset(self):
        self.sequence.clear()
        self.states = None
        self.send_data({
            'reset': True
        })
        time.sleep(0.1)

    def process_data(self, data):
        """Process incoming data from Ackermann Controller"""
        response = {
            'error': False
        }

        if response.get("error"):
            print(response['error'])

        if data.get("request_action"):
            """Store states"""
            self.can_send_actions = data['request_action']

        if data.get("states"):
            """store states when it is sent by the controller"""
            states = data.get("states")
            self.states = Vector(
                [states['vx'], states['vy'], states['omega']],
                keys=['vx', 'vy', 'omega']
            )
            self.coord = Vector(
                [states['x'], states['y'], states['yaw']],
                keys=['x', 'y', 'yaw']
            )

    @staticmethod
    def sample_action(bound):
        return np.random.uniform(-bound, bound)

    def simulate(self, steering=.15, throttle=4):
        _ = self.sample_action
        dt = 0.01
        t = 0
        i = 0
        start = time.time()
        while t <= 5:
            self.sequence(self(steering, throttle))
            # print(i, self.get_reward())
            t += dt
            i += 1
            time.sleep(1/60)
        print(f"Finish time: {time.time() - start}")

    def simulate_drift(self, steering=0.15, throttle=4):
        intervals = 20, 5
        t = 0
        i = 0
        s = 0
        dt = 0.01
        while t <= 5:
            if i <= intervals[0]:  # left turn
                s = steering
            elif intervals[0] < i < np.sum(intervals):  # right turn
                s = -steering
            else:  # reset
                i = 0
            self.sequence(self(s, throttle))
            i += 1
            t += dt

    def get_reward(self):
        # new_state = self.get_state() can get new_state filled in
        # call_return instead
        new_state = self.new_state
        return self.reward_func(self.desired_state, new_state, **self.reward_params)


if __name__ == '__main__':
    ack = Ackermann()
