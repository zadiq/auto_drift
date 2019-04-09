"""
Code for searching for the dynamic parameters that the model can control
optimally.
"""
import os
import tensorflow as tf
import numpy as np
# import dill
from collections import deque
from drift.robots import ETHModel
from drift.models.ddpg_tensor import ActorNetwork, CriticNetwork, Params
from skopt import gp_minimize, dump, load
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_evaluations, plot_objective


np.random.seed(777)


class DynamicSearch:
    """"""
    model_path = ""

    def __init__(self, run_search=True):
        self.eth = ETHModel(
            imp="drifty", ds="three",
            reward_version="v1",
            reward_params={'sigma': .9}
        )

        self.dynamics = [
            Integer(6, 10, name="B"),
            Real(8, 11, name="C"),
            Integer(500, 4000, name="D"),

            Integer(750, 1500, name="Cm1"),
            # Categorical([1, 3, 5, 7, 9, 11, 13, 15, 17], name="Cm2"),
            Integer(1, 17, name="Cm2"),
            Integer(50, 150, name="Cr"),
            Integer(30, 60, name="Cd"),
        ]
        self.dynamic_labels = [
            'B', 'C', 'D', 'Cm1', 'Cm2', 'Cr', 'Cd'
        ]

        self.gp_model = None
        self.gp_run = 0
        self.params = Params()

        with tf.Session() as self.sess:
            self.actor = ActorNetwork(
                self.sess, self.params.state_dim, self.params.action_dim,
                self.params.action_bound, self.params.actor_lr,
                self.params.tau, self.params.batch_size
            )

            self.critic = CriticNetwork(
                self.sess, self.params.state_dim, self.params.action_dim,
                self.params.critic_lr, self.params.tau, self.params.gamma,
                self.actor.get_num_trainable_vars()
            )
            self.sess.run(tf.global_variables_initializer())

            self.load_c_model(
                "26-02-2019",
                "3", "eps-38--171"
            )

            if run_search:
                self.run()
                self.dump_gp_model()

    def run(self):  # run search

        @use_named_args(self.dynamics)
        def run_inference(**params):
            self.eth.reset()
            self.eth.variables.__dict__.update(params)

            dt = 0
            state_sequence = deque(
                [np.zeros(self.params.state_dim) for _ in range(self.params.stack_len)],
                maxlen=self.params.stack_len
            )
            inf_reward = []

            while dt < self.params.drive_dur:
                current_state = self.eth.get_state().to_array()
                state_sequence.append(current_state)

                ss = state_sequence.copy()
                ss.reverse()
                cs = np.array(ss).ravel().reshape(1, -1)
                action = (self.actor.predict(cs) + 0).ravel()[0]
                _, _, reward, _, _ = self.eth(action, duty_cycle=1)
                inf_reward.append(reward)

                dt += self.eth.variables.dt
                self.eth.sequence(self.eth.variables.group)

            #  negate to turn reward into cost
            cost = -1 * np.array(inf_reward).sum()
            if np.isnan(cost):
                cost = 2000

            print(f"Run ({self.gp_run}) >> {cost}")
            self.gp_run += 1
            
            return cost

        self.gp_model = gp_minimize(
            run_inference,
            self.dynamics,
            n_calls=200, n_jobs=2,
            random_state=42,
        )

    def get_gp_path(self):
        return self.model_path + ".gp_model"

    def dump_gp_model(self):
        dump_path = self.get_gp_path()
        # with open(dump_path, 'wb') as fp:
        #     dill.dump(self.gp_model, fp)
        dump(self.gp_model, dump_path, store_objective=False)

    def load_gp_model(self):
        self.gp_model = load(self.get_gp_path())

    def load_c_model(self,  date, run, name, base_path=None):
        """Load control model"""
        saver = tf.train.Saver()
        _ = os.path.join
        base_path = base_path or "/home/zadiq/dev/@p/drift_project/data"
        self.model_path = _(base_path, f"{date}", f"ddpg_run_{run}", "model", f"{name}.model")
        saver.restore(self.sess, self.model_path)

    def plot(self):
        if self.gp_model is None:
            self.load_gp_model()

        plot_evaluations(self.gp_model, dimensions=self.dynamic_labels)
        plot_objective(self.gp_model, dimensions=self.dynamic_labels)
        plot_convergence(self.gp_model)

    def best_dynamics(self, as_list=False):
        """Get best dynamic parameters from gp_model used for search"""
        if self.gp_model is None:
            raise ValueError("Need to load gp model first")

        if as_list:
            return self.gp_model.x
        return dict(zip(self.dynamic_labels, self.gp_model.x))


if __name__ == "__main__":
    dy_search = DynamicSearch(False)
