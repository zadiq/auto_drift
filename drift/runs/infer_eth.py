"""
Infer a DDPG model on an ETHModel vehicle.
"""
import os
import glob
import tensorflow as tf
import numpy as np
from drift.robots import ETHModel
from collections import deque
from drift.models.ddpg import DDPG, DDPGParams, DDPGSummary
from drift.commons import ParamsBase


class Params(DDPGParams, ParamsBase):
    """Params class"""
    # path to base path containing models and params.json
    base_path = "/home/zadiq/dev/@p/drift_project/data/02-03-2019/ddpg_run_4"

    # the episode of the model that will be inferred
    episode = 108


def infer():
    """"""
    dt = 0
    # deque sequence for stacking states
    state_sequence = deque(
        [np.zeros(params.state_dim) for _ in range(params.stack_len)],
        maxlen=params.stack_len
    )
    inf_reward = []

    while dt < params.drive_dur:
        current_state = eth.get_state().to_array()
        state_sequence.append(current_state)

        ss = state_sequence.copy()
        ss.reverse()
        cs = np.array(ss).ravel().reshape(1, -1)
        action = (ddpg.predict(cs) + 0).ravel()[0]
        _, _, reward, _, _ = eth(action, duty_cycle=1)
        inf_reward.append(reward)

        dt += eth.variables.dt

        eth.sequence(eth.variables.group)

    inf_reward = np.array(inf_reward)
    print({
        "Total Reward: ": inf_reward.sum(),
        "Max Reward: ": inf_reward.max(),
        "Mean Reward: ": inf_reward.mean(),
    })


def main():
    ...


if __name__ == "__main__":
    params = Params.from_json(Params.base_path)

    with tf.Session() as sess:

        # set random random seed for reproducibility
        np.random.seed(params.seed)
        tf.random.set_random_seed(params.seed)

        # buffer = Sequence()
        eth = ETHModel(
            imp=params.dynamics,
            reward_version=params.reward_version,
            ds=params.ds,
            reward_params=params.reward_params
        )

        ddpg = DDPG(params, sess)
        summary = DDPGSummary(sess)
        sess.run(tf.global_variables_initializer())

        # load and restore weights
        model_path = os.path.join(
            params.base_path, "model",
            f"eps-{params.episode}*"
        )
        model_path = glob.glob(model_path)[0].split(".")
        model_path = ".".join(model_path[:2])
        summary.load_model(model_path)

        infer()
