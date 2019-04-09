"""
This script is used for training the ETHModel vehicle with DDPG to achieve
drifting. The Params class is used to configure the Dynamic Model and DDPG
architecture.
"""

import tensorflow as tf
import tflearn
import numpy as np
import os
from collections import deque
from drift.robots import ETHModel
from drift.models.ddpg import DDPG, DDPGParams, DDPGSummary
from drift.commons import Sequence, OrnsteinUhlenbeckActionNoise, ParamsBase


def pprint(e, obj):  # pretty print episode updates
    print(f"Episode: {e}")
    print("============")
    [print(f"{key}: {v}") for key, v in obj.items()]


class Params(DDPGParams, ParamsBase):
    path = "/home/zadiq/temp/ddpg"
    seed = 4112
    use_batch_norm = False
    max_episodes = int(10e6)
    stack_len = 1
    drive_dur = 20
    sigma = 0.005
    print_interval = 10
    save_ep_over = -299

    # eth model params
    dynamics = {
        'B': 8,
        'C': 9.5,
        'D': 2500,

        'Cm1': 1250,
        'Cm2': 3,  # this must be an odd number
        'Cr': 100,
        'Cd': 45,
    }
    reward_version = "v1"
    ds = "three"
    reward_params = {'sigma': .9}


def train():
    ddpg.update_target_weights()
    tflearn.is_training(params.use_batch_norm)
    max_ep_rew = -9999

    for e in range(params.max_episodes):
        episode_playback = Sequence()
        eth.reset()
        ddpg.reset()

        # deque sequence for stacking states
        state_sequence = deque(
            [np.zeros(params.state_dim) for _ in range(params.stack_len)],
            maxlen=params.stack_len
        )

        dt = 0
        rewards = []
        terminated = False

        while dt <= params.drive_dur:
            current_state = eth.get_state().to_array()
            state_sequence.append(current_state)

            ss = state_sequence.copy()
            ss.reverse()
            cs = np.array(ss).ravel().reshape(1, -1)

            action = (ddpg.predict(cs, False) + actor_noise()).ravel()[0]
            info = eth(action, duty_cycle=1)

            dt += eth.variables.dt
            if dt > params.drive_dur:
                info['terminal'] = True
                terminated = True

            buffer(info)
            episode_playback(eth.variables.group)

            if len(buffer) > params.batch_size:
                batch, _ = buffer.get_batch(params.batch_size)
                ddpg.train(batch)
                rewards.append(info['reward'])

            if terminated:
                break

        ep_updates = {
            'ep_reward': np.sum(rewards),
            'ep_mean_reward': np.mean(rewards),
            'ep_max_reward': max(rewards),
        }
        summary_values = [
            ep_updates['ep_reward'],
            ddpg.ep_avg_max_q,
            ep_updates['ep_max_reward']
        ]
        summary.write_summary(e, summary_values)

        if ((ep_updates['ep_reward'] > max_ep_rew) or
                (ep_updates['ep_reward'] >= params.save_ep_over)):
            """save best score"""
            max_ep_rew = int(max(max_ep_rew, (ep_updates['ep_reward'])))
            prefix = f"eps-{e}-{int(ep_updates['ep_reward'])}"
            episode_playback.dump(params.path, prefix=prefix)
            model_path = os.path.join(params.path, "model", prefix + ".model")
            summary.save_model(model_path)
            print(f"Saving good model with reward: {ep_updates['ep_reward']}")

        if (e % params.print_interval) == 0:
            pprint(e, ep_updates)
        else:
            msg = "(Terminated)" if terminated else ""
            print(f"Episode[{e}/{params.max_episodes}]: "
                  f"{ep_updates['ep_reward']} "
                  f"MAX({max_ep_rew}) {msg}")


if __name__ == "__main__":
    with tf.Session() as sess:
        params = Params()
        params.to_json(params.path)

        # set random random seed for reproducibility
        np.random.seed(params.seed)
        tf.random.set_random_seed(params.seed)

        buffer = Sequence()
        eth = ETHModel(
            imp=params.dynamics,
            reward_version=params.reward_version,
            ds=params.ds,
            reward_params=params.reward_params
        )
        actor_noise = OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(params.action_dim), sigma=params.sigma
        )

        ddpg = DDPG(params, sess)
        summary = DDPGSummary(sess, params.path)
        sess.run(tf.global_variables_initializer())

        train()
