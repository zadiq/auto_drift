import tensorflow as tf
import tflearn
import numpy as np
import os
from drift.robots.ackermann import Ackermann
from drift.models.ddpg import DDPGParams, DDPG, DDPGSummary
from drift.commons import (
    ParamsBase, Sequence,
    OrnsteinUhlenbeckActionNoise
)
from collections import deque


def pprint(e, obj):  # pretty print episode updates
    print(f"Episode: {e}")
    print("============")
    [print(f"{key}: {v}") for key, v in obj.items()]


class Params(DDPGParams, ParamsBase):
    """"""
    path = "/home/zadiq/dev/@p/auto_drift/data/3D/ddpg_run_1"  # path to model
    seed = 434
    reward_v = "v1"
    desired_state = 10
    reward_params = {'sigma': .9}
    sigma = 0.001
    # sigma = 0.001

    use_batch_norm = False
    max_episodes = int(10e6)
    stack_len = 1
    drive_dur = 10
    save_ep_over = -299
    print_interval = 10

    action_bound = [0.2]  # [steering, throttle] bounds
    throttle = 4  # default throttle value
    # the applied throttle will be (throttle_lim + x),
    # where x is the predicted throttle, x in
    # [-throttle_bound, throttle_bound].
    # throttle_bound = action_bound[1]
    throttle_lim = 4

    def __init__(self):
        self.action_dim = len(self.action_bound)


def train():
    ddpg.update_target_weights()
    tflearn.is_training(params.use_batch_norm)
    max_ep_rew = -9999

    for e in range(params.max_episodes):
        episode_playback = Sequence()
        ack.reset()
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
            current_state = ack.get_state().to_array()
            state_sequence.append(current_state)

            ss = state_sequence.copy()
            ss.reverse()
            cs = np.array(ss).ravel().reshape(1, -1)

            action = list((ddpg.predict(cs, False) + actor_noise()).ravel())

            # use default throttle if throttle is not predicted
            if params.action_dim > 1:
                action.append(params.throttle)
            info = ack(*action)

            dt += ack.dt
            if dt > params.drive_dur:
                info['terminal'] = True
                terminated = True

            buffer(info)
            episode_playback(info)

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


if __name__ == '__main__':
    """
    [+]: use bigger radius
    [+]: retune friction
    [ ]: train the same model used to train ddpg_run_4 to train on states:
        [+]: 1, -1, 2 (a must): failed
        [ ]: 1, -2, 2 wor
        [ ]: 2, -1, 2
        [ ]: 3, -1, 2
    [ ]: train with both action and throttle on 0.4, 0.4
    """
    params = Params()
    params.to_json(params.path)

    # set random random seed for reproducibility
    np.random.seed(params.seed)
    tf.random.set_random_seed(params.seed)

    buffer = Sequence()
    ack = Ackermann(
        action_dim=params.action_dim,
        reward_v=params.reward_v,
        desired_state=params.desired_state,
        reward_params=params.reward_params,
        thr_lim=params.throttle_lim
    )
    actor_noise = OrnsteinUhlenbeckActionNoise(
        mu=np.zeros(params.action_dim), sigma=params.sigma
    )

    with tf.Session() as sess:
        ddpg = DDPG(params, sess)
        summary = DDPGSummary(sess, params.path)
        sess.run(tf.global_variables_initializer())
        start = False

        while not start:
            start = input("Is the gazebo model running?(y/n) >> ")
            if start == 'y':
                start = True
            else:
                start = False

        train()
