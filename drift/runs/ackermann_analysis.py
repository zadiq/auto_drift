import os
import tensorflow as tf
import glob
import numpy as np
import pickle
from collections import deque
from drift.models.ddpg import DDPG, DDPGParams, DDPGSummary
from drift.robots.ackermann import Ackermann
from drift.commons import ParamsBase, Sequence


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


class Params(ParamsBase, DDPGParams):
    base_path = "/home/zadiq/uni/Personal Project/data/3D Simulation/15-03-2018/ddpg_run_3"


def load_ack_model(params):
    ack = Ackermann(
        action_dim=params.action_dim,
        reward_v=params.reward_v,
        desired_state=params.desired_state,
        reward_params=params.reward_params,
        thr_lim=params.throttle_lim
    )
    start = False

    while not start:
        start = input("Is the gazebo model running?(y/n) >> ")
        if start == 'y':
            start = True
        else:
            start = False

    return ack


def infer(ack, params):
    tf.reset_default_graph()
    with tf.Session() as sess:
        # set random random seed for reproducibility
        np.random.seed(params.seed)
        tf.random.set_random_seed(params.seed)

        ack = ack
        seq = Sequence()
        coords = Sequence()

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

        dt = 0
        # deque sequence for stacking states
        state_sequence = deque(
            [np.zeros(params.state_dim) for _ in range(params.stack_len)],
            maxlen=params.stack_len
        )
        inf_reward = []

        while dt <= params.drive_dur:
            current_state = ack.get_state().to_array()
            coords(ack.coord)
            state_sequence.append(current_state)

            ss = state_sequence.copy()
            ss.reverse()
            cs = np.array(ss).ravel().reshape(1, -1)

            action = ddpg.predict(cs, False).ravel().tolist()
            info = ack(*action)
            inf_reward.append(info['reward'])
            dt += ack.dt
            seq(info)

        return np.array(inf_reward), seq, coords


def get_inf_score(ep):
    return analysis["scores"][ep][1]


if __name__ == "__main__":
    analysis = {
        "rewards": [],
        "scores": {},
        "sequence": [],
        "episodes": [],
        "coord": [],
        "states": []
    }
    _params = Params.from_json(Params.base_path)
    episodes_path = glob.glob(os.path.join(_params.base_path, "eps-*"))
    _ack = load_ack_model(_params)
    for path in episodes_path:
        base_split = os.path.basename(path).split("-")
        train_score = int(base_split[3].split(".")[0])
        ep = _params.episode = int(base_split[1])
        try:
            re, sequence, coord = infer(_ack, _params)
        except Exception as e:
            print(e)
            print(f"Error occurred in ep: {ep}")
        else:
            analysis["episodes"].append(ep)
            analysis["sequence"].append(sequence)
            analysis["coord"].append(coord)
            analysis["rewards"].append(re)
            analysis["scores"][ep] = (train_score, int(re.sum()))
            _ack.reset()

    print("\t Eps \t|\t Train Score \t|\t Inf Score \t")
    print("\t --- \t|\t ----------- \t|\t --------- \t")
    for ep in sorted(analysis['scores'], key=get_inf_score):
        train, inf = analysis['scores'][ep]
        print(f"\t {ep} \t|\t {train} \t\t|\t {inf} \t")

    with open(os.path.join(_params.base_path, 'inference', 'analysis.pkl'), "wb") as fp:
        pickle.dump(analysis, fp)
