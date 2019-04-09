"""
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task
and developed with tflearn + Tensorflow

https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html#Tensorflow

Author: Patrick Emami
"""
import tensorflow as tf
import numpy as np
import tflearn
import os
import json
from drift.robots.eth import ETHModel
from drift.commons import Sequence, Animation
from collections import deque


# ===========================
#   Actor and Critic DNNzs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size, m_params):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.params = m_params

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, self.params.actor_layer_1)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, self.params.actor_layer_2)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init, name="actor")
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound, name="scaled-actor")
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars, m_params):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.params = m_params

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, self.params.critic_layer_1)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, self.params.critic_layer_2)
        t2 = tflearn.fully_connected(action, self.params.critic_layer_2)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init, name="critic")
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on htp://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
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


# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Total Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)
    max_reward = tf.Variable(0.)
    tf.summary.scalar("Max Reward", max_reward)

    summary_vars = [episode_reward, episode_ave_max_q, max_reward]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


# ===========================
#   Agent Training
# ===========================

class Params:

    max_episodes = int(10e6)
    state_dim = 3
    action_dim = 1
    stack_len = 1
    drive_dur = 20  # drive duration
    action_bound = np.radians(np.radians(80))  # double radian (due to code bug when tuning parameters)
    tau = 0.001  # tau used to update target weights,
    gamma = 0.99  # discount factor
    batch_size = 40
    buffer_size = int(1e6)
    actor_lr = 0.0001  # actor learning rate
    critic_lr = 0.001  # critic learning rate
    print_interval = 10
    ignore = True  # ignore early stop
    sigma = 0.005  # actor's noise
    save_ep_over = -289  # save any model with this reward or more
    seed = 4112
    dynamics = {
        'B': 8,
        'C': 9.5,
        'D': 2500,

        'Cm1': 1250,
        'Cm2': 3,  # this must be an odd number
        'Cr': 100,
        'Cd': 45,
    }
    actor_layer_1 = 405
    actor_layer_2 = 300
    critic_layer_1 = 405
    critic_layer_2 = 300
    use_batch_norm = False

    # ETHModel parameters
    reward_version = "v1"
    ds = "three"
    reward_params = {'sigma': .9}

    def to_json(self, save_path):
        save_path = os.path.join(save_path, 'params.json')

        with open(save_path, 'w') as fp:
            data = {}
            for a in dir(self):
                if not a.startswith("__"):
                    data[a] = getattr(self, a)

            data.pop("to_json")
            data.pop("from_json")
            json.dump(data, fp, indent=4)
            print("dumping params")

    def from_json(self, save_path):
        with open(save_path, 'r') as fp:
            data = json.load(fp)
            for k, v in data.items():
                setattr(self, k, v)


def pprint(e, obj):  # pretty episode updates
    print(f"Episode: {e}")
    print("============")
    [print(f"{key}: {v}") for key, v in obj.items()]


def model_is_done(m: ETHModel, ignore=False):

    if ignore:
        return False

    var = m.variables

    if var['X'] >= 6.5:
        return True

    if var['X'] <= -1:
        return True

    if var['Y'] >= 6.5:
        return True

    if var['Y'] <= -1:
        return True

    return False


def train():
    summary_ops, summary_vars = build_summaries()
    actor.update_target_network()
    critic.update_target_network()

    writer = tf.summary.FileWriter(path, sess.graph)
    max_ep_rew = -9999

    tflearn.is_training(params.use_batch_norm)
    for e in range(params.max_episodes):

        # sequence data for every episode
        episode_playback = Sequence()
        eth.reset()

        # deque sequence for stacking states
        state_sequence = deque(
            [np.zeros(params.state_dim) for _ in range(params.stack_len)],
            maxlen=params.stack_len
        )

        dt = 0
        ep_avg_max_q = 0
        rewards = []
        terminated = False

        while dt <= params.drive_dur:
            # for i in range(params.batch_size + 1):
            current_state = eth.get_state().to_array()
            state_sequence.append(current_state)

            ss = state_sequence.copy()
            ss.reverse()
            cs = np.array(ss).ravel().reshape(1, -1)

            action = (actor.predict(cs) + actor_noise()).ravel()[0]
            info = eth(action, duty_cycle=1)

            dt += eth.variables.dt
            if dt > params.drive_dur or model_is_done(eth, params.ignore):
                if dt < params.drive_dur:
                    print("terminated early")
                info['terminal'] = True
                terminated = True

            buffer(info)
            episode_playback(eth.variables.group)

            if len(buffer) > params.batch_size:  # train network
                batch, _ = buffer.get_batch(params.batch_size)

                t_actions = actor.predict_target(batch['new_state'])
                target_q_values = critic.predict_target(batch['new_state'], t_actions)

                est_q_values = []

                for r, t, t_q in zip(batch['reward'], batch['terminal'], target_q_values):
                    if t:
                        est_q_values.append(r)
                    else:
                        est_q_values.append(r + (params.gamma * t_q))

                est_q_values = np.reshape(est_q_values, (len(batch), 1))
                pred_q_value,  _ = critic.train(
                    batch['current_state'], batch['action'], est_q_values
                )

                ep_avg_max_q += np.amax(pred_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(batch['current_state'])
                grads = critic.action_gradients(batch['current_state'], a_outs)
                actor.train(batch['current_state'], grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

                rewards.append(info['reward'])

            if terminated:
                break

        ep_updates = {
            'ep_reward': np.sum(rewards),
            'ep_mean_reward': np.mean(rewards),
            'ep_max_reward': max(rewards),
        }

        if ((ep_updates['ep_reward'] > max_ep_rew) or
                (ep_updates['ep_reward'] >= params.save_ep_over)):
            """save best score"""
            max_ep_rew = int(max(max_ep_rew, (ep_updates['ep_reward'])))
            prefix = f"eps-{e}-{int(ep_updates['ep_reward'])}"
            episode_playback.dump(
                path,
                prefix=prefix
            )
            saver.save(sess, os.path.join(path, "model", prefix + ".model"))
            print(f"Saving good model with reward: {ep_updates['ep_reward']}")

        summary_str = sess.run(summary_ops, feed_dict={
            summary_vars[0]: ep_updates['ep_reward'],
            summary_vars[1]: ep_avg_max_q,
            summary_vars[2]: ep_updates['ep_max_reward'],
        })

        writer.add_summary(summary_str, e)
        writer.flush()

        if (e % params.print_interval) == 0:
            pprint(e, ep_updates)
        else:
            msg = "(Terminated)" if terminated else ""
            print(f"Episode[{e}/{params.max_episodes}]: {ep_updates['ep_reward']} MAX({max_ep_rew}) {msg}")


def infer():
    """Load a model and infer"""

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
        action = (actor.predict(cs) + 0).ravel()[0]
        _, _, reward, _, _ = eth(action, duty_cycle=1)
        inf_reward.append(reward)

        dt += eth.variables.dt

        eth.sequence(eth.variables.group)
    ani = Animation(eth.sequence)
    ani.show()
    eth.sequence.dump(path=infer_path, prefix=infer_prefix)
    inf_reward = np.array(inf_reward)
    print({
        "Total Reward: ": inf_reward.sum(),
        "Max Reward: ": inf_reward.max(),
        "Mean Reward: ": inf_reward.mean(),
    })


def fill_model_var(model, m_vars=None):
    m_vars = m_vars or {}
    model.variables.__dict__.update(m_vars)
    return model


if __name__ == '__main__':

    with tf.Session() as sess:
        path = '/home/zadiq/dev/@p/drift_project/data/02-03-2019/ddpg_run_5'

        # inference path
        _ = os.path.join
        base_path = "/home/zadiq/dev/@p/drift_project/data"
        model_date = "02-03-2019"
        model_run = "1"
        model_name = "eps-27--244"

        model_path = _(base_path, f"{model_date}", f"ddpg_run_{model_run}", "model", f"{model_name}.model")
        infer_path = _(base_path, "inference")
        infer_prefix = f"{model_date}-{model_run}-{model_name}-inference"
        param_path = _(base_path, f"{model_date}", f"ddpg_run_{model_run}", "params.json")

        is_train = False  # true to Train and False to infer
        warm_start = False  # set true if you want start training with a trained model

        params = Params()
        if not is_train:
            params.from_json(param_path)

        np.random.seed(params.seed)
        tf.random.set_random_seed(params.seed)

        buffer = Sequence(params.buffer_size)

        eth = ETHModel(
            imp=params.dynamics,
            reward_version=params.reward_version,
            ds=params.ds,
            reward_params=params.reward_params
        )

        actor = ActorNetwork(
            sess, params.state_dim, params.action_dim,
            params.action_bound, params.actor_lr, params.tau,
            params.batch_size, params
        )

        critic = CriticNetwork(
            sess, params.state_dim, params.action_dim,
            params.critic_lr, params.tau,
            params.gamma, actor.get_num_trainable_vars(), params
        )

        actor_noise = OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(params.action_dim), sigma=params.sigma
        )

        saver = tf.train.Saver(
            # TODO Remember to change this when training for a long time
            max_to_keep=100
        )
        sess.run(tf.global_variables_initializer())

        if warm_start or not is_train:
            """load model if warm starting or inferring"""
            print(f"Loading model from: {model_path}")
            saver.restore(sess, model_path)
            params.warm_start = warm_start
            if warm_start:
                params.warm_start_model = model_path
        try:
            if is_train:
                params.to_json(path)
                train()
            else:
                infer()
        except KeyboardInterrupt:
            print("Exited Gracefully!!!")


# TODO:
"""
- Check the effect of early stopping
- CHeck the effect of Replay buffer on exploration and exploitation
- Layers, learning rate, sigma, tau
- Try decaying gamma (gamma is desired reward)
- check the effect of batch normalisation (by setting is_trainable to true)
- Change length of time
- Include plane incline 
- USED GRIDSEARCH or BAYESIAN SEARCH TO FIND THE OPTIMAL COMBINATION OF DYNAMIC
 PARAMETER THAN ACHIEVES HIGHEST DRIFT REWARD FOR A SINGLE MODEL.
"""
