"""
Implementation of Deep Deterministic Policy Gradient

Based On: CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING
Link: https://arxiv.org/pdf/1509.02971.pdf
Implementation aided with: https://github.com/pemami4911/deep-rl/blob/master/ddpg/ddpg.py
"""
import tensorflow as tf
import tflearn
import numpy as np
from tflearn.layers import fully_connected
from tflearn.layers.normalization import batch_normalization


class DDPGLayer:
    """Abstract class for a neural layer"""

    def __init__(self, dim, batch=False, activation="relu"):
        self.dim = dim
        self.batch = batch
        self.activation = activation


def build_layers(x, layers):
    """Construct neural layers"""

    for l in layers:
        x = fully_connected(x, l.dim)
        if l.batch:
            x = batch_normalization(x)
        if l.activation:
            x = getattr(tflearn.activations, l.activation)(x)
    return x


def assert_name(name):
    """Check DDPG name"""
    assert name in ['online', 'target']


class DDPGActorCriticBase:
    """Base class for DDPG Actor and Critic"""

    def __init__(self):
        self.weight_params = None
        self.online_network = None
        self.target_network_weights = []

    def set_weight_params(self, weight_params, online_network=None):
        _ = tf.multiply
        self.weight_params = weight_params

        # create tensor for updating target weights using tau
        if online_network:
            self.online_network = online_network
            for i in range(len(self.weight_params)):
                self.target_network_weights.append(
                    self.weight_params[i].assign(
                        _(self.online_network.weight_params[i], self.params.tau) +
                        _(self.weight_params[i], 1. - self.params.tau)
                    )
                )

    def update_weights(self):
        """Update a target actor's weight with online weights"""
        self.sess.run(self.target_network_weights)


class DDPGParams:
    """DDPG architecture params"""

    action_dim = 1  # steering
    state_dim = 3  # vx, vy, w
    tau = 0.001  # target update tau
    batch_size = 20
    gamma = 0.99  # discount factor

    # actor's params
    actor_network = [
        (400, False, "relu"),  # dim, batch, activation
        (300, False, "relu"),
    ]
    actor_activator = "tanh"  # output layer activation
    action_bound = np.radians(np.radians(80))  # double rad: a bug that was discovered after tuning dynamic parameters
    actor_lr = 0.0001

    # critic's params
    critic_network = [
        (400, False, "relu"),  # dim, batch, activation
    ]
    critic_last_layer = 300
    critic_activator = "relu"
    critic_lr = 0.001

    # weights initialisation params
    layer_weights_min = -3e-3
    layer_weights_max = 3e-3


class DDPGActor(DDPGActorCriticBase):
    """
    DDPG Actor's Network
    An actor takes states as input and predicts an action that leads
    to the next state.
    """

    def __init__(self, params: DDPGParams, sess=None, name="online"):
        self.params = params
        self.sess = sess
        assert_name(name)
        self.name = name
        self._build_network()

        super().__init__()
        self.grads = None
        self.optimise = None

        # this will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.params.action_dim])

    def _build_network(self):
        layers = [DDPGLayer(*l) for l in self.params.actor_network]
        self.states = tflearn.input_data([None, self.params.state_dim])
        x = build_layers(self.states, layers)
        weights_init = tflearn.initializations.uniform(
            minval=self.params.layer_weights_min,
            maxval=self.params.layer_weights_max
        )
        self.action = tflearn.fully_connected(
            x, self.params.action_dim, self.params.actor_activator,
            weights_init=weights_init, name="actor"
        )
        self.scaled_action = tf.multiply(
            self.action, self.params.action_bound,
            name="scaled-actor"
        )

    def set_info(self, weight_params, online_actor=None):
        """
        used to set actor variable info
        :param weight_params: learning parameters after building network
        :param online_actor: provide online actor when setting info for target actor
        :return:
        """
        self.set_weight_params(weight_params, online_actor)

        if not online_actor:
            unnorm_gradients = tf.gradients(
                self.scaled_action, self.weight_params,
                -self.action_gradient
            )
            self.grads = list(map(lambda x: tf.div(x, self.params.batch_size), unnorm_gradients))
            # optimiser = tf.train.AdamOptimizer(self.params.actor_lr)
            self.optimise = tf.train.AdamOptimizer(self.params.actor_lr).apply_gradients(
                zip(self.grads, self.weight_params)
            )

    def train(self, states, action_grads):
        self.sess.run(self.optimise, feed_dict={
            self.states: states,
            self.action_gradient: action_grads
        })

    def predict(self, states):
        """Predict action based on provided states"""
        return self.sess.run(self.scaled_action, feed_dict={
            self.states: states
        })


class DDPGCritic(DDPGActorCriticBase):
    """
    DDPG Critic Network
    A critic takes in states and action(s) as inputs and predicts a q_value given
    a the old state and action.
    """

    def __init__(self, params: DDPGParams, sess, name="online"):
        self.params = params
        self.sess = sess
        assert_name(name)
        self.name = name
        self._build()
        super().__init__()

    def _build(self):
        self.states = tflearn.input_data(shape=[None, self.params.state_dim])
        self.actions = tflearn.input_data(shape=[None, self.params.action_dim])
        layers = [DDPGLayer(*l) for l in self.params.critic_network]
        x = build_layers(self.states, layers)

        # As stated in the paper, the action is not added to the layers
        # until the last one for low dimensional action space. Separate
        # last layer is created for actions and then the weights are then
        # combined through addition.
        t1 = fully_connected(x, self.params.critic_last_layer)
        t2 = fully_connected(self.actions, self.params.critic_last_layer)
        _ = tf.matmul
        combined = _(x, t1.W) + _(self.actions, t2.W) + t2.b
        x = tflearn.activation(combined, activation=self.params.critic_activator)

        weights_init = tflearn.initializations.uniform(
            minval=self.params.layer_weights_min,
            maxval=self.params.layer_weights_max
        )
        self.q_values = fully_connected(x, 1, weights_init=weights_init, name="critic")

        if self.name == "online":
            # Estimated y value (the future expected reward value also known).
            self.est_y_values = tf.placeholder(tf.float32, [None, 1])
            self.loss = tflearn.mean_square(self.est_y_values, self.q_values)
            optimiser = tf.train.AdamOptimizer(self.params.critic_lr)
            self.optimise = optimiser.minimize(self.loss)
            self.action_gradient = tf.gradients(self.q_values, self.actions)

    def set_info(self, weights_params, online_critic=None):
        self.set_weight_params(weights_params, online_critic)

    def train(self, states, actions, est_y_values):
        return self.sess.run(
            [self.q_values, self.optimise],
            feed_dict={
                self.states: states,
                self.actions: actions,
                self.est_y_values: est_y_values
            }
        )

    def predict(self, states, actions):
        """Predict q_values given state and actions"""
        return self.sess.run(self.q_values, feed_dict={
            self.states: states,
            self.actions: actions
        })

    def get_action_gradients(self, states, actions):
        return self.sess.run(self.action_gradient, feed_dict={
            self.states: states,
            self.actions: actions
        })


class DDPGSummary:
    """
    Handler class for handling summaries
    and saving and loading models.
    """

    def __init__(self, sess, sum_path=None, max_to_keep=100):
        """
        :param sum_path: path for saving checkpoints
        """
        self.sess = sess
        self.writer = None
        if sum_path:
            self.summary_vars, self.summary_ops = self.build_summaries()
            self.writer = tf.summary.FileWriter(sum_path, sess.graph)
        self.saver = tf.train.Saver(max_to_keep=max_to_keep)

    @staticmethod
    def build_summaries():
        episode_reward = tf.Variable(-1999, trainable=False)
        tf.summary.scalar("Total Reward", episode_reward)
        eps_ave_max_q = tf.Variable(0., trainable=False)
        tf.summary.scalar("Qmax Value", eps_ave_max_q)
        max_reward = tf.Variable(0., trainable=False)
        tf.summary.scalar("Max Reward", max_reward)

        summary_vars = [episode_reward, eps_ave_max_q, max_reward]
        summary_ops = tf.summary.merge_all()

        return summary_vars, summary_ops

    def write_summary(self, ep, values: list, sess=None):
        """Write an episode summary to file"""
        if not self.writer:
            AttributeError("write_summary is only available "
                           "when sum_path is provided")
        sess = sess or self.sess
        feed = dict(zip(self.summary_vars, values))
        summary_str = sess.run(self.summary_ops, feed_dict=feed)
        self.writer.add_summary(summary_str, ep)
        self.writer.flush()

    def save_model(self, path, sess=None):
        sess = sess or self.sess
        self.saver.save(sess, path)

    def load_model(self, path, sess=None):
        sess = sess or self.sess
        self.saver.restore(sess, path)


class DDPG:
    """DDPG main class"""

    def __init__(self, params: DDPGParams, sess):
        self.params = params
        self.var_count = 0  # tf variables count
        self.ep_avg_max_q = 0

        # create Actor(s)
        self.online_actor = DDPGActor(params, sess, "online")
        self.online_actor.set_info(self.get_tf_variables())

        self.target_actor = DDPGActor(params, sess, "target")
        self.target_actor.set_info(self.get_tf_variables(), self.online_actor)

        # create Critic(s)
        self.online_critic = DDPGCritic(params, sess, "online")
        self.online_critic.set_info(self.get_tf_variables())

        self.target_critic = DDPGCritic(params, sess, "target")
        self.target_critic.set_info(self.get_tf_variables(), self.online_critic)

    def reset(self):
        self.ep_avg_max_q = 0

    def get_tf_variables(self):
        t_var = tf.trainable_variables()[self.var_count:]
        self.var_count += len(t_var)
        return t_var

    def predict(self, states, target=True):
        """
        Predict actions given states using either
        target or online actor
        """
        if target:
            return self.target_actor.predict(states)
        return self.online_actor.predict(states)

    def train(self, batch):
        """
        An episodic train step using data provided as a batch
        Batch should have __get_item__ with the following keys:
            - current_state
            - new_state
            - action
            - reward
            - terminal
            - gamma
        """
        current_states = batch['current_state']
        new_states = batch['new_state']
        actions = batch['action']
        rewards = batch['reward']
        terminals = batch['terminal']
        gamma = self.params.gamma

        t_act = self.target_actor.predict(new_states)
        t_q_values = self.target_critic.predict(new_states, t_act)

        # calculate y values for the batch set
        est_y_values = []
        for r, t, t_q in zip(rewards, terminals, t_q_values):
            if t:
                est_y_values.append(r)
            else:
                est_y_values.append(r + (gamma * t_q))
        est_y_values = np.reshape(est_y_values, (len(batch), 1))

        o_q_values, _ = self.online_critic.train(
            current_states, actions,
            est_y_values
        )

        # Update the actor policy using the sampled gradient
        self.ep_avg_max_q += np.amax(o_q_values)
        a_outs = self.online_actor.predict(current_states)
        grads = self.online_critic.get_action_gradients(current_states, a_outs)
        self.online_actor.train(current_states, grads[0])

        self.update_target_weights()

    def update_target_weights(self):
        self.target_actor.update_weights()
        self.target_critic.update_weights()
