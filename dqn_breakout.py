import gym
import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt
tf.enable_eager_execution()


def preprocess(x, frame_height=84, frame_width=84):
    cropped_x = tf.image.crop_to_bounding_box(x, 20, 0, 180, 160)
    downsize_x = tf.image.resize_images(cropped_x, size=[frame_height, frame_width],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    gray_x = tf.image.rgb_to_grayscale(downsize_x)
    return tf.squeeze(gray_x)


# Since the scale of scores varies greatly from game to game
# we fixed all positive rewards to be 1 and all negative rewards to be −1, leaving 0 rewards unchanged.
# Clipping the rewards in this manner limits the scale of the error derivatives
# and makes it easier to use the same learning rate across multiple games.
def clipping_reward(r):
    return np.sign(r) # The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0.


class Memory:
    def __init__(self, size, obs_dims, batch_size, history_length):
        self.memory_size = size
        self.observation_dims = obs_dims
        self.batch_size = batch_size
        self.history_length = history_length
        self.state_shape = [self.history_length, self.observation_dims[0], self.observation_dims[1]]
        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.int8)
        self.observations = np.empty([self.memory_size] + self.observation_dims, dtype=np.uint8)
        self.terminals = np.empty(self.memory_size, dtype=np.bool)
        self.__count = 0
        self.__current = 0

        self.prestates = np.empty([self.batch_size] + self.state_shape, dtype=np.uint8)
        self.nextstates = np.empty([self.batch_size] + self.state_shape, dtype=np.uint8)

    @property
    def get_count(self):
        return self.__count

    @property
    def get_current(self):
        return self.__current

    def add(self, next_state, action, r, terminal):
        processed_state = preprocess(next_state)
        clipped_r = clipping_reward(r)

        self.actions[self.__current] = action
        self.rewards[self.__current] = clipped_r
        self.observations[self.__current, ...] = processed_state
        self.terminals[self.__current] = terminal
        # maximum is memory size
        self.__count = max(self.__count, self.__current + 1)
        # a finite memory size
        self.__current = (self.__current + 1) % self.memory_size

    def initialise(self, size, env):
        while True:
            env.reset()
            is_done = False
            initial_lives = 5
            while True:
                if is_done or self.__count == size:
                    break
                action = env.action_space.sample()
                next_state, r, is_done, info = env.step(action)

                current_lives = info['ale.lives']

                if current_lives < initial_lives:
                    is_done = True
                    r = 0.0

                self.add(next_state, action, r, is_done)

            if self.__count == size:
                break

    def get_initial_state(self, state):
        processed_state = preprocess(state)
        states = [processed_state for _ in range(self.history_length)]
        return tf.stack(states)

    def get_states_in_training(self, idx, frames):
        next_states = frames[idx - self.history_length:idx]
        return tf.stack(next_states)

    # stack 4 frames
    def get_state(self, index):
        assert self.__count > 0, "Memory empty."
        assert index >= self.history_length - 1, "Index must be greater than or equal to 3."
        return self.observations[(index - (self.history_length - 1)):(index + 1), ...]

    def sample(self):
        indexes = []
        while len(indexes) < self.batch_size:
            while True:
                index = np.random.randint(self.history_length, self.__count - 1)
                if index < self.history_length:
                    continue
                if self.__current <= index <= self.__current + self.history_length:
                    continue
                if self.terminals[index - self.history_length:index].any():
                    continue
                break

            self.prestates[len(indexes), ...] = self.get_state(index - 1)
            self.nextstates[len(indexes), ...] = self.get_state(index)
            indexes.append(index)

        action = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        return self.prestates, action, rewards, terminals, self.nextstates


class QNetwork(tf.keras.Model):
    def __init__(self, n_actions, gamma):
        super(QNetwork, self).__init__()

        self.n_actions = n_actions
        self.gamma = gamma

        self.normaliser = tf.keras.layers.Lambda(lambda x: x / 255.0)

        self.conv_layer1 = tf.keras.layers.Conv2D(filters=32, kernel_size=[8, 8], strides=[4, 4],
                                                  kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                                                  activation='relu')
        self.conv_layer2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[4, 4], strides=[2, 2],
                                                  kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                                                  activation='relu')
        self.conv_layer3 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1],
                                                  kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                                                  activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.hidden_dense_layer = tf.keras.layers.Dense(512,
                                                        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                                                        activation='relu')
        self.output_layer = tf.keras.layers.Dense(n_actions,
                                                  kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2))

    def call(self, inputs, training=None, mask=None):
        inputs = tf.reshape(inputs, [-1, 84, 84, 4])
        normalised = self.normaliser(inputs)

        conv1 = self.conv_layer1(normalised)
        conv2 = self.conv_layer2(conv1)
        conv3 = self.conv_layer3(conv2)

        flat = self.flatten(conv3)
        h1 = self.hidden_dense_layer(flat)
        outputs = self.output_layer(h1)
        return outputs

    def compute_target(self, next_states, r, is_terminal):
        values = self.predict(next_states)
        q_values = np.empty(len(r))
        for i in range(len(r)):
            if is_terminal[i]:
                q_values[i] = r[i]
            else:
                q_values[i] = r[i] + self.gamma * tf.reduce_max(values[i, :])
        return q_values


def compute_epsilon(starting_eps, iterations):
    decay_step = 434294.0
    decay_rate = 1.0 / math.e
    calculate_eps = tf.train.exponential_decay(starting_eps, iterations, decay_step, decay_rate)
    eps = calculate_eps()
    return max(0.1, eps)


def epsilon_greedy_policy(epsilon, actions, values):
    if np.random.binomial(1, epsilon) == 1:
        return np.random.choice(actions)
    else:
        return np.random.choice([action_ for action_, value_ in enumerate(values) if value_ == np.max(values)])


def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    condition = tf.keras.backend.abs(error) < clip_delta

    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

    return tf.where(condition, squared_loss, linear_loss)


def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
    return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))


def display(x):
    plt.imshow(x, cmap='gray')
    plt.show()


def plot_loss(score):
    plt.plot(score)
    plt.ylabel('Average Score per Episode')
    plt.show()


def main():
    # '-v0' will randomly skip 2, 3, 4 frames
    # '-v4' allows the agent to see and select actions on every 4th frame instead of every frame
    #  when using '-v0', you will have to implement your own frame skipping for some games
    env = gym.make('BreakoutDeterministic-v4')

    n_actions = env.action_space.n
    actions = np.arange(0, n_actions)
    input_dim = [84, 84]
    history_len = 4
    batch_size = 32
    memory_size = 500000
    starting_epsilon = 1.0
    discount_factor = 0.99

    experience = Memory(memory_size, input_dim, batch_size, history_len)
    # experience initialisation
    replay_init_size = 50000
    experience.initialise(replay_init_size, env)

    policy_network = QNetwork(n_actions, discount_factor)
    target_network = QNetwork(n_actions, discount_factor)
    target_network.set_weights(policy_network.get_weights())
    optimiser = tf.train.RMSPropOptimizer(learning_rate=0.00025, momentum=0.95, epsilon=0.01)

    training_iterations = 0
    episodes = 150
    loss_list = []
    score_list = []

    for e in range(episodes):
        print('Episode:', e + 1)
        print('Count:', experience.get_count)
        score = 0.0
        ep_counts = 0
        update_counts = 0

        while update_counts < 50000:
            state = env.reset()
            is_done = False
            stacked_frames = []
            processed_state = preprocess(state)
            stacked_frames.append(processed_state)

            while not is_done:
                if len(stacked_frames) >= history_len:
                    # get next states
                    current_idx = len(stacked_frames)
                    next_states = experience.get_states_in_training(current_idx, stacked_frames)
                else:
                    next_states = experience.get_initial_state(state)

                q_values = policy_network.predict(next_states)
                q_values = np.squeeze(q_values)
                epsilon = compute_epsilon(starting_epsilon, training_iterations)
                action = epsilon_greedy_policy(epsilon, actions, q_values)

                next_s, r, is_done, _ = env.step(action)
                processed_next_s = preprocess(next_s)
                stacked_frames.append(processed_next_s)

                experience.add(next_s, action, r, is_done)
                score += r

                if is_done:
                    ep_counts += 1

                training_iterations += 1
                update_counts += 1

                pre_states, action_values, rewards, terminals, post_states = experience.sample()

                with tf.GradientTape() as tape:
                    q = policy_network.call(pre_states, training=True)
                    one_hot_action = tf.one_hot(action_values, n_actions, 1.0, 0.0)
                    current_q = tf.reduce_sum(tf.multiply(q, one_hot_action), axis=1)

                    target = target_network.compute_target(post_states, rewards, terminals)
                    loss = huber_loss_mean(target, current_q)

                grads = tape.gradient(loss, policy_network.variables)
                optimiser.apply_gradients(zip(grads, policy_network.variables))
                loss_list.append(loss)

                if len(loss_list) % 1000 == 0 and len(loss_list) > 0:
                    print("Current parameter update step:", len(loss_list))

                if len(loss_list) % 10000 == 0 and len(loss_list) > 0:
                    print("Update target network")
                    target_network.set_weights(policy_network.get_weights())

        print('Average Score per episode:', score/ep_counts)
        score_list.append(score/ep_counts)

    plot_loss(score_list)

    # after training
    env = gym.wrappers.Monitor(env, './', force=True)
    state = env.reset()
    is_done = False
    stacked_frames = []
    processed_state = preprocess(state)
    stacked_frames.append(processed_state)
    trained_eps = 0.05
    score = 0.0

    while not is_done:
        env.render()
        if len(stacked_frames) >= history_len:
            # get next states
            current_idx = len(stacked_frames)
            next_states = experience.get_states_in_training(current_idx, stacked_frames)
        else:
            next_states = experience.get_initial_state(state)

        q_values = policy_network.predict(next_states)
        q_values = np.squeeze(q_values)
        action = epsilon_greedy_policy(trained_eps, actions, q_values)

        next_s, r, is_done, _ = env.step(action)
        processed_next_s = preprocess(next_s)
        stacked_frames.append(processed_next_s)
        score += r

    env.close()
    print('Final Score:', score)


if __name__ == '__main__':
    main()
