import matplotlib.pyplot as plt
import numpy as np
import gym
import tensorflow as tf
tf.enable_eager_execution()


def preprocess(x, frame_height=80, frame_width=80):
    cropped_x = tf.image.crop_to_bounding_box(x, 35, 0, 165, 160)
    downsize_x = tf.image.resize_images(cropped_x, size=[frame_height, frame_width],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    gray_x = tf.image.rgb_to_grayscale(downsize_x)
    gray_x = tf.to_float(gray_x)
    return tf.reshape(gray_x, [-1, frame_width * frame_height])


def discount_rewards(r, gamma):
    r = np.array(r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    # we go from last reward to first one so we don't have to do exponentiations
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # if the game ended (in Pong), reset the reward sum
        running_add = running_add * gamma + r[t]  # the point here is to use Horner's method to compute those rewards efficiently
        discounted_r[t] = running_add
    discounted_r -= np.mean(discounted_r)  # normalizing the result
    discounted_r /= np.std(discounted_r)  # idem
    return discounted_r


class Policy(tf.keras.Model):
    def __init__(self, input_size):
        super(Policy, self).__init__()
        self.hidden = 200
        self.input_size = input_size

        self.dense1 = tf.keras.layers.Dense(self.hidden, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs, training=None, mask=None):
        h1 = self.dense1(inputs)
        logits = self.dense2(h1)
        prob = tf.sigmoid(logits)
        return tf.squeeze(logits), tf.squeeze(prob)


def main():
    update_frequency = 10
    learning_rate = 1e-3
    gamma = 0.99
    decay_rate = 0.99
    n_pixels = 6400
    n_episodes = 10000

    policy_net = Policy(n_pixels)
    optimiser = tf.train.RMSPropOptimizer(learning_rate, decay=decay_rate)

    env = gym.make("Pong-v0")
    loss_list = []
    mini_batch = []
    updates = 1
    total_rewards = 0.0

    for e in range(1, n_episodes+1):
        state = env.reset()
        prev_state = None
        is_done = False
        rewards = []
        labels = []
        inputs = []

        while not is_done:
            state = preprocess(state)
            x = state - prev_state if prev_state is not None else tf.zeros((1, n_pixels))
            prev_state = state

            _, prob = policy_net.call(x, training=False)
            action = 3 if np.random.uniform() < prob else 2
            y = 1.0 if action == 3 else 0.0 # fake labels

            state, r, is_done, _ = env.step(action)
            rewards.append(r)
            labels.append(y)
            inputs.append(x)

            if is_done:
                # compute discounted rewards and standardise to 0 mean and unit variance to control the variance
                total_rewards += np.sum(rewards)
                discount_epr = discount_rewards(rewards, gamma)
                mini_batch.append([inputs, labels, discount_epr])

                if e % update_frequency == 0:
                    print('Updates:', updates)
                    for (batch, (x, label, reward)) in enumerate(mini_batch):
                        with tf.GradientTape() as tape:
                            logits, _ = policy_net.call(x, training=True)
                            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits)
                            loss = tf.reduce_mean(tf.multiply(reward, cross_entropy))

                        grads = tape.gradient(loss, policy_net.variables)
                        optimiser.apply_gradients(zip(grads, policy_net.variables))
                        loss_list.append(loss)

                    updates += 1
                    print('Average reward per every 10 episodes:', total_rewards / update_frequency)
                    # reset mini_batch and total rewards for batch_size number of episodes
                    del mini_batch[:]
                    total_rewards = 0.0


if __name__ == '__main__':
    main()