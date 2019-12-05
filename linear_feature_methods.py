import gym
import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Approximator:
    def __init__(self, obs, dim, n_a, alpha, discount):
        self.__discount = discount
        self.__alpha = alpha
        self.__n_actions = n_a
        self.__obs_samples = obs
        self.__feature_dim = dim
        self.__w_size = 4 * self.__feature_dim
        self.__w = np.zeros((self.__n_actions, self.__w_size))
        self.__scaler = None
        self.__featuriser = None
        self.__initialised = False

    def initialise_scaler_featuriser(self):
        self.__scaler = sklearn.preprocessing.StandardScaler().fit(self.__obs_samples)
        self.__featuriser = sklearn.pipeline.FeatureUnion(
                [("rbf1", RBFSampler(gamma=5.0, n_components=self.__feature_dim)),
                 ("rbf2", RBFSampler(gamma=2.0, n_components=self.__feature_dim)),
                 ("rbf3", RBFSampler(gamma=1.0, n_components=self.__feature_dim)),
                 ("rbf4", RBFSampler(gamma=0.5, n_components=self.__feature_dim))])
        self.__featuriser.fit(self.__scaler.transform(self.__obs_samples))
        self.__initialised = True

    @property
    def get_w(self):
        return self.__w

    def feature_transformation(self, state):
        if not self.__initialised:
            self.initialise_scaler_featuriser()

        scaled = self.__scaler.transform([state])
        features = self.__featuriser.transform(scaled)
        return features

    # linear_features
    def action_value_estimator(self, features, a):
        return np.inner(features, self.__w[a])

    # minimising MSE between q(replaced by td target) and q_hat
    def update_w(self, r, q, next_q, features, a):
        target = r + self.__discount * next_q
        td_error = target - q
        w_gradient = self.__alpha * td_error * features
        self.__w[a] = self.__w[a] + w_gradient

    def cost_to_go(self, state):
        features = self.feature_transformation(state)
        v_s = []
        for i in range(self.__n_actions):
            v_s.append(self.action_value_estimator(features, i))
        return - np.max(v_s)


def epsilon_greedy_policy(epsilon, actions, values):
    if np.random.binomial(1, epsilon) == 1:
        return np.random.choice(actions)
    else:
        return np.random.choice([action_ for action_, value_ in enumerate(values) if value_ == np.max(values)])


def plot_v(estimator):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca(projection='3d')

    x = np.linspace(-1.2, 0.6, 30)
    y = np.linspace(-0.07, 0.07, 30)

    X, Y = np.meshgrid(x, y)
    states = np.dstack((X, Y))

    values = np.apply_along_axis(estimator.cost_to_go, 2, states)

    ax.plot_surface(X, Y, values, cmap=plt.cm.coolwarm, linewidth=1, rstride=1, cstride=1)

    ax.set_title("Cost to go function")
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("Value")
    plt.show()


def main():
    env = gym.envs.make('MountainCar-v0')
    n_actions = env.action_space.n
    actions = np.arange(n_actions)

    # create obs samples
    sample_size = 10000
    obs_samples = np.array([env.observation_space.sample() for i in range(sample_size)])

    alpha = 0.1
    gamma = 1.0
    epsilon = 0.1
    dim = 100
    estimator = Approximator(obs_samples, dim, n_actions, alpha, gamma)
    estimator.initialise_scaler_featuriser()
    # plot_v(estimator)

    episodes = 9000

    for i in range(1, episodes + 1):
        state = env.reset()
        a = env.action_space.sample()

        while True:
            next_state, r, done, _ = env.step(a)  # check the openAI github repo

            if done:
                break

            # compute q_sa
            features = estimator.feature_transformation(state)
            q_sa = estimator.action_value_estimator(features, a)

            # compute all actions in the next state for optimal policy
            next_feature = estimator.feature_transformation(next_state)
            q_values = []
            for j in actions:
                q_values.append(estimator.action_value_estimator(next_feature, j))

            next_a = epsilon_greedy_policy(epsilon, actions, q_values)
            next_q_sa = estimator.action_value_estimator(next_feature, next_a)

            # update weights for current action
            estimator.update_w(r, q_sa, next_q_sa, features, a)

            a = next_a
            state = next_state

    plot_v(estimator)

    # use trained approximator to solve the mountain car problem
    env = gym.wrappers.Monitor(env, './', force=True)
    env.reset()
    action = env.action_space.sample()
    while True:
        env.render()
        s, r, done, _ = env.step(action)
        if done:
            break

        feature = estimator.feature_transformation(s)
        q_values = []
        for j in actions:
            q_values.append(estimator.action_value_estimator(feature, j))

        action = epsilon_greedy_policy(epsilon, actions, q_values)


if __name__ == '__main__':
    main()

