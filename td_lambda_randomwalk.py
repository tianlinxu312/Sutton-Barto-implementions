import numpy as np
import matplotlib.pyplot as plt

# actions
left = 0
right = 1


def random_policy():
    return np.random.binomial(1, 0.5)


class RandomWalk:
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.state = self.initial_state
        self.reward = 0.0
        self.is_terminal = False

    # write step function that returns obs(next state), reward, is_done
    def step(self, action):
        if self.state == 18 and action == right:
            self.state += 1
            self.is_terminal = True
            self.reward = 1.0
        elif self.state == 1 and action == left:
            self.state -= 1
            self.is_terminal = True
            self.reward = -1.0
        else:
            if action == left:
                self.state -= 1
                self.is_terminal = False
                self.reward = 0.0

            else:
                self.state += 1
                self.is_terminal = False
                self.reward = 0.0

        return self.state, self.reward, self.is_terminal

    def reset(self):
        self.state = self.initial_state
        self.reward = 0.0
        self.is_terminal = False
        return self.state


# Pass the reward trajectory from time step t up to n'th step
def n_step_reward(r, values, current_t, total_time, state_history, lam):
    n_steps = len(r)
    lambda_power = 1.0
    r_n = 0.0

    for n in range(n_steps):
        r_n += lambda_power * r[n]
        lambda_power *= lam

    end_time = min(current_t + n_steps, total_time)
    state = state_history[end_time]
    r_n += lambda_power * values[state]

    return r_n


def temporal_difference_lambda(values, state_history, r_trajectory, current_t, alpha=0.1, lam=1.0):
    lambda_r = 0.0
    lambda_power = 1.0
    total_time = len(state_history)
    total_n = total_time - current_t

    for n in range(1, max(total_n, 1)):
        lambda_r += lambda_power * n_step_reward(r_trajectory[current_t:current_t+n], values, current_t,
                                                 total_time, state_history, lam)
        lambda_power *= lam

    lambda_r = (1.0 - lam) * lambda_r + lambda_power * np.sum(r_trajectory[current_t:total_time])
    current_state = state_history[current_t]
    return values[current_state] + alpha * (lambda_r - values[current_state])


def root_mean_square_error(values, true_values):
    diff = values - true_values
    mse = np.mean(diff ** 2)
    return np.sqrt(mse)


def plot_rms_error(errors, alphas, lambdas):
    for i in range(len(lambdas)):
        plt.plot(alphas[i], errors[i, :], label=r'$\lambda$ = ' + str(lambdas[i]))
    plt.ylabel('RMS error averaged over states')
    plt.xlabel(r'$\alpha$')
    plt.title(r'Off-line $\lambda$-return')
    plt.ylim(0.25, 0.55)
    plt.legend()
    plt.show()


def main():
    true_values = np.arange(-20, 22, 2) / 20.0
    true_values[0] = true_values[20] = 0.0

    initial_state = 10
    env = RandomWalk(initial_state)

    episodes = 10
    n_runs = 10

    lambdas = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
    alphas = [np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 0.55, 0.05),
              np.arange(0, 0.22, 0.02),
              np.arange(0, 0.11, 0.01)]

    # arrays to store rms for different values of alphas
    rms = np.empty((n_runs, len(lambdas), len(alphas[0])))

    for n in range(n_runs):
        print("Number of current run:", n + 1)
        for lambda_idx, lam in zip(range(len(lambdas)), lambdas):
            for alpha_idx, alpha in zip(range(len(alphas[lambda_idx])), alphas[lambda_idx]):
                values = np.zeros(21)
                error = 0.0
                for i in range(1, episodes + 1):
                    state = env.reset()
                    state_history = [state]
                    done = False
                    reward_trajectory = []

                    while not done:
                        action = random_policy()
                        next_state, r, done = env.step(action)
                        reward_trajectory.append(r)
                        state_history.append(next_state)

                    for t, state in zip(range(len(state_history)), state_history):
                        values[state] = temporal_difference_lambda(values, state_history, reward_trajectory, t,
                                                                   alpha=alpha, lam=lam)

                    error += root_mean_square_error(values[1:20], true_values[1:20])

                rms[n, lambda_idx, alpha_idx] = error / episodes

    rms = np.mean(rms, axis=0)

    plot_rms_error(rms, alphas, lambdas)


if __name__ == '__main__':
    main()
