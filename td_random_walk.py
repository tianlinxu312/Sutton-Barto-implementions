import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

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
        if self.state == 5 and action == right:
            self.state += 1
            self.is_terminal = True
            self.reward = 1.0
        elif self.state == 1 and action == left:
            self.state -= 1
            self.is_terminal = True
            self.reward = 0.0
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


def temporal_difference(value, next_value, r, alpha=0.1):
    return value + alpha * (r + next_value - value)


def constant_mc(value, g, alpha=0.01):
    return value + alpha * (g - value)


def root_mean_square_error(values, true_values):
    diff = values - true_values
    mse = np.mean(diff ** 2)
    return np.sqrt(mse)


def plot_state_value(history, true_values):
    plt.figure(1)
    x = [1, 2, 3, 4, 5]
    my_xticks = ['A', 'B', 'C', 'D', 'E']
    plt.xticks(x, my_xticks)
    episodes = [0, 1, 10, 100]
    for i in range(len(history)):
        if i in episodes:
            plt.plot(x, history[i][1:6], 'o-', label=str(i)+' episode(s)')
    plt.plot(x, true_values[1:6], label='true values')
    plt.xlabel('state')
    plt.ylabel('estimated value')
    plt.legend()
    plt.show()


def plot_rms_error(err1, err2, err3, err4, err5, err6):
    plt.plot(err1, 'r-', label=r'MC $\alpha$=0.01')
    plt.plot(err2, 'y-', label=r'MC $\alpha$=0.02')
    plt.plot(err3, 'c-', label=r'MC $\alpha$=0.03')
    plt.plot(err4, 'o-', label=r'TD $\alpha$=0.05')
    plt.plot(err5, 'o-', label=r'TD $\alpha$=0.10')
    plt.plot(err6, 'o-', label=r'TD $\alpha$=0.15')
    plt.ylabel('RMS error averaged over states')
    plt.xlabel('Episodes')
    plt.legend()
    plt.show()


def main():
    # 0 is the left terminal state
    # 6 is the right terminal state
    # 1 ... 5 represents A ... E
    values = np.zeros(7)
    # For convenience, we assume all rewards are 0
    # and the left terminal state has value 0, the right terminal state has value 1
    # This trick has been used in Gambler's Problem
    # values[1:6] = 0.5
    # values[6] = 1.0

    initial_state = 3
    env = RandomWalk(initial_state)

    episodes = 100

    # Uncomment to produce estimated value
    '''
    value_history = [np.copy(values)]
    
    for i in range(1, episodes + 1):
        state = env.reset()
        done = False

        while not done:
            a = random_policy()
            next_state, r, done = env.step(a)
            values[state] = temporal_difference(values[state], values[next_state], r)
            state = next_state

        value_history.append(np.copy(values))

    true_values = np.zeros(7)
    true_values[1:6] = np.arange(1, 6) / 6.0
    true_values[6] = 1.0
    plot_state_value(value_history, true_values)
    '''

    true_values = np.zeros(7)
    true_values[1:6] = np.arange(1, 6) / 6.0
    true_values[6] = 1.0

    mc_values1 = np.zeros(7)
    mc_values1[1:6] = 0.5
    mc_values2 = np.copy(mc_values1)
    mc_values3 = np.copy(mc_values2)

    err_mc1 = []
    err_mc2 = []
    err_mc3 = []

    values1 = np.zeros(7)
    values1[1:6] = 0.5
    values2 = np.copy(values1)
    values3 = np.copy(values2)

    err_td1 = []
    err_td2 = []
    err_td3 = []

    for i in range(1, episodes + 1):
        state = env.reset()
        state_history = [state]
        done = False
        g = 0.0

        while not done:
            a = random_policy()
            next_state, r, done = env.step(a)
            values1[state] = temporal_difference(values1[state], values1[next_state], r, alpha=0.05)
            values2[state] = temporal_difference(values2[state], values2[next_state], r, alpha=0.10)
            values3[state] = temporal_difference(values3[state], values3[next_state], r, alpha=0.15)
            g += r
            state = next_state
            state_history.append(state)

        for state in state_history:
            mc_values1[state] = constant_mc(mc_values1[state], g, alpha=0.01)
            mc_values2[state] = constant_mc(mc_values2[state], g, alpha=0.02)
            mc_values3[state] = constant_mc(mc_values3[state], g, alpha=0.03)

        # compute error per episode
        rms_mc1 = root_mean_square_error(mc_values1[1:6], true_values[1:6])
        rms_td1 = root_mean_square_error(values1[1:6], true_values[1:6])

        rms_mc2 = root_mean_square_error(mc_values2[1:6], true_values[1:6])
        rms_td2 = root_mean_square_error(values2[1:6], true_values[1:6])

        rms_mc3 = root_mean_square_error(mc_values3[1:6], true_values[1:6])
        rms_td3 = root_mean_square_error(values3[1:6], true_values[1:6])

        err_mc1.append(rms_mc1)
        err_mc2.append(rms_mc2)
        err_mc3.append(rms_mc3)
        err_td1.append(rms_td1)
        err_td2.append(rms_td2)
        err_td3.append(rms_td3)

    plot_rms_error(err_mc1, err_mc2, err_mc3, err_td1, err_td2, err_td3)


if __name__ == '__main__':
    main()
