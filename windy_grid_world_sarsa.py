import numpy as np
import matplotlib.pyplot as plt

world_height = 7
world_width = 10

wind_strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

up = 0
left = 1
right = 2
down = 3
actions = [up, left, right, down]


class WindyGridworld:
    def __init__(self, init_position, goal_position):
        self.initial_state = init_position
        self.goal_state = goal_position
        self.state = self.initial_state
        self.reward = 0.0
        self.is_terminal = False

    # return (next state, reward, is_done)
    def step(self, action):
        i, j = self.state

        if self.state == self.goal_state:
            self.reward = 0.0
            self.is_terminal = True
        else:
            if action == up:
                self.state = [max(i - 1 - wind_strength[j], 0), j]
            elif action == left:
                # the next state (j-1) is the action + the wind in the previous state (j)
                self.state = [max(i - wind_strength[j], 0), max(j - 1, 0)]
            elif action == right:
                self.state = [max(i - wind_strength[j], 0), min(j + 1, world_width - 1)]
            elif action == down:
                self.state = [max(min(i + 1 - wind_strength[j], world_height - 1), 0), j]
            else:
                assert False, "Actions should be in the range of (0, 4)."
            self.reward = -1.0
            self.is_terminal = False

        return self.state, self.reward, self.is_terminal

    def reset(self):
        self.state = self.initial_state
        self.reward = 0.0
        self.is_terminal = False
        return self.state


def sarsa(qsa, next_qsa, r, alpha=0.1, gamma=1.0):
    return qsa + alpha * (r + gamma * next_qsa - qsa)


def epsilon_greedy_policy(q_values, epsilon=0.1):
    if np.random.binomial(1, epsilon) == 1:
        return np.random.choice(actions)
    else:
        return np.random.choice([action_ for action_, value_ in enumerate(q_values) if value_ == np.max(q_values)])


def plot_episode_steps(steps, episodes):
    plt.plot(steps, np.arange(1, episodes + 1))
    plt.title("Time steps vs Episodes")
    plt.xlabel("Time steps")
    plt.ylabel("Episodes")
    plt.show()


def print_optimal_policy(policy, terminal_position):
    policy_display = np.empty_like(policy, dtype=str)
    wind = np.empty_like(wind_strength, dtype=str)
    for i in range(0, world_height):
        for j in range(0, world_width):
            wind[j] = str(wind_strength[j])
            if [i, j] == terminal_position:
                policy_display[i, j] = 'G'
                continue
            a = policy[i, j]
            if a == up:
                policy_display[i, j] = 'U'
            elif a == left:
                policy_display[i, j] = 'L'
            elif a == right:
                policy_display[i, j] = 'R'
            elif a == down:
                policy_display[i, j] = 'D'
    print('Optimal policy is:')
    for row in policy_display:
        print(row)
    print(wind)


def main():
    q_sa = np.zeros((world_height, world_width, len(actions)))
    episodes = 1000
    timesteps = []

    start_position = [3, 0]
    terminal_position = [3, 7]

    env = WindyGridworld(start_position, terminal_position)

    for i in range(1, episodes + 1):
        state = env.reset()
        is_done = False
        row, col = state
        # initialise a
        a = epsilon_greedy_policy(q_sa[row, col, :])
        timesteps_per_epi = 1

        while not is_done:
            next_state, r, is_done = env.step(a)
            row, col = state
            n_row, n_col = next_state

            next_a = epsilon_greedy_policy(q_sa[n_row, n_col, :])
            q_sa[row, col, a] = sarsa(q_sa[row, col, a], q_sa[n_row, n_col, next_a], r)

            state = next_state
            a = next_a
            timesteps_per_epi += 1
        timesteps.append(timesteps_per_epi)

    optimal_policy = np.argmax(q_sa, axis=2)

    timesteps = np.add.accumulate(timesteps)
    plot_episode_steps(timesteps, episodes)
    print_optimal_policy(optimal_policy, terminal_position)


if __name__ == '__main__':
    main()

