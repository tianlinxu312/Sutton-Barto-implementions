import numpy as np
import matplotlib.pyplot as plt

world_height = 4
world_width = 12

up = 0
left = 1
right = 2
down = 3
actions = [up, left, right, down]


class CliffWalking:
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.state = self.initial_state
        self.reward = 0.0
        self.is_terminal = False

    def is_cliff(self):
        cliff = np.zeros((world_height, world_width), dtype=np.bool)
        cliff[3, 1: -1] = True
        return cliff[tuple(self.state)]

    # return next_state, reward, done
    def step(self, action):
        i, j = self.state

        if action == up:
            self.state = [max(i - 1, 0), j]
        elif action == left:
            self.state = [i, max(j - 1, 0)]
        elif action == right:
            self.state = [i, min(j + 1, world_width - 1)]
        elif action == down:
            self.state = [min(i + 1, world_height - 1), j]
        else:
            assert False, "Actions should be in the range of (0, 4)"

        if self.is_cliff():
            self.state = self.initial_state
            self.reward = -100.0
            self.is_terminal = False
        elif self.state == self.goal_state:
            self.state = self.state
            self.reward = 0.0
            self.is_terminal = True
        else:
            self.reward = -1.0
            self.is_terminal = False
        return self.state, self.reward, self.is_terminal

    def reset(self):
        self.state = self.initial_state
        self.reward = 0.0
        self.is_terminal = False
        return self.state


def eps_greedy_policy(qsa, epsilon=0.1):
    if np.random.binomial(1, epsilon) == 1:
        return np.random.choice(actions)
    else:
        return np.random.choice([action_ for action_, value_ in enumerate(qsa) if value_ == np.max(qsa)])


def sarsa(qsa, next_qsa, r, alpha=0.1, gamma=1.0):
    return qsa + alpha * (r + gamma * next_qsa - qsa)


# Note that you need to pass all the state-action pairs of the next state
def q_learning(qsa, next_qs, r, alpha=0.1, gamma=1.0):
    return qsa + alpha * (r + gamma * np.max(next_qs) - qsa)


# Need to rewrite this function
def print_optimal_policy(q, goal, method='Sarsa'):
    optimal_policy = np.empty((world_height, world_width), dtype=str)
    for i in range(world_height):
        for j in range(world_width):
            if [i, j] == goal:
                optimal_policy[i, j] = 'G'
                continue
            # if is_cliff([i, j]):
            #   optimal_policy[i, j] = 'X'
            #   continue
            a = np.argmax(q[i, j, :])
            if a == up:
                optimal_policy[i, j] = 'U'
            elif a == left:
                optimal_policy[i, j] = 'L'
            elif a == right:
                optimal_policy[i, j] = 'R'
            elif a == down:
                optimal_policy[i, j] = 'D'
    if method == 'Sarsa':
        print('Sarsa optimal policy is:')
    else:
        print('Q-learning optimal policy is:')
    for row in optimal_policy:
        print(row)


def plot_rewards(r_sarsa, r_qlearning):
    plt.figure()
    plt.plot(r_sarsa, label='Sarsa')
    plt.plot(r_qlearning, label='Q-learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episodes')
    plt.ylim([-100, 0])
    plt.legend()
    plt.show()


def main():
    start_position = [3, 0]
    goal = [3, 11]

    runs = 50
    episodes = 500
    rewards_sarsa = np.zeros(episodes)
    rewards_qlearning = np.zeros_like(rewards_sarsa)

    # Create two instances of the environments for comparison
    env_sarsa = CliffWalking(start_position, goal)
    env_q_learning = CliffWalking(start_position, goal)

    for r in range(runs):
        q_sarsa = np.zeros((world_height, world_width, len(actions)))
        q_qlearning = np.zeros_like(q_sarsa)
        for i in range(episodes):
            state_sarsa = env_sarsa.reset()
            state_q = env_q_learning.reset()
            done_sarsa = False
            done_q = False

            # choose an action based on current state
            row, col = state_sarsa
            a_sarsa = eps_greedy_policy(q_sarsa[row, col, :])
            g_sarsa = 0.0
            g_q = 0.0

            while not done_sarsa:
                next_state_sarsa, r_sarsa, done_sarsa = env_sarsa.step(a_sarsa)

                # choose an action for the next state
                row, col = state_sarsa
                n_row, n_col = next_state_sarsa
                next_a_sarsa = eps_greedy_policy(q_sarsa[n_row, n_col, :])
                g_sarsa += r_sarsa
                # sarsa updates
                q_sarsa[row, col, a_sarsa] = sarsa(q_sarsa[row, col, a_sarsa], q_sarsa[n_row, n_col, next_a_sarsa], r_sarsa)

                state_sarsa = next_state_sarsa
                a_sarsa = next_a_sarsa

            while not done_q:
                row_q, col_q = state_q
                a_q = eps_greedy_policy(q_qlearning[row_q, col_q, :])
                next_state_q, r_q, done_q = env_q_learning.step(a_q)
                g_q += r_q

                # Q-learning updates, note the second argument
                n_row_q, n_col_q = next_state_q
                q_qlearning[row_q, col_q, a_q] = q_learning(q_qlearning[row_q, col_q, a_q],
                                                            q_qlearning[n_row_q, n_col_q, :], r_q)
                state_q = next_state_q

            rewards_sarsa[i] += g_sarsa
            rewards_qlearning[i] += g_q

    print_optimal_policy(q_sarsa, goal)
    print_optimal_policy(q_qlearning, goal, method='Q-learning')

    rewards_sarsa /= runs
    rewards_qlearning /= runs
    plot_rewards(rewards_sarsa, rewards_qlearning)


if __name__ == '__main__':
    main()
