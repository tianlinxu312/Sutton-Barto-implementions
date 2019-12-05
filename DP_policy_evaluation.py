import numpy as np

up = 0
right = 1
left = 2
down = 3

n_states = 16
n_actions = 4
max_row = 4
max_col = 4


def grid_world():
    p = {}
    grid = np.arange(n_states).reshape([max_row, max_col])
    it = np.nditer(grid, flags=['multi_index'])

    with it:
        while not it.finished:
            s = it.iterindex
            row, col = it.multi_index

            p[s] = {a: [] for a in range(n_actions)}

            is_done = lambda x: x == 0 or x == (n_states - 1)
            reward = 0.0 if is_done(s) else -1.0

            if is_done(s):
                # 4 variables: probability of ending up in the next state after action, next state, reward, done
                p[s][up] = [(1.0, s, reward, True)]
                p[s][right] = [(1.0, s, reward, True)]
                p[s][left] = [(1.0, s, reward, True)]
                p[s][down] = [(1.0, s, reward, True)]

            else:
                s_up = s if row == 0 else s - max_row
                s_right = s if col == (max_col - 1) else s + 1
                s_left = s if col == 0 else s - 1
                s_down = s if row == (max_row - 1) else s + max_row

                p[s][up] = [(1.0, s_up, reward, is_done(s_up))]
                p[s][right] = [(1.0, s_right, reward, is_done(s_right))]
                p[s][left] = [(1.0, s_left, reward, is_done(s_left))]
                p[s][down] = [(1.0, s_down, reward, is_done(s_down))]

            it.iternext()
    return p


def main():
    # Initial state values - 0s
    state_values = np.zeros(16)
    # Assume a random policy
    pi_a = 0.25
    lam = 1.0
    theta = 1e-10
    iteration_counter = 1

    transition_probs = grid_world()

    while True:
        v_old = np.copy(state_values)
        delta = 0.0
        for s in range(n_states):
            v_s = 0.0

            for a in range(n_actions):
                current_entry = transition_probs[s][a][0]
                p_sa = current_entry[0]
                next_s = current_entry[1]
                reward = current_entry[2]
                v_s += pi_a * p_sa * (reward + lam * v_old[next_s])

            state_values[s] = v_s

            # stopping criteria
            delta = np.maximum(delta, np.abs(state_values[s] - v_old[s]))
        print('After %s iteration(s):\n' % iteration_counter, np.reshape(state_values, [max_row, max_col]))
        iteration_counter += 1
        if delta < theta:
            break


if __name__ == '__main__':
    main()