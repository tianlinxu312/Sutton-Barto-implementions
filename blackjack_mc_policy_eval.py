import numpy as np
import gym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(123)

# actions = (hit = 1, stick = 0)

# observation space = (the player's current sum, the dealer's one showing card,
# whether or not the player holds a usable ace)

# rewards = (+1, 0, -1)
# 200 states


def plot_v(values, ace=True):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca(projection='3d')

    x = np.arange(12, 22)
    y = np.arange(1, 11)

    X, Y = np.meshgrid(y, x)

    Z = values.reshape(X.shape)

    ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, linewidth=1, rstride=1, cstride=1)
    if ace:
        ax.set_title("With Ace")
    else:
        ax.set_title("Without Ace")
    ax.set_xlabel("Dealer Showing")
    ax.set_ylabel("Player Hand")
    ax.set_zlabel("State Value")
    plt.show()


def policy(hand_sum):
    if hand_sum > 20:
        return 0
    else:
        return 1


def mc_policy_evaluation(state_count, r, value):
    return value + (r - value) / state_count


def main():
    env = gym.make('Blackjack-v0')
    values_usable_ace = np.zeros((10, 10))
    values_no_usable_ace = np.zeros_like(values_usable_ace)
    # for every visit MC method
    state_count_ace = np.zeros_like(values_usable_ace)
    state_count_no_ace = np.zeros_like(state_count_ace)
    episodes = 10000

    for e in range(episodes):
        done = False
        # obs = the players current sum, the dealer's one showing card (1-10 where 1 is ace),
        # and whether or not the player holds a usable ace (0 or 1)
        obs = env.reset()
        state_history = []
        g = []

        if obs[0] == 11:
            done = True

        state_history.append(obs)

        while not done:
            a = policy(obs[0])
            obs, r, done, info = env.step(a)
            g.append(r)
            if done:
                break
            state_history.append(obs)

        final_reward = sum(g)

        for player_idx, dealer_idx, ace in state_history:
            player_idx -= 12
            dealer_idx -= 1

            if ace:
                state_count_ace[player_idx, dealer_idx] += 1.0
                values_usable_ace[player_idx, dealer_idx] = mc_policy_evaluation(state_count_ace[player_idx, dealer_idx],
                                                                                 final_reward,
                                                                                 values_usable_ace[player_idx, dealer_idx])
            else:
                state_count_no_ace[player_idx, dealer_idx] += 1.0
                values_no_usable_ace[player_idx, dealer_idx] = mc_policy_evaluation(state_count_no_ace[player_idx, dealer_idx],
                                                                                    final_reward,
                                                                                    values_no_usable_ace[player_idx, dealer_idx])

    plot_v(values_usable_ace)
    plot_v(values_no_usable_ace, ace=False)


if __name__ == '__main__':
    main()
