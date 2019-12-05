import numpy as np
import gym
import matplotlib.pyplot as plt
import matplotlib as mpl


def policy_evaluation(counts, qsa, g):
    return qsa + (g - qsa) / counts


def plot_policy(policy, ace=True):
    # Get colors
    cmap = plt.cm.get_cmap("Paired")
    colors = list([cmap(0.2), cmap(0.8)])
    label = ["Stick", "Hit"]

    # Plot results
    plt.figure(figsize=(15, 6))

    player_range = np.arange(12, 22)
    dealer_range = np.arange(1, 11)

    plt.pcolor(dealer_range, player_range, policy, label=label, cmap=mpl.colors.ListedColormap(colors))
    plt.axis([dealer_range.min(), dealer_range.max(), player_range.min(), player_range.max()])
    col_bar = plt.colorbar()
    col_bar.set_ticks([0.25, 0.75])
    col_bar.set_ticklabels(label)
    plt.grid()
    plt.xlabel("Dealer Showing")
    plt.ylabel("Player Score")
    if ace:
        plt.title("Optimal Policy With a Usable Ace ($\pi_*$)")
    else:
        plt.title("Optimal Policy Without a Usable Ace ($\pi_*$)")
    plt.show()


def main():
    env = gym.make('Blackjack-v0')
    player = 10
    dealer = 10
    n_action = 2
    qsa_with_ace = np.zeros([player, dealer, n_action])
    qsa_without_ace = np.zeros_like(qsa_with_ace)
    a_s_counts_ace = np.zeros_like(qsa_with_ace)
    a_s_counts_no_ace = np.zeros_like(a_s_counts_ace)

    # Initialise policy: stick if policies >= 20 (action = 0), else hit
    policy_with_ace = np.ones([player, dealer], dtype=np.int)
    policy_with_ace[7:, :] = 0
    policy_without_ace = np.ones([player, dealer], dtype=np.int)
    policy_without_ace[7:, :] = 0

    episodes = 500000

    for e in range(episodes):
        done = False
        obs = env.reset()
        state_action_history = []
        g = []
        # random first action
        a = env.action_space.sample()

        if obs[0] == 11:
            done = True

        else:
            state_action_history.append([obs[0], obs[1], obs[2], a])

        while not done:
            obs, r, done, info = env.step(a)
            g.append(r)
            if done:
                break
            current_player_idx = obs[0] - 12
            current_dealer_idx = obs[1] - 1
            if obs[2]:
                a = policy_with_ace[current_player_idx, current_dealer_idx]
            else:
                a = policy_without_ace[current_player_idx, current_dealer_idx]

            state_action_history.append([obs[0], obs[1], obs[2], a])

        final_reward = sum(g)
        for player_idx, dealer_idx, ace, action in state_action_history:
            player_idx -= 12
            dealer_idx -= 1

            if ace:
                a_s_counts_ace[player_idx, dealer_idx, action] += 1.0
                qsa_with_ace[player_idx, dealer_idx, action] = policy_evaluation(a_s_counts_ace[player_idx, dealer_idx, action],
                                                                            qsa_with_ace[player_idx, dealer_idx, action],
                                                                            final_reward)
                # improve policy
                policy_with_ace[player_idx, dealer_idx] = np.argmax(qsa_with_ace[player_idx, dealer_idx])

            else:
                a_s_counts_no_ace[player_idx, dealer_idx, action] += 1.0
                qsa_without_ace[player_idx, dealer_idx, action] = policy_evaluation(a_s_counts_no_ace[player_idx, dealer_idx, action],
                                                                               qsa_without_ace[player_idx, dealer_idx, action],
                                                                               final_reward)
                policy_without_ace[player_idx, dealer_idx] = np.argmax(qsa_without_ace[player_idx, dealer_idx])

    plot_policy(policy_with_ace)
    plot_policy(policy_without_ace, ace=False)


if __name__ == '__main__':
    main()

