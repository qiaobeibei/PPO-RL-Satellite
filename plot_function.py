import matplotlib.pyplot as plt

def plot_train_reward(epsiode_rewards, epsiode_mean_rewards):
    """
    绘制训练过程智能体的奖励图
    :param epsiode_rewards: 训练过程智能体获得的全部奖励回报
    :param epsiode_mean_rewards: 训练过程智能体获得的平均奖励回报
    """
    plt.plot(epsiode_rewards)
    plt.plot(epsiode_mean_rewards)
    plt.xlabel("epsiodes")
    plt.ylabel("rewards")
    plt.title("Continuous PPO With Optimization")
    plt.legend(["rewards", "mean_rewards"])
    plt.show()

def plot_trajectory(pursuer_position, escaper_position):
    """

    :param pursuer_position: 追捕者轨迹集
    :param escaper_position: 逃逸者轨迹集
    :return:
    """
    fig1 = plt.figure(1)
    ax = fig1.add_subplot(111, projection='3d')

    ax.plot([pos[0] for pos in pursuer_position], [pos[1] for pos in pursuer_position],
            [pos[2] for pos in pursuer_position], '-o', color='r', markersize=1)
    ax.plot([pos[0] for pos in escaper_position], [pos[1] for pos in escaper_position],
            [pos[2] for pos in escaper_position], '-o', color='b', markersize=1)
    ax.text(pursuer_position[0][0], pursuer_position[0][1], pursuer_position[0][2], 'P1',
            fontsize=10)
    ax.text(escaper_position[0][0], escaper_position[0][1], escaper_position[0][2], 'E',
            fontsize=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend(["pursuer", "evader"])
    plt.title('Relative PE Trajectories')
    plt.grid(True)
    plt.show()