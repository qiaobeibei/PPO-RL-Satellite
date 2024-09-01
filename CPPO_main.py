# --coding:utf-8--
import matplotlib.pyplot as plt
import numpy as np
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous import PPO_continuous
from environment import satellites
from tqdm import tqdm
import plot_function as pf


# 先定义一个参数类，用来储存超参数
class args_param(object):
    def __init__(self, max_train_steps=int(3e6),
                 evaluate_freq=5e3,
                 save_freq=20,
                 policy_dist="Gaussian",
                 batch_size=2048,
                 mini_batch_size=64,
                 hidden_width=256,
                 hidden_width2 =128,
                 lr_a=0.0002,
                 lr_c=0.0002,
                 gamma=0.99,
                 lamda=0.95,
                 epsilon=0.1,
                 K_epochs=10,
                 max_episode_steps=1000,
                 use_adv_norm=True,
                 use_state_norm=True,
                 use_reward_norm=False,
                 use_reward_scaling=True,
                 entropy_coef=0.01,
                 use_lr_decay=True,
                 use_grad_clip=True,
                 use_orthogonal_init=True,
                 set_adam_eps=True,
                 use_tanh=True,
                 chkpt_dir="/mnt/datab/home/yuanwenzheng/PICTURE1"):
        self.max_train_steps = max_train_steps
        self.evaluate_freq = evaluate_freq
        self.save_freq = save_freq
        self.policy_dist = policy_dist
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.hidden_width = hidden_width
        self.hidden_width2 = hidden_width2
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.gamma = gamma
        self.lamda = lamda
        self.epsilon = epsilon
        self.K_epochs = K_epochs
        self.use_adv_norm = use_adv_norm
        self.use_state_norm = use_state_norm
        self.use_reward_norm = use_reward_norm
        self.use_reward_scaling = use_reward_scaling
        self.entropy_coef = entropy_coef
        self.use_lr_decay = use_lr_decay
        self.use_grad_clip = use_grad_clip
        self.use_orthogonal_init = use_orthogonal_init
        self.set_adam_eps = set_adam_eps
        self.use_tanh = use_tanh
        self.max_episode_steps = max_episode_steps
        self.chkpt_dir = chkpt_dir

    def print_information(self):
        print("Maximum number of training steps:", self.max_train_steps)
        print("Evaluate the policy every 'evaluate_freq' steps:", self.evaluate_freq)
        print("Save frequency:", self.save_freq)
        print("Beta or Gaussian:", self.policy_dist)
        print("Batch size:", self.batch_size)
        print("Minibatch size:", self.mini_batch_size)
        print("The number of neurons in hidden layers of the neural network:", self.hidden_width)
        print("Learning rate of actor:", self.lr_a)
        print("Learning rate of critic:", self.lr_c)
        print("Discount factor:", self.gamma)
        print("GAE parameter:", self.lamda)
        print("PPO clip parameter:", self.epsilon)
        print("PPO parameter:", self.K_epochs)
        print("Trick 1:advantage normalization:", self.use_adv_norm)
        print("Trick 2:state normalization:", self.use_state_norm)
        print("Trick 3:reward normalization:", self.use_reward_norm)
        print("Trick 4:reward scaling:", self.use_reward_scaling)
        print("Trick 5: policy entropy:", self.entropy_coef)
        print("Trick 6:learning rate Decay:", self.use_lr_decay)
        print("Trick 7: Gradient clip:", self.use_grad_clip)
        print("Trick 8: orthogonal initialization:", self.use_orthogonal_init)
        print("Trick 9: set Adam epsilon=1e-5:", self.set_adam_eps)
        print("Trick 10: tanh activation function:", self.use_tanh)


# 下面函数用来训练网络
def train_pursuer_network(args, env, show_picture=True, pre_train=False, d_capture=0):
    epsiode_rewards = []
    epsiode_mean_rewards = []
    # 下面将导入env环境参数
    env.d_capture = d_capture
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space[0][1])
    # 下面将定义一个缓冲区
    replay_buffer = ReplayBuffer(args)
    # 下面将定义PPO智能体类
    pursuer_agent = PPO_continuous(args, 'pursuer')
    evader_agent = PPO_continuous(args, 'evader')
    if pre_train:
        pursuer_agent.load_checkpoint()
    # 下面开始进行训练过程
    pbar = tqdm(range(args.max_train_steps), desc="Training of pursuer", unit="episode")
    for epsiode in pbar:
        # 每个回合首先对值进行初始化
        epsiode_reward = 0.0
        done = False
        epsiode_count = 0
        # 再赋予一个新的初始状态
        s = env.reset(0)
        # 设置一个死循环，后面若跳出便在死循环中跳出
        while True:
            # 每执行一个回合，count次数加1
            epsiode_count += 1
            puruser_a, puruser_a_logprob = pursuer_agent.choose_action(s)
            evader_a, evader_a_logprob = evader_agent.choose_action(s)
            # 根据参数的不同选择输出是高斯分布/Beta分布调整
            if args.policy_dist == "Beta":
                puruser_action = 2 * (puruser_a - 0.5) * args.max_action
                evader_action = 2 * (evader_a - 0.5) * args.max_action
            else:
                puruser_action = puruser_a
                evader_action = evader_a
            # 下面是执行环境交互操作
            s_, r, done = env.step(puruser_action, evader_action, epsiode_count)  ## !!! 这里的环境是自己搭建的，输出每个人都不一样
            epsiode_reward += r

            # 下面考虑回合的最大运行次数(只要回合结束或者超过最大回合运行次数)
            if done or epsiode_count >= args.max_episode_steps:
                dw = True
            else:
                dw = False
            # 将经验存入replayBuffer中
            replay_buffer.store(s, puruser_action, puruser_a_logprob, r, s_, dw, done)
            # 重新赋值状态
            s = s_
            # 当replaybuffer尺寸到达batchsize便会开始训练
            if replay_buffer.count == args.batch_size:
                pursuer_agent.update(replay_buffer, epsiode)
                replay_buffer.count = 0
            # 如果回合结束便退出
            if done:
                epsiode_rewards.append(epsiode_reward)
                epsiode_mean_rewards.append(np.mean(epsiode_rewards))
                pbar.set_postfix({'回合': f'{epsiode}', '奖励': f'{epsiode_reward:.1f}', '平均奖励': f'{epsiode_mean_rewards[-1]:.1f}'})
                break

    # 存储训练模型
    pursuer_agent.save_checkpoint()
    # 如果需要画图的话
    if show_picture:
        pf.plot_train_reward(epsiode_rewards, epsiode_mean_rewards)

    return pursuer_agent

def train_evader_network(args, env, show_picture=True, pre_train=False, d_capture=0):
    epsiode_rewards = []
    epsiode_mean_rewards = []
    # 下面将导入env环境参数
    env.d_capture = d_capture
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space[0][1])
    # 下面将定义一个缓冲区
    replay_buffer = ReplayBuffer(args)
    # 下面将定义PPO智能体类
    pursuer_agent = PPO_continuous(args, 'pursuer')
    evader_agent = PPO_continuous(args, 'evader')
    if pre_train:
        evader_agent.load_checkpoint()
    # 下面开始进行训练过程
    pbar = tqdm(range(args.max_train_steps), desc="Training of evader", unit="episode")
    for epsiode in pbar:
        # 每个回合首先对值进行初始化
        epsiode_reward = 0.0
        done = False
        epsiode_count = 0

        s = env.reset(1)

        while True:
            # 每执行一个回合，count次数加1
            epsiode_count += 1
            puruser_a, puruser_a_logprob = pursuer_agent.choose_action(s)
            evader_a, evader_a_logprob = evader_agent.choose_action(s)
            # 根据参数的不同选择输出是高斯分布/Beta分布调整
            if args.policy_dist == "Beta":
                puruser_action = 2 * (puruser_a - 0.5) * args.max_action
                evader_action = 2 * (evader_a - 0.5) * args.max_action
            else:
                puruser_action = puruser_a
                evader_action = evader_a
            # 下面是执行环境交互操作
            s_, r, done = env.step(puruser_action, evader_action, epsiode_count)
            epsiode_reward += r

            # 下面考虑回合的最大运行次数(只要回合结束或者超过最大回合运行次数)
            if done or epsiode_count >= args.max_episode_steps:
                dw = True
            else:
                dw = False
            # 将经验存入replayBuffer中
            replay_buffer.store(s, evader_action, evader_a_logprob, r, s_, dw, done)
            # 重新赋值状态
            s = s_
            # 当replaybuffer尺寸到达batchsize便会开始训练
            if replay_buffer.count == args.batch_size:
                pursuer_agent.update(replay_buffer, epsiode)
                replay_buffer.count = 0
            # 如果回合结束便退出
            if done:
                epsiode_rewards.append(epsiode_reward)
                epsiode_mean_rewards.append(np.mean(epsiode_rewards))
                pbar.set_postfix({'回合': f'{epsiode}', '奖励': f'{epsiode_reward:.1f}', '平均奖励': f'{epsiode_mean_rewards[-1]:.1f}'})
                break

    # 存储训练模型
    evader_agent.save_checkpoint()
    # 如果需要画图的话
    if show_picture:
        pf.plot_train_reward(epsiode_rewards, epsiode_mean_rewards)

    return evader_agent

# 下面将用训练好的网络跑一次例程
def test_network(args, env, show_pictures=True, d_capture=0):
    epsiode_reward = 0.0
    done = False
    epsiode_count = 0
    pursuer_position = []
    escaper_position = []
    pursuer_velocity = []
    escaper_velocity = []

    env.d_capture = d_capture
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space[0][1])
    # 下面将定义PPO智能体类
    pursuer_agent = PPO_continuous(args, 'pursuer')
    evader_agent = PPO_continuous(args, 'evader')
    pursuer_agent.load_checkpoint()

    s = env.reset(0)
    while True:
        epsiode_count += 1
        puruser_a, puruser_a_logprob = pursuer_agent.choose_action(s)
        evader_a, evader_a_logprob = evader_agent.choose_action(s)
        if args.policy_dist == "Beta":
            puruser_action = 2 * (puruser_a - 0.5) * args.max_action
            evader_action = 2 * (evader_a - 0.5) * args.max_action
        else:
            puruser_action = puruser_a
            evader_action = evader_a
        # 下面是执行环境交互操作
        s_, r, done = env.step(puruser_action, evader_action, epsiode_count, )  ## !!! 这里的环境是自己搭建的，输出每个人都不一样
        epsiode_reward += r
        pursuer_position.append(s_[6:9])
        pursuer_velocity.append(s_[9:12])
        escaper_position.append(s_[12:15])
        escaper_velocity.append(s_[15:18])

        # 下面考虑回合的最大运行次数(只要回合结束或者超过最大回合运行次数)
        if done or epsiode_count >= args.max_episode_steps:
            dw = True
        else:
            dw = False

        s = s_
        if done :
            print("当前测试得分为{}".format(epsiode_reward))
            break
    # 下面开始画图
    if show_pictures:
        pf.plot_trajectory(pursuer_position, escaper_position)

def train_elliptical_network(args, env, epsiodes=200, d_capture=0):
    epsiode_reward = 0.0
    done = False
    epsiode_count = 0
    env.d_capture = d_capture
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space[0][1])
    # 下面将定义PPO智能体类
    pursuer_agent = PPO_continuous(args, 'pursuer')
    evader_agent = PPO_continuous(args, 'evader')
    pursuer_agent.load_checkpoint()

    for epsiode in range(epsiodes):
        s = env.reset(2)

        while True:
            epsiode_count += 1
            puruser_a, puruser_a_logprob = pursuer_agent.choose_action(s)
            evader_a, evader_a_logprob = evader_agent.choose_action(s)
            if args.policy_dist == "Beta":
                puruser_action = 2 * (puruser_a - 0.5) * args.max_action
                evader_action = 2 * (evader_a - 0.5) * args.max_action
            else:
                puruser_action = puruser_a
                evader_action = evader_a
            # 下面是执行环境交互操作
            s_, r, done = env.step(puruser_action, evader_action, epsiode_count, )  ## !!! 这里的环境是自己搭建的，输出每个人都不一样


            # 下面考虑回合的最大运行次数(只要回合结束或者超过最大回合运行次数)
            if done or epsiode_count >= args.max_episode_steps:
                dw = True
            else:
                dw = False

            s = s_
            if done :
                break


if __name__ == "__main__":
    # 声明环境
    args = args_param(max_episode_steps=64, batch_size=64, max_train_steps=5000, K_epochs=3,
                      chkpt_dir="D:\项目\可达域\ppo_flight-0704\ppo_flight\model_file\one_layer")
    # 声明参数
    env = satellites(Pursuer_position=np.array([2000000, 2000000 ,1000000]),
                     Pursuer_vector=np.array([1710, 1140, 1300]),
                     Escaper_position=np.array([1850000, 2000000, 1000000]),
                     Escaper_vector=np.array([1710, 1140, 1300]),
                     d_capture=50000,
                     args=args)
    Sign = 1
    if Sign == 0:
        pursuer_agent = train_pursuer_network(args, env, show_picture=True, pre_train=True, d_capture=15000)
        evader_agent = train_evader_network(args, env, show_picture=True, pre_train=False, d_capture=15000)
    elif Sign == 1:
        test_network(args, env, show_pictures=False, d_capture=20000)
    elif Sign == 2:
        train_elliptical_network(args, env, epsiodes=500, d_capture=0)

"""
1. 网络结构和代码优化? 已解决
2. 对奖励系数PSO寻优? 待定
3. 无干预情况下脉冲的自动寻优? 已解决
4. 对逃逸星的动作范围加以限制? 待定
5. 目标系下的坐标如何转换至惯性系下,危险区的求解? 已解决
6. 构建新的奖励函数激励网络输出脉冲的自动寻优? 已解决
7. 神经网络优化可达域计算速率？待定(CNN or fully_connected_layer)/曲面拟合，已解决
   只要知道大概的曲面方程便可通过神经网络近似求出可达域. 已解决,但只能做出来平面可达域曲线拟合
8. 其实可达域大可不必求出空间的交集，可以求出对应xoy平面的交集，因为可达域关于xoy平面对称，且只要xy面有交集，
    就代表空间也一定有交集。所以可通过俯视图的方式减少计算。而对于俯视图的可达域，其实可以将图像分解为两个椭圆，
    空心部分为一个椭圆，外部也为一个椭圆，两个椭圆的中心点不同。  (使用平面可达域的方式取代空间可达域的求解)已解决
"""