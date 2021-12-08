from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import lbc_husky
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
import os
import math
import time

from raisimGymTorch.algo.sac.replay_memory import ReplayMemory
from raisimGymTorch.algo.sac.sac import SAC

import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse
import wandb

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Deterministic",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')

args = parser.parse_args()

# task specification
task_name = "husky_navigation"

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
env = VecEnv(lbc_husky.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs # 37
act_dim = env.num_acts # 4

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_iteration_steps = n_steps * env.num_envs
start = 0
total_steps = start

avg_rewards = []

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])

action_space = None
agent = SAC(ob_dim, action_space, args)
memory = ReplayMemory(args.replay_size, args.seed)

updates = 0

# wandb.init(project=task_name)

for update in range(start, 1000000):
    start = time.time()
    env.reset()
    reward_sum = 0
    done_sum = 0
    completed_sum = 0
    average_dones = 0.

    if update % cfg['environment']['eval_every_n'] == 0: # eval_every_n : 200
        print("Visualizing and evaluating the current policy")
        agent.save_checkpoint("raisim", suffix = "", ckpt_path = saver.data_dir+"/full_"+str(update)+'.pt')

        env.turn_on_visualization()
        env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

        # collect data for visualization
        for step in range(n_steps):
            frame_start = time.time()
            obs = env.observe(False)
            action = agent.select_action(torch.from_numpy(obs).cpu())
            reward, dones, completed = env.step(action)
            frame_end = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
            if wait_time > 0.:
                time.sleep(wait_time)

        env.stop_video_recording()
        env.turn_off_visualization()

        env.reset()
        env.save_scaling(saver.data_dir, str(update))

    # actual training
    for step in range(n_steps):
        obs = env.observe()
        action = agent.select_action(obs)

        if len(memory) > args.batch_size:
            for i in range(args.updates_per_step):
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                updates += 1

        reward, dones, not_completed = env.step(action)

        next_obs = env.observe()

        for i in range(len(reward)):
            mask = 1 if step == n_steps-1 else float(not dones[i])
            memory.push(obs[i], action[i], reward[i], next_obs[i], mask)  # Append transition to memory

        done_sum = done_sum + sum(dones)
        reward_sum = reward_sum + sum(reward)
        completed_sum = completed_sum + sum(not_completed)

    # data constraints - DO NOT CHANGE THIS BLOCK
    total_steps += env.num_envs * n_steps
    if total_steps > 20000000:
        break

    average_performance = reward_sum / total_iteration_steps
    average_dones = done_sum / total_iteration_steps
    avg_rewards.append(average_performance)

    # curriculum update. Implement it in Environment.hpp
    env.curriculum_callback()

    end = time.time()

    # log_dict = dict()
    # log_dict['Completion Time'] = completed_sum / env.num_envs * cfg['environment']['control_dt']
    # log_dict['Number of Dones'] = done_sum
    # wandb.log(log_dict)

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("avg reward: ", '{:0.10f}'.format(average_performance)))
    print('{:<40} {:>6}'.format("avg completion time: ", '{:0.6f}'.format(completed_sum / env.num_envs * cfg['environment']['control_dt'])))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_iteration_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_iteration_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('std: ')
    # print(np.exp(actor.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')