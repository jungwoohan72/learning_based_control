from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import lbc_husky
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import raisimGymTorch.algo.ppo.module as ppo_module
import os
import math
import time
import torch
import argparse

import matplotlib.pyplot as plt

from raisimGymTorch.algo.sac.model import DeterministicPolicy

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

fig = plt.figure(1)
ax = fig.add_subplot(111)
plt.show(block = False)

for i in range(6000,6900,100):
    weight_path = home_path + "/raisimGymTorch/data/husky_navigation/2021-12-02-12-53-12-Curriculum1000/full_{}.pt".format(6500)

    # config
    cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

    # create environment from the configuration file
    cfg['environment']['num_envs'] = 200
    env = VecEnv(lbc_husky.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

    # shortcuts
    ob_dim = env.num_obs
    act_dim = env.num_acts

    iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
    weight_dir = weight_path.rsplit('/', 1)[0] + '/'

    if weight_path == "":
        print("Can't find trained weight, please provide a trained weight with --weight switch\n")
    else:
        print("Loaded weight from {}\n".format(weight_path))
        start = time.time()
        env.reset()
        completion_sum = 0
        done_sum = 0
        average_dones = 0.
        n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
        total_steps = n_steps * 1
        start_step_id = 0

        print("Visualizing and evaluating the policy: ", weight_path)
        loaded_graph = DeterministicPolicy(ob_dim, act_dim, 128, None)
        loaded_graph.load_state_dict(torch.load(weight_path)['policy_state_dict'])

        env.load_scaling(weight_dir, int(iteration_number))
        # env.turn_on_visualization()

        max_steps = 80 ## 8 secs
        completed_sum = 0

        for step in range(max_steps):
            time.sleep(cfg['environment']['control_dt'])
            obs = env.observe(False)
            action, _, _ = loaded_graph.sample(torch.FloatTensor(obs).unsqueeze(0))
            action = action.detach().cpu().numpy()[0]
            reward, dones, not_completed = env.step(action)
            completed_sum = completed_sum + sum(not_completed)

        print('----------------------------------------------------')
        print('{:<40} {:>6}'.format("avg completion time: ", '{:0.6f}'.format(completed_sum / env.num_envs * cfg['environment']['control_dt'])))
        print('----------------------------------------------------\n')

        plt.scatter(i, completed_sum / env.num_envs * cfg['environment']['control_dt'])
        plt.pause(0.001)
        # env.turn_off_visualization()