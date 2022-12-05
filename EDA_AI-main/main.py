import sys
import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from place_env import place_envs
#from fullplace_env import fullplace_envs

"Guangxi"
from Train_Dataset import *



def main():
    """Prepare the data file"""
    #area_filename = "./InputDataSample/sample45_compact/45-1/placement_info.txt"
    #print(area_filename)
    #link_filename = "./InputDataSample/connect_file/connect_45.txt"
    #print(link_filename)
    #f=open('file_names.txt','r')
    #f_list=f.readlines()
    #f.close()
    args = get_args()
    area_filename=args.area_file
    link_filename=args.link_file
    print(area_filename)
    print(link_filename)
    num_of_macro, case_id = TrainingData(area_filename, link_filename)

    """Guangxi: Define some parameters"""
    build_graph_n = num_of_macro
    """grid_size_n Must be 32"""
    grid_size_n = 32
    """Useless"""
    #evaluate_actions_n = 49
    

    """Define command parameters"""
  
    """Guangxi"""
    args.num_steps = build_graph_n * args.num_mini_batch

    """Set the random seed"""
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    """Make empty directories named 'log_dir' and 'eval_log_dir'"""
    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    """Set # of cores and device type"""
    torch.set_num_threads(10)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    """Define the environment"""
    if args.task == 'place':
        """Guangxi: 'obs_space must be 84 due to the network architecture"""
        envs = place_envs(grid_size = grid_size_n, num_cell = build_graph_n, obs_space = 84, case_id=case_id)

 #   elif args.task == 'fullplace':
 #      envs = fullplace_envs()

    """Guangxi"""
    actor_critic = Policy(
        envs.obs_space,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy},
        #n_in_evaluate_actions = evaluate_actions_n,
        n_in_build_graph_Policy = build_graph_n)

    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.obs_space) == 1
        discr = gail.Discriminator(
            envs.obs_space[0] + envs.action_space[1], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.obs_space, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(envs.transform(obs))
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    """Guangxi"""
    features = torch.zeros(build_graph_n, 2)
    #features = torch.zeros(50, 2)
    
    for j in range(100):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            n = len(envs.results)
            #print("\n*******************Start!*************************")
            #print(f"Step: {n}")
            #print(f"Out: {n}")
            #print(envs.results)
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step], features, n)

            # Obser reward and next obs
            """Guangxi"""
            obs, done, reward = envs.step(action, grid_size_n)
            """Guangxi"""
            features[n][0] = action // grid_size_n
            features[n][1] = action % grid_size_n   
            #print("\n*******************Next Step!*************************")         
            #features[n][0] = action // 32
            #features[n][1] = action % 32
            #print(n)
            #print(envs.results)
            masks = torch.FloatTensor(
                [[0.0] if done else [1.0]])
            bad_masks = torch.FloatTensor([[1.0]])
   
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
            if done:
                episode_rewards.append(reward)
                obs = envs.reset()
                """Guangxi"""
                features = torch.zeros(build_graph_n, 2)
                #features = torch.zeros(50, 2)
            #print(envs.results)
        #print("***************************************")
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1], features, n).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts, features)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            torch.save([
                actor_critic,
                None
            ], "./trained_models/placement_" + str(j) + ".pt")

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            listrewards = list(episode_rewards)
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean reward {:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards,dtype=float),
                        
                        dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = None
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
