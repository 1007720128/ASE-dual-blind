import math
import os
import os.path
import pickle
import shutil
import time
import warnings
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from numpy import ndarray as arr
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

from algos.MAPPO import GR_MAPPO, GR_MAPPOPolicy
from algos.gnn_util import format_training_duration, compute_mcs_edge_adj, \
    compute_disseminated_workload
from algos.ppo_buffer import GraphReplayBuffer

warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cur_dir = Path(os.path.dirname(__file__))
models_dir = Path(cur_dir, "model state")
models_dir.mkdir(exist_ok=True)
gma_model = Path(models_dir, "gma")
gma_model.mkdir(exist_ok=True)
results_dir = Path("/Users/xxxx/Project/GatMicroservice/simulation/Evaluation", "Results")
results_dir.mkdir(exist_ok=True, parents=True)
pict_dir = Path("/Users/xxxx/Project/GatMicroservice/simulation/Evaluation", "Pictures")
pict_dir.mkdir(exist_ok=True, parents=True)


def _t2n(x):
    return x.detach().cpu().numpy()


class GMPERunner:
    """
    Runner class to perform training, evaluation and data
    collection for the MPEs. See parent class for details
    """

    dt = 0.1

    def __init__(self, num_servers=8, num_services=5):
        self.n_rollout_threads = 1
        self.envs = EdgeCloudSim(n_server=num_servers)
        self.svc_adj = self.envs.application.adj[np.newaxis, ...].repeat(num_servers, axis=0)
        self.num_env_steps = 10000
        self.episode_length = 200
        self.all_args = None
        self.num_agents = num_servers
        self.num_services = num_services
        self.use_linear_lr_decay = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy = GR_MAPPOPolicy(26, num_services, self.svc_adj)
        self.trainer = GR_MAPPO(self.policy, device=self.device)
        self.buffer = GraphReplayBuffer(26, self.num_agents, self.num_services)
        self.recurrent_N = 1
        self.use_centralized_V = True
        self.hidden_size = 64
        self.save_dir = "/Users/xxxx/Project/Paper-Scalable/simulation/模型状态保存/InfoPPO"
        self.training_log_dir = Path(self.save_dir, "training_log")
        self.training_log_dir.mkdir(parents=True, exist_ok=True)
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.restore()

    def save(self):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        try:
            policy_actor_state_dict = torch.load(
                str(self.save_dir) + "/actor.pt", map_location=torch.device("cpu")
            )
            self.policy.actor.load_state_dict(policy_actor_state_dict)
        except FileNotFoundError:
            print("actor model not saved")
        try:
            policy_critic_state_dict = torch.load(
                str(self.save_dir) + "/critic.pt", map_location=torch.device("cpu")
            )
            self.policy.critic.load_state_dict(policy_critic_state_dict)
        except FileNotFoundError:
            print("critic model not saved")

    def train(self):
        """Train policies with data in buffer."""
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_infos

    def run(self):
        self.warmup()

        start_time = time.time()
        episodes = 500
        rewards_log = []
        # This is where the episodes are actually run.
        for episode in range(episodes):
            print("Episode {}/{}".format(episode + 1, episodes))
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            rewards_episode = []
            # Reset the environment at the beginning.
            self.envs.reset()
            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                # Obs reward and next obs
                obs, svc_obs, agent_id, node_obs, adj, rewards, dones, infos = self.envs.step(
                    torch.tensor(actions_env)
                )
                rewards_episode.append(rewards)

                data = (
                    obs,
                    svc_obs,
                    agent_id,
                    node_obs,
                    adj,
                    self.envs.msc_edge_adj[step + 1],
                    agent_id,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)
            rewards_log.append(np.array(rewards_episode).sum())

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (
                    (episode + 1) * self.episode_length * self.n_rollout_threads
            )

            # save model
            if episode % 5 == 0 or episode == episodes - 1:
                self.save()
                torch.save(np.array(rewards_log), Path(self.training_log_dir, "rewards.pt"))

            # # log information
            # if episode % self.log_interval == 0:
            #     end = time.time()
            #
            #     env_infos = self.process_infos(infos)
            #
            #     avg_ep_rew = np.mean(self.buffer.rewards) * self.episode_length
            #     train_infos["average_episode_rewards"] = avg_ep_rew
            #     print(
            #         f"Average episode rewards is {avg_ep_rew:.3f} \t"
            #         f"Total timesteps: {total_num_steps} \t "
            #         f"Percentage complete {total_num_steps / self.num_env_steps * 100:.3f}"
            #     )
            #     self.log_train(train_infos, total_num_steps)
            #     self.log_env(env_infos, total_num_steps)
            #
            # # eval
            # if episode % self.eval_interval == 0 and self.use_eval:
            #     self.eval(total_num_steps)
        end_time = time.time()
        print("training done, total time taken:", format_training_duration(end_time - start_time))

    def warmup(self):
        # reset env
        obs, svc_obs_, agent_id, node_obs, adj = self.envs.reset()

        # replay buffer
        # (n_rollout_threads, n_agents, feats) -> (n_rollout_threads, n_agents*feats)
        share_obs = obs.reshape(self.n_rollout_threads, -1)
        # (n_rollout_threads, n_agents*feats) -> (n_rollout_threads, n_agents, n_agents*feats)
        share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        # (n_rollout_threads, n_agents, 1) -> (n_rollout_threads, n_agents*1)
        share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
        # (n_rollout_threads, n_agents*1) -> (n_rollout_threads, n_agents, n_agents*1)
        share_agent_id = np.expand_dims(share_agent_id, 1).repeat(
            self.num_agents, axis=1
        )
        svc_obs = svc_obs_[np.newaxis, np.newaxis, ...].repeat(self.num_agents, axis=1)

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.svc_obs[0] = svc_obs.copy()
        self.buffer.node_obs[0] = node_obs.copy()
        self.buffer.adj[0] = adj.copy()
        self.buffer.svc_adj[0] = self.envs.msc_edge_adj[0].copy()
        self.buffer.agent_id[0] = agent_id.copy()
        self.buffer.share_agent_id[0] = share_agent_id.copy()

    @torch.no_grad()
    def collect(self, step: int) -> Tuple[arr, arr, arr, arr, arr, arr]:
        self.trainer.prep_rollout()
        (
            value,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
        ) = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.node_obs[step]),
                                            np.concatenate(self.buffer.adj[step]),
                                            np.concatenate(self.buffer.agent_id[step]),
                                            np.concatenate(self.buffer.share_agent_id[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]),
                                            svc_obs=np.concatenate(self.buffer.svc_obs[step]),
                                            svc_adj=np.concatenate(self.buffer.svc_adj[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions,
        )

    def insert(self, data):
        (
            obs,
            svc_obs,
            agent_id,
            node_obs,
            adj,
            svc_adj,
            agent_id,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # if centralized critic, then shared_obs is concatenation of obs from all agents
        if self.use_centralized_V:
            # TODO stack agent_id as well for agent specific information
            # (n_rollout_threads, n_agents, feats) -> (n_rollout_threads, n_agents*feats)
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            # (n_rollout_threads, n_agents*feats) -> (n_rollout_threads, n_agents, n_agents*feats)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
            # (n_rollout_threads, n_agents, 1) -> (n_rollout_threads, n_agents*1)
            share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
            # (n_rollout_threads, n_agents*1) -> (n_rollout_threads, n_agents, n_agents*1)
            share_agent_id = np.expand_dims(share_agent_id, 1).repeat(
                self.num_agents, axis=1
            )
        else:
            share_obs = obs
            share_agent_id = agent_id

        self.buffer.insert(share_obs, obs, node_obs, adj, agent_id, share_agent_id, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, svc_obs=svc_obs, svc_adj=svc_adj)

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                     np.concatenate(self.buffer.node_obs[-1]),
                                                     np.concatenate(self.buffer.adj[-1]),
                                                     np.concatenate(self.buffer.share_agent_id[-1]),
                                                     np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                     np.concatenate(self.buffer.masks[-1]),
                                                     np.concatenate(self.buffer.svc_obs[-1]),
                                                     np.concatenate(self.buffer.svc_adj[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def eval_AMA_scalable(self):
        self.__init__(num_servers=8)
        eval_rnn_states = np.zeros(
            (1, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (1, self.num_agents, 1), dtype=np.float32
        )
        obs, svc_obs, agent_id, node_obs, adj = self.envs.reset()
        res = []
        pods = []
        pattern_score = []
        for times in range(10):
            res_cur = []
            pods_cur = []
            pattern_score_cur = []
            for eval_step in range(100):
                self.trainer.prep_rollout()
                eval_action, eval_rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    node_obs,
                    np.ones((self.num_agents, self.num_agents, self.num_agents)),
                    np.concatenate(agent_id),
                    np.concatenate(eval_rnn_states),
                    np.concatenate(eval_masks),
                    deterministic=False,
                    svc_obs=svc_obs[np.newaxis, :].repeat(self.num_agents, 0),
                    svc_adj=self.envs.msc_edge_adj[eval_step]
                )
                obs, svc_obs, agent_id, node_obs, adj, rewards, dones, infos = self.envs.step(
                    torch.tensor(eval_action)
                )
                workload_on_server = np.sum(self.envs.actual_workload[eval_step], 0, keepdims=True)
                mask = workload_on_server != 0
                workload_on_server[mask] = workload_on_server[mask] / np.min(workload_on_server)
                resource_pattern = np.sum(Cluster.s_m, 0, keepdims=True)
                cosine_score = cosine_similarity(workload_on_server, resource_pattern).item()
                pattern_score_cur.append(cosine_score)
                res_cur.append(self.envs.send_dataset_workload(eval_step)[2])
                pods_cur.append(Cluster.s_m.sum())
                obs = np.array(obs)[np.newaxis, ...]
                node_obs = np.array(node_obs)[np.newaxis, ...].repeat(self.num_agents, 0)
                eval_rnn_states = np.array(
                    np.split(_t2n(eval_rnn_states), 1)
                )
                eval_rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                eval_masks = np.ones(
                    (1, self.num_agents, 1), dtype=np.float32
                )
                eval_masks[dones == True] = np.zeros(
                    ((dones == True).sum(), 1), dtype=np.float32
                )
            res.append(res_cur)
            pods.append(pods_cur)
            pattern_score.append(pattern_score_cur)
        res = np.array(res)
        pods = np.array(pods)
        pattern_score = np.array(pattern_score)
        np.savez("/Users/xxxx/Project/Paper-Scalable/simulation/实验中间结果/实验结果/流量rps200扩展性实验",
                 res=res,
                 pods=pods,
                 pattern_score=pattern_score)
        print(res.mean())
        # plt.figure()
        # sns.lineplot(x=np.tile(np.arange(res.shape[-1]), 7), y=res.flatten(), label="Response",
        #              errorbar="se",
        #              legend=False, color="b", linewidth=1.1)
        # plt.show()
        #
        # plt.figure()
        # sns.lineplot(x=np.tile(np.arange(pods.shape[-1]), 7), y=pods.flatten(), label="Response",
        #              errorbar="se",
        #              legend=False, color="b", linewidth=1.1)
        # plt.show()
        #
        # plt.figure()
        # sns.lineplot(x=np.tile(np.arange(pattern_score.shape[-1]), 7), y=pattern_score.flatten(), label="Response",
        #              errorbar="se",
        #              legend=False, color="b", linewidth=1.1)
        # plt.show()
        #
        # plt.figure()
        # res_distribution = np.mean(res, 0)
        # res_sorted = np.sort(res_distribution)
        # cdf = np.arange(1, len(res_sorted) + 1) / len(res_sorted)
        # plt.plot(res_sorted, cdf, marker='.', linestyle='none')
        # plt.xlabel('响应时间')
        # plt.ylabel('累积分布 (CDF)')
        # plt.title('响应时间的累积分布图')
        # plt.grid(True)
        # plt.savefig(
        #     "/Users/xxxx/Project/Paper-Scalable/simulation/实验中间结果/结果图/CMA响应时间累积分布图.png")
        # plt.show()

if __name__ == '__main__':
    GMPERunner().train()
