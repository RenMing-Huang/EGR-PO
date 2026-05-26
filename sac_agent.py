from model import Actor, Double_Q_Critic, Q_Critic
import torch.nn.functional as F
import numpy as np
import torch
import copy
import os


class Guided_SAC_countinuous:
    def __init__(self,
                 state_dim,
                 action_dim,
                 dvc,
                 adaptive_alpha=True,
                 alpha=0.12,
                 batch_size=1024,
                 a_lr=0.0003,
                 c_lr=0.0008,
                 gamma=0.99,
                 net_width=512,
                 goal_encoder=None,
                 max_timesteps=1000,
                 rep_dim=10,
                 way_steps=25,
                 rewards_fn=None,
                 ag_dim=None):

        self.dvc = dvc
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.adaptive_alpha = adaptive_alpha
        self.alpha = alpha
        self.batch_size = batch_size
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.gamma = gamma
        self.net_width = net_width
        self.max_timesteps = max_timesteps
        self.ag_dim = ag_dim
        self.rewards_fn = rewards_fn

        self.tau = 0.005
        self.actor = Actor(self.state_dim + rep_dim, self.action_dim,
                           (self.net_width, self.net_width, self.net_width)).to(self.dvc)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

        self.q_critic = Double_Q_Critic(self.state_dim + rep_dim, self.action_dim,
                                        (self.net_width, self.net_width, self.net_width)).to(self.dvc)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)

        self.g_critic = Q_Critic(self.state_dim + rep_dim, self.action_dim,
                                 (self.net_width, self.net_width, self.net_width)).to(self.dvc)
        self.g_critic_optimizer = torch.optim.Adam(self.g_critic.parameters(), lr=self.c_lr)

        self.coff = 0.3
        self.coff_end = 0.3
        self.step = 0
        self.decay_step = 100_000

        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        if self.adaptive_alpha:
            self.target_entropy = torch.tensor(-self.action_dim, dtype=float,
                                               requires_grad=True, device=self.dvc)
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float,
                                          requires_grad=True, device=self.dvc)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.c_lr)

        max_size = 1_000 * max_timesteps
        self.replay_buffer = ReplayBuffer(
            self.state_dim, self.action_dim, self.max_timesteps,
            goal_encoder, max_size=max_size, dvc=self.dvc,
            rep_dim=rep_dim, goal_dim=ag_dim, way_steps=way_steps,
            rewards_fn=rewards_fn,
        )

    def step_decay(self):
        self.step += 1
        self.coff = max(self.coff_end, 1 - self.step / self.decay_step)

    def select_action(self, state, goal, deterministic):
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis, :]).to(self.dvc)
            goal = torch.FloatTensor(goal[np.newaxis, :]).to(self.dvc)
            a, _ = self.actor(torch.cat([state, goal], dim=-1), deterministic, with_logprob=False)
        return a.cpu().numpy()[0]

    def HER_train(self):
        s, s_next, a, r, goal, done, i_r, sub_goal = self.replay_buffer.HER_sample(self.batch_size)

        s = torch.FloatTensor(s).to(self.dvc)
        s_next = torch.FloatTensor(s_next).to(self.dvc)
        a = torch.FloatTensor(a).to(self.dvc)
        r = torch.FloatTensor(r).to(self.dvc).unsqueeze(-1)
        goal = torch.FloatTensor(goal).to(self.dvc)
        done = torch.FloatTensor(done).to(self.dvc).unsqueeze(-1)
        i_r = torch.FloatTensor(i_r).to(self.dvc).unsqueeze(-1)
        sub_goal = torch.FloatTensor(sub_goal).to(self.dvc)

        obs = torch.cat([s, goal], dim=-1)
        sub_obs = torch.cat([s, sub_goal], dim=-1)
        n_obs = torch.cat([s_next, goal], dim=-1)

        with torch.no_grad():
            a_next, log_pi_a_next = self.actor(n_obs, deterministic=False, with_logprob=True)
            target_Q1, target_Q2 = self.q_critic_target(n_obs, a_next)
            target_Q = r + (1.0 - done) * self.gamma * (
                torch.min(target_Q1, target_Q2) - self.alpha * log_pi_a_next
            )

        current_Q1, current_Q2 = self.q_critic(obs, a)
        q_loss = (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)) / 2
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        g_value = self.g_critic(sub_obs, a)
        g_loss = F.mse_loss(g_value, i_r)
        self.g_critic_optimizer.zero_grad()
        g_loss.backward()
        self.g_critic_optimizer.step()

        for params in self.q_critic.parameters():
            params.requires_grad = False

        a_samp, log_pi_a = self.actor(obs, deterministic=False, with_logprob=True)
        _a, _ = self.actor(sub_obs, deterministic=True, with_logprob=False)
        current_Q1, current_Q2 = self.q_critic(obs, a_samp)
        q_value = torch.min(current_Q1, current_Q2)
        Q = q_value + self.g_critic(sub_obs, _a) * self.coff

        a_loss = (self.alpha * log_pi_a - Q).mean()
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        for params in self.q_critic.parameters():
            params.requires_grad = True

        if self.adaptive_alpha:
            alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.step_decay()
        return {
            'a_loss': a_loss.item(),
            'q_loss': q_loss.item(),
            'alpha': self.alpha,
            'g_loss': g_loss.item(),
            'g_value': g_value.mean().item(),
            'q_value': q_value.mean().item(),
            'log_pi_a': log_pi_a.mean().item(),
        }

    def save(self, env_name, timestep):
        root_path = f"./models/{env_name}"
        os.makedirs(root_path, exist_ok=True)
        torch.save(self.actor.state_dict(), f"./models/{env_name}/actor{timestep}.pth")
        torch.save(self.q_critic.state_dict(), f"./models/{env_name}/q_critic{timestep}.pth")

    def load(self, env_name, timestep):
        self.actor.load_state_dict(torch.load(f"./models/{env_name}/actor{timestep}.pth"))
        self.q_critic.load_state_dict(torch.load(f"./models/{env_name}/q_critic{timestep}.pth"))


class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, max_timesteps, goal_encoder,
                 max_size, dvc, rep_dim, goal_dim, way_steps, rewards_fn):
        self.dvc = dvc
        self.ptr = 0
        self.size = 0
        self.T = max_timesteps
        self.max_size = max_size // self.T
        self.future_p = 1 - 1 / 5
        self.goal_encoder = goal_encoder
        self.way_steps = way_steps
        self.rewards_fn = rewards_fn

        self.buffers = {
            'obs': np.empty([self.max_size, self.T, obs_dim]),
            'n_obs': np.empty([self.max_size, self.T, obs_dim]),
            'sub_g': np.empty([self.max_size, self.T, rep_dim]),
            'actions': np.empty([self.max_size, self.T, action_dim]),
            'i_r': np.empty([self.max_size, self.T]),
            'rewards': np.empty([self.max_size, self.T]),
            'episode_len': np.empty([self.max_size]),
            'ag': np.empty([self.max_size, self.T, goal_dim]),
            'done': np.empty([self.max_size, self.T]),
        }

    def add_episode(self, ep_obs, ep_sg, ep_ir, ep_actions, ep_r, ep_ag, ep_done):
        ep_obs = np.array(ep_obs)
        ep_sg = np.array(ep_sg)
        ep_ir = np.array(ep_ir)
        ep_actions = np.array(ep_actions)
        ep_r = np.array(ep_r)
        length = len(ep_obs) - 1

        self.buffers['obs'][self.ptr][:length] = ep_obs[:-1]
        self.buffers['n_obs'][self.ptr][:length] = ep_obs[1:]
        self.buffers['sub_g'][self.ptr][:length] = ep_sg
        self.buffers['i_r'][self.ptr][:length] = ep_ir
        self.buffers['actions'][self.ptr][:length] = ep_actions
        self.buffers['rewards'][self.ptr][:length] = ep_r
        self.buffers['episode_len'][self.ptr] = length
        self.buffers['ag'][self.ptr][:length] = ep_ag
        self.buffers['done'][self.ptr][:length] = ep_done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    @torch.no_grad()
    def goal_encode(self, obs):
        obs = torch.FloatTensor(obs).to(self.dvc)
        return self.goal_encoder(obs, None).cpu().numpy()

    def HER_sample(self, batch_size):
        episode_idxs = np.random.randint(0, self.size, batch_size)
        episode_lengths = self.buffers['episode_len'][episode_idxs].astype(int)
        t_samples = np.array([np.random.randint(0, length) for length in episode_lengths])

        obs = self.buffers['obs'][episode_idxs, t_samples]
        ag = self.buffers['ag'][episode_idxs, t_samples]
        n_obs = self.buffers['n_obs'][episode_idxs, t_samples]
        sub_g = self.buffers['sub_g'][episode_idxs, t_samples]
        actions = self.buffers['actions'][episode_idxs, t_samples]
        i_r = self.buffers['i_r'][episode_idxs, t_samples]

        r = np.full_like(i_r, -1.0)
        done = np.zeros_like(i_r)
        g = sub_g.copy()

        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)[0]

        future_offset = (
            np.random.exponential(scale=self.way_steps, size=batch_size)
            .clip(1, episode_lengths - t_samples) - 1
        ).astype(int)
        future_offset[np.random.uniform(size=batch_size) < 0.3] = 0
        future_t = (t_samples + future_offset)[her_indexes]

        future_ag = self.buffers['ag'][episode_idxs[her_indexes], future_t]

        if self.rewards_fn is None:
            achieved_goal_mask = future_offset[her_indexes] == 0
        else:
            achieved_goal_mask = self.rewards_fn(future_ag, ag[her_indexes])

        achieved_goal_indexes = np.flatnonzero(achieved_goal_mask)
        future_ag_encoded = self.goal_encode(future_ag)

        done[her_indexes[achieved_goal_indexes]] = 1.0
        r[her_indexes[achieved_goal_indexes]] = 0.0
        g[her_indexes] = future_ag_encoded

        return obs, n_obs, actions, r, g, done, i_r, sub_g

    def save_replay(self, env_name):
        path = f"buffers/{env_name}.npy"
        np.save(path, {
            'observation': self.buffers['obs'],
            'length': self.buffers['episode_len'],
            'size': self.size,
        })
