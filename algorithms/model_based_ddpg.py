import gym
import numpy as np
import torch
from torch import nn

from algorithms.model_free_ddpg import *


class Model(nn.Module):
    def __init__(self, state_dim, action_dim, h1, h2, init_w=1e-4):
        super(Model, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)

        self.linear_state = nn.Linear(state_dim, h1)
        self.linear_state.weight.data = fanin_(self.linear_state.weight.data.size())

        self.linear_common = nn.Linear(h1 + action_dim, h2)
        self.linear_common.weight.data = fanin_(self.linear_common.weight.data.size())

        self.linear_reward = nn.Linear(h2, 1)
        self.linear_reward.weight.data.uniform_(-init_w, init_w)

        self.linear_dynamics = nn.Linear(h2, state_dim)
        self.linear_dynamics.weight.data.uniform_(-init_w, init_w)

        self.linear_done = nn.Linear(h2, 1)
        self.linear_done.weight.data.uniform_(-init_w, init_w)


    def forward(self, state, action):
        x = self.relu(self.linear_state(state))
        x = torch.cat([x, action], 1)
        x = self.relu(self.linear_common(x))
        x = self.dropout(x)
        reward = self.linear_reward(x)
        dynamics = self.linear_dynamics(x)
        done = self.sigmoid(self.linear_done(x))
        return reward, dynamics, done


class ModelBasedDDPG(DdpgAgent):

    def __init__(self, env_name, actor=None, critic=None, target_actor=None,
            target_critic=None, h1=400, h2=300, device=None,
            buffer_size=1_000_000, buffer_start_model=200, buffer_start_policy=200,
            gamma=0.95, init_w=1e-3):
        """
        DDPG agent.
        Arguments:
            - env_name: str
                Environment name from Gym.
            - actor: Actor
                Actor network from the Actor class.
            - critic: Critic
                Critic network from the Critic class.
            - target_actor: Actor
                Target actor network from the Actor class.
            - target_critic: Critic
                Target critic network from the Critic class.
            - h1: int
                Number of neurons in the first hidden layer of both networks.
            - h2: int
                Number of neurons in the second hidden layer of both networks.
            - device: str ("cpu" or "cuda")
                Device to store and later train the models on.
            - buffer_size: int
                Size of the buffer used during training.
            - buffer_start: int
                Number of samples after which training using the buffer is enabled.
            - gamma: float
                Discount factor for the environment.
            - init_w: float
                Initialization of last layer of actor and critic.
        """
        
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.h1 = h1
        self.h2 = h2
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.a_bound = min(abs(self.env.action_space.low[0]), abs(self.env.action_space.high[0]))
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if actor is None:
            self.actor = Actor(
                state_dim, action_dim, self.a_bound, h1, h2, init_w=init_w).to(self.device)
        else:
            self.actor = actor.to(self.device)

        if critic is None:
            self.critic = Critic(
                state_dim, action_dim, h1, h2, init_w=init_w).to(self.device)
        else:
            self.critic = critic.to(self.device)

        if target_actor is None:
            self.target_actor = Actor(
                state_dim, action_dim, self.a_bound, h1, h2, init_w=init_w).to(self.device)
        else:
            self.target_actor = target_actor.to(self.device)

        if target_critic is None:
            self.target_critic = Critic(
                state_dim, action_dim, h1, h2, init_w=init_w).to(self.device)
        else:
            self.target_critic = target_critic.to(self.device)


        # Copy networks initial values to target networks
        for target_param, param in zip(
                self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(
                self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.noise = OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(action_dim), sigma=0.1*self.a_bound)
        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim)
        self.buffer_size = buffer_size
        self.buffer_start_model = buffer_start_model
        self.buffer_start_policy = buffer_start_policy

        self.gamma = gamma

        self.model = Model(state_dim, action_dim, h1, h2, init_w=init_w).to(self.device)


    def train(self, nb_steps, max_steps_ep=None, batch_size=32, summary_writer_path=None,
            set_device=None, lr_actor=0.0001, lr_critic=0.0001, lr_model=0.0001,
            lambda_dyna=1, lambda_done=1, tau=0.001, sigma_noise=None, policy_iter=1,
            state_noise=0.01, save_steps=None, assess_every_nb_steps=None,
            assess_nb_episodes=1, save_path=None, verbose=True):
        """
        Train the DDPG agent.
        Arguments:
            - nb_steps: int
                Number of environment steps to train the agent on.
            - max_steps_ep: int
                Maximum number of steps per episode.
            - batch_size: int
                Batch size of experience sampled from buffer and trained on
                at each step in the environment.
            - summary_writer_path: str
                Path to write a tensorboard summary for monitoring.
            - set_device: str ("cpu" or "cuda")
                Device to use for training. Default to device mentioned in __init__.
            - lr_actor: float
                Learning rate used in the actor network.
            - lr_critic: float
                Learning rate used in the critic network.
            - tau: float
                Hyperparameter that determines the rate at which the networks'
                parameters move towards the targets ones.
            - sigma_noise: float
                Set the noise added in the actions.
            - save_steps: List[int]
                Save models at specific number of steps.
            - assess_every_nb_steps: int
                Assess the model every assess_every_nb_steps steps.
            - assess_nb_episodes: int
                Number of episodes to assess the model over.
            - save_path: str
                Path where to save intermediate models.
        """
        
        if set_device is not None:
            self.device = set_device

        if summary_writer_path is not None:
            writer = SummaryWriter(summary_writer_path)
        else:
            writer = None
        
        if save_steps is not None and save_path is None:
            save_path = "models/agent_" + \
                datetime.datetime.today().strftime("%Y_%m_%d_%H%M")
        verbose_message = ""

        if sigma_noise is not None:
            self.noise.sigma = sigma_noise*self.a_bound
        
        for model in [self.actor, self.critic, self.target_actor, self.target_critic]:
            model = model.to(self.device)
        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        model_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_model)
        loss_fn = torch.nn.MSELoss()
        done_loss_fn = torch.nn.BCELoss()
        done, steps_in_episode, episode = True, 0, 0
        LOG_FREQ = 100
        t = trange(1, nb_steps + 1)
        for step in t:
            if done or max_steps_ep is not None and steps_in_episode >= max_steps_ep:
                # Reset the episode
                s = torch.from_numpy(self.env.reset()).float()
                self.noise.reset()
                episode += 1
                steps_in_episode = 0
            
            a = self.actor.get_action(s, self.device)
            a = np.clip(a + self.noise(), -self.a_bound, self.a_bound)
            s2, reward, done, _ = self.env.step(a)
            self.buffer.add(s, a, reward, done, s2)
            s = torch.from_numpy(s2).float()
            steps_in_episode +=1
            # Experience replay
            if len(self.buffer) > self.buffer_start_model:
                # Train model
                self.model.train()
                s_b, a_b, reward_b, done_b, s2_b = self.buffer.sample(
                    batch_size, self.device)
                reward_model, dynamics_model, done_model = self.model(s_b, a_b)
                s2_model = s_b.detach() + dynamics_model
                model_optimizer.zero_grad()
                dynamics_loss = loss_fn(s2_model, s2_b)
                reward_loss = loss_fn(reward_model, reward_b)
                done_loss = done_loss_fn(done_model, done_b)
                model_loss = lambda_dyna*dynamics_loss + reward_loss + lambda_done*done_loss
                model_loss.backward()
                model_optimizer.step()
                if writer is not None and step % LOG_FREQ == 0:
                    add_to_summary(writer, "Model/dynamics_loss", dynamics_loss.item(), step)
                    add_to_summary(writer, "Model/reward_loss", reward_loss.item(), step)
                    add_to_summary(writer, "Model/done_loss", done_loss.item(), step)
                    add_to_summary(writer, "Model/total_loss", model_loss.item(), step)

            if len(self.buffer) > self.buffer_start_policy:
                total_actor_loss, total_critic_loss = 0, 0
                # Train policy
                self.model.eval()
                for iter in range(policy_iter+1):
                    if iter > 0:
                        s_b, a_b, reward_b, done_b, s2_b = self.sample_from_model(
                            batch_size, self.device, state_noise)

                    # Compute loss for critic
                    a2_b = self.target_actor(s2_b)
                    target_q = self.target_critic(s2_b, a2_b)
                    y = reward_b + (1 - done_b) * self.gamma * target_q.detach()  # detach to avoid backprop target
                    q = self.critic(s_b, a_b)

                    critic_optimizer.zero_grad()
                    critic_loss = loss_fn(q, y)
                    critic_loss.backward()
                    critic_optimizer.step()
                    total_critic_loss += critic_loss.detach()

                    # Compute loss for actor
                    actor_optimizer.zero_grad()
                    actor_loss = - self.critic(s_b, self.actor(s_b))
                    actor_loss = actor_loss.mean()
                    actor_loss.backward()
                    actor_optimizer.step()
                    total_actor_loss += actor_loss.detach()
                    
                    # Soft update of the networks towards the target networks
                    for target_param, param in zip(
                            self.target_critic.parameters(), self.critic.parameters()):
                        target_param.data.copy_(
                            target_param.data*(1 - tau) + param.data*tau)
                    for target_param, param in zip(
                            self.target_actor.parameters(), self.actor.parameters()):
                        target_param.data.copy_(
                            target_param.data*(1 - tau) + param.data*tau)
                
                if writer is not None and step % LOG_FREQ == 0:
                    add_to_summary(writer, "Loss/actor",
                        total_actor_loss.item()/(policy_iter+1), step)
                    add_to_summary(writer, "Loss/critic",
                        total_critic_loss.item()/(policy_iter+1), step)


            if save_steps is not None and step in save_steps:
                # Save the model
                path = os.path.join(save_path, "step_{}".format(step))
                comment = "Agent trained on {} steps.".format(step)
                verbose_message += self.save(path, comment, verbose=True) + "\n"
            if step % assess_every_nb_steps == 0:
                # Evaluate the model
                return_, steps_per_ep = self.assess(assess_nb_episodes)
                add_to_summary(writer, "Episode/reward", return_, step)
                add_to_summary(writer, "Episode/steps", steps_per_ep, step)
                add_to_summary(writer, "Episode/index", episode, step)

        if writer is not None:
            writer.close()
        if verbose:
            print(verbose_message)

    def sample_from_model(self, size, device, state_noise):
        with torch.no_grad():
            s_b, _, _, _, _ = self.buffer.sample(size, device)
            s_b += state_noise*torch.randn_like(s_b)
            a_b = self.actor(s_b)
            a_b = torch.clip(
                a_b + self.noise.sigma*torch.randn_like(a_b, device=device),
                -self.a_bound, self.a_bound)
            reward, dynamics, done = self.model(s_b, a_b)
            reward_b = reward
            done_b = torch.round(done)
            s2_b = s_b + dynamics
        return s_b, a_b, reward_b, done_b, s2_b
