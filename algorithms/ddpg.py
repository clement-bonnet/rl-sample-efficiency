import copy
import datetime
import os
import json
import time
import random

import gym
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

def fanin_(size):
    fan_in = size[0]
    weight = 1./np.sqrt(fan_in)
    return torch.Tensor(size).uniform_(-weight, weight)

def add_to_summary(writer, name, value, timestep):
    try:
        writer.add_scalar(name, value, timestep)
    except Exception:
        pass

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, h1, h2, init_w=1e-4):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(state_dim, h1)
        self.linear1.weight.data = fanin_(self.linear1.weight.data.size())
        self.linear2 = nn.Linear(h1 + action_dim, h2)
        self.linear2.weight.data = fanin_(self.linear2.weight.data.size())
        self.linear3 = nn.Linear(h2, 1)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.relu = nn.ReLU()
        
    def forward(self, state, action):
        x = self.linear1(state)
        x = self.relu(x)
        x = self.linear2(torch.cat([x, action], 1))
        x = self.relu(x)
        x = self.linear3(x)
        return x

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, a_bound, h1, h2, init_w=1e-4):
        super(Actor, self).__init__()
        self.a_bound = a_bound
        self.linear1 = nn.Linear(state_dim, h1)
        self.linear1.weight.data = fanin_(self.linear1.weight.data.size())
        self.linear2 = nn.Linear(h1, h2)
        self.linear2.weight.data = fanin_(self.linear2.weight.data.size())
        self.linear3 = nn.Linear(h2, action_dim)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, state):
        x = self.linear1(state)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.a_bound*self.tanh(x)    # rescale output in [-a_bound, +a_bound]
        return x
    
    def get_action(self, state, device="cpu"):
        state = torch.FloatTensor(state.float()).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]
    
    
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu=0, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self._sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta*(self.mu - self.x_prev)*self.dt + self._sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return "OrnsteinUhlenbeckActionNoise(mu={}, sigma={})".format(self.mu, self._sigma)
    
    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, new_sigma):
        if new_sigma > 0:
            self._sigma = new_sigma
        else:
            print("The value of sigma has not changed. Please enter a positive value.")

class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim, buffer_name=""):
        assert buffer_size > 0, "buffer_size must be > 0."
        self.buffer_size = buffer_size
        self.buffer_name = buffer_name
        self.buffer_s = np.empty((buffer_size, state_dim))
        self.buffer_a = np.empty((buffer_size, action_dim))
        self.buffer_r = np.empty(buffer_size)
        self.buffer_d = np.empty(buffer_size, dtype=np.bool)
        self.buffer_s2 = np.empty((buffer_size, state_dim))
        self.index = -1

    def add(self, s, a, reward, done, s2):
        self.index += 1
        index = (self.index)%self.buffer_size
        self.buffer_s[index] = s
        self.buffer_a[index] = a
        self.buffer_r[index] = reward
        self.buffer_d[index] = done
        self.buffer_s2[index] = s2
        
    def size(self):
        return self.buffer_size        
    
    def __len__(self):
        return min(self.index+1, self.buffer_size)
    
    def sample(self, batch_size, device):
        indices = np.random.choice(len(self), batch_size)
        s_b = torch.FloatTensor(self.buffer_s[indices]).to(device)
        a_b = torch.FloatTensor(self.buffer_a[indices]).to(device)
        reward_b = torch.FloatTensor(self.buffer_r[indices]).unsqueeze(1).to(device)
        done_b = torch.FloatTensor(np.float32(self.buffer_d[indices])).unsqueeze(1).to(device)
        s2_b = torch.FloatTensor(self.buffer_s2[indices]).to(device)
        return s_b, a_b, reward_b, done_b, s2_b
    
    def clear(self):
        self.buffer_s = 0
        self.buffer_a = 0
        self.buffer_r = 0
        self.buffer_d = False
        self.buffer_s2 = 0
        self.index = -1
    
    def __repr__(self):
        return "ReplayBuffer named {} with a buffer size of {}.".format(self.buffer_name, self.buffer_size)


class DdpgAgent:
    """
    Deep Deterministic Policy Gradient agent.
    """

    def __init__(self, env_name, actor=None, critic=None, target_actor=None,
            target_critic=None, h1=400, h2=300, device=None,
            buffer_size=1_000_000, buffer_start=200, gamma=0.95, init_w=1e-3):
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
        self.buffer_start = buffer_start

        self.gamma = gamma

    def train(self, nb_steps, max_steps_ep=None, batch_size=32, summary_writer_path=None,
            set_device=None, lr_actor=0.0001, lr_critic=0.0001, tau=0.001,
            sigma_noise=None, save_steps=None, assess_every_nb_steps=None,
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
            self.noise.sigma = sigma_noise
        
        for model in [self.actor, self.critic, self.target_actor, self.target_critic]:
            model = model.to(self.device)
        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        loss_fn = torch.nn.MSELoss()
        done, steps_in_episode = True, 0
        t = trange(1, nb_steps + 1)
        for step in t:
            if done or max_steps_ep is not None and steps_in_episode >= max_steps_ep:
                # Reset the episode
                s = torch.from_numpy(self.env.reset())
                self.noise.reset()
                steps_in_episode = 0
            
            a = self.actor.get_action(s, self.device)
            a = np.clip(a + self.noise(), -self.a_bound, self.a_bound)
            s2, reward, done, info = self.env.step(a)
            self.buffer.add(s, a, reward, done, s2)
            steps_in_episode +=1
            # Experience replay
            if len(self.buffer) > self.buffer_start:
                s_b, a_b, reward_b, done_b, s2_b = self.buffer.sample(batch_size, self.device)

                # Compute loss for critic
                a2_b = self.target_actor(s2_b)
                target_q = self.target_critic(s2_b, a2_b)
                y = reward_b + (1 - done_b) * self.gamma * target_q.detach()  # detach to avoid backprop target
                q = self.critic(s_b, a_b)

                critic_optimizer.zero_grad()
                critic_loss = loss_fn(q, y)
                critic_loss.backward()
                critic_optimizer.step()

                # Compute loss for actor
                actor_optimizer.zero_grad()
                actor_loss = - self.critic(s_b, self.actor(s_b))
                actor_loss = actor_loss.mean()
                actor_loss.backward()
                actor_optimizer.step()

                if writer is not None:
                    add_to_summary(writer, "Loss/actor", actor_loss.item(), step)
                    add_to_summary(writer, "Loss/critic", critic_loss.item(), step)
                
                # Soft update of the networks towards the target networks
                for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                    target_param.data.copy_(target_param.data*(1 - tau) + param.data*tau)
                for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                    target_param.data.copy_(target_param.data*(1 - tau) + param.data*tau)

            s = torch.from_numpy(s2)

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

        if writer is not None:
            writer.close()
        if verbose:
            print(verbose_message)

    def assess(self, nb_episodes=1):
        """
        Evaluate the model and return the average return and number of steps.
        """
        return_ = 0
        nb_steps = 0
        env = gym.make(self.env_name)
        actor = copy.deepcopy(self.actor).to("cpu")
        for _ in range(nb_episodes):
            s = torch.from_numpy(env.reset())
            done = False
            while not done:
                a = actor.get_action(s)
                s2, reward, done, _ = env.step(a)
                return_ += reward
                nb_steps += 1
                s = torch.from_numpy(s2)
        return_ = return_/nb_episodes
        nb_steps = nb_steps/nb_episodes
        return return_, nb_steps



    def test_display(self, nb_episodes=5, max_steps=None, sleep_step=0, sleep_episode=1):
        """
        Test the agent on the environment with a given number of episodes.
        """

        actor_loaded = copy.deepcopy(self.actor).to("cpu")
        self.env.reset()
        self.env.render()
        for _ in range(nb_episodes):
            done = False
            s = torch.from_numpy(self.env.reset())
            step = 1
            while not done:
                if max_steps is not None and step > max_steps:
                    break
                a = actor_loaded.get_action(s)
                s2, reward, done, info = self.env.step(a)
                self.env.render()
                time.sleep(sleep_step)
                if done: break
                s = torch.from_numpy(s2)
                step += 1
            time.sleep(sleep_episode)
        self.env.close()


    def save(self, path, comment=None, verbose=False):
        """
        Save an agent to the given path. Will be saved in the folder:
        - agent_param.json
        - actor.pt
        - critic.pt
        - target_actor.pt
        - target_critic.pt
        """

        if not os.path.exists(os.path.join(".", path)):
            os.makedirs(path)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pt"))
        torch.save(self.target_actor.state_dict(), os.path.join(path, "target_actor.pt"))
        torch.save(self.target_critic.state_dict(), os.path.join(path, "target_critic.pt"))
        agent_param = {
            "env_name": self.env_name,
            "h1": self.h1,
            "h2": self.h2,
            "buffer_size": self.buffer_size,
            "buffer_start": self.buffer_start,
            "gamma": self.gamma
        }
        if comment is not None:
            agent_param["comment"] = comment
        with open(os.path.join(path, "agent_param.json"), "w") as fp:
            json.dump(agent_param, fp, indent=4)
        message = comment + "\nAgent saved to: " + path
        if verbose:
            return message


    def load(path, device="cpu", verbose=True):
        """
        Load an agent from a path that contains:
        - agent_param.json
        - actor.pt
        - critic.pt
        - target_actor.pt
        - target_critic.pt
        """

        if not os.path.exists(os.path.join(path, "agent_param.json")):
            raise ValueError("agent_param.json is not found at the address: {}".format(path))
        with open(os.path.join(path, "agent_param.json"), "r") as fp:
            agent_param = json.load(fp)
        env_name = agent_param["env_name"]
        h1 = agent_param["h1"]
        h2 = agent_param["h2"]
        buffer_size = agent_param["buffer_size"]
        buffer_start = agent_param["buffer_start"]
        gamma = agent_param["gamma"]
        comment = agent_param.get("comment", "")

        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        a_bound = min(abs(env.action_space.low[0]), abs(env.action_space.high[0]))
        
        actor = Actor(state_dim, action_dim, a_bound, h1, h2)
        actor.load_state_dict(torch.load(
            os.path.join(path, "actor.pt"),
            map_location=torch.device(device)))
        actor.eval()

        critic = Critic(state_dim, action_dim, h1, h2)
        critic.load_state_dict(torch.load(
            os.path.join(path, "critic.pt"),
            map_location=torch.device(device)))
        critic.eval()

        target_actor = Actor(state_dim, action_dim, a_bound, h1, h2)
        target_actor.load_state_dict(torch.load(
            os.path.join(path, "target_actor.pt"),
            map_location=torch.device(device)))
        target_actor.eval()

        target_critic = Critic(state_dim, action_dim, h1, h2)
        target_critic.load_state_dict(torch.load(
            os.path.join(path, "target_critic.pt"),
            map_location=torch.device(device)))
        target_critic.eval()

        agent = DdpgAgent(env_name, actor=actor, critic=critic,
            target_actor=target_actor, target_critic=target_critic, h1=h1,
            h2=h2, buffer_size=buffer_size, buffer_start=buffer_start,
            gamma=gamma)
        if verbose:
            print("Agent loaded!", "Comment: {}".format(comment) if comment else "")
        return agent