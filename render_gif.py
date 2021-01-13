import copy

from matplotlib import animation
import matplotlib.pyplot as plt
from mujoco_py import GlfwContext
import torch

from algorithms.ddpg import DdpgAgent

"""
Ensure you have imagemagick installed with 
sudo apt-get install imagemagick

Open file in CLI with:
xgd-open <filelname>
"""
######
FILE_NAME = "humanoid.gif"
AGENT_PATH = "models/humanoid/exp_1/step_200000"
INTERVAL = 100
FPS = 60
EPISODE_LENGTH = 100
######

def save_frames_as_gif(frames, path='./gif/', filename=FILE_NAME):

    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=INTERVAL)
    anim.save(path + filename, writer='pillow', fps=FPS)

# Prevent from GLEW error
GlfwContext(offscreen=True)
# Run the env
agent = DdpgAgent.load(AGENT_PATH, verbose=False)
actor = copy.deepcopy(agent.actor).to("cpu")
env = agent.env
frames = []
s = torch.from_numpy(env.reset())
for t in range(EPISODE_LENGTH):
    # Render to frames buffer
    frames.append(env.render(mode="rgb_array"))
    a = actor.get_action(s)
    next_state, _, done, _ = env.step(a)
    if done:
        break
    s = torch.from_numpy(next_state)
env.close()
save_frames_as_gif(frames)