import copy
import sys

from gym import wrappers
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
K_STEPS = sys.argv[1]
FILE_NAME = "mb_{}.gif".format(K_STEPS)
AGENT_PATH = "models/mb/exp_17/step_{}000".format(K_STEPS)
INTERVAL = 10
FPS = 60
EPISODE_LENGTH = 500
######

def save_frames_as_gif(frames, path='./gif/', filename=FILE_NAME):

    # Mess with this to change frame size
    plt.figure(figsize=(
        frames[0].shape[1] / 72.0,
        frames[0].shape[0] / 72.0),
        dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames),
        interval=INTERVAL, repeat=False)
    anim.save(path + filename, writer='pillow', fps=FPS)

# Prevent from GLEW error
GlfwContext(offscreen=True)
# Run the env
agent = DdpgAgent.load(AGENT_PATH, verbose=False)
actor = copy.deepcopy(agent.actor).to("cpu")
env = agent.env
frames = []
s = torch.from_numpy(env.reset())
done_ind = 0
for t in range(EPISODE_LENGTH):
    # Render to frames buffer
    frames.append(env.render(mode="rgb_array"))
    a = actor.get_action(s)
    next_state, _, done, _ = env.step(a)
    if done:
        done_ind += 1
        if done_ind > 80:
            break
    s = torch.from_numpy(next_state)
env.close()
save_frames_as_gif(frames)