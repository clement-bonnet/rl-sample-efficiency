# rl-sample-efficiency
Implementation of the classic model-free Deep Deterministic Policy Gradient as
well as its model-based version in Python. This work was done in the context of
a research project on sample efficiency as part of the Deep Learning course
from the MVA master, 2020-2021.

To install required packages, run the following:
>>> python -m pip install -r requirements.txt

Warning: you have to build mujoco_py and you therefore need a mujoco license.
More info on how to install mujoco: https://github.com/openai/mujoco-py.

To run a training of DDPG, run the following for the model-free version:
>>> python model_free.py paper_experiments/model_free_1.json

And the following for the model-based version:
>>> python model_based.py paper_experiments/model_based_1.json

You can create your own config files in the config folder, and then run
training accordingly.

To capture an episode in a gif, change the path in record_episode.py and run the
following (replacing 1000000 with the number of steps you want, as long as a
model exists):
>>> python record_episode.py 1000000

Videos of the experiments are available here: https://youtu.be/KeDRQ7d-ckk