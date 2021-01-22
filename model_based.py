import argparse
import json
import os

from algorithms.model_based_ddpg import ModelBasedDDPG
from utils.utils import seed_python, seed_agent


def run(config_file):
    with open(config_file) as f:
        config = json.load(f)

    ENV_NAME = config["ENV_NAME"]
    NB_STEPS = config["NB_STEPS"]
    MAX_STEPS_EP = config["MAX_STEPS_EP"]
    BATCH_SIZE = config["BATCH_SIZE"]
    BUFFER_SIZE = config["BUFFER_SIZE"]
    BUFFER_START_MODEL = config["BUFFER_START_MODEL"]
    BUFFER_START_POLICY = config["BUFFER_START_POLICY"]
    GAMMA = config["GAMMA"]
    LR_ACTOR = config["LR_ACTOR"]
    LR_CRITIC = config["LR_CRITIC"]
    LR_MODEL = config["LR_MODEL"]
    LAMBDA_DYNA = config["LAMBDA_DYNA"]
    LAMBDA_DONE = config["LAMBDA_DONE"]
    POLICY_ITER = config["POLICY_ITER"]
    TAU = config["TAU"]
    H1 = config["H1"]
    H2 = config["H2"]
    INIT_W = config["INIT_W"]
    ASSESS_EVERY_NB_STEPS = config["ASSESS_EVERY_NB_STEPS"]
    ASSESS_NB_EPISODES = config["ASSESS_NB_EPISODES"]
    SW_PATH = config["SW_PATH"]
    SAVE_PATH = config["SAVE_PATH"]
    RANDOM_SEED = config["RANDOM_SEED"]
    SAVE_STEPS = config["SAVE_STEPS"]
    SIGMA_NOISE = config.get("SIGMA_NOISE", None)
    STATE_NOISE = config.get("STATE_NOISE", 0)

    seed_python(RANDOM_SEED)
    agent = ModelBasedDDPG(
        ENV_NAME, h1=H1, h2=H2, buffer_size=BUFFER_SIZE,
        buffer_start_model=BUFFER_START_MODEL,
        buffer_start_policy=BUFFER_START_POLICY,
        gamma=GAMMA, init_w=INIT_W)
    seed_agent(RANDOM_SEED, agent.env)
    agent.train(
        NB_STEPS, MAX_STEPS_EP, BATCH_SIZE, summary_writer_path=SW_PATH,
        lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, lr_model=LR_MODEL,
        lambda_dyna=LAMBDA_DYNA, lambda_done=LAMBDA_DONE, tau=TAU,
        save_steps=SAVE_STEPS, assess_every_nb_steps=ASSESS_EVERY_NB_STEPS,
        sigma_noise=SIGMA_NOISE, policy_iter=POLICY_ITER, state_noise=STATE_NOISE,
        assess_nb_episodes=ASSESS_NB_EPISODES, save_path=SAVE_PATH, verbose=False)


if __name__ == "__main__":
    config_folder = "config"
    # argument parser    
    parser = argparse.ArgumentParser(description="Train using ModelBasedDDPG.")
    parser.add_argument("file")
    args = parser.parse_args()
    config_file = os.path.join(config_folder, args.file)
    run(config_file)
    

