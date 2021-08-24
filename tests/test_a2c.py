import os
import pathlib

import pytest
import gym 

from stable_baselines import A2C
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv


def test_a2c_update_n_batch_on_load(tmp_path):
    env = make_vec_env('CartPole-v1',n_envs=2)
    model = A2C('MlpPolicy', env, n_steps = 10)
    
    model.learn(total_timesteps = 100)
    model.save(tmp_path / "a2c_cartpole.zip")
    
    del model
    
    model = A2C.load(tmp_path / "a2c_cartpole.zip")
    test_env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
    
    model.set_env(test_env)
    model.learn(total_timesteps = 100)
    assert(model.n_batch == 10)
    os.remove(tmp_path / "a2c_cartpole.zip")