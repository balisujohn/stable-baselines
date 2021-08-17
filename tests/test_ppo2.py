import os

import pytest
import gym 

from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv


@pytest.mark.parametrize("cliprange", [0.2, lambda x: 0.1 * x])
@pytest.mark.parametrize("cliprange_vf", [None, 0.2, lambda x: 0.3 * x, -1.0])
def test_clipping(cliprange, cliprange_vf):
    """Test the different clipping (policy and vf)"""
    model = PPO2('MlpPolicy', 'CartPole-v1',
                 cliprange=cliprange, cliprange_vf=cliprange_vf,
                 noptepochs=2, n_steps=64).learn(100)
    model.save('./ppo2_clip.zip')
    env = model.get_env()
    model = PPO2.load('./ppo2_clip.zip', env=env)
    model.learn(100)

    if os.path.exists('./ppo2_clip.zip'):
        os.remove('./ppo2_clip.zip')

def test_update_n_batch_on_load():
    env = make_vec_env('CartPole-v1',n_envs=2)
    model = PPO2('MlpPolicy', env, n_steps = 10, nminibatches=1)
    
    model.learn(total_timesteps = 100)
    model.save("ppo2_cartpole")
    
    del model
    
    model = PPO2.load("ppo2_cartpole")
    test_env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
    
    model.set_env(test_env)
    model.learn(total_timesteps = 100)
    if os.path.exists('./ppo2_cartpole.zip'):
        os.remove('./ppo2_cartpole.zip')