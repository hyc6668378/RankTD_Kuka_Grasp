from alg.ppo.ppo2 import PPO2
from env.KukaGymEnv import KukaDiverseObjectEnv
from stable_baselines.common.vec_env import SubprocVecEnv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

env_fn = lambda : KukaDiverseObjectEnv(renders=False,
                                       maxSteps=32,
                                       blockRandom=0.2, cameraRandom=0.5,
                                       actionRepeat=200, numObjects=1,
                                       single_img=False,
                                       isTest=False,
                                       verbose=True)


train_from_init = False

if __name__ == '__main__':
    env = SubprocVecEnv( [env_fn]*4 )

    if train_from_init:

        from alg.ppo.common.policies import CnnPolicy
        model = PPO2(CnnPolicy, env=env, seed=0, verbose=2, gamma=0.9,
                          learning_rate=2.5e-4, nminibatches=4, n_steps=128,
                          cliprange_vf=-1, max_grad_norm=0.5,  # 不可少
                          tensorboard_log='logs/ppo2')
    else:
        model = PPO2.load('model/19.zip', env=env, tensorboard_log='logs/ppo2')

    for i in range(20):
        # _ = os.system("clear")
        _ = model.learn(total_timesteps=int(5e+5), reset_num_timesteps=False)
        # model.save('model/'+str(i)+'.zip')
