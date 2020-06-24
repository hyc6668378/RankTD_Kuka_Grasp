# coding=utf-8
"""
这个脚本是， 跑了n多个模型， 想挑一个最好的。

方法很简单，开n个进程，一个进程测试一个model

把model编号放到 model_num_list 里面.  跑完会统计成功率。
"""


#---------------超参数搁这调------------------------
model_num_list = [11, 12, 13, 14, 15, 16, 17]  # 对应 model/11.zip ...  model/17.zip
EVALUATE_EPISODE = 1000
#----------------------------------------------------



import numpy as np
from env.KukaGymEnv import KukaDiverseObjectEnv

from multiprocessing import Pool
import multiprocessing
import os
from stable_baselines.common.vec_env import DummyVecEnv
from alg.ppo.ppo2 import PPO2
from tqdm import tqdm

env_fn = lambda : KukaDiverseObjectEnv(renders=False,
                                       maxSteps=32,
                                       blockRandom=0.2, cameraRandom=0.5,
                                       actionRepeat=200, numObjects=1,
                                       single_img=False,
                                       isTest=False,
                                       verbose=False)

def evaluate_worker(model_index):

    """
    不要用 DummyVecEnv 当测试环境。
    前面 env = DummyVecEnv( [env_fn] )
    当交互用环境，准确率 0%. 给我吓尿了，为啥不能用，原因不明。
    """

    env = env_fn()
    model = PPO2.load('model/{}.zip'.format(model_index),
                      env=DummyVecEnv([env_fn]),
                      tensorboard_log=None)

    local_success = 0

    for _ in tqdm(range( EVALUATE_EPISODE )):
        o, done = env.reset(), False

        while not done:
            a = model.predict( o[np.newaxis, :] )
            o, r, done, _info = env.step(a[0][0])

        if _info['is_success']:
            local_success += 1

    # update global_dict
    global_success_dict[model_index] = local_success

if __name__ == '__main__':

    global_success_dict = multiprocessing.Manager().dict()
    for each_model in model_num_list:
        assert isinstance(each_model, int)
        global_success_dict[ each_model ] = 0


    _ = os.system("clear")
    print('Parent process %s.' % os.getpid())
    p = Pool( len(model_num_list) )

    for k in  model_num_list:

        assert isinstance(k, int)

        p.apply_async(evaluate_worker, args=(k,))
    p.close()
    p.join()

    _ = os.system("clear")
    for model_index, success_episode in global_success_dict.items():
        print('model/{}.zip:\tget {:.2f}% successes rate in 1000 episodes'.format(model_index, EVALUATE_EPISODE*success_episode/100))
