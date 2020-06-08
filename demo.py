#-*-coding:utf-8-*-
"""
Author:  Huang ShengKai

Graduate student, Graduate School of Informatics and Engineering, The University of Electro-Communications.

E-mail: miraclehsk@gmail.com

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from env.KukaGymEnv import KukaDiverseObjectEnv
env = KukaDiverseObjectEnv(renders=True,
                           maxSteps=32,
                           blockRandom=0.0, cameraRandom=0,
                           actionRepeat=200, numObjects=1,
                           single_img=False,
                           isTest=False,
                           verbose=True)


from alg.frozen_policy import RankTD_policy
demo_policy = RankTD_policy()


if __name__ == '__main__':
    os.system("clear")
    env.seed(0)
    for i in range(10):
        _ = os.system("clear")
        print('\n---------------------------------------------------\n')
        print('\n\t\t\tEpisode {}\n'.format(i+1))
        print('---------------------------------------------------\n')

        o, done = env.reset(), False
        while not done:
            a = demo_policy(o)
            o, r, done, info = env.step(a)

