# coding=utf-8

from env.kuka import Kuka
import os
from gym import spaces
import pybullet as p
from env import kuka
import numpy as np
import pybullet_data
import glob
import gym
from gym.utils import seeding
import random
import pyinter
import time


class KukaDiverseObjectEnv(Kuka, gym.Env):

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=200,
                 renders=False,
                 maxSteps=32,
                 blockRandom=0.2,
                 cameraRandom=0,
                 width=128,
                 height=128,
                 numObjects=1,
                 isTest=False,
                 proce_num=0,
                 single_img=False,
                 verbose=True):

        # Environment Characters
        self._timeStep,     self._urdfRoot     =   1. / 240.     , urdfRoot
        self._actionRepeat, self._isTest       =   actionRepeat  , isTest
        self._renders,      self._maxSteps     =   renders       , maxSteps
        self.terminated,    self._verbose      =   0             , verbose
        self._blockRandom,  self._cameraRandom =   blockRandom   , cameraRandom
        self.finger_angle_max = 0.25
        self._width,        self._height       =   width         , height
        self._success,      self._numObjects   =   False         , numObjects
        self._proce_num,  self._single_img     =   proce_num     , single_img

        # [256,128,3]
        self.observation_space = spaces.Box(low=0, high=255, shape=(self._width*2, self._height, 3), dtype=np.uint32)
        if self._single_img:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self._width, self._height, 3),
                                                dtype=np.uint32)

        if self._renders:
            self.cid = p.connect(p.GUI,
                                 options = "--window_backend=2 --render_device=0")
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.cid = p.connect(p.DIRECT)

        self.seed()
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,))  # dx, dy, dz, da

    def reset(self):
        fov = 20. + self._cameraRandom * np.random.uniform(-2, 2)
        aspect = self._width / self._height
        near = 0.01
        far = 10
        self._proj_matrix = p.computeProjectionMatrixFOV(
            fov, aspect, near, far)

        # counter and flag
        self.goal_rotation_angle = 0
        self._env_step = 0
        self.terminated = 0
        self.finger_angle = 0.3
        self._success = False

        self.out_of_range = False
        self._collision_box = False
        self.drop_down = False
        self._rank_before = 0
        self.inverse_rank = 0

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=300)
        p.setTimeStep(self._timeStep)

        self._planeUid = p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])
        self._tableUid = p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"),
           [0.5000000, 0.00000, -.820000], p.getQuaternionFromEuler( np.radians([0,0,90.])))

        p.setGravity(0, 0, -10)
        self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        p.stepSimulation()

        # Choose the objects in the bin.
        urdfList = self._get_random_object(
            self._numObjects, test=self._isTest)
        self._objectUids = self._randomly_place_objects(urdfList)
        self._init_obj_high = self._obj_high()

        self._rank_before = self._rank_1()

        self._domain_random()
        return self._get_observation()

    def _domain_random(self):
        p.changeVisualShape(self._objectUids[0], -1,
                            rgbaColor=[random.random(), random.random(), random.random(), 1.])
        p.changeVisualShape(self._planeUid, -1,
                            rgbaColor=[random.random(), random.random(), random.random(), 1.])
        p.changeVisualShape(self._tableUid, -1,
                            rgbaColor=[random.random(), random.random(), random.random(), 1.])
        for jointIndex in range(self._kuka.numJoints):
            p.changeVisualShape(self._kuka.kukaUid, linkIndex=jointIndex,
                            rgbaColor=[random.random(), random.random(), random.random(), 1.])

        p.changeVisualShape(self._kuka.trayUid, -1,
                            rgbaColor=[random.random(), random.random(), random.random(), 1.])

    def _obj_high(self):
        assert len(self._objectUids) == 1
        # obj.z
        return p.getBasePositionAndOrientation(self._objectUids[0])[0][2]

    def _Current_End_EffectorPos(self):
        state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
        current_End_EffectorPos = np.array(state[0])
        return current_End_EffectorPos

    def _release(self):
        current_End_EffectorPos = self._Current_End_EffectorPos() + np.array([0., 0., 0.02]) # z轴高2cm 抵消重力。

        # release
        for _ in range(500):
            self._kuka.applyAction( current_End_EffectorPos, self.goal_rotation_angle, fingerAngle=self.finger_angle)

            p.stepSimulation()

            self.finger_angle += 0.15 / 100.
            self.finger_angle = np.clip(self.finger_angle, 0., 0.35)

        return self._get_observation()

    def _atomic_action(self, action, repeat_action=200):
        # 执行原子action

        # descale + gravity offset(z axis)
        act_descale = np.array([0.05, 0.05, 0.05, np.radians(90), 1.])
        action = action * act_descale + np.array([0., 0., 0.02, 0., 0.])

        state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
        current_End_EffectorPos = np.array( state[0] )

        goal_pose = np.clip( current_End_EffectorPos + action[:3], a_min=np.array([0.1758, -0.4499, 0.0848]),
                                                                    a_max=np.array([0.79640, 0.5972 , 0.56562]) )
        out_of_range = np.sum( (goal_pose != current_End_EffectorPos + action[:3])[:2] )
        self.out_of_range = True if out_of_range else False

        if self._isTest:
            if out_of_range and self._verbose:
                print("goal_pose:\t", goal_pose )
                print("cur_pose:\t", current_End_EffectorPos + action[:3])
                print("action:\t", action[:3])

        self.goal_rotation_angle += action[-2]  # angel

        # execute
        for _ in range(repeat_action):
            self._kuka.applyAction( goal_pose, self.goal_rotation_angle, fingerAngle=self.finger_angle)
            p.stepSimulation()
            # if self._isTest: time.sleep(self._timeStep)

            if action[-1]>0:
                self.finger_angle += 0.2 / 100.
            else:
                self.finger_angle -= 0.3 / 100.
            self.finger_angle = np.clip(self.finger_angle, 0., 0.3 )

    def _randomly_place_objects(self, urdfList):
        """Randomly places the objects in the bin.

        Args:
          urdfList: The list of urdf files to place in the bin.

        Returns:
          The list of object unique ID's.
        """

        # Randomize positions of each object urdf.
        objectUids = []
        for urdf_name in urdfList:
            xpos = 0.35 + self._blockRandom * random.random()
            ypos = 0.28 + self._blockRandom * (random.random() - .5)
            angle = np.pi / 2 + self._blockRandom * np.pi * random.random()
            orn = p.getQuaternionFromEuler([0, 0, angle])
            urdf_path = os.path.join(self._urdfRoot, urdf_name)
            uid = p.loadURDF(urdf_path, [xpos, ypos, .05],
                             [orn[0], orn[1], orn[2], orn[3]])
            objectUids.append(uid)
            # Let each object fall to the tray individual, to prevent object
            # intersection.
            for _ in range(300):
                p.stepSimulation()
        return objectUids

    def seed(self, seed=None):
        _, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def _get_observation(self):

        look = [1.9+self._cameraRandom * np.random.uniform(-0.05, 0.05),
                0.5+self._cameraRandom * np.random.uniform(-0.05, 0.05),
                 1 +self._cameraRandom * np.random.uniform(-0.05, 0.05)]
        roll = -10
        pitch = -35
        yaw = 110

        look_2 = [-0.3+self._cameraRandom * np.random.uniform(-0.05, 0.05),
                   0.5+self._cameraRandom * np.random.uniform(-0.05, 0.05),
                   1.3+self._cameraRandom * np.random.uniform(-0.05, 0.05)]

        # print("Camera 1:\tx: {:.4f}m\ty: {:.4f}m\tz: {:.4f}m".format(look[0],
        #                                       look[1],
        #                                       look[2]))
        # print("Camera 2:\tx: {:.4f}m\ty: {:.4f}m\tz: {:.4f}m".format( look_2[0],
        #                                       look_2[1],
        #                                       look_2[2]))
        pitch_2 = -56
        yaw_2 = 245
        roll_2 = 0
        distance = 1.

        _view_matrix_2 = p.computeViewMatrixFromYawPitchRoll(
            look_2, distance, yaw_2, pitch_2, roll_2, 2)

        _view_matrix = p.computeViewMatrixFromYawPitchRoll(
            look, distance, yaw, pitch, roll, 2)

        img_arr = p.getCameraImage(width=self._width,
                                   height=self._height,
                                   viewMatrix=_view_matrix,
                                   projectionMatrix=self._proj_matrix,
                                   shadow=True,
                                   lightDirection=[1, 1, 1],
                                   renderer=p.ER_BULLET_HARDWARE_OPENGL
                                   )
        rgb_1 = np.reshape(img_arr[2], (self._height, self._width, 4))[:, :, :3]

        if self._single_img: return rgb_1

        img_arr_2 = p.getCameraImage(width=self._width,
                                   height=self._height,
                                   viewMatrix=_view_matrix_2,
                                   projectionMatrix=self._proj_matrix,
                                   shadow=True,
                                   lightDirection=[1, 1, 1],
                                   renderer=p.ER_BULLET_HARDWARE_OPENGL
                                   )
        rgb_2 = np.reshape(img_arr_2[2], (self._height, self._width, 4))[:, :, :3]

        # (256,128,3)
        return np.concatenate([rgb_1, rgb_2], axis=0)

    def step(self, action):
        self._env_step += 1
        self._atomic_action( action )

        if not self._isTest: self._domain_random()

        obs= self._get_observation()

        reward = self._reward()

        done = self._termination()

        debug = { 'is_success': self._success }

        return obs, reward, done, debug

    def _reward(self):
        # 如果机器人碰到框子。 直接惩罚
        if len(p.getContactPoints(bodyA=self._kuka.trayUid,
                                  bodyB=self._kuka.kukaUid)) != 0:
            self._collision_box = True

            if self._verbose:  print("process: {}\tXiang Zi".format(self._proce_num))

            return -1

        # 出界 有惩罚
        if self.out_of_range:
            if self._verbose: print("process: {}\tout_of_range".format(self._proce_num))
            return -1

        rank = self._rank_1()

        reward = rank - self._rank_before
        self._rank_before = rank

        if reward < 0:
            self.inverse_rank +=1
        return reward

    def _rank_2(self):

        dis_to_box = self._dis_obj_2_tray()

        if dis_to_box > 0.57: rank = 14
        elif dis_to_box in pyinter.openclosed(0.5, 0.57): rank = 15
        elif dis_to_box in pyinter.openclosed(0.4, 0.5): rank = 16
        elif dis_to_box in pyinter.openclosed(0.3, 0.4): rank = 17
        elif dis_to_box in pyinter.openclosed(0.23, 0.3): rank = 18
        elif dis_to_box <= 0.23:
            self.drop_down = True
            rank = 19
        if self._verbose: print("process: {}\tPhase_2 !!\trank: {}\tdis_to_box:{:.3f}".format(self._proce_num, rank, dis_to_box))
        return rank

    def _rank_1(self):

        # obj leave table
        if len(p.getContactPoints(bodyA=self._objectUids[0],
                                  bodyB=self._tableUid)) == 0 or self.drop_down:
            if self.drop_down:
                self._release()
                if len(p.getContactPoints(bodyA=self._objectUids[0],
                                      bodyB=self._kuka.trayUid)) != 0:
                    rank = 20
                    if self._verbose: print("process: {}\tYes! Baby you did it!!!".format(self._proce_num))
                else:
                    rank = 19
                    if self._verbose: print("process: {}\trelease but not in frame..".format(self._proce_num))
                self._success = True
                return rank

            h = self._obj_high() - self._init_obj_high
            if h <= 0.01:
                rank = 9
                if self._verbose: print("process: {}\tobj rising up [0.5-1] cm !".format(self._proce_num))
            elif h in pyinter.openclosed(0.01, 0.04):
                rank = 10
                if self._verbose: print("process: {}\tobj rising up [1-4] cm !".format(self._proce_num))
            elif h in pyinter.openclosed(0.04, 0.07):
                rank = 11
                if self._verbose: print("process: {}\tobj rising up [4-7] cm !".format(self._proce_num))
            elif h in pyinter.openclosed(0.07, 0.1):
                rank = 12
                if self._verbose: print("process: {}\tobj rising up [7-10] cm !".format(self._proce_num))
            elif h in pyinter.openclosed(0.1, 0.15):
                rank = 13
                if self._verbose: print("process: {}\tobj rising up [10-15] cm !".format(self._proce_num))
            elif 0.15 < h:
                rank = self._rank_2()
            return rank

        else:
            dis = self._dis_gripper_2_obj()

            if dis>0.57: rank = 0
            elif dis in pyinter.openclosed(0.37, 0.57): rank = 1
            elif dis in pyinter.openclosed(0.27, 0.37): rank = 2
            elif dis in pyinter.openclosed(0.18, 0.27): rank = 3
            elif dis in pyinter.openclosed(0.14, 0.18): rank = 4
            elif dis in pyinter.openclosed(0.09, 0.14): rank = 5
            elif dis in pyinter.openclosed(0.05, 0.09): rank = 6

            # joint Angle difference.
            else:
                gripper_joint_ = p.getJointState(bodyUniqueId=self._kuka.kukaUid, jointIndex=11)[0] - \
                                 p.getJointState(bodyUniqueId=self._kuka.kukaUid, jointIndex=8)[0]

                if gripper_joint_ not in pyinter.openclosed(0.02, 0.5):
                    rank = 7
                    if self._verbose: print("process: {}\tNot Grasping something.\tRank 7.\t{:.2f}".format(self._proce_num, gripper_joint_))
                else:
                    rank = 8
                    if self._verbose: print("process: {}\tGrasping something.\tRank 8.\t{:.2f}".format(self._proce_num, gripper_joint_))
            return rank

    def _dis_gripper_2_obj(self):
        obj, _ = p.getBasePositionAndOrientation(self._objectUids[0])
        current_End_EffectorPos = np.array(p.getLinkState(self._kuka.kukaUid,
                                                          self._kuka.kukaEndEffectorIndex)[0])
        obj_offset = np.array([0.0, 0.02, 0.0], dtype=np.float32)  # 物体的坐标和真实稍稍错位一点， 一点点调出来的。
        gripper_offset = np.array([0.0, 0.0, 0.25], dtype=np.float32)
        dis = obj - current_End_EffectorPos + obj_offset + gripper_offset
        dis = np.sqrt(np.sum(dis ** 2))
        return dis

    def _dis_obj_2_tray(self):
        obj, _ = p.getBasePositionAndOrientation(self._objectUids[0])
        obj_xy = obj[:2]  # don't use z
        box_pos_xy = np.array(p.getBasePositionAndOrientation(self._kuka.trayUid)[0])[:2]
        dis_to_box = np.sqrt(np.sum((obj_xy - box_pos_xy) ** 2))
        return dis_to_box

    def _termination(self):
        if self._isTest: return (self._env_step >= self._maxSteps) or self.out_of_range or self._success

        if self.inverse_rank > 2 and self._verbose: print("process: {}\tinverse_rank>2".format(self._proce_num))

        return (self._env_step >= self._maxSteps) or \
               self._success or self.out_of_range or \
               self._collision_box or self.inverse_rank>2

    def _get_random_object(self, num_objects, test):
        """Randomly choose an object urdf from the random_urdfs directory.

        Args:
          num_objects:
            Number of graspable objects.

        Returns:
          A list of urdf filenames.
        """
        if test:
            urdf_pattern = os.path.join(self._urdfRoot, 'random_urdfs/*0/*.urdf')
        else:
            urdf_pattern = os.path.join(self._urdfRoot, 'random_urdfs/*[^0]/*.urdf')
        found_object_directories = glob.glob(urdf_pattern)
        total_num_objects = len(found_object_directories)
        selected_objects = np.random.choice(np.arange(total_num_objects),
                                            num_objects)
        selected_objects_filenames = []
        for object_index in selected_objects:
            selected_objects_filenames += [found_object_directories[object_index]]
        return selected_objects_filenames
