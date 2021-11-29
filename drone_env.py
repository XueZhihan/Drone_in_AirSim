import airsim
import time
import copy
import numpy as np
from PIL import Image
import cv2
import math
from sklearn import preprocessing
import dense_solution as ds
goal_threshold = 3
np.set_printoptions(precision=3, suppress=True)
IMAGE_VIEW = True
speed_limit = 0.2
goals = [4.0, 9.0, 14.0, 19.0, 24.0, 29.0, 34.0, 39.0, 44.0, 49.0, 54.0, 59.0]
level_end = 0
clockspeed = 1
timeslice = 1.0 / clockspeed
class drone_env:
    def __init__(self, start, aim):
        self.start = np.array(start)
        self.aim = np.array(aim)
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.threshold = goal_threshold
        self.lv = 0
        self.dis = 0
    def reset(self, dis):

        position = airsim.Vector3r(self.dis, 0, -3.0)
        heading = airsim.utils.to_quaternion(0, 0, 0)
        pose = airsim.Pose(position, heading)
        self.client.simSetVehiclePose(pose, True)
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        #self.client.moveToPositionAsync(self.start.tolist()[0], self.start.tolist()[1], self.start.tolist()[2], 5,)
        self.client.takeoffAsync()
        time.sleep(5)

    def isDone(self):
        pos = self.client.simGetVehiclePose()
        if distance(self.aim, pos) < self.threshold:
            return True
        return False

    def moveByDist(self, diff, timeslice, forward=False):
        temp = airsim.YawMode()
        temp.is_rate = not forward
        m = math.pi/2
        self.client.moveByVelocityAsync(diff[0], diff[1], diff[2], timeslice, drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                   yaw_mode=temp)

        # self.client.moveByAngleRatesThrottleAsync(diff[0], diff[1]*0.5, diff[2]*0.5, diff[3]*0.5+0.5, duration=timeslice)
        # time.sleep(0.5)

        return 0

    def render(self, extra1="", extra2=""):
        pos = v2t(self.client.simGetVehiclePose())
        goal = distance(self.aim, pos)
        print(extra1, "distance:", int(goal), "position:", pos.astype("int"), extra2)

    def help(self):
        print("drone simulation environment")

 # -------------------------------------------------------
    # height control
    # continuous control

class drone_env_block(drone_env):
    def __init__(self, start=[0, 0, 0], aim=[58, 125, 10], scaling_factor=2, img_size=[64, 64]):
        drone_env.__init__(self, start, aim)
        self.scaling_factor = scaling_factor
        self.aim = np.array(aim)
        self.height_limit = -30
        self.rand = False
        self.level = 0
        #self.rand = True
        self.start = np.array([0, 0, -3])
        self.lv = 0

    def reset_aim(self):
        self.aim = (np.random.rand(3) * 300).astype("int") - 150
        self.aim[2] = -np.random.randint(10) - 5
        print("Our aim is: {}".format(self.aim).ljust(80, " "), end='\r')
        self.aim_height = self.aim[2]

    def reset(self, env_switch):
        if self.rand:
            self.reset_aim()
        if env_switch:
            self.level += 2
            self.lv += 2
            if self.lv >= 10:
                self.lv = 0
            if self.lv != 0:
                self.dis = goals[self.lv-1]
            else:
                self.dis = 0
        drone_env.reset(self, self.dis)
        self.img0 = self.getImg()
        #self.state = self.getState()
        return self.img0

    def getState(self):
        pos = v2t(self.client.simGetVehiclePose().position)
        #vel = v2t(self.client.simGetVehiclePose().position)
        img = self.getImg()
        #state = ds.dense_op_dis(self.img0, img)
        # state= ds.op(self.img0, img)
        # self.img0 = img
        return img

    def step(self, action):
        level_end = 0
        # pos = v2t(self.client.getMultirotorState().kinematics_estimated.position)
        # quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity

        #dpos = self.aim - pos
        # for i in range(2):
        #     if abs(action[i]) > 1:
        #         print("action value error")
        #         action = action / abs(action)
        depth = self.getImg()
        depth = depth[:, :, 24: 40, 24: 40][0][0]
        depthshow = np.array(depth*255, dtype=np.uint8)
        cv2.imshow('depth', depthshow)
        k = cv2.waitKey(1) & 0xff
        depth = np.mean(depth)
        #temp = np.sqrt(dpos[0] ** 2 + dpos[1] ** 2)
        dx = action[0] * self.scaling_factor
        dy = - action[1] * self.scaling_factor
        dz = - action[2] * self.scaling_factor
        #dt = - action[3] * self.scaling_factor
        # print (dx,dy,dz)
        #self.client.simPause(False)

        has_collided = False
        landed = False

        drone_env.moveByDist(self, [dx, dy, dz], timeslice, forward=True)
        collision_count = 0
        start_time = time.time()
        while time.time() - start_time < timeslice:


            pos = self.client.getMultirotorState().kinematics_estimated.position
            vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
            collided = self.client.simGetCollisionInfo().has_collided
            landed = (vel.x_val == 0 and vel.y_val == 0 and vel.z_val == 0)
            landed = landed or pos.z_val > 0
            collision = collided or landed
            if collision:
                collision_count += 1
            if collision_count > 50:
                has_collided = True
                break
        #self.client.simPause(True)

        info = None
        done = False
        state_ = self.getState()
        new_depth = self.getDepth()
        new_depth0 = new_depth.copy()
        #new_depth1 = new_depth1[:, :, 36: 108, 64: 190][0][0]
        new_depth1 = new_depth0[:, :, 24: 40, 24: 40][0][0]
        newdepthshow = np.array(new_depth1 * 255, dtype=np.uint8)
        cv2.imshow('new_depth', newdepthshow)
        k = cv2.waitKey(1) & 0xff
        new_depth = np.mean(new_depth1)
        crash_threshold = 0.07
        sigdepth = new_depth0.copy()
        m_depth = np.median(sigdepth)
        # sigdepth[sigdepth < 0.1] = 0
        # sigdepth[sigdepth > m_depth] = 1
        # sigdepth[sigdepth <= m_depth] = 0
        sigdepth = m_depth
        #cv2.imshow('sigdepth', sigdepth[0][0])
        k = cv2.waitKey(1) & 0xff
        # if pos.x_val >= goals[self.level]:
        #     reward += 10 * (1+1/6)
        # if pos.x_val >= 18:
        #     reward += 10 * (1+2/6)
        # if pos.x_val >= 28:
        #     reward += 10 * (1+3/6)
        # if pos.x_val >= 38:
        #     reward += 10 * (1+4/6)
        # if pos.x_val >= 48:
        #     reward += 10 * (1+5/6)
        pos = self.client.getMultirotorState().kinematics_estimated.position
        vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity

        if self.isDone():

            done = True
            #reward += 10 * (1+6/6)
            info = "success"

        if has_collided:
            #reward += -10
            done = True
            info = "collision"


        # if landed:
        #     done = True
        #     info = "collision"
        """
        if (pos + self.aim_height) < self.height_limit:
            done = True
            info = "too high"
            reward = -50
        """
        reward, info = self.rewardf(depth, new_depth, vel, pos, info, sigdepth)
        self.state = state_
        if done:
            level_end = self.level
            self.level = self.lv

        #reward /= 50
        norm_state = copy.deepcopy(state_)
        #norm_state[1] = norm_state[1] / 100

        return norm_state, reward, done, info, sigdepth, level_end

    def isDone(self):
        pos = self.client.getMultirotorState().kinematics_estimated.position
        #pos[2] = self.aim[2]
        if pos.x_val >= goals[9]:
            return True
        return False

    def rewardf(self, depth, new_depth, vel, pos, info,sigdepth):
        vel = np.array([vel.x_val, vel.y_val, vel.z_val], dtype=np.float)
        speed = np.linalg.norm(vel)
        reward = 0

        # if pos.x_val >= goals[self.level] - 1:
        #     self.level += 1
        #     info = "success"
        #     print("success")

        if info is "collision":
            reward = -2.0
            print("CR+: %r" %(reward))
        elif pos.x_val >= goals[self.level]-1:
            self.level += 1
            # reward = config.reward['forward'] * (1 + self.level / len(goals))
            #reward = 2.0 * (1 + self.level / len(goals))
            reward = 0.5
            info = "success"
            print("SR+: %r" %(reward))

        # elif speed < speed_limit:
        #     reward = -0.05
        # elif vel[0] > 0:
        #     reward = float(vel[0]) * 0.1
        #     print("VR+: %r" %(reward))
        # else:
        #     reward = float(vel[0]) * 0.5
        #     print("VR-: %r" %(reward))
        # L_new, C_new, R_new = self.avg_depth(d_new, thresh)
        # if C_new < crash_threshold:
        #     done = True
        #     reward = -1
        # else:
        #     done = False
        #     if action[0] == 0:
        #         reward = C_new
        #     else:
        #         # reward = C_new/3
        #         reward = C_new
        #pos = state[1][0]
        #pos_ = state_[1][0]

        else:
            # if vel[0] > 0:
            #     v_reward = float(vel[0]) * 0.1
            #     print("VR+: %r" % (v_reward))
            # else:
            #     v_reward = float(vel[0]) * 0.5
            #     print("VR-: %r" % (v_reward))
            action_reward = (new_depth - sigdepth)*10

            #action_reward = 0.1
            if action_reward > 0:
                action_reward = 0.2
            else:
                action_reward = -0.2
            reward += action_reward
            #reward += v_reward
            print("AR+: %r" % (action_reward))
            # depth_reward = np.mean(sigdepth)
            # reward += depth_reward * 5
            # print("DR+: %r" % (depth_reward))
            action_reward = new_depth - depth
            # if np.abs(action_reward) >= 0.1 and np.abs(reward)< 3:
            if action_reward <= 0:
                action_reward = -0.2
            else:
                action_reward = 0.2
            reward += action_reward
            # print("AR+: %r" % (action_reward))
            # print("R+: %r" % (action_reward+depth_reward))
        # print(new_depth - depth)
        # if new_depth - depth >= 0.2 and info is "collision":
        #     info = "through"
        print(": %r" % (reward))
        return reward, info

    def getDepth(self):
        responses = self.client.simGetImages(
              [airsim.ImageRequest(0, airsim.ImageType.DepthVis, True)])
        response = responses[0]
        while response.height == 0:
            print('img stuck!')
            response = self.client.simGetImages(
                [airsim.ImageRequest(0, airsim.ImageType.DepthVis, True)])[0]

        img1d = np.array(response.image_data_float, dtype=np.float)
        img1d = np.array(np.clip(255 * 3 * img1d, 0, 255), dtype=np.uint8)
        img2d = np.reshape(img1d, (response.height, response.width))
        image = Image.fromarray(img2d)
        image = np.array(image.convert('L'))

        # cv2.imwrite('view.png', image)
        image = cv2.resize(image, (128, 72))

        #cv2.imshow('frame', image)
        k = cv2.waitKey(1) & 0xff
        image = np.float32(image.reshape((1, 1, 72, 128)))
        image /= 255.0

        return image
    def getImg(self):

        # responses = self.client.simGetImages(
        #     [airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])
        # response = responses[0]
        # img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        # #img1d = np.array(responses[0].image_data_float, dtype=np.float)
        # #img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
        # img_rgb = img1d.reshape(response.height, response.width, 3)
        # image = Image.fromarray(img_rgb)
        # im_final = np.array(image.resize((64, 64)).convert('L'), dtype=np.float) / 255
        # im_final.resize((64, 64, 1))
        # if IMAGE_VIEW:
        #     cv2.imshow("view", im_final)
        #     key = cv2.waitKey(1) & 0xFF;
        # return im_final

        # responses = self.client.simGetImages(
        #     [airsim.ImageRequest(0, airsim.ImageType.DepthPerspective, True, False)])
        # img1d = np.array(responses[0].image_data_float, dtype=np.float)
        # img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
        # image = Image.fromarray(img2d)
        # im_final = np.array(image.resize((64, 64)).convert('L'), dtype=np.float) / 255
        # im_final.resize((64, 64, 1))
        # if IMAGE_VIEW:
        #     cv2.imshow("view", im_final)
        #     key = cv2.waitKey(1) & 0xFF;
        # return im_final
        # responses = self.client.simGetImages(
        #      [airsim.ImageRequest(0, airsim.ImageType.DepthVis, True)])
        responses = self.client.simGetImages(
            [airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])
        response = responses[0]
        while response.height == 0:
            print('img stuck!')
            response = self.client.simGetImages(
                [airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])[0]
        # img1d = np.array(response.image_data_float, dtype=np.float)  # get numpy array
        # img_rgba = img1d.reshape(response.height, response.width)
        # img = Image.fromarray(img_rgba)
        # img_l = img.convert('L')
        # camera_image_L = np.asarray(img_l)
        # camera_image = camera_image_L
        #
        # if IMAGE_VIEW:
        #     cv2.imshow("view", camera_image)
        #     key = cv2.waitKey(1) & 0xFF

        # # state = cv2.resize(camera_image, (103, 103), cv2.INTER_LINEAR)
        # state = cv2.normalize(camera_image, camera_image, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        # state_rgb = []
        # # state_rgb.append(state[:, :, 0:3])
        # state_rgb = np.array(state)
        # state_rgb = np.float32(camera_image.reshape((1, response.height, response.width, 1)))

        #
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img2d = np.reshape(img1d, (response.height, response.width, 3))
        image = Image.fromarray(img2d)
        #image = np.array(image.convert('L'))
        image = np.array(image)
        #cv2.imwrite('view.png', image)
        image = cv2.resize(image, (128, 72))
        #image.resize((128, 72, 3),  refcheck=False)
        # image[image > 50] = 255
        cv2.imshow('frame', image)
        k = cv2.waitKey(1) & 0xff
        image = np.float32(image.reshape((1, 3, 72, 128)))
        image /= 255.0

        return image
    def get_CustomDepth(self):
        max_tries = 5
        tries = 0
        correct = False
        while not correct and tries < max_tries:
            camera_name = 2
            responses = self.client.simGetImages(
                [airsim.ImageRequest(camera_name, airsim.ImageType.DepthVis, False, False)])
            while responses[0].height == 0:
                print('depth stuck!')
                responses = self.client.simGetImages(
                    [airsim.ImageRequest(camera_name, airsim.ImageType.DepthVis, False, False)])
                self.client.reset()
            img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
            if np.max(img1d)==255 and np.mean(img1d)<0.05:
                correct = False
            else:
                correct = True
        depth = img1d.reshape(responses[0].height, responses[0].width, 3)[:, :, 0]
        thresh = 50

        # To make sure the wall leaks in the unreal environment doesn't mess up with the reward function
        super_threshold_indices = depth > thresh
        depth[super_threshold_indices] = thresh
        depth = depth / thresh
        return depth, thresh

    def avg_depth(self, depth_map1, thresh):
        depth_map = depth_map1
        global_depth = np.mean(depth_map)
        n = max(global_depth * thresh / 3, 1)
        H = np.size(depth_map, 0)
        W = np.size(depth_map, 1)
        grid_size = (np.array([H, W]) / n)

        # scale by 0.9 to select the window towards top from the mid line
        h = max(int(0.9 * H * (n - 1) / (2 * n)), 0)
        w = max(int(W * (n - 1) / (2 * n)), 0)
        grid_location = [h, w]

        x_start = int(round(grid_location[0]))
        y_start_center = int(round(grid_location[1]))
        x_end = int(round(grid_location[0] + grid_size[0]))
        y_start_right = min(int(round(grid_location[1] + grid_size[1])), W)
        y_start_left = max(int(round(grid_location[1] - grid_size[1])), 0)
        y_end_right = min(int(round(grid_location[1] + 2 * grid_size[1])), W)

        fract_min = 0.05

        L_map = depth_map[x_start:x_end, y_start_left:y_start_center]
        C_map = depth_map[x_start:x_end, y_start_center:y_start_right]
        R_map = depth_map[x_start:x_end, y_start_right:y_end_right]

        if not L_map.any():
            L1 = 0
        else:
            L_sort = np.sort(L_map.flatten())
            end_ind = int(np.round(fract_min * len(L_sort)))
            L1 = np.mean(L_sort[end_ind])

        if not R_map.any():
            R1 = 0
        else:
            R_sort = np.sort(R_map.flatten())
            end_ind = int(np.round(fract_min * len(R_sort)))
            R1 = np.mean(R_sort[end_ind])

        if not C_map.any():
            C1 = 0
        else:
            C_sort = np.sort(C_map.flatten())
            end_ind = int(np.round(fract_min * len(C_sort)))
            C1 = np.mean(C_sort[end_ind])

        return L1, C1, R1
def v2t(vect):
    if isinstance(vect, airsim.Vector3r):
        res = np.array([vect.x_val, vect.y_val, vect.z_val])
    else:
        res = np.array(vect)
    return res

def distance(pos1, pos2):
    pos1 = v2t(pos1)
    pos2 = v2t(pos2)
    # dist = np.sqrt(abs(pos1[0]-pos2[0])**2 + abs(pos1[1]-pos2[1])**2 + abs(pos1[2]-pos2[2]) **2)
    dist = np.linalg.norm(pos1 - pos2)

    return dist