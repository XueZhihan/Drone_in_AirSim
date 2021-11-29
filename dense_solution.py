import cv2 as cv
import numpy as np
import airsim
from PIL import Image
def dense_op(img_1, img_2):
    #img1 = cv.imread(img_1)
    first_frame = img_1
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(first_frame)
    # Sets image saturation to maximum
    mask[..., 1] = 255
    cv.imshow("input1", first_frame)
    key = cv.waitKey(1) & 0xFF
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    #frame = cv.imread(img_2)
    frame =img_2
    # Opens a new window and displays the input frame
    cv.imshow("input2", frame)
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Calculates dense optical flow by Farneback method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    # Sets image hue according to the optical flow direction
    #mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 1] = 0
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    cv.imshow("output", rgb)

    img = Image.fromarray(rgb)
    img_rgb = img.convert('RGB')
    camera_image_rgb = np.asarray(img_rgb)
    camera_image = camera_image_rgb

    # state = cv.resize(camera_image, (244, 244), cv.INTER_LINEAR)
    state = cv.normalize(camera_image, camera_image, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
    state_rgb = []
    state_rgb.append(state[:, :, 0:3])
    state_rgb = np.array(state_rgb)
    state_rgb = state_rgb.astype('float32')
    return state_rgb
def prepaer_input(response):
    # get numpy array
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
    #png = cv.imdecode(rawImage, cv.IMREAD_UNCHANGED)
    img_rgba = img1d.reshape(response.height, response.width, 3)
    return img_rgba

def dense_op_dis(img_1, img_2):
    #img1 = cv.imread(img_1)
    first_frame = img_1
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(first_frame)
    # Sets image saturation to maximum
    mask[..., 1] = 255
    cv.imshow("input1", first_frame)
    key = cv.waitKey(1) & 0xFF
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    #frame = cv.imread(img_2)
    frame =img_2
    # Opens a new window and displays the input frame
    cv.imshow("input2", frame)
    key = cv.waitKey(1) & 0xFF
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Calculates dense optical flow by Farneback method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    dis = cv.DISOpticalFlow_create(2)
    flow = dis.calc(prev_gray, gray, None, )
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    # Sets image hue according to the optical flow direction
    #mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 1] = 0
    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    cv.imshow("flow", rgb)
    img = Image.fromarray(rgb)
    img_rgb = img.convert('RGB')
    camera_image_rgb = np.asarray(img_rgb)
    camera_image = camera_image_rgb




    #state = cv.resize(camera_image, (244, 244), cv.INTER_LINEAR)
    state = cv.normalize(camera_image, camera_image, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
    state_rgb = []
    state_rgb.append(state[:, :, 0:3])
    state_rgb = np.array(state_rgb)
    state_rgb = state_rgb.astype('float32')

    return state_rgb

def op(img_1, img_2):
    #cap = cv.VideoCapture(0)

    # ShiTomasi 角点检测参数
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # lucas kanade光流法参数
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # 创建随机颜色
    color = np.random.randint(0, 255, (100, 3))

    # 获取第一帧，找到角点
    #ret, old_frame = cap.read()
    # 找到原始灰度图
    old_gray = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)

    # 获取图像中的角点，返回到p0中
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # 创建一个蒙版用来画轨迹
    mask = np.zeros_like(img_1)


    #ret, frame = cap.read()
    frame_gray = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)

    # 计算光流
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # 选取好的跟踪点
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # 画出轨迹
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        img_2 = cv.circle(img_2, (a, b), 5, color[i].tolist(), -1)
    img = cv.add(img_2, mask)

    cv.imshow('frame', img)
    k = cv.waitKey(1) & 0xff

    #img_rgb = img.convert('RGB')
    camera_image_rgb = np.asarray(img)
    camera_image = camera_image_rgb

    # state = cv.resize(camera_image, (244, 244), cv.INTER_LINEAR)
    state = cv.normalize(camera_image, camera_image, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
    state_rgb = []
    state_rgb.append(state[:, :, 0:3])
    state_rgb = np.array(state_rgb)
    state_rgb = state_rgb.astype('float32')
    # # 更新上一帧的图像和追踪点
    # old_gray = frame_gray.copy()
    # p0 = good_new.reshape(-1, 1, 2)
    #
    # cv.destroyAllWindows()
    return state_rgb