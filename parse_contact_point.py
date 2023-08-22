import os
import glob
import json
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tqdm
import copy


def read_jsons(result_dir):
    json_files = glob.glob(os.path.join(result_dir, '*.json'))
    json_files.sort(key = lambda x: int(os.path.basename(x).split('.')[0]))
    # json_files = json_files[-10:]
    image_files = [os.path.basename(file).replace(".json", ".jpg") for file in json_files]
    detections = []
    detections_dict = {}
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f) # data is a dict with key "hand_det" and "obj_det"
            detections.append(data)
            detections_dict[os.path.basename(json_file).split(".")[0]] = data
    json.dump(detections_dict, open(f"{result_dir}_merged.json", "w"))
    return detections, image_files


def parse_img_files(image_dir, image_files):
    img_files = []
    for image_file in tqdm.tqdm(image_files, desc="read images"):
        image_path = os.path.join(image_dir, image_file)
        # image = cv2.imread(image_path)
        img_files.append(image_path)
    return img_files


def _calculate_center(bb):
    return [(bb[0] + bb[2])/2, (bb[1] + bb[3])/2]


def _filter_object(obj_dets, hand_dets):
    filtered_object = []
    object_cc_list = []
    for j in range(obj_dets.shape[0]):
        object_cc_list.append(_calculate_center(obj_dets[j,:4]))
    object_cc_list = np.array(object_cc_list)
    img_obj_id = []
    for i in range(hand_dets.shape[0]):
        if hand_dets[i, 5] <= 0:
            img_obj_id.append(-1)
            continue
        hand_cc = np.array(_calculate_center(hand_dets[i,:4]))
        point_cc = np.array([(hand_cc[0]+hand_dets[i,6]*10000*hand_dets[i,7]), (hand_cc[1]+hand_dets[i,6]*10000*hand_dets[i,8])])
        dist = np.sum((object_cc_list - point_cc)**2,axis=1)
        dist_min = np.argmin(dist)
        img_obj_id.append(dist_min)
    return img_obj_id


def skin_extract(image):
    # image = cv2.imread("/mnt/cephfs/home/zhihongyan/llm4embodied/hand_object_detector/images_det/21127_det.png")
    def color_segmentation():
        lower_HSV_values = np.array([0, 40, 0], dtype="uint8")
        upper_HSV_values = np.array([25, 255, 255], dtype="uint8")
        lower_YCbCr_values = np.array((0, 138, 67), dtype="uint8")
        upper_YCbCr_values = np.array((255, 173, 133), dtype="uint8")
        mask_YCbCr = cv2.inRange(YCbCr_image, lower_YCbCr_values, upper_YCbCr_values)
        mask_HSV = cv2.inRange(HSV_image, lower_HSV_values, upper_HSV_values)
        binary_mask_image = cv2.add(mask_HSV, mask_YCbCr)
        return binary_mask_image

    HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    YCbCr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    binary_mask_image = color_segmentation()
    image_foreground = cv2.erode(binary_mask_image, None, iterations=3)
    dilated_binary_image = cv2.dilate(binary_mask_image, None, iterations=3)
    ret, image_background = cv2.threshold(dilated_binary_image, 1, 128, cv2.THRESH_BINARY)

    image_marker = cv2.add(image_foreground, image_background)
    image_marker32 = np.int32(image_marker)
    cv2.watershed(image, image_marker32)
    m = cv2.convertScaleAbs(image_marker32)
    ret, image_mask = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((20, 20), np.uint8)
    image_mask = cv2.morphologyEx(image_mask, cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite("./test.png", image_mask)
    return image_mask


def _contact_point(obj_dets, hand_det, img):
    contact_point = None
    if obj_dets is not None and hand_det is not None:
        contact_state = hand_det[5]
        if contact_state == 3 or contact_state == 4:
            # bbox:[x0,y0,x1,y1]
            hand_bbox = hand_det[:4]
            mask_img = copy.deepcopy(img)
            mask_img[0:int(hand_bbox[1]),:]=(0,0,0)
            mask_img[int(hand_bbox[3]):,:]=(0,0,0)
            mask_img[:,0:int(hand_bbox[0])]=(0,0,0)
            mask_img[:,int(hand_bbox[2]):]=(0,0,0)
            # 0:none 255:hand 
            mask_hand = skin_extract(mask_img)
            mask_indexs = np.where(mask_hand==255)

            # img_obj_id = filter_object(obj_dets, hand_det)
            obj_bbox = obj_dets[0][:4]  # TODO select the most possible obj
            valid_contact_point_x = []
            valid_contact_point_y = []
            # mask_obj_img = copy.deepcopy(origin_image)
            # mask_obj_img[int(obj_bbox[2]):int(obj_bbox[3]),int(obj_bbox[0]):int(obj_bbox[1])] = 1
            for y,x in zip(mask_indexs[0], mask_indexs[1]):
                if (int(obj_bbox[1])<= y and y <= int(obj_bbox[3])) and \
                    (int(obj_bbox[0])<= x and x <= int(obj_bbox[2])):
                    valid_contact_point_x.append(x)
                    valid_contact_point_y.append(y)
            # 平均所有的有效的点
            if len(valid_contact_point_x) == 0 or len(valid_contact_point_y) == 0:
                contact_point = None
            else:
                contact_point = [sum(valid_contact_point_y)/len(valid_contact_point_y), sum(valid_contact_point_x)/len(valid_contact_point_x)]
    return contact_point


def savgol_filter(contact_state_cache, window_size=7, polyorder=3, vis=True):
    '''
    polyorder:多项式拟合系数，越大越接近原曲线，越小越平滑
    '''
    filter_contact_state_cache_l = scipy.signal.savgol_filter(contact_state_cache['L'], window_size, polyorder)
    filter_contact_state_cache_r = scipy.signal.savgol_filter(contact_state_cache['R'], window_size, polyorder)
    filter_contact_state_cache = {'L':filter_contact_state_cache_l, 'R':filter_contact_state_cache_r}
    if vis:
        plt.figure()
        x = np.linspace(1, len(contact_state_cache['L']) ,len(contact_state_cache['L']))
        plt.plot(x, np.array(filter_contact_state_cache_l), 'r', label = 'l_savgol_filter')
        plt.plot(x, np.array(contact_state_cache['L']), label = 'l_origin')
        plt.legend()
        plt.savefig("./state_left.png")

        plt.figure()
        x = np.linspace(1, len(contact_state_cache['R']) ,len(contact_state_cache['R']))
        plt.plot(x, np.array(filter_contact_state_cache_r), 'b', label = 'r_savgol_filter')
        plt.plot(x, np.array(contact_state_cache['R']), label = 'r_origin')
        plt.legend()
        plt.savefig("./state_right.png")
    return filter_contact_state_cache


def get_homography(img1, img2):
    img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    akaze_detector = cv2.AKAZE_create()
    kp1, desc1 = akaze_detector.detectAndCompute(img1_grey, mask=None)
    kp2, desc2 = akaze_detector.detectAndCompute(img2_grey, mask=None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m,n in matches:
        if m.distance < 0.4 * n.distance:
            good_matches.append(m)

    pt1 = np.zeros((len(good_matches), 2))
    pt2 = np.zeros((len(good_matches), 2))
    for i, p in enumerate(good_matches):
        pt1[i,:] = kp1[p.queryIdx].pt
        pt2[i,:] = kp2[p.trainIdx].pt

    if len(good_matches) >= 4:
        h, mask = cv2.findHomography(pt1, pt2, cv2.RANSAC)
    else:
        h, mask = None, None
        print(f"The match points are less than 4, cannot calculate homography matrix!")

    # H,W,C = img1.shape
    # img_warp = cv2.warpPerspective(img1, h, (W, H))
    return h


def parse_contact_state(detections):
    """
     detections: list of dict with key "hand_det" and "obj_det" in multiple frames
    """
    num_frame = len(detections)
    contact_state_cache = {'L':[0]*num_frame, 'R':[0]*num_frame}
    
    for index_frame, detection in enumerate(tqdm.tqdm(detections, desc="parse contact state")):    
        hand_dets = detection['hand_det']
        obj_dets = detection['obj_det']
        log_left_hand_flag, log_right_hand_flag = False, False
        if hand_dets is None:
            continue
        
        for i in range(len(hand_dets)):
            # step1:处理contac状态（3:接触可移动obj 4:接触不可移动obj 0:没有接触）
            contact_state = hand_dets[i][5]
            contact_hand = hand_dets[i][-1] # 0:左手 1:右手
            
            # left hand
            if contact_hand == 0 and log_left_hand_flag==False:
                log_left_hand_flag = True
                if contact_state == 3 or contact_state == 4:    # 和obj接触
                    contact_state_cache['L'][index_frame] = 1
                else:                                           # 没有和obj接触
                    contact_state_cache['L'][index_frame] = 0
            # right hand     
            elif contact_hand == 1 and log_right_hand_flag==False:
                log_right_hand_flag = True
                if contact_state == 3 or contact_state == 4:    # 和obj接触
                    contact_state_cache['R'][index_frame] = 1
                else:                                           # 没有和obj接触
                    contact_state_cache['R'][index_frame] = 0

        # step2: avgpooling
        # contact_state_cache = savgol_filter(contact_state_cache, window_size=7, polyorder=3, vis=True)

           
    return contact_state_cache


def parse_contact_point(contact_state_cache, detections, img_files):
    num_frame = len(contact_state_cache['L'])
    contact_point_cache = {
        "L" : [None] * num_frame,
        "R" : [None] * num_frame
    }

    for hand in ['L', 'R']:
        hand_indicator = 0 if hand == 'L' else 1
        contact_state = contact_state_cache[hand]
        for index_frame, state in enumerate(tqdm.tqdm(contact_state, desc="parse contact point")):
            if state >= 0.8:
                hand_dets = detections[index_frame]['hand_det']
                obj_dets = detections[index_frame]['obj_det']
                hand_det = [hand for hand in hand_dets if hand[-1] == hand_indicator][0]    # select the first left or right hand
                img = cv2.imread(img_files[index_frame])
                contact_point = _contact_point(obj_dets, hand_det, img)
                contact_point_cache[hand][index_frame] = contact_point

    return contact_point_cache


def parse_homography(img_files):
    """
    img_homography_cache[ind] indicates the homography matrix from img_files[ind+1] to img_files[ind]
    """
    img_homography_cache = [None] * len(img_files)
    curr_img = cv2.imread(img_files[0])
    for img_index in tqdm.tqdm(range(len(img_files)-1), desc="parse transformation"):
        next_img = cv2.imread(img_files[img_index+1])
        homography_matrix = get_homography(next_img, curr_img)
        homography_matrix = homography_matrix.tolist() if homography_matrix is not None else None
        img_homography_cache[img_index] = homography_matrix
        curr_img = next_img
    return img_homography_cache


if __name__ == "__main__":
    detected_videos = os.listdir('/home/winnie/chenpeihao/Projects/hand_object_detector/results/EPIC_handobj_frame')
    detected_videos.sort(reverse=False)
    print(detected_videos)
    for detected_video in detected_videos:
        print(f"processing {detected_video}")
        detection_dir = f'/home/winnie/chenpeihao/Projects/hand_object_detector/results/EPIC_handobj_frame/{detected_video}'
        image_dir = f'/home/winnie/AICD/AICDzhnf/data/EPIC_clip/{detected_video}'
        if not os.path.isdir(detection_dir):
            continue

        detections, image_files = read_jsons(detection_dir)
        img_files = parse_img_files(image_dir, image_files)

        contact_state_cache = parse_contact_state(detections)
        contact_point_cache = parse_contact_point(contact_state_cache, detections, img_files)
        print(contact_state_cache)
        print(contact_point_cache)
        img_homography_cache = parse_homography(img_files)
        result = {"contact_states": contact_state_cache, "contact_points": contact_point_cache, "img_homography": img_homography_cache}
        json.dump(result, open(f"results/EPIC_handobj_video/{detected_video}.json", 'w'))