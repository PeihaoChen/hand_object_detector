
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import glob
import copy
import cv2
import numpy as np
import json
import os 
import tqdm
import time

# 提取跟物体第一帧算法实现-------------------------------------#
class First_Contact_Extract():
   
  def __init__(self, save_dir) -> None:
     # (state:[0,1], bbox:[x,y])
     self.contact_state_cache = {"L":[], "R":[]}
     self.save_dir = save_dir
     self.SAM = SAM()

  def skin_extract(self, image):
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
  
  def read_jsons(self, result_dir):
    self.contact_state_cache = {"L":[], "R":[]}
    for json_path in glob.glob(result_dir + "/*.json"):
      with open(json_path, 'r') as f:
        data = json.load(f)
    return data

  def do_savgol_filter(self, window_size=7, polyorder=3, vis=True):
     '''
        polyorder:多项式拟合系数，越大越接近原曲线，越小越平滑
     '''
     filter_contact_state_cache_l = scipy.signal.savgol_filter(self.contact_state_cache['L'], window_size, polyorder)
     filter_contact_state_cache_r = scipy.signal.savgol_filter(self.contact_state_cache['R'], window_size, polyorder)
     if vis:
        plt.figure()
        x = np.linspace(1, len(self.contact_state_cache['L']) ,len(self.contact_state_cache['L']))
        plt.plot(x, np.array(filter_contact_state_cache_l), 'r', label = 'l_savgol_filter')
        plt.plot(x, np.array(self.contact_state_cache['L']), label = 'l_origin')
        plt.legend()
        plt.savefig("./state_left.png")

        plt.figure()
        x = np.linspace(1, len(self.contact_state_cache['R']) ,len(self.contact_state_cache['R']))
        plt.plot(x, np.array(filter_contact_state_cache_r), 'b', label = 'r_savgol_filter')
        plt.plot(x, np.array(self.contact_state_cache['R']), label = 'r_origin')
        plt.legend()
        plt.savefig("./state_right.png")
     return filter_contact_state_cache_l,filter_contact_state_cache_r

  def record_contact_state_into_image(self, filter_contact_state_cache_l, filter_contact_state_cache_r):
    state2text_map = {0: 'None', 1: 'first contact', 2: 'contact'}
    new_filter_contact_state_cache_l = []
    new_filter_contact_state_cache_r = []
    for i in range(len(filter_contact_state_cache_l)):
      if i == 0 :
        # 第一帧如果是contact则是first contact
        if filter_contact_state_cache_l[i]:
          new_filter_contact_state_cache_l.append(1)
        else:
          new_filter_contact_state_cache_l.append(0)
      elif filter_contact_state_cache_l[i] >= 0.75 and filter_contact_state_cache_l[i-1] < 0.75:
          new_filter_contact_state_cache_l.append(1)
      elif filter_contact_state_cache_l[i] >= 0.75 and filter_contact_state_cache_l[i-1] >= 0.75:
          new_filter_contact_state_cache_l.append(2)
      else:
          new_filter_contact_state_cache_l.append(0)
    for i in range(len(filter_contact_state_cache_r)):
      if i == 0 :
        # 第一帧如果是contact则是first contact
        if filter_contact_state_cache_r[i]:
          new_filter_contact_state_cache_r.append(1)
        else:
          new_filter_contact_state_cache_r.append(0)
      elif filter_contact_state_cache_r[i] >= 0.75 and filter_contact_state_cache_r[i-1] < 0.75:
          new_filter_contact_state_cache_r.append(1)
      elif filter_contact_state_cache_r[i] >= 0.75 and filter_contact_state_cache_r[i-1] >= 0.75:
          new_filter_contact_state_cache_r.append(2)
      else:
          new_filter_contact_state_cache_r.append(0)
        
    image_path = f"{self.save_dir}/*_contact.png"
    images = glob.glob(image_path)
    images.sort(key=lambda x:int(x.split('/')[-1].split('_')[0]), reverse=True)
    # assert len(images) == len(new_filter_contact_state_cache_r)
    print(len(images))
    print(len(new_filter_contact_state_cache_r))
    print(len(new_filter_contact_state_cache_l))
    for j, img in tqdm.tqdm(enumerate(images), desc="record contact state into image"):
      temp = cv2.imread(img)
      ls = state2text_map[new_filter_contact_state_cache_l[j]]
      cv2.putText(temp, f"Left_hand_state:[{ls}]", (5,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
      rs = state2text_map[new_filter_contact_state_cache_r[j]]
      cv2.putText(temp, f"Right_hand_state:[{rs}]", (5,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
      cv2.imwrite(img, temp)


  def log_contact_state(self, hand_dets, obj_dets, origin_image):
     '''
        hand_dets: [boxes(4), score(1), state(1), offset_vector(3), left/right(1)]
        obj_dets: [boxes(4), score(1), state(1), offset_vector(3), left/right(1)]
     '''
     log_left_hand_flag = False
     log_right_hand_flag = False
     contact_point = {
      "left": None,
      "right": None 
     }
    #  contact_img = copy.deepcopy(origin_image)
     
     if not hand_dets is None: 
         
      # # 使用sam去补偿没有检测到的obj bbox
      # if obj_dets is None:
      #     # TODO:offset怎么用呢
      #     obj_center = None
      #     sam_obj_mask = self.SAM.call_sam_to_general_mask(origin_image)
      #     sam_obj_bbox = self.SAM.match_obj(sam_obj_mask)
      #     obj_dets = [sam_obj_bbox,0,0,0,0,0,0]
      #     pass
          
             
      for i in range(len(hand_dets)):
        # step1:处理contac状态---------------------------------#
        # 3:接触可移动obj 4:接触不可移动obj 0:没有接触
        contact_state = hand_dets[i][5]
        # 0:左手 1:右手
        contact_hand = "left" if hand_dets[i][-1]==0 else "right"
        # 处理左手
        if contact_hand == "left" and log_left_hand_flag==False:
          log_left_hand_flag = True
          # 和obj接触
          if contact_state == 3 or contact_state == 4:
            self.contact_state_cache['L'].append(1)
          # 没有和obj接触
          else:
            self.contact_state_cache['L'].append(0)
        # 处理右手      
        elif contact_hand == "right" and log_right_hand_flag==False:
          log_right_hand_flag = True
          # 和obj接触
          if contact_state == 3 or contact_state == 4:
            self.contact_state_cache['R'].append(1)
          # 没有和obj接触
          else:
            self.contact_state_cache['R'].append(0)
          
        # step2:处理hand的bbox返回肤色的像素位置-------------------------#
        # 如果有接触
        if not obj_dets is None and not hand_dets is None:
          if contact_state == 3 or contact_state == 4:
            # bbox:[x0,y0,x1,y1]
            hand_bbox = hand_dets[i][:4]
            mask_img = copy.deepcopy(origin_image)
            mask_img[0:int(hand_bbox[1]),:]=(0,0,0)
            mask_img[int(hand_bbox[3]):,:]=(0,0,0)
            mask_img[:,0:int(hand_bbox[0])]=(0,0,0)
            mask_img[:,int(hand_bbox[2]):]=(0,0,0)
            # 0:none 255:hand 
            mask_hand = self.skin_extract(mask_img)
            mask_indexs = np.where(mask_hand==255)
            
            obj_bbox = obj_dets[0][:4]
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
              continue
            contact_point[contact_hand] = sum(valid_contact_point_y)/len(valid_contact_point_y), sum(valid_contact_point_x)/len(valid_contact_point_x)

            # cv2.circle(contact_img, (int(avg_x), int(avg_y)), 10, (0,255,0), -1)
            # pass
          
      if not log_left_hand_flag:
          self.contact_state_cache['L'].append(0)
      if not log_right_hand_flag:
          self.contact_state_cache['R'].append(0)       
            
      return contact_point
    
     else:
      self.contact_state_cache['L'].append(0)
      self.contact_state_cache['R'].append(0)
      # return np.zeros_like(origin_image)
      return contact_point

  def draw_contact_point(self, contact_point, origin_image):
    contact_img = copy.deepcopy(origin_image)
    for hand in ["right", "left"]:
      point = contact_point[hand]
      if point is not None:    
        cv2.circle(contact_img, (int(point[1]), int(point[0])), 10, (0,255,0), -1)
    return contact_img  


class SAM:
    
    def __init__(self) -> None:
       self.CACHE_DIR = "tools/Semantic-SAM/CACHE"
       
    def call_sam_to_general_mask(self, img):
        if not os.path.exists(self.CACHE_DIR):
            os.makedirs(self.CACHE_DIR)
        cv2.imwrite(f"{self.CACHE_DIR}/temp.png", img)
        st = time.time()
        os.system("cd tools/Semantic-SAM; /mnt/cephfs/home/zhihongyan/anaconda3/envs/sam/bin/python auto_gen_mask_demo.py")
        print(f"Finish Call SAM In {(time.time()-st):.2f} Seconds")
        
        mask_json  = json.load(open(f'{self.CACHE_DIR}/temp.json',"r"))
        self.IMG_SHAPE = np.array(mask_json['mask_img']).shape
        os.system(f"rm -r {self.CACHA_DIR}")
        print(f"Clean SAM Cache")
        
        return mask_json
    
    def match_obj(self, mask_info, obj_center):
        '''
            y行 x列
            obj_center[y,x]
        '''
        mask_img = np.array(mask_info['mask_img'])
        vis_img = np.zeros((mask_img.shape[0], mask_img.shape[1],3)).astype(np.uint8)
        center_list = []
        for id in range(max(mask_img)-1):
            ys, xs = np.where(mask_img==(id+1))
            if not len(ys) == 0:
                center_list.append([[int(sum(ys)/len(ys)), int(sum(xs)/len(xs))],id])
            else:
                center_list.append([[0, 0],id])
        center_list.sort(key=lambda x:np.linalg.norm((x[0][0]-obj_center[0], x[0][1]-obj_center[1])))
        cloest_mask_id = center_list[0][1]
        cloest_obj_pt_ys, cloest_obj_pt_xs= np.where(mask_img==(cloest_mask_id+1))
        
        # 返回x0y0 x1y1的格式
        bbox =  [min(cloest_obj_pt_xs),min(cloest_obj_pt_ys),max(cloest_obj_pt_xs),max(cloest_obj_pt_ys)]
        # cv2.rectangle(vis_img, (bbox['bbox'][0], bbox['bbox'][1]), (bbox['bbox'][2], bbox['bbox'][3]), (0,255,0), 2)
        # cv2.circle(vis_img, (obj_center[1], obj_center[0]), 10, (0,255,255), -1)
        return bbox, [cloest_obj_pt_ys, cloest_obj_pt_ys]
    
    def find_obj_findContours(self, obj_mask_pts):
        mask = np.zeros(self.IMG_SHAPE)
        for y,x in zip(obj_mask_pts):
            mask[y,x] = 1
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        mask = cv2.drawContours(mask,contours,-1,(0,255,0),5) 
        
        return contours
    
    def get_intersection_pt(self, cont1, cont2):
        pts = cv2.bitwise_and(cont1, cont2)