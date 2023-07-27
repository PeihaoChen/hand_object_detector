# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import tqdm
import os
import sys
import numpy as np
import json
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from PIL import Image

import torchvision.transforms as transforms
import torchvision.datasets as dset
# from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_PIL, vis_detections_filtered_objects_PIL, vis_detections_filtered_objects # (1) here add a function to viz
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/res101.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',
                      default="models")
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default="images")
  parser.add_argument('--mp4',
                    action='store_true')
  parser.add_argument('--debug',
                    action='store_true')
  parser.add_argument('--mp4_file',
                    help='directory to load video for demo',
                    default="data/video/e19b1be3-8ee9-44cc-ae7d-99a3591ebba5.mp4")
  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save results',
                      default="images_det")
  parser.add_argument('--cuda', dest='cuda', 
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=8, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=89999, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      default=True)
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=-1, type=int)
  parser.add_argument('--thresh_hand',
                      type=float, default=0.5,
                      required=False)
  parser.add_argument('--thresh_obj', default=0.5,
                      type=float,
                      required=False)

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)


# 提取跟物体第一帧算法实现-------------------------------------#
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import glob
import copy
class First_Contact_Extract():
   
  def __init__(self) -> None:
     # (state:[0,1], bbox:[x,y])
     self.contact_state_cache = {"L":[], "R":[]}
  
  # def load_dataset(self, dataset_path='./images_det/*.json'):
  #   #FIXME
  #   print("Loading Dataset")
  #   info_datas = glob.glob(dataset_path)
  # #  print(info_datas)
  #   for data in tqdm.tqdm(info_datas, desc="Loading Dataset"):
  #       try:
  #           info = json.load(open(data,"r"))
  #           L_state = 1 if info['hand_det'][0][5] > 0 else 0
  #           R_state = 1 if info['hand_det'][1][5] > 0 else 0
  #       except:
  #           continue
  #           self.contact_state_cache['L'].append(L_state)
  #           self.contact_state_cache['R'].append(R_state)
  #           # print(R_state)


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

  def do_savgol_filter(self, window_size=7, polyorder=6, vis=False):
     '''
        polyorder:多项式拟合系数，越小越接近原曲线，越大越平滑
     '''
     filter_contact_state_cache_l = scipy.signal.savgol_filter(self.contact_state_cache['L'], window_size, polyorder)
     filter_contact_state_cache_r = scipy.signal.savgol_filter(self.contact_state_cache['R'], window_size, polyorder)
     if vis:
        x = np.linspace(1, len(self.contact_state_cache['L']) ,len(self.contact_state_cache['L']))
        plt.plot(x, np.array(filter_contact_state_cache_l), 'r', label = 'l_savgol_filter')
        plt.plot(x, np.array(self.contact_state_cache['L']), label = 'l_origin')
        
        x = np.linspace(1, len(self.contact_state_cache['R']) ,len(self.contact_state_cache['R']))
        plt.plot(x, np.array(filter_contact_state_cache_r), 'b', label = 'r_savgol_filter')
        plt.plot(x, np.array(self.contact_state_cache['R']), label = 'r_origin')
        
        plt.legend()
        plt.savefig("./state.png")
     return filter_contact_state_cache_l,filter_contact_state_cache_r

  def record_contact_state_into_image(self, filter_contact_state_cache_l, filter_contact_state_cache_r, image_path="./images_det/*_contact.png"):
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
      
    pass
       

  def log_contact_state(self, hand_dets, obj_dets, origin_image):
     '''
        hand_dets: [boxes(4), score(1), state(1), offset_vector(3), left/right(1)]
        obj_dets: [boxes(4), score(1), state(1), offset_vector(3), left/right(1)]
     '''
     log_left_hand_flag = False
     log_right_hand_flag = False
     contact_img = copy.deepcopy(origin_image)
     
     if not hand_dets is None: 
      for i in range(len(hand_dets)):
        # step1:处理contac状态---------------------------------#
        # 3:接触可移动obj 4:没有接触可移动obj 0:没有接触
        contact_state = hand_dets[i][5]
        # 0:左手 1:右手
        contact_hand = hand_dets[i][-1]
        # 处理左手
        if contact_hand == 0 and log_left_hand_flag==False:
          log_left_hand_flag = True
          # 没有和obj接触
          if contact_state == 4 or contact_state == 0:
            self.contact_state_cache['L'].append(0)
          # 和obj接触
          else:
            self.contact_state_cache['L'].append(1)
            # # 整个视频第一帧
            # if len(self.contact_state_cache['L']) == 0 :
            #   self.contact_state_cache['L'].append(1)
            # # 上一帧不是第一帧接触
            # elif self.contact_state_cache['L'][-1] == 0:
            #   self.contact_state_cache['L'].append(1)
            # # 上一帧是第一帧接触
            # else:
            #   self.contact_state_cache['L'].append(0)
        # 处理右手      
        elif log_right_hand_flag==False:
          log_right_hand_flag = True
          # 没有和obj接触
          if contact_state == 4 or contact_state == 0:
            self.contact_state_cache['R'].append(0)
          # 和obj接触
          else:
            self.contact_state_cache['R'].append(1)
            # # 整个视频第一帧
            # if len(self.contact_state_cache['R']) == 0 :
            #   self.contact_state_cache['R'].append(1)
            # # 上一帧不是第一帧接触
            # elif self.contact_state_cache['R'][-1] == 0:
            #   self.contact_state_cache['R'].append(1)
            # # 上一帧是第一帧接触
            # else:
            #   self.contact_state_cache['R'].append(0)
          
        
        # step2:处理hand的bbox返回肤色的像素位置-------------------------#
        # 如果有接触
        if not obj_dets is None and not hand_dets is None:
          if contact_state == 1 or contact_state == 3:
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
            avg_y, avg_x = sum(valid_contact_point_y)/len(valid_contact_point_y), sum(valid_contact_point_x)/len(valid_contact_point_x)
            
            cv2.circle(contact_img, (int(avg_x), int(avg_y)), 10, (0,255,0), -1)
            pass
          
      if not log_left_hand_flag:
          self.contact_state_cache['L'].append(0)
      if not log_right_hand_flag:
          self.contact_state_cache['R'].append(0)       
            
      return contact_img
    
     else:
      self.contact_state_cache['L'].append(0)
      self.contact_state_cache['R'].append(0)
      return np.zeros_like(origin_image)



if __name__ == '__main__':

  args = parse_args()

  # print('Called with args:')
  # print(args)

  extractor = First_Contact_Extract()

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  cfg.USE_GPU_NMS = args.cuda
  np.random.seed(cfg.RNG_SEED)

  # load model
  model_dir = args.load_dir + "/" + args.net + "_handobj_100K" + "/" + args.dataset
  if not os.path.exists(model_dir):
    raise Exception('There is no input directory for loading network from ' + model_dir)
  load_name = os.path.join(model_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  pascal_classes = np.asarray(['__background__', 'targetobject', 'hand']) 
  args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]'] 

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  if args.cuda > 0:
    checkpoint = torch.load(load_name)
  else:
    checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']

  print('load model successfully!')


  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)
  box_info = torch.FloatTensor(1) 

  # ship to cuda
  if args.cuda > 0:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  with torch.no_grad():
    if args.cuda > 0:
      cfg.CUDA = True

    if args.cuda > 0:
      fasterRCNN.cuda()

    fasterRCNN.eval()

    start = time.time()
    max_per_image = 100
    thresh_hand = args.thresh_hand 
    thresh_obj = args.thresh_obj
    vis = args.vis

    # print(f'thresh_hand = {thresh_hand}')
    # print(f'thnres_obj = {thresh_obj}')

    webcam_num = args.webcam_num
    # Set up webcam or get image directories
    if args.mp4:
      cap = cv2.VideoCapture(args.mp4_file)
      imglist = []
      fps = int(cap.get(cv2.CAP_PROP_FPS))
      num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
      print(num_frames)
      print(fps)
      n = 0
      if args.debug:
        num_frames = 3000
      lpbar = tqdm.tqdm(total=int(num_frames), desc="load mp4")
      while n < num_frames:
          #lpbar.update(1)
          success, image = cap.read()
          imglist.append(image)
          lpbar.update(1)
          n+=1
      num_images = int(len(imglist))-1
    elif webcam_num >= 0 :
      cap = cv2.VideoCapture(webcam_num)
      num_images = 0
    else:
      print(f'image dir = {args.image_dir}')
      print(f'save dir = {args.save_dir}')
      imglist = os.listdir(args.image_dir)
      num_images = len(imglist)

    print('Loaded Photo: {} images.'.format(num_images))


    pbar = tqdm.tqdm(total=num_images)
    while (num_images >= 0):
        pbar.update(1)
        total_tic = time.time()
        if webcam_num == -1:
          num_images -= 1

        # Get image from the webcam
        if webcam_num >= 0 :
          if not cap.isOpened():
            raise RuntimeError("Webcam could not open. Please check connection.")
          ret, frame = cap.read()
          im_in = np.array(frame)
        elif args.mp4:
          im_in = imglist[num_images]
        # Load the demo image
        else:
          im_file = os.path.join(args.image_dir, imglist[num_images])
          im_in = cv2.imread(im_file)
        # bgr
        im = im_in

        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        with torch.no_grad():
                im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
                im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
                gt_boxes.resize_(1, 1, 5).zero_()
                num_boxes.resize_(1).zero_()
                box_info.resize_(1, 1, 5).zero_() 

        # pdb.set_trace()
        det_tic = time.time()

        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info) 

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        # extact predicted params
        contact_vector = loss_list[0][0] # hand contact state info
        offset_vector = loss_list[1][0].detach() # offset vector (factored into a unit vector and a magnitude)
        lr_vector = loss_list[2][0].detach() # hand side info (left/right)

        # get hand contact 
        _, contact_indices = torch.max(contact_vector, 2)
        contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

        # get hand side 
        lr = torch.sigmoid(lr_vector) > 0.5
        lr = lr.squeeze(0).float()

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
              if args.class_agnostic:
                  if args.cuda > 0:
                      box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                  else:
                      box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                  box_deltas = box_deltas.view(1, -1, 4)
              else:
                  if args.cuda > 0:
                      box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                  else:
                      box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                  box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im2show = np.copy(im)
        obj_dets, hand_dets = None, None
        for j in xrange(1, len(pascal_classes)):
            # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
            if pascal_classes[j] == 'hand':
              inds = torch.nonzero(scores[:,j]>thresh_hand).view(-1)
            elif pascal_classes[j] == 'targetobject':
              inds = torch.nonzero(scores[:,j]>thresh_obj).view(-1)

            # if there is det
            if inds.numel() > 0:
              cls_scores = scores[:,j][inds]
              _, order = torch.sort(cls_scores, 0, True)
              if args.class_agnostic:
                cls_boxes = pred_boxes[inds, :]
              else:
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
              
              cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
              cls_dets = cls_dets[order]
              keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
              cls_dets = cls_dets[keep.view(-1).long()]
              if pascal_classes[j] == 'targetobject':
                obj_dets = cls_dets.cpu().numpy()
              if pascal_classes[j] == 'hand':
                hand_dets = cls_dets.cpu().numpy()
              
        if vis:
          # visualization
          im2show = vis_detections_filtered_objects_PIL(im2show, obj_dets, hand_dets, thresh_hand, thresh_obj)

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        if webcam_num == -1:
            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                            .format(num_images + 1, len(imglist), detect_time, nms_time))
            sys.stdout.flush()

        if vis and webcam_num == -1:
            
            folder_name = args.save_dir
            os.makedirs(folder_name, exist_ok=True)
            if args.mp4:
              result_path = os.path.join(folder_name, str(num_images) + "_det.png")
              json_path = os.path.join(folder_name, str(num_images) + ".json")
              with open(json_path, "w") as f:
                json.dump({
                            "hand_det":hand_dets.tolist() if hand_dets is not None else "none" ,
                            "obj_det":obj_dets.tolist()  if obj_dets is not None else "none"
                          },
                          f
                          )
              contact_img = extractor.log_contact_state(hand_dets, obj_dets, np.copy(im))
              contact_img_path = os.path.join(folder_name, str(num_images) + "_contact.png")
              cv2.imwrite(contact_img_path, contact_img)
                
              
            else:
              result_path = os.path.join(folder_name, imglist[num_images][:-4] + "_det.png")
            im2show.save(result_path)
            
            
        else:
            im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
            cv2.imshow("frame", im2showRGB)
            total_toc = time.time()
            total_time = total_toc - total_tic
            frame_rate = 1 / total_time
            print('Frame rate:', frame_rate)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    filter_contact_state_cache_l,filter_contact_state_cache_r = extractor.do_savgol_filter()
    extractor.record_contact_state_into_image(filter_contact_state_cache_l, filter_contact_state_cache_r)
              
    if webcam_num >= 0:
        cap.release()
        cv2.destroyAllWindows()
