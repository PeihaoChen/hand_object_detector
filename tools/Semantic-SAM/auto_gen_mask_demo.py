from semantic_sam import prepare_image, plot_results, build_semantic_sam, SemanticSamAutomaticMaskGenerator
import numpy as np
import matplotlib.pyplot as plt
import json

def pack_mask_into_json(masks):
    json_img = np.zeros_like(masks[0]['segmentation']).astype(np.uint8)
    bbox_list = []
    for m,mask in enumerate(masks):
        json_img[mask['segmentation']] = m
        bbox_list.append(mask['bbox'])
    # plt.imsave("./vis/mask.png", json_img)
    op_dict = {'mask_img' : json_img.tolist(),
               'bbox' : bbox_list}
    with open('CACHE/temp.json',"w") as f:
        json.dump(op_dict, f)    
    


original_image, input_image = prepare_image(image_pth='CACHE/temp.png')  # change the image path to your image

mask_generator = SemanticSamAutomaticMaskGenerator(build_semantic_sam(model_type='L', ckpt='model/swinl_only_sam_many2many.pth')) # model_type: 'L' / 'T', depends on your checkpint
masks = mask_generator.generate(input_image)

pack_mask_into_json(masks)
# plot_results(masks, original_image, save_path='./vis/')  # results and original images will be saved at save_path


