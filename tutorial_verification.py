from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2
import os
import glob
# Specify the path to model config and checkpoint file

############# Faster R-CNN #############
# config_file = 'configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py'
# checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'
config_file = 'configs/triage_config/triage_config.py'
checkpoint_file = 'work_dirs/triage1600/epoch_7.pth'
########################################

##### triage #####
# config_file = 'configs/triage_config/triage_config.py'
# checkpoint_file = 'work_dirs/triage/latest.pth'
##################

##### person #####
# config_file = 'configs/triage_config/person_config.py'
# checkpoint_file = 'work_dirs/person_config/latest.pth'
##################


# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
# img = 'demo/demo.jpg'  # or img = mmcv.imread(img), which will only load it once
# img = 'our_dataset/official_example_crop/set01_drone03_triage012.jpg'  # or img = mmcv.imread(img), which will only load it once

im_folder = 'our_dataset/set01_test/test/'
# im_folder = 'our_dataset/official_example_crop/'
# im_folder = 'data/our_dataset/triage/onlyhandmade/1023/test/'
imges = []
# for filename in os.listdir(im_folder):
i = 0
# import pdb; pdb.set_trace()
for filename in glob.glob(im_folder + '*.jpg'):
    # post-processing
    result = inference_detector(model, filename) # person
    # result2 = inference_detector(model2, filename) # triage
    # model.show_result(filename, result, out_file='demo/our_dataset/onlyhandmade/triage1023/' + str(i) +".jpg")
    model.show_result(filename, result, out_file='demo/test_epoch7/' + str(i) +".jpg")
    # model.show_result(filename, result, out_file='demo/our_dataset/person_faster_rcnn/onlyperson/finetune/' + str(i) +".jpg")
    i +=1 

# result = inference_detector(model, imges)
# # visualize the results in a new window
# # model.show_result(img, result)
# # or save the visualization results to image files
# model.show_result(imges, result, out_file='demo/our_dataset/onlyhandmade/triage1023/')

# # test a video and show the results
# video = mmcv.VideoReader('demo/demo.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     model.show_result(frame, result, wait_time=1)