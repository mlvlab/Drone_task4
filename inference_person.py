from mmdet.apis import init_detector, inference_detector
# inference_detector_twoStage
import mmcv
import cv2
import os
import glob


##### person #####
#config_file_person = 'configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py'
#checkpoint_file_person = 'checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'
config_file_person = 'configs/triage_config/person_finetune_config2.py'
checkpoint_file_person = 'work_dirs/person_finetune/latest.pth'
##################




# build the model from a config file and a checkpoint file
model_person = init_detector(config_file_person, checkpoint_file_person, device='cuda:0')

im_folder = 'our_dataset/set01_test/test/'

j=0
person_results = []
#### Extract Region of Person ####
for filename in glob.glob(im_folder + '*.jpg'):
    result_person = inference_detector(model_person, filename) # person
    model_person.show_result(filename, result_person, out_file='demo/triage_version_1/person/' + str(j) +".jpg")
    #model_person.show_result(filename, result_person, out_file='demo/personfinetune/latestpth/' + str(j) +".jpg")
    j +=1 
    person_results.append(result_person)
