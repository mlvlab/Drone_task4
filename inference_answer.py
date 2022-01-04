from mmdet.apis import init_detector, inference_detector
# inference_detector_twoStage
import mmcv
import cv2
import os
import glob


##### person #####
# config_file_person = 'configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py'
# checkpoint_file_person = 'checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'

##################

##### triage #####
# config_file_triage = 'configs/triage_config/triage_config.py'
config_file_triage = 'configs/triage_config/triage_config_deformable_detr.py'


##################

im_folder = 'our_dataset/set01_test/test/'
# im_folder = 'our_dataset/official_example_crop/'

# j=0
# person_results = []
# #### Extract Region of Person ####
# for filename in glob.glob(im_folder + '*.jpg'):
#     result_person = inference_detector(model_person, filename) # person
#     # model_person.show_result(filename, result_person, out_file='demo/triage_version_1/person/' + str(j) +".jpg")
#     model_person.show_result(filename, result_person, score_thr=0.2, out_file='demo/demothreshold/0_2/person/' + str(j) +".jpg")
#     #model_person.show_result(filename, result_person, out_file='demo/personfinetune/latestpth/' + str(j) +".jpg")
#     j +=1 
#     person_results.append(result_person)

# import pdb; pdb.set_trace()
# i = 1
triage_results = []
for filename in glob.glob(im_folder + '*.jpg'): # 'our_dataset/set01_test/test/set01_drone01_triage01_jpg.rf.390020b4f83de586a1e6adfa66f3e8fa.jpg'
   # post-processing
   # result_person = inference_detector(model_triage, filename) # person
    range_num = 50
    if filename == 'our_dataset/set01_test/test/set01_drone01_triage01_jpg.rf.390020b4f83de586a1e6adfa66f3e8fa.jpg':
        for epoch in range(1,range_num+1):
            checkpoint_file_triage = 'work_dirs/deformableDETR/epoch_' + str(epoch) + '.pth'
            model_triage = init_detector(config_file_triage, checkpoint_file_triage, device='cuda:0')
            result_triage = inference_detector(model_triage, filename)
            model_triage.show_result(filename, result_triage, out_file='demo/deformableDETR/set01_drone01_triage01/' + str(epoch) +".jpg")

    if filename == 'our_dataset/set01_test/test/set01_drone01_triage02_jpg.rf.83bfc9fde2918b7719eb6f79799c4e56.jpg':
        for epoch in range(1,range_num+1):
            checkpoint_file_triage = 'work_dirs/deformableDETR/epoch_' + str(epoch) + '.pth'
            model_triage = init_detector(config_file_triage, checkpoint_file_triage, device='cuda:0')
            result_triage = inference_detector(model_triage, filename)
            model_triage.show_result(filename, result_triage, out_file='demo/deformableDETR/set01_drone01_triage02/' + str(epoch) +".jpg")

    if filename == 'our_dataset/set01_test/test/set01_drone02_triage01_jpg.rf.5e85604fbfd68a1d58929187ddf896e6.jpg':
        for epoch in range(1,range_num+1):
            checkpoint_file_triage = 'work_dirs/deformableDETR/epoch_' + str(epoch) + '.pth'
            model_triage = init_detector(config_file_triage, checkpoint_file_triage, device='cuda:0')
            result_triage = inference_detector(model_triage, filename)
            model_triage.show_result(filename, result_triage, out_file='demo/deformableDETR/set01_drone02_triage01/' + str(epoch) +".jpg")

    if filename == 'our_dataset/set01_test/test/set01_drone02_triage02_jpg.rf.59e8b762cf37e316c4e6574bce96582e.jpg':
        for epoch in range(1,range_num+1):
            checkpoint_file_triage = 'work_dirs/deformableDETR/epoch_' + str(epoch) + '.pth'
            model_triage = init_detector(config_file_triage, checkpoint_file_triage, device='cuda:0')
            result_triage = inference_detector(model_triage, filename)
            model_triage.show_result(filename, result_triage, out_file='demo/deformableDETR/set01_drone02_triage02/' + str(epoch) +".jpg")

    if filename == 'our_dataset/set01_test/test/set01_drone03_triage01_jpg.rf.c7623b8b9abfc3ef8aa0a6f5bf98a954.jpg':
        for epoch in range(1,range_num+1):
            checkpoint_file_triage = 'work_dirs/deformableDETR/epoch_' + str(epoch) + '.pth'
            model_triage = init_detector(config_file_triage, checkpoint_file_triage, device='cuda:0')
            result_triage = inference_detector(model_triage, filename)
            model_triage.show_result(filename, result_triage, out_file='demo/deformableDETR/set01_drone03_triage01/' + str(epoch) +".jpg")

                

          
        # model_triage.show_result(filename, result_triage, score_thr=0.2, out_file='demo/set01_drone01_triage02/' + str(i) +".jpg")
    # i +=1
#    triage_results.append(result_triage)

# k = 0
# for filename in glob.glob(im_folder + '*.jpg'):
#     # post-processing
#     # result_person = inference_detector(model_triage, filename) # person
#     # result_triage = inference_detector_twoStage(model_triage, filename, result_person)

#     # model_triage.show_result_postprocessing(filename, triage_results[k], person_results[k], out_file='demo/triage_version_1/post/' + str(k) +".jpg")
#     model_triage.show_result_postprocessing(filename, triage_results[k], person_results[k],  score_thr=0.2, out_file='demo/demothreshold/0_2/post/' + str(k) +".jpg")
#     k +=1

