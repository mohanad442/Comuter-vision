#!/usr/bin/env python
# coding: utf-8

# # 0. Setup Paths

# In[1]:


WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'


# # 1. Create Label Map

# In[2]:


labels = [{'name':'with_mask', 'id':1}, {'name':'without_mask', 'id':2},{'name':'mask_weared_incorrect', 'id':3}]

with open(ANNOTATION_PATH + '/label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')


# # 2. Create TF records

# In[2]:


get_ipython().system("python {SCRIPTS_PATH + '/generate_tfrecord.py'} -x {IMAGE_PATH + '/train'} -l {ANNOTATION_PATH + '/label_map.pbtxt'} -o {ANNOTATION_PATH + '/train.record'}")
get_ipython().system("python {SCRIPTS_PATH + '/generate_tfrecord.py'} -x {IMAGE_PATH + '/test'} -l {ANNOTATION_PATH + '/label_map.pbtxt'} -o {ANNOTATION_PATH + '/test.record'}")


# # 3. Download TF Models Pretrained Models from Tensorflow Model Zoo

# In[16]:


get_ipython().system('cd Tensorflow && git clone https://github.com/tensorflow/models')


# In[7]:


#wget.download('http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz')
#!mv ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz {PRETRAINED_MODEL_PATH}
#!cd {PRETRAINED_MODEL_PATH} && tar -zxvf ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz


# # 4. Copy Model Config to Training Folder

# In[3]:


CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 


# In[9]:


get_ipython().system("mkdir {'Tensorflow\\workspace\\models\\\\'+CUSTOM_MODEL_NAME}")


# In[10]:


get_ipython().system("cp {PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config'} {MODEL_PATH+'/'+CUSTOM_MODEL_NAME}")


# # 5. Update Config For Transfer Learning

# In[2]:


import tensorflow as tf 
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


# In[4]:


CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'


# In[6]:


config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)


# In[7]:


config


# In[8]:


pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(CONFIG_PATH, 'r') as f:
    proto_str = f.read()
    text_format.Merge(proto_str,pipeline_config)


# In[9]:


pipeline_config.model.ssd.num_classes = 3
pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/train.record']
pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/test.record']


# In[10]:


config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:                                                                                                                                                                                                                     
    f.write(config_text)   


# In[11]:


pipeline_config # modified pipeline


# # 6. Train the model

# In[ ]:


tf.test.is_gpu_available()


# In[19]:


print("""python {}/research/object_detection/model_main_tf2.py --model_dir={}/{} --pipeline_config_path={}/{}/pipeline.config --num_train_steps=5000""".format(APIMODEL_PATH, MODEL_PATH,CUSTOM_MODEL_NAME,MODEL_PATH,CUSTOM_MODEL_NAME))


# In[33]:


get_ipython().system('python Tensorflow\\models\\research\\object_detection\\model_main_tf2.py     --pipeline_config_path=${Tensorflow\\workspace\\models\\my_ssd_mobnet\\pipeline.config}     --model_dir=${Tensorflow\\workspace\\models\\my_ssd_mobnet}     --num_train_steps=5000     --alsologtostderr')


# # 7. Load Train Model From Checkpoint

# In[3]:


import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


# In[4]:


# Load modeified pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-6')).expect_partial()

# Pass our images to detect function and get back our detections
@tf.function   # tensorflow function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)                # resize the image down to 320*320
    prediction_dict = detection_model.predict(image, shapes)         # Using the detection_model to make a prediction
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


# # 8. Detect in Real-Time

# In[5]:


import cv2 
import numpy as np


# In[6]:


category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')


# In[12]:


category_index


# In[7]:


# Setup video capture from local webcam
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# In[8]:


while True: 
    # Get a video capture, And convert each image to a numpy array in order to work with tensor flow
    ret, frame = cap.read()
    if ret:
        image_np = np.array(frame)

        # Convert the image to a tensor (Tensors are multi-dimensional arrays with a uniform type (called a dtype))
        # Cause the model expects a multiple image to come through (in this case we're only passing through one)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

        # Making a detection
        detections = detect_fn(input_tensor)

        # Getting out the number of detection in order to do preprocessing on it
        num_detections = int(detections.pop('num_detections'))

        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)


        # Gettinig a copy of the image to visualize detections on it
        image_np_with_detections = image_np.copy() 

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes'] + 1 , # Detection classes starts at 0 while category_index stasts at 1
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.8,                 # minimum threshold is 80% to get high accuracy
                    agnostic_mode=False)

        cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            break


# In[ ]:




