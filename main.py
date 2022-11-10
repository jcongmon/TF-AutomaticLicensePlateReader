import os
import object_detection

# CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
# PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
# PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
# TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
# LABEL_MAP_NAME = 'label_map.pbtxt'
#
# paths = {
#     'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
#     'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
#     'APIMODEL_PATH': os.path.join('Tensorflow','models'),
#     'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
#     'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
#     'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
#     'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
#     'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME),
#     'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'),
#     'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'),
#     'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'),
#     'PROTOC_PATH':os.path.join('Tensorflow','protoc')
#  }
#
# files = {
#     'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
#     'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
#     'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
# }

# for path in paths.values():
#     if not os.path.exists(path):
#         os.makedirs(path)

# VCS from https://github.com/tensorflow/models to Tensorflow/models
# from the terminal: brew install protobuf
# check with protoc --version

# https://medium.com/@marklabinski/installing-tensorflow-object-detection-api-on-windows-10-7a4eb83e1e7b
# for setting up pythonpath https://bic-berkeley.github.io/psych-214-fall-2016/using_pythonpath.html

# labels = [{'name':'licence', 'id':1}]

# with open(files['LABELMAP'], 'w') as f:
#     for label in labels:
#         f.write('item { \n')
#         f.write('\tname:\'{}\'\n'.format(label['name']))
#         f.write('\tid:{}\n'.format(label['id']))
#         f.write('}\n')

import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file("/Users/Jonathan/PycharmProjects/anpr/Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config")
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore("/Users/Jonathan/PycharmProjects/anpr/Tensorflow/workspace/models/ckpt-11").expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


category_index = label_map_util.create_category_index_from_labelmap("/Users/Jonathan/PycharmProjects/anpr/Tensorflow/workspace/annotations/label_map.pbtxt")
IMAGE_PATH = "/Users/Jonathan/PycharmProjects/anpr/Tensorflow/workspace/images/test/Cars413.png"

img = cv2.imread(IMAGE_PATH)
image_np = np.array(img)

input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.8,
            agnostic_mode=False)

plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
plt.show()

# TODO 11/2/2022 OCR LIBRARY TO EXTRACT LICENSE PLATE VALUES