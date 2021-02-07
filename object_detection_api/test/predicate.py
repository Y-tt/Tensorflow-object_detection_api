import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

#%matplotlib inline

#frozen_inference_graph.pb文件就是后面需要导入的文件，它保存了网络的结构和数据
PATH_TO_FROZEN_GRAPH = 'C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output/frozen_inference_graph.pb'
# mscoco_label_map.pbtxt文件中保存了index到类别名的映射，该文件就在object_dection/data文件夹下
#PATH_TO_LABELS = os.path.join('data', 'pascal_label_map.pbtxt')
PATH_TO_LABELS = 'C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output/labelmap.pbtxt'

#新建一个图
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

#这个函数也是一个方便使用的帮助函数，功能是将图片转换为Numpy数组的形式
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

#检测
PATH_TO_TEST_IMAGES_DIR = 'C:\\Users\\tt\\Desktop\\vov_data\\JPEGImages'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.JPG'.format(i)) for i in range(1,3 ) ]
# 输出图像的大小（单位是in）
IMAGE_SIZE = (12, 8)
with tf.Session(graph=detection_graph) as sess:
    for image_path in TEST_IMAGE_PATHS:
        image = Image.open(image_path)
        #将图片转换为numpy格式
        image_np = load_image_into_numpy_array(image)
        #将图片扩展一维，最后进入神经网络的图片格式应该是[1,?,?,3]，括号内参数分别为一个batch传入的数量，宽，高，通道数
        image_np_expanded = np.expand_dims(image_np,axis = 0)

        #获取模型中的tensor
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
 
        #boxes变量存放了所有检测框
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0') 
        #score表示每个检测结果的confidence
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        #classes表示每个框对应的类别
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        #num_detections表示检测框的个数
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
 
        #开始检测
        boxes,scores,classes,num_detections = sess.run([boxes,scores,classes,num_detections],
        feed_dict={image_tensor:image_np_expanded})
 
        #可视化结果
        #squeeze函数：从数组的形状中删除单维度条目，即把shape中为1的维度去掉
        vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    
        plt.figure(figsize=IMAGE_SIZE) 
        plt.imshow(image_np)
