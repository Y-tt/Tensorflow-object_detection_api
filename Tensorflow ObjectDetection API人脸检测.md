# 1  项目环境

## 1.1 Anaconda 3.5.2.0

[Anaconda3.5.2.0 清华镜像下载](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)

tensorflow==1.12.0   （pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==1.12.0   使用清华镜像源下载更快）

[model==1.13.0](https://github.com/tensorflow/models)

python==3.6

#### GPU support

tensorflow-gpu==1.12.0   |  tensorflow-gpu==1.15.4

python==3.6                       |  python==3.7

[CUDA==10.0.130](https://developer.nvidia.com/cuda-10.0-download-archive)

cuDNN==7.4.1.5

## 1.2 Proto

[Proto](https://github.com/protocolbuffers/protobuf/releases)       下载适合自己系统版本的proto.exe放到C:/Windows目录下



# 2  资料准备

[model==1.13.0](https://github.com/tensorflow/models)

[下载需要的预训练模型](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)



# 3  环境搭建

## 3.1  object_detection_api安装

```shell
# 来自https://github.com/tensorflow/models/blob/v1.13.0/research/object_detection/g3doc/installation.md中Python Package Installation
cd C:\Users\tt\Desktop\tf_objectdetection_api\research
protoc ./object_detection/protos/*.proto --python_out=.
输出了一个换行（即无内容无报错）即可

# tf1    建议科学上网
尝试运行models/research/object_detection/目录下的.py文件时出现：ModuleNotFoundError: No module named 'object_detection'
需在models/research/目录下执行：
python setup.py install

若出现：ImportError: No module named nets
需在models/research/slim目录下执行：
python setup.py install
若出现：error: could not create 'build': 当文件已存在时，无法创建该文件。
原因是github下载下来的代码库中有个BUILD文件，而build和install指令需要新建build文件夹，名字冲突导致问题。暂时不清楚BUILD文件的作用。将该文件移动到其他目录或删除掉，再运行上述指令，即可成功安装。

若后续出现：ModuleNotFoundError: No module named 'pycocotools'
conda install pycocotools
若不行，则
pip installpycocotools

到此安装结束
python object_detection/builders/model_builder_test.py
运行测试，程序正常跑log，无报错即可，我测试运行之后最后会有个 OK(skipped=1)

————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# tf2
把object_detection/packages/tf2/setup.py复制到research目录下，替换原有的setup.py
python -m pip install --use-feature=2020-resolver .
这个过程会比较漫长，建议科学上网，报错了就重新运行，如无意外，Windows在pip install pycocotools会因为没有cl.exe报错，所以，之后我们手动使用
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
即可安装，之后再重新运行一下setup.py确认无报错，全部都是already satisfied（库已符合），这次建议盯着，不然报错了信息也刷走了

到此安装结束
python object_detection/builders/model_builder_tf2_test.py
运行测试，程序正常跑log，无报错即可，我测试运行之后最后会有个 OK(skipped=1)
```



## 3.2  GPU配置

### 3.2.1  CUDA10.0.130配置

注意要先安装好**VS**，否则会报错。选择 **Custom(Advanced)**，然后NEXT。将 **select driver components** 下的选项全部选上，要记住安装的路径，在后面配置环境变量的时候需要用到。安装完成后，配置系统环境变量。首先确认系统变量 path 中是否有 **CUDA_PATH **和 **CUDA_PATH_V10_0 **两项环境变量，然后将刚才CUDA安装路径添下的相应文件夹添加到**Path**中

```shell
CUDA_PATH         C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0
CUDA_PATH_V10_0   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0

path              C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin

# cmd 输入nvcc -V 测试CUDA是否安装成功，若输出CUDA版本号，即配置成功
```



### 3.2.2  cuDNN7.4.1.5配置

下载完成后解压文件，解压出来的cuda目录下包含如下内容：

```
+cuda
  +bin
  +include
  +ilb
  -NVIDIA_SIA_cuDNN_Support.txt
```

接着，我们将以上文件夹中的文件，复制到已安装的CUDA文件夹的相应位置下（根据实际安装）：

- 将bin中的 **cudnn64_7.dll** ，复制到**D:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin**；
- 将include中的 **cudnn.h** ，复制到**D:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include**；
- 将lib\x64中的 **cudnn.lib** ，复制到**D:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64**.



以上文件都复制完以后，将库路径**D:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64**，添加到环境变量中（根据实际安装）

```shell
path    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64

# 至此cuDNN安装、配置完毕！

# 测试GPU
# tf1
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) 

# tf2
sess=tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) 
```



# 4  数据准备

## 图片数据准备

### 1.rename.py

```python
# -*- coding:utf8 -*-
 
import os
class BatchRename():
    '''
    批量重命名文件夹中的图片文件
    '''
    def __init__(self):
        self.path = 'C:\\Users\\tt\\Desktop\\VOC_DATA\\JPEGImages'     #存放图片的文件夹路径
    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 1
        for item in filelist:
            if item.endswith('.jpg') or item.endswith('.JPG'):  #图片格式为jpg、JPG
 
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), str(i).zfill(5) + '.jpg')      #设置新的图片名称
                try:
                    os.rename(src, dst)
                    print ("converting %s to %s ..." % (src, dst))
                    i = i + 1        
                except:
                    continue
 
        print ("total %d to rename & converted %d jpgs" % (total_num, i))
if __name__ == '__main__':
    demo = BatchRename()
 
    demo.rename()
```



### 2.labelimg

[labelimg](https://github.com/tzutalin/labelImg)
生成xml放到Annotations文件夹中



### 3.CreateTXT.py

```python
import os
import random

trainval_percent = 0.8  # trainval数据集占所有数据的比例
train_percent = 0.5  # train数据集占trainval数据的比例
xmlfilepath = 'C:/Users/tt/Desktop/VOC/Annotations'
txtsavepath = 'C:/Users/tt/Desktop/VOC/ImageSets/Main'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
print('total number is ', num)
list = range(num)
tv = int(num * trainval_percent)
print('trainVal number is ', tv)
tr = int(tv * train_percent)
print('train number is ', tr)
print('test number is ', num - tv)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open('C:/Users/tt/Desktop/VOC/ImageSets/Main/trainval.txt', 'w')
ftest = open('C:/Users/tt/Desktop/VOC/ImageSets/Main/test.txt', 'w')
ftrain = open('C:/Users/tt/Desktop/VOC/ImageSets/Main/train.txt', 'w')
fval = open('C:/Users/tt/Desktop/VOC/ImageSets/Main/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
```



### 4.train_test_split.py

```python
import os  
import random  
import time  
import shutil

xmlfilepath=r'C:/Users/tt/Desktop/VOC/Annotations'  
saveBasePath=r"C:/Users/tt/Desktop/VOC/Annotations"

trainval_percent=0.8  
train_percent=0.8  
total_xml = os.listdir(xmlfilepath)  
num=len(total_xml)  
list=range(num)  
tv=int(num*trainval_percent)  
tr=int(tv*train_percent)  
trainval= random.sample(list,tv)  
train=random.sample(trainval,tr)  
print("train and val size",tv)  
print("train size",tr) 

start = time.time()

test_num=0  
val_num=0  
train_num=0  

for i in list:  
    name=total_xml[i]
    if i in trainval:  #train and val set 
        if i in train: 
            directory="train"  
            train_num += 1  
            xml_path = os.path.join(os.getcwd(), 'Annotations/{}'.format(directory))  
            if(not os.path.exists(xml_path)):  
                os.mkdir(xml_path)  
            filePath=os.path.join(xmlfilepath,name)  
            newfile=os.path.join(saveBasePath,os.path.join(directory,name))  
            shutil.copyfile(filePath, newfile)
        else:
            directory="validation"  
            xml_path = os.path.join(os.getcwd(), 'Annotations/{}'.format(directory))  
            if(not os.path.exists(xml_path)):  
                os.mkdir(xml_path)  
            val_num += 1  
            filePath=os.path.join(xmlfilepath,name)   
            newfile=os.path.join(saveBasePath,os.path.join(directory,name))  
            shutil.copyfile(filePath, newfile)

    else:
        directory="test"  
        xml_path = os.path.join(os.getcwd(), 'Annotations/{}'.format(directory))  
        if(not os.path.exists(xml_path)):  
                os.mkdir(xml_path)  
        test_num += 1  
        filePath=os.path.join(xmlfilepath,name)  
        newfile=os.path.join(saveBasePath,os.path.join(directory,name))  
        shutil.copyfile(filePath, newfile)

end = time.time()  
seconds=end-start  
print("train total : "+str(train_num))  
print("validation total : "+str(val_num))  
print("test total : "+str(test_num))  
total_num=train_num+val_num+test_num  
print("total number : "+str(total_num))  
print( "Time taken : {0} seconds".format(seconds))
```



### 5.xml_to_csv.py

```python
import os  
import glob  
import pandas as pd  
import xml.etree.ElementTree as ET 

def xml_to_csv(path):  
    xml_list = []  
    for xml_file in glob.glob(path + '/*.xml'):  
        tree = ET.parse(xml_file)  
        root = tree.getroot()
        
        print(root.find('filename').text)  
        for member in root.findall('object'): 
            value = (root.find('filename').text,  
                int(root.find('size')[0].text),   #width  
                int(root.find('size')[1].text),   #height  
                member[0].text,  
                int(member[4][0].text),  
                int(float(member[4][1].text)),  
                int(member[4][2].text),  
                int(member[4][3].text)  
                )  
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)  
    return xml_df      

def main():  
    for directory in ['train','test','validation']:  
        xml_path = os.path.join(os.getcwd(), 'Annotations/{}'.format(directory))  

        xml_df = xml_to_csv(xml_path)  
        # xml_df.to_csv('whsyxt.csv', index=None)  
        xml_df.to_csv('C:/Users/tt/Desktop/VOC/people_{}_labels.csv'.format(directory), index=None)  
        print('Successfully converted xml to csv.')

main()

```



### 6.generate_tfrecord.py

```python
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags

flags.DEFINE_string('csv_input', 'C:/Users/tt/Desktop/VOC/people_test_labels.csv', 'Path to the CSV input')#csv文件
flags.DEFINE_string('output_path', 'C:/Users/tt/Desktop/VOC/people_test.record', 'Path to output TFRecord')#TFRecord文件
flags.DEFINE_string('image_dir', 'C:/Users/tt/Desktop/VOC/JPEGImages', 'Path to images')#对应的图片位置

FLAGS = flags.FLAGS

# TO-DO replace this with label map
#从1开始根据自己训练的类别数和标签来写
def class_text_to_int(row_label):
    if row_label == 'ChenHe':
        return 1
    elif row_label == 'DengChao':
        return 2
    elif row_label =='HeJiong':
        return 3
    elif row_label =='HuGe':
        return 4
    elif row_label == 'LiuHaoran':
        return 5
    elif row_label =='LiuYifei':
        return 6
    elif row_label =='ShiyuanLimei':
        return 7
    elif row_label == 'TongLiya':
        return 8
    elif row_label =='WangBaoqiang':
        return 9
    elif row_label =='YangMi':
        return 10
    elif row_label == 'YangZi':
        return 11
    elif row_label =='ZhangHan':
        return 12
    elif row_label =='ZhangYishan':
        return 13
    elif row_label == 'ZhouDongyu':
        return 14
    elif row_label =='ZhouJielun':
        return 15
    elif row_label =='WangJunkai':
        return 16
    else:
        None

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    print(os.path.join(path, '{}'.format(group.filename)))
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={

        'image/height': dataset_util.int64_feature(height),

        'image/width': dataset_util.int64_feature(width),

        'image/filename': dataset_util.bytes_feature(filename),

        'image/source_id': dataset_util.bytes_feature(filename),

        'image/encoded': dataset_util.bytes_feature(encoded_jpg),

        'image/format': dataset_util.bytes_feature(image_format),

        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),

        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),

        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),

        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),

        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),

        'image/object/class/label': dataset_util.int64_list_feature(classes),

    }))

    return tf_example


def main(_):

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    path = os.path.join(FLAGS.image_dir)

    examples = pd.read_csv(FLAGS.csv_input)

    grouped = split(examples, 'filename')

    for group in grouped:

        tf_example = create_tf_example(group, path)

        writer.write(tf_example.SerializeToString())


    writer.close()

    output_path = os.path.join(os.getcwd(), FLAGS.output_path)

    print('Successfully created the TFRecords: {}'.format(output_path))



if __name__ == '__main__':

    tf.app.run()
```



### 7.labelmap.pbtxt

```shell
item {
  id: 1
  name: 'keyboard'
}

item {
  id: 2
  name: 'mouse'
}
命名为label_map.txt放到object_detection/data下
注意若是使用文本文件，生成的是.txt，需要得到的是.pbtxt，否则会报错：
tensorflow.python.framework.errors_impl.NotFoundError: NewRandomAccessFile failed to Create/Open: C:/Users/tt/Desktop/models-master/research/object_detection/vov_data/labelmap.pbtxt : ϵͳ\udcd5Ҳ\udcbb\udcb5\udcbdָ\udcb6\udca8\udcb5\udcc4\udcceļ\udcfe\udca1\udca3
; No such file or directory
```



### 8.pipeline.config

```shell
# 选择需要使用的模型，我这里选择的是ssd_mobilenet_v1_coco.config
# 来自https://github.com/tensorflow/models/blob/v1.13.0/research/object_detection/g3doc/configuring_jobs.md中Configuring the Object Detection Training Pipeline

配置文件分成五个部分：
1.model模型的框架 meta-architecture, feature extractor…
2.train_config，定义 optimizer (Momentum, Adam, Adagrad…), fine-tune model
3.eval_config，定义valuation估值指标
4.train_input_config，定义作为训练数据集与标签映射路径
5.eval_input_config，定义作为估值数据集的路径与标签映射路径

主要修改这四部分：
1：自定义路径指定模型位置
fine_tune_checkpoint: “PATH_TO_BE_CONFIGURED/model.ckpt”
通常在进行训练时不会从头开始训练，大部份会利用别人已经训练好的参数来微调以减少训练的时间fine_tune_checkpoint的数值为：你定义的faster_rcnn_resnet101_coco_11_06_2017位置（例如："object_detection/faster_rcnn_resnet101_coco_11_06_2017/model.ckpt"）

2：指定训练数据的label和record数据文件
label文件 官方已经有提供放在 object_detection/pascal_val.record
train_input_reader: {
tf_record_input_reader { input_path: "PATH_TO_BE_CONFIGURED/pascal_train.record" }
label_map_path: "PATH_TO_BE_CONFIGURED/pascal_label_map.pbtxt"}

3：指定测试数据的label和record数据文件
eval_input_reader: {
tf_record_input_reader { input_path: "PATH_TO_BE_CONFIGURED/pascal_val.record" }
label_map_path: "PATH_TO_BE_CONFIGURED/pascal_label_map.pbtxt"
}

4.
eval_config: {
num_examples: 5823#验证集图片数修改
# Note: The below line limits the evaluation process to 10 evaluations.
# Remove the below line to evaluate indefinitely.
max_evals: 10
}
```



# 5  train

## model_main.py

```shell
# 来自https://github.com/tensorflow/models/blob/v1.13.0/research/object_detection/g3doc/running_locally.md中TrainRunning Locally

# From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH={path to pipeline config file}
MODEL_DIR={path to model directory}
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr

# 本机脚本
# From the tensorflow/models/research/ directory
python object_detection/model_main.py --pipeline_config_path=C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/ssd_mobilenet_v1_coco.config --model_dir=C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output --num_train_steps=200 --sample_1_of_n_eval_examples=1 --alsologtostderr

```



# 6  Running Tensorboard

```shell
# 来自https://github.com/tensorflow/models/blob/v1.13.0/research/object_detection/g3doc/running_locally.md中TrainRunning Locally

# 通过tensorboard命令可以在浏览器很轻松的监控训练进程，在浏览器输入localhost:6006（默认）即可
tensorboard --logdir=${MODEL_DIR}
```



# 7  导出模型并预测单张图片

新版本export_inference_graph.py export_tflite_ssd_graph.py可以直接固化计算流图

## 7.1  export_inference_graph.py

```shell
# 来自https://github.com/tensorflow/models/blob/v1.13.0/research/object_detection/g3doc/exporting_models.md中Exporting a trained model for inference
# From tensorflow/models/research/
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH={path to pipeline config file}
TRAINED_CKPT_PREFIX={path to model.ckpt}
EXPORT_DIR={path to folder that will be used for export}
python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
    
```



## 7.2  export_tflite_ssd_graph.py

```shell
object_detection/export_tflite_ssd_graph.py \
--pipeline_config_path=$CONFIG_FILE \
--trained_checkpoint_prefix=$CHECKPOINT_PATH \
--output_directory=$OUTPUT_DIR \
--add_postprocessing_op=true
```



## 7.3  predicate

### test1.py

##### 出现的问题：
1. ModuleNotFoundError: No module named 'utils'
   解决方法：

```python
# from utils import  label_map_util
# from utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
```



2. Traceback (most recent call last):
     File "test233.py", line 52, in <module>
       cv.imshow("input", image)
   cv2.error: OpenCV(4.5.1) C:\Users\appveyor\AppData\Local\Temp\1\pip-req-build-vijyisc5\opencv\modules\highgui\src\window.cpp:376: error: (-215:Assertion failed) size.width>0 && size.height>0 in function 'cv::imshow'
   解决方法：把文件放在源码路径下



```python
# test1.py
import os
import sys
import tarfile

import cv2 as cv
import numpy as np
import tensorflow as tf

#from utils import  label_map_util
#from utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

'''
import sys
MODEL_ROOT = "C:/Users/tt/Desktop"
sys.path.append(MODEL_ROOT)  # 应用和训练的目录在不同的地方
import preprocess
'''
# What model to download.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
#MODEL_FILE = 'D:/tensorflow/' + MODEL_NAME + '.tar'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
#PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_FROZEN_GRAPH = 'C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('D:/tensorflow/models/research/object_detection/data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = 'C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output/labelmap.pbtxt'

NUM_CLASSES = 16

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categorys = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categorys)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image = cv.imread("18.jpg" , 1 )
        cv.imshow("input", image)
        print(image.shape)
        image_np_expanded = np.expand_dims(image, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                                                            feed_dict={image_tensor: image_np_expanded})

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            min_score_thresh=0.1,                            #0.5
            use_normalized_coordinates=True,
            line_thickness=4
        )
        cv.imshow("SSD - object detection image", image)
        cv.waitKey(0)
        cv.destroyAllWindows()

```



### test2.py（未成功，不明原因）

然后可以参考上面我们介绍的jupyter notebook代码，自行编写利用导出模型对单张图片做目标检测的脚本。然后将PATH_TO_FROZEN_GRAPH的值赋值为voc/export/frozen_inference_graph.pb，即导出模型文件。将PATH_TO_LABELS修改为voc/pascal_label_map.pbtxt，即各个类别的名称。其它代码都可以不改变，然后测试我们的图片

```python
# test2.py
# 来自https://www.cnblogs.com/wind-chaser/p/11339284.html
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

# frozen_inference_graph.pb文件就是后面需要导入的文件，它保存了网络的结构和数据
PATH_TO_FROZEN_GRAPH = 'C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output/frozen_inference_graph.pb'
# mscoco_label_map.pbtxt文件中保存了index到类别名的映射，该文件就在object_dection/data文件夹下
# PATH_TO_LABELS = os.path.join('data', 'pascal_label_map.pbtxt')
PATH_TO_LABELS = 'C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output/labelmap.pbtxt'

# 新建一个图
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# 这个函数也是一个方便使用的帮助函数，功能是将图片转换为Numpy数组的形式
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# 检测
PATH_TO_TEST_IMAGES_DIR = 'C:\\Users\\tt\\Desktop\\vov_data\\JPEGImages'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.JPG'.format(i)) for i in range(1,3 ) ]
# 输出图像的大小（单位是in）
IMAGE_SIZE = (12, 8)
with tf.Session(graph=detection_graph) as sess:
    for image_path in TEST_IMAGE_PATHS:
        image = Image.open(image_path)
        # 将图片转换为numpy格式
        image_np = load_image_into_numpy_array(image)
        # 将图片扩展一维，最后进入神经网络的图片格式应该是[1,?,?,3]，括号内参数分别为一个batch传入的数量，宽，高，通道数
        image_np_expanded = np.expand_dims(image_np,axis = 0)

        # 获取模型中的tensor
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
 
        # boxes变量存放了所有检测框
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0') 
        # score表示每个检测结果的confidence
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        # classes表示每个框对应的类别
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # num_detections表示检测框的个数
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
 
        # 开始检测
        boxes,scores,classes,num_detections = sess.run([boxes,scores,classes,num_detections],
        feed_dict={image_tensor:image_np_expanded})
 
        # 可视化结果
        # squeeze函数：从数组的形状中删除单维度条目，即把shape中为1的维度去掉
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
```



# 8  从源码编译tensorflow

## 8.1  参考文档

[tensorflow官方文档](https://tensorflow.google.cn/install/source_windows?hl=zh-cn)

[bazel官方文档](https://docs.bazel.build/versions/master/windows.html#build-c-with-clang)

[csdn的一篇文章，很详细](https://blog.csdn.net/atpalain_csdn/article/details/97945536)

[tensorflow工具用法](https://blog.csdn.net/chenyuping333/article/details/82108509?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.control)



## 8.2  编译环境

**Win10**

**tensorflow==1.13.1**

**bazel==0.19.1**

**python==3.6**

**JDK8**

**MSYS2**

**VS2015**



## 8.3  编译准备

[下载**tensorflow==1.13.1**](https://github.com/tensorflow/tensorflow)

[下载**bazel==0.19.1**](https://github.com/bazelbuild/bazel)

[下载**python==3.6**](https://www.python.org/downloads/)

[下载**MSYS2**](https://www.msys2.org/)

[下载**VS2015**（Visual Studio Community 2015 with Update 3）](https://my.visualstudio.com/Downloads?q=visual%20studio%202015&wt.mc_id=o~msft~vscom~older-downloads)



## 8.4  环境配置

### 8.4.1  安装VS2015

注意默认安装时有些包没有装上，应勾选上Windows 10 SDK等开发工具包，否则后续编译时会报错：

```shell
The target you are compiling requires Visual C++ build tools.
Bazel couldn't find a valid Visual C++ build tools installation on your machine.

Visual C++ build tools seems to be installed at D:\Microsoft Visual Studio 14.0\VC\
But Bazel can't find the following tools:
    VCVARSALL.BAT, cl.exe, link.exe, lib.exe, ml64.exe

```



### 8.4.2  python环境配置

将python可执行文件添加入系统变量path中，eg：

```shell
# python环境变量配置
D:\Program Files\Python\Python36

# 安装 Python 和 TensorFlow 软件包依赖项
pip3 install six numpy wheel
pip3 install keras_applications==1.0.6 --no-deps
pip3 install keras_preprocessing==1.0.5 --no-deps

```



### 8.4.3  Java环境配置

新建两个系统变量JAVA_HOME、CLASSPATH

```shell
JAVA_HOME： C:\Program Files\Java\jdk1.8.0_91  #电脑上JDK安装的绝对路径
CLASSPATH： .;%JAVA_HOME%\lib\dt.jar;%JAVA_HOME%\lib\tools.jar;
path：      %JAVA_HOME%\bin;%JAVA_HOME%\jre\bin

# cmd java  javac  java -version测试
```



### 8.4.4  MSYS2配置

一路默认即可，当安装完弹出类似cmd的命令窗口时，输入以下命令：

```shell
# 询问是否安装，输入y
pacman -Syu
pacman -S git
pacman -S patch unzip grep

# 包安装完后，将以下路径添加到系统变量path中（根据实际安装路径）
C:\msys64
C:\msys64\usr\bin

# 至此MSYS2安装配置完毕！
```



### 8.4.5  bazel配置

将下载好的bazel.x.exe文件复制到C:\msys64下（根据实际安装路径），更名为bazel.exe。复制完毕后，配置bazel环境变量

```shell
# 新建三个系统变量：BAZEL_SH，BAZEL_VC，BAZEL_VS。相应的路径如以下：
BAZEL_SH  C:\msys64\usr\bin\bash.exe
BAZEL_VC  C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC
BAZEL_VS  C:\Program Files (x86)\Microsoft Visual Studio 14.0

# 至此bazel安装配置完毕！
```



## 8.5  编译tensorflow工具summarize_graph、freeze_graph、toco

### 8.5.1  配置build

```shell
# using tensorflow==1.13.1  bazel==0.19.1
cd tensorflow-1.13.1/   
python ./configure.py

# 出现以下询问：
Please specify the location of python. [Default is D:\Python36\python.exe]:   # 回车（默认）

Please input the desired Python library path to use.  Default is [D:\Python36\lib\site-packages]  # 回车（默认）

Do you wish to build TensorFlow with XLA JIT support? [y/N]:  # 输入n
  
Do you wish to build TensorFlow with ROCm support? [y/N]:  # 输入n

Do you wish to build TensorFlow with CUDA support? [y/N]:  # 输入n
    
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is /arch:AVX]:  # 回车（默认）
    
Would you like to override eigen strong inline for some C++ compilation to reduce the compilation time? [Y/n]:  # 输入y
    
# build配置完毕
```



### 8.5.2  开始编译tensorflow工具

```shell
# 编译summarize_graph工具  |  summarize_graph可以查看网络节点，在只有一个固化的权重文件而不知道具体的网络结构时非常有用。
bazel build tensorflow/tools/graph_transforms:summarize_graph
    
# 编译freeze_graph工具     |  freeze_graph是模型固化工具。
bazel build tensorflow/python/tools:freeze_graph
    
# 编译toco工具             |  toco是用来生成一个可供TensorFlow Lite框架使用tflite文件。
bazel build tensorflow/lite/toco:toco
 
################################################################################################################################
 
###构建 pip 软件包###

# TensorFlow 2.x
# tensorflow:master 代码库已经默认更新为 build 2.x。请安装 Bazel 并使用 bazel build 创建 TensorFlow 软件包。
bazel build //tensorflow/tools/pip_package:build_pip_package

# TensorFlow 1.x
# 如需从 master 分支构建 TensorFlow 1.x，请使用 bazel build --config=v1 创建 TensorFlow 1.x 软件包。
bazel build --config=v1 //tensorflow/tools/pip_package:build_pip_package

# 仅支持CPU   |   使用 bazel 构建仅支持 CPU 的 TensorFlow 软件包构建器
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
# GPU支持     |   构建支持 GPU 的 TensorFlow 软件包编译器
bazel build --config=opt --config=cuda --define=no_tensorflow_py_deps=true //tensorflow/tools/pip_package:build_pip_package

# Bazel 构建选项
# 在构建时使用以下选项，避免在创建软件包时出现问题：tensorflow:issue#22390 （https://github.com/tensorflow/tensorflow/issues/22390）
--define=no_tensorflow_py_deps=true

# 其他需要：https://www.tensorflow.org/install/source_windows

###出现的问题###

## ”no such package '@icu//': java.io.IOException..." ##
# 解决方法1（无效）：https://blog.csdn.net/xiaolt90/article/details/104970944
# 解决方法2（无效）：这里主要是由于访问github下载包的网络延迟较大，使用手机连接热点解决问题。

# 编译pip软件包报错，其他的单个工具包则成功
```



# 9  convert

## pb2tflite

```shell
# 来自https://github.com/tensorflow/models/blob/v1.13.0/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md中Running on mobile with TensorFlow Lite

'''
# 比较旧的版本了  新版本export_inference_graph.py export_tflite_ssd_graph.py可以直接固化计算流图
# freeze_graph  在编译好的tensorflow源码目录下的bazel-bin中进行，首先要确认bazel-bin文件夹存在。
bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=/tmp/inception_v3_inf_graph.pb \
  --input_checkpoint=/tmp/checkpoints/inception_v3.ckpt \
  --input_binary=true --output_graph=/tmp/frozen_inception_v3.pb \
  --output_node_names=InceptionV3/Predictions/Reshape_1
'''

# summarize_graph 
bazel run --config=opt tensorflow/tools/graph_transforms:summarize_graph -- --in_graph=C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output/tflite_graph.pb
  
# toco 
# floating point model
bazel run --config=opt tensorflow/lite/toco:toco -- \
--input_file=$OUTPUT_DIR/tflite_graph.pb \
--output_file=$OUTPUT_DIR/detect.tflite \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
--inference_type=FLOAT \
--allow_custom_ops

# quantized model
bazel run --config=opt tensorflow/lite/toco:toco -- \
--input_file=$OUTPUT_DIR/tflite_graph.pb \
--output_file=$OUTPUT_DIR/detect.tflite \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=QUANTIZED_UINT8 \
--mean_values=128 \
--std_values=128 \
--change_concat_input_ranges=false \
--allow_custom_ops


# 本次训练脚本  报错：详见C:\Users\tt\Desktop\object_detection_api\pb2tflite\pb2tflite_build_run_log.txt
bazel run --config=opt tensorflow/lite/toco:toco -- --input_file=C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output/tflite_graph.pb --output_file=C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output/detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  --inference_type=FLOAT --allow_custom_ops
            
# 改为：
bazel run --config=opt tensorflow/lite/toco:toco -- --input_file=C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output/tflite_graph.pb --output_file=C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output/detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays=TFLite_Detection_PostProcess  --inference_type=FLOAT --allow_custom_ops

# 注意通过  summarize_graph 查看网络  output_arrays
--output_arrays=TFLite_Detection_PostProcess
```



# 10  迁移至Android studio

## 10.1  环境配置

下载并安装Android studio3.5.3（科学上网）

```shell
# 来自 https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md

# 将自己的 detect.tflite 和 labelmap.txt 复制到
examples-master/lite/examples/object_detection/android/app/src/main/assets
# 将文件名修改为detect.tflite labelmap.txt

# 将 apply from:'download_model.gradle' 注释掉，避免自动下载模型
cd TF_EXAMPLES/lite/examples/object_detection/android/app/build.gradle
// apply from:'download_model.gradle

# 若自定义文件名和路径，可自行修改DetectorActivity.java
cd TF_EXAMPLES/lite/examples/object_detection/android/app/src/main/java/org/tensorflow/demo/DetectorActivity.java

  private static final boolean TF_OD_API_IS_QUANTIZED = true/false;
  private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
  private static final String TF_OD_API_LABELS_FILE = "labels_list.txt";
```



## 10.2  build配置(旧版本，可不看)

```shell
# 来自C:\Users\tt\Desktop\models-r1.13.0\research\object_detection\g3doc\running_on_mobile_tensorflowlite.md

# 将自己的 detect.tflite 和 labelmap.txt 复制到
tensorflow1.13.1/tensorflow/lite/examples/android/app/src/main/assets

# cd tensorflow1.13.1/tensorflow/lite/examples/android/BUILD 
# 找到 assets =[]，把 "@tflite_mobilenet_ssd//:mobilenet_ssd.tflite"
(which by default points to a COCO pretrained model)替换成自己的detect.tflite的路径 "F:/tensorflow-1.13.1.0/tensorflow/lite/examples/android/app/src/main/assets：detect.tflite"
# 找到"//tensorflow/lite/examples/android/app/src/main/assets:coco_labels_list.txt"替换成自己的labelmap.txt的路径"F:/tensorflow-1.13.1.0/tensorflow/lite/examples/android/app/src/main/assets：labelmap.txt"

# 告诉app使用新的labelmap.txt
# cd tensorflow1.13.1/tensorflow/lite/examples/android/app/src/main/java/org/tensorflow/demo/DetectorActivity.java
# 找到 TF_OD_API_LABELS_FILE 将其值替换为自己 labelmap.txt 的路径 "F:/tensorflow-1.13.1.0/tensorflow/lite/examples/android/app/src/main/assets：labelmap.txt"
# 具体修改如下：
  private static final boolean TF_OD_API_IS_QUANTIZED = true/false;
  private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
  
# build the demo app
cd tensorflow1.13.1
 bazel build -c opt --config=android_arm{,64} --cxxopt='--std=c++11'
"//tensorflow/lite/examples/android:tflite_demo"

# install the demo via  [Android Debug Bridge](https://developer.android.com/studio/command-line/adb) (adb):
adb install bazel-bin/tensorflow/lite/examples/android/tflite_demo.apk
```



# 11  参考文档

## 11.1  object_detection_api

[ModuleNotFoundError: No module named 'object_detection'](https://www.jianshu.com/p/df42f49e7e9c)

[使用TensorFlow Object Detection API进行目标检测](https://www.cnblogs.com/wind-chaser/p/11339284.html)

https://www.cnblogs.com/arkenstone/p/7237292.html

https://cloud.tencent.com/developer/article/1341546



## 11.2  bazel交叉编译tensorflow

[tensorflow官方文档](https://tensorflow.google.cn/install/source_windows?hl=zh-cn)

[bazel官方文档](https://docs.bazel.build/versions/master/windows.html#build-c-with-clang)

[csdn的一篇文章，很详细](https://blog.csdn.net/atpalain_csdn/article/details/97945536)

[tensorflow工具用法](https://blog.csdn.net/chenyuping333/article/details/82108509?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.control)

