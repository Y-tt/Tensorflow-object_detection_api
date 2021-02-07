# From tensorflow/models/research/
#其中model.ckpt-400表示使用第400步保存的模型。我们需要根据训练文件夹下checkpoint的实际步数改成对应的值。导出的模型是frozen_inference_graph.pb文件。

python object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix=C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output/model.ckpt-400 --output_directory=C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output