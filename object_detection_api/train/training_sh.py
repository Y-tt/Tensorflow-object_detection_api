# From the tensorflow/models/research/ directory

#float
python object_detection/model_main.py --pipeline_config_path=C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/ssd_mobilenet_v1_coco.config --model_dir=C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output --num_train_steps=800 --sample_1_of_n_eval_examples=1 --alsologtostderr

# quantize
python object_detection/model_main.py --pipeline_config_path=C:/Users/tt/Desktop/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18/ssd_mobilenet_v1_quantized_300x300_coco14_sync.config --model_dir=C:/Users/tt/Desktop/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18/output --num_train_steps=400 --sample_1_of_n_eval_examples=1 --alsologtostderr

# python legacy/train.py --train_dir C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output --pipeline_config_path C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/ssd_mobilenet_v1_coco.config

# 通过tensorboard命令可以在浏览器很轻松的监控训练进程，在浏览器输入localhost:6006（默认）即可
tensorboard --logdir=C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output