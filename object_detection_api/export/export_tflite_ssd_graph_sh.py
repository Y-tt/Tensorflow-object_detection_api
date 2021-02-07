python object_detection/export_tflite_ssd_graph.py --pipeline_config_path C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output/model.ckpt-400 --output_directory C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output

r"""Exports an SSD detection model to use with tf-lite.

Outputs file:
* A tflite compatible frozen graph - $output_directory/tflite_graph.pb

The exported graph has the following input and output nodes.

Inputs:
'normalized_input_image_tensor': a float32 tensor of shape
[1, height, width, 3] containing the normalized input image. Note that the
height and width must be compatible with the height and width configured in
the fixed_shape_image resizer options in the pipeline config proto.

In floating point Mobilenet model, 'normalized_image_tensor' has values
between [-1,1). This typically means mapping each pixel (linearly)
to a value between [-1, 1]. Input image
values between 0 and 255 are scaled by (1/128.0) and then a value of
-1 is added to them to ensure the range is [-1,1).
In quantized Mobilenet model, 'normalized_image_tensor' has values between [0,
255].
In general, see the `preprocess` function defined in the feature extractor class
in the object_detection/models directory.

Outputs:
If add_postprocessing_op is true: frozen graph adds a
  TFLite_Detection_PostProcess custom op node has four outputs:
  detection_boxes: a float32 tensor of shape [1, num_boxes, 4] with box
  locations
  detection_classes: a float32 tensor of shape [1, num_boxes]
  with class indices
  detection_scores: a float32 tensor of shape [1, num_boxes]
  with class scores
  num_boxes: a float32 tensor of size 1 containing the number of detected boxes
else:
  the graph has two outputs:
   'raw_outputs/box_encodings': a float32 tensor of shape [1, num_anchors, 4]
    containing the encoded box predictions.
   'raw_outputs/class_predictions': a float32 tensor of shape
    [1, num_anchors, num_classes] containing the class scores for each anchor
    after applying score conversion.

Example Usage:
--------------
python object_detection/export_tflite_ssd_graph \
    --pipeline_config_path path/to/ssd_mobilenet.config \
    --trained_checkpoint_prefix path/to/model.ckpt \
    --output_directory path/to/exported_model_directory

The expected output would be in the directory
path/to/exported_model_directory (which is created if it does not exist)
with contents:
 - tflite_graph.pbtxt
 - tflite_graph.pb
Config overrides (see the `config_override` flag) are text protobufs
(also of type pipeline_pb2.TrainEvalPipelineConfig) which are used to override
certain fields in the provided pipeline_config_path.  These are useful for
making small changes to the inference graph that differ from the training or
eval config.

Example Usage (in which we change the NMS iou_threshold to be 0.5 and
NMS score_threshold to be 0.0):
python object_detection/export_tflite_ssd_graph \
    --pipeline_config_path path/to/ssd_mobilenet.config \
    --trained_checkpoint_prefix path/to/model.ckpt \
    --output_directory path/to/exported_model_directory
    --config_override " \
            model{ \
            ssd{ \
              post_processing { \
                batch_non_max_suppression { \
                        score_threshold: 0.0 \
                        iou_threshold: 0.5 \
                } \
             } \
          } \
       } \
       "
"""