#bazel run --config=opt tensorflow/lite/toco:toco -- --input_file=C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output/tflite_graph.pb --output_file=C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output/detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  --inference_type=FLOAT --allow_custom_ops


#bazel run --config=opt tensorflow/lite/toco:toco -- --input_file=C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output/tflite_graph.pb --output_file=C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output/detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess'  --inference_type=FLOAT --allow_custom_ops

bazel run --config=opt tensorflow/lite/toco:toco -- --input_file=C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output/tflite_graph.pb --output_file=C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output/detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays=TFLite_Detection_PostProcess  --inference_type=FLOAT --allow_custom_ops

--output_arrays=TFLite_Detection_PostProcess

#floating point model
bazel run --config=opt tensorflow/lite/toco:toco -- \
--input_file=$OUTPUT_DIR/tflite_graph.pb \
--output_file=$OUTPUT_DIR/detect.tflite \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
--inference_type=FLOAT \
--allow_custom_ops

#quantized model
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