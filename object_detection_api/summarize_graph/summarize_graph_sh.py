#bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output/tflite_graph.pb
bazel build tensorflow/tools/graph_transforms:summarize_graph
bazel run --config=opt tensorflow/tools/graph_transforms:summarize_graph -- --in_graph=C:/Users/tt/Desktop/ssd_mobilenet_v1_coco_2018_01_28/output/tflite_graph.pb