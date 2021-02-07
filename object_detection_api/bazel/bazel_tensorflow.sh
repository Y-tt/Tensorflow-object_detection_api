# using tensorflow==1.13.1  bazel==0.19.1
cd tensorflow-1.13.1/   
python ./configure.py
bazel build tensorflow/python/tools:freeze_graph
bazel build tensorflow/lite/toco:toco
