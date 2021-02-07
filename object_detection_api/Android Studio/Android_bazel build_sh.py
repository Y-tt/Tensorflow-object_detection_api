 bazel build -c opt --config=android_arm{,64} --cxxopt='--std=c++11' "//tensorflow/lite/examples/android:tflite_demo" 
