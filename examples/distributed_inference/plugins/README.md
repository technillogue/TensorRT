## Build instructions

```
export TENSORRT_INCLUDE_DIR=~/Downloads/TensorRT-9.2.0.0.Linux.x86_64-gnu.cuda-12.2/TensorRT-9.2.0.0/include/
export TENSORRT_LIB_DIR=~/Downloads/TensorRT-9.2.0.0.Linux.x86_64-gnu.cuda-12.2/TensorRT-9.2.0.0/lib/
cmake .. -DTENSORRT_INCLUDE_DIR=$TENSORRT_INCLUDE_DIR -DTENSORRT_LIB_DIR=$TENSORRT_LIB_DIR
```
