Object Detection Sample on Jetson
======================================
Please get the requirements installed with the information shared in 'requirement'.
</br>
</br>

# Pipeline-1: TensorFlow Object Detection API
**Dependency**
```C
$ sudo apt-get install git protobuf-compiler python-pil python-lxml python-tk
```

**Object Detection API**

```C
$ git clone https://github.com/tensorflow/models.git
$ cd models/research/
$ protoc object_detection/protos/*.proto --python_out=.
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
$ cd ../../
```

**Download SSD Model**
```C
$ wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
$ tar -xvzf ssd_mobilenet_v1_coco_2017_11_17.tar.gz 
```

**Test Pipeline 1**
```C
$ python object_detection-1.py
```
</br>
</br>
</br>
</br>
</br>
