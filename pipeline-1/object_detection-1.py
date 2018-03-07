import tensorflow as tf
import numpy as np
import time
import copy
import sys
import cv2

sys.path.append('./models/research')
sys.path.append('./models/research/object_detection')

from tensorflow.core.framework import graph_pb2
from utils import label_map_util


# Convert model
def _node_name(n):
  if n.startswith("^"):
    return n[1:]
  else:
    return n.split(":")[0]

input_graph = tf.Graph()
with tf.Session(graph=input_graph):
    score = tf.placeholder(tf.float32, shape=(None, 1917, 90), name="Postprocessor/convert_scores")
    expand = tf.placeholder(tf.float32, shape=(None, 1917, 1, 4), name="Postprocessor/ExpandDims_1")
    for node in input_graph.as_graph_def().node:
        if node.name == "Postprocessor/convert_scores":
            score_def = node
        if node.name == "Postprocessor/ExpandDims_1":
            expand_def = node

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile('./ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb', 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    dest_nodes = ['Postprocessor/convert_scores','Postprocessor/ExpandDims_1']

    edges = {}
    name_to_node_map = {}
    node_seq = {}
    seq = 0
    for node in od_graph_def.node:
      n = _node_name(node.name)
      name_to_node_map[n] = node
      edges[n] = [_node_name(x) for x in node.input]
      node_seq[n] = seq
      seq += 1

    for d in dest_nodes:
      assert d in name_to_node_map, "%s is not in graph" % d

    nodes_to_keep = set()
    next_to_visit = dest_nodes[:]
    while next_to_visit:
      n = next_to_visit[0]
      del next_to_visit[0]
      if n in nodes_to_keep:
        continue
      nodes_to_keep.add(n)
      next_to_visit += edges[n]

    nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: node_seq[n])

    nodes_to_remove = set()
    for n in node_seq:
      if n in nodes_to_keep_list: continue
      nodes_to_remove.add(n)
    nodes_to_remove_list = sorted(list(nodes_to_remove), key=lambda n: node_seq[n])

    keep = graph_pb2.GraphDef()
    for n in nodes_to_keep_list:
      keep.node.extend([copy.deepcopy(name_to_node_map[n])])

    remove = graph_pb2.GraphDef()
    remove.node.extend([score_def])
    remove.node.extend([expand_def])
    for n in nodes_to_remove_list:
      remove.node.extend([copy.deepcopy(name_to_node_map[n])])

    with tf.device('/gpu:0'):
      tf.import_graph_def(keep, name='')
    with tf.device('/cpu:0'):
      tf.import_graph_def(remove, name='')


# Read label
NUM_CLASSES = 90
label_map = label_map_util.load_labelmap('./models/research/object_detection/data/mscoco_label_map.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

# Inference
with detection_graph.as_default():
  with tf.Session(graph=detection_graph,config=config) as sess:
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    score_out = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
    expand_out = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
    score_in = detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
    expand_in = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    gst = "v4l2src device=/dev/video0 ! video/x-raw, width=(int)640, height=(int)480, format=RGB ! videoconvert ! appsink"
    cap = cv2.VideoCapture(gst)

    if not cap.isOpened():
        print "Failed to open camera."
        sys.exit(0)
    cv2.namedWindow("MyCameraPreview", cv2.WINDOW_AUTOSIZE)

    while True:
        start_time = time.time()
        ret, img = cap.read()
        image_np_expanded = np.expand_dims(img, axis=0)

        (score, expand) = sess.run([score_out, expand_out], feed_dict={image_tensor: image_np_expanded})
        (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={score_in:score, expand_in: expand})

        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        for i in range(min(20, boxes.shape[0])):
            if scores[i]>0.5:
                up_x = int(boxes[i][1] * img.shape[1]);
                up_y = int(boxes[i][0] * img.shape[0]);
                bottom_x = int(boxes[i][3] * img.shape[1]);
                bottom_y = int(boxes[i][2] * img.shape[0]);
                if classes[i] in category_index.keys(): name = category_index[classes[i]]['name']
                else: name = 'N/A'
                img = cv2.rectangle(img,(up_x,up_y),(bottom_x,bottom_y),(0,255,0),3)
                cv2.putText(img, name, (up_x,up_y), cv2.FONT_ITALIC, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow('MyCameraPreview', img)
        cv2.waitKey(1)
        print 'Execution time: %.2f sec'%(time.time()-start_time)

    cap.release()
    cv2.destroyAllWindows()
