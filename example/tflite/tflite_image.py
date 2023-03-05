import numpy as np
import tensorflow as tf
import cv2

ON_IMAGE = True

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return 2. / (1 + np.exp(-2 * x)) - 1

def preprocess(src_img, size):
    # 352x352x3
    output = cv2.resize(src_img,(size[0], size[1]),interpolation=cv2.INTER_AREA)
    # 3,352,352
    output = output.transpose(2,0,1)
    # 0-255 -> 0-1
    output = output.reshape((1, 3, size[1], size[0])) / 255
    #  float32[3,352,352]
    return output.astype('float32')

def nms(dets, thresh=0.45):
    try :
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1) 
        order = scores.argsort()[::-1]  
        keep = []  
    except :
        return []

    while order.size > 0:
        i = order[0]  
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    
    output = []
    for i in keep:
        output.append(dets[i].tolist())

    return output

def detection(feature_map, img, thresh):
    pred = []
    H, W, _ = img.shape
    feature_map = feature_map.transpose(1, 2, 0)
    feature_map_height = feature_map.shape[0]
    feature_map_width = feature_map.shape[1]
    for h in range(feature_map_height):
        for w in range(feature_map_width):
            data = feature_map[h][w]
            obj_score, cls_score = data[0], data[5:].max()
            score = (obj_score ** 0.6) * (cls_score ** 0.4)
            if score > thresh:
                cls_index = np.argmax(data[5:])
                x_offset, y_offset = tanh(data[1]), tanh(data[2])
                box_width, box_height = sigmoid(data[3]), sigmoid(data[4])
                box_cx = (w + x_offset) / feature_map_width
                box_cy = (h + y_offset) / feature_map_height
                
                # cx,cy,w,h => x1, y1, x2, y2
                x1, y1 = box_cx - 0.5 * box_width, box_cy - 0.5 * box_height
                x2, y2 = box_cx + 0.5 * box_width, box_cy + 0.5 * box_height
                x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
                pred.append([x1, y1, x2, y2, score, cls_index])

    return nms(np.array(pred))


def load_image(path) :
    return cv2.imread(path, cv2.IMREAD_ANYCOLOR)

def resize(image):
    return cv2.resize(image, (352,352))

def setuptflite_model(path):   
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

def get_input_details(interpreter):
    return interpreter.get_input_details()

def get_output_details(interpreter):
    return interpreter.get_output_details()

def get_model_details(interpreter) :
    return get_input_details(interpreter), get_output_details(interpreter)
     
def predict(interpreter, input_details, input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

if __name__ == '__main__':
    MODEL_PATH = "custom_fasestdet.tflite"
    IMG_PATH = "test1.png"
    
    interpreter = setuptflite_model(MODEL_PATH)
    # Get input and output tensors.
    input_details, output_details = get_model_details(interpreter)

    sample = None
    raw_image = None
    input_shape = input_details[0]['shape']
    print("Input Shape " ,input_shape)
    if ON_IMAGE:
        # sample image
        sample_image = load_image(IMG_PATH)
        sample_image = resize(sample_image)
        raw_image = sample_image
        sample_image = preprocess(sample_image, [352, 352])
        # sample_image = np.expand_dims(sample_image,0)
        # sample_image = sample_image.transpose(0,3,2,1)
        print(sample_image.shape)
        sample = sample_image
    else:
        sample = np.random.random_sample(input_shape)

    input_data = np.array(sample, dtype=np.float32)
    output_data = predict(interpreter, input_details, input_data)
    output_data = np.squeeze(output_data)
    outs = detection(output_data,raw_image, 0.7)
    for b in outs:
        print(b)
        obj_score, cls_index = b[4], int(b[5])
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])

        cv2.rectangle(raw_image, (x1,y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(raw_image, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
        # cv2.putText(raw_image, "names[cls_index]", (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

    cv2.imwrite("result.jpg", raw_image)
