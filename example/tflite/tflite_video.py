import example_tflite as helper
import numpy as np
import tensorflow as tf
import cv2
import time

MODEL_PATH = "custom_fasestdet.tflite""

cap = cv2.VideoCapture('my_video-12.mkv')

interpreter =  helper.setuptflite_model(MODEL_PATH)
input_details, output_details = helper.get_model_details(interpreter)

while(cap.isOpened()):
    start_time = time.time() # start time of the loop
    ret, frame = cap.read()
    ori_frame = helper.resize(frame.copy())
    frame = helper.preprocess(frame, [352,352])
    frame = np.array(frame, dtype=np.float32)
    output_data = helper.predict(interpreter, input_details, frame)
    output_data = np.squeeze(output_data)
    outs = helper.detection(output_data,ori_frame, 0.6)
    FPS = int(1.0 / (time.time() - start_time))
    print("FPS: ", FPS) # FPS = 1 / time to process loop
    for b in outs:
        print(b)
        obj_score, cls_index = b[4], int(b[5])
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])

        cv2.rectangle(ori_frame, (x1,y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(ori_frame, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
        cv2.putText(ori_frame, f"{FPS}", (30,30), 0, 0.7, (0, 255, 0), 2)


    cv2.imshow('frame',ori_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()