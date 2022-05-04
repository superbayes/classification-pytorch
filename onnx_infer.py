from cv2 import COLOR_BGR2RGB, IMREAD_COLOR
import numpy as np
import onnx
import onnxruntime as ort
import cv2 as cv

ort_session = ort.InferenceSession("model_data\logocls8_mobileNetv2_alpha075_train134_val087_0502.onnx")
img = cv.imread("img\ocv_logo_rgb_224x224.jpg",IMREAD_COLOR)
img = cv.cvtColor(img,COLOR_BGR2RGB)
img = np.array(img,dtype=np.float32)
img = (img - 128.0)/128.0
img = np.transpose(img,(2,0,1))
img = img[np.newaxis,:]

outputs = ort_session.run(
    output_names=["output"],
    input_feed = {"input": img.astype(np.float32)}
)
print(outputs[0])
