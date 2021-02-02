
from io import BytesIO, StringIO
from typing import Union
import tensorflow
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate,Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D,AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize
import random
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Multiply
import pandas as pd
import streamlit as st
from tempfile import NamedTemporaryFile

    




def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tensorflow.to_int32(y_pred > t)
        score, up_opt = tensorflow.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tensorflow.local_variables_initializer())
        with tensorflow.control_dependencies([up_opt]):
            score = tensorflow.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

dependencies = {
    'mean_iou': mean_iou
}
model = tensorflow.keras.models.load_model('first_model.h5',custom_objects=dependencies)
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])


def import_and_predict(image_data, model):
    
    X_test = np.zeros((1, 128, 128, 1), dtype=np.uint8)
    sizes_test = []
    #img = load_img(image_data)
    x = img_to_array(image_data)[:,:,1]
    sizes_test.append([x.shape[0], x.shape[1]])
    x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
    X_test[0] = x
    preds_test = model.predict(X_test, verbose=1)
    preds_test_t = (preds_test > 0.5).astype(np.uint8)
    tmp = np.squeeze(preds_test_t[0]).astype(np.float32)
    return np.dstack((tmp,tmp,tmp))
    
    
        
if file is None:
    st.text("Please upload an image file")
else:
    
    image = Image.open(file)
    #image = image.convert('LA')
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    st.image(prediction)    