import tensorflow as tf
import keras
from tensorflow.keras.callbacks import TensorBoard
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

tf.compat.v1.disable_eager_execution()

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test,axis = 1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer = "adam" , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'], experimental_run_tf_function=False)
model.fit(x_train,y_train,epochs = 3 )

#Test part

img = Image.open(r"number3.png")
#convert to gray
gray_img = img.convert('L')
final_img = gray_img.point(lambda  x: 0 if x<100 else 255, '1')
final_img.save("final_img.png")

#rescale into smaller image
img_ar = cv2.imread("final_img.png", cv2.IMREAD_GRAYSCALE)
img_ar = cv2.bitwise_not(img_ar)
size = 28
new_img_ar = cv2.resize(img_ar, (size, size))
plt.imshow(new_img_ar, cmap = plt.cm.binary)
plt.show()

tester = tf.keras.utils.normalize(new_img_ar, axis = 1)
prediction = model.predict([[tester]])
a = prediction[0][0]
for i in range(0,10):
    b = prediction[0][i]
    print("Probability distribution for ", i, " is ",b)

print("Prediction:",np.argmax(prediction[0]))