import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
len(x_train)
len(x_test)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x_train[0]
plt.matshow(x_train[0])

y_train[:5] 
x_train = x_train/255  
x_test = x_test/255 

#neural network with hidden layer
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(x_train,y_train,epochs=5)

model.evaluate(x_test,y_test)

plt.matshow(x_test[1])

y_predicted=model.predict(x_test)
y_predicted[1]
np.argmax(y_predicted[1]) 

y_predicted_labels=[np.argmax(i) for i in y_predicted]  
y_predicted_labels[:5]
y_test[:5]

#Confusion Matrix
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
cm

import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Truth")
