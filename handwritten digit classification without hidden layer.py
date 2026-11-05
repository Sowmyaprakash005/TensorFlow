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

x_train_flattened=x_train.reshape(len(x_train),28*28) 
x_test_flattened=x_test.reshape(len(x_test),28*28) 
print(x_train_flattened.shape)
print(x_test_flattened.shape)
x_train_flattened[0]

#neural network without hidden layer
model=keras.Sequential([
    keras.layers.Dense(10,input_shape=(784,),activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(x_train_flattened,y_train,epochs=5)

model.evaluate(x_test_flattened,y_test)

plt.matshow(x_test[1])

y_predicted=model.predict(x_test_flattened)
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
