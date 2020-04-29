import tensorflow as tf
import os
from glob import glob
import numpy as np
from preprocessing import imageSet
from datetime import datetime
from tensorflow.keras.utils import plot_model


class PneumoniaModel(tf.keras.Model):
    def __init__(self):
        super(PneumoniaModel,self).__init__()
        #Define layers:
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(32, kernel_size=3, activation='softmax')
        self.flatten = tf.keras.layers.Flatten()
        self.denseOutput = tf.keras.layers.Dense(3,activation=tf.nn.relu)

    def call(self,inputs):
        #Manipulation between layer:
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        return self.denseOutput(x)


input_shape = (100,100)
batch_size = 50
epochs = 3

working_dir = os.path.dirname(os.path.realpath(__file__))

#Data flow from various sources
#For initial test import saved numpys from DoG calculations
def getXy(pathName): 
    data_dir = os.path.join(working_dir,'chest_xray',pathName)   
    negative = imageSet(True,os.path.join(data_dir,'NORMAL'),input_shape)
    positive = imageSet(True,os.path.join(data_dir,'PNEUMONIA'),input_shape)

    X = np.concatenate([negative.X,positive.X])
    y = np.concatenate([negative.y,positive.y])
    X = X.reshape(len(X),input_shape[0],input_shape[1],1)
    return X,y


log_dir = "logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


#Define and compile a PneumoniaModel:
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss = tf.keras.losses.MeanAbsoluteError()
currentModel = PneumoniaModel()
print('Defined model')
currentModel.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
print('Compiled model')


X_train,y_train = getXy('train')
X_val,y_val = getXy('val')
X_test,y_test = getXy('test')

start = datetime.now()


currentModel.fit(x=X_train,y=y_train,batch_size=batch_size,epochs = epochs, validation_data=(X_val,y_val),shuffle=True,callbacks=[tensorboard_callback])
#currentModel.fit(x=X_train,y=y_train,batch_size=batch_size,epochs=epochs,shuffle=True)
print('fit model')
end = datetime.now()
print('Train time',str(end-start))

currentModel.evaluate(x=X_test,y=y_test,batch_size=batch_size)