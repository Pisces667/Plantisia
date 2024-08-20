import tensorflow  as tf
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns

Training_set= tf.keras.utils.image_dataset_from_directory(
     'train',

    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    data_format=None,
    verbose=True,
)

validation_set= tf.keras.utils.image_dataset_from_directory(
     'train',

    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    data_format=None,
    verbose=True,
)
for x,y in Training_set:
    print(x,x.shape)
    print(y,y.shape)
    break


from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout
from tensorflow.keras.models import Sequential

model= Sequential ()

model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[256,256,3]))
model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[256,256,3]))
model.add(Conv2D(filters=32,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[256,256,3]))
model.add(Conv2D(filters=32,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[256,256,3]))
model.add(Conv2D(filters=32,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[256,256,3]))
model.add(Conv2D(filters=32,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))





model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(units=1800,activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(units=41,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
 
training_accuracy=model.fit(x=Training_set,validation_data=validation_set,epochs=10)

train_loss,train_acc=model.evaluate(Training_set)

Val_loss,val_acc=model.evaluate(validation_set)

print (Val_loss,val_acc)

model.save("trained_model.keras")

training_accuracy.history

import json

with open("training_acc.json","w") as f:
    json.dump(training_accuracy.history,f)

epochs={i for i in range (1,11)}
plt.plot(epochs,train_acc.history['accuracy'],color='blue',label="Training Accuracy")
plt.plot(epochs,val_acc.history['val_accuracy'],color='Green',label="Validation Accuracy")
plt.xlabel("No of epochs")
plt.ylabel("Result for Accuracy")
plt.title('Model accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

class_name= validation_set.class_name

test_set= tf.keras.utils.image_dataset_from_directory(
     'train',

    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=False,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    data_format=None,
    verbose=True,
)

y_pred= model.predict(test_set)
y_pred,y_pred.shape

predicted_categories= tf.argmax(y_pred,axis=1)

true_categories= tf.concat([y for x,y in test_set],axis=0)
y_true=tf.argsmax(true_categories,axis=1)

predicted_categories= tf.argmax(true_categories,axis=1)

from sklearn.metrics import classification_report,confusion_matrix
classification_report(y_true,predicted_categories,target_names=class_name)

cm=confusion_matrix(y_true,predicted_categories)
cm.shape
plt.figure(figsize=(40,40))
sns.heatmap(cm,annot=True,annot_kws=('size=12'))
plt.xlabel("Predicted class",fontsize=20)
plt.ylabel("Actual Class",fontsize=20)
plt.title("Detection forPlant disease confusion Matrix",fontsize=25)
plt.show()
