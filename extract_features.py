import csv 
import numpy as np
from keras.datasets import cifar100
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

model = VGG16(weights='imagenet', include_top=False)

with open('cifar100/cifar_train.csv', mode='w') as cifar_image:
    cifar_writer = csv.writer(cifar_image, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for img, label in zip(x_train, y_train):
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feature = np.array(model.predict(img)) 
        data = [str(x) for x in feature.flatten()]
        data.insert(0, str(label[0]))
        cifar_writer.writerow(data)

with open('cifar100/cifar_test.csv', mode='w') as cifar_image:
    cifar_writer = csv.writer(cifar_image, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for img, label in zip(x_test, y_test):
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feature = np.array(model.predict(img)) 
        data = [str(x) for x in feature.flatten()]
        data.insert(0, str(label[0]))
        cifar_writer.writerow(data)