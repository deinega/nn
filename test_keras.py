from keras import models
from keras import layers
from keras import optimizers
from keras.datasets import mnist
from keras import utils

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
input_tensor = train_images.reshape((60000, 28*28)).astype('float32')/255
#target_tensor = train_labels.reshape((60000, 1))
target_tensor = utils.to_categorical(train_labels, 10)

test_images = test_images.reshape((10000, 28*28)).astype('float32')/255
test_labels = utils.to_categorical(test_labels, 10)

print(input_tensor.shape)
print(target_tensor.shape)

print(test_images.shape)
print(test_labels.shape)

model = models.Sequential()

model.add(layers.Dense(512, activation='relu', input_shape=(784,)))
model.add(layers.Dense(10, activation='softmax'))
#model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='mse', metrics=['accuracy'])
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(input_tensor, target_tensor, batch_size=128, epochs=1, validation_data=(test_images, test_labels))

score = model.evaluate(test_images, test_labels, verbose=0)
print('Evaluated')
print('Test loss:', score[0])
print('Test accuracy:', score[1])