import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from sklearn import preprocessing

batch_size = 25
epochs = 10
n = 2000
en = n - 50

K.set_image_dim_ordering('tf')

train_data = np.load("traningdata-1.npy")

X_all = train_data[:, 0]
y_all = train_data[:, 1]

npX = np.array([])
npY = np.array([])

for x in X_all:
    npX = np.append(npX, x)
for y in y_all:
    npY = np.append(npY, y)

X_all = npX.reshape(-1, 60, 80, 1)
y_all = npY.reshape(-1, 3)

X_train, X_test = X_all[:en], X_all[en:]
y_train, y_test = y_all[:en], y_all[en:]

model = Sequential()
model.add(Conv2D(32, input_shape=(60, 80, 1), activation='relu', kernel_size=(3,3)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(3, activation='tanh'))


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(), metrics=['accuracy'])


model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])



