import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import model_from_json

batch_size = 25
epochs = 5
n = 1000
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
test = model.predict(X_test)


print('Test loss:', score[0])
print('Test accuracy:', score[1])

#serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to hdf5
model.save_weights("model.h5")
print ("Saved model to disk")

#load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

#evaluate loaded model on test data
values = loaded_model.predict(X_test)
print (values)




