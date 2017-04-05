# coding: utf-8
from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np

drivers = np.loadtxt('drivers_5000.csv', delimiter = ',', dtype=np.float32)

X = drivers[:,0:14]
Y = drivers[:,14]

model = Sequential()
# first arg is num of neurones
model.add(Dense(21, input_dim=14, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=150, batch_size=10)

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
