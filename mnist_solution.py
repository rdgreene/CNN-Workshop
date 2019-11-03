# this defines the input size of the images we will be feeding into our model
target_size = 28

# create an instance of a sequential model
model = models.Sequential()

# first block of convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(target_size, target_size, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# final dense layer for classification
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# train the model using 20% of the data for validation
history = model.fit(X, y, batch_size=256, epochs = 5, validation_split = 0.2)