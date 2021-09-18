import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from sklearn.model_selection import train_test_split

path = 'archive (1)/myData'
img_rows = 32
img_cols = 32
img_channels = 3
n_epochs = 20

datagen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             rotation_range=10,
                             shear_range=0.1,
                             rescale=1.0/255.0)

data = datagen.flow_from_directory(path,
                                   target_size=(32, 32),
                                   class_mode='categorical',
                                   shuffle=True,
                                   batch_size=73139)

X, y = data.next()
print('Data shape is {} and Labels shape is {}'.format(X.shape, y.shape))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print("Train Shape: {}\nValidation Shape: {}\nTest Shape : {}".format(X_train.shape, X_validation.shape, X_test.shape))

model = Sequential()
model.add(Conv2D(64, (5,5), input_shape=(32,32,3), activation='relu'))
model.add(Conv2D(64, (5,5), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (5,5), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(500, activation='relu',
                kernel_regularizer=regularizers.l2(l=0.016),
                activity_regularizer=regularizers.l1(0.006),
                bias_regularizer=regularizers.l1(0.006)))
model.add(Dropout(0.5))
model.add(Dense(43, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_check = ModelCheckpoint('best_model.h5', monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=0, mode='max', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
csv_logger = CSVLogger('train_log.csv', separator=',')

history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=n_epochs,
                    verbose=1,
                    validation_data=(X_validation, y_validation),
                    callbacks=[model_check, early, reduce_lr, csv_logger])


plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')

plt.show()


score = model.evaluate(X_test, y_test)
print('Test loss(Score) = {}\nTest Accuracy = {}'.format(score[0], score[1]*100))

model.save('my_model', save_format='tf')
