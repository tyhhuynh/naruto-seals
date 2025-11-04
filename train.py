import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceExtended

CollectiveAllReduceExtended._enable_check_health = False

PATH_TO_TRAIN_DATA = "data/train_data"
PATH_TO_TEST_DATA = "data/test_data"
BATCH_SIZE = 16
# TRAIN_SIZE = 3638
# TEST_SIZE = 368

if __name__ ==  '__main__':
  def get_datagen(dataset, aug=False, subset=None):
      if aug:
          datagen = ImageDataGenerator(
                              preprocessing_function=preprocess_input,
                              rotation_range=15,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              shear_range=0.1,
                              zoom_range=0.1,
                              horizontal_flip=True,
                              brightness_range=[0.9,1.1],
                              validation_split=0.15)
      else:
          datagen = ImageDataGenerator(
              preprocessing_function=preprocess_input,
              validation_split=0.15)

      return datagen.flow_from_directory(
              dataset,
              target_size=(224, 224),
              color_mode='rgb',
              shuffle = True,
              class_mode='categorical',
              batch_size=BATCH_SIZE,
              subset=subset)

  train_generator  = get_datagen(PATH_TO_TRAIN_DATA, True, 'training')
  val_generator = get_datagen(PATH_TO_TRAIN_DATA, False, 'validation')
  test_generator   = get_datagen(PATH_TO_TEST_DATA, False)

  def create_vgg16_model():
        pretrained_model = VGG16(include_top=False,
                                input_shape=(224,224,3),
                                pooling='avg',classes=5,
                                weights='imagenet')
        
        for layer in pretrained_model.layers[:15]:
            layer.trainable=False

        vgg_model = Flatten()(pretrained_model.output)
        vgg_model = Dropout(0.5)(vgg_model)
        vgg_model = Dense(4096, activation='relu')(vgg_model)
        vgg_model = Dropout(0.5)(vgg_model)
        vgg_model = Dense(1024, activation='relu')(vgg_model)
        vgg_model = Dropout(0.5)(vgg_model)
        vgg_model = Dense(12, activation='softmax')(vgg_model)

        model = Model(pretrained_model.input, vgg_model, name='VGG16_Model')
        return model

  model = create_vgg16_model()
  model.summary()

  adam = tf.keras.optimizers.Adam(learning_rate=0.00001)
  rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',mode='max',factor=0.5, patience=10, min_lr=0.001, verbose=1)
  early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1,
                                            mode='auto', baseline=None, restore_best_weights=True)
  model.compile(loss='categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])

  history = model.fit(
      train_generator,
      validation_data=val_generator, 
      shuffle=True,
      epochs=50,
      callbacks=[early_stopper],
  )

  model.save("./VGG_Naruto_Model.keras")
