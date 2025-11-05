import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

PATH_TO_TRAIN_DATA = "/content/drive/My Drive/cnn/data/train_data"
PATH_TO_TEST_DATA = "/content/drive/My Drive/cnn/data/test_data"
BATCH_SIZE = 128

if __name__ ==  '__main__':
    def get_datagen(dataset, aug=False):
        if aug:
            datagen = ImageDataGenerator(
                        rescale=1./255,
                        rotation_range=25,
                        width_shift_range=0.3,
                        height_shift_range=0.3,
                        shear_range=0.5,
                        zoom_range=0.3,
                        horizontal_flip=False,
                        brightness_range=[0.8,1.1])
        else:
            datagen = ImageDataGenerator(rescale=1./255)
        return datagen.flow_from_directory(
                    dataset,
                    target_size=(224, 224),
                    color_mode='rgb',
                    shuffle = True,
                    class_mode='categorical',
                    batch_size=BATCH_SIZE)

    train_generator  = get_datagen(PATH_TO_TRAIN_DATA, True)
    test_generator   = get_datagen(PATH_TO_TEST_DATA, False)

    def create_vgg16_model():
        pretrained_model = VGG16(include_top=False,
                                input_shape=(224,224,3),
                                pooling='avg',
                                weights='imagenet')
        
        for layer in pretrained_model.layers[:15]:
            layer.trainable = False

        vgg_model = pretrained_model.output

        vgg_model = Dropout(0.5)(vgg_model)
        vgg_model = Dense(4096, activation='relu')(vgg_model)
        vgg_model = Dropout(0.5)(vgg_model)
        vgg_model = Dense(1024, activation='relu')(vgg_model)
        vgg_model = Dropout(0.5)(vgg_model)
        vgg_model = Dense(12, activation='softmax')(vgg_model)

        model = Model(pretrained_model.input, vgg_model, name='VGG16_Fine_Tuned')
        return model

    model = create_vgg16_model()
    model.summary()

    adam = tf.keras.optimizers.Adam(learning_rate=0.00002)

    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, verbose=1, restore_best_weights = True)
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)

    model.compile(
            loss='categorical_crossentropy', 
            optimizer=adam, 
            metrics=['accuracy'])

    history = model.fit(
        train_generator,
        validation_data=test_generator,
        shuffle=True,
        epochs=50,
        callbacks=[early_stopper, rlrop],
    )

    model.save("/content/drive/My Drive/cnn/VGG16-final.keras")