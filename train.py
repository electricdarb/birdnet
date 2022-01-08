import tensorflow as tf
import utils

TRAIN_DIR = './cub200data/CUB_200_2011/train'
TEST_DIR = './cub200data/CUB_200_2011/test'

CLASSES = 200

BATCH_SIZE = 64
TARGET_SIZE = (256, 256)
EPOCHS = 30

# optimizer 
MOMENTUM = .9
INIT_LR = 0.005
OPT = tf.keras.optimizers.SGD(INIT_LR, momentum = MOMENTUM)

LR_CB = tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor = .3, patience = 3, verbose = 1, min_lr = INIT_LR/100)

# loss
LABEL_SMOOTHING = 0.1
LOSS = tf.keras.losses.CategoricalCrossentropy(label_smoothing = LABEL_SMOOTHING)
WEIGHT_DECAY = 1e-5 

DROPOUT_RATE = .5

METRICS = [
    tf.keras.metrics.CategoricalAccuracy(),
    tf.keras.metrics.Recall(),
    tf.keras.metrics.Precision()
]

CALLBACKS = [LR_CB]

def main():
    model = utils.load_model('base_frozen_220108171850')

    model.layers[1].trainable = True

    for layer in model.layers[1].layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)): # check if it a conv layer and add dropout
            layer.kernel_regularizer = tf.keras.regularizers.l2(WEIGHT_DECAY)
            
    model.layers[3].rate = DROPOUT_RATE

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 15,
        height_shift_range = .1,
        zoom_range = .1,
        horizontal_flip = True,
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1./255
    )

    train_gen = train_datagen.flow_from_directory(TRAIN_DIR, batch_size = BATCH_SIZE, target_size = TARGET_SIZE)
    val_gen = test_datagen.flow_from_directory(TEST_DIR, batch_size = BATCH_SIZE, target_size = TARGET_SIZE)

    model.compile(optimizer = OPT,
        loss = LOSS,
        metrics = METRICS
        )

    runname = utils.make_runname('finetuned')

    history = model.fit(train_gen, 
        validation_data = val_gen,
        epochs = EPOCHS, 
        callbacks = CALLBACKS)
    
    utils.log_history(runname, history)
    utils.save_model(runname, model)
    print(f'run name: {runname}')

if __name__ == "__main__":
    main()
