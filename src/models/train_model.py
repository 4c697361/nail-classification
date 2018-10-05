import os
import sys
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import tensorflow as tf


import src.utils.utils as ut
import src.models.model as md


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k,
                    v in logs.items() if k.startswith('val_')}

        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


def train(modelname,
          batch_size,
          epochs,
          learning_rate,
          augment,
          image_width,
          image_heigth):
    input_shape = (image_width, image_heigth, 3)

    # training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True)
    # testing data augmentation (only rescaling)
    test_datagen = ImageDataGenerator(rescale=1./255)

    if(augment is False):
        train_datagen = test_datagen

    # training data generator
    train_generator = train_datagen.flow_from_directory(
        ut.dirs.train_dir,
        target_size=(image_width, image_heigth),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='binary')
    # validation data generator
    validation_generator = test_datagen.flow_from_directory(
        ut.dirs.validation_dir,
        target_size=(image_width, image_heigth),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='binary')

    mod = md.custom_models(input_shape, 1)

    if(modelname == 'vgg16'):
        model = mod.vgg16()
    elif(modelname == "cnn"):
        model = mod.CNN()
    else:
        print('invalid model selection.\n\
               please choose from one of the available models:\n\
                 vgg16, cnn')
        sys.exit()

    # Do not forget to compile it
    model.compile(
                    loss='binary_crossentropy',
                    optimizer=optimizers.RMSprop(lr=learning_rate),
                    metrics=['accuracy']
                    )

    model.summary()

    save_model_to = os.path.join(ut.dirs.model_dir, modelname + '.h5')

    Checkpoint = ModelCheckpoint(save_model_to,
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=False,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)
    Earlystop = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=0,
                              mode='auto',
                              baseline=None)

    model.fit_generator(
        train_generator,
        callbacks=[
                    TrainValTensorBoard(write_graph=False),
                    Checkpoint#,
                    #Earlystop
                    ],
        steps_per_epoch=150//batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=12//batch_size
    )


@click.command()
@click.option('--modelname', type=str, default='vgg16',
                    help='choose a model:\n\
                            vgg16:  pretrained vgg16\n\
                            cnn:    simple CNN\n\
                            (default: vgg16)')
@click.option('--ep', type=float, default=ut.params.epochs,
                    help='number of epochs (default: {})'.
                    format(ut.params.epochs))
@click.option('--lr', type=float, default=ut.params.learning_rate,
                    help='learning rate (default: {})'.
                    format(ut.params.learning_rate))
@click.option('--augment', type=int, default=1,
                    help='data augmentation\n\
                    0: False, 1: True (default: 1)')
@click.option('--bs', type=int, default=ut.params.batch_size,
                    help='batch size (default: {})'.
                    format(ut.params.batch_size))
@click.option('--width', type=int, default=ut.params.image_width,
                    help='width of the sample images (default: {})'.
                    format(ut.params.image_width))
@click.option('--heigth', type=int, default=ut.params.image_heigth,
                    help='heigth of the sample images (default: {})'.
                    format(ut.params.image_heigth))
def main(modelname, bs, ep, lr, augment, width, heigth):

    augmentation = True
    if(augment == 0):
        augmentation = False

    train(modelname, bs, ep, lr, augmentation, width, heigth)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
