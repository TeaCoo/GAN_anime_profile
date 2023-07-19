import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv2DTranspose, Conv2D, LeakyReLU, Flatten, Dropout, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
"""
Create a GAN net work to generate anime profile image
image size: 64x64
latent_dim: 50
We need a generator and a discriminator
Generator input is a (50, 1) random vector and output a 64x64x3 image
Predictor input is a 64x64x3 image and output a score (binary cross entropy 0 or 1)
"""


def gan_generator(latent_dim=100):
    """
    generator is a up scale multi-layer network
    input shape is (latent_dim, 1)
    :param latent_dim: the shape of vector
    :param image_shape: the shape of generated image
    :return: model of generator, shape is (64x64x3)
    """
    model = Sequential()
    n_nodes = 8 * 8 * 128
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((8, 8, 128)))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(3, (8, 8), activation='tanh', padding='same'))
    return model


def gan_discriminator(input_shape=(64, 64, 3)):
    """
    build discriminator
    :param input_shape: input image shape is 64x64x3
    :return: discriminator model
    """
    model = Sequential()
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def train_gan(g_model, d_model, gan_model, dataset, image_shape, latent_dim,
              epoch=100, batch_size=128, load_weight=False):
    """
    train GAN model need two for loop, one for epoch, one for batches.
    use train_on_batch to train the model.
    :param input_data: all input training data with shape 64x64x3
    :param epoch: the epoch for training
    :param batch_size: the batch size for each epoch
    :return: the training history/loss
    """
    bat_per_epo = int(dataset.shape[0] / batch_size)
    half_batch = int(batch_size / 2)
    epoch_loss_d1 = []
    epoch_loss_d2 = []
    epoch_loss_g = []

    if load_weight is not False:
        gan_model.load_weights("./GAN_model_100.h5")

    for e in range(epoch):
        d1 = 0
        d2 = 0
        d3 = 0
        if e % 200 == 0:
            predict_model(gan_model, current_index=e)
        for b in range(bat_per_epo):
            # 1. sample real image
            real_x, real_y = get_real_data(dataset, image_shape, half_batch)

            # 2. sample fake image
            fake_x, fake_y = get_fake_data(g_model, latent_dim, half_batch)

            # 3. train discriminator using real and fake images
            d_loss_true, _ = d_model.train_on_batch(real_x, real_y)
            d_loss_fake, _ = d_model.train_on_batch(fake_x, fake_y)

            # 4. generate some random vector for generator
            gan_x = get_random_vector((batch_size, latent_dim))
            gan_y = np.ones((batch_size, 1))

            # 5. train generator
            g_loss = gan_model.train_on_batch(gan_x, gan_y)

            # 6. save model's weight
            gan_model.save("./GAN_model.h5")
            print("Epoch[", e, "/", epoch, "]",
                  "Batch[", b, "/", bat_per_epo,
                  "]  d_loss_true=", d_loss_true, " d_loss_fake=", d_loss_fake, " g_loss=", g_loss)
            d1 += d_loss_true
            d2 += d_loss_fake
            d3 += g_loss
        epoch_loss_d1.append(d1/bat_per_epo)
        epoch_loss_d2.append(d2/bat_per_epo)
        epoch_loss_g.append(d3/bat_per_epo)
    show_training_loss_image(epoch_loss_d1, epoch_loss_d2, epoch_loss_g)


def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)

    return model


def get_real_data(dataset, image_shape, batch_size):
    random_image = dataset[np.random.choice(dataset.shape[0], size=batch_size, replace=False)]

    x_train = []
    for item in random_image:
        image = Image.open(item)
        resized_image = image.resize((image_shape[0], image_shape[1]))
        x_train.append(np.array(resized_image))
    x_train = np.array(x_train)/127.5 - 1
    y_train = np.ones((batch_size, 1))
    return x_train, y_train


def get_fake_data(g_model, latent_dim, batch_size):
    random_vector = get_random_vector((batch_size, latent_dim))
    fake_images = g_model.predict(random_vector)
    fake_y = np.zeros((batch_size, 1))
    return fake_images, fake_y


def get_random_vector(latent_dim):
    random_numbers = np.random.normal(loc=0, scale=1, size=latent_dim)
    random_numbers_clipped = np.clip(random_numbers, -1, 1)
    return random_numbers_clipped


def load_dataset(path):
    images = os.listdir(path)
    images = [item for item in images if "_" in item]
    images = [os.path.join(path, item) for item in images]
    return np.array(images)


def show_images_debug(image_array, current_index):
    num_images = image_array.shape[0]
    rows = 8
    cols = num_images // rows

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for i in range(num_images):
        row = i // cols
        col = i % cols
        axes[row, col].imshow(image_array[i])
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig("../test/test_" + str(current_index) + ".png")
    plt.close(fig)


def show_training_loss_image(data1, data2, data3):
    x = np.arange(len(data1))
    plt.plot(x, data1, marker='', linestyle='-', color='r')
    plt.plot(x, data2, marker='', linestyle='-', color='b')
    plt.plot(x, data3, marker='', linestyle='-', color='g')

    plt.title('Line Chart Example')
    plt.xlabel('Epoch')
    plt.ylabel('Training loss')
    plt.show()


def predict_model(model, current_index=0):
    test_input = get_random_vector((64, latent_dim))
    test_output = ge.predict(test_input)
    # convert image data from range -1 to 1 into 0 to 255
    rgb_image = ((test_output + 1) * 127.5).astype(int)
    show_images_debug(rgb_image, current_index)


latent_dim = 100
epoch = 10000
batch_size = 512
image_shape = (64, 64, 3)

di = gan_discriminator(input_shape=image_shape)
di.summary()

ge = gan_generator(100)
ge.summary()

gan_model = build_gan(ge, di)

image_path = "D:\\GAN_anime_profile\\data\\images"
dataset = load_dataset(image_path)

train_gan(ge, di, gan_model, dataset,
          image_shape=image_shape, latent_dim=latent_dim, epoch=epoch, batch_size=batch_size, load_weight=True)
# gan_model.load_weights("./GAN_model_100.h5")
# predict_model(gan_model, current_index=1)







