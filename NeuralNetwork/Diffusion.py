import os
from PIL import Image
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Embedding, Dense, Input, Conv2DTranspose, Conv2D, ReLU, MaxPooling2D, Concatenate, Reshape, Layer, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import math


def cos_beta(t, timestep):
    """
    From paper "Improved Denoising Diffusion Probabilistic Models"
    :param t: current t
    :param timestep: total timestep T
    :return: beta t
    """
    s = 0.008
    return math.cos((t / timestep + s) / (1 + s) * math.pi / 2) ** 2


def linear_beta(timestep, start=0.0001, end=0.02):
    return np.linspace(start, end, timestep)


def alpha_bar_t(t, timestep):
    return cos_beta(t, timestep) / cos_beta(0, timestep)


def set_beta_t(timestep):
    """
    alpha_bar_1 = [alpha_bar_t(t, timestep) for t in range(timestep)]
    alpha_bar_prev_1 = np.pad(alpha_bar_1[:-1], (1, 0), mode='constant', constant_values=1)
    alpha_1 = [alpha_bar_t(t, timestep) if t == 0 else alpha_bar_t(t, timestep)/alpha_bar_t(t-1, timestep) for t in range(timestep)]
    beta_1 = [1 - t for t in alpha_1]
    sigma_t_1 = [beta_1[t]*(1-alpha_bar_prev_1[t])/(1-alpha_bar_1[t]) for t in range(timestep)]
    """
    beta = linear_beta(timestep)
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha, axis=0)
    alpha_bar_prev = np.pad(alpha_bar[:-1], (1, 0), mode='constant', constant_values=1)
    sigma_t = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
    sqrt_alpha_bar = np.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = np.sqrt(1 - alpha_bar)

    alpha_t_tensor = tf.constant(alpha, dtype=tf.float32)
    alpha_bar_tensor = tf.constant(alpha_bar, dtype=tf.float32)
    beta_t_tensor = tf.constant(beta, dtype=tf.float32)
    sigma_t_tensor = tf.constant(sigma_t, dtype=tf.float32)
    sqrt_alpha_bar_tensor = tf.constant(sqrt_alpha_bar, dtype=tf.float32)
    sqrt_one_minus_alpha_bar_tensor = tf.constant(sqrt_one_minus_alpha_bar, dtype=tf.float32)

    return alpha_t_tensor, alpha_bar_tensor, beta_t_tensor, sigma_t_tensor, sqrt_alpha_bar_tensor, sqrt_one_minus_alpha_bar_tensor


class NoiseLayer(Layer):
    def __init__(self):
        super(NoiseLayer, self).__init__()

    def call(self, inputs):
        image = inputs
        shape = tf.shape(image)

        if IsTraining:
            t = tf.random.uniform(shape=(shape[0],), minval=0, maxval=diffusion_timestep, dtype=tf.float32)
        else:
            # t = tf.random.uniform(shape=(shape[0],), minval=0, maxval=diffusion_timestep, dtype=tf.float32)
            t = tf.fill((shape[0],), value=diffusion_timestep-1)
        t = tf.cast(t, dtype=tf.int32)

        noise = tf.random.normal(shape=shape, mean=0.0, stddev=1.0)

        total_elements = tf.reduce_prod(shape)

        alpha_t = tf.gather(sqrt_alpha, t)
        beta_t = tf.gather(sqrt_one_alpha, t)

        alpha_t = tf.repeat(alpha_t[:, tf.newaxis], total_elements // shape[0], axis=-1)
        beta_t = tf.repeat(beta_t[:, tf.newaxis], total_elements // shape[0], axis=-1)

        alpha_t = tf.reshape(alpha_t, shape)
        beta_t = tf.reshape(beta_t, shape)

        add_noise = alpha_t * image + beta_t * noise
        add_noise = tf.tanh(add_noise)
        noise = tf.tanh(noise)

        return add_noise, noise, t


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
    # plt.savefig("../test/test_" + str(current_index) + ".png")
    plt.show()
    plt.close(fig)


def get_image_data(dataset, image_shape, batch_size):
    random_image = dataset[np.random.choice(dataset.shape[0], size=batch_size, replace=False)]

    x_train = []
    for item in random_image:
        image = Image.open(item)
        resized_image = image.resize((image_shape[0], image_shape[1]))
        x_train.append(np.array(resized_image))
    x_train = np.array(x_train)/127.5 - 1
    return x_train


def load_dataset(path):
    images = os.listdir(path)
    images = [item for item in images if "_" in item]
    images = [os.path.join(path, item) for item in images]
    return np.array(images)


def conv_block(inputs, filters, kernel_size=(3, 3), activation='relu', padding='same'):
    conv1 = Conv2D(filters, kernel_size, activation=activation, padding=padding)(inputs)
    conv2 = Conv2D(filters, kernel_size, activation=activation, padding=padding)(conv1)
    return conv2


def unet(input_shape, filters):
    inputs = Input(shape=input_shape, name='noise_image')
    t = Input(shape=(1,), dtype=tf.int32, name="timestep")
    new_shape = np.array(input_shape)
    new_shape[-1] = 1
    shape = np.prod(new_shape)
    em = Embedding(diffusion_timestep, 64)(t)
    em = Dense(shape)(em)
    em = Reshape(new_shape)(em)

    merge = Concatenate()([inputs, em])

    conv_layers = []
    for i, f in enumerate(filters):
        if i == 0:
            conv = conv_block(merge, f)
        else:
            conv = conv_block(MaxPooling2D()(conv), f)
        bnorm = BatchNormalization()(conv)
        relu = ReLU()(bnorm)
        conv_layers.append(relu)

    for i, f in enumerate(filters[:-1][::-1]):
        up = Conv2DTranspose(f, (2, 2), strides=(2, 2), padding='same')(conv)
        conv = Concatenate()([up, conv_layers[-i-2]])
        conv = conv_block(conv, f)
        bnorm = BatchNormalization()(conv)
        relu = ReLU()(bnorm)
    relu = Dropout(0.1)(relu)
    outputs = Conv2D(input_shape[-1], (1, 1), activation='tanh')(relu)

    model = Model([inputs, t], outputs)
    return model


def build_denoise_model(input_shape):
    model = unet(input_shape, filters=[64, 128, 256, 512, 1024])
    return model


def build_diffusion_model(input_shape):
    model_input = Input(shape=input_shape)

    noise_image, noise, t = NoiseLayer()(model_input)
    # range of noise_image and noise are (-1, 1) after tanh

    generator = build_denoise_model(input_shape)

    if IsTraining:
        noise_predict = generator([noise_image, t])
        # range of noise_predict is (-1, 1) after tanh

        noise_output = Concatenate(axis=-1)([noise_predict, noise])
        model = Model(inputs=model_input, outputs=noise_output)
    else:
        model_output = [noise_image, noise, t]
        model = Model(inputs=model_input, outputs=model_output)

    opt = Adam(learning_rate=0.0002)
    model.compile(optimizer=opt, loss=denoise_loss)

    model.summary()
    generator.summary()

    return model, generator


@tf.function
def denoise_loss(y_true, y_pred):
    last_channel = tf.shape(y_pred)[-1]
    predict = y_pred[..., 0:int(last_channel/2)]
    noise = y_pred[..., int(last_channel / 2):last_channel]
    loss = tf.reduce_mean(tf.square(noise - predict))
    return loss


def get_random_vector(latent_dim):
    random_numbers = np.random.normal(loc=0.0, scale=1.0, size=latent_dim)
    return random_numbers


def predict_model_64(model, ge, input_shape=(64, 64, 3)):
    a_t = np.array(alpha)
    a_bar_t = np.array(alpha_bar)
    sigma_t = np.array(sigma)

    dataset = load_dataset("D:\\GAN_anime_profile\\data\\images")
    test_input = get_image_data(dataset, (64, 64), 64)

    test_output, noise, t = model.predict(test_input)
    predict_noise = ge.predict([test_output, t])
    predict_noise = replace_extreme_values(predict_noise)

    test_output = np.arctanh(test_output)
    noise = np.arctanh(noise)
    predict_noise = np.arctanh(predict_noise)

    test_output = test_output[0:32]
    noise = noise[0:32]
    predict_noise = predict_noise[0:32]

    a = np.array([a_t[index] for index in t])[0:32]
    b = np.array([a_bar_t[index] for index in t])[0:32]
    c = np.array([sigma_t[index] if index != 1 else 0 for index in t])[0:32]

    z = get_random_vector((32,) + input_shape)
    test_1 = (1 / np.sqrt(a))[:, np.newaxis, np.newaxis, np.newaxis]
    test_2 = ((1 - a) / np.sqrt(1 - b))[:, np.newaxis, np.newaxis, np.newaxis]
    test_2 = test_2 * noise
    test_2 = test_output - test_2
    test_3 = c[:, np.newaxis, np.newaxis, np.newaxis] * z
    x = test_1 * test_2 + test_3

    z_ = get_random_vector((32,) + input_shape)
    test_1_ = 1 / np.sqrt(a)
    test_1_ = test_1_[:, np.newaxis, np.newaxis, np.newaxis]
    test_2_ = (1 - a) / np.sqrt(1 - b)
    test_2_ = test_2_[:, np.newaxis, np.newaxis, np.newaxis]
    test_2_ = test_2_ * predict_noise
    test_2_ = test_output - test_2_
    test_3_ = c[:, np.newaxis, np.newaxis, np.newaxis] * z_
    x_ = test_1_ * test_2_ + test_3_

    image = np.concatenate([test_output[0:16], x[0:16], x_], axis=0)

    rgb_image = ((np.tanh(image) + 1) * 127.5).astype(int)
    show_images_debug(rgb_image, 0)


def replace_extreme_values(arr):
    min_val = (np.min(arr[arr != -1]) - 1) / 2
    max_val = (np.max(arr[arr != 1]) + 1) / 2

    arr[arr == -1] = min_val
    arr[arr == 1] = max_val

    return arr


def sampling_new_image(ge, input_shape=(64, 64, 3), image_count=1, model=None, current_index=0):
    # 1. get a noise image
    test_input = get_random_vector((image_count,) + input_shape)

    """
    dataset = load_dataset("D:\\GAN_anime_profile\\data\\images")
    data_size = len(dataset)
    test_input = get_image_data(dataset, (64, 64), 1)
    """

    fig, axes = plt.subplots(12, 12, figsize=(10, 10))
    rgb_image = ((np.tanh(test_input) + 1) * 127.5).astype(int)
    for i in range(image_count):
        axes[i, 0].imshow(rgb_image[i])
        axes[i, 0].axis('off')

    """
    test_input, _, _ = model.predict(test_input)
    test_input = replace_extreme_values(test_input)
    test_input = np.arctanh(test_input)
    """

    # 2. for loop denoise
    a_t = np.array(alpha)
    a_bar_t = np.array(alpha_bar)
    sigma_t = np.array(sigma)

    x = test_input
    index = 1
    step = int(diffusion_timestep / 10)
    for t in range(0, diffusion_timestep)[::-1]:
        if t % 50 == 0:
            print("Current time step:", t)
        if t == 0:
            z = 0
        else:
            z = get_random_vector((image_count,) + input_shape)
        t_test = np.tile(t, np.shape(x)[0])
        t_input = np.tanh(x)
        predict_noise = ge.predict([t_input, t_test])
        # remove -1 and 1 in predict_noise. change them to a smaller value
        predict_noise = replace_extreme_values(predict_noise)
        predict_noise = np.arctanh(predict_noise)

        if t % step == 0:
            rgb_image = ((np.tanh(x) + 1) * 127.5).astype(int)
            for i in range(image_count):
                axes[i, index].imshow(rgb_image[i])
                axes[i, index].axis('off')
            index += 1
            print("image: ", t)

        x = 1 / np.sqrt(a_t[t]) \
            * (x - (1 - a_t[t])/np.sqrt(1-a_bar_t[t]) * predict_noise) \
            + np.sqrt(sigma_t[t]) * z

    rgb_image = ((np.tanh(x) + 1) * 127.5).astype(int)
    for i in range(image_count):
        axes[i, 11].imshow(rgb_image[i])
        axes[i, 11].axis('off')
    plt.tight_layout()
    plt.savefig("../test/diffusion/output_" + str(current_index) + ".png")
    plt.close(fig)

    # image_array = map_to_minus_one_to_one(x)
    image_array = x
    return image_array


def prepare_data(image_path):
    dataset = load_dataset(image_path)
    print("Loading images...")
    data_size = len(dataset)
    images = get_image_data(dataset, (64, 64), data_size)
    print("Finish loading training data!")

    test_input = images
    return test_input


def show_training_loss_image(data1):
    plt.plot(data1.history['loss'], marker='', linestyle='-', color='r')

    plt.title('Line Chart Example')
    plt.xlabel('Epoch')
    plt.ylabel('Training loss')
    plt.show()


def map_to_minus_one_to_one(arr):
    min_value = np.min(arr)
    max_value = np.max(arr)

    mapped_arr = -1 + 2 * (arr - min_value) / (max_value - min_value)
    return mapped_arr


def self_train():
    input_shape = (64, 64, 3)

    # prepare train data
    image_path = "D:\\GAN_anime_profile\\data\\images"

    # build model
    model, generator = build_diffusion_model(input_shape=input_shape)

    if IsTraining:
        train_x = prepare_data(image_path)
        reduce_lr_callback = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=2e-6)
        # we don't need y_train, so we give a zero array to fit the blank. int8 requires smaller amount of memory
        history = model.fit(train_x, np.zeros(len(train_x), dtype=np.int8), epochs=100, batch_size=128, shuffle=True, callbacks=[reduce_lr_callback])
        show_training_loss_image(history)
        generator.save("./diffusion_model_1000_100.h5")

    else:
        # test model prediction
        generator.load_weights("./diffusion_model_1000_100.h5")

        # predict_model_64(model, generator)

        for i in range(1, 25, 1):
            sampling_new_image(generator, input_shape=input_shape, image_count=12, model=model, current_index=i)

        # test_output = sampling_new_image_gpu(generator, input_shape=input_shape, image_count=64)
        # test_output = test_output.numpy()
        #
        # convert image data from range -1 to 1 into 0 to 255

        # rgb_image = ((test_output + 1) * 127.5).astype(int)
        # show_images_debug(rgb_image, 0)


IsTraining = True
diffusion_timestep = 1000
alpha, alpha_bar, beta, sigma, sqrt_alpha, sqrt_one_alpha = set_beta_t(timestep=diffusion_timestep)

self_train()
