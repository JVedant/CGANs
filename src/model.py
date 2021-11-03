import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, BatchNormalization, Dense, Reshape, Flatten
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, Activation, MaxPool2D, LeakyReLU
from tensorflow.keras.models import Model
import config
import matplotlib.pyplot as plt

def UpConv_block(tensor, filters, strides=(2, 2)):
    x = Conv2DTranspose(
            filters=filters,
            kernel_size=(5, 5),
            strides=strides,
            padding="same",
            kernel_initializer="he_normal",
            use_bias=False
        )(tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.25)(x)
    return x


def Generator():
    image_height, image_width, image_channel = [config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS]

    inputs = Input(shape=(config.N_DIMS, ))
    x = Dense(
            units=16*16*image_height,
            use_bias=False
        )(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(
        alpha=0.25
    )(x)

    x = Reshape(
        (16, 16, image_height)
    )(x)
    
    up_conv_1 = UpConv_block(tensor=x, filters=256, strides=(1, 1))
    up_conv_2 = UpConv_block(tensor=up_conv_1, filters=128)
    #up_conv_3 = UpConv_block(tensor=up_conv_2, filters=128)
    up_conv_4 = UpConv_block(tensor=up_conv_2, filters=64)
    up_conv_5 = UpConv_block(tensor=up_conv_4, filters=32)
    
    up_conv_out = Conv2DTranspose(
        filters=image_channel, 
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=False,
        activation="tanh"
    )(up_conv_5)
    
    model = Model(inputs=[inputs], outputs=[up_conv_out])

    return model


'''class Generator():
    def __init__(self, n_dim, image_height=config.IMAGE_HEIGHT, image_width=config.IMAGE_WIDTH, image_channel=config.IMAGE_CHANNELS) -> None:
        self.n_dim = n_dim
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = image_channel
        self.kernel_size = (3, 3)
        
    def forward(self) -> Sequential:

        inputs = Input(shape=(self.n_dim, ))
        x = Dense(
                    units=8*8*self.image_height,
                    use_bias=False
                )(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU(
            alpha=0.25
        )(x)
        x = Reshape(
            (8, 8, self.image_height)
        )(x)

        up_conv_1 = UpConv_block(tensor=x, filters=512, strides=(1, 1))
        up_conv_2 = UpConv_block(tensor=up_conv_1, filters=256)
        up_conv_3 = UpConv_block(tensor=up_conv_2, filters=128)
        up_conv_4 = UpConv_block(tensor=up_conv_3, filters=64)
        up_conv_5 = UpConv_block(tensor=up_conv_4, filters=32)

        up_conv_out = Conv2DTranspose(
            filters=3, 
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            activation="tanh"
        )(up_conv_5)

        model = Model(inputs=[inputs], outputs=[up_conv_out])

        return model
'''

def conv_lr_d(tensor, filters):
    x = Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        padding="same"
    )(tensor)
    x = LeakyReLU(
        alpha=0.25
    )(x)
    x = Dropout(
        rate=0.25
    )(x)
    return x


def Discriminator():
    inputs = Input(shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS))
    x = conv_lr_d(tensor=inputs, filters = 32)
    x = conv_lr_d(tensor=x, filters = 64)
    x = conv_lr_d(tensor=x, filters = 128)
    x = conv_lr_d(tensor=x, filters = 256)

    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[inputs], outputs=[output])
    return model


#if __name__ == "__main__":
#
#    generator = Generator()
#    noise = tf.random.normal([1, config.N_DIMS])
#    generated_image = generator(noise, training=False)
#
#    plt.imshow(generated_image[0, :, :, 0], cmap="gray")
#
#    plt.show()