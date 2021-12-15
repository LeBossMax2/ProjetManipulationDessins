from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Conv2DTranspose, Reshape


compressed_size = 94

def get_model():
    optimizer = optimizers.Adam(learning_rate=0.0005)
    loss = "binary_crossentropy"
    metrics = ["mae", "accuracy"]

    e_input_layer = keras.Input(shape=(28, 28))

    layer = Conv2D(32, 3, activation="relu", strides=2, padding="same")(Reshape((28, 28, 1))(e_input_layer))
    conv_output = Conv2D(64, 3, activation="relu", strides=2, padding="same")(layer)
    layer = Flatten()(conv_output)
    z = Dense(compressed_size, activation="relu")(layer)

    encoder = keras.Model(e_input_layer, z, name="encoder")

    encoder.summary()
    encoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    d_input_layer = keras.Input(shape=(compressed_size,))
    layer = layers.Dense(conv_output.shape[1] * conv_output.shape[2] * conv_output.shape[3], activation="relu")(d_input_layer)
    layer = layers.Reshape(conv_output.shape[1:4])(layer)
    layer = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(layer)
    layer = Conv2DTranspose(1, 3, activation="sigmoid", strides=2, padding="same")(layer)
    layer = Reshape((28, 28))(layer)

    decoder = keras.Model(d_input_layer, layer, name="decoder")

    decoder.summary()
    decoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    autoencoder = keras.Model(e_input_layer, decoder(encoder(e_input_layer)), name="auto-encoder")

    autoencoder.summary()
    autoencoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return autoencoder, encoder, decoder
