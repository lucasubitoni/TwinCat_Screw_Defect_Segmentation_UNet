import tensorflow as tf

def unet_model_OLD(input_size, num_filters=[4, 8, 16, 32]):
    
    # INPUT
    inputs = tf.keras.Input(input_size)
    x = inputs
    skips = list()

    #  ENCODER
    for filters in num_filters[:-1]:  # Skip last element (bottleneck)
        x = tf.keras.layers.Conv2D(filters, (7, 7), activation="relu", padding="same")(x)
        #x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters, (5, 5), activation="relu", padding="same")(x)
        skips.append(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)  

    #  BOTTLENECK
    x = tf.keras.layers.Conv2D(num_filters[-1], (5, 5), activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(num_filters[-1], (5, 5), activation="relu", padding="same")(x)

    #  DECODER
    for filters, skip in zip(reversed(num_filters[:-1]), reversed(skips)):
        x = tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(x)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = tf.keras.layers.Conv2D(filters, (5, 5), activation="relu", padding="same")(x)
        #x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters, (7, 7), activation="relu", padding="same")(x)
        

    #  OUTPUT LAYER
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)



def unet_model(input_size, num_filters=[4, 8, 16, 32]):

    def conv_block(x, filters, kernel_size=(5, 5)):
        x = tf.keras.layers.Conv2D(filters, kernel_size, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        return x

    # Input
    inputs = tf.keras.Input(input_size)
    x = inputs
    skips = []

    # Encoder
    for filters in num_filters[:-1]:
        x = conv_block(x, filters)
        x = conv_block(x, filters)
        skips.append(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # Bottleneck
    x = conv_block(x, num_filters[-1])
    x = conv_block(x, num_filters[-1])

    # Decoder
    for filters, skip in zip(reversed(num_filters[:-1]), reversed(skips)):
        x = tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(x)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = conv_block(x, filters)
        x = conv_block(x, filters)

    # Output
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)