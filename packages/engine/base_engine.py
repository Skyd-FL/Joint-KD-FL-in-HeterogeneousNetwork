from utils.header import *

class BaseNetwork(Model):
    def __init__(self):
        super(BaseNetwork,self).__init__()
        self.predictor = tf.keras.models.Sequential([
            tf.keras.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(128, kernel_size=(5, 5), activation="relu", padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),        
            tf.keras.layers.Conv2D(64, kernel_size=(5, 5), activation="relu", padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),        

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(47, name = "logits"),
            tf.keras.layers.Activation('softmax')
        ])
    # def call(self, x):
    #     predictor = self.predictor(x)
    #     return predictor