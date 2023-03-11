import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPool2D, Input


#https://github.com/marload/DeepRL-TensorFlow2
class DeepQNetwork_3D:
  def __init__(self, input_dim, output_dim): 
    self.input_dims = input_dim
    self.output_dim = output_dim 

  def build_model(self): 
    input = Input((self.input_dims))
    x = Conv2D(32, 8, strides=(4, 4), activation='relu', data_format="channels_first")(input)
    x = Conv2D(64, 4, strides=(2, 2), activation='relu', data_format="channels_first")(x)
    x = Conv2D(128, 3, strides=(1, 1), activation='relu', data_format="channels_first")(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation="relu")(x)
    output = Dense(self.output_dim)(x)

    model = keras.Model(inputs=[input], outputs=[output])

    return model

  
class DeepQNetwork_2D:
  def __init__(self, input_dim, output_dim): 
    self.input_dims = input_dim
    self.output_dim = output_dim 

  def build_model(self): 
    input = Input((self.input_dims))
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation="relu")(x)
    output = Dense(self.output_dim)(x)

    model = keras.Model(inputs=[input], outputs=[output])

    return model
