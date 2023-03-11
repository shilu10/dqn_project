from network import *
import tensorflow.keras as keras 

class ModelBuilder: 
  
  def build_model(self, input_dim, output_dim, lr):  
    network = DeepQNetwork(input_dim, output_dim)
    model = network.build_model()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))
    return model 