#Model used in paper

  def build_model(self): 
    input = Input((self.input_dims))
    x = Conv2D(32, (3, 3), activation='relu')(input)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), activation='relu')(input)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.2)(x) 
    x = Flatten()(x)

    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation="relu")(x)

    output = Dense(self.output_dim)(x)

    model = keras.Model(inputs=[input], outputs=[output])

    return model