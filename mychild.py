from keras import models, layers, optimizers

class Child:
    # architecture from: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    def __init__(self, model, batch_size, epochs):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, gen, nbatches):
        self.model.fit_generator(
            gen, nbatches, self.epochs, verbose=0, use_multiprocessing=True)
        return self

    def evaluate(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1]

def create_simple_conv(input_shape):
    x = input_layer = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(10, activation='softmax')(x)
    model = models.Model(input_layer, x)
    optimizer = optimizers.SGD(decay=1e-4)
    model.compile(optimizer, 'categorical_crossentropy', ['accuracy'])
    return model
