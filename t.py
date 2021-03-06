import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


a = tf.constant([[1, 2], [3, 4]])

TRUE_W = 3.0
TRUE_B = 2.0
NUM_EXAMPLES = 201
x = tf.linspace(-2, 2, NUM_EXAMPLES)
x = tf.cast(x, tf.float32)


def f(x):
    return x*TRUE_W + TRUE_B


noise = tf.random.normal(shape=[201])
y = f(x) + noise


plt.plot(x, y, '.')


class mymodel(tf.Module):
    def __init__(self, **kwags):
        super().__init__(**kwags)
        self.w = tf.Variable(5.0, name='w')
        self.b = tf.Variable(0.0, name='b')

    def __call__(self, x):
        return self.w*x + self.b


model = mymodel()
model(x)
model.variables


def myloss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred-y_true))


def train(model, x, y, learning_rate):
    with tf.GradientTape as tape:
        current_loss = myloss(y, model(x))

    dw, db = tape.gradient(current_loss, [model.w, model.b])
    model.w.assign_sub(learning_rate*dw)
    model.b.assign_sub(learning_rate*db)


model = tf.keras.Sequential(
    [tf.keras.Input(shape=(3, )),
     tf.keras.layers.Dense(2, activation='relu', name='layer1'),
     tf.keras.layers.Dense(2, activation='relu', name='layer2'),
     tf.keras.layers.Dense(4, name='layer3')])

x = tf.ones((6, 3))
y = model(x)


model.layers
model.weights
model.summary()


initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu", name="my_intermediate_layer"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)

feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=initial_model.get_layer(name="my_intermediate_layer").output,
)


type(initial_model.get_layer(name="my_intermediate_layer").output)
x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)


inputs = keras.Input(shape=(784,))
x = layers.Dense(64, activation='relu')(input)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs=input, outputs=outputs, name='mnist_model')
model.summary()


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255


model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=64, epochs=2,
                    validation_split=0.2)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

features_list = [layer.output for layer in model.layers]
model.summary()


class CustomDense(tf.keras.layers.Layer):
    def __init__(self,units=32):
        super(CustomDense,self).__init__()
        self.units = units
        
    def build(self,input_shape):
        self.w = self.add_weight(shape=(input_shape[-1],self.units),
                                 initializer='random_normal',trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", 
                                 trainable=True)
        
    def call(self,inputs):
        return tf.matmul(inputs,self.w) + self.b



inputs = tf.keras.Input(shape=[784],name='digits')
x = tf.keras.layers.Dense(64,activation='relu',name='dense_1')(inputs)
x = tf.keras.layers.Dense(64,activation='relu',name='dense_2')(x)
outputs = tf.keras.layers.Dense(10,activation='softmax',name='predicitons')(x)
model = tf.keras.Model(inputs=inputs,outputs=outputs)


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000,784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

model.compile(optimizer=keras.optimizers.RMSprop(),
              loss = keras.losses.SparseCategoricalCrossentropy(),
              metrics = keras.metrics.SparseCategoricalAccuracy())

history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=3,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val))
results = model.evaluate(x_test, y_test, batch_size=128)
model.predict(x_test[:3])



a = tf.constant([1,2])
b = tf.constant([3,4])

tf.cast(a,'float32')






















