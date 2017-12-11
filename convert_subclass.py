import sys, os
import numpy as np

import keras
from keras.models import *
from keras.layers import *
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K

import coremltools
from coremltools.proto import NeuralNetwork_pb2



from keras.engine.topology import Layer

# This is the custom activation function implemented as a Layer subclass.
class Swish(Layer):
    def __init__(self, beta=1., **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.beta = beta

    def build(self, input_shape):
        super(Swish, self).build(input_shape)

    def call(self, x):
        return K.sigmoid(self.beta * x) * x

    def compute_output_shape(self, input_shape):
        return input_shape


# This version of Swish can automatically learn the best value of beta.
class LearnableSwish(Layer):
    def __init__(self, **kwargs):
        super(LearnableSwish, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(name="beta", 
                                    shape=(input_shape[3], ),
                                    initializer=keras.initializers.Constant(value=1),
                                    trainable=True)
        super(LearnableSwish, self).build(input_shape)

    def call(self, x):
        return K.sigmoid(self.beta * x) * x

    def compute_output_shape(self, input_shape):
        return input_shape


# Create a silly model that has our custom activation function as a new layer.
def create_model():
    inp = Input(shape=(256, 256, 3))
    x = Conv2D(6, (3, 3), padding="same")(inp)
    x = Swish(beta=0.01)(x)
    #x = LearnableSwish()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation="softmax")(x)
    return Model(inp, x)


# Build the model.
model = create_model()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()


# Here is where you would train the model... To keep things simple, we don't
# actually do any training but give the model a fixed set of random weights. 
# These weights do not mean anything; they're just here to test that Core ML
# gives the same output as the Keras model.
W = model.get_weights()
np.random.seed(12345)
for i in range(len(W)):
    if i != 2:  # skip the LearnableSwish layer
        W[i] = np.random.randn(*(W[i].shape)) * 2 - 1
model.set_weights(W)


# Test the model with an image. We'll do the same thing with Core ML in the
# iOS app and it should give the same output for the same input.
img = load_img("floortje.png", target_size=(256, 256))
img = np.expand_dims(img_to_array(img), 0)
pred = model.predict(img)

print("Predicted output:")
print(pred)


# The conversion function for the Swish layer.
def convert_swish(layer):
    params = NeuralNetwork_pb2.CustomLayerParams()

    # The name of the Swift or Obj-C class that implements this layer.
    params.className = "Swish"

    # The desciption is shown in Xcode's mlmodel viewer.
    params.description = "A fancy new activation function"

    # Set configuration parameters
    params.parameters["beta"].doubleValue = layer.beta
    return params


def convert_learnable_swish(layer):
    params = NeuralNetwork_pb2.CustomLayerParams()

    # The name of the Swift or Obj-C class that implements this layer.
    params.className = "Swish"

    # The desciption is shown in Xcode's mlmodel viewer.
    params.description = "A fancy new activation function"

    # Add the weights
    beta_weights = params.weights.add()
    beta_weights.floatValue.extend(layer.get_weights()[0].astype(float))
    return params


print("\nConverting the model:")

# Convert the model to Core ML.
coreml_model = coremltools.converters.keras.convert(
    model,
    input_names="image",
    image_input_names="image",
    output_names="output",
    add_custom_layers=True,
    custom_conversion_functions={ "Swish": convert_swish,
                                  "LearnableSwish" : convert_learnable_swish })


# This is the alternative method of filling in the CustomLayerParams:
# grab the layer and change its properties directly.
#layer = coreml_model._spec.neuralNetwork.layers[1]
#layer.custom.className = "Swish"


# Look at the layers in the converted Core ML model.
print("\nLayers in the converted model:")
for i, layer in enumerate(coreml_model._spec.neuralNetwork.layers):
    if layer.HasField("custom"):
        print("Layer %d = %s --> custom layer = %s" % (i, layer.name, layer.custom.className))
    else:
        print("Layer %d = %s" % (i, layer.name))


# Fill in the metadata and save the model.
coreml_model.author = "AuthorMcAuthorName"
coreml_model.license = "Public Domain"
coreml_model.short_description = "Playing with custom Core ML layers"
coreml_model.input_description["image"] = "Input image"
coreml_model.output_description["output"] = "The predictions"
coreml_model.save("NeuralMcNeuralNet.mlmodel")
