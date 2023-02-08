import tensorflow as tf
from tensorflow import keras

### Gradient descent
### The trainable part of the model is the initial noise, added as weight
### as part of the build method
### The initial noise can be randomly initialized (default),
### or it can be passed as input

class DescentModel(keras.Model):
    def __init__(self, reference_model, n, starting_noise = None):
        super().__init__()
        self.reference_model = reference_model
        self.n = n
        self.starting_noise = starting_noise
        self.reference_model.trainable = False #the model must not be trainable
    
    def build(self, input_shape):
        if self.starting_noise is None:
            #Warning the default keras stddev is 0.05 
            initializer = tf.keras.initializers.RandomNormal(mean=0.,stddev=1.)
        else:
            initializer = tf.keras.initializers.Constant(self.starting_noise)
        self._initial_noise = self.add_weight(name='initial_noise', shape=(self.n,)+input_shape[1:], initializer=initializer, trainable=True)
  
    def call(self, inputs):
        #we add a small quadratic penalty
        self.add_loss(.005 * tf.reduce_mean(self._initial_noise**2))
        generated_images = self.reference_model.reverse_diffusion(self._initial_noise, 10)
        generated_images = self.reference_model.normalizer.mean + generated_images * self.reference_model.normalizer.variance**0.5
        return generated_images
    
    def compute_output_shape(self, input_shape):
        return (self.n,) + input_shape[1:]
 
def loss_function(ground_truth, predicted):
    return tf.reduce_mean(tf.math.abs(ground_truth - predicted))

def diffusion_descent(model, ground_truth, n, starting_noise=None, epochs=1500):
    ground_truth = tf.expand_dims(ground_truth, 0)
    descent_model = DescentModel(model, n, starting_noise=starting_noise)
    descent_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-2),
        loss=loss_function
    )
    descent_model.fit(
        x=ground_truth,
        y=ground_truth,
        epochs=epochs
    )

    output = descent_model.predict(
        ground_truth
    )

    return output, descent_model._initial_noise