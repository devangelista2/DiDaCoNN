import DDIM

import tensorflow as tf

### Embedding Network
def get_embedding_model(model, depths, block_depth):
    model.trainable = False
    embedding_model = DDIM.get_Unet(64, depths, block_depth)
    weights_name = 'weights/networkUnet_full.hdf5'
    embedding_model.load_weights(weights_name)

def get_embedding_loss(model):
    def visible_loss(y_true, y_pred):
        #y_pred is the seed 
        generated = model.reverse_diffusion(y_pred, 10)
        generated = model.denormalize(generated)
        err = tf.reduce_mean(tf.math.abs(y_true - generated))
        return err

    return visible_loss