#%%
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import RNN_ver4 

def NetD(input_shape=360):
    inputs = keras.Input(shape=(input_shape))
    inputs_ = tf.reshape(inputs,(-1,input_shape,1))
    l1 = layers.Conv1D(128,3,strides=3)(inputs_)
    l1b = layers.BatchNormalization()(l1)
    l1a = layers.LeakyReLU()(l1b)
    l2 = layers.Conv1D(64,3,strides=3)(l1a)
    l2b= layers.BatchNormalization()(l2)
    l2a = layers.LeakyReLU()(l2b)
    l3 = layers.Conv1D(64,3,strides=3)(l2a)
    l3a = layers.LeakyReLU()(l3)
    l3f = layers.Flatten()(l3a)
    features = layers.Dense(64,activation='selu')(l3f)
    l4d = layers.Dropout(0.2)(features)
    out = layers.Dense(1,activation='sigmoid')(l4d)
    discriminator = keras.Model(inputs,[features,out])
    return discriminator
    
class SurroGAN(keras.Model):
    def __init__(self, n_cell=256, n_layers=2, FN=[128,64], dr_rates=0.2, PE=None, PE_d=4, loss_weight = [1,10],**kwargs):
        super(SurroGAN, self).__init__(**kwargs)
        # 초기 파라미터 설정
        # net_G PARAMS
        self.n_cell=n_cell
        self.n_layers=n_layers
        self.FN=[128,64]
        self.dr_rates=0.2
        self.PE=PE
        self.PE_d=4
        self.loss_weight=loss_weight
        self.net_G = RNN_ver4.RNN_model_v4(n_cell=self.n_cell,n_layers=self.n_layers, 
                        FN=self.FN, dr_rates=self.dr_rates, PE=self.PE,PE_d=self.PE_d)        
        self.net_D = NetD()
        
    def compile(self, d_optimizer, g_optimizer):
        super(SurroGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    @tf.function
    def test_step(self, data):
        x,y = data
        gen_time = self.net_G(x,training=False)
        self.compiled_metrics.update_state(y_true=y,y_pred=gen_time)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as net_G_tape:
            # Train Generator
            # data = x
            x,y = data
            gen_time = self.net_G(x, training=True)

            real_feat, real_out = self.net_D(y,training=False)
            fake_feat, fake_out = self.net_D(gen_time,training=False)

            L_adv = tf.reduce_mean(tf.keras.losses.MSE(y_true = real_feat, y_pred = fake_feat))
            L_con = tf.reduce_mean(tf.keras.losses.MSE(y_true = y,y_pred = gen_time))

            loss_gen = self.loss_weight[0]*L_adv + self.loss_weight[1]*L_con

        gvar_list = self.net_G.trainable_variables
        grads_of_gen = net_G_tape.gradient(loss_gen, gvar_list)
        # gradient clipping for version2
        grads_of_gen, _ = tf.clip_by_global_norm(grads_of_gen, 5.0)
        self.g_optimizer.apply_gradients(zip(grads_of_gen, gvar_list))

        with tf.GradientTape() as net_D_tape:
            # Train Discriminator   
            _, real_o = self.net_D(y, training = True) # 1
            _, fake_o = self.net_D(gen_time, training = True) # 0

            loss_disc = self.discriminator_loss(real_o,fake_o)

        dvar_list = self.net_D.trainable_variables
        grads_of_disc = net_D_tape.gradient(loss_disc, dvar_list)
        # gradient clipping for version2
        grads_of_disc, _ = tf.clip_by_global_norm(grads_of_disc, 5.0)

        self.d_optimizer.apply_gradients(zip(grads_of_disc, dvar_list))

        self.compiled_metrics.update_state(y_true = y, y_pred=gen_time)

        return {
            "total loss": loss_gen,
            "Adv loss" : L_adv,
            "Recon loss" : L_con,
            "Disc loss" : loss_disc}        

    def call(self,inputs):
        gen_time = self.net_G(inputs)
        return gen_time

    @staticmethod
    def discriminator_loss(real, fake):
        real_label = tf.ones_like(real)
        fake_label = tf.zeros_like(fake)
        cross_entropy = tf.keras.losses.BinaryCrossentropy()
        real_loss = cross_entropy(real_label, real)
        fake_loss = cross_entropy(fake_label, fake)
        bce_loss = (real_loss + fake_loss)/2
        return bce_loss      
