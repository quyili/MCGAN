# _*_ coding:utf-8 _*_
import tensorflow as tf
from encoder import Encoder
from decoder import Decoder
import logging


class GAN:
    def __init__(self,
                 image_size,
                 learning_rate=2e-5,
                 batch_size=1,
                 ngf=64,
                 ):
        """
        Args:
          batch_size: integer, batch size
          learning_rate: float, initial learning rate for Adam
          ngf: number of gen filters in first conv layer
        """
        self.learning_rate = learning_rate
        self.input_shape = [int(batch_size / 4), image_size[0], image_size[1], image_size[2]]
        self.ones = tf.ones(self.input_shape, name="ones")
        # segment model result : label probability dict
        self.prob_list = {}
        # encoder result dict
        self.code_list = {}
        # Discriminator result dict
        self.judge_list = {}
        self.tensor_name = {}


        self.EC_L = Encoder('EC_L', ngf=ngf)
        self.DC_L = Decoder('DC_L', ngf=ngf, output_channl=5)

    """
    Get input image x segmentation result, and turn to ont hot vector
    then normalize 
    """
    def segmentation(self, x):
        l_prob = self.DC_L(self.EC_L(x))
        l_f = tf.reshape(tf.cast(tf.argmax(l_prob, axis=-1), dtype=tf.float32) * 0.25, shape=self.input_shape)
        return l_prob, l_f

    def model(self, l_x, l_y, l_z, l_w, x, y, z, w):

        image_list = {}
        self.tensor_name["l_x"] = str(l_x)
        self.tensor_name["l_y"] = str(l_y)
        self.tensor_name["l_z"] = str(l_z)
        self.tensor_name["l_w"] = str(l_w)
        self.tensor_name["x"] = str(x)
        self.tensor_name["y"] = str(y)
        self.tensor_name["z"] = str(z)
        self.tensor_name["w"] = str(w)

        label_expand_x = tf.reshape(tf.one_hot(tf.cast(l_x, dtype=tf.int32), axis=-1, depth=5),
                                    shape=[self.input_shape[0], self.input_shape[1],
                                           self.input_shape[2], 5])
        label_expand_y = tf.reshape(tf.one_hot(tf.cast(l_y, dtype=tf.int32), axis=-1, depth=5),
                                    shape=[self.input_shape[0], self.input_shape[1],
                                           self.input_shape[2], 5])
        label_expand_z = tf.reshape(tf.one_hot(tf.cast(l_z, dtype=tf.int32), axis=-1, depth=5),
                                    shape=[self.input_shape[0], self.input_shape[1],
                                           self.input_shape[2], 5])
        label_expand_w = tf.reshape(tf.one_hot(tf.cast(l_w, dtype=tf.int32), axis=-1, depth=5),
                                    shape=[self.input_shape[0], self.input_shape[1],
                                           self.input_shape[2], 5])
        l_x = l_x * 0.25
        l_y = l_y * 0.25
        l_z = l_z * 0.25
        l_w = l_w * 0.25

        l_f_prob_by_x, l_f_by_x = self.segmentation(x)
        l_f_prob_by_y, l_f_by_y = self.segmentation(y)
        l_f_prob_by_z, l_f_by_z = self.segmentation(z)
        l_f_prob_by_w, l_f_by_w = self.segmentation(w)

        self.tensor_name["l_f_by_x"] = str(l_f_by_x)
        self.tensor_name["l_f_by_y"] = str(l_f_by_y)
        self.tensor_name["l_f_by_z"] = str(l_f_by_z)
        self.tensor_name["l_f_by_w"] = str(l_f_by_w)

        G_loss = 0.0
        G_loss += self.mse_loss(label_expand_x[:, :, :, 0],
                                l_f_prob_by_x[:, :, :, 0]) \
                  + self.mse_loss(label_expand_x[:, :, :, 1],
                                  l_f_prob_by_x[:, :, :, 1]) * 15 \
                  + self.mse_loss(label_expand_x[:, :, :, 2],
                                  l_f_prob_by_x[:, :, :, 2]) * 85 \
                  + self.mse_loss(label_expand_x[:, :, :, 3],
                                  l_f_prob_by_x[:, :, :, 3]) * 85 \
                  + self.mse_loss(label_expand_x[:, :, :, 4],
                                  l_f_prob_by_x[:, :, :, 4]) * 85
        G_loss += self.mse_loss(l_x, l_f_by_x) * 25

        G_loss += self.mse_loss(label_expand_y[:, :, :, 0],
                                l_f_prob_by_y[:, :, :, 0]) \
                  + self.mse_loss(label_expand_y[:, :, :, 1],
                                  l_f_prob_by_y[:, :, :, 1]) * 15 \
                  + self.mse_loss(label_expand_y[:, :, :, 2],
                                  l_f_prob_by_y[:, :, :, 2]) * 85 \
                  + self.mse_loss(label_expand_y[:, :, :, 3],
                                  l_f_prob_by_y[:, :, :, 3]) * 85 \
                  + self.mse_loss(label_expand_y[:, :, :, 4],
                                  l_f_prob_by_y[:, :, :, 4]) * 85
        G_loss += self.mse_loss(l_y, l_f_by_y) * 25

        G_loss += self.mse_loss(label_expand_z[:, :, :, 0],
                                l_f_prob_by_z[:, :, :, 0]) \
                  + self.mse_loss(label_expand_z[:, :, :, 1],
                                  l_f_prob_by_z[:, :, :, 1]) * 15 \
                  + self.mse_loss(label_expand_z[:, :, :, 2],
                                  l_f_prob_by_z[:, :, :, 2]) * 85 \
                  + self.mse_loss(label_expand_z[:, :, :, 3],
                                  l_f_prob_by_z[:, :, :, 3]) * 85 \
                  + self.mse_loss(label_expand_z[:, :, :, 4],
                                  l_f_prob_by_z[:, :, :, 4]) * 85
        G_loss += self.mse_loss(l_z, l_f_by_z) * 25

        G_loss += self.mse_loss(label_expand_w[:, :, :, 0],
                                l_f_prob_by_w[:, :, :, 0]) \
                  + self.mse_loss(label_expand_w[:, :, :, 1],
                                  l_f_prob_by_w[:, :, :, 1]) * 15 \
                  + self.mse_loss(label_expand_w[:, :, :, 2],
                                  l_f_prob_by_w[:, :, :, 2]) * 85 \
                  + self.mse_loss(label_expand_w[:, :, :, 3],
                                  l_f_prob_by_w[:, :, :, 3]) * 85 \
                  + self.mse_loss(label_expand_w[:, :, :, 4],
                                  l_f_prob_by_w[:, :, :, 4]) * 85
        G_loss += self.mse_loss(l_w, l_f_by_w) * 25

        image_list["l_x"] = l_x
        image_list["l_y"] = l_y
        image_list["l_z"] = l_z
        image_list["l_w"] = l_w
        image_list["x"] = x
        image_list["y"] = y
        image_list["z"] = z
        image_list["w"] = w

        self.prob_list["label_expand_x"] = label_expand_x
        self.prob_list["label_expand_y"] = label_expand_y
        self.prob_list["label_expand_z"] = label_expand_z
        self.prob_list["label_expand_w"] = label_expand_w

        self.prob_list["l_f_prob_by_x"] = l_f_prob_by_x
        self.prob_list["l_f_prob_by_y"] = l_f_prob_by_y
        self.prob_list["l_f_prob_by_z"] = l_f_prob_by_z
        self.prob_list["l_f_prob_by_w"] = l_f_prob_by_w

        image_list["l_f_by_x"] = l_f_by_x
        image_list["l_f_by_y"] = l_f_by_y
        image_list["l_f_by_z"] = l_f_by_z
        image_list["l_f_by_w"] = l_f_by_w

        return G_loss, image_list

    def get_variables(self):
        return [self.EC_L.variables
                + self.DC_L.variables
                ]

    def optimize(self):
        def make_optimizer(name='Adam'):
            learning_step = (
                tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, name=name)
            )
            return learning_step

        G_optimizer = make_optimizer(name='Adam_G')

        return G_optimizer

    def loss_summary(self, G_loss):
        tf.summary.scalar('loss/G_loss', G_loss)

    def image_summary(self, image_dirct):
        for key in image_dirct:
            tf.summary.image('image/' + key, image_dirct[key])

    def mse_loss(self, x, y):
        """ supervised loss (L2 norm)
        """
        loss = tf.reduce_mean(tf.square(x - y))
        return loss

    def dice_score(self, output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
        inse = tf.reduce_sum(output * target, axis=axis)
        if loss_type == 'jaccard':
            l = tf.reduce_sum(output * output, axis=axis)
            r = tf.reduce_sum(target * target, axis=axis)
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(output, axis=axis)
            r = tf.reduce_sum(target, axis=axis)
        else:
            raise Exception("Unknow loss_type")
        dice = (2. * inse + smooth) / (l + r + smooth)
        dice = tf.reduce_mean(dice)
        return dice

    def MSE(self, output, target):
        mse = tf.reduce_mean(tf.square(output - target))
        return mse

    def norm(self, input):
        output = (input - tf.reduce_min(input, axis=[1, 2, 3])
                  ) / (tf.reduce_max(input, axis=[1, 2, 3]) - tf.reduce_min(input, axis=[1, 2, 3]))
        return output
