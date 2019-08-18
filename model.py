# _*_ coding:utf-8 _*_
import tensorflow as tf
from discriminator import Discriminator
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
          input_size：list [H, W, C]
          batch_size: integer, batch size
          learning_rate: float, initial learning rate for Adam
          ngf: number of gen filters in first conv layer
        """
        self.learning_rate = learning_rate
        self.input_shape = [int(batch_size / 4), image_size[0], image_size[1], image_size[2]]
        self.ones = tf.ones(self.input_shape, name="ones")
        self.prob_list = {}
        self.code_list = {}
        self.judge_list = {}
        self.tensor_name = {}

        self.EC_L = Encoder('EC_L', ngf=ngf)
        self.DC_L = Decoder('DC_L', ngf=ngf, output_channl=5)

        self.EC_M = Encoder('EC_M', ngf=ngf)

        self.DC_X = Decoder('DC_X', ngf=ngf)
        self.DC_Y = Decoder('DC_Y', ngf=ngf)
        self.DC_Z = Decoder('DC_Z', ngf=ngf)
        self.DC_W = Decoder('DC_W', ngf=ngf)

        self.D_M = Discriminator('D_M', ngf=ngf)

    def get_mask(self, m, p=0):
        mask = 1.0 - self.ones * tf.cast(m > 0.0, dtype="float32")
        shape = m.get_shape().as_list()
        mask = tf.image.resize_images(mask, size=[shape[1] + p, shape[2] + p], method=1)
        mask = tf.image.resize_image_with_crop_or_pad(mask, shape[1], shape[2])
        return mask

    def segmentation(self, x):
        l_prob = self.DC_L(self.EC_L(x))
        l_f = tf.reshape(tf.cast(tf.argmax(l_prob, axis=-1), dtype=tf.float32) * 0.25, shape=self.input_shape)
        return l_prob, l_f

    def model(self,
              l_x, l_y, l_z, l_w,
              x, y, z, w):

        image_list = {}

        self.tensor_name["l_x"] = str(l_x)
        self.tensor_name["l_y"] = str(l_y)
        self.tensor_name["l_z"] = str(l_z)
        self.tensor_name["l_w"] = str(l_w)
        self.tensor_name["x"] = str(x)
        self.tensor_name["y"] = str(y)
        self.tensor_name["z"] = str(z)
        self.tensor_name["w"] = str(w)
        cx = 0.0
        cy = 1.0
        cz = 2.0
        cw = 3.0

        l_x = l_x * 0.25
        l_y = l_y * 0.25
        l_z = l_z * 0.25
        l_w = l_w * 0.25

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

        l_f_prob_by_x, l_f_by_x = self.segmentation(x)
        l_f_prob_by_y, l_f_by_y = self.segmentation(y)
        l_f_prob_by_z, l_f_by_z = self.segmentation(z)
        l_f_prob_by_w, l_f_by_w = self.segmentation(w)

        mask_x = self.get_mask(x)
        mask_y = self.get_mask(y)
        mask_z = self.get_mask(z)
        mask_w = self.get_mask(w)

        code_x = self.EC_M(x)
        code_y = self.EC_M(y)
        code_z = self.EC_M(z)
        code_w = self.EC_M(w)

        x_r = self.DC_X(code_x)
        y_r = self.DC_Y(code_y)
        z_r = self.DC_Z(code_z)
        w_r = self.DC_W(code_w)

        l_prob_x_r, l_x_r = self.segmentation(x_r)
        l_prob_y_r, l_y_r = self.segmentation(y_r)
        l_prob_z_r, l_z_r = self.segmentation(z_r)
        l_prob_w_r, l_w_r = self.segmentation(w_r)

        y_t_by_x = self.DC_Y(code_x)
        code_y_t_by_x = self.EC_M(y_t_by_x)
        x_r_c_by_y = self.DC_X(code_y_t_by_x)
        z_t_by_x = self.DC_Z(code_x)
        code_z_t_by_x = self.EC_M(z_t_by_x)
        x_r_c_by_z = self.DC_X(code_z_t_by_x)
        w_t_by_x = self.DC_W(code_x)
        code_w_t_by_x = self.EC_M(w_t_by_x)
        x_r_c_by_w = self.DC_X(code_w_t_by_x)

        l_prob_x_r_c_by_y, l_x_r_c_by_y = self.segmentation(x_r_c_by_y)
        l_prob_x_r_c_by_z, l_x_r_c_by_z = self.segmentation(x_r_c_by_z)
        l_prob_x_r_c_by_w, l_x_r_c_by_w = self.segmentation(x_r_c_by_w)

        x_t_by_y = self.DC_X(code_y)
        code_x_t_by_y = self.EC_M(x_t_by_y)
        y_r_c_by_x = self.DC_Y(code_x_t_by_y)
        z_t_by_y = self.DC_Z(code_y)
        code_z_t_by_y = self.EC_M(z_t_by_y)
        y_r_c_by_z = self.DC_Y(code_z_t_by_y)
        w_t_by_y = self.DC_W(code_y)
        code_w_t_by_y = self.EC_M(w_t_by_y)
        y_r_c_by_w = self.DC_Y(code_w_t_by_y)

        l_prob_y_r_c_by_x, l_y_r_c_by_x = self.segmentation(y_r_c_by_x)
        l_prob_y_r_c_by_z, l_y_r_c_by_z = self.segmentation(y_r_c_by_z)
        l_prob_y_r_c_by_w, l_y_r_c_by_w = self.segmentation(y_r_c_by_w)

        x_t_by_z = self.DC_X(code_z)
        code_x_t_by_z = self.EC_M(x_t_by_z)
        z_r_c_by_x = self.DC_Z(code_x_t_by_z)
        y_t_by_z = self.DC_Y(code_z)
        code_y_t_by_z = self.EC_M(y_t_by_z)
        z_r_c_by_y = self.DC_Z(code_y_t_by_z)
        w_t_by_z = self.DC_W(code_z)
        code_w_t_by_z = self.EC_M(w_t_by_z)
        z_r_c_by_w = self.DC_Z(code_w_t_by_z)

        l_prob_z_r_c_by_x, l_z_r_c_by_x = self.segmentation(z_r_c_by_x)
        l_prob_z_r_c_by_y, l_z_r_c_by_y = self.segmentation(z_r_c_by_y)
        l_prob_z_r_c_by_w, l_z_r_c_by_w = self.segmentation(z_r_c_by_w)

        x_t_by_w = self.DC_X(code_w)
        code_x_t_by_w = self.EC_M(x_t_by_w)
        w_r_c_by_x = self.DC_W(code_x_t_by_w)
        y_t_by_w = self.DC_Y(code_w)
        code_y_t_by_w = self.EC_M(y_t_by_w)
        w_r_c_by_y = self.DC_W(code_y_t_by_w)
        z_t_by_w = self.DC_Z(code_w)
        code_z_t_by_w = self.EC_M(z_t_by_w)
        w_r_c_by_z = self.DC_W(code_z_t_by_w)

        l_prob_w_r_c_by_x, l_w_r_c_by_x = self.segmentation(w_r_c_by_x)
        l_prob_w_r_c_by_y, l_w_r_c_by_y = self.segmentation(w_r_c_by_y)
        l_prob_w_r_c_by_z, l_w_r_c_by_z = self.segmentation(w_r_c_by_z)

        l_prob_x_t_by_y, l_x_t_by_y = self.segmentation(x_t_by_y)
        l_prob_x_t_by_z, l_x_t_by_z = self.segmentation(x_t_by_z)
        l_prob_x_t_by_w, l_x_t_by_w = self.segmentation(x_t_by_w)

        l_prob_y_t_by_x, l_y_t_by_x = self.segmentation(y_t_by_x)
        l_prob_y_t_by_z, l_y_t_by_z = self.segmentation(y_t_by_z)
        l_prob_y_t_by_w, l_y_t_by_w = self.segmentation(y_t_by_w)

        l_prob_z_t_by_x, l_z_t_by_x = self.segmentation(z_t_by_x)
        l_prob_z_t_by_y, l_z_t_by_y = self.segmentation(z_t_by_y)
        l_prob_z_t_by_w, l_z_t_by_w = self.segmentation(z_t_by_w)

        l_prob_w_t_by_x, l_w_t_by_x = self.segmentation(w_t_by_x)
        l_prob_w_t_by_y, l_w_t_by_y = self.segmentation(w_t_by_y)
        l_prob_w_t_by_z, l_w_t_by_z = self.segmentation(w_t_by_z)

        j_x, j_x_c = self.D_M(x)
        j_y, j_y_c = self.D_M(y)
        j_z, j_z_c = self.D_M(z)
        j_w, j_w_c = self.D_M(w)

        j_x_t_by_y, j_x_t_c_by_y = self.D_M(x_t_by_y)
        j_x_t_by_z, j_x_t_c_by_z = self.D_M(x_t_by_z)
        j_x_t_by_w, j_x_t_c_by_w = self.D_M(x_t_by_w)

        j_y_t_by_x, j_y_t_c_by_x = self.D_M(y_t_by_x)
        j_y_t_by_z, j_y_t_c_by_z = self.D_M(y_t_by_z)
        j_y_t_by_w, j_y_t_c_by_w = self.D_M(y_t_by_w)

        j_z_t_by_x, j_z_t_c_by_x = self.D_M(z_t_by_x)
        j_z_t_by_y, j_z_t_c_by_y = self.D_M(z_t_by_y)
        j_z_t_by_w, j_z_t_c_by_w = self.D_M(z_t_by_w)

        j_w_t_by_x, j_w_t_c_by_x = self.D_M(w_t_by_x)
        j_w_t_by_y, j_w_t_c_by_y = self.D_M(w_t_by_y)
        j_w_t_by_z, j_w_t_c_by_z = self.D_M(w_t_by_z)

        D_loss = 0.0
        G_loss = 0.0
        S_loss = 0.0

        # label supervise loss
        S_loss += self.mse_loss(label_expand_x[:, :, :, 0],
                                l_f_prob_by_x[:, :, :, 0]) * 0.5 * 10 \
                  + self.mse_loss(label_expand_x[:, :, :, 1],
                                  l_f_prob_by_x[:, :, :, 1]) * 5 * 10 \
                  + self.mse_loss(label_expand_x[:, :, :, 2],
                                  l_f_prob_by_x[:, :, :, 2]) * 25 * 10 \
                  + self.mse_loss(label_expand_x[:, :, :, 3],
                                  l_f_prob_by_x[:, :, :, 3]) * 25 * 10 \
                  + self.mse_loss(label_expand_x[:, :, :, 4],
                                  l_f_prob_by_x[:, :, :, 4]) * 25 * 10
        S_loss += self.mse_loss(l_x, l_f_by_x) * 25 * 10

        S_loss += self.mse_loss(label_expand_y[:, :, :, 0],
                                l_f_prob_by_y[:, :, :, 0]) * 0.5 * 10 \
                  + self.mse_loss(label_expand_y[:, :, :, 1],
                                  l_f_prob_by_y[:, :, :, 1]) * 5 * 10 \
                  + self.mse_loss(label_expand_y[:, :, :, 2],
                                  l_f_prob_by_y[:, :, :, 2]) * 25 * 10 \
                  + self.mse_loss(label_expand_y[:, :, :, 3],
                                  l_f_prob_by_y[:, :, :, 3]) * 25 * 10 \
                  + self.mse_loss(label_expand_y[:, :, :, 4],
                                  l_f_prob_by_y[:, :, :, 4]) * 25 * 10
        S_loss += self.mse_loss(l_y, l_f_by_y) * 25 * 10

        S_loss += self.mse_loss(label_expand_z[:, :, :, 0],
                                l_f_prob_by_z[:, :, :, 0]) * 0.5 * 10 \
                  + self.mse_loss(label_expand_z[:, :, :, 1],
                                  l_f_prob_by_z[:, :, :, 1]) * 5 * 10 \
                  + self.mse_loss(label_expand_z[:, :, :, 2],
                                  l_f_prob_by_z[:, :, :, 2]) * 25 * 10 \
                  + self.mse_loss(label_expand_z[:, :, :, 3],
                                  l_f_prob_by_z[:, :, :, 3]) * 25 * 10 \
                  + self.mse_loss(label_expand_z[:, :, :, 4],
                                  l_f_prob_by_z[:, :, :, 4]) * 25 * 10
        S_loss += self.mse_loss(l_z, l_f_by_z) * 25 * 10

        S_loss += self.mse_loss(label_expand_w[:, :, :, 0],
                                l_f_prob_by_w[:, :, :, 0]) * 0.5 * 10 \
                  + self.mse_loss(label_expand_w[:, :, :, 1],
                                  l_f_prob_by_w[:, :, :, 1]) * 5 * 10 \
                  + self.mse_loss(label_expand_w[:, :, :, 2],
                                  l_f_prob_by_w[:, :, :, 2]) * 25 * 10 \
                  + self.mse_loss(label_expand_w[:, :, :, 3],
                                  l_f_prob_by_w[:, :, :, 3]) * 25 * 10 \
                  + self.mse_loss(label_expand_w[:, :, :, 4],
                                  l_f_prob_by_w[:, :, :, 4]) * 25 * 10
        S_loss += self.mse_loss(l_w, l_f_by_w) * 25 * 10

        # contrast loss
        D_loss += self.mse_loss(j_x, 1.0) * 50 * 5
        D_loss += self.mse_loss(j_x_t_by_y, 0.0) * 15 * 5
        D_loss += self.mse_loss(j_x_t_by_z, 0.0) * 15 * 5
        D_loss += self.mse_loss(j_x_t_by_w, 0.0) * 15 * 5
        G_loss += self.mse_loss(j_x_t_by_y, 1.0) * 5 * 5
        G_loss += self.mse_loss(j_x_t_by_z, 1.0) * 5 * 5
        G_loss += self.mse_loss(j_x_t_by_w, 1.0) * 5 * 5

        D_loss += self.mse_loss(j_y, 1.0) * 50 * 5
        D_loss += self.mse_loss(j_y_t_by_x, 0.0) * 15 * 5
        D_loss += self.mse_loss(j_y_t_by_z, 0.0) * 15 * 5
        D_loss += self.mse_loss(j_y_t_by_w, 0.0) * 15 * 5
        G_loss += self.mse_loss(j_y_t_by_x, 1.0) * 5 * 5
        G_loss += self.mse_loss(j_y_t_by_z, 1.0) * 5 * 5
        G_loss += self.mse_loss(j_y_t_by_w, 1.0) * 5 * 5

        D_loss += self.mse_loss(j_z, 1.0) * 50 * 5
        D_loss += self.mse_loss(j_z_t_by_x, 0.0) * 15 * 5
        D_loss += self.mse_loss(j_z_t_by_y, 0.0) * 15 * 5
        D_loss += self.mse_loss(j_z_t_by_w, 0.0) * 15 * 5
        G_loss += self.mse_loss(j_z_t_by_x, 1.0) * 5 * 5
        G_loss += self.mse_loss(j_z_t_by_y, 1.0) * 5 * 5
        G_loss += self.mse_loss(j_z_t_by_w, 1.0) * 5 * 5

        D_loss += self.mse_loss(j_w, 1.0) * 50 * 5
        D_loss += self.mse_loss(j_w_t_by_x, 0.0) * 15 * 5
        D_loss += self.mse_loss(j_w_t_by_y, 0.0) * 15 * 5
        D_loss += self.mse_loss(j_w_t_by_z, 0.0) * 15 * 5
        G_loss += self.mse_loss(j_w_t_by_x, 1.0) * 5 * 5
        G_loss += self.mse_loss(j_w_t_by_y, 1.0) * 5 * 5
        G_loss += self.mse_loss(j_w_t_by_z, 1.0) * 5 * 5

        D_loss += self.mse_loss(j_x_c, cx) * 50 * 5
        D_loss += self.mse_loss(j_y_c, cy) * 50 * 5
        D_loss += self.mse_loss(j_z_c, cz) * 50 * 5
        D_loss += self.mse_loss(j_w_c, cw) * 50 * 5

        G_loss += self.mse_loss(j_x_t_c_by_y, cx) * 50 * 5
        G_loss += self.mse_loss(j_x_t_c_by_z, cx) * 50 * 5
        G_loss += self.mse_loss(j_x_t_c_by_w, cx) * 50 * 5

        G_loss += self.mse_loss(j_y_t_c_by_x, cy) * 50 * 5
        G_loss += self.mse_loss(j_y_t_c_by_z, cy) * 50 * 5
        G_loss += self.mse_loss(j_y_t_c_by_w, cy) * 50 * 5

        G_loss += self.mse_loss(j_z_t_c_by_x, cz) * 50 * 5
        G_loss += self.mse_loss(j_z_t_c_by_y, cz) * 50 * 5
        G_loss += self.mse_loss(j_z_t_c_by_w, cz) * 50 * 5

        G_loss += self.mse_loss(j_w_t_c_by_x, cw) * 50 * 5
        G_loss += self.mse_loss(j_w_t_c_by_y, cw) * 50 * 5
        G_loss += self.mse_loss(j_w_t_c_by_z, cw) * 50 * 5

        # label loss
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 0], l_prob_x_r[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 1], l_prob_x_r[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 2], l_prob_x_r[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 3], l_prob_x_r[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 4], l_prob_x_r[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 0], l_prob_y_r[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 1], l_prob_y_r[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 2], l_prob_y_r[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 3], l_prob_y_r[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 4], l_prob_y_r[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 0], l_prob_z_r[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 1], l_prob_z_r[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 2], l_prob_z_r[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 3], l_prob_z_r[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 4], l_prob_z_r[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 0], l_prob_w_r[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 1], l_prob_w_r[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 2], l_prob_w_r[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 3], l_prob_w_r[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 4], l_prob_w_r[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(l_f_by_x, l_x_r) * 0.5 * 50
        G_loss += self.mse_loss(l_f_by_y, l_y_r) * 0.5 * 50
        G_loss += self.mse_loss(l_f_by_z, l_z_r) * 0.5 * 50
        G_loss += self.mse_loss(l_f_by_w, l_w_r) * 0.5 * 50

        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 0], l_prob_x_t_by_y[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 1], l_prob_x_t_by_y[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 2], l_prob_x_t_by_y[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 3], l_prob_x_t_by_y[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 4], l_prob_x_t_by_y[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 0], l_prob_x_t_by_z[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 1], l_prob_x_t_by_z[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 2], l_prob_x_t_by_z[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 3], l_prob_x_t_by_z[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 4], l_prob_x_t_by_z[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 0], l_prob_x_t_by_w[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 1], l_prob_x_t_by_w[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 2], l_prob_x_t_by_w[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 3], l_prob_x_t_by_w[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 4], l_prob_x_t_by_w[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(l_f_by_z, l_x_t_by_z) * 0.5 * 50
        G_loss += self.mse_loss(l_f_by_y, l_x_t_by_y) * 0.5 * 50
        G_loss += self.mse_loss(l_f_by_w, l_x_t_by_w) * 0.5 * 50

        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 0], l_prob_y_t_by_x[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 1], l_prob_y_t_by_x[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 2], l_prob_y_t_by_x[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 3], l_prob_y_t_by_x[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 4], l_prob_y_t_by_x[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 0], l_prob_y_t_by_z[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 1], l_prob_y_t_by_z[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 2], l_prob_y_t_by_z[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 3], l_prob_y_t_by_z[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 4], l_prob_y_t_by_z[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 0], l_prob_y_t_by_w[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 1], l_prob_y_t_by_w[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 2], l_prob_y_t_by_w[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 3], l_prob_y_t_by_w[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 4], l_prob_y_t_by_w[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(l_f_by_x, l_y_t_by_x) * 0.5 * 50
        G_loss += self.mse_loss(l_f_by_z, l_y_t_by_z) * 0.5 * 50
        G_loss += self.mse_loss(l_f_by_w, l_y_t_by_w) * 0.5 * 50

        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 0], l_prob_z_t_by_x[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 1], l_prob_z_t_by_x[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 2], l_prob_z_t_by_x[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 3], l_prob_z_t_by_x[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 4], l_prob_z_t_by_x[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 0], l_prob_z_t_by_y[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 1], l_prob_z_t_by_y[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 2], l_prob_z_t_by_y[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 3], l_prob_z_t_by_y[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 4], l_prob_z_t_by_y[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 0], l_prob_z_t_by_w[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 1], l_prob_z_t_by_w[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 2], l_prob_z_t_by_w[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 3], l_prob_z_t_by_w[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 4], l_prob_z_t_by_w[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(l_f_by_x, l_z_t_by_x) * 0.5 * 50
        G_loss += self.mse_loss(l_f_by_y, l_z_t_by_y) * 0.5 * 50
        G_loss += self.mse_loss(l_f_by_w, l_z_t_by_w) * 0.5 * 50

        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 0], l_prob_w_t_by_x[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 1], l_prob_w_t_by_x[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 2], l_prob_w_t_by_x[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 3], l_prob_w_t_by_x[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 4], l_prob_w_t_by_x[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 0], l_prob_w_t_by_y[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 1], l_prob_w_t_by_y[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 2], l_prob_w_t_by_y[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 3], l_prob_w_t_by_y[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 4], l_prob_w_t_by_y[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 0], l_prob_w_t_by_z[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 1], l_prob_w_t_by_z[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 2], l_prob_w_t_by_z[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 3], l_prob_w_t_by_z[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 4], l_prob_w_t_by_z[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(l_f_by_x, l_w_t_by_x) * 0.5 * 50
        G_loss += self.mse_loss(l_f_by_y, l_w_t_by_y) * 0.5 * 50
        G_loss += self.mse_loss(l_f_by_z, l_w_t_by_z) * 0.5 * 50

        # label supervise loss
        G_loss += self.mse_loss(label_expand_y[:, :, :, 0], l_prob_x_t_by_y[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(label_expand_y[:, :, :, 1], l_prob_x_t_by_y[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(label_expand_y[:, :, :, 2], l_prob_x_t_by_y[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(label_expand_y[:, :, :, 3], l_prob_x_t_by_y[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(label_expand_y[:, :, :, 4], l_prob_x_t_by_y[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(label_expand_z[:, :, :, 0], l_prob_x_t_by_z[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(label_expand_z[:, :, :, 1], l_prob_x_t_by_z[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(label_expand_z[:, :, :, 2], l_prob_x_t_by_z[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(label_expand_z[:, :, :, 3], l_prob_x_t_by_z[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(label_expand_z[:, :, :, 4], l_prob_x_t_by_z[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(label_expand_w[:, :, :, 0], l_prob_x_t_by_w[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(label_expand_w[:, :, :, 1], l_prob_x_t_by_w[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(label_expand_w[:, :, :, 2], l_prob_x_t_by_w[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(label_expand_w[:, :, :, 3], l_prob_x_t_by_w[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(label_expand_w[:, :, :, 4], l_prob_x_t_by_w[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(l_z, l_x_t_by_z) * 0.5 * 50
        G_loss += self.mse_loss(l_y, l_x_t_by_y) * 0.5 * 50
        G_loss += self.mse_loss(l_w, l_x_t_by_w) * 0.5 * 50

        G_loss += self.mse_loss(label_expand_x[:, :, :, 0], l_prob_y_t_by_x[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(label_expand_x[:, :, :, 1], l_prob_y_t_by_x[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(label_expand_x[:, :, :, 2], l_prob_y_t_by_x[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(label_expand_x[:, :, :, 3], l_prob_y_t_by_x[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(label_expand_x[:, :, :, 4], l_prob_y_t_by_x[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(label_expand_z[:, :, :, 0], l_prob_y_t_by_z[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(label_expand_z[:, :, :, 1], l_prob_y_t_by_z[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(label_expand_z[:, :, :, 2], l_prob_y_t_by_z[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(label_expand_z[:, :, :, 3], l_prob_y_t_by_z[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(label_expand_z[:, :, :, 4], l_prob_y_t_by_z[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(label_expand_w[:, :, :, 0], l_prob_y_t_by_w[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(label_expand_w[:, :, :, 1], l_prob_y_t_by_w[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(label_expand_w[:, :, :, 2], l_prob_y_t_by_w[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(label_expand_w[:, :, :, 3], l_prob_y_t_by_w[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(label_expand_w[:, :, :, 4], l_prob_y_t_by_w[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(l_x, l_y_t_by_x) * 0.5 * 50
        G_loss += self.mse_loss(l_z, l_y_t_by_z) * 0.5 * 50
        G_loss += self.mse_loss(l_w, l_y_t_by_w) * 0.5 * 50

        G_loss += self.mse_loss(label_expand_x[:, :, :, 0], l_prob_z_t_by_x[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(label_expand_x[:, :, :, 1], l_prob_z_t_by_x[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(label_expand_x[:, :, :, 2], l_prob_z_t_by_x[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(label_expand_x[:, :, :, 3], l_prob_z_t_by_x[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(label_expand_x[:, :, :, 4], l_prob_z_t_by_x[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(label_expand_y[:, :, :, 0], l_prob_z_t_by_y[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(label_expand_y[:, :, :, 1], l_prob_z_t_by_y[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(label_expand_y[:, :, :, 2], l_prob_z_t_by_y[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(label_expand_y[:, :, :, 3], l_prob_z_t_by_y[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(label_expand_y[:, :, :, 4], l_prob_z_t_by_y[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(label_expand_w[:, :, :, 0], l_prob_z_t_by_w[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(label_expand_w[:, :, :, 1], l_prob_z_t_by_w[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(label_expand_w[:, :, :, 2], l_prob_z_t_by_w[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(label_expand_w[:, :, :, 3], l_prob_z_t_by_w[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(label_expand_w[:, :, :, 4], l_prob_z_t_by_w[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(l_x, l_z_t_by_x) * 0.5 * 50
        G_loss += self.mse_loss(l_y, l_z_t_by_y) * 0.5 * 50
        G_loss += self.mse_loss(l_w, l_z_t_by_w) * 0.5 * 50

        G_loss += self.mse_loss(label_expand_x[:, :, :, 0], l_prob_w_t_by_x[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(label_expand_x[:, :, :, 1], l_prob_w_t_by_x[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(label_expand_x[:, :, :, 2], l_prob_w_t_by_x[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(label_expand_x[:, :, :, 3], l_prob_w_t_by_x[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(label_expand_x[:, :, :, 4], l_prob_w_t_by_x[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(label_expand_y[:, :, :, 0], l_prob_w_t_by_y[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(label_expand_y[:, :, :, 1], l_prob_w_t_by_y[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(label_expand_y[:, :, :, 2], l_prob_w_t_by_y[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(label_expand_y[:, :, :, 3], l_prob_w_t_by_y[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(label_expand_y[:, :, :, 4], l_prob_w_t_by_y[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(label_expand_z[:, :, :, 0], l_prob_w_t_by_z[:, :, :, 0]) * 0.5 * 50
        G_loss += self.mse_loss(label_expand_z[:, :, :, 1], l_prob_w_t_by_z[:, :, :, 1]) * 1 * 50
        G_loss += self.mse_loss(label_expand_z[:, :, :, 2], l_prob_w_t_by_z[:, :, :, 2]) * 5 * 50
        G_loss += self.mse_loss(label_expand_z[:, :, :, 3], l_prob_w_t_by_z[:, :, :, 3]) * 5 * 50
        G_loss += self.mse_loss(label_expand_z[:, :, :, 4], l_prob_w_t_by_z[:, :, :, 4]) * 5 * 50

        G_loss += self.mse_loss(l_x, l_w_t_by_x) * 0.5 * 50
        G_loss += self.mse_loss(l_y, l_w_t_by_y) * 0.5 * 50
        G_loss += self.mse_loss(l_z, l_w_t_by_z) * 0.5 * 50

        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 0], l_prob_x_r_c_by_y[:, :, :, 0]) * 0.5 * 1
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 1], l_prob_x_r_c_by_y[:, :, :, 1]) * 5 * 1
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 2], l_prob_x_r_c_by_y[:, :, :, 2]) * 25 * 1
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 3], l_prob_x_r_c_by_y[:, :, :, 3]) * 25 * 1
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 4], l_prob_x_r_c_by_y[:, :, :, 4]) * 25 * 1

        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 0], l_prob_x_r_c_by_z[:, :, :, 0]) * 0.5 * 1
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 1], l_prob_x_r_c_by_z[:, :, :, 1]) * 5 * 1
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 2], l_prob_x_r_c_by_z[:, :, :, 2]) * 25 * 1
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 3], l_prob_x_r_c_by_z[:, :, :, 3]) * 25 * 1
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 4], l_prob_x_r_c_by_z[:, :, :, 4]) * 25 * 1

        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 0], l_prob_x_r_c_by_w[:, :, :, 0]) * 0.5 * 1
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 1], l_prob_x_r_c_by_w[:, :, :, 1]) * 1 * 1
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 2], l_prob_x_r_c_by_w[:, :, :, 2]) * 25 * 1
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 3], l_prob_x_r_c_by_w[:, :, :, 3]) * 25 * 1
        G_loss += self.mse_loss(l_f_prob_by_x[:, :, :, 4], l_prob_x_r_c_by_w[:, :, :, 4]) * 25 * 1

        G_loss += self.mse_loss(l_f_by_x, l_x_r_c_by_z) * 1 * 10
        G_loss += self.mse_loss(l_f_by_x, l_x_r_c_by_y) * 1 * 10
        G_loss += self.mse_loss(l_f_by_x, l_x_r_c_by_w) * 1 * 10
        G_loss += self.mse_loss(l_x_r_c_by_z, l_x_r_c_by_y) * 0.5
        G_loss += self.mse_loss(l_x_r_c_by_z, l_x_r_c_by_w) * 0.5 * 10
        G_loss += self.mse_loss(l_x_r_c_by_y, l_x_r_c_by_w) * 0.5 * 10

        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 0], l_prob_y_r_c_by_x[:, :, :, 0]) * 0.5 * 1
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 1], l_prob_y_r_c_by_x[:, :, :, 1]) * 5 * 1
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 2], l_prob_y_r_c_by_x[:, :, :, 2]) * 25 * 1
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 3], l_prob_y_r_c_by_x[:, :, :, 3]) * 25 * 1
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 4], l_prob_y_r_c_by_x[:, :, :, 4]) * 25 * 1

        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 0], l_prob_y_r_c_by_z[:, :, :, 0]) * 0.5 * 1
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 1], l_prob_y_r_c_by_z[:, :, :, 1]) * 5 * 1
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 2], l_prob_y_r_c_by_z[:, :, :, 2]) * 25 * 1
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 3], l_prob_y_r_c_by_z[:, :, :, 3]) * 25 * 1
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 4], l_prob_y_r_c_by_z[:, :, :, 4]) * 25 * 1

        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 0], l_prob_y_r_c_by_w[:, :, :, 0]) * 0.5 * 1
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 1], l_prob_y_r_c_by_w[:, :, :, 1]) * 5 * 1
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 2], l_prob_y_r_c_by_w[:, :, :, 2]) * 25 * 1
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 3], l_prob_y_r_c_by_w[:, :, :, 3]) * 25 * 1
        G_loss += self.mse_loss(l_f_prob_by_y[:, :, :, 4], l_prob_y_r_c_by_w[:, :, :, 4]) * 25 * 1

        G_loss += self.mse_loss(l_f_by_y, l_y_r_c_by_x) * 5 * 1
        G_loss += self.mse_loss(l_f_by_y, l_y_r_c_by_z) * 5 * 1
        G_loss += self.mse_loss(l_f_by_y, l_y_r_c_by_w) * 5 * 1
        G_loss += self.mse_loss(l_y_r_c_by_x, l_y_r_c_by_z) * 0.5 * 1
        G_loss += self.mse_loss(l_y_r_c_by_x, l_y_r_c_by_w) * 0.5 * 1
        G_loss += self.mse_loss(l_y_r_c_by_z, l_y_r_c_by_w) * 0.5 * 1

        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 0], l_prob_z_r_c_by_x[:, :, :, 0]) * 0.5 * 1
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 1], l_prob_z_r_c_by_x[:, :, :, 1]) * 5 * 1
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 2], l_prob_z_r_c_by_x[:, :, :, 2]) * 25 * 1
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 3], l_prob_z_r_c_by_x[:, :, :, 3]) * 25 * 1
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 4], l_prob_z_r_c_by_x[:, :, :, 4]) * 25 * 1

        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 0], l_prob_z_r_c_by_y[:, :, :, 0]) * 0.5 * 1
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 1], l_prob_z_r_c_by_y[:, :, :, 1]) * 5 * 1
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 2], l_prob_z_r_c_by_y[:, :, :, 2]) * 25 * 1
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 3], l_prob_z_r_c_by_y[:, :, :, 3]) * 25 * 1
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 4], l_prob_z_r_c_by_y[:, :, :, 4]) * 25 * 1

        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 0], l_prob_z_r_c_by_w[:, :, :, 0]) * 0.5 * 1
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 1], l_prob_z_r_c_by_w[:, :, :, 1]) * 5 * 1
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 2], l_prob_z_r_c_by_w[:, :, :, 2]) * 25 * 1
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 3], l_prob_z_r_c_by_w[:, :, :, 3]) * 25 * 1
        G_loss += self.mse_loss(l_f_prob_by_z[:, :, :, 4], l_prob_z_r_c_by_w[:, :, :, 4]) * 25 * 1

        G_loss += self.mse_loss(l_f_by_z, l_z_r_c_by_x) * 5 * 1
        G_loss += self.mse_loss(l_f_by_z, l_z_r_c_by_y) * 5 * 1
        G_loss += self.mse_loss(l_f_by_z, l_z_r_c_by_w) * 5 * 1
        G_loss += self.mse_loss(l_z_r_c_by_x, l_z_r_c_by_y) * 0.5 * 1
        G_loss += self.mse_loss(l_z_r_c_by_x, l_z_r_c_by_w) * 0.5 * 1
        G_loss += self.mse_loss(l_z_r_c_by_y, l_z_r_c_by_w) * 0.5 * 1

        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 0], l_prob_w_r_c_by_x[:, :, :, 0]) * 0.5 * 1
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 1], l_prob_w_r_c_by_x[:, :, :, 1]) * 5 * 1
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 2], l_prob_w_r_c_by_x[:, :, :, 2]) * 25 * 1
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 3], l_prob_w_r_c_by_x[:, :, :, 3]) * 25 * 1
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 4], l_prob_w_r_c_by_x[:, :, :, 4]) * 25 * 1

        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 0], l_prob_w_r_c_by_y[:, :, :, 0]) * 0.5 * 1
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 1], l_prob_w_r_c_by_y[:, :, :, 1]) * 5 * 1
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 2], l_prob_w_r_c_by_y[:, :, :, 2]) * 25 * 1
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 3], l_prob_w_r_c_by_y[:, :, :, 3]) * 25 * 1
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 4], l_prob_w_r_c_by_y[:, :, :, 4]) * 25 * 1

        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 0], l_prob_w_r_c_by_z[:, :, :, 0]) * 0.5 * 1
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 1], l_prob_w_r_c_by_z[:, :, :, 1]) * 5 * 1
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 2], l_prob_w_r_c_by_z[:, :, :, 2]) * 25 * 1
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 3], l_prob_w_r_c_by_z[:, :, :, 3]) * 25 * 1
        G_loss += self.mse_loss(l_f_prob_by_w[:, :, :, 4], l_prob_w_r_c_by_z[:, :, :, 4]) * 25 * 1

        G_loss += self.mse_loss(l_f_by_w, l_w_r_c_by_x) * 5 * 1
        G_loss += self.mse_loss(l_f_by_w, l_w_r_c_by_y) * 5 * 1
        G_loss += self.mse_loss(l_f_by_w, l_w_r_c_by_z) * 5 * 1
        G_loss += self.mse_loss(l_w_r_c_by_x, l_w_r_c_by_y) * 0.5 * 1
        G_loss += self.mse_loss(l_w_r_c_by_x, l_w_r_c_by_z) * 0.5 * 1
        G_loss += self.mse_loss(l_w_r_c_by_y, l_w_r_c_by_z) * 0.5 * 1

        # self supervise loss between rebuild and original image
        G_loss += self.mse_loss(x, x_r) * 5 * 0.01
        G_loss += self.mse_loss(y, y_r) * 5 * 0.01
        G_loss += self.mse_loss(z, z_r) * 5 * 0.01
        G_loss += self.mse_loss(w, w_r) * 5 * 0.01

        # self supervise loss between cycle rebuild and original image
        G_loss += self.mse_loss(x, x_r_c_by_y) * 10 * 0.01
        G_loss += self.mse_loss(x, x_r_c_by_z) * 10 * 0.01
        G_loss += self.mse_loss(x, x_r_c_by_w) * 10 * 0.01
        G_loss += self.mse_loss(x_r_c_by_y, x_r_c_by_z) * 2 * 0.01
        G_loss += self.mse_loss(x_r_c_by_y, x_r_c_by_w) * 2 * 0.01
        G_loss += self.mse_loss(x_r_c_by_z, x_r_c_by_w) * 2 * 0.01

        G_loss += self.mse_loss(y, y_r_c_by_x) * 10 * 0.01
        G_loss += self.mse_loss(y, y_r_c_by_z) * 10 * 0.01
        G_loss += self.mse_loss(y, y_r_c_by_w) * 10 * 0.01
        G_loss += self.mse_loss(y_r_c_by_x, y_r_c_by_z) * 2 * 0.01
        G_loss += self.mse_loss(y_r_c_by_x, y_r_c_by_w) * 2 * 0.01
        G_loss += self.mse_loss(y_r_c_by_z, y_r_c_by_w) * 2 * 0.01

        G_loss += self.mse_loss(z, z_r_c_by_x) * 10 * 0.01
        G_loss += self.mse_loss(z, z_r_c_by_y) * 10 * 0.01
        G_loss += self.mse_loss(z, z_r_c_by_w) * 10 * 0.01
        G_loss += self.mse_loss(z_r_c_by_x, z_r_c_by_y) * 2 * 0.01
        G_loss += self.mse_loss(z_r_c_by_x, z_r_c_by_w) * 2 * 0.01
        G_loss += self.mse_loss(z_r_c_by_y, z_r_c_by_w) * 2 * 0.01

        G_loss += self.mse_loss(w, w_r_c_by_x) * 10 * 0.01
        G_loss += self.mse_loss(w, w_r_c_by_y) * 10 * 0.01
        G_loss += self.mse_loss(w, w_r_c_by_z) * 10 * 0.01
        G_loss += self.mse_loss(w_r_c_by_x, w_r_c_by_y) * 2 * 0.01
        G_loss += self.mse_loss(w_r_c_by_x, w_r_c_by_z) * 2 * 0.01
        G_loss += self.mse_loss(w_r_c_by_y, w_r_c_by_z) * 2 * 0.01

        # self supervise loss between cycle rebuild and original image
        G_loss += self.mse_loss(0.0, x_t_by_y * mask_y) * 5
        G_loss += self.mse_loss(0.0, x_t_by_z * mask_z) * 5
        G_loss += self.mse_loss(0.0, x_t_by_w * mask_w) * 5

        G_loss += self.mse_loss(0.0, y_t_by_x * mask_x) * 5
        G_loss += self.mse_loss(0.0, y_t_by_z * mask_z) * 5
        G_loss += self.mse_loss(0.0, y_t_by_w * mask_w) * 5

        G_loss += self.mse_loss(0.0, z_t_by_x * mask_x) * 5
        G_loss += self.mse_loss(0.0, z_t_by_y * mask_y) * 5
        G_loss += self.mse_loss(0.0, z_t_by_w * mask_w) * 5

        G_loss += self.mse_loss(0.0, w_t_by_x * mask_x) * 5
        G_loss += self.mse_loss(0.0, w_t_by_y * mask_y) * 5
        G_loss += self.mse_loss(0.0, w_t_by_z * mask_z) * 5

        G_loss += self.mse_loss(0.0, x_r * mask_x) * 0.5 * 0.01
        G_loss += self.mse_loss(0.0, y_r * mask_y) * 0.5 * 0.01
        G_loss += self.mse_loss(0.0, z_r * mask_z) * 0.5 * 0.01
        G_loss += self.mse_loss(0.0, w_r * mask_w) * 0.5 * 0.01

        # code loss
        G_loss += self.mse_loss(code_x, code_y_t_by_x) * 5 * 0.01
        G_loss += self.mse_loss(code_x, code_z_t_by_x) * 5 * 0.01
        G_loss += self.mse_loss(code_x, code_w_t_by_x) * 5 * 0.01
        G_loss += self.mse_loss(code_y_t_by_x, code_z_t_by_x) * 0.01
        G_loss += self.mse_loss(code_y_t_by_x, code_w_t_by_x) * 0.01
        G_loss += self.mse_loss(code_z_t_by_x, code_w_t_by_x) * 0.01

        G_loss += self.mse_loss(code_y, code_x_t_by_y) * 5 * 0.01
        G_loss += self.mse_loss(code_y, code_z_t_by_y) * 5 * 0.01
        G_loss += self.mse_loss(code_y, code_w_t_by_y) * 5 * 0.01
        G_loss += self.mse_loss(code_x_t_by_y, code_z_t_by_y) * 0.01
        G_loss += self.mse_loss(code_x_t_by_y, code_w_t_by_y) * 0.01
        G_loss += self.mse_loss(code_z_t_by_y, code_w_t_by_y) * 0.01

        G_loss += self.mse_loss(code_z, code_x_t_by_z) * 5 * 0.01
        G_loss += self.mse_loss(code_z, code_y_t_by_z) * 5 * 0.01
        G_loss += self.mse_loss(code_z, code_w_t_by_z) * 5 * 0.01
        G_loss += self.mse_loss(code_x_t_by_z, code_y_t_by_z) * 0.01
        G_loss += self.mse_loss(code_x_t_by_z, code_w_t_by_z) * 0.01
        G_loss += self.mse_loss(code_y_t_by_z, code_w_t_by_z) * 0.01

        G_loss += self.mse_loss(code_w, code_x_t_by_w) * 5 * 0.01
        G_loss += self.mse_loss(code_w, code_y_t_by_w) * 5 * 0.01
        G_loss += self.mse_loss(code_w, code_z_t_by_w) * 5 * 0.01
        G_loss += self.mse_loss(code_x_t_by_w, code_y_t_by_w) * 0.01
        G_loss += self.mse_loss(code_x_t_by_w, code_z_t_by_w) * 0.01
        G_loss += self.mse_loss(code_y_t_by_w, code_z_t_by_w) * 0.01

        # sobel loss
        G_loss += self.sobel_loss(x, x_r_c_by_y) * 10 * 0.01
        G_loss += self.sobel_loss(x, x_r_c_by_z) * 10 * 0.01
        G_loss += self.sobel_loss(x, x_r_c_by_w) * 10 * 0.01

        G_loss += self.sobel_loss(y, y_r_c_by_x) * 10 * 0.01
        G_loss += self.sobel_loss(y, y_r_c_by_z) * 10 * 0.01
        G_loss += self.sobel_loss(y, y_r_c_by_w) * 10 * 0.01

        G_loss += self.sobel_loss(z, z_r_c_by_x) * 10 * 0.01
        G_loss += self.sobel_loss(z, z_r_c_by_y) * 10 * 0.01
        G_loss += self.sobel_loss(z, z_r_c_by_w) * 10 * 0.01

        G_loss += self.sobel_loss(w, w_r_c_by_x) * 10 * 0.01
        G_loss += self.sobel_loss(w, w_r_c_by_y) * 10 * 0.01
        G_loss += self.sobel_loss(w, w_r_c_by_z) * 10 * 0.01

        # ssim loss
        G_loss += self.ssim_loss(x, x_r_c_by_y) * 10 * 0.01
        G_loss += self.ssim_loss(x, x_r_c_by_z) * 10 * 0.01
        G_loss += self.ssim_loss(x, x_r_c_by_w) * 10 * 0.01

        G_loss += self.ssim_loss(y, y_r_c_by_x) * 10 * 0.01
        G_loss += self.ssim_loss(y, y_r_c_by_z) * 10 * 0.01
        G_loss += self.ssim_loss(y, y_r_c_by_w) * 10 * 0.01

        G_loss += self.ssim_loss(z, z_r_c_by_x) * 10 * 0.01
        G_loss += self.ssim_loss(z, z_r_c_by_y) * 10 * 0.01
        G_loss += self.ssim_loss(z, z_r_c_by_w) * 10 * 0.01

        G_loss += self.ssim_loss(w, w_r_c_by_x) * 10 * 0.01
        G_loss += self.ssim_loss(w, w_r_c_by_y) * 10 * 0.01
        G_loss += self.ssim_loss(w, w_r_c_by_z) * 10 * 0.01

        image_list["l_x"] = l_x
        image_list["l_y"] = l_y
        image_list["l_z"] = l_z
        image_list["l_w"] = l_w
        image_list["x"] = x
        image_list["y"] = y
        image_list["z"] = z
        image_list["w"] = w

        image_list["mask_x"] = mask_x
        image_list["mask_y"] = mask_y
        image_list["mask_z"] = mask_z
        image_list["mask_w"] = mask_w

        self.code_list["code_x"] = code_x
        self.code_list["code_y"] = code_y
        self.code_list["code_z"] = code_z
        self.code_list["code_w"] = code_w

        image_list["l_x_r"] = l_x_r
        image_list["l_y_r"] = l_y_r
        image_list["l_z_r"] = l_z_r
        image_list["l_w_r"] = l_w_r

        self.prob_list["l_f_prob_by_x"] = l_f_prob_by_x
        self.prob_list["l_f_prob_by_y"] = l_f_prob_by_y
        self.prob_list["l_f_prob_by_z"] = l_f_prob_by_z
        self.prob_list["l_f_prob_by_w"] = l_f_prob_by_w
        image_list["l_f_by_x"] = l_f_by_x
        image_list["l_f_by_y"] = l_f_by_y
        image_list["l_f_by_z"] = l_f_by_z
        image_list["l_f_by_w"] = l_f_by_w

        image_list["x_r"] = x_r
        image_list["y_r"] = y_r
        image_list["z_r"] = z_r
        image_list["w_r"] = w_r

        image_list["y_t_by_x"] = y_t_by_x
        self.code_list["code_y_t_by_x"] = code_y_t_by_x
        image_list["x_r_c_by_y"] = x_r_c_by_y
        image_list["z_t_by_x"] = z_t_by_x
        self.code_list["code_z_t_by_x"] = code_z_t_by_x
        image_list["x_r_c_by_z"] = x_r_c_by_z
        image_list["w_t_by_x"] = w_t_by_x
        self.code_list["code_w_t_by_x"] = code_w_t_by_x
        image_list["x_r_c_by_w"] = x_r_c_by_w

        image_list["x_t_by_y"] = x_t_by_y
        self.code_list["code_x_t_by_y"] = code_x_t_by_y
        image_list["y_r_c_by_x"] = y_r_c_by_x
        image_list["z_t_by_y"] = z_t_by_y
        self.code_list["code_z_t_by_y"] = code_z_t_by_y
        image_list["y_r_c_by_z"] = y_r_c_by_z
        image_list["w_t_by_y"] = w_t_by_y
        self.code_list["code_w_t_by_y"] = code_w_t_by_y
        image_list["y_r_c_by_w"] = y_r_c_by_w

        image_list["x_t_by_z"] = x_t_by_z
        self.code_list["code_x_t_by_z"] = code_x_t_by_z
        image_list["z_r_c_by_x"] = z_r_c_by_x
        image_list["y_t_by_z"] = y_t_by_z
        self.code_list["code_y_t_by_z"] = code_y_t_by_z
        image_list["z_r_c_by_y"] = z_r_c_by_y
        image_list["w_t_by_z"] = w_t_by_z
        self.code_list["code_w_t_by_z"] = code_w_t_by_z
        image_list["z_r_c_by_w"] = z_r_c_by_w

        image_list["x_t_by_w"] = x_t_by_w
        self.code_list["code_x_t_by_w"] = code_x_t_by_w
        image_list["w_r_c_by_x"] = w_r_c_by_x
        image_list["y_t_by_w"] = y_t_by_w
        self.code_list["code_y_t_by_w"] = code_y_t_by_w
        image_list["w_r_c_by_y"] = w_r_c_by_y
        image_list["z_t_by_w"] = z_t_by_w
        self.code_list["code_z_t_by_w"] = code_z_t_by_w
        image_list["w_r_c_by_z"] = w_r_c_by_z

        image_list["l_x_t_by_y"] = l_x_t_by_y
        image_list["l_x_t_by_z"] = l_x_t_by_z
        image_list["l_x_t_by_w"] = l_x_t_by_w

        image_list["l_y_t_by_x"] = l_y_t_by_x
        image_list["l_y_t_by_z"] = l_y_t_by_z
        image_list["l_y_t_by_w"] = l_y_t_by_w

        image_list["l_z_t_by_x"] = l_z_t_by_x
        image_list["l_z_t_by_y"] = l_z_t_by_y
        image_list["l_z_t_by_w"] = l_z_t_by_w

        image_list["l_w_t_by_x"] = l_w_t_by_x
        image_list["l_w_t_by_y"] = l_w_t_by_y
        image_list["l_w_t_by_z"] = l_w_t_by_z

        image_list["l_x_r_c_by_y"] = l_x_r_c_by_y
        image_list["l_x_r_c_by_z"] = l_x_r_c_by_z
        image_list["l_x_r_c_by_w"] = l_x_r_c_by_w

        image_list["l_y_r_c_by_x"] = l_y_r_c_by_x
        image_list["l_y_r_c_by_z"] = l_y_r_c_by_z
        image_list["l_y_r_c_by_w"] = l_y_r_c_by_w

        image_list["l_z_r_c_by_x"] = l_z_r_c_by_x
        image_list["l_z_r_c_by_y"] = l_z_r_c_by_y
        image_list["l_z_r_c_by_w"] = l_z_r_c_by_w

        image_list["l_w_r_c_by_x"] = l_w_r_c_by_x
        image_list["l_w_r_c_by_y"] = l_w_r_c_by_y
        image_list["l_w_r_c_by_z"] = l_w_r_c_by_z

        self.judge_list["j_x"], self.judge_list["j_x_c"] = j_x, j_x_c
        self.judge_list["j_y"], self.judge_list["j_y_c"] = j_y, j_y_c
        self.judge_list["j_z"], self.judge_list["j_z_c"] = j_z, j_z_c
        self.judge_list["j_w"], self.judge_list["j_w_c"] = j_w, j_w_c

        self.judge_list["j_x_t_by_y"], self.judge_list["j_x_t_c_by_y"] = j_x_t_by_y, j_x_t_c_by_y
        self.judge_list["j_x_t_by_z"], self.judge_list["j_x_t_c_by_z"] = j_x_t_by_z, j_x_t_c_by_z
        self.judge_list["j_x_t_by_w"], self.judge_list["j_x_t_c_by_w"] = j_x_t_by_w, j_x_t_c_by_w

        self.judge_list["j_y_t_by_x"], self.judge_list["j_y_t_c_by_x"] = j_y_t_by_x, j_y_t_c_by_x
        self.judge_list["j_y_t_by_z"], self.judge_list["j_y_t_c_by_z"] = j_y_t_by_z, j_y_t_c_by_z
        self.judge_list["j_y_t_by_w"], self.judge_list["j_y_t_c_by_w"] = j_y_t_by_w, j_y_t_c_by_w

        self.judge_list["j_z_t_by_x"], self.judge_list["j_z_t_c_by_x"] = j_z_t_by_x, j_z_t_c_by_x
        self.judge_list["j_z_t_by_y"], self.judge_list["j_z_t_c_by_y"] = j_z_t_by_y, j_z_t_c_by_y
        self.judge_list["j_z_t_by_w"], self.judge_list["j_z_t_c_by_w"] = j_z_t_by_w, j_z_t_c_by_w

        self.judge_list["j_w_t_by_x"], self.judge_list["j_w_t_c_by_x"] = j_w_t_by_x, j_w_t_c_by_x
        self.judge_list["j_w_t_by_y"], self.judge_list["j_w_t_c_by_y"] = j_w_t_by_y, j_w_t_c_by_y
        self.judge_list["j_w_t_by_z"], self.judge_list["j_w_t_c_by_z"] = j_w_t_by_z, j_w_t_c_by_z

        loss_list = [G_loss + S_loss, D_loss, S_loss]

        return loss_list, image_list, self.code_list, self.judge_list

    def get_variables(self):
        return [self.EC_M.variables
                + self.DC_X.variables
                + self.DC_Y.variables
                + self.DC_Z.variables
                + self.DC_W.variables
            ,
                self.D_M.variables
            ,
                self.EC_L.variables
                + self.DC_L.variables
                ]

    def optimize(self):
        def make_optimizer(name='Adam'):
            learning_step = (
                tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, name=name)
            )
            return learning_step

        G_optimizer = make_optimizer(name='Adam_G')
        D_optimizer = make_optimizer(name='Adam_D')

        return G_optimizer, D_optimizer

    def histogram_summary(self, judge_dirct):
        for key in judge_dirct:
            tf.summary.image('discriminator/' + key, judge_dirct[key])

    def loss_summary(self, loss_list):
        G_loss, D_loss, S_loss = loss_list[0], loss_list[1], loss_list[2]
        tf.summary.scalar('loss/G_loss', G_loss)
        tf.summary.scalar('loss/D_loss', D_loss)
        tf.summary.scalar('loss/S_loss', S_loss)

    def image_summary(self, image_dirct):
        for key in image_dirct:
            tf.summary.image('image/' + key, image_dirct[key])

    def mse_loss(self, x, y):
        """ supervised loss (L2 norm)
        """
        loss = tf.reduce_mean(tf.square(x - y))
        return loss

    def sobel_loss(self, x, y):
        fx = self.norm(tf.reduce_max(tf.image.sobel_edges(x), axis=-1))
        fy = self.norm(tf.reduce_max(tf.image.sobel_edges(y), axis=-1))
        loss = self.mse_loss(fx, fy)
        return loss

    def ssim_loss(self, x, y):
        """ supervised loss (L2 norm)
        """
        loss = (1.0 - self.SSIM(x, y)) * 20
        return loss

    def PSNR(self, output, target):
        psnr = tf.reduce_mean(tf.image.psnr(output, target, max_val=1.0, name="psnr"))
        return psnr

    def SSIM(self, output, target):
        ssim = tf.reduce_mean(tf.image.ssim(output, target, max_val=1.0))
        return ssim

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

    def norm(self, input):
        output = (input - tf.reduce_min(input, axis=[1, 2, 3])
                  ) / (tf.reduce_max(input, axis=[1, 2, 3]) - tf.reduce_min(input, axis=[1, 2, 3]) + 1e-6)
        return output
