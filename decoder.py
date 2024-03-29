# _*_ coding:utf-8 _*_
import tensorflow as tf
import ops as ops
import logging


class Decoder:
    def __init__(self, name, ngf=64, keep_prob=1.0, output_channl=1, reuse=False):
        """
        Args:
          name: string, model name
          ngf: int, number of gen filters in first conv layer
          keep_prob: float, dropout rate
          output_channl: int, equal to class_nums
        """
        self.name = name
        self.reuse = reuse
        self.ngf = ngf
        self.keep_prob = keep_prob
        self.output_channl = output_channl

    def __call__(self, DC_input):
        """
        Args:
          input: batch_size x width x height x N
        Returns:
          output: same size as input
        """
        with tf.variable_scope(self.name, reuse=self.reuse):
            DC_input = tf.nn.dropout(DC_input, keep_prob=self.keep_prob)
            with tf.variable_scope("conv1", reuse=self.reuse):
                conv1 = tf.layers.conv2d(inputs=DC_input, filters=4 * self.ngf, kernel_size=3, strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * 4 * self.ngf), stddev=0.000001,
                                             dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv1')
                norm1 = ops._norm(conv1)
                relu1 = ops.relu(norm1)
            with tf.variable_scope("deconv1_r", reuse=self.reuse):
                resize1 = ops.uk_resize(relu1, reuse=self.reuse, name='resize1')
                deconv1_r = tf.layers.conv2d(inputs=resize1, filters=4 * self.ngf, kernel_size=3, strides=1,
                                             padding="SAME",
                                             activation=None,
                                             kernel_initializer=tf.random_normal_initializer(
                                                 mean=1.0 / (9.0 * 4 * self.ngf), stddev=0.000001,
                                                 dtype=tf.float32),
                                             bias_initializer=tf.constant_initializer(0.0),
                                             name='deconv1_r')
                deconv1_norm1_r = ops._norm(deconv1_r)
            with tf.variable_scope("deconv1_t", reuse=self.reuse):
                deconv1_t = tf.layers.conv2d_transpose(inputs=relu1, filters=4 * self.ngf, kernel_size=3,
                                                       strides=2,
                                                       padding="SAME",
                                                       activation=None,
                                                       kernel_initializer=tf.random_normal_initializer(
                                                           mean=1.0 / (9.0 * 4 * self.ngf), stddev=0.000001,
                                                           dtype=tf.float32),
                                                       bias_initializer=tf.constant_initializer(0.0),
                                                       name='deconv1_t')
                deconv1_norm1_t = ops._norm(deconv1_t)
            with tf.variable_scope("add1", reuse=self.reuse):
                add1 = ops.relu(tf.add(deconv1_norm1_r * 0.85, deconv1_norm1_t * 0.15))
            with tf.variable_scope("add1_conv1", reuse=self.reuse):
                add1_conv1 = tf.layers.conv2d(inputs=add1, filters=4 * self.ngf, kernel_size=3,
                                              strides=1,
                                              padding="SAME",
                                              activation=None,
                                              kernel_initializer=tf.random_normal_initializer(
                                                  mean=1.0 / (9.0 * 4 * self.ngf), stddev=0.000001,
                                                  dtype=tf.float32),
                                              bias_initializer=tf.constant_initializer(0.0),
                                              name='add1_conv1')
                add1_norm1 = ops._norm(add1_conv1)
                add1_relu1 = ops.relu(add1_norm1)
            with tf.variable_scope("deconv2_r", reuse=self.reuse):
                resize2 = ops.uk_resize(add1_relu1, reuse=self.reuse, name='resize2')
                deconv2_r = tf.layers.conv2d(inputs=resize2, filters=2 * self.ngf, kernel_size=3, strides=1,
                                             padding="SAME",
                                             activation=None,
                                             kernel_initializer=tf.random_normal_initializer(
                                                 mean=1.0 / (9.0 * 4 * self.ngf), stddev=0.000001,
                                                 dtype=tf.float32),
                                             bias_initializer=tf.constant_initializer(0.0),
                                             name='deconv2_r')
                deconv2_norm1_r = ops._norm(deconv2_r)
            with tf.variable_scope("deconv2_t", reuse=self.reuse):
                deconv2_t = tf.layers.conv2d_transpose(inputs=add1_relu1, filters=2 * self.ngf,
                                                       kernel_size=3,
                                                       strides=2,
                                                       padding="SAME",
                                                       activation=None,
                                                       kernel_initializer=tf.random_normal_initializer(
                                                           mean=1.0 / (9.0 * 4 * self.ngf), stddev=0.000001,
                                                           dtype=tf.float32),
                                                       bias_initializer=tf.constant_initializer(0.0),
                                                       name='deconv2_t')
                deconv2_norm1_t = ops._norm(deconv2_t)
            with tf.variable_scope("add2", reuse=self.reuse):
                add2 = ops.relu(tf.add(deconv2_norm1_r * 0.85, deconv2_norm1_t * 0.15))
            with tf.variable_scope("add2_conv1", reuse=self.reuse):
                add2_conv1 = tf.layers.conv2d(inputs=add2, filters=2 * self.ngf, kernel_size=3,
                                              strides=1,
                                              padding="SAME",
                                              activation=None,
                                              kernel_initializer=tf.random_normal_initializer(
                                                  mean=1.0 / (9.0 * 2 * self.ngf), stddev=0.000001,
                                                  dtype=tf.float32),
                                              bias_initializer=tf.constant_initializer(0.0),
                                              name='add2_conv1')
                add2_norm1 = ops._norm(add2_conv1)
                add2_relu1 = ops.relu(add2_norm1)
            with tf.variable_scope("conv2", reuse=self.reuse):
                conv2 = tf.layers.conv2d(inputs=add2_relu1, filters=self.ngf, kernel_size=3, strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv12')
                norm2 = ops._norm(conv2)
                relu2 = ops.relu(norm2)
            with tf.variable_scope("lastconv", reuse=self.reuse):
                lastconv = tf.layers.conv2d(inputs=relu2, filters=1, kernel_size=3, strides=1,
                                            padding="SAME",
                                            activation=tf.nn.sigmoid,
                                            kernel_initializer=tf.random_normal_initializer(
                                                mean=1.0 / (9.0 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                            bias_initializer=tf.constant_initializer(0.0), name='lastconv')
                lastnorm = ops._norm(lastconv)
                output = tf.nn.sigmoid(lastnorm)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return output
