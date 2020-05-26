#! -*- coding: utf-8 -*-
# 自定义层

import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.backend import search_layer
from bert4keras.backend import sequence_masking
from bert4keras.backend import pool1d
from bert4keras.backend import divisible_temporal_padding
from bert4keras.snippets import is_string
from keras import initializers, activations

from keras.layers import *


def integerize_shape(func):
    """装饰器，保证input_shape一定是int或None
    """
    def convert(item):
        if hasattr(item, '__iter__'):
            return [convert(i) for i in item]
        elif hasattr(item, 'value'):
            return item.value
        else:
            return item

    def new_func(self, input_shape):
        input_shape = convert(input_shape)
        return func(self, input_shape)

    return new_func



class Layer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Layer, self).__init__(**kwargs)
        self.supports_masking = True  # 本项目的自定义层均可mask


class Embedding(keras.layers.Embedding):
    """拓展Embedding层
    """
    def compute_mask(self, inputs, mask=None):
        """为了适配T5，保证第一个token不被mask
        """
        if self._current_mode == 'embedding':
            mask = super(Embedding, self).compute_mask(inputs, mask)
            if mask is not None:
                mask1 = K.ones_like(mask[:, :1], dtype='bool')
                mask2 = mask[:, 1:]
                return K.concatenate([mask1, mask2], 1)
        else:
            return mask

    def call(self, inputs, mode='embedding'):
        """新增mode参数，可以为embedding或dense。如果为embedding，
        则等价于普通Embedding层；如果为dense，则等价于无bias的Dense层。
        """
        self._current_mode = mode
        if mode == 'embedding':
            return super(Embedding, self).call(inputs)
        else:
            kernel = K.transpose(self.embeddings)
            return K.dot(inputs, kernel)

    def compute_output_shape(self, input_shape):
        if self._current_mode == 'embedding':
            return super(Embedding, self).compute_output_shape(input_shape)
        else:
            return input_shape[:2] + (K.int_shape(self.embeddings)[0],)


class BiasAdd(Layer):
    """加上偏置项
    """
    @integerize_shape
    def build(self, input_shape):
        super(BiasAdd, self).build(input_shape)
        output_dim = input_shape[-1]
        self.bias = self.add_weight(
            name='bias',
            shape=(output_dim,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        return K.bias_add(inputs, self.bias)


class MultiHeadAttention(Layer):
    """多头注意力机制
    """
    def __init__(
        self,
        heads,
        head_size,
        key_size=None,
        use_bias=True,
        scaled_dot_product=True,
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.out_dim = heads * head_size
        self.key_size = key_size or head_size
        self.use_bias = use_bias
        self.scaled_dot_product = scaled_dot_product
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = Dense(
            units=self.key_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.k_dense = Dense(
            units=self.key_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.v_dense = Dense(
            units=self.out_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.o_dense = Dense(
            units=self.out_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    def call(self, inputs, mask=None, a_mask=None, p_bias=None):
        """实现多头注意力
        q_mask: 对输入的query序列的mask。
                主要是将输出结果的padding部分置0。
        v_mask: 对输入的value序列的mask。
                主要是防止attention读取到padding信息。
        a_mask: 对attention矩阵的mask。
                不同的attention mask对应不同的应用。
        p_bias: 在attention里的位置偏置。
                一般用来指定相对位置编码的种类。
        """
        q, k, v = inputs[:3]
        q_mask, v_mask, n = None, None, 3
        if mask is not None:
            if mask[0] is not None:
                q_mask = K.cast(mask[0], K.floatx())
            if mask[2] is not None:
                v_mask = K.cast(mask[2], K.floatx())
        if a_mask:
            a_mask = inputs[n]
            n += 1
        # 线性变换
        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(q)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(k)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(v)[1], self.heads, self.head_size))
        # Attention
        a = tf.einsum('bjhd,bkhd->bhjk', qw, kw)
        # 处理位置编码
        if p_bias == 'typical_relative':
            pos_embeddings = inputs[n]
            a = a + tf.einsum('bjhd,jkd->bhjk', qw, pos_embeddings)
        elif p_bias == 't5_relative':
            pos_embeddings = K.permute_dimensions(inputs[n], (2, 0, 1))
            a = a + K.expand_dims(pos_embeddings, 0)
        # Attention（续）
        if self.scaled_dot_product:
            a = a / self.key_size**0.5
        a = sequence_masking(a, v_mask, 1, -1)
        if a_mask is not None:
            a = a - (1 - a_mask) * 1e12
        a = K.softmax(a)
        # 完成输出
        o = tf.einsum('bhjk,bkhd->bjhd', a, vw)
        if p_bias == 'typical_relative':
            o = o + tf.einsum('bhjk,jkd->bjhd', a, pos_embeddings)
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.o_dense(o)
        # 返回结果
        o = sequence_masking(o, q_mask, 0)
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return mask[0]

    def get_config(self):
        config = {
            'heads': self.heads,
            'head_size': self.head_size,
            'key_size': self.key_size,
            'use_bias': self.use_bias,
            'scaled_dot_product': self.scaled_dot_product,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LayerNormalization(Layer):
    """(Conditional) Layer Normalization
    hidden_*系列参数仅为有条件输入时(conditional=True)使用
    """
    def __init__(
        self,
        center=True,
        scale=True,
        epsilon=None,
        conditional=False,
        hidden_units=None,
        hidden_activation='linear',
        hidden_initializer='glorot_uniform',
        **kwargs
    ):
        super(LayerNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_activation = activations.get(hidden_activation)
        self.hidden_initializer = initializers.get(hidden_initializer)
        self.epsilon = epsilon or 1e-12

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)

        if self.conditional:
            shape = (input_shape[0][-1],)
        else:
            shape = (input_shape[-1],)

        if self.center:
            self.beta = self.add_weight(
                shape=shape, initializer='zeros', name='beta'
            )
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape, initializer='ones', name='gamma'
            )

        if self.conditional:

            if self.hidden_units is not None:
                self.hidden_dense = Dense(
                    units=self.hidden_units,
                    activation=self.hidden_activation,
                    use_bias=False,
                    kernel_initializer=self.hidden_initializer
                )

            if self.center:
                self.beta_dense = Dense(
                    units=shape[0], use_bias=False, kernel_initializer='zeros'
                )
            if self.scale:
                self.gamma_dense = Dense(
                    units=shape[0], use_bias=False, kernel_initializer='zeros'
                )

    def call(self, inputs):
        """如果是条件Layer Norm，则默认以list为输入，第二个是condition
        """
        if self.conditional:
            inputs, cond = inputs
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(K.ndim(inputs) - K.ndim(cond)):
                cond = K.expand_dims(cond, 1)
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = K.mean(outputs, axis=-1, keepdims=True)
            outputs = outputs - mean
        if self.scale:
            variance = K.mean(K.square(outputs), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs

    def compute_output_shape(self, input_shape):
        if self.conditional:
            return input_shape[0]
        else:
            return input_shape

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'conditional': self.conditional,
            'hidden_units': self.hidden_units,
            'hidden_activation': activations.serialize(self.hidden_activation),
            'hidden_initializer':
                initializers.serialize(self.hidden_initializer),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PositionEmbedding(Layer):
    """定义位置Embedding，这里的Embedding是可训练的。
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        merge_mode='add',
        embeddings_initializer='zeros',
        custom_position_ids=False,
        **kwargs
    ):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.custom_position_ids = custom_position_ids

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer
        )

    def call(self, inputs):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            inputs, position_ids = inputs
            if K.dtype(position_ids) != 'int32':
                position_ids = K.cast(position_ids, 'int32')
            pos_embeddings = K.gather(self.embeddings, position_ids)
        else:
            input_shape = K.shape(inputs)
            batch_size, seq_len = input_shape[0], input_shape[1]
            pos_embeddings = self.embeddings[:seq_len]
            pos_embeddings = K.expand_dims(pos_embeddings, 0)
            if self.merge_mode != 'add':
                pos_embeddings = K.tile(pos_embeddings, [batch_size, 1, 1])

        if self.merge_mode == 'add':
            return inputs + pos_embeddings
        else:
            return K.concatenate([inputs, pos_embeddings])

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode == 'add':
            return input_shape
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'embeddings_initializer':
                initializers.serialize(self.embeddings_initializer),
            'custom_position_ids': self.custom_position_ids,
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class FeedForward(Layer):
    """FeedForward层，其实就是两个Dense层的叠加
    """
    def __init__(
        self,
        units,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
        super(FeedForward, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)

    @integerize_shape
    def build(self, input_shape):
        super(FeedForward, self).build(input_shape)
        output_dim = input_shape[-1]

        self.dense_1 = Dense(
            units=self.units,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.dense_2 = Dense(
            units=output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    def call(self, inputs):
        x = inputs
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

