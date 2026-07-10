from tensorflow.keras.models import Sequential, Model # type: ignore
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPooling2D, DepthwiseConv2D, Reshape, concatenate, Input, BatchNormalization, Activation, Conv2D, InputLayer, UpSampling1D, AveragePooling2D, MaxPooling2D, LayerNormalization, Conv1D, SpatialDropout2D, Conv3D, SpatialDropout3D, Lambda, GlobalAveragePooling2D, MultiHeadAttention # type: ignore
from tensorflow.keras.regularizers import l1 # type: ignore
from tensorflow.keras import backend as K # type: ignore
import tensorflow as tf


def depthwise_model_ndms(input_shape, output_shape,
                         depth_mul_in=(4, 4), filters_out=(4, 4),
                         krnl_in=((1, 3), (1, 3)), krnl_out=(3, 3),
                         pad='valid', strides=((1, 1), (1, 1)),
                         dil=((1, 1), (1, 1)),
                         mpool=((0, 0), (0, 0)), dense=(100, 50), acts=('relu', 'relu', 'relu'),
                         feature_conv=None,
                         b_norm=False, l_norm=False,
                         dense_drp=False, conv_drp=False, drp=0.3):
    model = Sequential(name='Depthwise_model')
    if len(input_shape) < 3:
        model.add(Reshape((1, *input_shape), input_shape=input_shape))
    else:
        model.add(InputLayer(input_shape=input_shape))

    # model.add(AveragePooling2D(pool_size=(1,2), strides=(1, 2)))

    e_layers = encode_layers(depth_mul_in=depth_mul_in, krnl_in=krnl_in, pad=pad,
                             strides=strides, dil=dil, mpool=mpool, acts=acts, l_norm=l_norm, conv_drp=conv_drp,
                             drp=drp)

    model = add_seq_layers(model, e_layers)

    if feature_conv is not None:
        model.add(Conv2D(kernel_size=(1, 1), filters=feature_conv, activation=acts[0], padding='same', ))
        model.add(SpatialDropout2D(drp / 4))

    model.add(Flatten())

    d_layers = decode_layers(dense=dense, acts=acts, b_norm=b_norm, dense_drp=dense_drp, drp=drp)
    model = add_seq_layers(model, d_layers)

    model.add(Reshape((15, -1)))

    # TODO: Could try transposed conv layers too!
    for i in range(len(krnl_out)):
        model.add(UpSampling1D(size=2))
        model.add(Conv1D(filters=filters_out[i], kernel_size=krnl_out[i], padding='same', activation=acts[2]))
        if l_norm:
            model.add(LayerNormalization(axis=-1))

    model.add(Conv1D(kernel_size=1, filters=2, padding='same', activation='linear'))

    model.compile(loss=MeanSquaredError(), optimizer=Adam())  # metrics=[coeff_determination]
    return model


def depthwise_model_ndms_encode(input_shape, output_shape,
                                depth_mul_in=(4, 4),
                                krnl_in=((1, 3), (1, 3)),
                                pad='valid', strides=((1, 1), (1, 1)),
                                dil=((1, 1), (1, 1)),
                                mpool=((0, 0), (0, 0)), dense=(100, 50), acts=('relu', 'relu', 'relu'),
                                feature_conv=None,
                                b_norm=False, l_norm=False,
                                dense_drp=False, conv_drp=False, drp=0.3):
    model = Sequential(name='Depthwise_model_encode')
    if len(input_shape) < 3:
        model.add(Reshape((1, *input_shape), input_shape=input_shape))
    else:
        model.add(InputLayer(input_shape=input_shape))

        # model.add(AveragePooling2D(pool_size=(1,2), strides=(1, 2)))
    e_layers = encode_layers(depth_mul_in=depth_mul_in, krnl_in=krnl_in, pad=pad,
                             strides=strides, dil=dil, mpool=mpool, acts=acts, l_norm=l_norm, conv_drp=conv_drp,
                             drp=drp)

    model = add_seq_layers(model, e_layers)

    if feature_conv is not None:
        model.add(Conv2D(kernel_size=(1, 1), filters=feature_conv, activation=acts[0], padding='same', ))

    # model.add(SpatialDropout2D(drp / 2))
    # model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dropout(drp/2))

    d_layers = decode_layers(dense=dense, acts=acts, b_norm=b_norm, dense_drp=dense_drp, drp=drp)
    model = add_seq_layers(model, d_layers)

    model.add(Dense(np.product(output_shape), activation='linear'))
    model.add(Reshape(output_shape))

    model.compile(loss=MeanSquaredError(), optimizer=Adam())  # metrics=[coeff_determination]
    return model


def depthwise_encode_curve(input_shape, output_shape,
                            depth_mul_in=(4, 4),
                            krnl_in=((1, 3), (1, 3)),
                            pad='valid', strides=((1, 1), (1, 1)),
                            dil=((1, 1), (1, 1)),
                            mpool=((0, 0), (0, 0)), dense=(100, 50), acts=('relu', 'relu', 'relu'),
                            feature_conv=None,
                            b_norm=False, l_norm=False,
                            dense_drp=False, conv_drp=False, drp=0.3,
                            eig=-2., time_samples=None, train_eig=False):

    model = Sequential(name='Depthwise_model_encode_curve')
    if len(input_shape) < 3:
        model.add(Reshape((1, *input_shape), input_shape=input_shape))
    else:
        model.add(InputLayer(input_shape=input_shape))

        # model.add(AveragePooling2D(pool_size=(1,2), strides=(1, 2)))
    e_layers = encode_layers(depth_mul_in=depth_mul_in, krnl_in=krnl_in, pad=pad,
                             strides=strides, dil=dil, mpool=mpool, acts=acts, l_norm=l_norm, conv_drp=conv_drp,
                             drp=drp)

    model = add_seq_layers(model, e_layers)

    if feature_conv is not None:
        model.add(Conv2D(kernel_size=(1, 1), filters=feature_conv, activation=acts[0], padding='same', ))

    model.add(SpatialDropout2D(drp / 2))
    # model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    # model.add(Dropout(drp/2))

    d_layers = decode_layers(dense=dense, acts=acts, b_norm=b_norm, dense_drp=dense_drp, drp=drp)
    model = add_seq_layers(model, d_layers)

    model.add(Dense(np.product(output_shape), activation='linear'))
    model.add(Reshape(output_shape))
    # TODO: Custom params based on shape
    if time_samples is None:
        time_samples = np.linspace(0, 1, output_shape[0]+1)[1:]
    if train_eig:
        model.add(AdaptTrajEstimator(eig=eig, time_samples=time_samples))
    else:
        model.add(TrajEstimator(eig=eig, time_samples=time_samples))

    model.compile(loss=MeanSquaredError(), optimizer=Adam())  # metrics=[coeff_determination]
    return model


def depthwise_encode_curve_2(input_shape, output_shape,
                            depth_mul_in=(4, 4),
                            krnl_in=((1, 3), (1, 3)),
                            pad='valid', strides=((1, 1), (1, 1)),
                            dil=((1, 1), (1, 1)),
                            mpool=((0, 0), (0, 0)), dense=(100, 50), acts=('relu', 'relu', 'relu'),
                            feature_conv=None,
                            b_norm=False, l_norm=False,
                            dense_drp=False, conv_drp=False, drp=0.3,
                            eig=-2., time_samples=None, train_eig=False):

    model = Sequential(name='Depthwise_model_encode_curve')
    if len(input_shape) < 3:
        model.add(Reshape((1, *input_shape), input_shape=input_shape))
    else:
        model.add(InputLayer(input_shape=input_shape))

        # model.add(AveragePooling2D(pool_size=(1,2), strides=(1, 2)))
    e_layers = encode_layers(depth_mul_in=depth_mul_in, krnl_in=krnl_in, pad=pad,
                             strides=strides, dil=dil, mpool=mpool, acts=acts, l_norm=l_norm, conv_drp=conv_drp,
                             drp=drp)

    model = add_seq_layers(model, e_layers)

    if feature_conv is not None:
        model.add(Conv2D(kernel_size=(1, 1), filters=feature_conv, activation=acts[0], padding='same', ))

    model.add(SpatialDropout2D(drp / 2))
    # model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    # model.add(Dropout(drp/2))

    d_layers = decode_layers(dense=dense, acts=acts, b_norm=b_norm, dense_drp=dense_drp, drp=drp)
    model = add_seq_layers(model, d_layers)
    model.add(Dense(6, activation='linear'))
    model.add(Reshape(target_shape=(3, 2)))
    if time_samples is None:
        time_samples = np.linspace(0, 1, output_shape[0]+1)[1:]
    if train_eig:
        model.add(AdaptTrajEstimator(eig=eig, time_samples=time_samples))
    else:
        model.add(TrajEstimator(eig=eig, time_samples=time_samples))

    model.compile(loss=MeanSquaredError(), optimizer=Adam())  # metrics=[coeff_determination]
    return model


def depthwise_curve_params(input_shape, output_shape,
                            depth_mul_in=(4, 4),
                            krnl_in=((1, 3), (1, 3)),
                            pad='valid', strides=((1, 1), (1, 1)),
                            dil=((1, 1), (1, 1)),
                            mpool=((0, 0), (0, 0)), dense=(100, 50), acts=('relu', 'relu', 'relu'),
                            feature_conv=None,
                            b_norm=False, l_norm=False,
                            dense_drp=False, conv_drp=False, drp=0.3,
                            eig=-2., time_samples=None, train_eig=False):

    model = Sequential(name='Depthwise_model_encode_curve')
    if len(input_shape) < 3:
        model.add(Reshape((1, *input_shape), input_shape=input_shape))
    else:
        model.add(InputLayer(input_shape=input_shape))

        # model.add(AveragePooling2D(pool_size=(1,2), strides=(1, 2)))
    e_layers = encode_layers(depth_mul_in=depth_mul_in, krnl_in=krnl_in, pad=pad,
                             strides=strides, dil=dil, mpool=mpool, acts=acts, l_norm=l_norm, conv_drp=conv_drp,
                             drp=drp)

    model = add_seq_layers(model, e_layers)

    if feature_conv is not None:
        model.add(Conv2D(kernel_size=(1, 1), filters=feature_conv, activation=acts[0], padding='same', ))

    model.add(SpatialDropout2D(drp / 2))
    # model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    # model.add(Dropout(drp/2))

    d_layers = decode_layers(dense=dense, acts=acts, b_norm=b_norm, dense_drp=dense_drp, drp=drp)
    model = add_seq_layers(model, d_layers)
    model.add(Dense(6, activation='linear'))

    model.compile(loss=MeanSquaredError(), optimizer=Adam())  # metrics=[coeff_determination]
    return model


def depthwise_transformer_curve(input_shape, output_shape,
                                depth_mul_in=(4, 4),
                                krnl_in=((1, 3), (1, 3)),
                                pad='valid', strides=((1, 1), (1, 1)),
                                dil=((1, 1), (1, 1)),
                                mpool=((0, 0), (0, 0)), dense=(100, 50), acts=('relu', 'relu', 'relu'),
                                feature_conv=None,
                                b_norm=False, l_norm=False,
                                dense_drp=False, conv_drp=False, drp=0.3,
                                eig=-2., time_samples=None, train_eig=False):

    model = Sequential(name='Depthwise_model_encode_curve')
    if len(input_shape) < 3:
        model.add(Reshape((1, *input_shape), input_shape=input_shape))
    else:
        model.add(InputLayer(input_shape=input_shape))

        # model.add(AveragePooling2D(pool_size=(1,2), strides=(1, 2)))
    e_layers = encode_layers(depth_mul_in=depth_mul_in, krnl_in=krnl_in, pad=pad,
                             strides=strides, dil=dil, mpool=mpool, acts=acts, l_norm=l_norm, conv_drp=conv_drp,
                             drp=drp)

    model = add_seq_layers(model, e_layers)

    if feature_conv is not None:
        model.add(Conv1D(kernel_size=3, filters=feature_conv, activation=acts[0], padding='same', ))

    model.add(SpatialDropout2D(drp / 2))
    # model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    # model.add(Dropout(drp/2))

    d_layers = decode_layers(dense=dense, acts=acts, b_norm=b_norm, dense_drp=dense_drp, drp=drp)
    model = add_seq_layers(model, d_layers)

    model.add(Dense(np.product(output_shape), activation='linear'))
    model.add(Reshape(output_shape))
    # TODO: Custom params based on shape
    if time_samples is None:
        time_samples = np.linspace(0, 1, output_shape[0]+1)[1:]
    if train_eig:
        model.add(AdaptTrajEstimator(eig=eig, time_samples=time_samples))
    else:
        model.add(TrajEstimator(eig=eig, time_samples=time_samples))

    model.compile(loss=MeanSquaredError(), optimizer=Adam())  # metrics=[coeff_determination]
    return model

def depthwise_channelwise_mix(input_shape, output_shape,
                              depth_mul_in=(4, 4),
                              krnl_in=((1, 3), (1, 3)),
                              pad='valid', strides=((1, 1), (1, 1)),
                              dil=((1, 1), (1, 1)),
                              mpool=((0, 0), (0, 0)), dense=(100, 50), acts=('relu', 'relu', 'relu'),
                              feature_conv=None,
                              b_norm=False, l_norm=False,
                              dense_drp=False, conv_drp=False, drp=0.3):
    model = Sequential(name='Depthwise_model')
    if len(input_shape) < 3:
        model.add(Reshape((1, *input_shape), input_shape=input_shape))
    else:
        model.add(InputLayer(input_shape=input_shape))

        # model.add(AveragePooling2D(pool_size=(1,2), strides=(1, 2)))
    e_layers = encode_layers(depth_mul_in=depth_mul_in, krnl_in=krnl_in, pad=pad,
                             strides=strides, dil=dil, mpool=mpool, acts=acts, l_norm=l_norm, conv_drp=conv_drp,
                             drp=drp)

    model = add_seq_layers(model, e_layers)

    if feature_conv is not None:
        channel_features = np.product(depth_mul_in)
        model.add(Reshape([*model.layers[-1].output_shape[1:], 1]))
        model.add(Conv3D(kernel_size=(1, 1, channel_features), filters=feature_conv, activation=acts[0],
                         padding='valid', strides=(1, 1, channel_features)))
        model.add(SpatialDropout3D(drp / 4))
    else:
        model.add(SpatialDropout2D(drp / 2))

    model.add(Flatten())

    d_layers = decode_layers(dense=dense, acts=acts, b_norm=b_norm, dense_drp=dense_drp, drp=drp)
    model = add_seq_layers(model, d_layers)

    model.add(Dense(np.product(output_shape), activation='linear'))
    model.add(Reshape(output_shape))

    model.compile(loss=MeanSquaredError(), optimizer=Adam())  # metrics=[coeff_determination]
    return model


def functional_depthwise_mixer_model(input_shape, output_shape,
                                     depth_mul_in=(4, 4),
                                     krnl_in=((1, 3), (1, 3)),
                                     pad='valid', strides=((1, 1), (1, 1)),
                                     dil=((1, 1), (1, 1)),
                                     mpool=((0, 0), (0, 0)), dense=(100, 50), acts=('relu', 'relu', 'relu'),
                                     feature_conv=None,
                                     b_norm=False, l_norm=False,
                                     dense_drp=False, conv_drp=False, drp=0.3):
    if len(input_shape) < 3:
        inputs = Input(shape=input_shape)
        reshaped = Reshape((1, *input_shape))(inputs)
    else:
        inputs = Input(input=input_shape)
        reshaped = inputs

        # model.add(AveragePooling2D(pool_size=(1,2), strides=(1, 2)))
    e_layers = encode_layers(depth_mul_in=depth_mul_in, krnl_in=krnl_in, pad=pad,
                             strides=strides, dil=dil, mpool=mpool, acts=acts, l_norm=l_norm, conv_drp=conv_drp,
                             drp=drp)

    encoded = add_functional_layers(reshaped, e_layers)

    if feature_conv is not None:
        fc_layers = channelwise_feature_conv_layers(n_channels=input_shape[-1],
                                                    n_channel_features=np.prod(depth_mul_in),
                                                    acts=acts, feature_conv=feature_conv, spatial_drp=None)
        split_layers = split_to_branches(encoded, fc_layers)

        mixed = concat_branches(split_layers)
        mixed = SpatialDropout2D(drp / 2)(mixed)
    else:
        mixed = SpatialDropout2D(drp / 2)(encoded)

    decoded = Flatten()(mixed)

    d_layers = decode_layers(dense=dense, acts=acts, b_norm=b_norm, dense_drp=dense_drp, drp=drp)
    decoded = add_functional_layers(decoded, d_layers)

    outputs = Dense(np.product(output_shape), activation='linear')(decoded)
    outputs = Reshape(output_shape)(outputs)

    model = Model(inputs=inputs, outputs=outputs, name='Functional_Mixer')
    model.compile(loss=MeanSquaredError(), optimizer=Adam())
    return model

    pass


def add_seq_layers(model, layers):
    for i in range(len(layers)):
        model.add(layers[i])
    return model


def add_functional_layers(cur_layer, layers):
    for i in range(len(layers)):
        cur_layer = layers[i](cur_layer)
    return cur_layer


def encode_layers(depth_mul_in, krnl_in, pad, strides, dil, mpool, acts, l_norm, conv_drp, drp):
    layers = []
    for i in range(len(krnl_in)):
        layers.append(DepthwiseConv2D(kernel_size=krnl_in[i],
                                      depth_multiplier=depth_mul_in[i],
                                      activation=acts[0],
                                      padding=pad,
                                      strides=strides[i],
                                      dilation_rate=dil[i]))
        if mpool[i][0]:
            layers.append(MaxPooling2D(pool_size=mpool[i]))
        if l_norm:
            layers.append(LayerNormalization(axis=-1))
        if conv_drp:
            layers.append(SpatialDropout2D(drp / 4))
    return layers


def decode_layers(dense, acts, b_norm, dense_drp, drp):
    layers = []
    if b_norm:
        for i, d in enumerate(dense):
            layers.append(Dense(d))
            layers.append(Activation(acts[1]))
            # TODO: layers.append(BatchNormalization())
            layers.append(LayerNormalization())

            if dense_drp and i != len(dense) - 1:
                layers.append(Dropout(drp))
    else:
        for i, d in enumerate(dense):
            layers.append(Dense(d, activation=acts[1]))
            if dense_drp and i != len(dense) - 1:
                layers.append(Dropout(drp))
    return layers


def split_to_branches(cur_layer, branches):
    split_layers = []
    for i in range(len(branches)):
        split_layers.append(add_functional_layers(cur_layer, branches[i]))
    return split_layers


def concat_branches(branches, axis=-1):
    return concatenate(branches, axis=axis)
    pass


def channelwise_feature_conv_layers(n_channels, n_channel_features, feature_conv, acts, spatial_drp=None):
    branches =[]
    for i in range(n_channels):
        branch = []
        sl = Lambda(lambda x: x[:, :, :, i * n_channel_features:(i+1)*n_channel_features],
                    name=f'Split_{i * n_channel_features}_to_{(i+1)*n_channel_features}')
        branch.append(sl)
        cn = Conv2D(kernel_size=(1, 1), filters=feature_conv, activation=acts[0], padding='same')
        branch.append(cn)
        if spatial_drp is not None:
            branch.append(SpatialDropout2D(spatial_drp))
        branches.append(branch)
    return branches


def traj_estimator(tensor, eig=-2, t=(1/3,2/3,1)):
    vd = tensor[:, 0, :] # batch, var, dim
    c1 = tensor[:, 1, :]
    a0 = tensor[:, 2, :]
    c2 = a0 - eig*c1
    c3 = c2 - eig*c1

    t = tf.convert_to_tensor([1/3, 2/3, 1])

    c3 = c2 + 2*c1


class TrajEstimator(tf.keras.layers.Layer):
    def __init__(self, eig=-2., time_samples=(1/3, 2/3, 1), **kwargs):
        super(TrajEstimator, self).__init__(**kwargs)
        self.time_samples = time_samples
        self.T = tf.constant(time_samples, dtype=tf.float32)[None, None, ...]
        self.eig = eig
        self.eigsq = self.eig*self.eig
        self.exp_precalc = tf.exp(self.eig*self.T)

    def call(self, inputs, **kwargs):
        # inputs: (batch, var, dim)
        vd = inputs[:, 0, :, None]  # (batch, dim, time)
        c1 = inputs[:, 1, :, None]
        a0 = inputs[:, 2, :, None]
        
        # Apply bounds to prevent division by zero
        bounded_eig = tf.clip_by_value(self.eig, -4.0, -0.5)
        c2 = a0 - bounded_eig * c1
        c3 = c2 - bounded_eig * c1

        # Use bounded eigenvalue for exponential and squared terms too
        bounded_eigsq = bounded_eig * bounded_eig
        bounded_exp = tf.exp(bounded_eig * self.T)

        # output: (batch, t)
        integrated = vd * self.T + \
                     (c3 + (bounded_eig*c1 + bounded_eig*c2*self.T - c2)*bounded_exp)/bounded_eigsq
        return tf.transpose(integrated, perm=[0, 2, 1])

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(TrajEstimator, self).get_config()
        config = {'eig': self.eig,
                  'time_samples': self.time_samples}
        base_config.update(config)
        return base_config

    def test_from_params(self, vd, v0, a0):
        vd, v0, a0 = np.array(vd, dtype=np.float32), np.array(v0, dtype=np.float32), np.array(a0, dtype=np.float32)
        c1 = v0 - vd
        input = np.vstack((vd, c1, a0))[None, ...]
        input = tf.constant(input)
        return self(input)


class AdaptTrajEstimator(TrajEstimator):
    def __init__(self, **kwargs):
        super(AdaptTrajEstimator, self).__init__(**kwargs)
        self.eig_in = self.eig

    @property
    def eigsq(self):
        # Clamp eigenvalue to prevent division by zero and numerical instability
        bounded_eig = tf.clip_by_value(self.eig, -6.0, -0.1)  # Keep negative for stability
        return bounded_eig * bounded_eig

    @property
    def exp_precalc(self):
        # Use bounded eigenvalue for exponential
        bounded_eig = tf.clip_by_value(self.eig, -6.0, -0.1)
        return tf.exp(bounded_eig*self.T)

    def build(self, input_shape):
        self.eig = self.add_weight(name='shaping_eig', shape=[1], trainable=self.trainable,
                                   initializer=tf.keras.initializers.constant(self.eig))

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(TrajEstimator, self).get_config()
        config = {'eig': self.eig_in,
                  'time_samples': self.time_samples}
        base_config.update(config)
        return base_config

    # Hacky solution so we can reuse the parent class' call function, while still having dynamic values for these params
    @exp_precalc.setter
    def exp_precalc(self, value):
        pass
    @eigsq.setter
    def eigsq(self, value):
        pass


custom_classes = {'TrajEstimator': TrajEstimator, 'AdaptTrajEstimator': AdaptTrajEstimator}


def main():
    Est = TrajEstimator(eig=-4, time_samples=np.linspace(0,1,60))
    res = Est.test_from_params([1, 0], [0, 1], [20, 0]).numpy()[0, ...]
    import matplotlib.pyplot as plt

    plt.plot(res[0, :], res[1, :], marker='x')
    plt.plot(res[0, 0], res[1, 0], c='green', marker='o')
    plt.plot(res[0, -1], res[1, -1], c='red', marker='o')
    plt.axis('equal')
    plt.show()
    pass


if __name__ == '__main__':
    main()