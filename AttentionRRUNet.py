import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    add,
    concatenate,
    multiply,
    Activation,
    Input,
    SpatialDropout2D,
    BatchNormalization,
)


def ConvBnElu(inp, filters, kernel_size=3, strides=1, dilation_rate=1):

    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer="he_uniform",
        use_bias=False,
        dilation_rate=dilation_rate,
    )(inp)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    return x


def deconv(inp):
    """Deconv upsampling of x. Doubles x and y dimension and maintains z.
    """
    num_filters = inp.get_shape().as_list()[-1]

    x = Conv2DTranspose(
        filters=num_filters,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=False,
        kernel_initializer="he_uniform",
    )(inp)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)

    return x


def attention_gate(inp_1, inp_2, n_intermediate_filters):
    """Attention gate. Compresses both inputs to n_intermediate_filters filters before processing.
    """
    inp_1_conv = Conv2D(
        n_intermediate_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_uniform",
    )(inp_1)
    inp_2_conv = Conv2D(
        n_intermediate_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_uniform",
    )(inp_2)

    f = Activation("relu")(add([inp_1_conv, inp_2_conv]))
    g = Conv2D(
        filters=1,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_uniform",
    )(f)
    h = Activation("sigmoid")(g)
    return multiply([inp_1, h])


def attention_concat_upsample(across, below):
    """Upsamples below and concatenates with an attention gated version of across. Below needs to be 1/2 the size of across.
    """
    below_filters = below.get_shape().as_list()[-1]
    below_upsampled = deconv(below)
    attention_across = attention_gate(across, below_upsampled, below_filters)
    return concatenate([below_upsampled, attention_across])


def RR_block(inp, out_filters, dropout=0.2):
    """ Reccurent conv block with decreasing kernel size. Makes use of atrous convolutions to make large kernel sizes computationally feasible

    """

    initial = skip = ConvBnElu(inp, out_filters, kernel_size=1)

    c1 = ConvBnElu(initial, out_filters, kernel_size=8, dilation_rate=4)
    c1 = SpatialDropout2D(dropout)(c1)
    c2 = ConvBnElu(add([initial, c1]), out_filters,
                   kernel_size=6, dilation_rate=3)
    c2 = SpatialDropout2D(dropout)(c2)
    c3 = ConvBnElu(c2, out_filters, kernel_size=4, dilation_rate=2)
    c3 = SpatialDropout2D(dropout)(c3)
    c4 = ConvBnElu(add([c2, c3]), out_filters, kernel_size=3, dilation_rate=1)

    return add([skip, c4])


def MoNet(
    input_shape=(256, 256, 1),
    output_classes=1,
    depth=2,
    n_filters_init=16,
    dropout_enc=0.2,
    dropout_dec=0.2,
):

    inputs = x = Input(input_shape)
    skips = []
    features = n_filters_init

    # encoder
    for i in range(depth):
        x = RR_block(x, features, dropout=dropout_enc)
        skips.append(x)
        x = ConvBnElu(x, features, kernel_size=4, strides=2)
        features *= 2

    # bottleneck
    x = RR_block(x, features)

    # decoder
    for i in reversed(range(depth)):
        features //= 2
        x = attention_concat_upsample(across=skips[i], below=x)
        x = RR_block(x, features, dropout=dropout_dec)

    # head
    final_conv = Conv2D(
        output_classes,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_uniform",
        use_bias=False,
    )(x)
    final_bn = BatchNormalization()(final_conv)
    sigm = Activation("sigmoid")(final_bn)
    return Model(inputs, sigm)
