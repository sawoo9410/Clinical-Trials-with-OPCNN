"""
model.py - Dual-Modal Neural Network architectures for clinical trial outcome prediction.

Implements multiple fusion strategies for combining two input modalities:
  - Descriptor (des): molecular descriptor features (e.g., MolecularWeight, LogP, etc.)
  - Body: tissue/organ expression features (e.g., Blood, Liver, Kidney, etc.)

Fusion strategies (from the paper):
  - OPCNN: Outer Product CNN with residual Conv2D blocks (proposed method)
  - Early Fusion: Merge modalities before the main network (Add, Product, Concat, TFL, MCF)
  - Intermediate Fusion: Merge after independent sub-networks (Add, Product, Concat, TFL, MCF)
  - Late Fusion: Each modality produces its own output, then combined (Concat)

Reference:
  Frontiers in Pharmacology (2021)
  https://www.frontiersin.org/articles/10.3389/fphar.2021.670670/full
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Input, add, dot, multiply
from tensorflow.keras.layers import concatenate, Reshape, Conv2D
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


def circulant(shape):
    """
    Generate a circulant matrix from a 1D vector (used in MCF fusion).

    A circulant matrix is a square matrix where each row is a cyclic shift
    of the previous row. This enables compact bilinear pooling between modalities.

    Args:
        shape: 2D tensor of shape (batch_size, d), where d is the feature dimension.

    Returns:
        3D tensor of shape (batch_size, d*d, 1) representing the circulant matrix.
    """
    matrix_shape = shape.get_shape().as_list()
    circulant_set = []
    circulant_ = K.expand_dims(shape, 2)
    circulant_set.append(circulant_)

    for i in range(1, matrix_shape[1]):
        # Cyclic shift: move last i elements to the front
        pre=circulant_[:,(matrix_shape[1]-i):,:]
        host=circulant_[:,0:(matrix_shape[1]-i),:]
        circulant_1=tf.concat([pre,host],1)
        circulant_set.append(circulant_1)

    vector_ = K.concatenate(circulant_set)

    return vector_


class Circulant_layer(Layer):
    """
    Custom Keras layer implementing Multimodal Compact Fusion (MCF).

    Projects both modalities into a shared space via a learned kernel,
    then computes circulant-based bilinear interaction between them.
    This is more parameter-efficient than full bilinear pooling.

    Args:
        output_dim: Dimension of the shared projection space.
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Circulant_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Shared projection kernel for both modalities
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Circulant_layer, self).build(input_shape)

    def call(self, x):
        """
        Args:
            x: List of two tensors [des, body], each of shape (batch_size, features).

        Returns:
            Fused tensor from circulant-based bilinear interaction.
        """
        assert isinstance(x, list)
        des, body = x

        # Project both modalities into shared space
        des = K.dot(des, self.kernel)
        body = K.dot(body, self.kernel)

        # Circulant bilinear interaction (symmetric)
        multiple_vector1 = K.batch_dot(circulant(body), des)
        multiple_vector2 = K.batch_dot(circulant(des), body)

        sum_vector = multiple_vector1 + multiple_vector2
        sum_vector_1 = K.dot(sum_vector, self.kernel)

        return sum_vector


class network:
    """
    Collection of dual-modal neural network architectures.

    All models take two inputs (descriptor and body modalities) and output
    a binary prediction (sigmoid) for clinical trial success/failure.

    Args:
        des_shape: Number of descriptor features (e.g., 13).
        body_shape: Number of body/tissue features (e.g., 30).
        hid_layer: List of 3 hidden layer sizes, e.g. [100, 100, 50].
        output_layer: Output dimension (1 for binary classification).
    """
    def __init__(self, des_shape, body_shape, hid_layer, output_layer):
        self.des_shape = des_shape
        self.body_shape = body_shape
        self.hid_layer = hid_layer
        self.output_layer = output_layer

    # =========================================================================
    # OPCNN (Proposed Method)
    # =========================================================================

    def OPCNN(self):
        """
        Outer Product CNN - the main proposed architecture.

        Architecture:
          1. Each modality is projected to hid_layer[2] dims via Dense
          2. Reshaped to (1, d) and combined via outer product (dot with axes=1)
          3. Outer product matrix (d x d x 1) is fed into 3 residual Conv2D blocks
             - Each block: Conv -> Conv -> Conv, with skip connection (input + output)
          4. Flattened and passed through 2 Dense+BN layers
          5. Sigmoid output for binary classification

        Returns:
            Keras Model with inputs=[des_input, body_input], output=sigmoid prediction.
        """
        des_input = Input(shape = (self.des_shape,))
        body_input = Input(shape = (self.body_shape,))

        # Project each modality to shared dimension
        des_fc = Dense(self.hid_layer[2])(des_input)
        body_fc = Dense(self.hid_layer[2])(body_input)

        # Reshape for outer product: (batch, 1, d)
        des_reshape_2 = Reshape((1,self.hid_layer[2]))(des_fc)
        body_reshape_2 = Reshape((1,self.hid_layer[2]))(body_fc)

        # Outer product: (1, d) x (d, 1) -> (d, d) interaction matrix
        dot_layer = dot([des_reshape_2, body_reshape_2], axes = 1, normalize=False)
        dot_reshape = Reshape((self.hid_layer[2], self.hid_layer[2], 1))(dot_layer)

        # Residual Block A
        res_a = Conv2D(32, (3, 3), activation='relu', padding="same")(dot_reshape)
        res_a_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a)
        res_a_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a_1)
        res_ab = res_a + res_a_2  # Skip connection

        # Residual Block B
        res_b = Conv2D(32, (3, 3), activation='relu', padding="same")(res_ab)
        res_b_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_b)
        res_b_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_b_1)
        res_bc = res_b + res_b_2  # Skip connection

        # Residual Block C
        res_c = Conv2D(32, (3, 3), activation='relu', padding="same")(res_bc)
        res_c_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_c)
        res_c_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_c_1)
        res_cc = res_c + res_c_2  # Skip connection

        tensor_flat = Flatten()(res_cc)

        # Classification head
        x = Dense(self.hid_layer[2] , activation = 'relu')(tensor_flat)
        x = BatchNormalization()(x)

        x = Dense(self.hid_layer[2] , activation = 'relu')(x)
        x = BatchNormalization()(x)

        x = Dense(self.output_layer , activation = 'sigmoid')(x)

        OPCNN = keras.models.Model(inputs = [des_input, body_input], outputs = [x])

        return OPCNN

    # =========================================================================
    # Early Fusion Models
    # - Modalities are merged at the input level (after one projection layer),
    #   then the fused representation passes through the shared network.
    # =========================================================================

    def Early_Add(self):
        """
        Early Fusion with element-wise Addition.

        Each modality is projected to hid_layer[2] dims + BN,
        then merged by element-wise addition before the shared Dense layers.

        Returns:
            Keras Model with inputs=[des_input, body_input].
        """
        # des modality
        des_input = Input(shape = self.des_shape)
        des_input_x2 = Dense(self.hid_layer[2], activation = 'relu')(des_input)
        des_input_x2_batch = BatchNormalization()(des_input_x2)

        # body modality
        body_input = Input(shape = self.body_shape)
        body_input_x2 = Dense(self.hid_layer[2], activation = 'relu')(body_input)
        body_input_x2_batch = BatchNormalization()(body_input_x2)

        # Fusion: element-wise addition
        merge_layer = add([des_input_x2_batch , body_input_x2_batch])

        # Shared classification layers
        x = Dense(self.hid_layer[0] , activation = 'relu')(merge_layer)
        x = BatchNormalization()(x)

        x = Dense(self.hid_layer[1] , activation = 'relu')(x)
        x = BatchNormalization()(x)

        x = Dense(self.hid_layer[2] , activation = 'relu')(x)
        x = BatchNormalization()(x)

        x = Dense(self.output_layer , activation = 'sigmoid')(x)

        early_add = keras.models.Model(inputs = [des_input, body_input], outputs = [x])

        return early_add

    def Early_Product(self):
        """
        Early Fusion with element-wise Multiplication (Hadamard product).

        Returns:
            Keras Model with inputs=[des_input, body_input].
        """
        # des modality
        des_input = Input(shape = self.des_shape)
        des_input_x2 = Dense(self.hid_layer[2], activation = 'relu')(des_input)
        des_input_x2_batch = BatchNormalization()(des_input_x2)

        # body modality
        body_input = Input(shape = self.body_shape)
        body_input_x2 = Dense(self.hid_layer[2], activation = 'relu')(body_input)
        body_input_x2_batch = BatchNormalization()(body_input_x2)

        # Fusion: element-wise multiplication
        merge_layer = multiply([des_input_x2_batch , body_input_x2_batch])

        # Shared classification layers
        x = Dense(self.hid_layer[0] , activation = 'relu')(merge_layer)
        x = BatchNormalization()(x)

        x = Dense(self.hid_layer[1] , activation = 'relu')(x)
        x = BatchNormalization()(x)

        x = Dense(self.hid_layer[2] , activation = 'relu')(x)
        x = BatchNormalization()(x)

        x = Dense(self.output_layer , activation = 'sigmoid')(x)

        early_product = keras.models.Model(inputs = [des_input, body_input], outputs = [x])

        return early_product

    def Early_Concat(self):
        """
        Early Fusion with Concatenation.

        Projected modality vectors are concatenated along the feature axis,
        doubling the dimension before the shared layers.

        Returns:
            Keras Model with inputs=[des_input, body_input].
        """
        # des modality
        des_input = Input(shape = self.des_shape)
        des_input_x2 = Dense(self.hid_layer[2], activation = 'relu')(des_input)
        des_input_x2_batch = BatchNormalization()(des_input_x2)

        # body modality
        body_input = Input(shape = self.body_shape)
        body_input_x2 = Dense(self.hid_layer[2], activation = 'relu')(body_input)
        body_input_x2_batch = BatchNormalization()(body_input_x2)

        # Fusion: concatenation
        merge_layer = concatenate([des_input_x2_batch , body_input_x2_batch])

        # Shared classification layers
        x = Dense(self.hid_layer[0] , activation = 'relu')(merge_layer)
        x = BatchNormalization()(x)

        x = Dense(self.hid_layer[1] , activation = 'relu')(x)
        x = BatchNormalization()(x)

        x = Dense(self.hid_layer[2] , activation = 'relu')(x)
        x = BatchNormalization()(x)

        x = Dense(self.output_layer , activation = 'sigmoid')(x)

        early_concat = keras.models.Model(inputs = [des_input, body_input], outputs = [x])

        return early_concat

    def Early_TFL(self):
        """
        Early Fusion with Tensor Fusion Layer (TFL).

        Appends a bias term to each projected modality, then computes
        the outer product to capture all pairwise (including bias) interactions.
        Requires an additional bias_input of ones.

        Returns:
            Keras Model with inputs=[bias_input, des_input, body_input].
        """
        des_input = Input(shape = (self.des_shape,))
        body_input = Input(shape = (self.body_shape,))
        bias_input = Input(shape = (1,))  # Constant 1s for bias interaction

        # Project to shared dimension
        des_reshape = Dense(self.hid_layer[2])(des_input)
        body_reshape = Dense(self.hid_layer[2])(body_input)

        # Append bias to each modality: (d) -> (d+1)
        concat_1 = concatenate([des_reshape , bias_input])
        concat_2 = concatenate([body_reshape , bias_input])

        # Reshape for outer product: (d+1) -> (1, d+1)
        des_reshape_2 = Reshape((1,51))(concat_1)
        body_reshape_2 = Reshape((1,51))(concat_2)

        # Outer product: captures all pairwise feature interactions
        dot_layer = dot([des_reshape_2, body_reshape_2], axes = 1, normalize=False)
        tensor_flat = Flatten()(dot_layer)

        # Shared classification layers
        x = Dense(self.hid_layer[0] , activation = 'relu')(tensor_flat)
        x = BatchNormalization()(x)

        x = Dense(self.hid_layer[1] , activation = 'relu')(x)
        x = BatchNormalization()(x)

        x = Dense(self.hid_layer[2] , activation = 'relu')(x)
        x = BatchNormalization()(x)

        x = Dense(self.output_layer , activation = 'sigmoid')(x)

        early_TFL = keras.models.Model(inputs = [bias_input, des_input, body_input], outputs = [x])

        return early_TFL

    def Early_MCF(self):
        """
        Early Fusion with Multimodal Compact Fusion (MCF).

        Uses circulant-based bilinear pooling (Circulant_layer) for
        parameter-efficient cross-modal interaction. More compact than TFL.

        Returns:
            Keras Model with inputs=[des_input, body_input].
        """
        des_input = Input(shape = (self.des_shape,))
        body_input = Input(shape = (self.body_shape,))

        # Project to shared dimension
        des_reshape = Dense(self.hid_layer[2],)(des_input)
        body_reshape = Dense(self.hid_layer[2],)(body_input)

        # Circulant-based compact bilinear fusion
        fusion = Circulant_layer(self.hid_layer[2])([des_reshape, body_reshape])

        tensor_flat = Flatten()(fusion)

        # Shared classification layers
        x = Dense(self.hid_layer[0] , activation = 'relu')(tensor_flat)
        x = BatchNormalization()(x)

        x = Dense(self.hid_layer[1] , activation = 'relu')(x)
        x = BatchNormalization()(x)

        x = Dense(self.hid_layer[2] , activation = 'relu')(x)
        x = BatchNormalization()(x)

        x = Dense(self.output_layer , activation = 'sigmoid')(x)

        early_MCF = keras.models.Model(inputs = [des_input, body_input], outputs = [x])

        return early_MCF

    # =========================================================================
    # Intermediate Fusion Models
    # - Each modality passes through its own deep sub-network first,
    #   then the learned representations are merged before the final output.
    # =========================================================================

    def Intermediate_Add(self):
        """
        Intermediate Fusion with element-wise Addition.

        Each modality passes through 3 Dense+BN layers independently,
        then merged by addition before the output layer.

        Returns:
            Keras Model with inputs=[des_input, body_input].
        """
        # des modality sub-network
        des_input = Input(shape = self.des_shape)
        des_input_x2 = Dense(self.hid_layer[0], activation = 'relu')(des_input)
        des_input_x2_batch = BatchNormalization()(des_input_x2)
        des_input_x3 = Dense(self.hid_layer[1], activation = 'relu')(des_input_x2_batch)
        des_input_x3_batch = BatchNormalization()(des_input_x3)
        des_input_x4 = Dense(self.hid_layer[2], activation = 'relu')(des_input_x3_batch)
        des_input_x4_batch = BatchNormalization()(des_input_x4)

        # body modality sub-network
        body_input = Input(shape = self.body_shape)
        body_input_x2 = Dense(self.hid_layer[0], activation = 'relu')(body_input)
        body_input_x2_batch = BatchNormalization()(body_input_x2)
        body_input_x3 = Dense(self.hid_layer[1], activation = 'relu')(body_input_x2_batch)
        body_input_x3_batch = BatchNormalization()(body_input_x3)
        body_input_x4 = Dense(self.hid_layer[2], activation = 'relu')(body_input_x3_batch)
        body_input_x4_batch = BatchNormalization()(body_input_x4)

        # Fusion: element-wise addition of deep representations
        addition_layer = keras.layers.add([des_input_x4_batch, body_input_x4_batch])
        addition_input_x = Dense(self.output_layer, activation = 'sigmoid')(addition_layer)
        inter_add = keras.models.Model(inputs = [des_input, body_input], outputs = [addition_input_x])

        return inter_add

    def Intermediate_Product(self):
        """
        Intermediate Fusion with element-wise Multiplication.

        Returns:
            Keras Model with inputs=[des_input, body_input].
        """
        # des modality sub-network
        des_input = Input(shape = self.des_shape)
        des_input_x2 = Dense(self.hid_layer[0], activation = 'relu')(des_input)
        des_input_x2_batch = BatchNormalization()(des_input_x2)
        des_input_x3 = Dense(self.hid_layer[1], activation = 'relu')(des_input_x2_batch)
        des_input_x3_batch = BatchNormalization()(des_input_x3)
        des_input_x4 = Dense(self.hid_layer[2], activation = 'relu')(des_input_x3_batch)
        des_input_x4_batch = BatchNormalization()(des_input_x4)

        # body modality sub-network
        body_input = Input(shape = self.body_shape)
        body_input_x2 = Dense(self.hid_layer[0], activation = 'relu')(body_input)
        body_input_x2_batch = BatchNormalization()(body_input_x2)
        body_input_x3 = Dense(self.hid_layer[1], activation = 'relu')(body_input_x2_batch)
        body_input_x3_batch = BatchNormalization()(body_input_x3)
        body_input_x4 = Dense(self.hid_layer[2], activation = 'relu')(body_input_x3_batch)
        body_input_x4_batch = BatchNormalization()(body_input_x4)

        # Fusion: element-wise multiplication of deep representations
        product_layer = multiply([des_input_x4_batch, body_input_x4_batch])
        product_input_x = Dense(self.output_layer, activation = 'sigmoid')(product_layer)
        inter_product = keras.models.Model(inputs = [des_input, body_input], outputs = [product_input_x])

        return inter_product

    def Intermediate_Concat(self):
        """
        Intermediate Fusion with Concatenation.

        Returns:
            Keras Model with inputs=[des_input, body_input].
        """
        # des modality sub-network
        des_input = Input(shape = self.des_shape)
        des_input_x2 = Dense(self.hid_layer[0], activation = 'relu')(des_input)
        des_input_x2_batch = BatchNormalization()(des_input_x2)
        des_input_x3 = Dense(self.hid_layer[1], activation = 'relu')(des_input_x2_batch)
        des_input_x3_batch = BatchNormalization()(des_input_x3)
        des_input_x4 = Dense(self.hid_layer[2], activation = 'relu')(des_input_x3_batch)
        des_input_x4_batch = BatchNormalization()(des_input_x4)

        # body modality sub-network
        body_input = Input(shape = self.body_shape)
        body_input_x2 = Dense(self.hid_layer[0], activation = 'relu')(body_input)
        body_input_x2_batch = BatchNormalization()(body_input_x2)
        body_input_x3 = Dense(self.hid_layer[1], activation = 'relu')(body_input_x2_batch)
        body_input_x3_batch = BatchNormalization()(body_input_x3)
        body_input_x4 = Dense(self.hid_layer[2], activation = 'relu')(body_input_x3_batch)
        body_input_x4_batch = BatchNormalization()(body_input_x4)

        # Fusion: concatenation of deep representations
        concat_layer = concatenate([des_input_x4_batch, body_input_x4_batch])
        concat_input_x = Dense(self.output_layer, activation = 'sigmoid')(concat_layer)
        inter_concat = keras.models.Model(inputs = [des_input, body_input], outputs = [concat_input_x])

        return inter_concat

    def Intermediate_TFL(self):
        """
        Intermediate Fusion with Tensor Fusion Layer (TFL).

        Each modality is processed by 2 Dense+BN layers, then a bias term
        is appended before computing the outer product.

        Returns:
            Keras Model with inputs=[bias_input, des_input, body_input].
        """
        # bias modality (constant 1s for bias interaction)
        bias_input = Input(shape = (1,))

        # des modality sub-network
        des_input = Input(shape = self.des_shape)
        des_input_x1 = Dense(self.hid_layer[0], activation = 'relu')(des_input)
        des_input_x1_batch = BatchNormalization()(des_input_x1)
        des_input_x3 = Dense(self.hid_layer[2], activation = 'relu')(des_input_x1_batch)
        des_input_x3_batch = BatchNormalization()(des_input_x3)

        # body modality sub-network
        body_input = Input(shape = self.body_shape)
        body_input_x1 = Dense(self.hid_layer[0], activation = 'relu')(body_input)
        body_input_x1_batch = BatchNormalization()(body_input_x1)
        body_input_x3 = Dense(self.hid_layer[2], activation = 'relu')(body_input_x1_batch)
        body_input_x3_batch = BatchNormalization()(body_input_x3)

        # Append bias and compute outer product
        concat_layer1 = concatenate([bias_input, des_input_x3_batch])
        concat_layer2 = concatenate([bias_input, body_input_x3_batch])

        re_concat1 = Reshape((1, 51))(concat_layer1)
        re_concat2 = Reshape((1, 51))(concat_layer2)

        dot_layer = dot([re_concat1, re_concat2], axes = 1, normalize=False)
        tensor_flat = Flatten()(dot_layer)
        fusion_dense = Dense(self.hid_layer[2], activation = 'relu')(tensor_flat)
        fusion_out = Dense(self.output_layer, activation = 'sigmoid')(fusion_dense)

        inter_TFL = keras.models.Model(inputs = [bias_input, des_input, body_input], outputs = [fusion_out])

        return inter_TFL

    def Intermediate_MCF(self):
        """
        Intermediate Fusion with Multimodal Compact Fusion (MCF).

        Each modality is processed by 2 Dense+BN layers, then fused
        via circulant-based compact bilinear pooling.

        Returns:
            Keras Model with inputs=[des_input, body_input].
        """
        # des modality sub-network
        des_input = Input(shape = self.des_shape)
        des_input_x1 = Dense(self.hid_layer[0], activation = 'relu')(des_input)
        des_input_x1_batch = BatchNormalization()(des_input_x1)
        des_input_x2 = Dense(self.hid_layer[2], activation = 'relu')(des_input_x1_batch)
        des_input_x2_batch = BatchNormalization()(des_input_x2)

        # body modality sub-network
        body_input = Input(shape = self.body_shape)
        body_input_x1 = Dense(self.hid_layer[0], activation = 'relu')(body_input)
        body_input_x1_batch = BatchNormalization()(body_input_x1)
        body_input_x2 = Dense(self.hid_layer[2], activation = 'relu')(body_input_x1_batch)
        body_input_x2_batch = BatchNormalization()(body_input_x2)

        # Circulant-based compact bilinear fusion
        fusion = Circulant_layer(self.hid_layer[2])([des_input_x2_batch, body_input_x2_batch])
        fusion_out = Dense(self.output_layer, activation = 'sigmoid')(fusion)
        inter_MCF = keras.models.Model(inputs = [des_input, body_input], outputs = [fusion_out])

        return inter_MCF

    # =========================================================================
    # Late Fusion Models
    # - Each modality has its own complete sub-network with independent output,
    #   then the individual outputs are combined for the final prediction.
    # =========================================================================

    def Late_Concat(self):
        """
        Late Fusion with Concatenation.

        Each modality passes through a full sub-network (BN + 3 Dense+BN layers)
        and produces its own sigmoid output. The two outputs are then concatenated
        and passed through a final Dense layer for the combined prediction.

        Returns:
            Keras Model with inputs=[des_input, body_input].
        """
        # des modality: full independent sub-network with its own output
        des_input = Input(shape = (self.des_shape,))
        des_input_x1_batch = BatchNormalization()(des_input)
        des_input_x2 = Dense(self.hid_layer[0], activation = 'relu')(des_input_x1_batch)
        des_input_x2_batch = BatchNormalization()(des_input_x2)
        des_input_x3 = Dense(self.hid_layer[1], activation = 'relu')(des_input_x2_batch)
        des_input_x3_batch = BatchNormalization()(des_input_x3)
        des_input_x4 = Dense(self.hid_layer[2], activation = 'relu')(des_input_x3_batch)
        des_input_x4_batch = BatchNormalization()(des_input_x4)
        des_output = Dense(self.output_layer , activation = 'sigmoid')(des_input_x4_batch)

        # body modality: full independent sub-network with its own output
        body_input = Input(shape = (self.body_shape,))
        body_input_x1_batch = BatchNormalization()(body_input)
        body_input_x2 = Dense(self.hid_layer[0], activation = 'relu')(body_input_x1_batch)
        body_input_x2_batch = BatchNormalization()(body_input_x2)
        body_input_x3 = Dense(self.hid_layer[1], activation = 'relu')(body_input_x2_batch)
        body_input_x3_batch = BatchNormalization()(body_input_x3)
        body_input_x4 = Dense(self.hid_layer[2], activation = 'relu')(body_input_x3_batch)
        body_input_x4_batch = BatchNormalization()(body_input_x4)
        body_output = Dense(self.output_layer , activation = 'sigmoid')(body_input_x4_batch)

        # Late fusion: concatenate individual outputs and combine
        concat_layer = concatenate([des_output , body_output])
        f_layer = Dense(self.hid_layer[2], activation = 'relu')(concat_layer)
        concat_input_x = Dense(self.output_layer, activation = 'sigmoid')(f_layer)
        late_concat = keras.models.Model(inputs = [des_input, body_input], outputs = [concat_input_x])

        return late_concat
