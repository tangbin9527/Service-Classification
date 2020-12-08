<<<<<<< HEAD
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import top_k_categorical_accuracy, categorical_accuracy
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, GRU, Activation, Conv1D, Conv2D, Reshape, Average, Flatten, GlobalMaxPooling1D, Dot
from tensorflow.keras.layers import Embedding, BatchNormalization
from tensorflow.keras.layers import Concatenate, Lambda

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.regularizers import l2

from co_attention_layer import AttentionLayer
from weighted_attention_layer import WeightedLayer

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    # 定义输出到文件的log级别
    level=logging.DEBUG,
    # 定义输出log的格式
    format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
    datefmt='%Y-%m-%d %A %H:%M:%S',                                     # 时间
)


class COBERT:
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.nameLength = self.args.name_length
        self.inforWLen = self.args.informative_words_length
        self.descLength = self.args.description_length

        self.bert_path = self.args.bert_path
        self.embedding_size = self.args.embedding_size

        self.convolutional_filters = self.args.convolutional_filters
        self.l2 = self.args.l2_reg_lambda


    def build(self):
        logger.debug('Building Co-Attentive Representation Model')

        # INPUT - Service Name
        in_name_id = Input(shape=(self.nameLength, ), dtype=tf.int32, name="input_word_name_ids")
        in_name_mask = Input(shape=(self.nameLength, ), dtype=tf.int32, name="input_name_masks")
        in_name_segment = Input(shape=(self.nameLength, ), dtype=tf.int32, name="segment_name_ids")

        bert_name_inputs = [in_name_id, in_name_mask, in_name_segment]
        # BERT for Name
        bert_name_layer = hub.KerasLayer(self.bert_path, trainable=True, name="bert_name")
        _, name_sequence_output = bert_name_layer(bert_name_inputs)
        # CNN
        name_embeddings = Reshape((-1, self.embedding_size, 1))(name_sequence_output)
        name_features1 = Conv2D(self.convolutional_filters, kernel_size=(3, 3), padding='same', activation=None, name="name_conv1", kernel_regularizer=l2(self.l2))(name_embeddings)
        name_features1 = Dropout(0.3, name="methname_conv1_dropout")(name_features1)
        name_features2 = Conv2D(1, kernel_size=(1, 1), padding='same', activation=None, name="name_conv2", kernel_regularizer=l2(self.l2))(name_features1)
        name_features = Reshape((-1, self.embedding_size))(name_features2)
        methname_conv1 = Conv1D(1024, 2, padding='valid', strides=1, activation="tanh", name='methname_conv1', kernel_regularizer=l2(self.l2))(name_features)
        methname_conv1 = Dropout(0.5, name="methname_conv2_dropout")(methname_conv1)

        # INPUT - Service informative words
        in_ew_id = Input(shape=(self.inforWLen, ), dtype=tf.int32, name="input_word_ew_ids")
        in_ew_mask = Input(shape=(self.inforWLen, ), dtype=tf.int32, name="input_ew_masks")
        in_ew_segment = Input(shape=(self.inforWLen, ), dtype=tf.int32, name="segment_ew_ids")
        bert_ew_inputs = [in_ew_id, in_ew_mask, in_ew_segment]
        # BERT for ew
        bert_ew_layer = hub.KerasLayer(self.bert_path, trainable=True, name="bert_ew")
        # bert_ew_layer = hub.load(bert_path)
        ew_pooled_output, ew_sequence_output = bert_ew_layer(bert_ew_inputs)
        ew_embeddings = Reshape((-1, self.embedding_size, 1))(ew_sequence_output)
        ew_features1 = Conv2D(32, kernel_size=(3, 3), padding='same', activation=None, name="ew_conv1", kernel_regularizer=l2(self.l2))(ew_embeddings)
        ew_features1 = Dropout(0.3, name="ew_conv1_dropout")(ew_features1)
        ew_features2 = Conv2D(1, kernel_size=(1, 1), padding='same', activation=None, name="ew_conv2", kernel_regularizer=l2(self.l2))(ew_features1)
        ew_features = Reshape((-1, self.embedding_size))(ew_features2)
        methew_conv1 = Conv1D(1024, 2, padding="valid", strides=1, activation="tanh", name='methew_conv1', kernel_regularizer=l2(self.l2))(ew_features)
        methew_conv1 = Dropout(0.5, name="methew_conv1_dropout")(methew_conv1)

        merged_augumented_data = Concatenate(name='augumented_data_merge', axis=1)([methname_conv1, methew_conv1])

        # INPUT - Service Description
        in_id = Input(shape=(self.descLength, ), dtype=tf.int32, name="input_word_ids")
        in_mask = Input(shape=(self.descLength, ), dtype=tf.int32, name="input_masks")
        in_segment = Input(shape=(self.descLength, ), dtype=tf.int32, name="segment_ids")
        bert_description_inputs = [in_id, in_mask, in_segment]
        # BERT for Description
        bert_layer = hub.KerasLayer(self.bert_path, trainable=True, name="bert_description")
        # bert_layer = hub.load(bert_path)
        _, sequence_output = bert_layer(bert_description_inputs)
        description_features = Bidirectional(GRU(512, return_sequences=True, name="description_feature", kernel_regularizer=l2(self.l2)))(sequence_output)
        description_features = Dropout(0.5, name="desc_dropout")(description_features)


        attention = AttentionLayer(name='attention_layer')
        attention_out = attention([merged_augumented_data, description_features])

        gmp_1 = GlobalMaxPooling1D(name='Globalmaxpool_colum')
        att_1 = gmp_1(attention_out)
        activ1 = Activation('softmax', name='AP_active_colum')
        att_1_next = activ1(att_1)
        dot1 = Dot(axes=1, normalize=False, name='column_dot')
        desc_out = dot1([att_1_next, description_features])

        attention_trans_layer = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)), name='trans_attention')
        attention_transposed = attention_trans_layer(attention_out)
        gmp_2 = GlobalMaxPooling1D(name='globalmaxpool_row')
        att_2 = gmp_2(attention_transposed)
        activ2 = Activation('softmax', name='AP_active_row')
        att_2_next = activ2(att_2)
        dot2 = Dot(axes=1, normalize=False, name='row_dot')
        code_out = dot2([att_2_next, merged_augumented_data])

        all_features = WeightedLayer(name="weighted_layer")(desc_out, code_out)

        output = Dense(50, activation='softmax', kernel_regularizer=l2(self.l2), name="softmax")(all_features)
        self.model = Model(inputs=[bert_description_inputs, bert_name_inputs, bert_ew_inputs], outputs=output)
        print('\nsummary of co-attentive representation model')
        self.model.summary()

    
    def compile(self, optimizer):
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[top_k_categorical_accuracy, categorical_accuracy])

    def fit(self, traindata, steps_per_epoch, **kwargs):
        assert self.model is not None, 'Must compile the model before fitting data'
        return self.model.fit(traindata, steps_per_epoch=steps_per_epoch, epochs=1, verbose = 1, shuffle=False, **kwargs)

    def evaluate(self, testdata, steps, **kwargs):
        assert self.model is not None, 'Must compile the model before evaluatting data'
        return self.model.evaluate(testdata, steps=steps, **kwargs)

    def predict(self, testdata, **kwargs):
        assert self.model is not None, 'Must compile the model before predicting data'
        return self.model.predict(testdata, **kwargs)

    def save(self, model_path, **kwargs):
        assert self.model is not None, 'Must compile the model before saving weights'
        self.model.save(model_path, **kwargs)

    def load(self, model_path, **kwargs):
        assert self.model is not None, 'Must compile the model loading weights'
=======
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import top_k_categorical_accuracy, categorical_accuracy
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, GRU, Activation, Conv1D, Conv2D, Reshape, Average, Flatten, GlobalMaxPooling1D, Dot
from tensorflow.keras.layers import Embedding, BatchNormalization
from tensorflow.keras.layers import Concatenate, Lambda

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.regularizers import l2

from co_attention_layer import AttentionLayer
from weighted_attention_layer import WeightedLayer

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    # 定义输出到文件的log级别
    level=logging.DEBUG,
    # 定义输出log的格式
    format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
    datefmt='%Y-%m-%d %A %H:%M:%S',                                     # 时间
)


class COBERT:
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.nameLength = self.args.name_length
        self.inforWLen = self.args.informative_words_length
        self.descLength = self.args.description_length

        self.bert_path = self.args.bert_path
        self.embedding_size = self.args.embedding_size

        self.convolutional_filters = self.args.convolutional_filters
        self.l2 = self.args.l2_reg_lambda


    def build(self):
        logger.debug('Building Co-Attentive Representation Model')

        # INPUT - Service Name
        in_name_id = Input(shape=(self.nameLength, ), dtype=tf.int32, name="input_word_name_ids")
        in_name_mask = Input(shape=(self.nameLength, ), dtype=tf.int32, name="input_name_masks")
        in_name_segment = Input(shape=(self.nameLength, ), dtype=tf.int32, name="segment_name_ids")

        bert_name_inputs = [in_name_id, in_name_mask, in_name_segment]
        # BERT for Name
        bert_name_layer = hub.KerasLayer(self.bert_path, trainable=True, name="bert_name")
        _, name_sequence_output = bert_name_layer(bert_name_inputs)
        # CNN
        name_embeddings = Reshape((-1, self.embedding_size, 1))(name_sequence_output)
        name_features1 = Conv2D(self.convolutional_filters, kernel_size=(3, 3), padding='same', activation=None, name="name_conv1", kernel_regularizer=l2(self.l2))(name_embeddings)
        name_features1 = Dropout(0.3, name="methname_conv1_dropout")(name_features1)
        name_features2 = Conv2D(1, kernel_size=(1, 1), padding='same', activation=None, name="name_conv2", kernel_regularizer=l2(self.l2))(name_features1)
        name_features = Reshape((-1, self.embedding_size))(name_features2)
        methname_conv1 = Conv1D(1024, 2, padding='valid', strides=1, activation="tanh", name='methname_conv1', kernel_regularizer=l2(self.l2))(name_features)
        methname_conv1 = Dropout(0.5, name="methname_conv2_dropout")(methname_conv1)

        # INPUT - Service informative words
        in_ew_id = Input(shape=(self.inforWLen, ), dtype=tf.int32, name="input_word_ew_ids")
        in_ew_mask = Input(shape=(self.inforWLen, ), dtype=tf.int32, name="input_ew_masks")
        in_ew_segment = Input(shape=(self.inforWLen, ), dtype=tf.int32, name="segment_ew_ids")
        bert_ew_inputs = [in_ew_id, in_ew_mask, in_ew_segment]
        # BERT for ew
        bert_ew_layer = hub.KerasLayer(self.bert_path, trainable=True, name="bert_ew")
        # bert_ew_layer = hub.load(bert_path)
        ew_pooled_output, ew_sequence_output = bert_ew_layer(bert_ew_inputs)
        ew_embeddings = Reshape((-1, self.embedding_size, 1))(ew_sequence_output)
        ew_features1 = Conv2D(32, kernel_size=(3, 3), padding='same', activation=None, name="ew_conv1", kernel_regularizer=l2(self.l2))(ew_embeddings)
        ew_features1 = Dropout(0.3, name="ew_conv1_dropout")(ew_features1)
        ew_features2 = Conv2D(1, kernel_size=(1, 1), padding='same', activation=None, name="ew_conv2", kernel_regularizer=l2(self.l2))(ew_features1)
        ew_features = Reshape((-1, self.embedding_size))(ew_features2)
        methew_conv1 = Conv1D(1024, 2, padding="valid", strides=1, activation="tanh", name='methew_conv1', kernel_regularizer=l2(self.l2))(ew_features)
        methew_conv1 = Dropout(0.5, name="methew_conv1_dropout")(methew_conv1)

        merged_augumented_data = Concatenate(name='augumented_data_merge', axis=1)([methname_conv1, methew_conv1])

        # INPUT - Service Description
        in_id = Input(shape=(self.descLength, ), dtype=tf.int32, name="input_word_ids")
        in_mask = Input(shape=(self.descLength, ), dtype=tf.int32, name="input_masks")
        in_segment = Input(shape=(self.descLength, ), dtype=tf.int32, name="segment_ids")
        bert_description_inputs = [in_id, in_mask, in_segment]
        # BERT for Description
        bert_layer = hub.KerasLayer(self.bert_path, trainable=True, name="bert_description")
        # bert_layer = hub.load(bert_path)
        _, sequence_output = bert_layer(bert_description_inputs)
        description_features = Bidirectional(GRU(512, return_sequences=True, name="description_feature", kernel_regularizer=l2(self.l2)))(sequence_output)
        description_features = Dropout(0.5, name="desc_dropout")(description_features)


        attention = AttentionLayer(name='attention_layer')
        attention_out = attention([merged_augumented_data, description_features])

        gmp_1 = GlobalMaxPooling1D(name='Globalmaxpool_colum')
        att_1 = gmp_1(attention_out)
        activ1 = Activation('softmax', name='AP_active_colum')
        att_1_next = activ1(att_1)
        dot1 = Dot(axes=1, normalize=False, name='column_dot')
        desc_out = dot1([att_1_next, description_features])

        attention_trans_layer = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)), name='trans_attention')
        attention_transposed = attention_trans_layer(attention_out)
        gmp_2 = GlobalMaxPooling1D(name='globalmaxpool_row')
        att_2 = gmp_2(attention_transposed)
        activ2 = Activation('softmax', name='AP_active_row')
        att_2_next = activ2(att_2)
        dot2 = Dot(axes=1, normalize=False, name='row_dot')
        code_out = dot2([att_2_next, merged_augumented_data])

        all_features = WeightedLayer(name="weighted_layer")(desc_out, code_out)

        output = Dense(50, activation='softmax', kernel_regularizer=l2(self.l2), name="softmax")(all_features)
        self.model = Model(inputs=[bert_description_inputs, bert_name_inputs, bert_ew_inputs], outputs=output)
        print('\nsummary of co-attentive representation model')
        self.model.summary()

    
    def compile(self, optimizer):
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[top_k_categorical_accuracy, categorical_accuracy])

    def fit(self, traindata, steps_per_epoch, **kwargs):
        assert self.model is not None, 'Must compile the model before fitting data'
        return self.model.fit(traindata, steps_per_epoch=steps_per_epoch, epochs=1, verbose = 1, shuffle=False, **kwargs)

    def evaluate(self, testdata, steps, **kwargs):
        assert self.model is not None, 'Must compile the model before evaluatting data'
        return self.model.evaluate(testdata, steps=steps, **kwargs)

    def predict(self, testdata, **kwargs):
        assert self.model is not None, 'Must compile the model before predicting data'
        return self.model.predict(testdata, **kwargs)

    def save(self, model_path, **kwargs):
        assert self.model is not None, 'Must compile the model before saving weights'
        self.model.save(model_path, **kwargs)

    def load(self, model_path, **kwargs):
        assert self.model is not None, 'Must compile the model loading weights'
>>>>>>> 65f55c70c0449874e6f1d72320a242641fb2d0f4
        self.model.load_weights(model_path, **kwargs)