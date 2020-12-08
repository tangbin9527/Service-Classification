<<<<<<< HEAD
import argparse

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Args for Service Classification")

    # data parameters
    parser.add_argument("--data_path", default="./data/",
                        type=str, help="Parent path of data")
    parser.add_argument("--train_data_path", default="TrainData.pickle",
                        type=str, help="Path of train data")
    parser.add_argument("--test_data_path", default="TestData.pickle",
                        type=str, help="Path of test data")
    parser.add_argument("--bert_path", default="https://hub.tensorflow.google.cn/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
                        type=str, help="BERT vectors for word representation")
    parser.add_argument("--informative_words_length", default=8,
                        type=int, help="Num of the informative words")
    parser.add_argument("--description_length", default=None,
                        type=int, help="Max num of the description sequence length")
    parser.add_argument("--name_length", default=10,
                        type=int, help="Max num of the name sequence length")

    # model parameters
    parser.add_argument("--model_name", default="model/cobert.hdf5",
                        type=str, help="Name of model")
    parser.add_argument("--embedding_size", default=768,
                        type=int, help="Dimensionality of word embedding")
    parser.add_argument("--cnn1_dropout", default=0.3,
                        type=float, help="Dropout of the first cnn layer")
    parser.add_argument("--convolutional_filters", default=32,
                        type=int, help="Num of the convolutional filters")
    parser.add_argument("--cnn2_dropout", default=0.5,
                        type=float, help="Dropout of the second cnn layer")

    parser.add_argument("--hidden_size", default=512,
                        type=int, help="Dimensionality of LSTM hidden")
    parser.add_argument("--blstm_dropout", default=0.5,
                        type=float, help="Dropout of the LSTM layer")
    parser.add_argument("--mode", choices=["train", "eval"], default='train',
                        help="The mode to run. The `train` mode trains a model;"
                        " the `eval` mode evaluat models in a test set ")

    # train parameters
    parser.add_argument("--batch_size", default=32,
                        type=int, help="Batch size")
    parser.add_argument("--epoch", default=1,
                        type=int, help="Number of the training epoch")
    parser.add_argument("--evaluate_every", default=1,
                        type=int, help="Evaluate model on test set after this many steps")
    parser.add_argument("--learning_rate", default=0.01,
                        type=float, help="learning rate")
    parser.add_argument("--l2_reg_lambda", default=0.01,
                        type=float, help="L2 regularization lambda(hypa parameters)")

    parser.add_argument("--n_gpu", default=0,
                        type=int, help="# gpu to run on")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    for arg in vars(args):
        print("{}={}".format(arg.upper(), getattr(args, arg)))
    print(args.__dict__)
    print(args.train_data_path)


if __name__ == "__main__":
    main()
=======
import argparse

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Args for Service Classification")

    # data parameters
    parser.add_argument("--data_path", default="./data/",
                        type=str, help="Parent path of data")
    parser.add_argument("--train_data_path", default="TrainData.pickle",
                        type=str, help="Path of train data")
    parser.add_argument("--test_data_path", default="TestData.pickle",
                        type=str, help="Path of test data")
    parser.add_argument("--bert_path", default="https://hub.tensorflow.google.cn/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
                        type=str, help="BERT vectors for word representation")
    parser.add_argument("--informative_words_length", default=8,
                        type=int, help="Num of the informative words")
    parser.add_argument("--description_length", default=None,
                        type=int, help="Max num of the description sequence length")
    parser.add_argument("--name_length", default=10,
                        type=int, help="Max num of the name sequence length")

    # model parameters
    parser.add_argument("--model_name", default="model/cobert.hdf5",
                        type=str, help="Name of model")
    parser.add_argument("--embedding_size", default=768,
                        type=int, help="Dimensionality of word embedding")
    parser.add_argument("--cnn1_dropout", default=0.3,
                        type=float, help="Dropout of the first cnn layer")
    parser.add_argument("--convolutional_filters", default=32,
                        type=int, help="Num of the convolutional filters")
    parser.add_argument("--cnn2_dropout", default=0.5,
                        type=float, help="Dropout of the second cnn layer")

    parser.add_argument("--hidden_size", default=512,
                        type=int, help="Dimensionality of LSTM hidden")
    parser.add_argument("--blstm_dropout", default=0.5,
                        type=float, help="Dropout of the LSTM layer")
    parser.add_argument("--mode", choices=["train", "eval"], default='train',
                        help="The mode to run. The `train` mode trains a model;"
                        " the `eval` mode evaluat models in a test set ")

    # train parameters
    parser.add_argument("--batch_size", default=32,
                        type=int, help="Batch size")
    parser.add_argument("--epoch", default=1,
                        type=int, help="Number of the training epoch")
    parser.add_argument("--evaluate_every", default=1,
                        type=int, help="Evaluate model on test set after this many steps")
    parser.add_argument("--learning_rate", default=0.01,
                        type=float, help="learning rate")
    parser.add_argument("--l2_reg_lambda", default=0.01,
                        type=float, help="L2 regularization lambda(hypa parameters)")

    parser.add_argument("--n_gpu", default=0,
                        type=int, help="# gpu to run on")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    for arg in vars(args):
        print("{}={}".format(arg.upper(), getattr(args, arg)))
    print(args.__dict__)
    print(args.train_data_path)


if __name__ == "__main__":
    main()
>>>>>>> 65f55c70c0449874e6f1d72320a242641fb2d0f4
