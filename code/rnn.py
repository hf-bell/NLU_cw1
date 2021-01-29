# coding: utf-8
import sys
import time
import numpy as np

from utils import *
from rnnmath import *
from sys import stdout

vocab_size = 2


class RNN(object):
    '''
    This class implements Recurrent Neural Networks.

    You should implement code in the following functions:
            predict                         ->      predict an output sequence for a given input sequence
            acc_deltas                      ->      accumulate update weights for the RNNs weight matrices, standard Back Propagation
            acc_deltas_bptt         ->      accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time
            acc_deltas_np           ->      accumulate update weights for the RNNs weight matrices, standard Back Propagation -- for number predictions
            acc_deltas_bptt_np      ->      accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time -- for number predictions
            compute_loss            ->      compute the (cross entropy) loss between the desired output and predicted output for a given input sequence
            compute_mean_loss       ->      compute the average loss over all sequences in a corpus
            generate_sequence       ->      use the RNN to generate a new (unseen) sequnce

    Do NOT modify any other methods!
    Do NOT change any method signatures!
    '''

    def __init__(self, vocab_size, hidden_dims, out_vocab_size):
        '''
        initialize the RNN with random weight matrices.

        DO NOT CHANGE THIS

        vocab_size              size of vocabulary that is being used
        hidden_dims             number of hidden units
        out_vocab_size  size of the output vocabulary
        '''
        self.vocab_size = vocab_size
        self.hidden_dims = hidden_dims
        self.out_vocab_size = out_vocab_size

        # matrices V (input -> hidden), W (hidden -> output), U (hidden -> hidden)
        self.U = np.random.randn(self.hidden_dims, self.hidden_dims) * np.sqrt(0.1)
        self.V = np.random.randn(self.hidden_dims, self.vocab_size) * np.sqrt(0.1)
        self.W = np.random.randn(self.out_vocab_size, self.hidden_dims) * np.sqrt(0.1)

        # matrices to accumulate weight updates
        self.deltaU = np.zeros((self.hidden_dims, self.hidden_dims))
        self.deltaV = np.zeros((self.hidden_dims, self.vocab_size))
        self.deltaW = np.zeros((self.out_vocab_size, self.hidden_dims))

    def apply_deltas(self, learning_rate):
        '''
        update the RNN's weight matrices with corrections accumulated over some training instances

        DO NOT CHANGE THIS

        learning_rate   scaling factor for update weights
        '''
        # apply updates to U, V, W
        self.U += learning_rate * self.deltaU
        self.W += learning_rate * self.deltaW
        self.V += learning_rate * self.deltaV

        # reset matrices
        self.deltaU.fill(0.0)
        self.deltaV.fill(0.0)
        self.deltaW.fill(0.0)

    def predict(self, x):
        '''
        predict an output sequence y for a given input sequence x

        x       list of words, as indices, e.g.: [0, 4, 2]

        returns y,s
        y       matrix of probability vectors for each input word
        s       matrix of hidden layers for each input word

        '''

        # matrix s for hidden states, y for output states, given input x.
        # rows correspond to times t, i.e., input words
        # s has one more row, since we need to look back even at time 0 (s(t=0-1) will just be [0. 0. ....] )
        s = np.zeros((len(x) + 1, self.hidden_dims))
        y = np.zeros((len(x), self.out_vocab_size))

        for t in range(len(x)):
            Vxt = np.dot(self.V, make_onehot(x[t], self.out_vocab_size))
            Ust_1 = np.dot(self.U, s[t - 1])
            net_in = np.add(Vxt, Ust_1)
            s[t] = sigmoid(net_in)
            net_out = np.dot(self.W, s[t])
            y[t] = softmax(net_out)

        return y, s

    def acc_deltas(self, x, d, y, s):
        '''
        accumulate updates for V, W, U
        standard back propagation

        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time

        x       list of words, as indices, e.g.: [0, 4, 2]
        d       list of words, as indices, e.g.: [4, 2, 3]
        y       predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
                should be part of the return value of predict(x)
        s       predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
                should be part of the return value of predict(x)

        no return values
        '''

        del_W = 0
        del_V = 0
        del_U = 0
        for t in reversed(range(len(x))):
            der_softmax = (make_onehot(d[t], self.out_vocab_size) - y[t])

            der_sigmoid = (s[t] * (np.ones(s[t].shape) - s[t]))
            del_inputs = np.dot(self.W.T, der_softmax) * der_sigmoid

            del_W += np.outer(der_softmax, s[t])
            del_V += np.outer(del_inputs, make_onehot(x[t], self.out_vocab_size))
            del_U += np.outer(del_inputs, s[t - 1])

        self.deltaW = del_W
        self.deltaV = del_V
        self.deltaU = del_U

    def acc_deltas_bptt(self, x, d, y, s, steps):
        '''
        accumulate updates for V, W, U
        truncated back propagation through time

        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time

        x       list of words, as indices, e.g.: [0, 4, 2]
        d       list of words, as indices, e.g.: [4, 2, 3]
        y       predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
                should be part of the return value of predict(x)
        s       predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
                should be part of the return value of predict(x)

        no return values
        '''

        del_W = 0
        del_V = 0
        del_U = 0
        del_inputs = 0
        for t in reversed(range(len(x))):

            der_softmax = (make_onehot(d[t], self.out_vocab_size) - y[t])
            der_sigmoid = (s[t] * (np.ones(s[t].shape) - s[t]))

            del_inputs = np.dot(self.W.T, der_softmax) * der_sigmoid
            del_W += np.outer(der_softmax, s[t])
            del_V += np.outer(del_inputs, make_onehot(x[t], self.out_vocab_size))
            del_U += np.outer(del_inputs, s[t - 1])

            del_next_inputs = del_inputs

            for tau in reversed(range((t - steps), t)):
                der_sigmoid_tau = (s[tau] * (np.ones(s[tau].shape) - s[tau]))
                del_inputs = np.dot(self.U.T, del_next_inputs) * der_sigmoid_tau
                del_next_inputs = del_inputs
                del_V += np.outer(del_inputs, make_onehot(x[tau], self.out_vocab_size))
                del_U += np.outer(del_inputs, s[tau - 1])

        self.deltaW = del_W
        self.deltaV = del_V
        self.deltaU = del_U

    def compute_loss(self, x, d):
        '''
        compute the loss between predictions y for x, and desired output d.

        first predicts the output for x using the RNN, then computes the loss w.r.t. d

        x               list of words, as indices, e.g.: [0, 4, 2]
        d               list of words, as indices, e.g.: [4, 2, 3]

        return loss             the combined loss for all words
        '''

        loss = 0.
        y_pred, hiddens = self.predict(x)
        for t_index, t in enumerate(d):
            d_t_1h = make_onehot(t, self.out_vocab_size)
            t_loss = -np.sum(d_t_1h * np.log(y_pred[int(t_index)]))
            loss += t_loss
        return loss

    def compute_mean_loss(self, X, D):
        mean_loss = 0.
        N = 0
        tot_loss = 0
        for i in range(len(X)):
            tot_loss += self.compute_loss(X[i], D[i])
            N += len(D[i])  # Not robust?
            print(tot_loss)

        mean_loss = tot_loss / N
        return mean_loss

    def train(self, X, D, X_dev, D_dev, epochs=10, learning_rate=0.5, anneal=5, back_steps=0, batch_size=100,
              min_change=0.0001, log=True):
        if log:
            stdout.write(
                "\nTraining model for {0} epochs\ntraining set: {1} sentences (batch size {2})".format(epochs, len(X),
                                                                                                       batch_size))
            stdout.write("\nOptimizing loss on {0} sentences".format(len(X_dev)))
            stdout.write("\nVocab size: {0}\nHidden units: {1}".format(self.vocab_size, self.hidden_dims))
            stdout.write("\nSteps for back propagation: {0}".format(back_steps))
            stdout.write("\nInitial learning rate set to {0}, annealing set to {1}".format(learning_rate, anneal))
            stdout.write("\n\ncalculating initial mean loss on dev set")
            stdout.flush()

        t_start = time.time()
        loss_function = self.compute_loss

        loss_sum = sum([len(d) for d in D_dev])
        initial_loss = sum([loss_function(X_dev[i], D_dev[i]) for i in range(len(X_dev))]) / loss_sum

        if log or not log:
            stdout.write(": {0}\n".format(initial_loss))
            stdout.flush()

        prev_loss = initial_loss
        loss_watch_count = -1
        min_change_count = -1

        a0 = learning_rate

        best_loss = initial_loss
        bestU, bestV, bestW = self.U, self.V, self.W
        best_epoch = 0

        for epoch in range(epochs):
            if anneal > 0:
                learning_rate = a0 / ((epoch + 0.0 + anneal) / anneal)
            else:
                learning_rate = a0

            if log:
                stdout.write("\nepoch %d, learning rate %.04f" % (epoch + 1, learning_rate))
                stdout.flush()

            t0 = time.time()
            count = 0

            # use random sequence of instances in the training set (tries to avoid local maxima when training on batches)
            permutation = np.random.permutation(range(len(X)))
            if log:
                stdout.write("\tinstance 1")
            for i in range(len(X)):
                c = i + 1
                if log:
                    stdout.write("\b" * len(str(i)))
                    stdout.write("{0}".format(c))
                    stdout.flush()
                p = permutation[i]
                x_p = X[p]
                d_p = D[p]

                y_p, s_p = self.predict(x_p)
                if back_steps == 0:
                    self.acc_deltas(x_p, d_p, y_p, s_p)
                else:
                    self.acc_deltas_bptt(x_p, d_p, y_p, s_p, back_steps)

                if i % batch_size == 0:
                    self.deltaU /= batch_size
                    self.deltaV /= batch_size
                    self.deltaW /= batch_size
                    self.apply_deltas(learning_rate)

            if len(X) % batch_size > 0:
                mod = len(X) % batch_size
                self.deltaU /= mod
                self.deltaV /= mod
                self.deltaW /= mod
                self.apply_deltas(learning_rate)

            loss = sum([loss_function(X_dev[i], D_dev[i]) for i in range(len(X_dev))]) / loss_sum

            if log:
                stdout.write("\tepoch done in %.02f seconds" % (time.time() - t0))
                stdout.write("\tnew loss: {0}".format(loss))
                stdout.flush()

            if loss < best_loss:
                best_loss = loss
                bestU, bestV, bestW = self.U.copy(), self.V.copy(), self.W.copy()
                best_epoch = epoch

            # make sure we change the RNN enough
            if abs(prev_loss - loss) < min_change:
                min_change_count += 1
            else:
                min_change_count = 0
            if min_change_count > 2:
                print("\n\ntraining finished after {0} epochs due to minimal change in loss".format(epoch + 1))
                break

            prev_loss = loss

        t = time.time() - t_start

        if min_change_count <= 2:
            print("\n\ntraining finished after reaching maximum of {0} epochs".format(epochs))
        print("best observed loss was {0}, at epoch {1}".format(best_loss, (best_epoch + 1)))

        print("setting U, V, W to matrices from best epoch")
        self.U, self.V, self.W = bestU, bestV, bestW
        #print("U: {}, V:{}, W:{}".format(self.U, self.V, self.W))
        
        print("U = ", self.U)
        print("V = ", self.V)
        print("W = ", self.W)
 
        return best_loss


if __name__ == "__main__":

    mode = sys.argv[1].lower()
    data_folder = sys.argv[2]
    np.random.seed(2018)

    if mode == "train-lm":
        '''
        code for training language model.
        change this to different values, or use it to get you started with your own testing class
        '''
        train_size = 1000
        dev_size = 1000
        vocab_size = 2000

        hdim = int(sys.argv[3])
        lookback = int(sys.argv[4])
        lr = float(sys.argv[5])

        # get the data set vocabulary
        vocab = pd.read_table(data_folder + "/vocab.wiki.txt", header=None, sep="\s+", index_col=0,
                              names=['count', 'freq'], )
        num_to_word = dict(enumerate(vocab.index[:vocab_size]))
        word_to_num = invert_dict(num_to_word)

        # calculate loss vocabulary words due to vocab_size
        fraction_lost = fraq_loss(vocab, word_to_num, vocab_size)
        print(
            "Retained %d words from %d (%.02f%% of all tokens)\n" % (vocab_size, len(vocab), 100 * (1 - fraction_lost)))

        docs = load_lm_dataset(data_folder + '/wiki-train.txt')
        S_train = docs_to_indices(docs, word_to_num, 1, 1)
        X_train, D_train = seqs_to_lmXY(S_train)

        # Load the dev set (for tuning hyperparameters)
        docs = load_lm_dataset(data_folder + '/wiki-dev.txt')
        S_dev = docs_to_indices(docs, word_to_num, 1, 1)
        X_dev, D_dev = seqs_to_lmXY(S_dev)

        X_train = X_train[:train_size]
        D_train = D_train[:train_size]
        X_dev = X_dev[:dev_size]
        D_dev = D_dev[:dev_size]

        # q = best unigram frequency from omitted vocab
        # this is the best expected loss out of that set
        q = vocab.freq[vocab_size] / sum(vocab.freq[vocab_size:])
        r = RNN(vocab_size, hdim, vocab_size)

        r.train(X_train, D_train, X_dev, D_dev, learning_rate = lr, back_steps = lookback)

        run_loss = -1
        adjusted_loss = -1

        print("Unadjusted: %.03f" % np.exp(run_loss))
        print("Adjusted for missing vocab: %.03f" % np.exp(adjusted_loss))

    if mode == "predict-lm":
        data_folder = sys.argv[2]
        rnn_folder = sys.argv[3]

        # get saved RNN matrices and setup RNN
        U, V, W = np.load(rnn_folder + "/rnn.U.npy"), np.load(rnn_folder + "/rnn.V.npy"), np.load(
            rnn_folder + "/rnn.W.npy")
        vocab_size = len(V[0])
        hdim = len(U[0])

        dev_size = 1000

        r = RNN(vocab_size, hdim, vocab_size)
        r.U = U
        r.V = V
        r.W = W

        # get vocabulary
        vocab = pd.read_table(data_folder + "/vocab.wiki.txt", header=None, sep="\s+", index_col=0,
                              names=['count', 'freq'], )
        num_to_word = dict(enumerate(vocab.index[:vocab_size]))
        word_to_num = invert_dict(num_to_word)

        # Load the dev set (for tuning hyperparameters)
        docs = load_lm_np_dataset(data_folder + '/wiki-dev.txt')
        S_np_dev = docs_to_indices(docs, word_to_num, 1, 0)
        X_np_dev, D_np_dev = seqs_to_lmnpXY(S_np_dev)

        X_np_dev = X_np_dev[:dev_size]
        D_np_dev = D_np_dev[:dev_size]

        np_acc = r.compute_acc_lmnp(X_np_dev, D_np_dev)

        print('Number prediction accuracy on dev set:', np_acc)

        # load test data
        sents = load_lm_np_dataset(data_folder + '/wiki-test.txt')
        S_np_test = docs_to_indices(sents, word_to_num, 1, 0)
        X_np_test, D_np_test = seqs_to_lmnpXY(S_np_test)

        np_acc_test = r.compute_acc_lmnp(X_np_test, D_np_test)

        print('Number prediction accuracy on test set:', np_acc_test)
