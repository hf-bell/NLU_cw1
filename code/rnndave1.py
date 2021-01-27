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
		predict				->	predict an output sequence for a given input sequence
		acc_deltas			->	accumulate update weights for the RNNs weight matrices, standard Back Propagation
		acc_deltas_bptt		->	accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time
		acc_deltas_np		->	accumulate update weights for the RNNs weight matrices, standard Back Propagation -- for number predictions
		acc_deltas_bptt_np	->	accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time -- for number predictions
		compute_loss 		->	compute the (cross entropy) loss between the desired output and predicted output for a given input sequence
		compute_mean_loss	->	compute the average loss over all sequences in a corpus
		generate_sequence	->	use the RNN to generate a new (unseen) sequnce
	
	Do NOT modify any other methods!
	Do NOT change any method signatures!
	'''
	
	def __init__(self, vocab_size, hidden_dims, out_vocab_size):
		'''
		initialize the RNN with random weight matrices.
		
		DO NOT CHANGE THIS
		
		vocab_size		size of vocabulary that is being used
		hidden_dims		number of hidden units
		out_vocab_size	size of the output vocabulary
		'''
		self.vocab_size = vocab_size
		self.hidden_dims = hidden_dims
		self.out_vocab_size = out_vocab_size 
		
		# matrices V (input -> hidden), W (hidden -> output), U (hidden -> hidden)
		self.U = np.random.randn(self.hidden_dims, self.hidden_dims)*np.sqrt(0.1)
		self.V = np.random.randn(self.hidden_dims, self.vocab_size)*np.sqrt(0.1)
		self.W = np.random.randn(self.out_vocab_size, self.hidden_dims)*np.sqrt(0.1)
		
		# matrices to accumulate weight updates
		self.deltaU = np.zeros((self.hidden_dims, self.hidden_dims))
		self.deltaV = np.zeros((self.hidden_dims, self.vocab_size))
		self.deltaW = np.zeros((self.out_vocab_size, self.hidden_dims))

	def apply_deltas(self, learning_rate):
		'''
		update the RNN's weight matrices with corrections accumulated over some training instances
		
		DO NOT CHANGE THIS
		
		learning_rate	scaling factor for update weights
		'''
		# apply updates to U, V, W
		self.U += learning_rate*self.deltaU
		self.W += learning_rate*self.deltaW
		self.V += learning_rate*self.deltaV
		
		# reset matrices
		self.deltaU.fill(0.0)
		self.deltaV.fill(0.0)
		self.deltaW.fill(0.0)
	
	def predict(self, x):
		'''
		predict an output sequence y for a given input sequence x
		
		x	list of words, as indices, e.g.: [0, 4, 2]
		
		returns	y,s
		y	matrix of probability vectors for each input word
		s	matrix of hidden layers for each input word
		
		'''
		
		# matrix s for hidden states, y for output states, given input x.
		# rows correspond to times t, i.e., input words
		# s has one more row, since we need to look back even at time 0 (s(t=0-1) will just be [0. 0. ....] )
		s = np.zeros((len(x) + 1, self.hidden_dims))
		y = np.zeros((len(x), self.out_vocab_size))

		for t in range(len(x)):

			Vxt = np.dot(self.V, make_onehot(x[t], 3))
			Ust_1 = np.dot(self.U, s[t-1])
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

                x	list of words, as indices, e.g.: [0, 4, 2]
                d	list of words, as indices, e.g.: [4, 2, 3]
                y	predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
                        should be part of the return value of predict(x)
                s	predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
                        should be part of the return value of predict(x)

                no return values
                '''


                del_W = 0
                del_V = 0
                del_U = 0
                for t in reversed(range(len(x))):
                        der_softmax = (make_onehot(d[t],3) - y[t])

                        
                        der_sigmoid = (s[t]*(np.ones(s[t].shape) - s[t]))
                        del_inputs = np.dot(self.W.T,der_softmax)*der_sigmoid



                        
                        del_W += np.outer(der_softmax, s[t])
                        del_V += np.outer(del_inputs, make_onehot(x[t],3))
                        del_U += np.outer(del_inputs, s[t-1])

                self.deltaW = del_W
                self.deltaV = del_V
                self.deltaU = del_U


	def acc_deltas_bptt(self, x, d, y, s, steps):
                '''
                accumulate updates for V, W, U
                truncated back propagation through time

                this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time

                x	list of words, as indices, e.g.: [0, 4, 2]
                d	list of words, as indices, e.g.: [4, 2, 3]
                y	predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
                        should be part of the return value of predict(x)
                s	predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
                        should be part of the return value of predict(x)

                no return values
                '''


                del_W = 0
                del_V = 0
                del_U = 0
                del_inputs = 0
                for t in reversed(range(len(x))):
                        
                        der_softmax = (make_onehot(d[t],3) - y[t])
                        der_sigmoid = (s[t]*(np.ones(s[t].shape) - s[t]))


                        del_inputs = np.dot(self.W.T,der_softmax)*der_sigmoid
                        del_W += np.outer(der_softmax, s[t])
                        del_V += np.outer(del_inputs, make_onehot(x[t],3))
                        del_U += np.outer(del_inputs, s[t-1])

                        del_next_inputs = del_inputs


                        for tau in reversed(range((t - steps),t)):
                                der_sigmoid_tau = (s[tau]*(np.ones(s[tau].shape) - s[tau]))
                                del_inputs = np.dot(self.U.T, del_next_inputs)*der_sigmoid_tau   
                                del_next_inputs = del_inputs
                                del_V += np.outer(del_inputs, make_onehot(x[tau],3))
                                del_U += np.outer(del_inputs, s[tau-1])


                        
                        


                self.deltaW = del_W
                self.deltaV = del_V
                self.deltaU = del_U


	def compute_loss(self, x, d):
		'''
		compute the loss between predictions y for x, and desired output d.

		first predicts the output for x using the RNN, then computes the loss w.r.t. d

		x		list of words, as indices, e.g.: [0, 4, 2]
		d		list of words, as indices, e.g.: [4, 2, 3]

		return loss		the combined loss for all words
		'''

		loss = 0.
		y_pred, hiddens = self.predict(x)
		for t_index, t in enumerate(d):
			d_t_1h = make_onehot(t, 3)
			t_loss = -np.sum(d_t_1h*np.log(y_pred[int(t_index)]))
			loss += t_loss
		return loss



	def compute_mean_loss(self, X, D):
                mean_loss = 0.
                N = 0
                tot_loss = 0
                for i in range(len(X)):
                        tot_loss += self.compute_loss(X[i], D[i])
                        N+= len(D[i]) # Not robust?
                        print(tot_loss)

                mean_loss = tot_loss/N
                return mean_loss










if __name__ == "__main__":

	mode = sys.argv[1].lower()
	data_folder = sys.argv[2]
	np.random.seed(2018)
	

	
	if mode == "predict-lm":
		
		data_folder = sys.argv[2]
		rnn_folder = sys.argv[3]

		# get saved RNN matrices and setup RNN
		U,V,W = np.load(rnn_folder + "/rnn.U.npy"), np.load(rnn_folder + "/rnn.V.npy"), np.load(rnn_folder + "/rnn.W.npy")
		vocab_size = len(V[0])
		hdim = len(U[0])

		dev_size = 1000

		r = RNN(vocab_size, hdim, vocab_size)
		r.U = U
		r.V = V
		r.W = W

		# get vocabulary
		vocab = pd.read_table(data_folder + "/vocab.wiki.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
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
