import datetime
import logging
import os
import random
import warnings
from collections import OrderedDict
from typing import List, Tuple, Union

import flair
import numpy as np
import torch
import torch.nn as nn
from flair.data import Sentence, Dictionary, TaggedCorpus
from flair.models.sequence_tagger_model import clear_embeddings
from flair.nn import LockedDropout
from flair.training_utils import Metric, init_output_file, WeightExtractor
from torch import autograd
from torch.optim.lr_scheduler import ReduceLROnPlateau

from . import disambiguation
from . import treetransforms, treebanktransforms
from .containers import cellidx, cellstart, cellend
from .pcfg import DenseCFGChart
from .pruning import PruningMask
from .punctuation import applypunct


class SentenceWithChartInfo(Sentence):
	def __init__(self, *args, **kwargs):
		super(SentenceWithChartInfo, self).__init__(*args, **kwargs)
		self.chartinfo = []


class BeamWidthPredictor(nn.Module):
	def __init__(self,
				hidden_size: int,
				embeddings: flair.embeddings.TokenEmbeddings,
				tag_dictionary: Dictionary,
				tag_type: str,
				use_rnn: bool = True,
				rnn_layers: int = 1
				):

		super(BeamWidthPredictor, self).__init__()

		self.use_rnn = use_rnn
		self.hidden_size = hidden_size
		# self.use_crf: bool = use_crf
		self.rnn_layers: int = rnn_layers

		self.trained_epochs: int = 0
		self.assymetric_loss = 100.0

		self.embeddings = embeddings

		# set the dictionaries
		self.tag_dictionary: Dictionary = tag_dictionary
		self.tag_type: str = tag_type
		self.tagset_size: int = len(tag_dictionary)

		# initialize the network architecture
		self.nlayers: int = rnn_layers
		self.hidden_word = None

		# self.dropout = nn.Dropout(0.5)
		self.dropout: nn.Module = LockedDropout(0.5)

		rnn_input_dim: int = self.embeddings.embedding_length

		self.relearn_embeddings: bool = True

		if self.relearn_embeddings:
			self.embedding2nn = nn.Linear(rnn_input_dim, rnn_input_dim)

		# bidirectional LSTM on top of embedding layer
		self.rnn_type = 'LSTM'
		if self.rnn_type in ['LSTM', 'GRU']:

			if self.nlayers == 1:
				self.rnn = getattr(nn, self.rnn_type)(rnn_input_dim, hidden_size,
													  num_layers=self.nlayers,
													  bidirectional=True)
			else:
				self.rnn = getattr(nn, self.rnn_type)(rnn_input_dim, hidden_size,
													  num_layers=self.nlayers,
													  dropout=0.5,
													  bidirectional=True)

		self.nonlinearity = nn.Tanh()

		# final linear map to tag space
		if self.use_rnn:
			self.linear = nn.Linear(hidden_size * 4, len(tag_dictionary))
		else:
			self.linear = nn.Linear(self.embeddings.embedding_length * 2, len(tag_dictionary))

		if torch.cuda.is_available():
			self.cuda()

	def save(self, model_file: str):
		model_state = {
			'state_dict': self.state_dict(),
			'embeddings': self.embeddings,
			'hidden_size': self.hidden_size,
			'tag_dictionary': self.tag_dictionary,
			'tag_type': self.tag_type,
			'use_rnn': self.use_rnn,
			'rnn_layers': self.rnn_layers,
		}
		torch.save(model_state, model_file, pickle_protocol=4)

	@classmethod
	def load_from_file(cls, model_file):

		warnings.filterwarnings("ignore")
		state = torch.load(model_file, map_location={'cuda:0': 'cpu'})
		warnings.filterwarnings("default")

		model = BeamWidthPredictor(
			hidden_size=state['hidden_size'],
			embeddings=state['embeddings'],
			tag_dictionary=state['tag_dictionary'],
			tag_type=state['tag_type'],
			# use_crf=state['use_crf'],
			use_rnn=state['use_rnn'],
			rnn_layers=state['rnn_layers'])

		model.load_state_dict(state['state_dict'])
		model.eval()
		if torch.cuda.is_available():
			model = model.cuda()
		return model

	def forward(self, sentences: List[SentenceWithChartInfo]) \
		-> Tuple[List, List, List]:

		self.zero_grad()

		# first, sort sentences by number of tokens
		sentences.sort(key=lambda x: len(x), reverse=True)
		longest_token_sequence_in_batch: int = len(sentences[0])

		self.embeddings.embed(sentences)

		all_sentence_tensors = []
		lengths: List[int] = []
		tag_list: List = []

		padding = torch.FloatTensor(
			np.zeros(self.embeddings.embedding_length, dtype='float')).unsqueeze(0)

		for sentence in sentences:
			# get the chart in this sentence
			tag_idx: List[int] = [self.tag_dictionary.get_idx_for_item(cell)
									for cell in sentence.chartinfo]

			lengths.append(len(sentence.tokens))

			word_embeddings = []

			for token in sentence:
				# get the word embeddings
				word_embeddings.append(token.get_embedding().unsqueeze(0))

			# pad shorter sentences out
			for add in range(longest_token_sequence_in_batch - len(sentence.tokens)):
				word_embeddings.append(padding)

			word_embeddings_tensor = torch.cat(word_embeddings, 0)

			if torch.cuda.is_available():
				tag_list.append(torch.cuda.LongTensor(tag_idx))
			else:
				tag_list.append(torch.LongTensor(tag_idx))

			all_sentence_tensors.append(word_embeddings_tensor.unsqueeze(1))

		# padded tensor for entire batch
		# shape: (longest_token_sequence_in_batch, batch_size, embedding_dimension)
		sentence_tensor = torch.cat(all_sentence_tensors, 1)
		if torch.cuda.is_available():
			sentence_tensor = sentence_tensor.cuda()

		# --------------------------------------------------------------------
		# FF PART
		# --------------------------------------------------------------------
		sentence_tensor = self.dropout(sentence_tensor)

		if self.relearn_embeddings:
			sentence_tensor = self.embedding2nn(sentence_tensor)

		if self.use_rnn:
			packed = torch.nn.utils.rnn.pack_padded_sequence(sentence_tensor, lengths)

			rnn_output, hidden = self.rnn(packed)

			sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_output)

			sentence_tensor = self.dropout(sentence_tensor)

		# concatenating tensors for each chart cell
		chart_input_layer = []
		for sentence_no, _ in enumerate(sentences):
			rnn_outputs = []
			for l in range(longest_token_sequence_in_batch):
				for r in range(l, longest_token_sequence_in_batch):
					left = sentence_tensor[l, sentence_no, :].unsqueeze(0)
					right = sentence_tensor[r, sentence_no, :].unsqueeze(0)
					rnn_outputs.append(torch.cat([left, right], 1))
			chart_input_layer.append(torch.cat(rnn_outputs, 0).unsqueeze(1))

		chart_input_layer_tensor = torch.cat(chart_input_layer, 1)

		features = self.linear(chart_input_layer_tensor)

		# return features, lengths, tag_list

		predictions_list = []
		for sentence_no, length in enumerate(lengths):
			sentence_predictions = []
			for l in range(length):
				startidx = cellidx(l, l + 1, longest_token_sequence_in_batch, 1)
				for width in range(length - l):
					sentence_predictions.append(
						features[startidx + width, sentence_no, :].unsqueeze(0)
					)
			predictions_list.append(torch.cat(sentence_predictions, 0))

		return predictions_list, lengths, tag_list

	def bodenstab_loss(self, sentences: List[SentenceWithChartInfo]):
		all_feats, lengths, tags = self.forward(sentences)
		score = 0

		for feats, length, gold_tags in zip(all_feats, lengths, tags):
			if torch.cuda.is_available():
				tag_tensor = autograd.Variable(torch.cuda.LongTensor(gold_tags))
			else:
				tag_tensor = autograd.Variable(torch.LongTensor(gold_tags))

			scaling = []
			for backscore, gold_tag in zip(feats, gold_tags):
				_, tag_idx = torch.max(backscore, 0)
				scaling.append(
					0.0 if tag_idx == gold_tag else
				 	1.0 if tag_idx > gold_tag else
				 	self.assymetric_loss
				)

			scaling = torch.tensor(scaling)
			if torch.cuda.is_available():
				scaling = torch.tensor(scaling).cuda()
			score += nn.functional.cross_entropy(feats, tag_tensor, reduction='none').dot(scaling) / length

		return score

	def neg_log_likelihood(self, sentences: List[SentenceWithChartInfo]):
		features, lengths, tags = self.forward(sentences)
		score = 0

		for sentence_feats, sentence_tags, sentence_length in zip(features, tags, lengths):
			if torch.cuda.is_available():
				tag_tensor = autograd.Variable(torch.cuda.LongTensor(sentence_tags))
			else:
				tag_tensor = autograd.Variable(torch.LongTensor(sentence_tags))
			score += nn.functional.cross_entropy(sentence_feats, tag_tensor)

		return score

	def predict(self, sentences: Union[List[SentenceWithChartInfo],
				SentenceWithChartInfo], mini_batch_size=32) -> List[Sentence]:

		if type(sentences) is SentenceWithChartInfo:
			sentences = [sentences]

		# remove previous embeddings
		clear_embeddings(sentences)

		# make mini-batches
		batches = [sentences[x:x + mini_batch_size] for x in
					range(0, len(sentences), mini_batch_size)]

		for batch in batches:
			scores, predicted_ids = self._predict_scores_batch(batch)

			for sentence, chartinfo in zip(batch, predicted_ids):
				sentence.chartinfo = \
					[self.tag_dictionary.get_item_for_index(pred_id)
					for pred_id in chartinfo]

		return sentences

	def _predict_scores_batch(self, sentences: List[Sentence]):
		import torch.nn.functional as F

		all_feats, length, _ = self.forward(sentences)

		all_tags_seqs = []
		all_confidences = []

		for feats in all_feats:
			tag_seq = []
			confidences = []

			for backscore in feats:
				softmax = F.softmax(backscore, dim=0)
				_, idx = torch.max(backscore, 0)
				prediction = idx.item()
				tag_seq.append(prediction)
				confidences.append(softmax[prediction].item())

			all_tags_seqs.append(tag_seq)
			all_confidences.append(confidences)

		return all_confidences, all_tags_seqs


log = logging.getLogger()
# log = logging.get()


class BeamWidthPredictionTrainer:
	def __init__(self, model: BeamWidthPredictor, corpus: TaggedCorpus, test_mode: bool = False) -> None:
		self.model: BeamWidthPredictor = model
		self.corpus: TaggedCorpus = corpus
		self.test_mode: bool = test_mode

	def train(self,
			  base_path: str,
			  learning_rate: float = 0.1,
			  mini_batch_size: int = 32,
			  max_epochs: int = 100,
			  anneal_factor: float = 0.5,
			  patience: int = 4,
			  train_with_dev: bool = False,
			  embeddings_in_memory: bool = True,
			  checkpoint: bool = False,
			  save_final_model: bool = True,
			  ):

		# evaluation_method = 'accuracy'
		# evaluation_method = 'F1'
		evaluation_method = 'missclassification-costs'
		# if self.model.tag_type in ['pos', 'upos']: evaluation_method = 'accuracy'
		log.info('Evaluation method: {}'.format(evaluation_method))

		loss_txt = init_output_file(base_path, 'loss.tsv')
		with open(loss_txt, 'a') as f:
			f.write('EPOCH\tTIMESTAMP\tTRAIN_LOSS\t{}\tDEV_LOSS\t{}\tTEST_LOSS\t{}\n'.format(
				MetricWithMissclassificationCosts.tsv_header('TRAIN'),
				MetricWithMissclassificationCosts.tsv_header('DEV'),
				MetricWithMissclassificationCosts.tsv_header('TEST')))

		weight_extractor = WeightExtractor(base_path)

		optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

		anneal_mode = 'min' if train_with_dev else 'max'
		scheduler: ReduceLROnPlateau = ReduceLROnPlateau(optimizer,
				factor=anneal_factor, patience=patience, mode=anneal_mode)

		train_data = self.corpus.train

		# if training also uses dev data, include in training set
		if train_with_dev:
			train_data.extend(self.corpus.dev)

		# At any point you can hit Ctrl + C to break out of training early.
		try:

			for epoch in range(max_epochs):
				log.info('-' * 100)

				if not self.test_mode: random.shuffle(train_data)

				batches = [train_data[x:x + mini_batch_size] for x in range(0, len(train_data), mini_batch_size)]

				self.model.train()

				current_loss: float = 0
				seen_sentences = 0
				modulo = max(1, int(len(batches) / 10))

				for group in optimizer.param_groups:
					learning_rate = group['lr']

				for batch_no, batch in enumerate(batches):
					batch: List[SentenceWithChartInfo] = batch

					optimizer.zero_grad()

					# Step 4. Compute the loss, gradients, and update the parameters by calling optimizer.step()
					loss = self.model.bodenstab_loss(batch)
					# loss = self.model.neg_log_likelihood(batch)

					current_loss += loss.item()
					seen_sentences += len(batch)

					loss.backward()
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
					optimizer.step()

					if not embeddings_in_memory:
						self.clear_embeddings_in_batch(batch)

					if batch_no % modulo == 0:
						log.info("epoch {0} - iter {1}/{2} - loss {3:.8f}".format(
							epoch + 1, batch_no, len(batches), current_loss / seen_sentences))
						iteration = epoch * len(batches) + batch_no
						weight_extractor.extract_weights(self.model.state_dict(), iteration)

				current_loss /= len(train_data)

				# switch to eval mode
				self.model.eval()

				# if checkpointing is enable, save model at each epoch
				if checkpoint:
					self.model.save(base_path + "/checkpoint.pt")

				log.info('-' * 100)

				dev_score = dev_metric = None
				if not train_with_dev:
					dev_score, dev_metric = self.evaluate(self.corpus.dev, base_path,
						evaluation_method=evaluation_method, embeddings_in_memory=embeddings_in_memory)

				test_score, test_metric = self.evaluate(self.corpus.test, base_path,
						evaluation_method=evaluation_method, embeddings_in_memory=embeddings_in_memory)

				# anneal against train loss if training with dev, otherwise anneal against dev score
				scheduler.step(current_loss) if train_with_dev else scheduler.step(dev_score)

				log.info("EPOCH {0}: lr {1:.4f} - bad epochs {2}".format(epoch + 1, learning_rate, scheduler.num_bad_epochs))
				if not train_with_dev:
					log.info("{0:<4}: f-score {1:.4f} - acc {2:.4f} - tp {3} - fp {4} - fn {5} - tn {6} - mc {7}".format(
						'DEV', dev_metric.f_score(), dev_metric.accuracy(), dev_metric._tp, dev_metric._fp, dev_metric._fn, dev_metric._tn, dev_metric._mc))
				log.info("{0:<4}: f-score {1:.4f} - acc {2:.4f} - tp {3} - fp {4} - fn {5} - tn {6} - mc {7}".format(
					'TEST', test_metric.f_score(), test_metric.accuracy(), test_metric._tp, test_metric._fp, test_metric._fn, test_metric._tn, test_metric._mc))

				with open(loss_txt, 'a') as f:
					dev_metric_str = dev_metric.to_tsv() \
						if dev_metric is not None \
						else MetricWithMissclassificationCosts.to_empty_tsv()
					f.write('{}\t{:%H:%M:%S}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
						epoch, datetime.datetime.now(), '_',
						MetricWithMissclassificationCosts.to_empty_tsv(), '_',
						dev_metric_str, '_', test_metric.to_tsv()))

				# if we use dev data, remember best model based on dev evaluation score
				if not train_with_dev and dev_score == scheduler.best:
					self.model.save(base_path + "/best-model.pt")

			# if we do not use dev data for model selection, save final model
			if save_final_model:
				if train_with_dev:
					self.model.save(base_path + "/final-model.pt")

		except KeyboardInterrupt:
			log.info('-' * 100)
			log.info('Exiting from training early.')
			log.info('Saving model ...')
			self.model.save(base_path + "/final-model.pt")
			log.info('Done.')

	def evaluate(self, evaluation: List[SentenceWithChartInfo], out_path=None, evaluation_method: str = 'F1',
				 eval_batch_size: int = 32,
				 embeddings_in_memory: bool = True):

		batch_no: int = 0
		batches = [evaluation[x:x + eval_batch_size] for x in
				   range(0, len(evaluation), eval_batch_size)]

		metric = MetricWithMissclassificationCosts('')

		lines: List[str] = []

		for batch in batches:
			batch_no += 1

			all_scores, tag_seq = self.model._predict_scores_batch(batch)
			predictedbeams = tag_seq

			for sentence, scores, predictedbeam in zip(batch, all_scores, predictedbeams):
				# get the predicted tag

				predicted = [self.model.tag_dictionary.get_item_for_index(predicted_id) for predicted_id in predictedbeam]
				sentence.beaminfopredicted = predicted
				for idx, score in enumerate(scores):

					# append both to file for evaluation
					eval_line = '{} {}\n'.format(sentence.chartinfo[idx],
												 sentence.beaminfopredicted[idx])

					lines.append(eval_line)

				# check for true positives, false positives and false negatives
				assert len(predicted) == len(sentence.chartinfo)
				for gold, pred in zip(sentence.chartinfo, predicted):
					if gold == pred:
						metric.tp()
					else:
						metric.fp()
						metric.fn()
					if DEFAULTBEAMS[gold] > DEFAULTBEAMS[pred]:
						metric.mc(self.model.assymetric_loss)
					elif DEFAULTBEAMS[gold] < DEFAULTBEAMS[pred]:
						metric.mc(1)

			if not embeddings_in_memory:
				self.clear_embeddings_in_batch(batch)

		if out_path is not None:
			test_tsv = os.path.join(out_path, "test.tsv")
			with open(test_tsv, "w", encoding='utf-8') as outfile:
				outfile.write(''.join(lines))

		if evaluation_method == 'accuracy':
			score = metric.accuracy()
			return score, metric

		if evaluation_method == 'F1':
			score = metric.f_score()
			return score, metric

		if evaluation_method == 'missclassification-costs':
			score = metric.neg_missclassification_costs()
			return score, metric

	def clear_embeddings_in_batch(self, batch: List[Sentence]):
		for sentence in batch:
			for token in sentence.tokens:
				token.clear_embeddings()


PRUNE = 'PRUNE'
SENTS_SUFFIX = '_sents.txt'
GOLD_CELL_SUFFIX = '_goldcells.txt'
MAXBEAM = 5.0
STEPSIZE = 0.5

import math

def defaultbags(x: Union[str, float]):
	if x == PRUNE:
		return PRUNE
	if x > math.log(10**MAXBEAM):
		return "INF"
	if x <= 0:
		return "E0.0"
	else:
		for y in range(0, int(MAXBEAM / STEPSIZE) + 1):
			if x <= math.log(10**(y * STEPSIZE)):
				return "E%.1f" % (y * STEPSIZE)


DEFAULTBEAMS = OrderedDict()
DEFAULTBEAMS["PRUNE"] = float('-Inf')
# NB: beam_beta of 0.0 disables beam instead of just keeping the best item
# TODO: this is unfavourable!
DEFAULTBEAMS["E0.0"] = math.log(10**0.00001)
for y in range(1, int(MAXBEAM / STEPSIZE) + 1):
	DEFAULTBEAMS["E%.1f" % (y * STEPSIZE)] = math.log(10**(y * STEPSIZE))
DEFAULTBEAMS["INF"] = float('Inf')


class BeamWidthsCorpusReader:
	@staticmethod
	def read_beam_width_corpus(train_prefix, test_prefix, dev_prefix=None, categories=defaultbags):
		sentences_train: List[SentenceWithChartInfo] = BeamWidthsCorpusReader.__read_sentences(
			train_prefix + GOLD_CELL_SUFFIX, train_prefix + SENTS_SUFFIX, categories)
		if dev_prefix is not None:
			sentences_dev: List[SentenceWithChartInfo] = BeamWidthsCorpusReader.__read_sentences(
				dev_prefix + GOLD_CELL_SUFFIX, dev_prefix + SENTS_SUFFIX, categories
			)
		else:
			sentences_dev: List[SentenceWithChartInfo] = [sentences_train[i] for i in
											BeamWidthsCorpusReader.__sample()
														if i < len(sentences_train)]
			sentences_train = [x for x in sentences_train if x not in sentences_dev]

		sentences_test = BeamWidthsCorpusReader.__read_sentences(
			test_prefix + GOLD_CELL_SUFFIX, test_prefix + SENTS_SUFFIX, categories
		)
		return TaggedCorpus(sentences_train, sentences_dev, sentences_test)

	@staticmethod
	def __read_sentences(path_to_beam_file, path_to_sentence_file, categories):
		sentences = []
		with open(path_to_beam_file, 'r') as beam_file, open(path_to_sentence_file) as sent_file:
			for beamstr, sentstr in zip(beam_file, sent_file):
				if sentstr[-1] == '\n':
					sentstr = sentstr[:-1]
				sent = SentenceWithChartInfo(sentstr)
				chartsize = 1 + cellidx(len(sent) - 1, len(sent), len(sent), 1)
				sent.chartinfo = [categories(PRUNE) for _ in range(chartsize)]
				goldcells = eval(beamstr, {'inf': float('inf')})
				for cell in goldcells:
					sent.chartinfo[cell] = categories(goldcells[cell])
				sentences.append(sent)
		return sentences

	@staticmethod
	def __sample():
		sample = [7199, 2012, 7426, 1374, 2590, 4401, 7659, 2441, 4209, 6997, 6907, 4789, 3292,
				  4874, 7836, 2065, 1804,
				  2409,
				  6353, 86, 1412, 5431, 3275, 7696, 3325, 7678, 6888, 5326, 5782, 3739, 4972, 6350,
				  7167, 6558, 918,
				  6444,
				  5368, 731, 244, 2029, 6200, 5088, 4688, 2580, 2153, 5477, 714, 1570, 6651, 5724,
				  4090, 167, 1689,
				  6166,
				  7304, 3705, 256, 5689, 6282, 707, 5390, 1367, 4167, 16, 6554, 5093, 3944, 5008,
				  3510, 1741, 1, 4464,
				  173,
				  5362, 6827, 35, 1662, 3136, 1516, 3826, 1575, 6771, 5965, 1449, 7806, 632, 5870,
				  3566, 1434, 2361,
				  6348,
				  5140, 7765, 4800, 6541, 7910, 2021, 1041, 3171, 2137, 495, 2249, 7334, 4806, 844,
				  3848, 7396, 3861,
				  1337,
				  430, 1325, 36, 2289, 720, 4182, 3955, 3451, 192, 3715, 3144, 1140, 2874, 6728,
				  4877, 1876, 2551, 2910,
				  260,
				  7767, 7206, 5577, 6707, 3392, 1830, 842, 5264, 4042, 3572, 331, 6995, 2307, 5664,
				  2878, 1115, 1880,
				  1548,
				  3740, 860, 1799, 2099, 7359, 4648, 2264, 1018, 5417, 3052, 2480, 2256, 6672, 6647,
				  1272, 1986, 7063,
				  4071,
				  3199, 3652, 1797, 1693, 2008, 4138, 7428, 3083, 1494, 4911, 728, 1556, 7651, 2535,
				  2160, 4014, 1438,
				  6148,
				  551, 476, 4198, 3835, 1489, 6404, 7346, 1178, 607, 7693, 4146, 6655, 4355, 1571,
				  522, 5835, 622, 1267,
				  6778, 5236, 5211, 5039, 3836, 1751, 1019, 6952, 7610, 7677, 4224, 1485, 4101,
				  5793, 6708, 5741, 4630,
				  5857,
				  6959, 847, 4375, 3458, 4936, 6887, 5, 3150, 5551, 4840, 2618, 7456, 7600, 5995,
				  5270, 5496, 4316,
				  1479,
				  517, 2940, 2337, 7461, 3296, 4133, 491, 6408, 7609, 4290, 5028, 7471, 6337, 488,
				  5033, 5967, 1209,
				  5511,
				  5449, 3837, 4760, 4490, 6550, 2676, 371, 3962, 4507, 5268, 4285, 5257, 859, 14,
				  4487, 5669, 6594,
				  6544,
				  7427, 5624, 4882, 7425, 2378, 1498, 931, 7253, 2638, 2897, 5670, 6463, 5300, 6802,
				  4229, 7076, 6848,
				  6414,
				  1465, 7243, 989, 7204, 1926, 1255, 1794, 2115, 3975, 6987, 3166, 105, 3856, 3272,
				  3977, 4097, 2612,
				  2869,
				  6022, 153, 3357, 2439, 6491, 766, 3840, 2683, 5074, 159, 5407, 3029, 4815, 1782,
				  4970, 6250, 5377,
				  6473,
				  5151, 4687, 798, 5214, 3364, 6412, 7125, 3495, 2385, 4476, 863, 5493, 5830, 938,
				  2979, 7808, 4830,
				  4180,
				  1565, 4818, 702, 1442, 4673, 6920, 2089, 1930, 2036, 1436, 6632, 1006, 5256, 5666,
				  6401, 3415, 4693,
				  5890,
				  7124, 3853, 884, 4650, 4550, 7406, 3394, 6715, 6754, 3932, 599, 1816, 3273, 5016,
				  2918, 526, 6883,
				  3089,
				  64, 1305, 7442, 6837, 783, 4536, 100, 4951, 2933, 3750, 3232, 7150, 1934, 3576,
				  2900, 7883, 964, 4025,
				  28,
				  1732, 382, 166, 6053, 6320, 2058, 652, 3182, 6836, 4547, 419, 1600, 6891, 6235,
				  7208, 7190, 7144,
				  3133,
				  4775, 4892, 895, 4428, 7929, 7297, 7773, 5325, 2799, 5645, 1192, 1672, 2540, 6812,
				  5441, 2681, 342,
				  333,
				  2161, 593, 5463, 1568, 5252, 4194, 2280, 2423, 2118, 7455, 4553, 5960, 3163, 7147,
				  4305, 5599, 2775,
				  5334,
				  4727, 6926, 2189, 7778, 7245, 2066, 1259, 2074, 7866, 7403, 4642, 5490, 3563,
				  6923, 3934, 5728, 5425,
				  2369,
				  375, 3578, 2732, 2675, 6167, 6726, 4211, 2241, 4585, 4272, 882, 1821, 3904, 6864,
				  5723, 4708, 3226,
				  7151,
				  3911, 4274, 4945, 3719, 7467, 7712, 5068, 7181, 745, 2846, 2695, 3707, 1076, 1077,
				  2698, 5699, 1040,
				  6338,
				  631, 1609, 896, 3607, 6801, 3593, 1698, 91, 639, 2826, 2937, 493, 4218, 5958,
				  2765, 4926, 4546, 7400,
				  1909,
				  5693, 1871, 1687, 6589, 4334, 2748, 7129, 3332, 42, 345, 709, 4685, 6624, 377,
				  3204, 2603, 7183, 6123,
				  4249, 1531, 7, 703, 6978, 2856, 7871, 7290, 369, 582, 4704, 4979, 66, 1139, 87,
				  5166, 967, 2727, 5920,
				  6806, 5997, 1301, 5826, 1805, 4347, 4870, 4213, 4254, 504, 3865, 189, 6393, 7281,
				  2907, 656, 6617,
				  1807,
				  6258, 3605, 1009, 3694, 3004, 2870, 7710, 2608, 400, 7635, 4392, 3055, 942, 2952,
				  3441, 902, 5892,
				  574,
				  5418, 6212, 1602, 5619, 7094, 1168, 3877, 3888, 1618, 6564, 455, 4581, 3258, 2606,
				  4643, 2454, 2763,
				  5332,
				  6158, 940, 2343, 7902, 3438, 6117, 2198, 3842, 4773, 1492, 2424, 7662, 6559, 1196,
				  3203, 5286, 6764,
				  3829,
				  4746, 1117, 2120, 1378, 5614, 4871, 4024, 5489, 3312, 1094, 1838, 3964, 3151,
				  4545, 5795, 1739, 4920,
				  5690,
				  2570, 3530, 2751, 1426, 2631, 88, 7728, 3741, 5654, 3157, 5557, 6668, 7309, 7313,
				  807, 4376, 4512,
				  6786,
				  7898, 2429, 3890, 2418, 2243, 2330, 4561, 6119, 2864, 5570, 2485, 5499, 4983,
				  6257, 3692, 1563, 1939,
				  126,
				  3299, 2811, 7933, 465, 5976, 3712, 4478, 7671, 3143, 1947, 6133, 1928, 5725, 5747,
				  1107, 163, 3610,
				  3723,
				  1496, 7477, 53, 6548, 5548, 4357, 4963, 5896, 5361, 7295, 7632, 3559, 6740, 6312,
				  6890, 3303, 625,
				  7681,
				  7174, 6928, 1088, 2133, 4276, 5299, 4488, 5354, 3044, 3321, 409, 6218, 2255, 829,
				  2129, 673, 1588,
				  6824,
				  1297, 6996, 4324, 7423, 5209, 7617, 3041, 78, 5518, 5392, 4967, 3704, 497, 858,
				  1833, 5108, 6095,
				  6039,
				  6705, 5561, 5888, 3883, 1048, 1119, 1292, 5639, 4358, 2487, 1235, 125, 4453, 3035,
				  3304, 6938, 2670,
				  4322,
				  648, 1785, 6114, 6056, 1515, 4628, 5036, 37, 1226, 6081, 4473, 953, 5009, 217,
				  5952, 755, 2604, 3060,
				  3322,
				  6087, 604, 2260, 7897, 3129, 616, 1593, 69, 230, 1526, 6349, 6452, 4235, 1752,
				  4288, 6377, 1229, 395,
				  4326,
				  5845, 5314, 1542, 6483, 2844, 7088, 4702, 3300, 97, 7817, 6804, 471, 3624, 3773,
				  7057, 2391, 22, 3293,
				  6619, 1933, 6871, 164, 7796, 6744, 1589, 1802, 2880, 7093, 906, 389, 7892, 976,
				  848, 4076, 7818, 5556,
				  3507, 4740, 4359, 7105, 2938, 683, 4292, 1849, 3121, 5618, 4407, 2883, 7502, 5922,
				  6130, 301, 4370,
				  7019,
				  3009, 425, 2601, 3592, 790, 2656, 5455, 257, 1500, 3544, 818, 2221, 3313, 3426,
				  5915, 7155, 3110,
				  4425,
				  5255, 2140, 5632, 614, 1663, 1787, 4023, 1734, 4528, 3318, 4099, 5383, 3999, 722,
				  3866, 1401, 1299,
				  2926,
				  1360, 1916, 3259, 2420, 1409, 2817, 5961, 782, 1636, 4168, 1344, 4327, 7780, 7335,
				  3017, 6582, 4623,
				  7198,
				  2499, 2139, 3821, 4822, 2552, 4904, 4328, 6666, 4389, 3687, 1014, 7829, 4802,
				  5149, 4199, 1866, 1992,
				  2893,
				  6957, 3099, 1212, 672, 4616, 758, 6421, 2281, 6528, 3148, 4197, 1317, 4258, 1407,
				  6618, 2562, 4448,
				  6137,
				  6151, 1817, 3278, 3982, 5144, 3311, 3453, 1722, 4912, 3641, 5560, 2234, 6645,
				  3084, 4890, 557, 1455,
				  4152,
				  5784, 7221, 3078, 6961, 23, 4281, 6012, 156, 5109, 6984, 6140, 6730, 4965, 7123,
				  85, 2912, 5192, 1425,
				  1993, 4056, 598]
		return sample


def train_model(train_prefix: str, test_prefix: str, training_path: str):
	from flair.embeddings import CharLMEmbeddings, StackedEmbeddings
	fm = CharLMEmbeddings("german-forward")
	bm = CharLMEmbeddings("german-backward")
	embedding = StackedEmbeddings(embeddings=[fm, bm])

	tag_dict = Dictionary(add_unk=False)
	for item in DEFAULTBEAMS: tag_dict.add_item(item)

	model: BeamWidthPredictor = BeamWidthPredictor(hidden_size=256,
			embeddings=embedding, tag_dictionary=tag_dict, tag_type=None)

	corpus = BeamWidthsCorpusReader.read_beam_width_corpus(train_prefix,
			test_prefix)

	trainer: BeamWidthPredictionTrainer \
		= BeamWidthPredictionTrainer(model, corpus)

	trainer.train(training_path, embeddings_in_memory=False, max_epochs=20)

	return model


def sentencetopruningmask(sentence: SentenceWithChartInfo):
	mask: PruningMask = PruningMask()
	mask.openbracket = [True for _ in sentence]
	mask.closebracket = [True for _ in sentence]
	dynamicbeams = mask.dynamicbeams = []
	lensent = len(sentence)
	idx = 0
	for l in range(0, lensent):
		for r in range(l + 1, lensent + 1):
			dynamicbeams.append(DEFAULTBEAMS[sentence.chartinfo[idx]])
			idx += 1
	return mask


class MetricWithMissclassificationCosts(Metric):
	def __init__(self, name):
		super(MetricWithMissclassificationCosts, self).__init__(name)
		self._mc = 0.0

	def to_tsv(self):
		tsv = super(MetricWithMissclassificationCosts, self).to_tsv()
		return '{}\t{}'.format(tsv, self._mc)

	def print(self):
		log.info(self)

	def mc(self, cost=1.0):
		self._mc += cost

	def missclassification_costs(self):
		return self._mc

	def neg_missclassification_costs(self):
		return -self._mc

	@staticmethod
	def tsv_header(prefix=None):
		header = Metric.tsv_header(prefix)
		if prefix:
			return '{0}\t{1}_MISSCLASSIFICATION_COSTS'.format(header, prefix)

		return '{}\tMISSCLASSIFICATION_COSTS'.format(header)

	@staticmethod
	def to_empty_tsv():
		empty = Metric.to_empty_tsv()
		return '{}\t_'.format(empty)

	def __str__(self):
		s = super(MetricWithMissclassificationCosts, self).__str__()
		return '{0} - missclassification costs: {1}'.format(s, self._mc)


def predictdynamicbeams(testset, model: BeamWidthPredictor):
	model.eval()
	pruningmasks: OrderedDict[int, PruningMask] \
		= OrderedDict((n, PruningMask()) for (n, _) in testset.items())
	sentences = OrderedDict((n, SentenceWithChartInfo(' '.join([w for w, _ in sent])))
							for (n, (_, _, sent, _)) in testset.items())
	sentences_list = [s for _, s in sentences.items()]

	model.predict(sentences_list)

	for n in pruningmasks:
		sent = sentences[n]
		pruningmasks[n].openbracket = [True for _ in sent]
		pruningmasks[n].closebracket = [True for _ in sent]
		pruningmasks[n].dynamicbeams = [DEFAULTBEAMS[x] for x in sentences[n].chartinfo]

	return pruningmasks


def beamwidths(chart: DenseCFGChart, goldtree, sent, prm):
	# reproduce preprocessing so that gold items can be counted
	goldtree = goldtree.copy(True)
	applypunct(prm.punct, goldtree, sent[:])
	if prm.transformations:
		treebanktransforms.transform(goldtree, sent, prm.transformations)
	treetransforms.binarizetree(goldtree, prm.binarization, prm.relationalrealizational)
	treetransforms.addfanoutmarkers(goldtree)

	# todo: ask Andreas about markorigin

	goldtree = treetransforms.splitdiscnodes(goldtree.copy(True),
			prm.stages[0].markorigin)

	from .tree import DrawTree
	logging.log(5, DrawTree(goldtree, sent))

	for node in goldtree.subtrees():
		item = chart.itemid(node.label, node.leaves())
		cell = item // prm.stages[0].grammar.nonterminals
		logging.log(5, "%s %s %d %d" % (node.label, str(node.leaves()),
			item, cell))
	golditems = [chart.itemid(node.label, node.leaves())
				for node in goldtree.subtrees()]

	goldbeams = {}

	def updated_goldbeam(items, beams):
		_counter = 0
		for item in items:
			cell = item // prm.stages[0].grammar.nonterminals
			start = cellstart(cell, len(sent), 1)
			end = cellend(cell, len(sent), 1)
			# bestitemprob = chart.getbeambucket(cell) - beam
			bestitem = chart.bestsubtree(start, end, skipsplit=False)
			bestitemprob = chart._subtreeprob(bestitem)
			golditemprob = chart._subtreeprob(item)
			if golditemprob != float("Inf"):
				_counter += 1
			if bestitemprob == float("Inf"):
				goldbeam = golditemprob
			else:
				goldbeam = golditemprob - bestitemprob
			logging.log(5, "%d [%d-%d]: gold: %f goldbeam: %f" % (cell, start, end, golditemprob, goldbeam))
			try:
				beams[cell] = max(beams[cell], goldbeam)
			except KeyError:
				beams[cell] = goldbeam
			assert bestitemprob <= golditemprob
		return _counter

	counter = updated_goldbeam(golditems, goldbeams)
	# if gold tree not in chart, add items viterbi tree
	if counter != len(golditems) and chart:
		k = 1

		disambiguation.getderivations(chart, k, derivstrings=True)
		stage = prm.stages[0]
		if stage.objective == 'shortest':
			stage.grammar.switch('default'
								 if stage.estimator == 'rfe'
								 else stage.estimator, True)
		tags = None
		parsetrees, msg1 = disambiguation.marginalize('mpd',
			chart, sent=sent, tags=tags,
			k=stage.m, sldop_n=stage.sldop_n,
			mcplambda=stage.mcplambda,
			mcplabels=stage.mcplabels,
			ostag=stage.dop == 'ostag',
			require=set(),
			block=set())

		from operator import itemgetter
		from .tree import ParentedTree

		resultstr, prob, fragments = max(parsetrees, key=itemgetter(1))
		parsetree = ParentedTree(resultstr)

		viterbiitems = [chart.itemid(node.label, node.leaves())
						 for node in parsetree.subtrees()]

		updated_goldbeam(viterbiitems, goldbeams)

	return goldbeams, counter == len(golditems)


def generatebeamdata(prm, treeset, beampath: str, sentpath: str, top, usetags):
	from .parser import punctprune, alignsent, estimateitems, escape, \
		ptbescape, replaceraretestwords
	from .pcfg import parse

	mode = 'w'
	skip = 0
	# skip expensive training data generation if files already exist
	# and have `len(treeset)` many lines
	# append to file, if *consistent* partial training data already exists
	if os.path.isfile(sentpath) and os.path.isfile(sentpath):
		with open(beampath, 'r') as beamfile, open(sentpath, 'r') as sentfile:
			beams = sum(1 for _ in beamfile)
			sents = sum(1 for _ in sentfile)
			if beams == sents and sents == len(treeset):
				return
			elif beams == sents and beams < len(treeset):ü
				mode = 'a'
				skip = beams

	stage = prm.stages[0]

	with open(beampath, mode) as beamfile, open(sentpath, mode) as sentfile:
		parsed = 0
		counter = 0
		for _, (tagged_sent, tree, _, _) in treeset.items():
			counter += 1
			if counter <= skip:
				continue
			sent = origsent_ = [w for w, _ in tagged_sent]
			tags = [t for _, t in tagged_sent] if usetags else None
			if 'PUNCT-PRUNE' in (prm.transformations or ()):
				origsent = sent[:]
				punctprune(None, sent)
				if tags:
					newtags = alignsent(sent, origsent, dict(enumerate(tags)))
					tags = [newtags[n] for n, _ in enumerate(sent)]
			if 'PTBbrackets' in (prm.transformations or ()):
				sent = [ptbescape(token) for token in sent]
			else:
				sent = [escape(token) for token in sent]
			if prm.postagging and prm.postagging.method == 'unknownword':
				sent = list(replaceraretestwords(sent,
												 prm.postagging.unknownwordfun,
												 prm.postagging.lexicon,
												 prm.postagging.sigs))
			if tags is not None:
				tags = list(tags)

			# disabling beam_beta
			# beam_beta = 0.0
			# choosing small beam
			beam_beta = -math.log(1.0e-12)

			chart, msg = parse(
				sent, stage.grammar,
				tags=tags,
				start=top,
				whitelist=None,
				beam_beta=beam_beta,
				beam_delta=stage.beam_delta,
				itemsestimate=estimateitems(
					sent, stage.prune, stage.mode, stage.dop),
				postagging=prm.postagging,
				pruning=None)
			if not chart:
				logging.info("Could not parse: " + str(sent))

			goldbeam, goldinchart = beamwidths(chart, tree, sent, prm)
			if goldinchart: parsed += 1
			beamfile.write(str(goldbeam) + '\n')
			sentfile.write(' '.join(origsent_) + '\n')
			if counter % 10 == 0:
				beamfile.flush()
				sentfile.flush()

		logging.info("Found gold trees for %s out of %s sentences in parse charts." % (
		parsed, len(treeset.items())))
