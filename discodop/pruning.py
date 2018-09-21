from .tree import Tree
import logging
from typing import List, Union
from flair.data import Sentence, Token
from flair.models import SequenceTagger
from flair.training_utils import clear_embeddings
import torch
import warnings
import os


def bracketannotation(tree: Tree, sent):
	spans = [t2.leaves() for t2 in tree.subtrees() if len(t2.leaves()) > 1]
	starts = {min(span) for span in spans}
	ends = {max(span) for span in spans}
	return [(w, i in starts, i in ends) for i, w in enumerate(sent)]


def annotation2file(annotation, file):
	file.write('\n'.join("%s\t%s\t%s" % w for w in annotation) + '\n\n')


class SequenceTaggerWithProbs(SequenceTagger):
	def set_goal_tags(self, goal_tags):
		self.goal_tags = goal_tags
		self.goal_tag_ids \
			= [self.tag_dictionary.get_idx_for_item(t) for t in goal_tags]
		self.selection_tensor = torch.tensor(self.goal_tag_ids)

	def foo(self, sentence: List[str]):
		return self.predictprobs(Sentence(' '.join(sentence)))

	def predictprobs(self, sentences: Union[List[Sentence], Sentence], mini_batch_size=32) -> List[
		Sentence]:

		if type(sentences) is Sentence:
			sentences = [sentences]

		# remove previous embeddings
		clear_embeddings(sentences)

		# make mini-batches
		batches = [sentences[x:x + mini_batch_size] for x in
				   range(0, len(sentences), mini_batch_size)]

		for batch in batches:
			tag_seq, all_probabilities = self._predict_probs_batch(batch)
			predicted_id = tag_seq
			all_tokens = []
			for sentence in batch:
				all_tokens.extend(sentence.tokens)

			for (token, pred_id, probs) in zip(all_tokens, predicted_id,
											   all_probabilities):
				token: Token = token
				# get the predicted tag
				predicted_tag = self.goal_tags[pred_id]
				# self.tag_dictionary.get_item_for_index(pred_id)
				token.add_tag(self.tag_type, predicted_tag)
				token.tag_prob = probs.data[pred_id].item()

		return sentences

	def _predict_probs_batch(self, sentences: List[Sentence]):
		all_feats, tags = self.forward(sentences)

		all_tags_seqs = []
		all_probabilities = []

		for feats in all_feats:
			assert not self.use_crf

			feats_select = torch.index_select(feats, 1, self.selection_tensor)
			probs = torch.softmax(feats_select, 1).cpu()
			_, tag_seq = torch.max(feats_select, 1)
			tag_seq = list(tag_seq.cpu().data)

			# overall_score += score
			all_tags_seqs.extend(tag_seq)
			all_probabilities.extend(probs)

		return all_tags_seqs, all_probabilities

	@classmethod
	def load_from_file(cls, model_file):

		warnings.filterwarnings("ignore")
		state = torch.load(model_file, map_location={'cuda:0': 'cpu'})
		warnings.filterwarnings("default")

		model = SequenceTaggerWithProbs(
			hidden_size=state['hidden_size'],
			embeddings=state['embeddings'],
			tag_dictionary=state['tag_dictionary'],
			tag_type=state['tag_type'],
			use_crf=state['use_crf'],
			use_rnn=state['use_rnn'],
			rnn_layers=state['rnn_layers'])

		model.load_state_dict(state['state_dict'])
		model.eval()
		if torch.cuda.is_available():
			model = model.cuda()
		return model


def pruning_training(pruningdir):
	logging.info("Starting pruning training")
	from flair.data_fetcher import NLPTaskDataFetcher
	columns = {0: 'text', 1: 'ob', 2: 'cb'}
	corpus = NLPTaskDataFetcher.fetch_column_corpus(pruningdir, columns,
			train_file='train.txt', dev_file="dev.txt", test_file="test.txt")
	tag_type = 'ob'
	logging.info("Reading corpus from %s" % pruningdir)
	tag_dict = corpus.make_tag_dictionary(tag_type=tag_type)

	logging.info(str(tag_dict.idx2item))

	from flair.embeddings import TokenEmbeddings, WordEmbeddings, \
		StackedEmbeddings, CharLMEmbeddings
	embedding_types: List[TokenEmbeddings] = [
		# WordEmbeddings('de-fasttext'),
		CharLMEmbeddings('german-forward'),
		CharLMEmbeddings('german-backward')
	]

	embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

	from flair.models import SequenceTagger

	tagger: SequenceTaggerWithProbs = SequenceTaggerWithProbs(hidden_size=256,
		embeddings=embeddings, tag_dictionary=tag_dict, tag_type=tag_type,
		use_crf=False)

	from flair.trainers import SequenceTaggerTrainer

	trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger,
														corpus,
														test_mode=True)

	trainer.train(pruningdir, learning_rate=0.1, mini_batch_size=32,
			max_epochs=150)
	tagger.set_goal_tags(["True", "False"])
	return tagger


def loadmodel(pruningdir):
	tagger = SequenceTaggerWithProbs.load_from_file(os.path.join(pruningdir, "best-model.pt"))
	tagger.set_goal_tags(["True", "False"])
	return tagger
