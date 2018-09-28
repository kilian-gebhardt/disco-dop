from .tree import Tree
import logging
from typing import List
from flair.data import Sentence
from flair.models import SequenceTagger
from collections import OrderedDict
import os


def bracketannotation(tree: Tree, sent):
	spans = [t2.leaves() for t2 in tree.subtrees() if len(t2.leaves()) > 1]
	starts = {min(span) for span in spans}
	ends = {max(span) for span in spans}
	return [(w, i in starts, i in ends) for i, w in enumerate(sent)]


def annotation2file(annotation, file):
	file.write('\n'.join("%s\t%s\t%s" % w for w in annotation) + '\n\n')


class PruningMask:
	def __init__(self):
		self.openbracket = []
		self.closebracket = []


def predictpruningmask(testset: OrderedDict, obtagger: SequenceTagger,
			cbtagger: SequenceTagger, obthreshold: float,
			cbthreshold: float):
	obtagger.eval()
	cbtagger.eval()
	pruningmasks: OrderedDict[int, PruningMask] \
		= OrderedDict((n, PruningMask()) for (n, _) in testset.items())
	sentences = OrderedDict((n, Sentence(' '.join([w for w, _ in sent])))
							for (n, (_, _, sent, _)) in testset.items())
	sentences_list = [s for _, s in sentences.items()]
	obtagger.predict(sentences_list)
	cbtagger.predict(sentences_list)
	for n in pruningmasks:
		sent = sentences[n]
		# only block bracket if confidence is above threshold
		pruningmasks[n].openbracket = [t.get_tag(obtagger.tag_type).value == 'True'
				or t.get_tag(obtagger.tag_type).value == 'False'
					and t.get_tag(obtagger.tag_type).score < obthreshold
									for t in sent]
		pruningmasks[n].closebracket = [t.get_tag(cbtagger.tag_type).value == 'True'
				or t.get_tag(cbtagger.tag_type).value == 'False'
					and t.get_tag(cbtagger.tag_type).score < cbthreshold
									for t in sent]
	return pruningmasks


def pruning_training(pruningdir, tag_type, max_epochs=150, use_crf=False):
	logging.info("Starting pruning training")
	from flair.data_fetcher import NLPTaskDataFetcher
	columns = {0: 'text', 1: 'ob', 2: 'cb'}
	corpus = NLPTaskDataFetcher.fetch_column_corpus(pruningdir, columns,
			train_file='train.txt', dev_file="dev.txt", test_file="test.txt")
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

	tagger: SequenceTagger = SequenceTagger(hidden_size=256,
			embeddings=embeddings, tag_dictionary=tag_dict, tag_type=tag_type,
			use_crf=use_crf)

	from flair.trainers import SequenceTaggerTrainer

	trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus,
			test_mode=True)

	trainer.train(os.path.join(pruningdir, tag_type + "model"),
			learning_rate=0.1, mini_batch_size=32, max_epochs=max_epochs, embeddings_in_memory=False)
	# tagger.set_goal_tags(["True", "False"])
	return tagger


def loadmodel(pruningdir, tag_type):
	path = os.path.join(os.path.join(pruningdir, tag_type + "model"),
			"best-model.pt")
	tagger = SequenceTagger.load_from_file(path)
	return tagger