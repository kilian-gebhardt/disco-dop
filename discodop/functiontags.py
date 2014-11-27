"""Function tags classifier."""
from discodop.tree import Tree
from discodop.treebanktransforms import base, function, functions, FUNC
from discodop import heads

FIELDS = tuple(range(8))
WORD, LEMMA, TAG, MORPH, FUNC, PARENT, SECEDGETAG, SECEDGEPARENT = FIELDS


def trainfunctionclassifier(trees, sents, numproc):
	"""Train a classifier to predict functions tags in trees."""
	from sklearn import linear_model, multiclass
	from sklearn import preprocessing, feature_extraction
	vectorizer = feature_extraction.DictVectorizer(sparse=True)
	# PTB has no function tags on pretermintals, Negra etc. do.
	posfunc = any(functions(node) for tree in trees
			for node in tree.subtrees(lambda n: n is not tree and n
				and isinstance(n[0], int)))
	target = [functions(node) for tree in trees
			for node in tree.subtrees(lambda n: n is not tree and n
				and (posfunc or isinstance(n[0], Tree)))]
	# PTB may have multiple tags (or 0) per node.
	# Negra etc. have exactly 1 tag for every node.
	multi = any(len(a) > 1 for a in target)
	if multi:
		encoder = preprocessing.MultiLabelBinarizer()
	else:
		encoder = preprocessing.LabelEncoder()
		target = [a[0] if a else '--' for a in target]
	# binarize features (output is a sparse array)
	trainfeats = vectorizer.fit_transform(functionfeatures(node, sent)
			for tree, sent in zip(trees, sents)
				for node in tree.subtrees(lambda n: n is not tree and n
					and (posfunc or isinstance(n[0], Tree))))
	trainfuncs = encoder.fit_transform(target)
	classifier = linear_model.SGDClassifier(loss='hinge', penalty='elasticnet')
	if multi:
		classifier = multiclass.OneVsRestClassifier(
				classifier, n_jobs=numproc or -1)
	# train classifier
	classifier.fit(trainfeats, trainfuncs)
	msg = ('trained classifier; multi=%r, posfunc=%r; score on training set: '
			'%g %%\nfunction tags: %s' % (multi, posfunc,
			100.0 * sum((a == b).all() for a, b
				in zip(trainfuncs, classifier.predict(trainfeats)))
				/ len(trainfuncs),
			' '.join(str(a) for a in encoder.classes_)))
	return (classifier, vectorizer, encoder, posfunc, multi), msg


def applyfunctionclassifier(funcclassifier, tree, sent):
	"""Add predicted function tags to tree using classifier."""
	classifier, vectorizer, encoder, posfunc, multi = funcclassifier
	# get features and use classifier
	funclabels = encoder.inverse_transform(classifier.predict(
			vectorizer.transform(functionfeatures(node, sent)
			for node in tree.subtrees(lambda n: n is not tree and n
				and (posfunc or isinstance(n[0], Tree))))))
	# store labels in tree
	for node, func in zip(tree.subtrees(lambda n: n is not tree and n
			and (posfunc or isinstance(n[0], Tree))), funclabels):
		if not getattr(node, 'source', None):
			node.source = ['--'] * 8
		elif isinstance(node.source, tuple):
			node.source = list(node.source)
		if not func:
			node.source[FUNC] = '--'
		elif multi:
			node.source[FUNC] = '-'.join(func)
		else:
			node.source[FUNC] = func


def functionfeatures(node, sent):
	"""Return a list of features for node to predict its function tag.

	The node must be a ParentedTree, with head information.

	The features are based on Blaheta & Charniak (2000),
	Assigning Function Tags to Parsed Text."""

	headsib = headsibpos = None
	for sib in node.parent:
		if heads.ishead(sib):
			headsib = sib
			headsibpos = heads.getheadpos(headsib)
			break
	result = {
			# 4. head sister const label
			'hsc': headsib.label if headsib else '',
			# 5. head sister head word POS
			'hsp': headsibpos.label if headsibpos else '',
			# 6. head sister head word
			'hsf': sent[headsibpos[0]] if headsibpos else '',
			# 10. parent label
			'moc': node.parent.label,
			# 11. grandparent label
			'grc': node.parent.parent.label
					if node.parent.parent else '',
			# 12. Offset of this node to head sister
			'ohs': (node.parent_index - headsib.parent_index)
					if headsib is not None else -1,
			}
	result.update(basefeatures(node, sent))
	# add similar features for neighbors
	if node.parent_index > 0:
		result.update(basefeatures(
				node.parent[node.parent_index - 1], sent, prefix='p'))
	if node.parent_index + 1 < len(node.parent):
		result.update(basefeatures(
				node.parent[node.parent_index + 1], sent, prefix='n'))
	return result


def basefeatures(node, sent, prefix=''):
	"""A set features describing this particular node."""
	headpos = heads.getheadpos(node)
	if base(node, 'PP'):
		# NB: we skip the preposition here; need way to identify it.
		altheadpos = heads.getheadpos(node[1:])
	else:
		altheadpos = None
	return {
			# 1. syntactic category
			prefix + 'cat': node.label,
			# 2. head POS
			prefix + 'hwp': headpos and headpos.label,
			# 3. head word
			prefix + 'hwf': headpos and sent[headpos[0]],
			# 7. alt (for PPs, non-prep. node) head POS
			prefix + 'ahc': altheadpos.label if altheadpos else '',
			# 8. alt head word
			prefix + 'ahf': sent[altheadpos[0]] if altheadpos else '',
			# 9 yield length
			prefix + 'yis': len(node.leaves()),
			}


__all__ = ['applyfunctionclassifier', 'trainfunctionclassifier',
		'functionfeatures']