from math import isinf, exp, log, fsum
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcmp
cimport cython

ctypedef unsigned long long ULLong
ctypedef unsigned long ULong
ctypedef unsigned int UInt
ctypedef unsigned char UChar


cdef extern:
	int __builtin_ffsll (ULLong)
	int __builtin_ctzll (ULLong)
	int __builtin_clzll (ULLong)
	int __builtin_ctzl (ULong)
	int __builtin_popcountl (ULong)
	int __builtin_popcountll (ULLong)


cdef extern from "macros.h":
	int BITSIZE
	int BITSLOT(int b)
	ULong BITMASK(int b)
	int BITNSLOTS(int nb)
	void SETBIT(ULong a[], int b)
	ULong TESTBIT(ULong a[], int b)
	#int SLOTS # doesn't work


DEF SLOTS = 3 # FIXME: make this a constant, yet shared across modules.


cdef struct Rule:
	double prob # 8 bytes
	UInt args # 4 bytes => 32 max vars per rule
	UInt lengths # 4 bytes => same
	UInt lhs # 4 bytes
	UInt rhs1 # 4 bytes
	UInt rhs2 # 4 bytes
	UInt no # 4 bytes
	# total: 32 bytes.


@cython.final
cdef class LexicalRule:
	cdef UInt lhs
	cdef unicode word
	cdef double prob


cdef class ChartItem:
	cdef UInt label


@cython.final
cdef class SmallChartItem(ChartItem):
	cdef ULLong vec


@cython.final
cdef class FatChartItem(ChartItem):
	cdef ULong vec[SLOTS]


@cython.final
cdef class CFGChartItem(ChartItem):
	cdef UChar start, end


cdef SmallChartItem CFGtoSmallChartItem(UInt label, UChar start, UChar end)
cdef FatChartItem CFGtoFatChartItem(UInt label, UChar start, UChar end)


cdef class Edge:
	cdef double inside
	cdef Rule *rule


@cython.final
cdef class LCFRSEdge(Edge):
	cdef double score # inside probability + estimate score
	cdef ChartItem left
	cdef ChartItem right


@cython.final
cdef class CFGEdge(Edge):
	cdef UChar mid


@cython.final
cdef class RankedEdge:
	cdef ChartItem head
	cdef LCFRSEdge edge
	cdef int left
	cdef int right


@cython.final
cdef class RankedCFGEdge:
	cdef UInt label
	cdef UChar start, end
	cdef CFGEdge edge
	cdef int left
	cdef int right


# start scratch
#cdef union VecType:
#	ULLong vec
#	ULong *vecptr

#cdef class ParseForest:
#	""" the chart representation of bitpar. seems to require parsing
#	in 3 stages: recognizer, enumerate analyses, get probs. """
#	#keys
#	cdef UInt *catnum			#lhs
#	cdef size_t *firstanalysis	#idx to arrays below.
#	# from firstanalysis[n] to firstanalysis[n+1] or end
#	#values.
#	cdef size_t *firstchild
#	cdef UInt *ruleno
#	#positive means index to lists above, negative means terminal index
#	cdef UInt *child
#
# no, instead of explicitly recording ruleno & mid, record them as part of
# indices / hashes
#cdef struct PackedCFGEdge: # 18 bytes
#	double inside # float here could help squeeze this in 16 bytes
#	UInt ruleno
#	short mid
#
#cdef struct PackedParseForest:
#	PackedCFGEdge *edges
#	size_t *celloffset
#	size_t *labeloffset
#	size_t *edgeoffset
#	ULong numedges
#	#celloffset[left * lensent + right] => idx to edges
#	#labeloffset[left * lensent + right][lhs] => idx to edges
#	#edgeoffset[left * lensent + right][mid][ruleno] => idx to edges
#
#cdef inline setedge(PackedParseForest forest, size_t celloffset,
#		left, right, mid, lhs, ruleno, inside):
#	cdef PackedCFGEdge *newedge
#	if ...: # already in chart
#		newedge = &(forest.edges[celloffset
#				+ forest.edgeoffset[celloffset][mid][ruleno]])
#	else:
#		newedge =
#		forest.edges += 1
#	newedge.ruleno = ruleno
#	newedge.inside = inside
#	newedge.mid = mid
#cdef inline PackedCFGEdge getviterbi(PackedParseForest forest,
#		size_t celloffset, left, right, lhs):
#	return forest.edges[celloffset + forest.labeloffset[celloffset][lhs]]
#cdef inline PackedCFGEdge getedge(PackedParseForest forest, size_t celloffset,
#		left, right, mid, ruleno, inside):
#	return forest.edges[celloffset + forest.edgeoffset[celloffset][mid][ruleno]]
#
#@cython.final
#cdef class ParseForest:
#	""" A packed parse forest represented in contiguous arrays.
#	This only works if for each cell, all rules for each lhs are added
#	in order. """
#	cdef PackedCFGEdge **edges
#	#
#	cdef UInt *rules
#	cdef short *midpoints
#	cdef double *inside
#	# offset[lhs] contains the index where the edges for that label start.
#	cdef size_t *offset
#	# unaries are stored separately:
#	# unaryrules[left * lensent + right][edgeno]
#	# unaryinside[left * lensent + right]edgeno]
#	cdef Rule **unaryrules
#	cdef double **unaryinside
#	#cdef __init__(self, Grammar grammar, short lensent):
#	#	self.lensent = lensent
#	#	self.numrules = grammar.numrules
#	#	# this is for a dense array,
#	#	# start with conservative estimate and then re-alloc along the way?
#	#	self.midpoints = calloc(sizeof(self.midpoints), lensent * lensent)
#	#	self.rules = calloc(sizeof(self.rules), lensent * lensent)
#	#	self.inside = calloc(sizeof(self.inside), lensent * lensent)
#	#	self.offset = calloc(sizeof(self.offset), len(grammar.tolabel))
#	#	self.unaryrules = calloc(sizeof(self.unaryrules), lensent * lensent
#	#			* len(grammar.numunary))
#	#	self.unaryinside = calloc(sizeof(self.unaryinside), lensent * lensent
#	#			* len(grammar.numunary))
#	cdef inline getinside(self, short left, short right, UInt label):
#		cdef size_t dest = self.offsets[label + 1]
#		return self.inside[left * self.lensent + right][self.offset[label]]
#	cdef inline getrule(self, short left, short right, UInt label):
#		return self.rules[left * self.lensent + right][self.offset[label]]
#	cdef inline getmidpoint(self, short left, short right, UInt label):
#		return self.midpoints[left * self.lensent + right][self.offset[label]]
#	cdef inline setitem(self, short left, short right, UInt label,
#			UInt ruleno, short mid, double inside):
#		cdef size_t dest = self.offsets[label + 1]
#		self.rules[dest] = ruleno
#		self.midpoints[dest] = mid
#		self.inside[dest] = inside
#		self.offsets[label + 1] += 1
#	#cdef __dealloc__(self):
#	#	if self.inside is not None:
#	#		free(self.inside)
#	#		self.inside = None
#	#	if self.rules is not None:
#	#		free(self.rules)
#	#		self.rules = None
#	#	if self.midpoints is not None:
#	#		free(self.midpoints)
#	#		self.midpoints = None
#
#cdef class DiscNode:
#	cdef int label
#	cdef tuple children
#	cdef CBitset leaves
# end scratch

# start fragments stuff

cdef struct Node:  # a node of a binary tree
	int prod # non-negative, ID of a phrasal or lexical production
	short left # >= 0: array idx to child Node; <0: idx sent[-left - 1];
	short right # >=0: array idx to child Node; -1: empty (unary Node)


cdef struct NodeArray:  # a tree as an array of Node structs
	size_t offset # index to array of nodes in treebank where this tree starts
	short len, root # number of nodes, index to root node


@cython.final
cdef class Ctrees:
	cdef Node *nodes
	cdef NodeArray *trees
	cdef long nodesleft, max
	cdef readonly size_t numnodes
	cdef readonly short maxnodes
	cdef readonly int len
	cdef list treeswithprod
	cpdef alloc(self, int numtrees, long numnodes)
	cdef realloc(self, int len)
	cpdef add(self, list tree, dict prods)
	cdef addnodes(self, Node *source, int cnt, int root)

# end fragments stuff


# ---------------------------------------------------------------
#                          INLINED FUNCTIONS
# ---------------------------------------------------------------


cdef inline FatChartItem new_FatChartItem(UInt label):
	cdef FatChartItem item = FatChartItem.__new__(FatChartItem)
	item.label = label
	# NB: since item.vec is a static array, its elements are initialized to 0.
	return item


cdef inline SmallChartItem new_ChartItem(UInt label, ULLong vec):
	cdef SmallChartItem item = SmallChartItem.__new__(SmallChartItem)
	item.label = label
	item.vec = vec
	return item


cdef inline CFGChartItem new_CFGChartItem(UInt label, UChar start, UChar end):
	cdef CFGChartItem item = CFGChartItem.__new__(CFGChartItem)
	item.label = label
	item.start = start
	item.end = end
	return item


cdef inline LCFRSEdge new_LCFRSEdge(double score, double inside, Rule *rule,
		ChartItem left, ChartItem right):
	cdef LCFRSEdge edge = LCFRSEdge.__new__(LCFRSEdge)
	cdef long h = 0x345678UL
	edge.score = score
	edge.inside = inside
	edge.rule = rule
	edge.left = left
	edge.right = right
	return edge


cdef inline CFGEdge new_CFGEdge(double inside, Rule *rule, UChar mid):
	cdef CFGEdge edge = CFGEdge.__new__(CFGEdge)
	edge.inside = inside
	edge.rule = rule
	edge.mid = mid
	return edge


cdef object log1e200 = log(1e200)


cdef inline logprobadd(x, y):
	""" Add two log probabilities in log space; i.e.:

	>>> a = b = 0.25
	>>> logprobadd(log(a), log(b)) == log(a + b) == log(0.5)
	True

	:param x, y: Python floats with log probabilities; -inf <= x, y <= 0.
	:source: https://facwiki.cs.byu.edu/nlp/index.php/Log_Domain_Computations
	"""
	if isinf(x):
		return y
	elif isinf(y):
		return x
	# If one value is much smaller than the other, keep the larger value.
	elif x < (y - 460):  # log(1e200)
		return y
	elif y < (x - 460):  # log(1e200)
		return x
	diff = y - x
	assert not isinf(diff)
	if isinf(exp(diff)):	# difference is too large
		return x if x > y else y
	# otherwise return the sum.
	return x + log(1.0 + exp(diff))


cdef inline double logprobsum(list logprobs):
	""" Takes a list of log probabilities and sums them producing a
	normal probability 0 < p <= 1.0; i.e.:

	>>> a = b = c = 0.25
	>>> logprobsum([log(a), log(b), log(c)]) == sum([a, b, c]) == 0.75
	True

	:param logprobs: a list of Python floats with negative log probilities,
		s.t. 0 <= p <= inf for each p in ``logprobs``.
	:source: http://blog.smola.org/post/987977550/log-probabilities-semirings-and-floating-point-numbers

	Comparison of different methods: https://gist.github.com/andreasvc/6204982
	"""
	maxprob = max(logprobs)
	return exp(maxprob) * fsum([exp(prob - maxprob) for prob in logprobs])
