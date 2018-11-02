"""Computation of outside estimates for best-first or A* parsing.

- PCFG A* estimate (Klein & Manning 2003).
  http://aclweb.org/anthology/N03-1016
- PLCFRS LR context-summary estimate (Kallmeyer & Maier 2010).
  http://aclweb.org/anthology/C10-1061

The latter ported almost directly from rparse
(except for sign reversal of log probs)."""

from __future__ import print_function
from math import exp
import numpy as np
from .util import PyAgenda

from cython.operator cimport dereference
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libc.math cimport HUGE_VAL as INFINITY
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from .bit cimport nextset, nextunset, bitcount, bitlength, testbit
from .containers cimport (SmallChartItemAgenda, Agenda, Chart, Grammar,
		Label, Prob, ProbRule, LexicalRule, SmallChartItem, StringIntDict)

# from libc.math cimport isnan, isfinite  # conflicts with C++ std::isfinite
cdef extern from "<cmath>" namespace "std" nogil:
	bint isfinite(double v)
	bint isnan(double v)

include "constants.pxi"


cdef class Item:
	"""Item class used in agenda for computing the outside LR estimate."""
	cdef int state, length, lr, gaps

	def __hash__(Item self):
		return (self.state * 1021
				+ self.length * 571
				+ self.lr * 311
				+ self.gaps)

	def __richcmp__(Item self, Item other, op):
		if op == 2:
			return (self.state == other.state and self.length == other.length
					and self.lr == other.lr and self.gaps == other.gaps)
		a = (self.state, self.length, self.lr, self.gaps)
		b = (other.state, other.length, other.lr, other.gaps)
		if op == 0:
			return a < b
		elif op == 1:
			return a <= b
		elif op == 3:
			return a != b
		elif op == 4:
			return a > b
		elif op == 5:
			return a >= b

	def __repr__(Item self):
		return "%s(%4d, %2d, %2d, %2d)" % (self.__class__.__name__,
				self.state, self.length, self.lr, self.gaps)


cdef inline Item new_Item(int state, int length, int lr, int gaps):
	cdef Item item = Item.__new__(Item)
	item.state = state
	item.length = length
	item.lr = lr
	item.gaps = gaps
	return item


cdef inline double getoutside(double [:, :, :, :] outside,
		uint32_t maxlen, uint32_t slen, Label label, uint64_t vec):
	"""Query for outside estimate.

	NB: if this would be used, it should be in a .pxd with `inline`.
	However, passing the numpy array may be slow."""
	cdef int length = bitcount(vec)
	cdef int left = nextset(vec, 0)
	cdef int gaps = bitlength(vec) - length - left
	cdef int right = slen - length - left - gaps
	cdef int lr = left + right
	if slen > maxlen or length + lr + gaps > maxlen:
		return 0.0
	return outside[label, length, lr, gaps]


def simpleinside(Grammar grammar, uint32_t maxlen, double [:, :] insidescores):
	"""Compute simple inside estimate in bottom-up fashion.
	Here vec is actually the length (number of terminals in the yield of
	the constituent)
	insidescores is a 2-dimensional matrix initialized with NaN to indicate
	values that have yet to be computed."""
	cdef SmallChartItemAgenda[double] agenda
	cdef pair[SmallChartItem, double] entry
	cdef SmallChartItem I
	cdef double x
	cdef ProbRule rule
	cdef size_t i
	cdef uint64_t vec
	cdef LexicalRule lexrule
	for lexrule in grammar.lexical:
		agenda.setifbetter(SmallChartItem(lexrule.lhs, 1), lexrule.prob)
	while not agenda.empty():
		entry = agenda.pop()
		I = entry.first
		x = entry.second

		# This comparison catches the case when insidescores has a higher
		# value than x, but also when it is NaN, because all comparisons
		# against NaN are false.
		# if not x >= insidescores[I.label, I.vec]:
		# Mory explicitly:
		if (isnan(insidescores[I.label, I.vec])
				or x < insidescores[I.label, I.vec]):
			insidescores[I.label, I.vec] = x

		for i in range(grammar.nonterminals):
			rule = grammar.unary[I.label][i]
			if rule.rhs1 != I.label:
				break
			elif isnan(insidescores[rule.lhs, I.vec]):
				agenda.setifbetter(
						SmallChartItem(rule.lhs, I.vec), rule.prob + x)

		for i in range(grammar.nonterminals):
			rule = grammar.lbinary[I.label][i]
			if rule.rhs1 != I.label:
				break
			for vec in range(1, maxlen - I.vec + 1):
				if (isfinite(insidescores[rule.rhs2, vec])
						and isnan(insidescores[rule.lhs, I.vec + vec])):
					agenda.setifbetter(
							SmallChartItem(rule.lhs, I.vec + vec),
							rule.prob + x + insidescores[rule.rhs2, vec])

		for i in range(grammar.nonterminals):
			rule = grammar.rbinary[I.label][i]
			if rule.rhs2 != I.label:
				break
			for vec in range(1, maxlen - I.vec + 1):
				if (isfinite(insidescores[rule.rhs1, vec])
						and isnan(insidescores[rule.lhs, vec + I.vec])):
					agenda.setifbetter(
							SmallChartItem(rule.lhs, vec + I.vec),
							rule.prob + insidescores[rule.rhs1, vec] + x)

	# anything not reached so far is still NaN and gets probability zero:
	insidescores.base[np.isnan(insidescores.base)] = np.inf


def inside(Grammar grammar, uint32_t maxlen, dict insidescores):
	"""Compute inside estimate in bottom-up fashion, with full bit vectors.

	(not used)."""
	cdef SmallChartItem I
	cdef SmallChartItemAgenda[double] agenda
	cdef pair[SmallChartItem, double] entry
	cdef size_t i
	cdef LexicalRule lexrule
	for lexrule in grammar.lexical:
		agenda.setifbetter(SmallChartItem(lexrule.lhs, 1), lexrule.prob)
	while not agenda.empty():
		entry = agenda.pop()
		I = entry.first
		x = entry.second
		if I.label not in insidescores:
			insidescores[I.label] = {}
		if x < insidescores[I.label].get(I.vec, INFINITY):
			insidescores[I.label][I.vec] = x

		for i in range(grammar.nonterminals):
			rule = grammar.unary[I.label][i]
			if rule.rhs1 != I.label:
				break
			elif (rule.lhs not in insidescores
					or I.vec not in insidescores[rule.lhs]):
				agenda.setifbetter(
						SmallChartItem(rule.lhs, I.vec), rule.prob + x)

		for i in range(grammar.nonterminals):
			rule = grammar.lbinary[I.label][i]
			if rule.rhs1 != I.label:
				break
			elif rule.rhs2 not in insidescores:
				continue
			for vec in insidescores[rule.rhs2]:
				left = insideconcat(I.vec, vec, rule, grammar, maxlen)
				if left and (rule.lhs not in insidescores
						or left not in insidescores[rule.lhs]):
					agenda.setifbetter(SmallChartItem(rule.lhs, left),
							rule.prob + x + insidescores[rule.rhs2][vec])

		for i in range(grammar.nonterminals):
			rule = grammar.rbinary[I.label][i]
			if rule.rhs2 != I.label:
				break
			elif rule.rhs1 not in insidescores:
				continue
			for vec in insidescores[rule.rhs1]:
				right = insideconcat(vec, I.vec, rule, grammar, maxlen)
				if right and (rule.lhs not in insidescores
						or right not in insidescores[rule.lhs]):
					agenda.setifbetter(SmallChartItem(rule.lhs, right),
							rule.prob + insidescores[rule.rhs1][vec] + x)

	return insidescores


cdef inline uint64_t insideconcat(uint64_t a, uint64_t b, ProbRule rule,
		Grammar grammar, uint32_t maxlen):
	cdef int subarg, resultpos
	cdef uint64_t result
	if grammar.fanout[rule.lhs] + bitcount(a) + bitcount(b) > maxlen + 1:
		return 0
	result = resultpos = l = r = 0
	for x in range(bitlength(rule.lengths)):
		if testbit(rule.args, x) == 0:
			subarg = nextunset(a, l) - l
			result |= (1UL << subarg) - 1UL << resultpos
			resultpos += subarg
			l = subarg + 1
		else:
			subarg = nextunset(b, r) - r
			result |= (1UL << subarg) - 1UL << resultpos
			resultpos += subarg
			r = subarg + 1
		if testbit(rule.lengths, x):
			resultpos += 1
			result &= ~(1UL << resultpos)
	return result


def outsidelr(Grammar grammar, double [:, :] insidescores,
		uint32_t maxlen, Label goal, double [:, :, :, :] outside):
	"""Compute the outside SX simple LR estimate in top down fashion."""
	cdef Item I
	cdef ProbRule rule
	cdef double x, insidescore, current, score
	cdef int n, totlen, addgaps, addright, leftfanout, rightfanout
	cdef int lenA, lenB, lr, ga
	cdef size_t i
	cdef bint stopaddleft, stopaddright
	agenda = PyAgenda()

	for n in range(1, maxlen + 1):
		agenda[new_Item(goal, n, 0, 0)] = 0.0
		outside[goal, n, 0, 0] = 0.0
	print("initialized")

	while agenda:
		I, x = agenda.popitem()
		if len(agenda) % 10000 == 0:
			print('agenda size: %dk top: %r, %g %s' % (
					len(agenda) / 1000, I, exp(-x),
					grammar.tolabel[I.state]))
		totlen = I.length + I.lr + I.gaps
		i = 0
		rule = grammar.bylhs[I.state][i]
		while rule.lhs == I.state:
			# X -> A
			if rule.rhs2 == 0:
				score = rule.prob + x
				if score < outside[rule.rhs1, I.length, I.lr, I.gaps]:
					agenda[new_Item(rule.rhs1, I.length, I.lr, I.gaps)
							] = score
					outside[rule.rhs1, I.length, I.lr, I.gaps] = score
				i += 1
				rule = grammar.bylhs[I.state][i]
				continue
			# X -> A B
			addgaps = addright = 0
			stopaddright = False
			for n in range(bitlength(rule.lengths) - 1, -1, -1):
				if testbit(rule.args, n):
					if stopaddright:
						addgaps += 1
					else:
						addright += 1
				elif not stopaddright:
					stopaddright = True

			leftfanout = grammar.fanout[rule.rhs1]
			rightfanout = grammar.fanout[rule.rhs2]

			# binary-left (A is left)
			for lenA in range(leftfanout, I.length - rightfanout + 1):
				lenB = I.length - lenA
				insidescore = insidescores[rule.rhs2, lenB]
				for lr in range(I.lr, I.lr + lenB + 2):  # FIXME: why 2?
					if addright == 0 and lr != I.lr:
						continue
					for ga in range(leftfanout - 1, totlen + 1):
						if (lenA + lr + ga == I.length + I.lr + I.gaps
								and ga >= addgaps):
							score = rule.prob + x + insidescore
							current = outside[rule.rhs1, lenA, lr, ga]
							if score < current:
								agenda[new_Item(rule.rhs1, lenA, lr, ga)
										] = score
								outside[rule.rhs1, lenA, lr, ga] = score

			# X -> B A
			addgaps = addright = 0
			stopaddleft = False
			for n in range(bitlength(rule.lengths)):
				if not stopaddleft and testbit(rule.args, n):
					stopaddleft = True
				if not testbit(rule.args, n):
					if stopaddleft:
						addgaps += 1

			stopaddright = False
			for n in range(bitlength(rule.lengths) - 1, -1, -1):
				if not stopaddright and testbit(rule.args, n):
					stopaddright = True
				if not testbit(rule.args, n) and not stopaddright:
					addright += 1
			addgaps -= addright

			# binary-right (A is right)
			for lenA in range(rightfanout, I.length - leftfanout + 1):
				lenB = I.length - lenA
				insidescore = insidescores[rule.rhs1, lenB]
				for lr in range(I.lr, I.lr + lenB + 2):  # FIXME: why 2?
					for ga in range(rightfanout - 1, totlen + 1):
						if (lenA + lr + ga == I.length + I.lr + I.gaps
								and ga >= addgaps):
							score = rule.prob + insidescore + x
							current = outside[rule.rhs2, lenA, lr, ga]
							if score < current:
								agenda[new_Item(rule.rhs2, lenA, lr, ga)
										] = score
								outside[rule.rhs2, lenA, lr, ga] = score
			i += 1
			rule = grammar.bylhs[I.state][i]
		# end while rule.lhs == I.state:
	# end while agenda:


def getestimates(Grammar grammar, uint32_t maxlen, str rootlabel):
	"""Compute table of outside SX simple LR estimates for a PLCFRS."""
	cdef Label goal = grammar.toid[rootlabel]
	print("allocating outside matrix:",
		(8 * grammar.nonterminals * (maxlen + 1) * (maxlen + 1)
			* (maxlen + 1) / 1024 ** 2), 'MB')
	insidescores = np.empty((grammar.nonterminals, (maxlen + 1)), dtype='d')
	outside = np.empty((grammar.nonterminals, ) + 3 * (maxlen + 1, ), dtype='d')
	insidescores[...] = np.NAN
	outside[...] = np.inf
	print("getting inside estimates")
	simpleinside(grammar, maxlen, insidescores)
	print("getting outside estimates")
	outsidelr(grammar, insidescores, maxlen, goal, outside)
	return outside


cdef inline double getpcfgoutside(dict outsidescores,
		uint32_t maxlen, uint32_t slen, Label label, uint64_t vec):
	"""Query for a PCFG A* estimate. For documentation purposes."""
	cdef int length = bitcount(vec)
	cdef int left = nextset(vec, 0)
	cdef int right = slen - length - left
	if slen > maxlen or length + left + right > maxlen:
		return 0.0
	return outsidescores[label, left, right]


cpdef getpcfgestimates(Grammar grammar, uint32_t maxlen, str rootlabel,
		bint debug=False):
	"""Compute table of outside SX estimates for a PCFG."""
	cdef Label goal = grammar.toid[rootlabel]
	insidescores = pcfginsidesx(grammar, maxlen)
	outside = pcfgoutsidesx(grammar, insidescores, goal, maxlen)
	if debug:
		print('inside:')
		for span in range(1, maxlen + 1):
			for k, v in sorted(insidescores[span].items()):
				if v < INFINITY:
					print("%s[%d] %g" % (grammar.tolabel[k],
							span, exp(-v)))
		print('infinite:', end=' ')
		for span in range(1, maxlen + 1):
			for k, v in sorted(insidescores[span].items()):
				if v == INFINITY:
					print("%s[%d]" % (grammar.tolabel[k],
							span), end=' ')
		print('\n\noutside:')
		for lspan in range(maxlen + 1):
			for rspan in range(maxlen - lspan + 1):
				for lhs in range(grammar.nonterminals):
					if outside[lhs, lspan, rspan] < INFINITY:
						print("%s[%d-%d] %g" % (
								grammar.tolabel[lhs], lspan,
								rspan, exp(-outside[lhs, lspan, rspan])))
	return outside


cdef pcfginsidesx(Grammar grammar, uint32_t maxlen):
	"""Compute insideSX estimate for a PCFG using agenda.

	Adapted from: Klein & Manning (2003), A* parsing: Fast Exact Viterbi Parse
	Selection."""
	cdef uint64_t vec
	cdef SmallChartItem I
	cdef SmallChartItemAgenda[double] agenda
	cdef pair[SmallChartItem, double] entry
	cdef ProbRule rule
	cdef double x
	cdef list insidescores = [{} for _ in range(maxlen + 1)]
	cdef LexicalRule lexrule
	for lexrule in grammar.lexical:
		agenda.setifbetter(SmallChartItem(lexrule.lhs, 1), lexrule.prob)
	while not agenda.empty():
		entry = agenda.pop()
		I = entry.first
		x = entry.second
		if (I.label not in insidescores[I.vec]
				or x < insidescores[I.vec][I.label]):
			insidescores[I.vec][I.label] = x

		for i in range(grammar.nonterminals):
			rule = grammar.unary[I.label][i]
			if rule.rhs1 != I.label:
				break
			elif rule.lhs not in insidescores[I.vec]:
				agenda.setifbetter(
						SmallChartItem(rule.lhs, I.vec), rule.prob + x)

		for i in range(grammar.nonterminals):
			rule = grammar.lbinary[I.label][i]
			if rule.rhs1 != I.label:
				break
			for vec in range(1, maxlen - I.vec + 1):
				if (rule.rhs2 in insidescores[vec]
						and rule.lhs not in insidescores[I.vec + vec]):
					agenda.setifbetter(
							SmallChartItem(rule.lhs, I.vec + vec),
							rule.prob + x + insidescores[vec][rule.rhs2])

		for i in range(grammar.nonterminals):
			rule = grammar.rbinary[I.label][i]
			if rule.rhs2 != I.label:
				break
			for vec in range(1, maxlen - I.vec + 1):
				if (rule.rhs1 in insidescores[vec]
						and rule.lhs not in insidescores[vec + I.vec]):
					agenda.setifbetter(
							SmallChartItem(rule.lhs, vec + I.vec),
							rule.prob + insidescores[vec][rule.rhs1] + x)
	return insidescores


cdef pcfgoutsidesx(Grammar grammar, list insidescores, Label goal,
		uint32_t maxlen):
	"""outsideSX estimate for a PCFG, agenda-based version."""
	cdef tuple I
	cdef ProbRule rule
	cdef double x, insidescore, current, score
	cdef int state, left, right
	cdef size_t i, sibsize
	cdef double [:, :, :, :] outside = np.empty(
			(grammar.nonterminals, maxlen + 1, maxlen + 1, 1), dtype='d')
	outside[...] = np.inf

	agenda = PyAgenda()
	agenda[goal, 0, 0] = outside[goal, 0, 0, 0] = 0.0
	while agenda:
		I, x = agenda.popitem()
		state, left, right = I
		if len(agenda) % 10000 == 0:
			print('agenda size: %dk top: %r, %g %s' % (
					len(agenda) / 1000, I, exp(-x),
					grammar.tolabel[state]))
		i = 0
		rule = grammar.bylhs[state][i]
		while rule.lhs == state:
			# X -> A
			if rule.rhs2 == 0:
				score = rule.prob + x
				if score < outside[rule.rhs1, left, right, 0]:
					agenda[(rule.rhs1, left, right)] = score
					outside[rule.rhs1, left, right, 0] = score
				i += 1
				rule = grammar.bylhs[state][i]
				continue

			# item is on the left: X -> A B.
			for sibsize in range(1, maxlen - left - right):
				insidescore = insidescores[sibsize].get(rule.rhs2, INFINITY)
				score = rule.prob + x + insidescore
				current = outside[rule.rhs1, left, right + sibsize, 0]
				if score < current:
					agenda[(rule.rhs1, left, right + sibsize)] = score
					outside[rule.rhs1, left, right + sibsize, 0] = score

			# item is on the right: X -> B A
			for sibsize in range(1, maxlen - left - right):
				insidescore = insidescores[sibsize].get(rule.rhs1, INFINITY)
				score = rule.prob + insidescore + x
				current = outside[rule.rhs2, left + sibsize, right, 0]
				if score < current:
					agenda[(rule.rhs2, left + sibsize, right)] = score
					outside[rule.rhs2, left + sibsize, right, 0] = score

			i += 1
			rule = grammar.bylhs[state][i]
		# end while rule.lhs == state:
	# end while agenda:
	return outside.base


cpdef getpcfgestimatesrec(Grammar grammar, uint32_t maxlen, Label goal,
	bint debug=False):
	insidescores = [{} for _ in range(maxlen + 1)]
	outsidescores = {}
	for span in range(1, maxlen + 1):
		insidescores[span][goal] = pcfginsidesxrec(
				grammar, insidescores, goal, span)
	for lspan in range(maxlen + 1):
		for rspan in range(maxlen - lspan + 1):
			for lhs in grammar.lexicallhs:
				if (lhs, lspan, rspan) in outsidescores:
					continue
				outsidescores[lhs, lspan, rspan] = pcfgoutsidesxrec(grammar,
						insidescores, outsidescores, goal, lhs, lspan, rspan)
	if debug:
		print('inside:')
		for span in range(1, maxlen + 1):
			for k, v in sorted(insidescores[span].items()):
				if v < INFINITY:
					print("%s[%d] %g" % (
							grammar.tolabel[k], span, exp(-v)))
		print('infinite:', end=' ')
		for span in range(1, maxlen + 1):
			for k, v in sorted(insidescores[span].items()):
				if v == INFINITY:
					print("%s[%d]" % (
							grammar.tolabel[k], span), end=' ')

		print('\n\noutside:')
		for k, v in sorted(outsidescores.items()):
			if v < INFINITY:
				print("%s[%d-%d] %g" % (
						grammar.tolabel[k[0]], k[1], k[2],
						exp(-v)))
		print('infinite:', end=' ')
		for k, v in sorted(outsidescores.items()):
			if v == INFINITY:
				print("%s[%d-%d]" % (
						grammar.tolabel[k[0]], k[1], k[2]),
						end=' ')
	outside = np.empty((grammar.nonterminals, maxlen + 1, maxlen + 1, 1),
			dtype='d')
	outside[...] = np.inf
	# convert sparse dictionary to dense numpy array
	for (state, lspan, rspan), prob in outsidescores.items():
		outside[state, lspan, rspan, 0] = prob
	return outside


cdef pcfginsidesxrec(Grammar grammar, list insidescores, Label state,
		int span):
	"""Compute insideSX estimate for a PCFG.

	Straight from Klein & Manning (2003), A* parsing: Fast Exact Viterbi Parse
	Selection."""
	# NB: does not deal correctly with unary rules.
	cdef size_t n, split
	cdef ProbRule rule
	if span == 0:
		return 0 if state == 0 else INFINITY
	it = grammar.lexicallhs.find(state)
	if span == 1 and it != grammar.lexicallhs.end():
		score = min([lexrule.prob for lexrule in grammar.lexical
				if lexrule.lhs == state])  # FIXME: slow
	else:
		score = INFINITY
	for split in range(1, span + 1):
		n = 0
		rule = grammar.bylhs[state][n]
		while rule.lhs == state:
			if rule.rhs1 in insidescores[split]:
				inleft = insidescores[split][rule.rhs1]
				if inleft == -1:
					n += 1
					rule = grammar.bylhs[state][n]
			else:
				insidescores[split][rule.rhs1] = -1  # mark to avoid cycles.
				inleft = pcfginsidesxrec(
						grammar, insidescores, rule.rhs1, split)
				insidescores[split][rule.rhs1] = inleft
			if rule.rhs2 == 0:
				if split == span:
					inright = 0
				else:
					n += 1
					rule = grammar.bylhs[state][n]
					continue
			elif rule.rhs2 in insidescores[span - split]:
				inright = insidescores[span - split][rule.rhs2]
			else:
				inright = pcfginsidesxrec(
						grammar, insidescores, rule.rhs2, span - split)
				insidescores[span - split][rule.rhs2] = inright
			cost = inleft + inright + rule.prob
			if cost < score:
				score = cost
			n += 1
			rule = grammar.bylhs[state][n]
	return score


cdef pcfgoutsidesxrec(Grammar grammar, list insidescores, dict outsidescores,
		Label goal, Label state, int lspan, int rspan):
	"""outsideSX estimate for a PCFG."""
	# NB: does not deal correctly with unary rules.
	cdef size_t n, sibsize
	cdef ProbRule rule
	cdef tuple item
	if lspan + rspan == 0:
		return 0 if state == goal else INFINITY
	score = INFINITY
	# unary productions: no sibling
	n = 0
	rule = grammar.unary[state][n]
	while rule.rhs1 == state:
		item = (rule.lhs, lspan, rspan)
		if item in outsidescores:
			out = outsidescores[item]
			if out == -1:
				n += 1
				rule = grammar.unary[state][n]
				continue
		else:
			outsidescores[item] = -1  # mark to avoid cycles
			outsidescores[item] = out = pcfgoutsidesxrec(
					grammar, insidescores, outsidescores, goal,
					rule.lhs, lspan, rspan)
		cost = out + rule.prob
		if cost < score:
			score = cost
		n += 1
		rule = grammar.unary[state][n]
	# could have a left sibling
	for sibsize in range(1, lspan + 1):
		n = 0
		rule = grammar.rbinary[state][n]
		while rule.rhs2 == state:
			item = (rule.lhs, lspan - sibsize, rspan)
			if item in outsidescores:
				out = outsidescores[item]
			else:
				outsidescores[item] = out = pcfgoutsidesxrec(
						grammar, insidescores, outsidescores, goal,
						rule.lhs, lspan - sibsize, rspan)
			cost = (insidescores[sibsize].get(rule.rhs1, INFINITY)
					+ out + rule.prob)
			if cost < score:
				score = cost
			n += 1
			rule = grammar.rbinary[state][n]
	# could have a right sibling
	for sibsize in range(1, rspan + 1):
		n = 0
		rule = grammar.lbinary[state][n]
		while rule.rhs1 == state:
			item = (rule.lhs, lspan, rspan - sibsize)
			if item in outsidescores:
				out = outsidescores[item]
			else:
				out = pcfgoutsidesxrec(
						grammar, insidescores, outsidescores, goal,
						rule.lhs, lspan, rspan - sibsize)
				outsidescores[item] = out
			cost = (insidescores[sibsize].get(rule.rhs2, INFINITY)
					+ out + rule.prob)
			if cost < score:
				score = cost
			n += 1
			rule = grammar.lbinary[state][n]
	return score


cpdef posboundaryestimates(trees, Grammar grammar):
	cdef:
		# double[: , :] leftboundary, rightboundary
		short END
		short tagidx = 0
		StringIntDict tagids = StringIntDict()
		vector[short] posconversion = []

	for tagidx, tag in enumerate(grammar.getpos()):
		tagids.ob[tag.encode('utf8')] = tagidx
		posconversion.push_back(grammar.toid[tag])

	tagids.ob['START'.encode('utf8')] = tagidx + 1
	tagids.ob['END'.encode('utf8')] = END = tagidx + 2

	leftboundary = np.zeros((grammar.nonterminals, END + 1), dtype='d')
	rightboundary = np.zeros((END + 1, grammar.nonterminals), dtype='d')

	for tree in trees:
		for t in tree.subtrees():
			mi = min(t.leaves()) - 1
			ma = max(t.leaves()) + 1
			pl = tree.pos()[mi][1].encode('utf8') if mi >= 0 else 'START'.encode('utf8')
			pr = tree.pos()[ma][1].encode('utf8') if ma < len(tree.leaves()) else 'END'.encode('utf8')
			leftboundary[grammar.toid[t.label], tagids.ob[pl]] += 1
			rightboundary[tagids.ob[pr], grammar.toid[t.label]] += 1

	# normalization
	rowsums = leftboundary.sum(axis=0)
	leftboundary = np.nan_to_num(leftboundary / rowsums[np.newaxis, :])

	rowsums = rightboundary.sum(axis=0)
	rightboundary = np.nan_to_num(rightboundary / rowsums[np.newaxis, :])
	return posconversion, leftboundary, rightboundary


cpdef testestimates(Grammar grammar, uint32_t maxlen, str rootlabel):
	cdef Label goal = grammar.toid[rootlabel]
	print("getting inside")
	insidescores = inside(grammar, maxlen, {})
	for a in insidescores:
		for b in insidescores[a]:
			assert 0 <= a < grammar.nonterminals
			assert 0 <= bitlength(b) <= maxlen
			# print(a, b)
			# print("%s[%d] =" % (grammar.tolabel[a], b),
			# 		exp(insidescores[a][b]))
	print(len(insidescores) * sum(map(len, insidescores.values())), '\n')
	insidescores = np.empty((grammar.nonterminals, (maxlen + 1)), dtype='d')
	insidescores[...] = np.NAN
	simpleinside(grammar, maxlen, insidescores)
	insidescores[np.isnan(insidescores)] = np.inf
	print("inside")
	for an, a in enumerate(insidescores):
		for bn, b in enumerate(a):
			if b < np.inf:
				print("%s len %d = %g" % (grammar.tolabel[an],
						bn, exp(-b)))
	# print(insidescores)
	# for a in range(maxlen):
	# 	print(grammar.tolabel[goal], "len", a, "=",
	# 			exp(-insidescores[goal, a]))

	print("getting outside")
	outside = np.empty((grammar.nonterminals, ) + 3 * (maxlen + 1, ), dtype='d')
	outside[...] = np.inf
	outsidelr(grammar, insidescores, maxlen, goal, outside)
	# print(outside)
	cnt = 0
	for an, a in enumerate(outside):
		for bn, b in enumerate(a):
			for cn, c in enumerate(b):
				for dn, d in enumerate(c):
					if d < np.inf:
						print("%s length %d lr %d gaps %d = %g" % (
								grammar.tolabel[an],
								bn, cn, dn, exp(-d)))
						cnt += 1
	print(cnt)
	print("done")
	return outside


def test():
	cdef Chart chart, estchart
	from . import plcfrs
	from .grammar import treebankgrammar
	from .containers import Grammar
	from .tree import Tree
	from .treebank import NegraCorpusReader
	from .treetransforms import addfanoutmarkers, binarize
	corpus = NegraCorpusReader('alpinosample.export')
	trees = list(corpus.trees().values())
	for a in trees:
		binarize(a, vertmarkov=1, horzmarkov=1)
		addfanoutmarkers(a)
	grammar = Grammar(treebankgrammar(trees, list(corpus.sents().values())))
	trees = [Tree('(ROOT (A (a 0) (b 1)))'),
			Tree('(ROOT (B (a 0) (c 2)) (b 1))'),
			Tree('(ROOT (B (a 0) (c 2)) (b 1))'),
			Tree('(ROOT (C (a 0) (c 2)) (b 1))')]
	sents =[["a", "b"],
			["a", "b", "c"],
			["a", "b", "c"],
			["a", "b", "c"]]
	print("treebank:")
	for a in trees:
		print(a)
	print("\ngrammar:")
	grammar = Grammar(treebankgrammar(trees, sents))
	print(grammar, '\n')
	testestimates(grammar, 4, 'ROOT')
	outside = getestimates(grammar, 4, 'ROOT')
	sent = ["a", "b", "c"]
	print("\nwithout estimates")
	chart, msg = plcfrs.parse(sent, grammar, estimates=None)
	print(msg)
	print(chart)
	print("\nwith estimates")
	estchart, msg = plcfrs.parse(sent, grammar,
			estimates=('SXlrgaps', outside))
	print(msg)
	print(estchart)
	print('items avoided:', chart.numitems() - estchart.numitems())

	trees = [Tree(a) for a in (
			'(ROOT (A (a 0) (b 1)))\n'
			'(ROOT (A (B (A (B (a 0) (b 1))))) (c 2))\n'
			'(ROOT (A (B (A (B (a 0) (b 1))))) (c 2))\n'
			'(ROOT (A (B (A (B (a 0) (b 1))))) (c 2))\n'
			'(ROOT (A (B (A (B (a 0) (b 1))))) (c 2))\n'
			'(ROOT (A (B (A (B (a 0) (b 1))))) (c 2))\n'
			'(ROOT (C (a 0) (b 1)) (c 2))\n').splitlines()]
	sents =[["a", "b"],
			["a", "b", "c"],
			["a", "b", "c"],
			["a", "b", "c"],
			["a", "b", "c"],
			["a", "b", "c"],
			["a", "b", "c"]]
	print("treebank:")
	for a in trees:
		print(a)
	print("\npcfg grammar:")
	grammar = Grammar(treebankgrammar(trees, sents))
	print(grammar, '\n')
	outside = getpcfgestimates(grammar, 4, 'ROOT', debug=True)
	sent = ["a", "b", "c"]
	print("\nwithout estimates")
	chart, msg = plcfrs.parse(sent, grammar, estimates=None)
	print(msg)
	print(chart)
	print("\nwith estimates")
	estchart, msg = plcfrs.parse(sent, grammar, estimates=('SX', outside))
	print(msg)
	print(estchart)
	print('items avoided:')
	print('items avoided:', chart.numitems() - estchart.numitems())

__all__ = ['Item', 'getestimates', 'getpcfgestimates', 'inside', 'outsidelr',
		'simpleinside']
