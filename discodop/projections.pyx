from .containers cimport Chart, RankedEdge, Edge, ItemNo, Label, Prob, Position, ProbRule
from .coarsetofine import getinside, getoutside
from libcpp.vector cimport vector
from libc.math cimport HUGE_VAL as INFINITY, exp, log
from sys import exit
from .plcfrs cimport SmallChartItem, SmallLCFRSChart, LCFRSChart_fused
from .tree import Tree
from cython.operator cimport postincrement, dereference
from libc.stdint cimport uint64_t
from .disambiguation import getderivations


cdef vector[vector[ItemNo]] recoverfragments_(ItemNo v, Edge edge, Chart chart,
		list backtransform):
	cdef RankedEdge child
	cdef list children = []
	cdef vector[ItemNo] childitems
	cdef vector[vector[ItemNo]] childitemss = []
	cdef vector[int] childranks
	cdef int n
	cdef str frag = backtransform[edge.rule.no]  # template
	ruleno = edge.rule.no  # FIXME only for debugging

	# nondeterministically collect all children w/on the fly left-factored debinarization
	if edge.rule.rhs2:  # is there a right child?
		return recover_rec([], v, edge, chart, backtransform)
	elif chart.grammar.selfmapping[edge.rule.rhs1] == 0:
		v = chart._left(v, edge)
		for edge in chart.parseforest[v]:
			childitems = []
			childitemss.push_back(recover_left_most(childitems, v, edge, chart, backtransform))
		return childitemss
	else:
		return [recover_left_most([], v, edge, chart, backtransform)]


cdef vector[ItemNo] recover_left_most(vector[ItemNo] childitems, ItemNo v, Edge edge, Chart chart, list backtransform):
	childitems.push_back(chart._left(v, edge))
	return childitems


cdef list recover_rec(list childitems, ItemNo v, Edge edge, Chart chart, list backtransform):
	if chart.grammar.selfmapping[edge.rule.rhs1] == 0:
		childitems.append(chart._right(v, edge))
		v = chart._left(v, edge)
		result = []
		for _edge in chart.parseforest[v]:
			result.extend(recover_rec(list(childitems), v, _edge, chart, backtransform))
		return result
	elif edge.rule.rhs2:  # is there a right child?
		childitems.append(chart._right(v, edge))
		return [recover_left_most(childitems, v, edge, chart, backtransform)]
	else:
		return [recover_left_most(childitems, v, edge, chart, backtransform)]


cdef class UnrankedItem:
	cdef:
		Label label
		vector[Label] children
		Prob prob
		Position pos


# @cython.cdivision(True)
cpdef decode2dop(SmallLCFRSChart chart, mode='maxruleproduct'):
	# 0. remove unreachable entries from chart
	# TODO

	# 1. compute inside/outside scores for all items in chart
	if not chart.inside.size():
		origlogprob = chart.grammar.logprob
		chart.grammar.switch(chart.grammar.currentmodel, logprob=False)
		getinside(chart)
		getoutside(chart)
		chart.grammar.switch(chart.grammar.currentmodel, logprob=origlogprob)
	sentprob = chart.inside[chart.root()]

	backtransform = chart.grammar.backtransform

	cdef:
		ItemNo item, leftitem, rightitem, left, right
		int n
		str frag
		Edge edge, _edge
		vector[vector[ItemNo]] children_s
		SmallChartItem newitem
		Prob prob
		uint64_t vec
		ProbRule * rule
		Label label
		bint found
		maxruleproduct = mode == 'maxruleproduct'
		vector[ProbRule*] rules

	# 2. create new chart by extending fragments

	cdef:
		SmallLCFRSChart projchart = SmallLCFRSChart(chart.grammar, chart.sent, logprob=False) # , itemsestimate=chart.items.size())
		# vector[Prob] lexprobs

	for n, a in enumerate(chart.grammar.backtransform):
		print(n, chart.grammar.rulestr(chart.grammar.revrulemap[n]),
					'\t', a)
		if n > 20:
			break

	print('\n\n\n')

	print(chart.parseforest.size())

	# traverse items in bottom-up order
	for n in range(0, chart.numitems() + 1):
		item = chart.getitemidx(n)

		if chart.grammar.selfmapping[chart.label(item)] == 0:
			continue
			# if chart.parseforest[item].size() != 1:
			# 	print(item, "size", chart.parseforest[item].size())

		for edge in chart.parseforest[item]:
			if edge.rule is not NULL:
				assert edge.rule.no < len(backtransform)
				frag = backtransform[edge.rule.no]
				# print(edge.rule.no, frag)
				# print(edge.rule.no, chart.grammar.tolabel[edge.rule.lhs], chart.grammar.tolabel[edge.rule.rhs1], chart.grammar.tolabel[edge.rule.rhs2])
				# print(edge.rule.no, edge.rule.lhs, edge.rule.rhs1, edge.rule.rhs2)
				children_s = recoverfragments_(item, edge, chart, backtransform)
				# print("fragments:", children_s)
				# print("nontfrags:", [[chart.label(item) for item in children] for children in children_s])
				if children_s.size() != 1:
					size = children_s[0].size()
					for x in children_s:
						if len(x) != size:
							exit(0)

				rep = tuple([str(i) for i in range(children_s[0].size())])
				ftree = Tree(frag.format(*rep))

				for children in children_s:
					item_lookup = {}
					prob = 1 / sentprob if maxruleproduct else 1.0
					prob *= chart.outside[n] * exp(-edge.rule.prob)
					children = list(reversed(children))
					for i, c in enumerate(children):
						# item_lookup[str(i)] = chart.label(c), chart.indices(c)
						item_lookup[str(i)] = chart.label(c), chart.items[c].vec
						prob *= chart.inside[c]

					for t in ftree.postorder():
						# x = { pos for c in t.children for pos in item_lookup[str(c)][1] }
						vec = 0b0
						for c in t.children:
							vec = vec | item_lookup[str(c)][1]
						# item_lookup[str(t)] = chart.grammar.toid[t.label], list(sorted(x))
						item_lookup[str(t)] = chart.grammar.toid[t.label], vec

						newitem.label = chart.grammar.toid[t.label]
						newitem.vec = vec
						itemidxit = projchart.itemindex.find(newitem)
						if itemidxit == projchart.itemindex.end():
							projchart.itemindex[newitem] = projchart.items.size()
							projchart.items.push_back(newitem)
							projchart.parseforest.resize(projchart.itemindex.size())
							projchart.probs.push_back(INFINITY)

					for t in ftree.postorder():
						newitem.label, newitem.vec = item_lookup[str(t)]
						root = dereference(projchart.itemindex.find(newitem)).second
						left = right = 0
						if len(t.children) > 1:
							newitem.label, newitem.vec = item_lookup[str(t.children[1])]
							right = dereference(projchart.itemindex.find(newitem)).second
						assert len(t.children) > 0
						newitem.label, newitem.vec = item_lookup[str(t.children[0])]
						left = dereference(projchart.itemindex.find(newitem)).second

						# search edge:
						found = False
						for _edge in projchart.parseforest[root]:
							if projchart._left(root, _edge) == left and \
								projchart._right(root, _edge) == right:
								found = True
								break

						if found:
							_edge.rule.prob += prob
						else:
							rule = new ProbRule()
							rule.prob = prob
							rule.lhs = item_lookup[str(t)][0]
							rule.rhs1 = newitem.label
							rule.rhs2 = 0 if right == 0 else item_lookup[str(t.children[1])][0]
							rules.push_back(rule)
							# TODO can we just ignore args / lengths ?!
							projchart.addedge(root, left, newitem, rule)

			# elif edge.rule is not NULL:
			# 	print(edge.rule.no, chart.grammar.tolabel[edge.rule.lhs], chart.grammar.tolabel[edge.rule.rhs1], chart.grammar.tolabel[edge.rule.rhs2])
			# 	print(edge.rule.no, edge.rule.lhs, edge.rule.rhs1, edge.rule.rhs2)
			else: # edge.rule is NULL and edge.rule.no < len(backtransform):
				assert edge.rule is NULL
				label = chart.label(item)
				newitem.label = label
				newitem.vec = chart.items[item].vec
				idcs = chart.indices(item)
				assert len(idcs) == 1
				wordidx = idcs[0]
				itemidxit = projchart.itemindex.find(newitem)
				prob = 1 / sentprob if maxruleproduct else 1.0
				prob *= chart.outside[item] * chart.inside[item]
				if itemidxit == projchart.itemindex.end():
					itemidx = projchart.itemindex[newitem] = projchart.items.size()
					projchart.itemindex[newitem] = projchart.items.size()
					projchart.items.push_back(newitem)
					projchart.parseforest.resize(projchart.itemindex.size())
					projchart.probs.push_back(INFINITY)
					projchart.addlexedge(itemidx, wordidx)
					projchart.updateprob(itemidx, -log(prob)) # Todo: handle different if variational
					# lexprobs.resize(projchart.itemindex.size())
					# lexprobs[itemidx] = prob
				else:
					assert False

				# print(n, item, "-> eps")
	# normalize (already normalized if maxruleprod)
	if not maxruleproduct:
		pass
		# TODO implement variational

	cdef vector[bint] processed
	processed = [False for _ in range(projchart.items.size())]
	# compute Viterbi probabilities bottom-up
	# FIXME: increasing item order should ok?!
	for n in list(range(1, projchart.numitems() + 1)) + [0]:
		item = projchart.getitemidx(n)
		processed[item] = True
		for edge in projchart.parseforest[item]:
			if edge.rule is NULL: # lexical rules are already processed
				continue
			left = projchart._left(item, edge)
			if not processed[left]:
					print(item, left, right)
					assert processed[left]
			prob = -log(edge.rule.prob)
			prob += projchart.probs[projchart._left(item, edge)]
			if edge.rule.rhs2:
				right = projchart._right(item, edge)
				prob += projchart.probs[right]
				if not processed[right]:
					print(item, left, right)
					assert processed[right]
			projchart.updateprob(item, prob)


	# select Viterbi parse
	print(projchart._root().label, bin(projchart._root().vec), projchart.items.size())
	getderivations(projchart, 10, derivstrings=True)

	print(projchart.derivations)

	# clean up rules
	for rule in rules:
		del rule
