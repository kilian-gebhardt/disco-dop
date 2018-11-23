from .containers cimport Chart, RankedEdge, Edge, ItemNo, Label, Prob, Position, ProbRule
from .coarsetofine import getinside, getoutside, topologicalorder
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.math cimport HUGE_VAL as INFINITY, exp, log
from sys import exit
from .plcfrs cimport SmallChartItem, SmallLCFRSChart, LCFRSChart_fused
from .tree import Tree
from cython.operator cimport postincrement, dereference
from libc.stdint cimport uint64_t
from .disambiguation import getderivations


cdef vector[vector[pair[ItemNo, Prob]]] recoverfragments_(ItemNo v, Edge edge, Chart chart,
		list backtransform):
	cdef RankedEdge child
	cdef list children = []
	cdef vector[pair[ItemNo, Prob]] childitems
	cdef vector[vector[pair[ItemNo, Prob]]] childitemss = []
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


cdef vector[pair[ItemNo, Prob]] recover_left_most(vector[pair[ItemNo, Prob]] childitems, ItemNo v, Edge edge, Chart chart, list backtransform):
	childitems.push_back((chart._left(v, edge), edge.rule.prob))
	return childitems


cdef list recover_rec(list childitems, ItemNo v, Edge edge, Chart chart, list backtransform):
	if chart.grammar.selfmapping[edge.rule.rhs1] == 0:
		childitems.append((chart._right(v, edge), edge.rule.prob))
		v = chart._left(v, edge)
		result = []
		for _edge in chart.parseforest[v]:
			result.extend(recover_rec(list(childitems), v, _edge, chart, backtransform))
		return result
	elif edge.rule.rhs2:  # is there a right child?
		childitems.append((chart._right(v, edge), edge.rule.prob))
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
	# TODO: maybe do elsewhere?
	chart.filter()
	cdef:
		vector[ItemNo] chart_top_order = topologicalorder(chart)
		int n


	# TODO: the charts are not necessarily acyclic!
	# TODO: implement fixed point approximation for inside/outside weights!
	if chart_top_order.size() != chart.numitems():
		print(chart_top_order.size(), chart.numitems())
	assert chart_top_order.size() == chart.numitems()


	# 1. compute inside/outside scores for all items in chart
	if not chart.inside.size():
		origlogprob = chart.grammar.logprob
		chart.grammar.switch(chart.grammar.currentmodel, logprob=False)
		getinside(chart, chart_top_order)
		getoutside(chart, chart_top_order)
		chart.grammar.switch(chart.grammar.currentmodel, logprob=origlogprob)
	sentprob = chart.inside[chart.root()]
	assert 0 <= sentprob <= 1

	backtransform = chart.grammar.backtransform

	cdef:
		ItemNo item, leftitem, rightitem, left, right
		int child
		str frag
		Edge edge, _edge
		vector[vector[pair[ItemNo, Prob]]] children_s
		SmallChartItem newitem, zero, left_item
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
	# print(projchart.itemindex.size())

	zero = SmallChartItem(0, 0)

	# print(chart.parseforest.size())

	# traverse items in bottom-up order
	for n in range(1, chart.numitems() + 1):
		item = chart.getitemidx(n)
		if chart.grammar.selfmapping[chart.label(item)] == 0:
			continue

		for edge in chart.parseforest[item]:
			if edge.rule is not NULL:
				assert edge.rule.no < len(backtransform)
				frag = backtransform[edge.rule.no]
				children_s = recoverfragments_(item, edge, chart, backtransform)
				# if children_s.size() != 1:
				# 	size = children_s[0].size()
				# 	for x in children_s:
				# 		if len(x) != size:
				# 			exit(0)

				ftree = Tree(frag.format(*[str(i) for i in range(children_s[0].size())]))

				for children in children_s:
					# for child, _ in children:
					# 	if child == 0:
					# 		print(children)
					# 		assert child != 0
					item_lookup = {}
					prob = 1 / sentprob if maxruleproduct else 1.0
					assert edge.rule.prob >= 0
					prob *= chart.outside[n] * exp(-edge.rule.prob)
					children = [(c[0], c[1]) for c in reversed(children)]
					for i, (c, p) in enumerate(children):
						# todo: are all the probabilities of the intermediate rules 1.0?
						# todo: if not, include them into prob!
						item_lookup[str(i)] = chart.label(c), chart.items[c].vec
						prob *= chart.inside[c]

						newitem = SmallChartItem(chart.label(c), chart.items[c].vec)
						itemidxit = projchart.itemindex.find(newitem)
						if itemidxit == projchart.itemindex.end():
							projchart.itemindex[newitem] = projchart.items.size()
							assert projchart.itemindex[newitem] > 0
							projchart.items.push_back(newitem)
							projchart.parseforest.resize(projchart.itemindex.size())
							projchart.probs.push_back(INFINITY)
						else:
							assert dereference(itemidxit).second != 0
							# print(dereference(itemidxit).first.label, dereference(itemidxit).first.vec, dereference(itemidxit).second)
							# print(projchart.itemindex.size())
							# print(n, dereference(projchart.itemindex.find(zero)).second)

					assert 1 + 1.e-5 >= prob >= 0 - 1e-5
					for t in ftree.postorder():
						assert(dereference(projchart.itemindex.find(zero)).second == 0)
						vec = 0b0
						for c in t.children:
							vec = vec | item_lookup[str(c)][1]

						assert t.label in chart.grammar.toid
						assert chart.grammar.toid.get(t.label) > 0
						assert str(t) not in item_lookup
						item_lookup[str(t)] = chart.grammar.toid.get(t.label), vec
						newitem = SmallChartItem(chart.grammar.toid.get(t.label), vec)
						itemidxit = projchart.itemindex.find(newitem)
						if itemidxit == projchart.itemindex.end():
							projchart.itemindex[newitem] = projchart.items.size()
							assert projchart.itemindex[newitem] > 0
							projchart.items.push_back(newitem)
							projchart.parseforest.resize(projchart.itemindex.size())
							projchart.probs.push_back(INFINITY)
						else:
							assert dereference(itemidxit).second != 0
						# print(dereference(itemidxit).first.label, dereference(itemidxit).first.vec, dereference(itemidxit).second)
						# print(projchart.itemindex.size())
						# print(n, dereference(projchart.itemindex.find(zero)).second)

					for t in ftree.postorder():
						root_entry = item_lookup[str(t)]
						# rootitem = SmallChartItem(item_lookup[str(t)][0], item_lookup[str(t)][1])
						itemidxit = projchart.itemindex.find(SmallChartItem(root_entry[0], root_entry[1]))
						assert itemidxit != projchart.itemindex.end()
						root = dereference(itemidxit).second
						left, right = 0, 0
						if len(t.children) > 1:
							assert len(t.children) == 2
							right_child = item_lookup[str(t.children[1])]
							# right_item = SmallChartItem(right_child[0], right_child[1])
							itemidxit = projchart.itemindex.find(SmallChartItem(right_child[0], right_child[1]))
							if not itemidxit != projchart.itemindex.end():
								print(right_child, t, item_lookup, children)
							assert itemidxit != projchart.itemindex.end()
							right = dereference(itemidxit).second
							assert right > 0

						assert len(t.children) > 0
						left_child = item_lookup[str(t.children[0])]
						left_item = SmallChartItem(left_child[0], left_child[1])
						itemidxit = projchart.itemindex.find(left_item)
						assert itemidxit != projchart.itemindex.end()
						left = dereference(itemidxit).second

						# check, if edge in projchart:
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
							# rule.rhs1 = left_item.label
							rule.rhs1 = left_child[0]
							rule.rhs2 = 0 if right == 0 else right_child[0] #right_item.label
							rule.args = 0
							rule.lengths = 0
							rules.push_back(rule)
							# TODO can we just ignore args / lengths ?!
							projchart.addedge(root, left, left_item, rule)

			else: # edge.rule is NULL and edge.rule.no < len(backtransform):
				assert edge.rule is NULL
				label = chart.label(item)
				# newitem = SmallChartItem(label, chart.items[item].vec)
				idcs = chart.indices(item)
				assert len(idcs) == 1
				wordidx = idcs[0]
				itemidxit = projchart.itemindex.find(SmallChartItem(label, chart.items[item].vec))
				prob = 1 / sentprob if maxruleproduct else 1.0
				prob *= chart.outside[item] * chart.inside[item]
				if itemidxit == projchart.itemindex.end():
					itemidx = projchart.itemindex[SmallChartItem(label, chart.items[item].vec)] = projchart.items.size()
					projchart.itemindex[SmallChartItem(label, chart.items[item].vec)] = projchart.items.size()
					assert projchart.itemindex[SmallChartItem(label, chart.items[item].vec)] > 0
					projchart.items.push_back(SmallChartItem(label, chart.items[item].vec))
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

	cdef vector[ItemNo] order = topologicalorder(projchart)

	# assume that edges in chart can be ordered topologically
	assert(order.size() + 1 == projchart.probs.size())
	# print("Order", order.size(), projchart.probs.size(), order)

	# compute Viterbi probabilities bottom-up

	# cdef vector[bint] processed
	# processed = [False for _ in range(projchart.items.size())]
	for n in order:
		item = projchart.getitemidx(n)
		# processed[item] = True
		for edge in projchart.parseforest[item]:
			if edge.rule is NULL: # lexical rules are already processed
				continue
			left = projchart._left(item, edge)
			# if not processed[left]:
			# 		print(item, left, right)
			# assert processed[left]
			# if not 1 >= edge.rule.prob >= 0:
			# 	print(edge.rule.prob)
			assert 1 + 1e-5 >= edge.rule.prob >= 0 - 1.e-5
			prob = -log(edge.rule.prob)
			prob += projchart.probs[projchart._left(item, edge)]
			if edge.rule.rhs2:
				right = projchart._right(item, edge)
				prob += projchart.probs[right]
				# if not processed[right]:
				# 	print(item, left, right)
				# assert processed[right]
			projchart.updateprob(item, prob)

	# select Viterbi parse
	# print(projchart._root().label, bin(projchart._root().vec), projchart.items.size())
	if projchart:
		getderivations(projchart, 10, derivstrings=True)
		for derivstr, prob in projchart.derivations:
			if prob < 0:
				print(derivstr, prob)
			assert prob >= 0
		results = [(derivstr, exp(-prob), []) for derivstr, prob in
					projchart.derivations]
	else:
		results = []

	# print(projchart.derivations)

	# clean up rules
	for rule in rules:
		del rule

	msg = "items in projected chart: %d" % chart.numitems()

	return results, msg
