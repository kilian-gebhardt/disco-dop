# See: http://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html
cdef extern int __builtin_ffsl (unsigned long)
cdef extern int __builtin_ctzl (unsigned long)
cdef extern int __builtin_clzl (unsigned long)
cdef extern int __builtin_popcountl (unsigned long)

cpdef inline int nextset(unsigned long vec, unsigned int pos):
	""" Return next set bit starting from pos, -1 if there is none.
	>>> nextset(0b001101, 1)
	2
	"""
	# return (vec >> pos) ? pos + __builtin_ctzl(vec >> pos) : -1
	return ((vec >> pos) > 0) * pos + __builtin_ffsl(vec >> pos) - 1

cpdef inline int nextunset(unsigned long vec, unsigned int pos):
	""" Return next unset bit starting from pos. There is always a next unset
	bit, so no bounds checking. __builtin_ctzl is undefined when input is
	zero, so the most significant bit should always remain unused.
	>>> nextunset(0b001101, 2)
	4
	"""
	return pos + __builtin_ctzl(~vec >> pos)

cpdef inline int bitcount(unsigned long vec):
	""" Number of set bits (1s)
	>>> bitcount(0b0011101)
	4
	"""
	return __builtin_popcountl(vec)

cpdef inline int bitlength(unsigned long vec):
	""" number of bits needed to represent vector
	(equivalently: index of most significant set bit, plus one)
	>>> bitlength(0b0011101)
	5"""
	return sizeof (vec) * 8 - __builtin_clzl(vec)

cpdef inline int fanout(unsigned long vec):
	""" number of contiguous components in bit vector (gaps plus one)
	>>> fanout(0b011011011)
	3"""
	gaps = 0; pos = 0
	while vec >> pos:
		pos = nextunset(vec, nextset(vec, pos))
		gaps += 1
	return gaps # this value is actually gaps+1

cpdef inline int testbit(unsigned long vec, unsigned int pos):
	""" Mask a particular bit, return nonzero if set 
	>>> testbit(0b0011101, 0)
	1
	>>> testbit(0b0011101, 1)
	0"""
	return vec & (1UL << pos)

# todo: see if this can be turned into a macro
cpdef inline int testbitc(unsigned char arg, unsigned int pos):
	""" Mask a particular bit, return nonzero if set 
	>>> testbit(0b0011101, 0)
	1
	>>> testbit(0b0011101, 1)
	0"""
	return arg & (1UL << pos)

cpdef inline int testbitshort(unsigned short arg, unsigned int pos):
	""" Mask a particular bit, return nonzero if set 
	>>> testbit(0b0011101, 0)
	1
	>>> testbit(0b0011101, 1)
	0"""
	return arg & (1UL << pos)

cpdef inline int testbitint(unsigned int arg, unsigned int pos):
	""" Mask a particular bit, return nonzero if set 
	>>> testbit(0b0011101, 0)
	1
	>>> testbit(0b0011101, 1)
	0"""
	return arg & (1UL << pos)

def main():
	assert nextset(0b001100110, 3) == 5
	assert nextunset(0b001100110, 1) == 3
	assert bitcount(0b001100110) == 4
	assert testbit(0b001100110, 1)
	assert not testbit(0b001100110, 3)
	assert fanout(0b0111100) == 1
	assert fanout(0b1000001) == 2
	assert fanout(0b011011011) == 3
	print "it worked"
	from doctest import testmod, NORMALIZE_WHITESPACE, ELLIPSIS
	#from cydoctest import testmod
	# do doctests, but don't be pedantic about whitespace (I suspect it is the
	# militant anti-tab faction who are behind this obnoxious default)
	fail, attempted = testmod(verbose=False, optionflags=NORMALIZE_WHITESPACE | ELLIPSIS)
	if attempted and not fail: print "%s: %d doctests succeeded!" % (__file__, attempted)
	else: print "attempted", attempted, "fail", fail

if __name__ == '__main__': main()
