'''

Segment tree.
This implementation is mostly taken from openai's segment tree's implementation:
https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
I wrote the implmentation to make it easier for me to understand what these
classes do and how they function.
I adapted the implementation in a very minimal way, adjusting it to my personal implementation
taste.

'''
import operator


class SegmentTree(object):
    ''' Segment Tree with basic functionaliyt to later be
        extended by classes 'SumSegmentTree' and
        'MinSegmentTree'
    '''

    def __init__(self, capacity: int, operation: __builtins__, neutral_element: any = None) -> None:
        ''' Build a Segment Tree data structure.

        https://en.wikipedia.org/wiki/Segment_tree

        Can be used as regular array, but with two
        important differences:

            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.

        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        '''
        assert capacity > 0 and capacity & (capacity -
                                            1) == 0, 'capacity must be positive and a power of 2.'
        self._capacity: int = capacity
        self._value: list = [neutral_element] * 2 * capacity
        self._operation = operation

    def _reduce_helper(self, start: int, end: int, node: int, node_start: int,
                       node_end: int) -> int or float:
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end))

    def reduce(self, start: int = 0, end: int = None) -> int or float:
        ''' Returns result of applying `self.operation`
        to a contiguous subsequence of the array.

            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))

        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences

        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        '''
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx: int, val: any) -> None:
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(self._value[2 * idx], self._value[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> int or float:
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):

    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(capacity=capacity,
                                             operation=operator.add,
                                             neutral_element=0.)

    def sum(self, start: int = 0, end: int = None) -> int or float:
        ''' Returns arr[start] + ... + arr[end] '''
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum: float) -> int:
        ''' Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum

        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.

        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix

        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        '''
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx: int = 1
        while idx < self._capacity:
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):

    def __init__(self, capacity: int):
        super(MinSegmentTree, self).__init__(capacity=capacity,
                                             operation=min,
                                             neutral_element=float('inf'))

    def min(self, start: int = 0, end: int = None) -> int or float:
        ''' Returns min(arr[start], ..., arr[end]) '''
        return super(MinSegmentTree, self).reduce(start, end)
