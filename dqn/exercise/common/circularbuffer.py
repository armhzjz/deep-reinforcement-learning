''' A simple and very basic Circular Buffer implementation.
    It exposes some properties for the user to undestand the
    internal state of the buffer (i.e. previous index, current index, size and
    if the buffer is full).
    It can be treated as if it were a list
    Author: armhzjz@pm.me
'''
import numpy as np


class CircularBuffer(list):
    ''' A simple and very basic Circular Buffer implementation.
        It exposes some properties for the user to undestand the
        internal state of the buffer (i.e. previous index, current index, size and
        if the buffer is full).
        It can be treated as if it were a list
    '''

    def __init__(self, buff_capacity: int = 1, init_values: any = None) -> None:
        super(CircularBuffer, self).__init__()
        self._buff_capacity: int = buff_capacity
        self._init_vals = init_values
        self._init(self._init_vals)

    def _init(self, init_values: any) -> None:
        if init_values is not None:
            super(CircularBuffer, self).extend([init_values for _ in range(self.buff_capacity)])
        self._current_index = self._prev_index = 0

    def append(self, element: any) -> int:
        ''' Add an element to the circular buffer and adjust the internals
            (i.e. buffer size, current and last indexes, etc.)
        '''
        self._prev_index = self._current_index
        if not self._is_buff_full():
            super(CircularBuffer, self).append(element)
        else:
            self.__setitem__(self._current_index, element)
        self._current_index = (self._current_index + 1) % self._buff_capacity
        return self._prev_index

    def clear(self) -> None:
        ''' Clears the buffer and returns it to its initial internal states
            Notice, that if any initial values were given when the instance was
            created, the buffer will be initialized to this value again.
        '''
        super(CircularBuffer, self).clear()
        self._init(self._init_vals)

    def _is_buff_full(self) -> bool:
        ''' Returns True if the buffer is full; return false otherwise'''
        return self.__len__() == self._buff_capacity

    def __getitem__(self, idx: int) -> any:
        ''' Returns the value pointed at by 'idx' '''
        try:
            return super(CircularBuffer, self).__getitem__(idx)
        except IndexError:
            raise IndexError('Index out of circular buffer\'s capacity range')

    def __setitem__(self, idx: int, value: any) -> None:
        ''' Set value 'value' on the specific buffer's position 'idx'
            Be careful on how you use this function - your values could be soon
            overwritten by the append method si
            nce the state of the buffer (i.e. indices)
            are not controlled, managed nor manipulated by this function
        '''
        try:
            return super(CircularBuffer, self).__setitem__(idx, value)
        except IndexError:
            raise IndexError(f'Index {idx} out of circular buffer\'s capacity range')

    @property
    def size(self) -> int:
        ''' The current buffer size (current numbers of elements on the buffer) '''
        return self.__len__()

    @property
    def isFull(self) -> bool:
        ''' Returns True if buffer is full. Returns False otherwise '''
        return self._is_buff_full()

    @property
    def last_written_index(self) -> int:
        ''' Returns the last index at which an element was written '''
        return self._prev_index

    @property
    def index(self) -> int:
        ''' Returns the current index at which the next value will be written '''
        return self._current_index

    def pop(self) -> None:
        ''' Not supported '''
        pass

    def extend(self) -> None:
        ''' Not supported '''
        pass

    def insert(self) -> None:
        ''' Not supported '''
        pass

    def remove(self) -> None:
        ''' Not supported '''
        pass

    def reverse(self) -> None:
        ''' Not supported '''
        pass

    def sort(self) -> None:
        ''' Not supported '''
        pass
