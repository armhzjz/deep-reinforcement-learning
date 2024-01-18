import numpy as np


class Scheduler(object):
    ''' Base scheduler class '''

    def __init__(self, sched_time_span: int, initial_val: float, final_val: float) -> None:
        if not (0 <= initial_val < final_val) and (initial_val != final_val):
            raise Exception(
                'Initial and / or final value do not match constraint (0<=initial<final)')
        if sched_time_span < 2:
            raise Exception('Time span must be at least 2 steps')
        self._time_span: int = sched_time_span
        self._init_v: float = initial_val
        self._final_v: float = final_val
        self._step: int = 0
        self._number_steps: float = final_val - initial_val
        self._step_size: float = self._number_steps / float(sched_time_span)

    @property
    def scheduler_time_span(self) -> int:
        ''' This is the time span the scheduler will go through '''
        return self._time_span

    @property
    def initial_sched_value(self) -> float:
        ''' Thsi is the initial value the schedule will visit '''
        return self._init_v

    @property
    def final_sched_value(self) -> float:
        ''' This is the final value the schedule will visit '''
        return self._final_v

    @property
    def scheduler_current_step(self) -> int:
        ''' This is the last step the scheduler has gone through '''
        return self._step


class LinearScheduler(Scheduler):
    '''
        Linear scheduler.
        Moves linearlly a value from "initial value" to the "final value" in
        steps of size "(final value - initial value) / number of steps" (number of steps
        equals the scheduler time span).
        The class maintains locally the numer of steps it has gone through, so the user
        only needs to call the "get_step" method on each step of the training phase to retrieve
        the next linearlly changed value.
        The user could also specify the number of step for which the value is required.
    '''

    def __init__(self, sched_time_span: int, initial_val: float, final_val: float) -> None:
        super(LinearScheduler, self).__init__(sched_time_span, initial_val, final_val)

    def __call__(self, step: int = None) -> float:
        ''' If step is None, this method returns the next linear step value.
            If step is given, this method returns the linear value that would correspond
            to such step.
        '''
        if not step:
            self._step += 1
            return min(self._final_v, self._init_v + self._step * self._step_size)
        return min(self._final_v, self._init_v + step * self._step_size)


class ExponentialAnnealingScheduler(Scheduler):

    def __init__(self, sched_time_span: int, initial_val: float, final_val: float,
                 rate: float) -> None:
        super(ExponentialAnnealingScheduler, self).__init__(sched_time_span, initial_val, final_val)
        self._rate = rate

    def __call__(self, step: int = None) -> float:
        if not step:
            self._step += 1
            return min(self._final_v, self._init_v + (1 - np.exp(-self._rate * self._step)))
        return min(self._final_v, self._init_v + (1 - np.exp(-self._rate * step)))


class ExponentialDecay(Scheduler):

    def __init__(self, sched_time_span: int, final_val: float, decay_factor: float) -> None:
        super(ExponentialDecay, self).__init__(sched_time_span, 0., final_val)
        self._decay_factor = decay_factor

    def __call__(self, step: int = None) -> float:
        if not step:
            retval = max(self._decay_factor**step, self._final_v)
            self._step += 1
            return retval
        return max(self._decay_factor**step, self._final_v)

    @property
    def initial_sched_value(self) -> float:
        ''' Thsi is the initial value the schedule will visit '''
        raise NotImplementedError("Property not implemented for class ExponentialDecay")
