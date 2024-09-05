import math

class IdealGasModel:

    def __init__(self, gas_const, c_p, p = None, v = None, t = None, *args, **kwargs):
        self.gas_const = gas_const
        self.c_p = c_p
        self.h = self.calc_h_from_params(t, *args, **kwargs)
        self.v = self.calc_v_from_params(p, t, *args, **kwargs )
        self.t = self.calc_t_from_params(p, self.v, *args, **kwargs)
        self.p = self.calc_p_from_params(self.v, self.t, *args, **kwargs)
        self.phase = self.get_phase(self.p, self.v, self.t, *args, **kwargs)
        self.x = self.calc_x(self.p, self.v, self.t, *args, **kwargs)

    def calc_p_from_params(self, v = None, t = None, *args, **kwargs):
        return (self.gas_const * t) / v

    def calc_v_from_params(self, p = None, t = None, *args, **kwargs):
        return (self.gas_const) * t / p

    def calc_t_from_params(self, p, v, *args, **kwargs):
        return (p * v) / self.gas_const

    def calc_h_from_params(self, t, *args, **kwargs):
        return self.c_p * t

    def _isobaric_t(self, v1, v2, t1, *args, **kwargs):
        return (v2 * t1) / v1

    def _isobaric_v(self, v1, t1, t2, *args, **kwargs):
        return (t1 * v1) / t2 

    def isobaric_process(self, v1, t1, v2 = None, t2 = None, *args, **kwargs):
        if v2 is not None:
            return self._isobaric_t(v1, t1, v2, *args, **kwargs)
        else:
            return self._isobaric_v(v1, t1, t2, *args, **kwargs)

    def _isothermal_p(self, p1, v1, v2, *args, **kwargs):
        return (p1 * v1) / v2

    def _isothermal_v(self, p1, v1, p2, *args, **kwargs):
        return (v1 * p1) / p2

    def isothermal_process(self, p1, v1, p2 = None, v2 = None, *args, **kwargs):
        if v2 is not None:
            return self._isothermal_p(p1, v1, v2, *args, **kwargs)
        else:
            return self._isothermal_v(p1, v1, p2, *args, **kwargs)

    def _isohoric_p(self, p1, t1, t2, *args, **kwargs):
        return (p1 * t2) / t1

    def _isohoric_t(self, p1, t1, p2, *args, **kwargs):
        return (t1 * p2) / p1

    def isochoric_process(self, p1, t1, p2 = None, t2 = None, *args, **kwargs):
        if t2 is not None:
            return self._isohoric_p(p1, t1, t2, *args, **kwargs)
        else:
            return self._isohoric_t(p1, t1, p2, *args, **kwargs)

    def _isentropic_v(self, p1, v1, t1, p2 = None, t2 = None, *args, **kwargs):      
        k = self.c_p / (self.gas_const - self.c_p)
        if p2 is not None:
            b = p1 / p2
            v2 = math.pow(b, 1/k)*v1
        else:
            d = t2 / t1
            v2 = math.pow(d, 1/(k-1))*v1
        return v2

    def _isentropic_t(self, p1, v1, t1, p2 = None, v2 = None, *args, **kwargs):
        k = self.c_p / (self.gas_const - self.c_p)
        b = v1 / v2
        c = p2 / p1
        if v2 is not None:
            t2 = (b**(k-1)) * t1
        else:
            t2 = c**((k-1)/k) * t1
        return t2

    def _isentropic_p(self, p1, v1, t1, v2 = None, t2 = None, *args, **kwargs):
        k = self.c_p / (self.gas_const - self.c_p)
        if v2 is not None:
            p2 = p1 * (v1/v2)**k
        else:
            d = t2 / t1
            p2 = math.pow(d, ((k-1)/k)) * p1
        return p2

    def isentropic_process(self, p1, v1, t1, p2 = None, v2 = None, t2 = None, *args, **kwargs):
        if p2 is not None and t2 is not None:
            return self._isentropic_v(p1, v1, t1, p2, t2, *args, **kwargs)
        elif p2 is not None and v2 is not None:
            return self._isentropic_t(p1, v1, t1, p2, v2, *args, **kwargs)
        elif v2 is not None and t2 is not None:
            return self._isentropic_p(p1, v1, t1, v2, t2, *args, **kwargs)

    def get_phase(self, p = None, v = None, t = None, *args, **kwargs):
        return "Gas"

    def calc_x(self, p = None, v = None, t = None, *args, **kwargs):
        return 1