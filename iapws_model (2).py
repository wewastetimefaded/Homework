import numpy as np
from iapws import IAPWS97 as gas
from scipy.optimize import minimize
MPa = 10 ** 6
to_kelvin = lambda x: x + 273.15 if x else None


class IapwsModel:

    def __init__(self, p = None, t = None, v = None, s = None, h = None, x = None, *args, **kwargs):
        self.h = self.calc_h_from_params(p, h, t, s, x, *args, **kwargs)
        self.p = self.calc_p_from_params(p, self.h, s, t, *args, **kwargs) 
        self.t = self.calc_t_from_params(self.p, t, v, self.h, s, x, *args, **kwargs)
        self.v = self.calc_v_from_params(self.p, self.h, self.t, v, s, *args, **kwargs)
        self.x = self.calc_x(self.p, self.h, self.t, s, x, *args, **kwargs)
        self.s = self._iso_s(self.p, self.h, self.t, s, self.x)
        self.phase = self.get_phase(self.p, self.h, self.t, self.s, self.x, *args, **kwargs)
      
        
    def calc_p_from_params(self, p = None, t = None, h = None, s = None, *args, **kwargs):    
         if h is not None and s is not None:
              p = gas(h = h, s = s).P
              return p
         else: 
              return p

    def calc_v_from_params(self, p = None, h = None, t = None, v = None, s = None, *args, **kwargs): 
        if p is not None and h is not None:
            v = gas(P = p, h = h).v
            return v
        elif p is not None and s is not None:
            v = gas(P = p, s = s).v
            return v
        elif p is not None and t is not None :
            v = gas(P= p, T = t).v
            return v
        elif h is not None and s is not None:
            v = gas(h = h, s = s).v
            return v
        else: 
            return v

    def calc_t_from_params(self, p = None, t = None, v = None, h = None, s  = None, x  = None, *args, **kwargs):
        if p is not None and h is not None:
            t = gas(P = p, h = h).T
            return t

        elif v is not None:  
            def error(t):
                v_ = gas(P=p, T=float(to_kelvin(t))).v
                return (v-v_) ** 2
     
            t = float(minimize(error, 600, tol=1e-8).x)
            return t

        elif p is not None and s is not None:
            t = gas(P = p, s = s).T
            return t
        elif h is not None and s is not None:
            t = gas(h = h, s = s).T 
            return t
        elif p is not None and x is not None:
            t = gas(P = p, x = x).T
            return t
        else:
            return t
    
    def calc_h_from_params(self, p = None, h = None, t = None, s = None, x = None, *args, **kwargs):  
        if  p is not None and t is not None:
            h = gas(P = p, T = t).h
            return h
        elif  p is not None and s is not None:
            h = gas(P = p, s = s).h
            return h
        elif p is not None  and  x is not None :
            h = gas(P = p, x = x).h
            return h
        else:  
            return h
      
    def _iso_s(self, p, h = None, t = None, s = None, x = None):
       if p is not None and h is not None:
           s = gas(P = p, h = h ).s
           return s
       elif p is not None and t is not None:
           s = gas(P = p, T = t).s
           return s
       elif p is not None and x is not None:
           s = gas(P = p, x = x).s
           return s
       else:  
           return s

    def get_phase (self, p = None, h = None, t = None, s = None, x = None, *args, **kwargs):
        if p is not None and h is not None:
            point = gas(P = p, h = h).phase
            return point
        elif p is not None and s is not None:
            point = gas(P = p, s = s).phase
            return point
        elif p is not None and t is not None:
            point = gas(P = p, T = t).phase
            return point
        elif p is not None and x is not None:
            point = gas(P = p, x = x).phase
            return point
        elif h is not None and s is not None:
            point = gas(h = h, s = s).phase
            return point
        else:
            return point
        
    def calc_x (self, p = None, h = None, t = None, s = None, x = None, *args, **kwargs):
        if p is not None and h is not None:
            x = gas(P = p, h = h).x
            return x
        elif p is not None and s is not None:
            x = gas(P = p, s = s).x
            return x
        elif p is not None and t is not None:
            x = gas(P = p, T = t).x
            return x
        elif h is not None and s is not None:
            x = gas(h = h, s = s).x
            return x
        return x

    def _isobaric_t(self, t1, v1, v2, *args, **kwargs):
        p = self.calc_p_from_params(t=t1, v=v1, *args, **kwargs)
        t = self.calc_t_from_params(p=p, v=v2)
        return t
        
    def _isobaric_v(self, t1, t2, v1, *args, **kwargs):
        p = self.calc_p_from_params(t=t1, v=v1, *args, **kwargs)
        v = self.calc_v_from_params(p=p, t=t2)
        return v
    
    def isobaric_process(self, t1, v1, t2 = None, v2 = None, *args, **kwargs):
        if t2 is None and v2 is None:
            raise ValueError("t2 and v2 dont should be None")
        if t2 is not None:
            return self._isobaric_t(t1=t1, v1=v1, v2=v2, *args, **kwargs)
        return self._isobaric_v(t1=t1, v1=v1, t2=t2, *args, **kwargs)

    def _isothermal_p(self, p1, v1, v2, *args, **kwargs):
        t = self.calc_t_from_params(p = p1, v = v1, *args, **kwargs)
        p = self.calc_p_from_params(t = t, v = v2)
        return p

    def _isothermal_v(self, p1, v1, p2, *args, **kwargs):
        t = self.calc_t_from_params(p = p1, v = v1, *args, **kwargs)
        v = self.calc_v_from_params(p = p2, t = t)
        return v
  
    def isotermal_process(self, p1, v1, p2 = None, v2 = None, *args, **kwargs):
        if p2 is None and v2 is None:
            raise ValueError("p2 and v2 dont should be None")
        if p2 is not None:
            return self._isothermal_p(p1=p1, v1=v1, v2=v2, *args, **kwargs)
        return self._isothermal_v(p1=p1, v1=v1, p2=p2, *args, **kwargs)
    
    def _isohoric_p(self, p1, t1, t2, *args, **kwargs):
        v1 = self.calc_v_from_params(p = p1, t = t1, *args, **kwargs)
        p2 = self.calc_p_from_params(t=t2, v=v1)
        return p2

    def _isohoric_t(self, p1, p2, t1, *args, **kwargs):
       v1 = self.calc_v_from_params(p = p1, t = t1, *args, **kwargs)
       t2 = self.calc_t_from_params(p = p2, v = v1) 
       return t2
    
    def isochoric_process(self, p1, t1, p2 = None, t2 = None, *args, **kwargs):
        if p2 is None and t2 is None:
            raise ValueError("p2 and t2 dont should be None")
        if p2 is not None:
            return self._isohoric_p(p1=p1, t1=t1, t2=t2, *args, **kwargs)
        return self._isohoric_t(p1=p1, t1=t1, p2=p2, *args, **kwargs)

    def _isentropic_v(self, p1, p2, t1, *args, **kwargs):
        s1 = self._iso_s(p = p1, t = t1, *args, **kwargs)
        v2 = self.calc_v_from_params(p = p2, s = s1)
        return v2
        
    def _isentropic_t(self, p1, p2, t1, *args, **kwargs):
        s1 = self._iso_s(p = p1, t = t1, *args, **kwargs)
        v2 = self.calc_v_from_params(p = p2, s = s1)
        return t2

    def _isentropic_p(self, p1, v1, v2, *args, **kwargs):
        s1 = self._iso_s(p = p1, v = v1, *args, **kwargs)
        def error(p):
            v = gas(P=p, s=s1).v
            return (v2-v) ** 2
        p2 = float(minimize(error, 1, tol=1e-8).x)  
        return p2
       
    def isentropic_process(self, p1=None, t1=None, v1=None, p2 = None, t2 = None, v2=None, *args, **kwargs):
        if p2 is not None and p1 is not None and v1 is not None:
           return self._isentropic_v(p1=p1, p2=p2, t1=t1, *args, **kwargs)  
        if p1 is not None and v1 is not None and v2 is not None:
           return self._isentropic_p(p1, v1, v2, *args, **kwargs)
        if p1 is not None and t1 is not None and p2 is not None:
           return self._isentropic_t(p1, p2, t1, *args, **kwargs)
           
         
         
         