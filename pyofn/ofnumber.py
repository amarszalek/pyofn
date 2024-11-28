# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy
from numpy.core.numeric import isscalar
from scipy.interpolate import interp1d
from scipy.integrate import trapz


class Branch(object):
    def __init__(self, fvalue_y, domain_x=None):
        super(Branch, self).__init__()
        assert len(fvalue_y), 'fvalue_y is empty'
        self.dim = len(fvalue_y)
        self.domain_x = domain_x
        if domain_x is None:
            self.domain_x = np.linspace(0., 1., self.dim)
        self.fvalue_y = np.asarray(fvalue_y, dtype=np.double)
    
    # str() method
    def __str__(self):
        return str(self.fvalue_y)
    
    # repr() method
    def __repr__(self):
        return str(self.fvalue_y)
    
    # initialize branch of linear type y = a * x + b
    @staticmethod
    def init_linear_ab(a, b, dim=11, domain_x=None):
        if domain_x is not None:
            dim = len(domain_x)
        else:
            assert (dim > 1), 'dim must be > 1'
        branch = Branch(np.zeros(dim), domain_x=domain_x)
        branch.fvalue_y = a * branch.domain_x + b
        return branch
    
    # initialize branch of linear type y = (x1 - x0) * x + x0
    @staticmethod
    def init_linear_x0x1(x0, x1, dim=11, domain_x=None):
        if domain_x is not None:
            dim = len(domain_x)
        else:
            assert (dim > 1), 'dim must be > 1'
        branch = Branch(np.zeros(dim), domain_x=domain_x)
        branch.fvalue_y = (x1 - x0) * branch.domain_x + x0
        return branch
    
    # initialize branch of gaussian type y = s * sqrt(-2 * ln(x)) + m
    @staticmethod
    def init_gauss(m, s, dim=11, x0=10, domain_x=None):
        if domain_x is not None:
            dim = len(domain_x)
        else:
            assert (dim > 1), 'dim must be > 1'
        branch = Branch(np.zeros(dim), domain_x=domain_x)
        branch.fvalue_y[1:] = s * np.sqrt(-2 * np.log(branch.domain_x[1:])) + m
        if isinstance(x0, int):
            branch.fvalue_y[0] = s * np.sqrt(-2 * np.log(branch.domain_x[1]/x0)) + m
        elif isinstance(x0, float):
            branch.fvalue_y[0] = s * np.sqrt(-2 * np.log(x0)) + m
        else:
            raise ValueError('x0 must be instance of int or float')
        return branch
    
    # initialize branch from scalar
    @staticmethod
    def init_from_scalar(s, dim=11, domain_x=None):
        assert isscalar(s), 's must be scalar'
        if domain_x is not None:
            dim = len(domain_x)
        else:
            assert (dim > 1), 'dim must be > 1'
        return Branch(np.ones(dim) * s, domain_x=domain_x)

    # deepcopy operator
    def copy(self):
        return deepcopy(self)
    
    # to_array method
    def to_array(self):
        return self.fvalue_y.copy()

    # add method left side
    def __add__(self, right):
        res = self.copy()
        if isinstance(right, Branch):
            assert np.all(res.domain_x == right.domain_x), 'self.domain_x and right.domain_x must be the same'
            res.fvalue_y = res.fvalue_y + right.fvalue_y
        elif isscalar(right):
            res.fvalue_y = res.fvalue_y + right
        else:
            raise ValueError('right must be instance of Branch class or scalar')
        return res

    # add method right side
    def __radd__(self, left):
        res = self.copy()
        if isinstance(left, Branch):
            assert np.all(res.domain_x == left.domain_x), 'self.domain_x and left.domain_x must be the same'
            res.fvalue_y = res.fvalue_y + left.fvalue_y
        elif isscalar(left):
            res.fvalue_y = res.fvalue_y + left
        else:
            raise ValueError('left must be instance of Branch class or scalar')
        return res

    # sub method left side
    def __sub__(self, right):
        res = self.copy()
        if isinstance(right, Branch):
            assert np.all(res.domain_x == right.domain_x), 'self.domain_x and right.domain_x must be the same'
            res.fvalue_y = res.fvalue_y - right.fvalue_y
        elif isscalar(right):
            res.fvalue_y = res.fvalue_y - right
        else:
            raise ValueError('right must be instance of Branch class or scalar')
        return res

    # sub method right side
    def __rsub__(self, left):
        res = self.copy()
        if isinstance(left, Branch):
            assert np.all(res.domain_x == left.domain_x), 'self.domain_x and left.domain_x must be the same'
            res.fvalue_y = left.fvalue_y - res.fvalue_y
        elif isscalar(left):
            res.fvalue_y = left - res.fvalue_y
        else:
            raise ValueError('left must be instance of Branch class or scalar')
        return res

    # mul method left side
    def __mul__(self, right):
        res = self.copy()
        if isinstance(right, Branch):
            assert np.all(res.domain_x == right.domain_x), 'self.domain_x and right.domain_x must be the same'
            res.fvalue_y = res.fvalue_y * right.fvalue_y
        elif isscalar(right):
            res.fvalue_y = res.fvalue_y * right
        else:
            raise ValueError('right must be instance of Branch class or scalar')
        return res

    # mul method right side
    def __rmul__(self, left):
        res = self.copy()
        if isinstance(left, Branch):
            assert np.all(res.domain_x == left.domain_x), 'self.domain_x and left.domain_x must be the same'
            res.fvalue_y = res.fvalue_y * left.fvalue_y
        elif isscalar(left):
            res.fvalue_y = res.fvalue_y * left
        else:
            raise ValueError('left must be instance of Branch class or scalar')
        return res

    # contains_zero method
    def contains_zero(self):
        min_val = np.min(self.fvalue_y)
        max_val = np.max(self.fvalue_y)
        return min_val <= 0.0 <= max_val
    
    # div method left side
    def __truediv__(self, right):
        res = self.copy()
        if isinstance(right, Branch):
            if right.contains_zero():
                raise ZeroDivisionError('division by zero')
            assert np.all(res.domain_x == right.domain_x), 'self.domain_x and right.domain_x must be the same'
            res.fvalue_y = res.fvalue_y / right.fvalue_y
        elif isscalar(right):
            if right == 0.:
                raise ZeroDivisionError('division by zero')
            res.fvalue_y = res.fvalue_y / right
        else:
            raise ValueError('right must be instance of Branch class or scalar')
        return res

    # div method right side
    def __rtruediv__(self, left):
        if self.contains_zero():
            raise ZeroDivisionError('division by zero')
        res = self.copy()
        if isinstance(left, Branch):
            assert np.all(res.domain_x == left.domain_x), 'self.domain_x and left.domain_x must be the same'
            res.fvalue_y = left.fvalue_y / res.fvalue_y
        elif isscalar(left):
            res.fvalue_y = left / res.fvalue_y
        else:
            raise ValueError('left must be instance of Branch class or scalar')
        return res

    # neg method
    def __neg__(self):
        res = self.copy()
        res.fvalue_y = -res.fvalue_y
        return res

    # < method
    def __lt__(self, val):
        if isinstance(val, Branch):
            return np.max(self.fvalue_y) < np.min(val.fvalue_y)
        elif isscalar(val):
            return np.max(self.fvalue_y) < val
        else:
            raise ValueError('val must be instance of Branch class or scalar')

    # <= method
    def __le__(self, val):
        if isinstance(val, Branch):
            return np.max(self.fvalue_y) <= np.min(val.fvalue_y)
        elif isscalar(val):
            return np.max(self.fvalue_y) <= val
        else:
            raise ValueError('val must be instance of Branch class or scalar')

    # > method
    def __gt__(self, val):
        if isinstance(val, Branch):
            return np.min(self.fvalue_y) > np.max(val.fvalue_y)
        elif isscalar(val):
            return np.min(self.fvalue_y) > val
        else:
            raise ValueError('val must be instance of Branch class or scalar')

    # >= method
    def __ge__(self, val):
        if isinstance(val, Branch):
            return np.min(self.fvalue_y) >= np.max(val.fvalue_y)
        elif isscalar(val):
            return np.min(self.fvalue_y) >= val
        else:
            raise ValueError('val must be instance of Branch class or scalar')

    # == method
    def __eq__(self, val):
        if isinstance(val, Branch):
            assert np.all(self.domain_x == val.domain_x), 'self.domain_x and val.domain_x must be the same'
            return np.all(self.fvalue_y == val.fvalue_y)
        elif isscalar(val):
            return np.all(self.fvalue_y == val)
        else:
            raise ValueError('val must be instance of Branch class or scalar')

    # != method
    def __ne__(self, val):
        if isinstance(val, Branch):
            assert np.all(self.domain_x == val.domain_x), 'self.domain_x and val.domain_x must be the same'
            return np.all(self.fvalue_y != val.fvalue_y)
        elif isscalar(val):
            return np.all(self.fvalue_y != val)
        else:
            raise ValueError('val must be instance of Branch class or scalar')

    # call method
    def __call__(self, x, kind='linear'):
        if isscalar(x):
            assert (0. <= x <= 1.), 'x must be in [0, 1]'
        elif isinstance(x, np.ndarray):
            assert ((0 <= x).all() and (x <= 1).all()), 'x must be in [0, 1]'
        else:
            raise ValueError('x must be instance of np.ndarray class or scalar')
        f = interp1d(self.domain_x, self.fvalue_y, kind=kind)
        return f(x)
            
    # alpha-cut method
    def acut(self, alpha, kind='linear'):
        assert (0. <= alpha <= 1.), 'x must be in [0, 1]'
        if alpha == 0.:
            return np.min(self.fvalue_y[:]), np.max(self.fvalue_y[:])
        elif alpha == 1.:
            return np.min(self.fvalue_y[-1:]), np.max(self.fvalue_y[-1:])
        elif alpha in self.domain_x:
            k = np.where(self.domain_x == alpha)[0][0]
            return np.min(self.fvalue_y[k:]), np.max(self.fvalue_y[k:])
        else:
            k = np.where(self.domain_x >= alpha)[0][0]
            fa = self(alpha, kind=kind)
            vmin = np.min(self.fvalue_y[k:])
            vmax = np.max(self.fvalue_y[k:])
            return np.min([fa, vmin]), np.max([fa, vmax])
            #i = int(alpha/(self.domain_x[1]-self.domain_x[0]))
            #xp = self.domain_x[i]
            #xk = self.domain_x[i + 1]
            #fp = self.fvalue_y[i]
            #fk = self.fvalue_y[i + 1]
            #factor = (alpha - xp) / (xk - xp)
            #vmin = np.min(self.fvalue_y[i+1:])
            #vmax = np.max(self.fvalue_y[i+1:])
            #y = factor * fk + (1 - factor) * fp
            #return np.min([y, vmin]), np.max([y, vmax])

    def change_domain(self, new_domain, kind='linear'):
        res = self.copy()
        new_y = res(new_domain, kind=kind)
        res.domain_x[:] = new_domain[:]
        res.fvalue_y[:] = new_y[:]
        return res

    # plot method
    def plot_branch(self, ax, plot_as='ordered', *args, **kwargs):
        if plot_as == 'ordered':
            ax.plot(self.domain_x, self.fvalue_y, *args, **kwargs)
        elif plot_as == 'classic':
            ax.plot(self.fvalue_y, self.domain_x, *args, **kwargs)
        else:
            raise ValueError('plot_as must be ordered or classic')


class OFNumber(object):
    def __init__(self, branch_f, branch_g, domain_x=None):
        super(OFNumber, self).__init__()
        if isinstance(branch_f, Branch):
            self.branch_f = branch_f
        elif isinstance(branch_f, (list, np.ndarray)):
            self.branch_f = Branch(branch_f, domain_x=domain_x)
        else:
            raise ValueError('branch_f must be instance of Branch, list or np.ndarray')
        if isinstance(branch_g, Branch):
            self.branch_g = branch_g
        elif isinstance(branch_g, (list, np.ndarray)):
            self.branch_g = Branch(branch_g, domain_x=domain_x)
        else:
            raise ValueError('branch_g must be instance of Branch, list or np.ndarray')
    
    # str() method
    def __str__(self):
        return '('+str(self.branch_f)+', '+str(self.branch_g)+')'
    
    # repr() method
    def __repr__(self):
        return '('+str(self.branch_f)+', '+str(self.branch_g)+')'
    
    # initialize OFN of linear type y = a * x + b, y = c * x + d
    @staticmethod
    def init_trapezoid_abcd(a, b, c, d, dim=11, domain_x=None):
        assert (dim > 1), 'dim must be > 1'
        branch_f = Branch.init_linear_ab(a, b, dim=dim, domain_x=domain_x)
        branch_g = Branch.init_linear_ab(c, d, dim=dim, domain_x=domain_x)
        return OFNumber(branch_f, branch_g)

    # initialize OFN of linear type
    # y = (fx1 - fx0) * x + fx0, y = (gx1 - gx0) * x + gx0
    @staticmethod
    def init_trapezoid_x0x1(fx0, fx1, gx0, gx1, dim=11, domain_x=None):
        assert (dim > 1), 'dim must be > 1'
        branch_f = Branch.init_linear_x0x1(fx0, fx1, dim=dim, domain_x=domain_x)
        branch_g = Branch.init_linear_x0x1(gx0, gx1, dim=dim, domain_x=domain_x)
        return OFNumber(branch_f, branch_g)

    # initialize branch of gaussian type y = s * sqrt(-2 * ln(x)) + m
    @staticmethod
    def init_gaussian(mf, sf, mg, sg, dim=11, x0=10, domain_x=None):
        assert (dim > 1), 'dim must be > 1'
        branch_f = Branch.init_gauss(mf, sf, dim=dim, x0=x0, domain_x=domain_x)
        branch_g = Branch.init_gauss(mg, sg, dim=dim, x0=x0, domain_x=domain_x)
        return OFNumber(branch_f, branch_g)

    # initialize ofn from scalar
    @staticmethod
    def init_from_scalar(s, dim=11, domain_x=None):
        assert isscalar(s), 's must be scalar'
        branch_f = Branch.init_from_scalar(s, dim=dim, domain_x=domain_x)
        branch_g = Branch.init_from_scalar(s, dim=dim, domain_x=domain_x)
        return OFNumber(branch_f, branch_g)
    
    # deepcopy operator
    def copy(self):
        return deepcopy(self)
    
    # to_array method
    def to_array(self, stack='vstack'):
        if stack == 'vstack':
            return np.vstack([self.branch_f.to_array(), self.branch_g.to_array()])
        elif stack == 'hstack':
            return np.hstack([self.branch_f.to_array(), self.branch_g.to_array()])
        else:
            raise ValueError('stack must be vstack or hstack')
    
    # plot method
    def plot_ofn(self, ax, plot_as='ordered', kwargs_f=None, kwargs_g=None):
        if kwargs_f is None:
            kwargs_f = {'c': 'k'}
        if kwargs_g is None:
            kwargs_g = {'c': 'k'}
        self.branch_f.plot_branch(ax, plot_as=plot_as, **kwargs_f)
        self.branch_g.plot_branch(ax, plot_as=plot_as, **kwargs_g)
        
    # add method left side
    def __add__(self, right):
        res = self.copy()
        if isinstance(right, OFNumber):
            res.branch_f = res.branch_f + right.branch_f
            res.branch_g = res.branch_g + right.branch_g
        elif isscalar(right):
            res.branch_f = res.branch_f + right
            res.branch_g = res.branch_g + right
        else:
            raise ValueError('right must be instance of OFNumber class or scalar')
        return res

    # add method right side
    def __radd__(self, left):
        res = self.copy()
        if isinstance(left, OFNumber):
            res.branch_f = res.branch_f + left.branch_f
            res.branch_g = res.branch_g + left.branch_g
        elif isscalar(left):
            res.branch_f = res.branch_f + left
            res.branch_g = res.branch_g + left
        else:
            raise ValueError('left must be instance of OFNumber class or scalar')
        return res
    
    # sub method left side
    def __sub__(self, right):
        res = self.copy()
        if isinstance(right, OFNumber):
            res.branch_f = res.branch_f - right.branch_f
            res.branch_g = res.branch_g - right.branch_g
        elif isscalar(right):
            res.branch_f = res.branch_f - right
            res.branch_g = res.branch_g - right
        else:
            raise ValueError('right must be instance of OFNumber class or scalar')
        return res

    # sub method right side
    def __rsub__(self, left):
        res = self.copy()
        if isinstance(left, OFNumber):
            res.branch_f = left.branch_f - res.branch_f
            res.branch_g = left.branch_g - res.branch_g
        elif isscalar(left):
            res.branch_f = left - res.branch_f
            res.branch_g = left - res.branch_g
        else:
            raise ValueError('left must be instance of OFNumber class or scalar')
        return res
    
    # mul method left side
    def __mul__(self, right):
        res = self.copy()
        if isinstance(right, OFNumber):
            res.branch_f = res.branch_f * right.branch_f
            res.branch_g = res.branch_g * right.branch_g
        elif isscalar(right):
            res.branch_f = res.branch_f * right
            res.branch_g = res.branch_g * right
        else:
            raise ValueError('right must be instance of OFNumber class or scalar')
        return res

    # mul method right side
    def __rmul__(self, left):
        res = self.copy()
        if isinstance(left, OFNumber):
            res.branch_f = res.branch_f * left.branch_f
            res.branch_g = res.branch_g * left.branch_g
        elif isscalar(left):
            res.branch_f = res.branch_f * left
            res.branch_g = res.branch_g * left
        else:
            raise ValueError('left must be instance of OFNumber class or scalar')
        return res
        
    # div method left side
    def __truediv__(self, right):
        res = self.copy()
        if isinstance(right, OFNumber):
            res.branch_f = res.branch_f / right.branch_f
            res.branch_g = res.branch_g / right.branch_g
        elif isscalar(right):
            res.branch_f = res.branch_f / right
            res.branch_g = res.branch_g / right
        else:
            raise ValueError('right must be instance of OFNumber class or scalar')
        return res

    # div method right side
    def __rtruediv__(self, left):
        res = self.copy()
        if isinstance(left, OFNumber):
            res.branch_f = left.branch_f / res.branch_f
            res.branch_g = left.branch_g / res.branch_g
        elif isscalar(left):
            res.branch_f = left / res.branch_f
            res.branch_g = left / res.branch_g
        else:
            raise ValueError('left must be instance of OFNumber class or scalar')
        return res
    
    # neg method
    def __neg__(self):
        res = self.copy()
        res.branch_f = -res.branch_f
        res.branch_g = -res.branch_g
        return res
    
    # contains_zero method
    def contains_zero(self):
        if self.branch_f.contains_zero() or self.branch_g.contains_zero():
            return True
        min_val = np.min([self.branch_f.fvalue_y, self.branch_g.fvalue_y])
        max_val = np.max([self.branch_f.fvalue_y, self.branch_g.fvalue_y])
        return min_val <= 0.0 <= max_val
    
    # < method
    def __lt__(self, val):
        if isinstance(val, OFNumber):
            f = self.branch_f < val.branch_f
            g = self.branch_g < val.branch_g
        elif isscalar(val):
            f = self.branch_f < val
            g = self.branch_g < val
        else:
            raise ValueError('val must be instance of OFNumber class or scalar')
        if f and g:
            return True
        return False

    # <= method
    def __le__(self, val):
        if isinstance(val, OFNumber):
            f = self.branch_f <= val.branch_f
            g = self.branch_g <= val.branch_g
        elif isscalar(val):
            f = self.branch_f <= val
            g = self.branch_g <= val
        else:
            raise ValueError('val must be instance of OFNumber class or scalar')
        if f and g:
            return True
        return False

    # > method
    def __gt__(self, val):
        if isinstance(val, OFNumber):
            f = self.branch_f > val.branch_f
            g = self.branch_g > val.branch_g
        elif isscalar(val):
            f = self.branch_f > val
            g = self.branch_g > val
        else:
            raise ValueError('val must be instance of OFNumber class or scalar')
        if f and g:
            return True
        return False

    # >= method
    def __ge__(self, val):
        if isinstance(val, OFNumber):
            f = self.branch_f >= val.branch_f
            g = self.branch_g >= val.branch_g
        elif isscalar(val):
            f = self.branch_f >= val
            g = self.branch_g >= val
        else:
            raise ValueError('val must be instance of OFNumber class or scalar')
        if f and g:
            return True
        return False

    # == method
    def __eq__(self, val):
        if isinstance(val, OFNumber):
            f = self.branch_f == val.branch_f
            g = self.branch_g == val.branch_g
        elif isscalar(val):
            f = self.branch_f == val
            g = self.branch_g == val
        else:
            raise ValueError('val must be instance of OFNumber class or scalar')
        if f and g:
            return True
        return False

    # != method
    def __ne__(self, val):
        if isinstance(val, OFNumber):
            f = self.branch_f != val.branch_f
            g = self.branch_g != val.branch_g
        elif isscalar(val):
            f = self.branch_f != val
            g = self.branch_g != val
        else:
            raise ValueError('val must be instance of OFNumber class or scalar')
        if f or g:
            return True
        return False
    
    # call method
    def __call__(self, x, kind='linear'):
        return self.branch_f(x, kind=kind), self.branch_g(x, kind=kind)
    
    # alpha-cut method
    def acut(self, alpha):
        f_min, f_max = self.branch_f.acut(alpha)
        g_min, g_max = self.branch_g.acut(alpha)
        return (f_min, f_max), (g_min, g_max)
    
    # support
    def supp(self):
        return self.acut(0.0)

    # kernel
    def ker(self):
        return self.acut(1.0)
    
    # defuzzy operator
    def defuzzy(self, method='scog', args=(0.5,)):
        if method == 'scog':
            return scog(self, args[0])
        elif method == 'expected':
            return expected(self)
        else:
            raise ValueError('wrong defuzzy method')
    
    # order operator
    def order(self, dfuzzy=None, proper=False, method='scog', args=(0.5,)):
        f = self.branch_f.fvalue_y
        odr = 0
        if proper:
            g = self.branch_g.fvalue_y
            if f[0] < g[0]:
                return 1
            elif f[0] > g[0]:
                return -1
            else:
                return 0
        if dfuzzy is None:
            dfuzzy = self.defuzzy(method=method, args=args)
        if (dfuzzy is None) or (dfuzzy == f[0]):
            odr = 0
        elif dfuzzy > f[0]:
            odr = 1
        elif dfuzzy < f[0]:
            odr = -1
        return odr
    
    def change_order(self):
        res = self.copy()
        res.branch_f = self.branch_g.copy()
        res.branch_g = self.branch_f.copy()
        return res

    def fnorm(self):
        f = self.branch_f.fvalue_y
        g = self.branch_g.fvalue_y
        x = self.branch_f.domain_x
        return np.sqrt(trapz(f**2, x)) + np.sqrt(trapz(g**2, x))
    
    def liquidity_ratio(self, w_sell=1.0, w_buy=1.0):
        f = self.branch_f.fvalue_y
        g = self.branch_g.fvalue_y
        x = self.branch_f.domain_x
        # return (w_buy*np.sqrt(trapz(f ** 2, x)) + w_sell*np.sqrt(trapz(g ** 2, x)))/(w_buy+w_sell)
        return (w_buy * trapz(np.abs(f), x) + w_sell * trapz(np.abs(g), x)) / (w_buy + w_sell)


# center of gravity operator
def scog(ofn, alpha):
    f = ofn.branch_f.fvalue_y
    g = ofn.branch_g.fvalue_y
    if np.all(f == g):
        return f[0] if f.max() == f.min() else None
    x = ofn.branch_f.domain_x
    y2 = np.abs(g - f)
    y1 = (alpha * f + (1.0 - alpha) * g) * y2
    return trapz(y1, x) / trapz(y2, x)


# expected value operator
def expected(ofn):
    y = 0.5 * (ofn.branch_f.fvalue_y + ofn.branch_g.fvalue_y)
    x = ofn.branch_f.domain_x
    return trapz(y, x)


def fabs(ofn):
    res = ofn.copy()
    res.branch_f = Branch(np.abs(res.branch_f.fvalue_y), domain_x=res.domain_x)
    res.branch_g = Branch(np.abs(res.branch_g.fvalue_y), domain_x=res.domain_x)
    return res


def flog(ofn):
    if (ofn.branch_f.fvalue_y <= 0.0).any() or (ofn.branch_g.fvalue_y <= 0.0).any():
        raise ValueError('ofn must be > 0')
    res = ofn.copy()
    res.branch_f = Branch(np.log(res.branch_f.fvalue_y), domain_x=res.domain_x)
    res.branch_g = Branch(np.log(res.branch_g.fvalue_y), domain_x=res.domain_x)
    return res


def fexp(ofn):
    res = ofn.copy()
    res.branch_f = Branch(np.exp(res.branch_f.fvalue_y), domain_x=res.domain_x)
    res.branch_g = Branch(np.exp(res.branch_g.fvalue_y), domain_x=res.domain_x)
    return res


def fpower(ofn, p):
    if (not isinstance(ofn, OFNumber)) and (not isinstance(p, OFNumber)):
        raise ValueError('at least one of arguments must be a OFNumber')
    elif isinstance(ofn, OFNumber) and isinstance(p, OFNumber):
        assert np.all(ofn.branch_f.domain_x == p.branch_f.domain_x), 'self.domain_x and val.domain_x must be the same'
        domain = ofn.branch_f.domain_x
    else:
        domain = ofn.branch_f.domain_x if isinstance(ofn, OFNumber) else p.branch_f.domain_x

    ofn_1 = OFNumber.init_from_scalar(ofn, dim=p.branch_f.dim, domain_x=domain) if isscalar(ofn) else ofn.copy()
    ofn_2 = OFNumber.init_from_scalar(p, dim=ofn.branch_f.dim, domain_x=domain) if isscalar(p) else p.copy()
    
    res = ofn_1.copy()
    res.branch_f = Branch(np.power(ofn_1.branch_f.fvalue_y, ofn_2.branch_f.fvalue_y), domain_x=domain)
    res.branch_g = Branch(np.power(ofn_1.branch_g.fvalue_y, ofn_2.branch_g.fvalue_y), domain_x=domain)
    return res


def fmax(ofn1, ofn2):
    if (not isinstance(ofn1, OFNumber)) and (not isinstance(ofn2, OFNumber)):
        raise ValueError('at least one of arguments must be a OFNumber')
    elif isinstance(ofn1, OFNumber) and isinstance(ofn2, OFNumber):
        assert np.all(ofn1.branch_f.domain_x == ofn2.branch_f.domain_x), 'self.domain_x and val.domain_x must be the same'
        domain = ofn1.branch_f.domain_x
    else:
        domain = ofn1.branch_f.domain_x if isinstance(ofn1, OFNumber) else ofn2.branch_f.domain_x

    ofn_1 = OFNumber.init_from_scalar(ofn1, dim=ofn2.branch_f.dim, domain_x=domain) if isscalar(ofn1) else ofn1.copy()
    ofn_2 = OFNumber.init_from_scalar(ofn2, dim=ofn1.branch_f.dim, domain_x=domain) if isscalar(ofn2) else ofn2.copy()

    res = ofn_1.copy()
    res.branch_f = Branch(np.max([ofn_1.branch_f.fvalue_y, ofn_2.branch_f.fvalue_y], axis=0), domain_x=domain)
    res.branch_g = Branch(np.max([ofn_1.branch_g.fvalue_y, ofn_2.branch_g.fvalue_y], axis=0), domain_x=domain)
    return res


def fmin(ofn1, ofn2):
    if (not isinstance(ofn1, OFNumber)) and (not isinstance(ofn2, OFNumber)):
        raise ValueError('at least one of arguments must be a OFNumber')
    elif isinstance(ofn1, OFNumber) and isinstance(ofn2, OFNumber):
        assert np.all(ofn1.branch_f.domain_x == ofn2.branch_f.domain_x), 'self.domain_x and val.domain_x must be the same'
        domain = ofn1.branch_f.domain_x
    else:
        domain = ofn1.branch_f.domain_x if isinstance(ofn1, OFNumber) else ofn2.branch_f.domain_x

    ofn_1 = OFNumber.init_from_scalar(ofn1, dim=ofn2.branch_f.dim, domain_x=domain) if isscalar(ofn1) else ofn1.copy()
    ofn_2 = OFNumber.init_from_scalar(ofn2, dim=ofn1.branch_f.dim, domain_x=domain) if isscalar(ofn2) else ofn2.copy()

    res = ofn_1.copy()
    res.branch_f = Branch(np.min([ofn_1.branch_f.fvalue_y, ofn_2.branch_f.fvalue_y], axis=0), domain_x=domain)
    res.branch_g = Branch(np.min([ofn_1.branch_g.fvalue_y, ofn_2.branch_g.fvalue_y], axis=0), domain_x=domain)
    return res
