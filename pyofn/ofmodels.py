# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
#import ofrandom as ofr
from copy import deepcopy
from pyofn.ofnumber import OFNumber, flog, fpower
from scipy.optimize import minimize
from collections import deque

#try:
#    from arch import arch_model
#except ModuleNotFoundError:
#    print('ARCH module not found')


class OFSeries(object):
    def __init__(self, ofns):
        super(OFSeries, self).__init__()
        self.values = np.array(ofns, dtype=object)
        
    def copy(self):
        return deepcopy(self)
    
    def __getitem__(self, i):
        return self.values[i]
    
    def __setitem__(self, i, ofn):
        self.values[i] = ofn

    def __len__(self):
        return len(self.values)
        
    def plot_ofseries(self, ax, s=0, e=None, color='black', shift=0, ord_method='expected'):
        if e is None:
            ofns = self[s:]
        else:
            ofns = self[s:e]
        for i, ofn in enumerate(ofns):
            f, g = ofn.branch_f, ofn.branch_g
            x = f.domain_x*0.5 + i-0.25 + shift
            o = ofn.order(method=ord_method)
            if o >= 0:
                ax.fill_between(x, f.fvalue_y, g.fvalue_y, facecolor='white', edgecolor='black')
            else:
                ax.fill_between(x, f.fvalue_y, g.fvalue_y, facecolor=color, edgecolor='black')
        ax.set_xlim(-1+shift, len(ofns)+shift)

    def apply(self, func, args=(), otypes=None):
        fv = np.vectorize(lambda x: func(x, *args), otypes=otypes)
        return OFSeries(fv(self.values))

    def to_positive_order(self, method='expected', args=()):
        fv = np.vectorize(lambda x: x if x.order(method=method, args=args) >= 0.0 else x.change_order(),
                          otypes=[OFNumber])
        return OFSeries(fv(self.values))
        
    def to_array(self, stack='vstack'): 
        fv = np.vectorize(lambda x: x.to_array(stack=stack), otypes=[np.ndarray])
        return fv(self.values)
    
    def defuzzy(self, method='scog', args=(0.5,)):
        fv = np.vectorize(lambda x: x.defuzzy(method=method, args=args), otypes=[np.double])
        return fv(self.values)
    
    def order(self, method='scog', args=(0.5,)):
        fv = np.vectorize(lambda x: x.order(method=method, args=args), otypes=[np.int])
        return fv(self.values)
    
    def mean_fuzzy(self):
        x = np.mean(self.to_array())
        return OFNumber(x[0], x[1])
    
    def mean_crisp(self):
        mu = self.mean_fuzzy()
        return mu.defuzzy(method='expected')
    
    def var_fuzzy(self, ddof=1):
        x = np.var(self.to_array(), ddof=ddof)
        return OFNumber(x[0], x[1])
    
    def var_crisp(self, ddof=1):
        defuzz = self.defuzzy(method='expected')
        return np.var(defuzz, ddof=ddof)
    
    def order_probability(self):
        ords = self.order(method='expected')
        return ords[ords >= 0].sum()/ords.shape[0]
    
    def plot_histogram(self, ax_f, ax_g, alpha, bins=20, density=False, s=0, e=None, kwargs_f=None, kwargs_g=None):
        if kwargs_f is None:
            kwargs_f = {}
        if kwargs_g is None:
            kwargs_g = {}
        if e is None:
            ofns = self.values[s:]
        else:
            ofns = self.values[s:e]
        fv_f = np.vectorize(lambda x: x.branch_f(alpha), otypes=[np.double])
        fv_g = np.vectorize(lambda x: x.branch_g(alpha), otypes=[np.double])
        data_f = fv_f(ofns)
        data_g = fv_g(ofns)
        ax_f.hist(data_f, bins=bins, density=density, **kwargs_f)
        ax_g.hist(data_g, bins=bins, density=density, **kwargs_g)

    def plot_3d_histogram(self, ax_f, ax_g, alphas=np.linspace(0, 1, 11), bins=20, density=False, s=0, e=None,
                          kwargs_f=None, kwargs_g=None, true_param=None):
        def normal_f(xx, a, p, m, sig2, s2):
            y_f = p*(1./np.sqrt(np.pi*2*(sig2.branch_f(a)+s2)))*np.exp((-(xx-m.branch_f(a))**2)/(2*(sig2.branch_f(a)+s2)))
            y_f += (1-p)*(1./np.sqrt(np.pi*2*(sig2.branch_g(a)+s2)))*np.exp((-(xx-m.branch_g(a))**2)/(2*(sig2.branch_g(a)+s2)))
            return y_f

        def normal_g(xx, a, p, m, sig2, s2):
            y_g = p*(1./np.sqrt(np.pi*2*(sig2.branch_g(a)+s2)))*np.exp((-(xx-m.branch_g(a))**2)/(2*(sig2.branch_g(a)+s2)))
            y_g += (1-p)*(1./np.sqrt(np.pi*2*(sig2.branch_f(a)+s2)))*np.exp((-(xx-m.branch_f(a))**2)/(2*(sig2.branch_f(a)+s2)))
            return y_g

        if kwargs_f is None:
            kwargs_f = {}
        if kwargs_g is None:
            kwargs_g = {}
        if e is None:
            ofns = self.values[s:]
        else:
            ofns = self.values[s:e]
        b_f_min = b_g_min = 10000
        b_f_max = b_g_max = -10000
        for a in alphas:
            fv_f = np.vectorize(lambda x: x.branch_f(a), otypes=[np.double])
            fv_g = np.vectorize(lambda x: x.branch_g(a), otypes=[np.double])
            data_f = fv_f(ofns)
            data_g = fv_g(ofns)
            h_f, b_f = np.histogram(data_f, bins=bins, density=density)
            b_f = (b_f[:-1] + b_f[1:]) / 2.
            h_g, b_g = np.histogram(data_g, bins=bins, density=density)
            b_g = (b_g[:-1] + b_g[1:]) / 2.
            ax_f.bar(b_f, h_f, zs=a, zdir='y', alpha=0.8, **kwargs_f)
            ax_g.bar(b_g, h_g, zs=a, zdir='y', alpha=0.8, **kwargs_g)
            if np.min(b_f) < b_f_min:
                b_f_min = np.min(b_f)
            if np.min(b_g) < b_g_min:
                b_g_min = np.min(b_g)
            if np.max(b_f) > b_f_max:
                b_f_max = np.max(b_f)
            if np.max(b_g) > b_g_max:
                b_g_max = np.max(b_g)
        if true_param is not None:
            for a in alphas:           
                xx_f = np.linspace(b_f_min, b_f_max, 100)
                y_f = normal_f(xx_f,a,*true_param)
                xx_g = np.linspace(b_g_min, b_g_max, 100)
                y_g = normal_g(xx_g,a,*true_param)
                ax_f.plot(xx_f, y_f, 'k--', zs=a-0.01, zdir='y', zorder=2000, alpha=1.0)
                ax_g.plot(xx_g, y_g, 'k--', zs=a-0.01, zdir='y', zorder=2000, alpha=1.0)
                
        ax_f.set_xlabel('$f(x)$')
        ax_f.set_ylabel('$x$')
        ax_f.set_zlabel('frequency')
        ax_g.set_xlabel('$g(x)$')
        ax_g.set_ylabel('$x$')
        ax_g.set_zlabel('frequency')
        
    def transform(self, method='diff'):
        arr = np.copy(self.values)
        if method == 'diff':
            new_ofns = arr[1:]-arr[:-1]
        elif method == 'ret':
            new_ofns = (arr[1:]-arr[:-1])/arr[:-1]
        elif method == 'logret':
            fv = np.vectorize(lambda x: flog(x), otypes=[OFNumber])
            new_ofns = fv(arr[1:]/arr[:-1])
        else:
            raise ValueError('method must be diff, ret or logret')
        return OFSeries(new_ofns)



