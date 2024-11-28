# -*- coding: utf-8 -*-

import numpy as np
import copy
import pyofn.ofnumber as ofn
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats


class OFCandle(ofn.OFNumber):
    """
    ctype : str, linear or gauss or empirical
    data : dic, {'price': bid price,
                 'volume': volume,
                 'spread': ask - bid}
    params : dic, {'wtype': 'clear', 'lt', 'et', 'volume', comb ..,
                   'param_s': 'average', 'lw_average', 'ew_average', 'mix_average',
                   'param_c': 'none', 'std', 'volatility',
                   'dim': int,
                   'domain_x': 1-dim array or None
                   'order': function or None}
    """
    def __init__(self, ctype, data, params):
        data, params = check_inputs(data, params)
        self.price = data['price']
        self.volume = data['volume']
        self.spread = data['spread']
        self.params = params

        if params['order'] is None:
            params['order'] = open_close_order

        if ctype == 'linear':
            ofc, s1, s2, c1, c2, field_a, field_b = compute_linear(self.price, self.volume, self.spread, self.params)
        elif ctype == 'gauss':
            pass
            ofc, s1, s2, c1, c2, field_a, field_b = compute_gauss(self.price, self.volume, self.spread, self.params)
        elif ctype == 'empirical':
            pass
            ofc, s1, s2, c1, c2, field_a, field_b = compute_empirical(self.price, self.volume, self.spread, self.params)
        else:
            raise ValueError('wrong candle type')

        super(OFCandle, self).__init__(ofc.branch_f, ofc.branch_g)
        self.param_s = (s1, s2)
        self.param_c = (c1, c2)
        self.param_ab = (field_a, field_b)

    def copy(self):
        return copy.deepcopy(self)

    def to_ofn(self):
        return ofn.OFNumber(self.branch_f, self.branch_g)

    def plot_candle(self):
        fig = plt.figure('OFCandlestick', figsize=(8, 8))
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
        ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=2)

        ymax = np.max([np.max(self.price), self.branch_f.fvalue_y[0], self.branch_g.fvalue_y[0]])
        ymin = np.min([np.min(self.price), self.branch_f.fvalue_y[0], self.branch_g.fvalue_y[0]])
        ymax = ymax + (ymax-ymin)*0.1
        ymin = ymin - (ymax-ymin)*0.1
        s1, s2 = self.param_s

        ax1.plot(self.price, 'k-', label='price (Bid)')
        ax1.plot(self.price + self.spread, 'k--', label='price (Ask)')
        ax1.set_title('Price')
        ax1.axhline(s1, c='g', ls='-.')
        ax1.axhline(s2, c='g', ls='-.')

        ax1.set_ylim(ymin, ymax)
        ax1.legend()
        ax2.plot(self.branch_f.domain_x, self.branch_f.fvalue_y, 'r', label='$f$')
        ax2.plot(self.branch_g.domain_x, self.branch_g.fvalue_y, 'b', label='$g$')
        ax2.axhline(s1, c='g', ls='-.')
        ax2.axhline(s2, c='g', ls='-.')
        ax2.set_ylim(ymin, ymax)
        ax2.set_title('OFNumber')
        ax2.legend()
        ax3.plot(self.volume)
        ax3.set_title('Volume')
        plt.tight_layout()
        plt.show()


def check_inputs(data, params):
    data_fields = ['price', 'volume', 'spread']
    for field in data_fields:
        if field in data:
            if not isinstance(data[field], np.ndarray):
                data[field] = np.array(data[field])
        else:
            raise ValueError('missing field ' + field + ' in data')
    params_fields = ['wtype', 'param_s', 'param_c', 'dim', 'domain_x']
    for field in params_fields:
        if field not in params:
            raise ValueError('missing field ' + field + ' in params')
    return data, params


def get_weighted_data(price, volume, spread, wtype):
    weights = np.ones(len(price))
    if isinstance(wtype, list):
        weights = np.ones(len(price))
        for wt in wtype:
            weights = weights * get_weights(price, volume, wt)
    elif isinstance(wtype, str):
        weights = get_weights(price, volume, wtype)
    else:
        ValueError('wrong weight type')

    maxw = np.max(weights)
    minw = np.min(weights)
    if maxw == minw:
        weights = np.ones(len(price))
    else:
        weights = 99.0 / (maxw - minw) * (weights - minw) + 1.0
    new_price = np.repeat(price, weights.astype(int))
    new_spread = np.repeat(spread, weights.astype(int))
    return new_price, new_spread


def get_weights(price, volume, wtype):
    d = len(price)
    weights = None
    if wtype == 'clear':
        weights = np.ones(d)
    elif wtype == 'lt':
        weights = np.arange(1, d + 1)
    elif wtype == 'et':
        factor = 2.0 / (1 + d)
        weights = np.linspace(0, d - 1, d)
        weights = ((1 - factor) ** (d - weights - 1)) * 100 + 1
    elif wtype == 'volume':
        weights = volume
    else:
        ValueError('wrong weight type')
    return weights


def compute_linear(price, volume, spread, params):
    new_price, new_spread = get_weighted_data(price, volume, spread, params['wtype'])
    s1, s2 = compute_params_s(new_price, new_spread, params['param_s'])
    c1, c2 = compute_params_c(new_price, new_spread, params['param_c'])
    field_a, field_b = compute_params_ab(new_price, new_spread, s1, s2)
    if params['order'](new_price):
        fx0 = np.min(new_price) - c1
        fx1 = s1
        gx1 = s2
        gx0 = (field_a / field_b) * (fx1 - fx0) + gx1
        ofc = ofn.OFNumber.init_trapezoid_x0x1(fx0, fx1, gx0, gx1, dim=params['dim'], domain_x=params['domain_x'])
    else:
        fx0 = np.max(new_price) + c2
        fx1 = s2
        gx1 = s1
        gx0 = (field_b / field_a) * (fx1 - fx0) + gx1
        ofc = ofn.OFNumber.init_trapezoid_x0x1(fx0, fx1, gx0, gx1, dim=params['dim'], domain_x=params['domain_x'])
    return ofc, s1, s2, c1, c2, field_a, field_b


def compute_gauss(price, volume, spread, params):
    new_price, new_spread = get_weighted_data(price, volume, spread, params['wtype'])
    s1, s2 = compute_params_s(new_price, new_spread, params['param_s'])
    c1, c2 = compute_params_c(new_price, new_spread, params['param_c'])
    field_a, field_b = compute_params_ab(new_price, new_spread, s1, s2)
    xmin = (1.0 / (params['dim'] - 1)) / 10
    if params['order'](new_price):
        mf = s1
        sf = (np.min(new_price) - c1 - s1) / (np.sqrt(-2 * np.log(xmin)))
        mg = s2
        sg = -(field_a / field_b) * sf
        ofc = ofn.OFNumber.init_gaussian(mf, sf, mg, sg, dim=params['dim'], domain_x=params['domain_x'])
    else:
        mf = s2
        sf = (np.max(new_price) + c1 - s2) / (np.sqrt(-2 * np.log(xmin)))
        mg = s1
        sg = -(field_b / field_a) * sf
        ofc = ofn.OFNumber.init_gaussian(mf, sf, mg, sg, dim=params['dim'], domain_x=params['domain_x'])
    return ofc, s1, s2, c1, c2, field_a, field_b


def compute_empirical(price, volume, spread, params):
    dim = params['dim']
    new_price, new_spread = get_weighted_data(price, volume, spread, params['wtype'])
    if dim > len(price):
        raise ValueError('dim > len(price)')
    s1, s2 = compute_params_s(new_price, new_spread, params['param_s'])
    c1, c2 = compute_params_c(new_price, new_spread, params['param_c'])
    field_a, field_b = compute_params_ab(new_price, new_spread, s1, s2)

    data_bid = np.copy(new_price)
    data_ask = np.copy(new_price + new_spread)
    data_bid = np.sort(data_bid)
    data_ask = np.sort(data_ask)

    y1 = data_bid[data_bid <= s1]
    y2 = data_ask[data_ask >= s2]
    if len(y1) == 1:
        y1 = np.full(dim, y1[0])
    if len(y2) == 1:
        y2 = np.full(dim, y2[0])

    step_c1 = c1 / (len(y1) - 1)
    step_c2 = c2 / (len(y2) - 1)
    y1 -= step_c1 * np.arange(len(y1) - 1, -1, -1)
    y2 += step_c2 * np.arange(len(y2))
    y1 = np.append(y1, s1)
    y2 = np.insert(y2, 0, s2)

    domain_x = params['domain_x'] if params['domain_x'] is not None else np.linspace(0, 1, dim)

    #step1 = len(y1) / (dim - 1.0)
    steps1 = domain_x * (len(y1)-1)  # 0, ..., len
    #step2 = len(y2) / (dim - 1.0)
    steps2 = domain_x * (len(y2)-1)
    if params['order'](new_price):
        #mask = np.arange(dim - 1) * step1
        #yf = y1[mask.astype(int)]
        yf = y1[steps1.astype(int)]
        #yf = np.append(yf, y1[-1])
        #mask = len(y2) - 1 - (np.arange(dim - 1) * step2)
        #yg = y2[mask.astype(int)]
        yg = y2[steps2[::-1].astype(int)]
        #yg = np.append(yg, y2[0])
        ofc = ofn.OFNumber(yf, yg, domain_x=params['domain_x'])
    else:
        #mask = np.arange(dim - 1) * step1
        #yg = y1[mask.astype(int)]
        yg = y1[steps1.astype(int)]
        #yg = np.append(yg, y1[-1])
        #mask = len(y2) - 1 - (np.arange(dim - 1) * step2)
        #yf = y2[mask.astype(int)]
        yf = y2[steps2[::-1].astype(int)]
        #yf = np.append(yf, y2[0])
        ofc = ofn.OFNumber(yf, yg, domain_x=params['domain_x'])
    return ofc, s1, s2, c1, c2, field_a, field_b


def compute_params_s(price, spread, stype):
    if stype == 'average':
        s1 = np.average(price)
        s2 = np.average(price + spread)
    elif stype == 'lw_average':
        s1 = lw_average(price)
        s2 = lw_average(price + spread)
    elif stype == 'ew_average':
        s1 = ew_average(price)
        s2 = ew_average(price + spread)
    elif stype == 'mix_average':
        a = np.average(price)
        b = lw_average(price)
        c = ew_average(price)
        s1 = np.min([a, b, c])
        a = np.average(price + spread)
        b = lw_average(price + spread)
        c = ew_average(price + spread)
        s2 = np.max([a, b, c])
    else:
        raise ValueError('wrong param_s')
    return s1, s2


def compute_params_c(price, spread, ctype):
    if ctype == 'none':
        c1 = 0.0
        c2 = c1
    elif ctype == 'std':
        c1 = np.std(price)
        c2 = np.std(price + spread)
    elif ctype == 'volatility':
        a = volatility(price)
        b = volatility(price + spread)
        c1 = np.min(price) * a
        c2 = np.max(price + spread) * b
    else:
        raise ValueError('wrong param_c')
    return c1, c2


def compute_params_ab(price, spread, s1, s2):
    data_bid = np.copy(price)
    data_ask = np.copy(price + spread)
    data_down = data_bid[data_bid < s1]
    data_up = data_ask[data_ask >= s2]
    data_down = np.abs(data_down - s1)
    data_up = np.abs(data_up - s2)
    a = np.sum(data_up)
    b = np.sum(data_down)
    return a, b


def lw_average(data):
    d = len(data)
    m = np.average(data, weights=np.arange(1, d + 1))
    return m


def ew_average(data):
    d = len(data)
    factor = 2.0 / (1 + d)
    weights = np.linspace(0, d - 1, d)
    weights = (1 - factor) ** (d - weights - 1)
    m = np.average(data, weights=weights)
    return m


def volatility(data):
    d = len(data)
    lnr = np.log(data[1:d] / data[:d - 1])
    m = np.std(lnr)
    return m


def open_close_order(x):
    return True if x[0] <= x[-1] else False


def amplitude_order(x):
    analytic_signal = signal.hilbert(x)
    amplitude_envelope = np.abs(analytic_signal)
    slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(amplitude_envelope)), amplitude_envelope)
    return True if slope >= 0 else False
