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
                   'domain_x': 1-dim array or None,
                   'eps': min value for division field a and b,
                   'order': function or None}
    """
    def __init__(self, ctype, data, params):
        data, params = check_inputs(data, params)
        self.price = data['price']
        self.volume = data['volume']
        self.spread = data['spread']
        self.params = params

        if 'order' not in params or params['order'] is None:
            params['order'] = open_close_order

        self.weights = compute_weights(self.price, self.volume, params['wtype'])

        if ctype == 'linear':
            res_data = compute_linear(self.price, self.spread, self.weights, self.params)
        elif ctype == 'gauss':
            res_data = compute_gauss(self.price, self.spread, self.weights, self.params)
        elif ctype == 'empirical':
            res_data = compute_empirical(self.price, self.spread, self.weights, self.params)
        else:
            raise ValueError(f"Unknown candle type: {ctype}")

        branch_f, branch_g, s1, s2, c1, c2, field_a, field_b = res_data
        super(OFCandle, self).__init__(branch_f, branch_g)

        self.param_s = (s1, s2)
        self.param_c = (c1, c2)
        self.param_ab = (field_a, field_b)

    def copy(self):
        return copy.deepcopy(self)

    def to_ofn(self):
        return ofn.OFNumber(self.branch_f, self.branch_g)

    def plot_candle(self):
        fig = plt.figure('OFCandlestick', figsize=(10, 8))
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
        ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=2)

        # Zakresy wykresu
        vals = np.concatenate([self.price, self.branch_f.fvalue_y, self.branch_g.fvalue_y])
        ymax = np.max(vals)
        ymin = np.min(vals)
        margin = (ymax - ymin) * 0.1 if ymax != ymin else 1.0
        ymax += margin
        ymin -= margin

        s1, s2 = self.param_s

        # Panel 1: Cena
        ax1.plot(self.price, 'k-', label='Bid')
        ax1.plot(self.price + self.spread, 'k--', label='Ask', alpha=0.5)
        ax1.set_title('Price History')
        ax1.axhline(s1, c='g', ls='-.', alpha=0.7, label='S1')
        ax1.axhline(s2, c='b', ls='-.', alpha=0.7, label='S2')
        ax1.set_ylim(ymin, ymax)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Panel 2: OFN
        ax2.plot( self.branch_f.domain_x, self.branch_f.fvalue_y, 'r', label='f (UP)')
        ax2.plot( self.branch_g.domain_x, self.branch_g.fvalue_y, 'b', label='g (DOWN)')
        ax2.axhline(s1, c='g', ls='-.')
        ax2.axhline(s2, c='b', ls='-.')
        ax2.set_ylim(ymin, ymax)  # Oś X tutaj to ceny
        ax2.set_title('Fuzzy Profile')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Panel 3: Wolumen
        ax3.bar(np.arange(len(self.volume)), self.volume, color='gray', alpha=0.7)
        ax3.set_title('Volume')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_candle_old(self):
        fig = plt.figure('OFCandlestick', figsize=(10, 8))
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
    required_data = ['price', 'volume', 'spread']
    for field in required_data:
        if field not in data:
            raise ValueError(f'missing field {field} in data')
        data[field] = np.asarray(data[field], dtype=float)

    required_params = ['wtype', 'param_s', 'param_c', 'dim']
    for field in required_params:
        if field not in params:
            raise ValueError(f'missing field {field} in params')

    if 'domain_x' not in params or params['domain_x'] is None:
        params['domain_x'] = np.linspace(0, 1, params['dim'])

    if 'eps' not in params or params['eps'] is None:
        params['eps'] = 1.0e-8

    return data, params


def compute_weights(price, volume, wtype):
    n = len(price)

    if wtype == 'clear':
        return np.ones(n)
    elif wtype == 'lt':  # Linear Time
        return np.arange(1, n + 1, dtype=float)
    elif wtype == 'et':  # Exponential Time
        factor = 2.0 / (1 + n)
        steps = np.arange(n)
        weights = (1 - factor) ** (n - 1 - steps)
        return weights * 100
    elif wtype == 'volume':
        return volume.copy()
    elif isinstance(wtype, list):
        # Kombinacja wag (mnożenie)
        w_total = np.ones(n)
        for wt in wtype:
            w_total *= compute_weights(price, volume, wt)
        return w_total
    else:
        raise ValueError(f'Unknown weight type: {wtype}')


def weighted_avg_and_std(values, weights):
    """Zwraca średnią ważoną i odchylenie standardowe ważone."""
    average = np.average(values, weights=weights)
    # Fast weighted variance: sum(w * (x - mean)^2) / sum(w)
    variance = np.average((values - average)**2, weights=weights)
    return average, np.sqrt(variance)


def weighted_sum_absolute_deviation(values, weights, center):
    """Oblicza sumę ważoną |x - center|."""
    return np.sum(weights * np.abs(values - center))


def weighted_percentile_interp(values, weights, percentile_points):
    """
    Oblicza ważone kwantyle.
    Odpowiednik np.percentile ale dla danych z wagami.
    """
    # Sortowanie
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    computed_quantiles = np.cumsum(weights) - 0.5 * weights
    computed_quantiles /= np.sum(weights)
    return np.interp(percentile_points, computed_quantiles, values)


def compute_params_s(price, spread, weights, stype):
    ask = price + spread

    if stype == 'one_average':
        s1 = np.average(price, weights=weights)
        s2 = np.average(ask, weights=weights)
    elif stype == 'mix_average':
        # Mix of Arithmetic, Linear, Exponential
        n = len(price)
        lw = np.arange(1, n + 1)
        ew = (1 - 2.0 / (1 + n)) ** (n - 1 - np.arange(n))
        avgs_bid = [np.average(price), np.average(price, weights=lw), np.average(price, weights=ew)]
        avgs_ask = [np.average(ask), np.average(ask, weights=lw), np.average(ask, weights=ew)]
        s1 = np.min(avgs_bid)
        s2 = np.max(avgs_ask)
    elif stype == 'close':
        s1 = price[-1]
        s2 = ask[-1]  # Close Ask
    else:
        s1 = np.average(price, weights=weights)
        s2 = np.average(ask, weights=weights)
    return s1, s2


def compute_params_c(price, spread, weights, ctype):
    ask = price + spread

    if ctype == 'none':
        return 0.0, 0.0
    elif ctype == 'std':
        # Weighted STD
        _, c1 = weighted_avg_and_std(price, weights)
        _, c2 = weighted_avg_and_std(ask, weights)
        return c1, c2
    elif ctype == 'volatility':
        # Volatility usually defined on log-returns
        # ln(P_t / P_{t-1})
        if len(price) < 2:
            return 0.0, 0.0
        # Log returns
        lnr_bid = np.log(price[1:] / price[:-1])
        lnr_ask = np.log(ask[1:] / ask[:-1])

        # Weights for returns
        w_ret = weights[1:]

        _, vol_bid = weighted_avg_and_std(lnr_bid, w_ret)
        _, vol_ask = weighted_avg_and_std(lnr_ask, w_ret)

        # Skalowanie o cenę minimalną/maksymalną
        c1 = np.min(price) * vol_bid
        c2 = np.max(ask) * vol_ask
        return c1, c2
    else:
        raise ValueError(f'wrong param_c: {ctype}')


def compute_params_ab_weighted(price, spread, weights, s1, s2):
    """
    Oblicza parametry A i B jako ważone sumy odchyleń od S1 i S2.
    A = Sum_{x > S2} weight * |x - S2| (dla Ask)
    B = Sum_{x < S1} weight * |x - S1| (dla Bid)
    """
    bid = price
    ask = price + spread
    # Dane poniżej s1 (Bid)
    mask_down = bid < s1
    if np.any(mask_down):
        b = np.sum(weights[mask_down] * np.abs(bid[mask_down] - s1))
    else:
        b = 0.0
    # Dane powyżej s2 (Ask)
    mask_up = ask >= s2
    if np.any(mask_up):
        a = np.sum(weights[mask_up] * np.abs(ask[mask_up] - s2))
    else:
        a = 0.0
    return a, b


def compute_linear(price, spread, weights, params):
    # Oblicz parametry statystyczne
    s1, s2 = compute_params_s(price, spread, weights, params['param_s'])
    c1, c2 = compute_params_c(price, spread, weights, params['param_c'])
    field_a, field_b = compute_params_ab_weighted(price, spread, weights, s1, s2)

    # Określ kierunek świecy
    is_positive_trend = params['order'](price)

    # Zbuduj gałęzie liniowe (Trapezoid)
    if is_positive_trend:
        # Long candle (wzrostowa), f: rosnąca od min do S1, g: malejąca od max do S2
        fx0 = np.min(price) - c1
        fx1 = s1
        gx1 = s2
        # Jeśli field_b jest małe, unikamy dzielenia przez zero/błędu
        if field_b < params['eps']:
            gx0 = gx1
        else:
            # Zachowanie proporcji pól: Pole(g)/B = Pole(f)/A
            ratio = field_a / field_b if field_b > 0 else 1.0
            gx0 = ratio * (fx1 - fx0) + gx1

        branch_f = ofn.Branch.init_linear_x0x1(fx0, fx1, dim=params['dim'], domain_x=params['domain_x'])
        branch_g = ofn.Branch.init_linear_x0x1(gx0, gx1, dim=params['dim'], domain_x=params['domain_x'])
    else:
        # Short candle (spadkowa), f: malejąca od max do S2, g: rosnąca od min do S1
        fx0 = np.max(price) + c2
        fx1 = s2
        gx1 = s1
        if field_a < params['eps']:
            gx0 = gx1
        else:
            ratio = field_b / field_a if field_a > 0 else 1.0
            gx0 = ratio * (fx1 - fx0) + gx1

        branch_f = ofn.Branch.init_linear_x0x1(fx0, fx1, dim=params['dim'], domain_x=params['domain_x'])
        branch_g = ofn.Branch.init_linear_x0x1(gx0, gx1, dim=params['dim'], domain_x=params['domain_x'])

    return branch_f, branch_g, s1, s2, c1, c2, field_a, field_b


def compute_gauss(price, spread, weights, params):
    s1, s2 = compute_params_s(price, spread, weights, params['param_s'])
    c1, c2 = compute_params_c(price, spread, weights, params['param_c'])
    field_a, field_b = compute_params_ab_weighted(price, spread, weights, s1, s2)

    is_positive_trend = params['order'](price)

    # Minimalna wartość x w dziedzinie (unikanie log(0))
    xmin = (1.0 / (params['dim'] - 1)) / 10.0

    if is_positive_trend:
        # Long candle
        mf = s1
        # Skalowanie szerokości gaussa tak, by przy xmin osiągał wartość min(price)-c1
        val_at_tail = np.min(price) - c1
        sf = (val_at_tail - mf) / (np.sqrt(-2 * np.log(xmin)))
        mg = s2
        if field_b < params['eps']:
            sg = 0
        else:
            # Skalowanie sg względem sf proporcjonalnie do "masy" A i B
            sg = -(field_a / field_b) * sf

        branch_f = ofn.Branch.init_gauss(mf, sf, dim=params['dim'], domain_x=params['domain_x'])
        branch_g = ofn.Branch.init_gauss(mg, sg, dim=params['dim'], domain_x=params['domain_x'])
    else:
        # Short candle
        mf = s2
        val_at_tail = np.max(price) + c2
        sf = (val_at_tail - mf) / (np.sqrt(-2 * np.log(xmin)))

        mg = s1
        if field_a < params['eps']:
            sg = 0
        else:
            sg = -(field_b / field_a) * sf

        branch_f = ofn.Branch.init_gauss(mf, sf, dim=params['dim'], domain_x=params['domain_x'])
        branch_g = ofn.Branch.init_gauss(mg, sg, dim=params['dim'], domain_x=params['domain_x'])

    return branch_f, branch_g, s1, s2, c1, c2, field_a, field_b


def compute_empirical(price, spread, weights, params):
    """
    Oblicza OFC metodą empiryczną z wykorzystaniem wag (wolumenu).
    Zamiast powielać dane, używamy ważonej dystrybuanty.
    """
    s1, s2 = compute_params_s(price, spread, weights, params['param_s'])
    c1, c2 = compute_params_c(price, spread, weights, params['param_c'])
    field_a, field_b = compute_params_ab_weighted(price, spread, weights, s1, s2)

    bid = price
    ask = price + spread

    is_positive_trend = params['order'](price)

    # Rozdzielamy dane na część "dolną" (pod s1) i "górną" (nad s2)
    # Zbiór 1: Bid <= S1
    mask1 = bid <= s1
    if np.sum(mask1) == 0:
        # Fallback if empty
        vals1 = np.array([s1])
        ws1 = np.array([1.0])
    else:
        vals1 = bid[mask1]
        ws1 = weights[mask1]
        # Dodajemy punkt skrajny (ogon)
        vals1 = np.append(vals1, np.min(bid) - c1)
        ws1 = np.append(ws1, np.mean(weights))  # Waga dla ogona (średnia)

    # Zbiór 2: Ask >= S2
    mask2 = ask >= s2
    if np.sum(mask2) == 0:
        vals2 = np.array([s2])
        ws2 = np.array([1.0])
    else:
        vals2 = ask[mask2]
        ws2 = weights[mask2]
        # Dodajemy punkt skrajny
        vals2 = np.append(vals2, np.max(ask) + c2)
        ws2 = np.append(ws2, np.mean(weights))

    # Generujemy wartości dla dziedziny X [0, 1] używając ważonych kwantyli
    domain_x = params['domain_x']

    # Obliczamy wartości "odwróconej dystrybuanty"
    # Dla zbioru 1 (lewa strona/dół): od min do S1
    # Dla zbioru 2 (prawa strona/góra): od S2 do max

    y1_values = weighted_percentile_interp(vals1, ws1, domain_x)
    y2_values = weighted_percentile_interp(vals2, ws2, domain_x)

    # Należy upewnić się, że końce pasują do S1 i S2 w punktach x=1 (lub x=0 zależnie od orientacji)
    # W OFN zwykle x=1 to jądro (S1, S2), a x=0 to nośnik (skrajne)

    # weighted_percentile_interp zwraca wartości posortowane rosnąco.
    # y1_values[0] ~ min, y1_values[-1] ~ S1
    # y2_values[0] ~ S2,  y2_values[-1] ~ max

    if is_positive_trend:
        # f: rosnąca (bid side) -> y1
        # g: malejąca (ask side) -> y2 (musimy odwrócić logicznie dla OFN)

        # Branch f (rosnąca):
        # x=0 -> min, x=1 -> S1. y1_values są rosnące, pasuje.
        yf = y1_values

        # Branch g (malejąca):
        # x=0 -> max, x=1 -> S2. y2_values są rosnące (S2 -> max).
        # Musimy wziąć y2 i je odwrócić, aby dla x=0 (początek tablicy brancha) była wartość max
        yg = y2_values[::-1]

    else:
        # Negative trend
        # f: malejąca (zbiór 2, ask side, max -> S2)
        # g: rosnąca (zbiór 1, bid side, min -> S1)

        # Branch f (malejąca):
        # x=0 -> max, x=1 -> S2. y2_values (S2->max). Odwracamy.
        yf = y2_values[::-1]

        # Branch g (rosnąca):
        # x=0 -> min, x=1 -> S1. y1_values (min->S1). Pasuje.
        yg = y1_values

    # Wymuszenie dokładnych wartości w jądrze (x=1)
    # yf[-1] musi być S1 (long) lub S2 (short)
    # yg[-1] musi być S2 (long) lub S1 (short)
    if is_positive_trend:
        yf[-1] = s1
        yg[-1] = s2
    else:
        yf[-1] = s2
        yg[-1] = s1

    branch_f = ofn.Branch(yf, domain_x=domain_x)
    branch_g = ofn.Branch(yg, domain_x=domain_x)

    return branch_f, branch_g, s1, s2, c1, c2, field_a, field_b


def compute_empirical_old(price, volume, spread, params):
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


def open_close_order(price):
    if len(price) < 2: return True
    return price[-1] >= price[0]


def amplitude_order(x):
    # Prosta regresja liniowa na amplitudzie obwiedni Hilberta
    try:
        analytic_signal = signal.hilbert(x)
        amplitude_envelope = np.abs(analytic_signal)
        slope, _, _, _, _ = stats.linregress(np.arange(len(x)), amplitude_envelope)
        return slope >= 0
    except Exception:
        # Fallback jeśli sygnał jest stały lub zbyt krótki
        return True
