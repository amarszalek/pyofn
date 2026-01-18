# -*- coding: utf-8 -*-

import numpy as np

from pyofn.ofnumber import OFNumber, fmax


def ofnormal(mu, sig2, s2, p):
    """
    Generuje pojedynczą realizację OFRV o rozkładzie normalnym.
    """
    mu_f = mu.branch_f.fvalue_y
    mu_g = mu.branch_g.fvalue_y
    dom = mu.branch_f.domain_x
    dim = len(dom)

    # Przesunięcie do wartości dodatnich
    minv = min(np.min(mu_f), np.min(mu_g))
    c = np.abs(minv) + 1.0 if minv <= 0.0 else 0.0

    eta_f = mu_f + c
    eta_g = mu_g + c

    # Szum multiplikatywny (kształt)
    std_x = np.sqrt(sig2.branch_f.fvalue_y) / eta_f
    std_y = np.sqrt(sig2.branch_g.fvalue_y) / eta_g

    x = np.random.normal(1.0, std_x)
    y = np.random.normal(1.0, std_y)

    # POPRAWKA: s to skalar (stałe przesunięcie dla całej funkcji), nie wektor szumu!
    s_scalar = np.random.normal(0, np.sqrt(s2))
    s = np.full(dim, s_scalar)

    if np.random.random() < p:
        res_f = x * eta_f + s
        res_g = y * eta_g + s
    else:
        res_f = x * eta_g + s
        res_g = y * eta_f + s

    return OFNumber(res_f - c, res_g - c, domain_x=dom)


def ofnormal_sample(n, mu, sig2, s2, p):
    """
    Generuje n próbek OFRV o rozkładzie normalnym.
    """
    # Import wewnątrz funkcji (unikamy cyklu importów)
    import pyofn.ofmodels as ofm

    mu_f = mu.branch_f.fvalue_y
    mu_g = mu.branch_g.fvalue_y
    dom = mu.branch_f.domain_x
    dim = len(dom)

    minv = min(np.min(mu_f), np.min(mu_g))
    c = np.abs(minv) + 1.0 if minv <= 0.0 else 0.0

    eta_f = mu_f + c
    eta_g = mu_g + c

    # Broadcasting parametrów
    sig_f_val = np.sqrt(sig2.branch_f.fvalue_y)
    sig_g_val = np.sqrt(sig2.branch_g.fvalue_y)

    # Generowanie macierzy losowych (N, Dim)
    x = np.random.normal(1.0, sig_f_val / eta_f, size=(n, dim))
    y = np.random.normal(1.0, sig_g_val / eta_g, size=(n, dim))

    # POPRAWKA: S musi być wektorem (N, 1), aby dodać tę samą wartość do wszystkich punktów danej próbki
    s = np.random.normal(0, np.sqrt(s2), size=(n, 1))

    r = np.random.random(n)
    mask = (r < p)[:, np.newaxis]

    # Obliczenie wartości
    term_f_pos = x * eta_f
    term_g_pos = y * eta_g

    term_f_neg = x * eta_g
    term_g_neg = y * eta_f

    # s (N, 1) broadcastuje się do (N, Dim)
    ksi_f = np.where(mask, term_f_pos, term_f_neg) + s - c
    ksi_g = np.where(mask, term_g_pos, term_g_neg) + s - c

    ofns = [OFNumber(ksi_f[i], ksi_g[i], domain_x=dom) for i in range(n)]

    return ofm.OFSeries(ofns)


def ofnormal_mu_est(ofs):
    # E[X]
    # Konwersja do pozytywnej orientacji jest wymagana wg teorii dla estymacji średniej rozkładu
    pofs = ofs.to_positive_order()
    return pofs.mean_fuzzy()


def ofnormal_sig2_est(ofs, ddof=1):
    # Var_fuzzy[X] - s^2
    s2 = ofnormal_s2_est(ofs, ddof=ddof)
    pofs = ofs.to_positive_order()
    # fmax zapewnia nieujemność wyniku (wariancja nie może być ujemna)
    return fmax(pofs.var_fuzzy(ddof=ddof) - s2, 0.0)


def ofnormal_s2_est(ofs, ddof=1):
    # Wariancja składnika skalarnego (szumu białego)
    return ofs.var_crisp(ddof=ddof)
    

def ofnormal_p_est(ofs):
    return ofs.order_probability()


# generate peudo ordered fuzzy random variable with uniform distributon
def ofuniform(mu, sig2, s2, p):
    dom = mu.branch_f.domain_x
    dim = len(dom)

    limit_f = np.sqrt(3 * sig2.branch_f.fvalue_y)
    limit_g = np.sqrt(3 * sig2.branch_g.fvalue_y)
    limit_s = np.sqrt(3 * s2)

    s_f = np.random.uniform(-limit_f, limit_f)
    s_g = np.random.uniform(-limit_g, limit_g)

    # POPRAWKA: Stałe przesunięcie losowe
    s_scalar = np.random.uniform(-limit_s, limit_s)
    s = np.full(dim, s_scalar)

    x = mu.branch_f.fvalue_y + s_f + s
    y = mu.branch_g.fvalue_y + s_g + s

    if np.random.random() < p:
        return OFNumber(x, y, domain_x=dom)
    else:
        return OFNumber(y, x, domain_x=dom)


def ofuniform_sample(n, mu, sig2, s2, p):
    import pyofn.ofmodels as ofm

    dom = mu.branch_f.domain_x
    dim = len(dom)

    limit_f = np.sqrt(3 * sig2.branch_f.fvalue_y)
    limit_g = np.sqrt(3 * sig2.branch_g.fvalue_y)
    limit_s = np.sqrt(3 * s2)

    raw_f = np.random.uniform(-1, 1, size=(n, dim))
    s_f = raw_f * limit_f

    raw_g = np.random.uniform(-1, 1, size=(n, dim))
    s_g = raw_g * limit_g

    # POPRAWKA: Stałe przesunięcie losowe dla każdej próbki
    s = np.random.uniform(-limit_s, limit_s, size=(n, 1))

    r = np.random.random(n)
    mask = (r < p)[:, np.newaxis]

    mu_f = mu.branch_f.fvalue_y
    mu_g = mu.branch_g.fvalue_y

    ksi_f = np.where(mask, mu_f + s_f, mu_g + s_g) + s
    ksi_g = np.where(mask, mu_g + s_g, mu_f + s_f) + s

    ofns = [OFNumber(ksi_f[i], ksi_g[i], domain_x=dom) for i in range(n)]
    return ofm.OFSeries(ofns)

    
ofuniform_mu_est = ofnormal_mu_est
ofuniform_sig2_est = ofnormal_sig2_est
ofuniform_s2_est = ofnormal_s2_est
ofuniform_p_est = ofnormal_p_est
