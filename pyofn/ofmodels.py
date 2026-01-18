# -*- coding: utf-8 -*-

import numpy as np

from copy import deepcopy
from pyofn.ofnumber import OFNumber, flog, fpower


class OFSeries(object):
    def __init__(self, ofns):
        super(OFSeries, self).__init__()
        self.values = np.array(ofns, dtype=object)

        # zakładamy, że w szeregu czasowym dziedzina jest stała dla wszystkich elementów
        if len(self.values) > 0:
            self._common_domain = self.values[0].branch_f.domain_x
            self._dim = self.values[0].branch_f.dim
        else:
            self._common_domain = None
            self._dim = None
        
    def copy(self):
        return deepcopy(self)
    
    def __getitem__(self, i):
        if isinstance(i, slice):
            return OFSeries(self.values[i])
        return self.values[i]
    
    def __setitem__(self, i, ofn):
        self.values[i] = ofn

    def __len__(self):
        return len(self.values)

    def to_numpy_3d(self):
        """
        Konwertuje serię obiektów do macierzy 3D NumPy (N_samples, 2, Dim).
        """
        if len(self) == 0:
            return np.array([])
        # Wynik: (N, 2, Dim)
        return np.array([ofn.to_array(include_domain=False) for ofn in self.values])

    def _from_numpy_3d(self, data_3d):
        """
        Rekonstruuje OFSeries z macierzy 3D (N, 2, Dim).
        """
        dom = self._common_domain
        new_ofns = []
        for i in range(data_3d.shape[0]):
            f_vals = data_3d[i, 0, :]
            g_vals = data_3d[i, 1, :]
            new_ofns.append(OFNumber(f_vals, g_vals, domain_x=dom))
        return OFSeries(new_ofns)

    def mean_fuzzy(self):
        """Oblicza średnią rozmytą (OFN) całego szeregu."""
        if len(self) == 0: return None
        data = self.to_numpy_3d()  # (N, 2, Dim)
        mean_vals = np.mean(data, axis=0)  # (2, Dim)
        return OFNumber(mean_vals[0], mean_vals[1], domain_x=self._common_domain)

    def mean_crisp(self, method='expected', args=(0.5,)):
        """Zwraca skalarną średnią (zdefuzyfikowaną)."""
        mu = self.mean_fuzzy()
        return mu.defuzzy(method=method, args=args)

    def var_fuzzy(self, ddof=1):
        """Oblicza wariancję rozmytą (OFN)."""
        if len(self) == 0: return None
        data = self.to_numpy_3d()
        var_vals = np.var(data, axis=0, ddof=ddof)
        return OFNumber(var_vals[0], var_vals[1], domain_x=self._common_domain)

    def var_crisp(self, ddof=1, method='expected', args=(0.5,)):
        """Oblicza wariancję na podstawie zdefuzyfikowanych wartości."""
        defuzz_vals = self.defuzzy(method=method, args=args)
        return np.var(defuzz_vals, ddof=ddof)

    def order_probability(self, method='expected', args=(0.5,)):
        """Prawdopodobieństwo wystąpienia pozytywnej orientacji."""
        if len(self) == 0: return 0.0
        ords = self.order(method='expected', args=args)
        return np.sum(ords >= 0) / len(ords)

    def apply(self, func, args=(), otypes=None):
        """Aplikuje funkcję do każdego elementu serii."""
        fv = np.vectorize(lambda x: func(x, *args), otypes=otypes)
        return OFSeries(fv(self.values))

    def defuzzy(self, method='scog', args=(0.5,)):
        """Zwraca wektor numpy ze zdefuzyfikowanymi wartościami."""
        fv = np.vectorize(lambda x: x.defuzzy(method=method, args=args), otypes=[np.double])
        return fv(self.values)

    def order(self, method='scog', args=(0.5,)):
        """Zwraca wektor intów z orientacją (1, -1, 0)."""
        fv = np.vectorize(lambda x: x.order(method=method, args=args), otypes=[int])
        return fv(self.values)

    def to_positive_order(self, method='expected', args=(0.5,)):
        """Zwraca nowy szereg, w którym wszystkie liczby mają pozytywną orientację."""
        orders = self.order(method=method, args=args)
        new_vals = self.values.copy()

        mask = (orders < 0)
        if np.any(mask):
            convert_func = np.vectorize(lambda x: x.change_order(), otypes=[object])
            new_vals[mask] = convert_func(new_vals[mask])
        return OFSeries(new_vals)

    def transform(self, method='diff'):
        """
        Przekształca szereg (np. ceny na zwroty).
        """
        data = self.to_numpy_3d()  # (N, 2, Dim)

        if method == 'diff':
            # X_t - X_{t-1}
            diffs = data[1:] - data[:-1]
            return self._from_numpy_3d(diffs)
        elif method == 'ret':
            # (X_t - X_{t-1}) / X_{t-1}
            prev = data[:-1]
            curr = data[1:]
            with np.errstate(divide='ignore', invalid='ignore'):
                rets = (curr - prev) / prev
            return self._from_numpy_3d(rets)
        elif method == 'logret':
            # log(X_t) - log(X_{t-1}) = log(X_t / X_{t-1})
            if np.any(data <= 0):
                raise ValueError("Log returns require strictly positive values")
            log_data = np.log(data)
            diff_logs = log_data[1:] - log_data[:-1]
            return self._from_numpy_3d(diff_logs)
        else:
            raise ValueError('method must be diff, ret or logret')

    def distance_to_mean(self):
        """
        Oblicza odległość każdego elementu od średniej rozmytej szeregu.
        """
        mu = self.mean_fuzzy()
        fv = np.vectorize(lambda x: x.distance(mu), otypes=[float])
        return fv(self.values)

    def plot_ofseries(self, ax, s=0, e=None, color_up='white', color_down='black', shift=0, ord_method='expected', ord_args=(0.5,)):
        """
        Wizualizuje szereg czasowy jako sekwencję świec OFN.
        """
        if e is None:
            ofns = self.values[s:]
        else:
            ofns = self.values[s:e]

        n = len(ofns)
        if n == 0: return

        orders = self.order(method=ord_method)[s:e] if e else self.order(method=ord_method)[s:]
        dom = self._common_domain

        # Rysowanie
        # Dla <1000 świec pętla jest akceptowalna.

        for i, ofn in enumerate(ofns):
            f = ofn.branch_f.fvalue_y
            g = ofn.branch_g.fvalue_y

            # Skalowanie domeny lokalnej (0..1) na oś czasu globalną (i..i+1)
            # Świeca scentrowana na i, szerokość np. 0.8
            width_scale = 0.8
            x_mapped = (dom - 0.5) * width_scale + (i + shift)
            fill_color = color_up if orders[i] >= 0 else color_down
            ax.fill_between(x_mapped, f, g, facecolor=fill_color, edgecolor='black', linewidth=0.5, alpha=0.9)

        ax.set_xlim(-1 + shift, n + shift)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Value')

    def _get_values_at_alpha(self, vals_3d, alpha):
        """
        Wewnętrzna metoda do szybkiej, wektorowej interpolacji wartości f i g
        dla zadanego alpha na podstawie macierzy 3D.
        """
        dom = self._common_domain

        # Znajdź indeksy sąsiadujące
        idx = np.searchsorted(dom, alpha)

        # Obsługa brzegów (alpha poza dziedziną lub na granicy)
        if idx == 0:
            return vals_3d[:, 0, 0], vals_3d[:, 1, 0]
        if idx >= len(dom):
            return vals_3d[:, 0, -1], vals_3d[:, 1, -1]

        idx_L = idx - 1
        idx_R = idx

        x_L = dom[idx_L]
        x_R = dom[idx_R]

        # Oblicz wagę interpolacji
        if x_R == x_L:
            w = 0.0
        else:
            w = (alpha - x_L) / (x_R - x_L)

        # Interpolacja dla f (indeks 0)
        # vals_3d[:, 0, idx] to wektor wartości f dla wszystkich próbek w punkcie idx
        f_vals = vals_3d[:, 0, idx_L] * (1 - w) + vals_3d[:, 0, idx_R] * w

        # Interpolacja dla g (indeks 1)
        g_vals = vals_3d[:, 1, idx_L] * (1 - w) + vals_3d[:, 1, idx_R] * w

        return f_vals, g_vals

    def plot_histogram(self, ax_f, ax_g, alpha, bins=20, density=False, s=0, e=None, kwargs_f={}, kwargs_g={}):
        """
        Rysuje histogram wartości f(alpha) i g(alpha) dla szeregu (lub jego wycinka).
        Obsługuje dowolne alpha poprzez interpolację liniową.
        """
        subset = self[s:e] if e is not None else self[s:]

        if len(subset) == 0:
            return

        # Konwersja do numpy (N_subset, 2, Dim)
        vals = subset.to_numpy_3d()
        # Pobranie interpolowanych wartości
        data_f, data_g = self._get_values_at_alpha(vals, alpha)

        # Rysowanie
        ax_f.hist(data_f, bins=bins, density=density, alpha=0.7, label=f'f(alpha={alpha:.2f})', **kwargs_f)
        ax_g.hist(data_g, bins=bins, density=density, alpha=0.7, label=f'g(alpha={alpha:.2f})', **kwargs_g)

        ax_f.set_title(f"Histogram f(alpha={alpha:.2f})")
        ax_g.set_title(f"Histogram g(alpha={alpha:.2f})")
        ax_f.legend()
        ax_g.legend()

    def plot_3d_histogram(self, ax_f, ax_g, alphas=np.linspace(0, 1, 11), bins=20, density=True, s=0, e=None,
                          true_param=None):
        """
        Rysuje histogramy 3D. Wykorzystuje interpolację dla precyzyjnego
        odwzorowania wartości na zadanych poziomach alpha.
        """

        def normal_f_pdf(x, a, p, mu_ofn, var_ofn, sigma_sq):
            # Pobieramy parametry teoretyczne w punkcie a (też interpolowane!)
            # Dla uproszczenia tutaj używamy metody __call__ obiektu OFNumber (branch_f(a))
            # ponieważ mu_ofn to pojedynczy obiekt, nie seria.
            mf = mu_ofn.branch_f(a)
            mg = mu_ofn.branch_g(a)
            vf = var_ofn.branch_f(a)
            vg = var_ofn.branch_g(a)

            s_f = np.sqrt(vf + sigma_sq)
            s_g = np.sqrt(vg + sigma_sq)

            y = p * (1 / (s_f * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mf) / s_f) ** 2)
            y += (1 - p) * (1 / (s_g * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mg) / s_g) ** 2)
            return y

        def normal_g_pdf(x, a, p, mu_ofn, var_ofn, sigma_sq):
            mf = mu_ofn.branch_f(a)
            mg = mu_ofn.branch_g(a)
            vf = var_ofn.branch_f(a)
            vg = var_ofn.branch_g(a)

            s_f = np.sqrt(vf + sigma_sq)
            s_g = np.sqrt(vg + sigma_sq)

            y = p * (1 / (s_g * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mg) / s_g) ** 2)
            y += (1 - p) * (1 / (s_f * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mf) / s_f) ** 2)
            return y

        # Przygotowanie danych
        subset = self[s:e] if e is not None else self[s:]
        if len(subset) == 0: return

        vals_3d = subset.to_numpy_3d()  # (N, 2, Dim)

        if true_param is not None:
            p, mu_ofn, var_ofn, sigma_sq = true_param

        # Iteracja po alphas
        for alpha in alphas:
            df, dg = self._get_values_at_alpha(vals_3d, alpha)

            # Histogramy
            hf, bf = np.histogram(df, bins=bins, density=density)
            hg, bg = np.histogram(dg, bins=bins, density=density)

            # Środki binów
            bf_centers = (bf[:-1] + bf[1:]) / 2
            bg_centers = (bg[:-1] + bg[1:]) / 2

            width_f = bf[1] - bf[0]
            width_g = bg[1] - bg[0]

            ax_f.bar(bf_centers, hf, zs=alpha, zdir='y', width=width_f, alpha=0.6, color='r', edgecolor='none')
            ax_g.bar(bg_centers, hg, zs=alpha, zdir='y', width=width_g, alpha=0.6, color='b', edgecolor='none')

            # Krzywa teoretyczna
            if true_param is not None:
                # Rysujemy tylko jeśli mamy dane w zakresie
                if len(df) > 0:
                    xf_line = np.linspace(np.min(df), np.max(df), 100)
                    yf_line = normal_f_pdf(xf_line, alpha, p, mu_ofn, var_ofn, sigma_sq)
                    ax_f.plot(xf_line, yf_line, zs=alpha, zdir='y', color='black', linewidth=1.5)

                if len(dg) > 0:
                    xg_line = np.linspace(np.min(dg), np.max(dg), 100)
                    yg_line = normal_g_pdf(xg_line, alpha, p, mu_ofn, var_ofn, sigma_sq)
                    ax_g.plot(xg_line, yg_line, zs=alpha, zdir='y', color='black', linewidth=1.5)

        ax_f.set_xlabel('Value f(alpha)')
        ax_f.set_ylabel('Alpha')
        ax_f.set_zlabel('Density')
        ax_f.set_title('Distribution of f-branch')

        ax_g.set_xlabel('Value g(alpha)')
        ax_g.set_ylabel('Alpha')
        ax_g.set_zlabel('Density')
        ax_g.set_title('Distribution of g-branch')


