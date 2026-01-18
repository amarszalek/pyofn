import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model, metrics

from pyofn.ofnumber import OFNumber


class Sampling(layers.Layer):
    """
    Warstwa próbkowania dla VAE (Reparameterization Trick).
    z = z_mean + exp(0.5 * z_log_var) * epsilon
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class MonotonicOFNLayer(layers.Layer):
    """
    Warstwa generująca geometrię OFN:
    1. f(t) monotonicznie rosnąca.
    2. spread(t) = g(t) - f(t) >= 0 (brak przecięć).
    3. spread(t) malejący od brzegów do jądra (styl świecy).
    """

    def __init__(self, ofn_dim=11, **kwargs):
        super(MonotonicOFNLayer, self).__init__(**kwargs)
        self.ofn_dim = ofn_dim

    def build(self, input_shape):
        # Parametry dla krzywej f (dolnej)
        self.dense_f_start = layers.Dense(1)
        # Softplus zapewnia dodatnie przyrosty -> monotoniczność f
        self.dense_f_deltas = layers.Dense(self.ofn_dim - 1, activation='softplus')

        # Parametry dla spreadu (odległość g - f)
        # Softplus zapewnia spread >= 0
        self.dense_spread_kernel = layers.Dense(1, activation='softplus')  # Minimalna szerokość w jądrze
        self.dense_spread_deltas = layers.Dense(self.ofn_dim - 1, activation='softplus')

    def call(self, inputs):
        # Konstrukcja f (rosnąca)
        f_start = self.dense_f_start(inputs)
        f_deltas = self.dense_f_deltas(inputs)

        # Cumsum na deltach + start
        f_steps = tf.cumsum(f_deltas, axis=1)
        f_steps = tf.concat([tf.zeros_like(f_start), f_steps], axis=1)
        f_curve = f_steps + f_start

        # Konstrukcja spreadu (malejącego ku środkowi)
        # Budujemy go "wstecz" od jądra (prawa strona) do nośnika (lewa strona)
        spread_kernel = self.dense_spread_kernel(inputs)
        spread_deltas = self.dense_spread_deltas(inputs)

        # Odwracamy delty, sumujemy, odwracamy wynik -> spread rośnie w lewo (czyli maleje w prawo)
        spread_rev = tf.cumsum(tf.reverse(spread_deltas, axis=[1]), axis=1)
        spread_steps = tf.reverse(spread_rev, axis=[1])
        spread_steps = tf.concat([spread_steps, tf.zeros_like(spread_kernel)], axis=1)
        spread_curve = spread_steps + spread_kernel

        # 3. Konstrukcja g
        g_curve = f_curve + spread_curve

        return f_curve, g_curve


class SoftFlipLayer(layers.Layer):
    """
    Miesza krzywe f i g na podstawie prawdopodobieństwa orientacji p.
    Umożliwia nienadzorowane uczenie kierunku trendu.
    """

    def call(self, inputs):
        f_curve, g_curve, p = inputs
        # p: (batch, 1) -> broadcasting na (batch, dim)

        # Jeśli p=1 (Long): Start=f, End=g
        # Jeśli p=0 (Short): Start=g, End=f
        directed_start = p * f_curve + (1.0 - p) * g_curve
        directed_end = p * g_curve + (1.0 - p) * f_curve

        return directed_start, directed_end


class VariationalNeuralOFC(Model):
    def __init__(self, window_size, n_features=2, ofn_dim=11, latent_dim=16):
        super(VariationalNeuralOFC, self).__init__()
        self.window_size = window_size
        self.n_features = n_features
        self.ofn_dim = ofn_dim
        self.latent_dim = latent_dim

        # --- ENCODER ---
        self.encoder_net = tf.keras.Sequential([
            layers.InputLayer(input_shape=(window_size, n_features)),
            layers.Conv1D(16, 5, activation='relu', padding='same'),
            layers.AveragePooling1D(2),
            layers.Conv1D(32, 3, activation='relu', padding='same'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(32, activation='relu')
        ], name="Encoder_Backbone")

        self.dense_mean = layers.Dense(latent_dim, name="z_mean")
        self.dense_log_var = layers.Dense(latent_dim, name="z_log_var")
        self.sampling = Sampling()

        # --- GENERATOR OFN (GEOMETRIA) ---
        self.ofn_layer = MonotonicOFNLayer(ofn_dim=ofn_dim)

        # --- GENERATOR ORIENTACJI (p) ---
        # Sigmoid zwraca p w [0, 1]. Nie uczymy tego na etykietach, lecz przez błąd rekonstrukcji Trendu!
        self.p_layer = layers.Dense(1, activation='sigmoid', name="P_Orientation")
        self.flipper = SoftFlipLayer()

        # --- DECODER 1: ROZKŁAD (SHAPE) ---
        # Rekonstruuje posortowane ceny (dystrybuantę). Zależy tylko od geometrii f i g.
        self.dist_decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(ofn_dim * 2,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(window_size)
        ], name="Dist_Decoder")

        # --- DECODER 2: TREND (DIRECTION) ---
        # Rekonstruuje [Open, Close]. Zależy od skierowanych krzywych.
        self.trend_decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(ofn_dim * 2,)),
            layers.Dense(16, activation='relu'),
            layers.Dense(2)  # [Start_Price, End_Price]
        ], name="Trend_Decoder")

        # Trackery metryk
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.rec_loss_tracker = metrics.Mean(name="rec_loss")
        self.trend_loss_tracker = metrics.Mean(name="trend_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.rec_loss_tracker,
                self.trend_loss_tracker, self.kl_loss_tracker]

    def call(self, inputs):
        # 1. Kodowanie VAE
        x = self.encoder_net(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling([z_mean, z_log_var])

        # 2. Generowanie OFN
        f_curve, g_curve = self.ofn_layer(z)
        p = self.p_layer(z)

        # 3. Dekodowanie Rozkładu (Shape) -> Posortowane Ceny
        # Wejście: surowa geometria [f, g]
        geom_vector = tf.concat([f_curve, g_curve], axis=1)
        rec_dist = self.dist_decoder(geom_vector)

        # 4. Dekodowanie Trendu (Direction) -> [Open, Close]
        # Wejście: skierowane krzywe po SoftFlip
        dir_start, dir_end = self.flipper([f_curve, g_curve, p])
        trend_vector = tf.concat([dir_start, dir_end], axis=1)
        rec_trend = self.trend_decoder(trend_vector)

        return rec_dist, rec_trend, z_mean, z_log_var, p, f_curve, g_curve

    def train_step(self, data):
        # Obsługa formatu danych (X, [Y_dist, Y_trend])
        if isinstance(data, tuple):
            x = data[0]
            # y_true[0] = Sorted Prices, y_true[1] = [Open, Close]
            y_dist = data[1][0]
            y_trend = data[1][1]
        else:
            raise ValueError('must be tuple')

        with tf.GradientTape() as tape:
            # Forward pass
            rec_dist, rec_trend, z_mean, z_log_var, _, _, _ = self(x)

            # 1. Reconstruction Loss (MSE Shape)
            loss_dist = tf.reduce_mean(tf.square(y_dist - rec_dist))

            # 2. Trend Loss (MSE Open/Close)
            loss_trend = tf.reduce_mean(tf.square(y_trend - rec_trend))

            # 3. KL Divergence (VAE Regularization)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # Total Loss (Wagi można stroić: beta-VAE)
            total_loss = loss_dist + 2.0 * loss_trend + 0.01 * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(loss_dist)
        self.trend_loss_tracker.update_state(loss_trend)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "rec_loss": self.rec_loss_tracker.result(),
            "trend_loss": self.trend_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def get_candle(self, x):
        """
        Zwraca obiekt OFNumber (OFCandle) i metadane dla pojedynczego okna.
        """
        if x.ndim == 1: x = x.reshape(1, -1, 1)
        if x.ndim == 2: x = np.expand_dims(x, axis=-1)

        _, _, _, _, p_val, f_vals, g_vals = self(x)

        f = f_vals.numpy()[0]
        g = g_vals.numpy()[0]
        p = p_val.numpy()[0, 0]

        # Interpretacja
        # p > 0.5 -> Model uznał, że Start=f, End=g (Long)
        # p < 0.5 -> Model uznał, że Start=g, End=f (Short)

        if p > 0.5:
            # Long: f (dół) -> g (góra)
            # W pyofn Long: Branch_f rosnąca, Branch_g malejąca (ale g>f)
            # Nasze g jest rosnące numerycznie, pyofn interpretuje to jako kształt.
            return OFNumber(f, g), "Long", p
        else:
            # Short: g (góra) -> f (dół)
            return OFNumber(g, f), "Short", p


# --- FUNKCJA POMOCNICZA DO TRENINGU ---

def train_vae_ofc(raw_prices, raw_volumes, window_size=60, epochs=20, ofn_dim=11):
    # 1. Przygotowanie danych
    X, Y_dist, Y_trend = [], [], []

    # Normalizacja globalna lub lokalna (tutaj lokalna - per okno)
    for i in range(len(raw_prices) - window_size):
        # 1. Pobieramy okno
        w_price = raw_prices[i: i + window_size]
        w_vol = raw_volumes[i: i + window_size]

        # 2. Normalizacja Ceny (Z-Score w oknie) - kluczowe dla kształtu
        p_mean = np.mean(w_price)
        p_std = np.std(w_price) + 1e-6
        w_price_norm = (w_price - p_mean) / p_std

        # 3. Normalizacja Wolumenu (dla Enkodera)
        # Wolumen często ma rozkład potęgowy, więc logarytmowanie pomaga
        # Dodatkowo skalujemy do zakresu 0-1 w oknie dla stabilności
        v_log = np.log1p(w_vol)
        v_min, v_max = np.min(v_log), np.max(v_log)
        if v_max - v_min == 0:
            w_vol_norm = np.zeros_like(v_log)
        else:
            w_vol_norm = (v_log - v_min) / (v_max - v_min)

        # 4. Input X: Stack Price i Volume
        # Shape: (window_size, 2)
        x_sample = np.stack([w_price_norm, w_vol_norm], axis=1)
        X.append(x_sample)

        # 5. Target Y_dist: WAŻONE KWANTYLE
        # Używamy znormalizowanej ceny i SUROWEGO (lub zlogarytmowanego) wolumenu jako wag
        # Ważne: wagi muszą być nieujemne. Używamy w_vol (surowego) dla precyzji wagi.
        weighted_q = get_weighted_quantiles(w_price_norm, w_vol, n_points=window_size)
        Y_dist.append(weighted_q)

        # 6. Target Y_trend: Open/Close (znormalizowane)
        Y_trend.append([w_price_norm[0], w_price_norm[-1]])

    X = np.array(X)
    Y_dist = np.array(Y_dist)
    Y_trend = np.array(Y_trend)

    print(f"Dane gotowe. X shape: {X.shape}")

    # Inicjalizacja modelu
    model = VariationalNeuralOFC(window_size, n_features=2, ofn_dim=ofn_dim)
    model.compile(optimizer='adam')

    # Trening
    model.fit(X, [Y_dist, Y_trend], epochs=epochs, batch_size=32, verbose=1)

    return model


def get_weighted_quantiles(prices, volumes, n_points):
    """
    Oblicza ważone kwantyle dla danego okna.
    """
    # Sortujemy ceny (i odpowiadające im wolumeny) rosnąco po cenie
    sort_idx = np.argsort(prices)
    sorted_prices = prices[sort_idx]
    sorted_vol = volumes[sort_idx]

    # Obliczamy skumulowany wolumen (Dystrybuanta empiryczna)
    cum_vol = np.cumsum(sorted_vol)

    total_vol = cum_vol[-1]
    if total_vol == 0:
        cum_vol = np.linspace(0, 1, len(prices))
    else:
        cum_vol = cum_vol / total_vol  # Normalizacja do [0, 1]

    # Interpolujemy do stałej liczby punktów (n_points)
    target_x = np.linspace(0, 1, n_points)

    # interp(x_new, x_points, y_points)
    weighted_dist = np.interp(target_x, cum_vol, sorted_prices)

    return weighted_dist


