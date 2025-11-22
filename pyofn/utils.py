import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from copy import deepcopy
from sortedcontainers import SortedDict
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict
from itertools import chain


def worker_min_func_gpw(key, fdata, freq, nlevels, target, batch_size):
    df = pd.read_hdf(fdata, key)
    ob = MinOrderBook()
    ob_prev = None
    df['time'] = (pd.to_datetime(df['time'], unit='ns', utc=True)).dt.tz_convert('Europe/Warsaw')
    df['priority_date'] = (pd.to_datetime(df['priority_date'], unit='ns', utc=True)).dt.tz_convert('Europe/Warsaw')
    df['order_date'] = (pd.to_datetime(df['order_date'], unit='ns', utc=True)).dt.tz_convert('Europe/Warsaw')
    df['price'] = df['price'] / (10**df['price_level'])
    #df = df.sort_values(by='time')

    target_fn = target + (fdata.split(r'/')[-1]).split('.')[0] + '_' + key[1:] + f'_{freq}_{nlevels}.parquet'
    writer = None
    schema = None

    current_batch_cols = defaultdict(list)
    current_rows_count = 0  # Licznik wierszy do batch_size

    n = 0
    if 'all' in freq:
        dre = (df.set_index('time')).groupby('time')
    else:
        dre = (df.set_index('time')).resample(freq)
    for t, g in dre:
        r = _func_min_obooks_gpw(g, ob, nlevels, ob_prev, current_batch_cols)
        if r != 'Empty':
            n+=1
            ob_prev = r
            current_rows_count = len(current_batch_cols['order_id'])
            if (batch_size>0) and (current_rows_count >= batch_size):
                table_batch = pa.Table.from_pydict(current_batch_cols)
                if writer is None:
                    schema = table_batch.schema
                    writer = pq.ParquetWriter(target_fn, schema=schema, compression='snappy')

                writer.write_table(table_batch)

                current_batch_cols = defaultdict(list)
                current_rows_count = 0

    if current_batch_cols:
        table_batch = pa.Table.from_pydict(current_batch_cols)
        if writer is None:
            schema = table_batch.schema
            writer = pq.ParquetWriter(target_fn, schema=schema, compression='snappy')

        writer.write_table(table_batch)
        current_batch_cols = defaultdict(list)
        current_rows_count = 0

    if writer:
        writer.close()

    return n


def _func_min_obooks_gpw(df, ob2, nlevels, obs_prev, current_batch_cols):
    if len(df) < 1:
        return 'Empty'
    for i, row in df.iterrows():
        stamp = i
        ob2.current_time = deepcopy(stamp)
        ob2.nmsg += 1

        order_dict = row.to_dict()

        if row.action_type == 'F':
            ob2.clear_orderbook()
        elif row.action_type == 'Y':
            ob2.retransmit_order(order_dict)
        elif row.action_type == 'A':
            ob2.add_order(order_dict)
        elif row.action_type == 'M':
            ob2.mod_order(order_dict)
        elif row.action_type == 'D':
            ob2.del_order(order_dict)
        else:
            raise ValueError(f"unsupported action_type: {row.action_type}")

    ts9 = deepcopy(stamp)
    ts9 = ts9.replace(hour=9, minute=0, second=0, microsecond=0)
    ts1650 = deepcopy(stamp)
    ts1650 = ts1650.replace(hour=16, minute=50, second=0, microsecond=0)
    try:
        if ob2.current_time < ts9 or ob2.current_time >= ts1650:
            return 'Empty'
    except Exception as e:
        print(e, stamp, ob2.current_time)
        raise e

    if ob2.isempty():
        return 'Empty'

    if ob2.has_zero_price():
        return 'Empty'

    if ob2.is_crossed():
        return 'Empty'

    ob3 = ob2.reduce_to_nlevels(nlevels=nlevels)

    if (obs_prev is None) or (not ob3.isthesame(obs_prev)):
        batch_cols = ob3.flat_order_map()
        for col_name, col_values in batch_cols.items():
            current_batch_cols[col_name].extend(col_values)
        return ob3
    else:
        return 'Empty'


def worker_func_gpw(key, fdata, freq, nlevels, target, batch_size):
    df = pd.read_hdf(fdata, key)
    ob = OrderBook()
    ob_prev = None
    df['time'] = (pd.to_datetime(df['time'], unit='ns', utc=True)).dt.tz_convert('Europe/Warsaw')
    df['priority_date'] = (pd.to_datetime(df['priority_date'], unit='ns', utc=True)).dt.tz_convert('Europe/Warsaw')
    df['order_date'] = (pd.to_datetime(df['order_date'], unit='ns', utc=True)).dt.tz_convert('Europe/Warsaw')
    df['price'] = df['price'] / (10**df['price_level'])
    #df = df.sort_values(by='time')

    target_fn = target + (fdata.split(r'/')[-1]).split('.')[0] + '_' + key[1:] + f'_{freq}_{nlevels}.parquet'
    writer = None
    schema = None
    all_orders_data = []
    n = 0
    if 'all' in freq:
        dre = (df.set_index('time')).groupby('time')
    else:
        dre = (df.set_index('time')).resample(freq)
    for t, g in dre:
        r = _func_obooks_gpw(g, ob, nlevels, ob_prev, all_orders_data)
        if r != 'Empty':
            n+=1
            if (batch_size>0) and (len(all_orders_data) >= batch_size):
                # Konwertuj partię na DataFrame i potem na Tabelę PyArrow
                df_batch = pd.DataFrame(all_orders_data)
                table_batch = pa.Table.from_pandas(df_batch)

                if writer is None:
                    # Pierwszy zapis: stwórz schemat i otwórz writer
                    # Schemat jest pobierany z pierwszej partii i musi być taki sam dla reszty
                    schema = table_batch.schema
                    writer = pq.ParquetWriter(target_fn, schema=schema, compression='snappy')

                writer.write_table(table_batch)
                all_orders_data.clear()

    if all_orders_data:
        df_final = pd.DataFrame(all_orders_data)
        table_final = pa.Table.from_pandas(df_final)
        if writer is None:
            schema = table_final.schema
            writer = pq.ParquetWriter(target_fn, schema=schema, compression='snappy')

        writer.write_table(table_final)
        all_orders_data.clear()

    if writer:
        writer.close()

    return n


def _func_obooks_gpw(df, ob2, nlevels, obs_prev, all_orders_data):
    if len(df) < 1:
        return 'Empty'
    for i, row in df.iterrows():
        stamp = i
        ob2.current_time = deepcopy(stamp)
        ob2.nmsg += 1

        if row.action_type == 'F':
            ob2.clear_orderbook()
        elif row.action_type == 'Y':
            ob2.retransmit_order(row)
        elif row.action_type == 'A':
            ob2.add_order(row)
        elif row.action_type == 'M':
            ob2.mod_order(row)
        elif row.action_type == 'D':
            ob2.del_order(row)
        else:
            raise ValueError('unsupported action_type')

    ts9 = deepcopy(stamp)
    ts9 = ts9.replace(hour=9, minute=0, second=0, microsecond=0)
    ts1650 = deepcopy(stamp)
    ts1650 = ts1650.replace(hour=16, minute=50, second=0, microsecond=0)
    try:
        if ob2.current_time < ts9 or ob2.current_time >= ts1650:
            return 'Empty'
    except Exception as e:
        print(e, stamp, ob2.current_time)
        raise e

    if 0.0 in ob2.buy_book or 0.0 in ob2.sell_book:
        return 'Empty'

    ob3 = ob2.reduce_to_nlevels(nlevels=nlevels, mode='gpw')

    if (obs_prev is None) or (not ob3.isthesame_l3(obs_prev)):
        obs_prev = deepcopy(ob3)

        batch = ob3.flat_order_map()
        all_orders_data.extend(batch)
    else:
        return 'Empty'


class MinOrderBook(object):
    def __init__(self, data=None):
        super(MinOrderBook, self).__init__()

        self.buy_map = {}
        self.sell_map = {}
        self.current_time = None
        self.nmsg = 0

        if data is not None:
            try:
                self.restore_order_map(data)
            except KeyError as e:
                print(f"Błąd ładowania danych: Brak oczekiwanego klucza w słowniku 'data': {e}")
            except Exception as e:
                print(f"Wystąpił nieoczekiwany błąd podczas ładowania danych: {e}")

    def flat_order_map_old(self):
        batch_data = []
        snapshot_time = self.current_time
        snapshot_nmsg = self.nmsg
        for order in self.buy_map.values():
            order_with_time = order.copy()
            order_with_time['snapshot_timestamp'] = snapshot_time
            order_with_time['snapshot_nmsg'] = snapshot_nmsg
            batch_data.append(order_with_time)
        for order in self.sell_map.values():
            order_with_time = order.copy()
            order_with_time['snapshot_timestamp'] = snapshot_time
            order_with_time['snapshot_nmsg'] = snapshot_nmsg
            batch_data.append(order_with_time)
        return batch_data

    def flat_order_map(self):
        cols = {
            'order_date': [], 'order_id': [], 'price': [], 'volume': [],
            'side': [], 'order_type': [], 'priority_date': [],
            'snapshot_timestamp': [], 'snapshot_nmsg': []
        }
        snapshot_time = self.current_time
        snapshot_nmsg = self.nmsg

        for order in chain(self.buy_map.values(), self.sell_map.values()):
            cols['order_date'].append(order.get('order_date'))
            cols['order_id'].append(order.get('order_id'))
            cols['price'].append(order.get('price'))
            cols['volume'].append(order.get('volume'))
            cols['side'].append(order.get('side'))
            cols['order_type'].append(order.get('order_type'))
            cols['priority_date'].append(order.get('priority_date'))
            cols['snapshot_timestamp'].append(snapshot_time)
            cols['snapshot_nmsg'].append(snapshot_nmsg)
        return cols

    @staticmethod
    def restore_order_map_old(batch_data):
        ob_restored = MinOrderBook()
        ob_restored.current_time = batch_data[0]['snapshot_timestamp']
        ob_restored.nmsg = batch_data[0]['snapshot_nmsg']
        for order_dict in batch_data:
            del order_dict['snapshot_timestamp']
            del order_dict['snapshot_nmsg']
            ob_restored.add_order(order_dict)
        return ob_restored

    @staticmethod
    def restore_order_map(batch_data):
        """
        Odtwarza instancję klasy z danych w formacie kolumnowym (słownik list).
        Jest to operacja odwrotna do flat_order_map_columnar.
        """
        ob_restored = MinOrderBook()

        # Sprawdzenie czy są jakiekolwiek dane (czy lista 'order_id' istnieje i nie jest pusta)
        if not batch_data or 'order_id' not in batch_data or not batch_data['order_id']:
            return ob_restored

        # Odtworzenie metadanych globalnych
        ob_restored.current_time = batch_data['snapshot_timestamp'][0]
        ob_restored.nmsg = batch_data['snapshot_nmsg'][0]

        # Przygotowanie kluczy do rekonstrukcji zleceń
        meta_keys = {'snapshot_timestamp', 'snapshot_nmsg'}
        order_keys = [k for k in batch_data.keys() if k not in meta_keys]

        # Transpozycja (Kolumny -> Wiersze)
        # Pobieramy listy wartości dla wybranych kluczy
        data_arrays = [batch_data[k] for k in order_keys]

        # Funkcja zip(*data_arrays) "zszywa" listy równolegle, tworząc krotki wierszy.
        for row_values in zip(*data_arrays):
            order_dict = dict(zip(order_keys, row_values))
            ob_restored.add_order(order_dict)

        return ob_restored

    def copy(self):
        return deepcopy(self)

    def isempty(self):
        if len(self.buy_map) == 0 and len(self.sell_map) == 0:
            return True
        return False

    def has_zero_price(self):
        """
        Sprawdza, czy w order_map znajduje się jakiekolwiek zlecenie z ceną 0.0.
        Zwraca True, jeśli znajdzie takie zlecenie, w przeciwnym razie False.
        """

        buy = any(order.get('price') == 0.0 for order in self.buy_map.values())
        sell = any(order.get('price') == 0.0 for order in self.sell_map.values())
        zero = buy or sell

        return zero

    def isthesame(self, other):
        """
        Sprawdza, czy stan L3 (każde pojedyncze zlecenie)
        tej księgi jest identyczny ze stanem księgi ob2.
        """
        return (self.buy_map == other.buy_map) and (self.sell_map == other.sell_map)

    def is_crossed(self):
        """
        Sprawdza, czy arkusz jest skrzyżowany (Best Bid >= Best Ask).
        Zwraca True, jeśli dane są niepoprawne (ujemny lub zerowy spread).
        Zwraca False, jeśli spread jest dodatni lub brakuje jednej ze stron.
        """
        if not self.buy_map or not self.sell_map:
            return False  # Nie można skrzyżować, jeśli brakuje jednej strony

        # Znajdujemy Best Bid (maksymalna cena kupna)
        best_bid = max(o['price'] for o in self.buy_map.values())

        # Znajdujemy Best Ask (minimalna cena sprzedaży)
        best_ask = min(o['price'] for o in self.sell_map.values())

        return best_bid >= best_ask

    def clear_orderbook(self, ctime=None, nmsg=None):  # F
        """
        Czyści bieżącą instancję OrderBook (in-place),
        usuwając wszystkie zlecenia i resetując metadane (lub nie).
        """
        # Wyczyść struktury L3 i L2
        self.buy_map.clear()
        self.sell_map.clear()
        # Zresetuj metadane
        if ctime is not None:
            self.current_time = ctime
        if nmsg is not None:
            self.nmsg = nmsg

    def _get_key(self, order):
        """Helper do tworzenia unikalnego klucza zlecenia."""
        return order['order_date'], order['order_id']

    def add_order(self, order):  # A
        key = self._get_key(order)
        side = order['side']

        if side == 1:
            if key not in self.buy_map:
                self.buy_map[key] = order
        elif side in [2, 5]:
            if key not in self.sell_map:
                self.sell_map[key] = order

    def del_order(self, order):  # D
        key = self._get_key(order)
        side = order['side']

        if side == 1:
            self.buy_map.pop(key, None)
        elif side in [2, 5]:
            self.sell_map.pop(key, None)


    def mod_order(self, order):  # M
        key = self._get_key(order)
        side = order.get('side')

        if order['side'] == 1:
            existing = self.buy_map.get(key)
        elif order['side'] in [2, 5]:
            existing = self.sell_map.get(key)
        else:
            existing = None

        if existing is None:
            existing = self.buy_map.get(key)
            if existing is None:
                existing = self.sell_map.get(key)

        if existing is None:
            self.add_order(order)  # Jeśli nie ma, to dodaj
            return

        # Update in-place
        for k, v in order.items():
            if v != -1:
                existing[k] = v

    def retransmit_order(self, order):  # Y
        try:
            self.mod_order(order)
        except ValueError:
            self.add_order(order)

    def reduce_to_nlevels(self, nlevels=10):
        """
        Zwraca nową instancję MinOrderBook zredukowaną do 'n' najlepszych poziomów
        lub do procentowego odchylenia od best bid/ask.
        """
        ob = deepcopy(self)  # Kopia całej struktury
        if ob.isempty():
            return ob

        # Wyciągamy ceny
        unique_buys = sorted({o['price'] for o in ob.buy_map.values()}, reverse=True)
        unique_sells = sorted({o['price'] for o in ob.sell_map.values()})

        price_up = float('inf')
        price_down = float('-inf')

        # Wyznaczanie progów (logika identyczna, ale kod czystszy)
        if isinstance(nlevels, int):
            if unique_sells:
                idx = min(nlevels, len(unique_sells)) - 1
                price_up = unique_sells[idx]
            if unique_buys:
                idx = min(nlevels, len(unique_buys)) - 1
                price_down = unique_buys[idx]
        else:
            # Logika procentowa (skrócona dla przykładu)
            if unique_sells:
                best_ask = unique_sells[0]
                price_up = best_ask * (1 + nlevels)
                # Specyficzne reguły dla niskich cen
                if best_ask < 0.2:
                    price_up = best_ask + 0.02
                elif best_ask < 0.3:
                    price_up = best_ask + 0.03
            if unique_buys:
                best_bid = unique_buys[0]
                price_down = best_bid * (1 - nlevels)
                # Specyficzne reguły dla niskich cen
                if best_bid < 0.2:
                    price_down = best_bid - 0.02
                elif best_bid < 0.3:
                    price_down = best_bid - 0.03

        # Czyścimy Sells
        keys_to_del = [k for k, o in ob.sell_map.items() if o['price'] > price_up]
        for k in keys_to_del:
            del ob.sell_map[k]
        # Czyścimy Buys
        keys_to_del = [k for k, o in ob.buy_map.items() if o['price'] < price_down]
        for k in keys_to_del:
            del ob.buy_map[k]

        return ob


class OrderBook(object):
    def __init__(self, data=None, verbose=False):
        super(OrderBook, self).__init__()
        self.verbose = verbose

        # self.order_map: Szybki dostęp po zleceń po kluczu
        # Klucz: (order_date, order_id)
        # Wartość: Słownik zlecenia (order_dict)
        self.order_map = {}

        # self.buy_book / self.sell_book: Utrzymują posortowane poziomy cen
        # Klucz: price (float)
        # Wartość: Słownik zleceń na tym poziomie
        self.buy_book = SortedDict()  # Ceny sortowane rosnąco (Best Bid jest na końcu)
        self.sell_book = SortedDict()  # Ceny sortowane rosnąco (Best Ask jest na początku)

        self.current_time = None
        self.nmsg = 0

        if data is not None:
            try:
                self.from_dict(data)
            except KeyError as e:
                print(f"Błąd ładowania danych: Brak oczekiwanego klucza w słowniku 'data': {e}")
            except Exception as e:
                print(f"Wystąpił nieoczekiwany błąd podczas ładowania danych: {e}")

    def copy(self):
        return deepcopy(self)

    def isempty(self):
        if len(self.buy_book) == 0 and len(self.sell_book) == 0:
            return True
        return False

    def _get_key(self, order):
        """Helper do tworzenia unikalnego klucza zlecenia."""
        return order['order_date'], order['order_id']

    def _get_l2_view(self, book):
        """
        Pomocnicza funkcja do generowania zagregowanego widoku L2 (cena: łączny wolumen)
        dla danej księgi (buy_book lub sell_book).
        """
        return {
            price: sum(order['volume'] for order in price_level_orders.values())
            for price, price_level_orders in book.items()
            if price_level_orders
        }

    def isthesame_l2(self, ob2):
        """
        Sprawdza, czy zagregowany stan L2 (cena i łączny wolumen)
        tej księgi jest identyczny ze stanem księgi ob2.
        """
        l_buy_view = self._get_l2_view(self.buy_book)
        r_buy_view = ob2._get_l2_view(ob2.buy_book)
        if l_buy_view != r_buy_view:
            return False
        l_sell_view = self._get_l2_view(self.sell_book)
        r_sell_view = ob2._get_l2_view(ob2.sell_book)
        if l_sell_view != r_sell_view:
            return False
        return True

    def isthesame_l3(self, ob2):
        """
        Sprawdza, czy stan L3 (każde pojedyncze zlecenie)
        tej księgi jest identyczny ze stanem księgi ob2.
        """
        return self.order_map == ob2.order_map

    def flat_order_map(self):
        batch_data = []
        snapshot_time = self.current_time
        snapshot_nmsg = self.nmsg
        for order in self.order_map.values():
            order_with_time = order.copy()
            order_with_time['snapshot_timestamp'] = snapshot_time
            order_with_time['snapshot_nmsg'] = snapshot_nmsg
            batch_data.append(order_with_time)
        return batch_data

    @staticmethod
    def restore_order_map(batch_data):
        ob_restored = OrderBook(verbose=False)
        ob_restored.current_time = batch_data[0]['snapshot_timestamp']
        ob_restored.nmsg = batch_data[0]['snapshot_nmsg']
        for order_dict in batch_data:
            del order_dict['snapshot_timestamp']
            del order_dict['snapshot_nmsg']
            ob_restored.add_order(order_dict)
        return ob_restored

    def to_dict(self):
        """
        Zwraca pełny stan księgi (L3) w oryginalnym formacie słownika list.
        Wymaga iteracji po wszystkich zleceniach w self.order_map.
        """

        buy_orders_dict = {'order_date': [], 'priority_date': [], 'order_id': [],
                           'price': [], 'volume': [], 'order_type': []}
        sell_orders_dict = {'order_date': [], 'priority_date': [], 'order_id': [],
                            'price': [], 'volume': [], 'order_type': []}

        for order in self.order_map.values():
            if order['side'] == 1:
                target_dict = buy_orders_dict
            elif order['side'] in [2, 5]:
                target_dict = sell_orders_dict
            else:
                raise ValueError('unsupported side')

            target_dict['order_date'].append(order.get('order_date'))
            target_dict['priority_date'].append(order.get('priority_date'))
            target_dict['order_id'].append(order.get('order_id'))
            target_dict['price'].append(order.get('price'))
            target_dict['volume'].append(order.get('volume'))
            target_dict['order_type'].append(order.get('order_type'))

        d = {
            'current_time': deepcopy(self.current_time),
            'nmsg': deepcopy(self.nmsg),
            'buy_orders': deepcopy(buy_orders_dict),
            'sell_orders': deepcopy(sell_orders_dict)
        }
        return d

    def from_dict(self, data):
        """
        Metoda do parsowania formatu danych (słownik list)
        i budowania struktur (order_map, buy_book, sell_book).
        """

        self.current_time = deepcopy(data.get('current_time'))
        self.nmsg = deepcopy(data.get('nmsg', 0))

        buy_data = data.get('buy_orders', {})
        if buy_data:
            try:
                buy_tuples = zip(
                    buy_data['order_date'],
                    buy_data['priority_date'],
                    buy_data['order_id'],
                    buy_data['price'],
                    buy_data['volume'],
                    buy_data['order_type']
                )
                for od, pd, oid, p, v, ot in buy_tuples:
                    order = {
                        'order_date': od,
                        'priority_date': pd,
                        'order_id': oid,
                        'price': p,
                        'volume': v,
                        'order_type': ot,
                        'side': 1
                    }
                    self.add_order(order)
            except KeyError as e:
                print(f"Brak klucza podczas przetwarzania 'buy_orders': {e}")
            except TypeError as e:
                print(f"Błąd typu (prawdopodobnie puste dane) w 'buy_orders': {e}")

        sell_data = data.get('sell_orders', {})
        if sell_data:
            try:
                sell_tuples = zip(
                    sell_data['order_date'],
                    sell_data['priority_date'],
                    sell_data['order_id'],
                    sell_data['price'],
                    sell_data['volume'],
                    sell_data['order_type']
                )
                for od, pd, oid, p, v, ot in sell_tuples:
                    order = {
                        'order_date': od,
                        'priority_date': pd,
                        'order_id': oid,
                        'price': p,
                        'volume': v,
                        'order_type': ot,
                        'side': 2
                    }
                    self.add_order(order)
            except KeyError as e:
                print(f"Brak klucza podczas przetwarzania 'sell_orders': {e}")
            except TypeError as e:
                print(f"Błąd typu (prawdopodobnie puste dane) w 'sell_orders': {e}")

    def get_best_buys(self, num=1):
        prices = []
        volumes = []

        if num is None:
            num = len(self.buy_book)

        # Iterujemy od końca (najwyższe ceny)
        for i in range(1, min(num, len(self.buy_book)) + 1):
            price, price_level_orders = self.buy_book.peekitem(-i)  # -i -> i-ty od końca
            total_volume = sum(o['volume'] for o in price_level_orders.values())
            prices.append(price)
            volumes.append(total_volume)
        return {'best_buy_prices': np.array(prices), 'best_buy_volumes': np.array(volumes)}

    def get_best_sells(self, num=1):
        prices = []
        volumes = []

        if num is None:
            num = len(self.sell_book)

        # Iterujemy od początku (najniższe ceny)
        for i in range(min(num, len(self.sell_book))):
            price, price_level_orders = self.sell_book.peekitem(i)  # i -> i-ty od początku
            total_volume = sum(o['volume'] for o in price_level_orders.values())
            prices.append(price)
            volumes.append(total_volume)
        return {'best_sell_prices': np.array(prices), 'best_sell_volumes': np.array(volumes)}

    def get_bests(self, num=1):
        if self.isempty():
            return {'best_buy_prices': np.array([]), 'best_buy_volumes': np.array([]),
                    'best_sell_prices': np.array([]), 'best_sell_volumes': np.array([])}

        bb = self.get_best_buys(num=num)
        bs = self.get_best_sells(num=num)
        bb.update(bs)
        return bb

    def to_min_dict(self):
        """
        Zwraca zagregowany stan L2 (wszystkie poziomy cen i ich łączny wolumen).
        """
        d = {'current_time': deepcopy(self.current_time), 'nmsg': deepcopy(self.nmsg)}
        bests = self.get_bests(num=None)
        d['buy_prices'] = deepcopy(bests['best_buy_prices'])
        d['sell_prices'] = deepcopy(bests['best_sell_prices'])
        d['buy_volumes'] = deepcopy(bests['best_buy_volumes'])
        d['sell_volumes'] = deepcopy(bests['best_sell_volumes'])
        return d

    def reduce_to_nlevels(self, nlevels=10, mode=None):
        """
        Zwraca nową instancję OrderBook zredukowaną do 'n' najlepszych poziomów
        lub do procentowego odchylenia od best bid/ask.
        """
        ob = self.copy()
        if ob.isempty():
            return ob

        price_up = float('inf')
        price_down = float('-inf')
        if isinstance(nlevels, int):
            bests = ob.get_bests(num=nlevels)
            if bests['best_sell_prices'].size > 0:
                price_up = bests['best_sell_prices'][-1]
            if bests['best_buy_prices'].size > 0:
                price_down = bests['best_buy_prices'][-1]
        else:
            bests = ob.get_bests(num=1)
            if bests['best_sell_prices'].size > 0:
                best_ask = bests['best_sell_prices'][0]
                price_up = (1 + nlevels) * best_ask
                if bests['best_sell_prices'][0] < 0.2:
                    price_up = bests['best_sell_prices'][0] + 0.02
                elif bests['best_sell_prices'][0] < 0.3:
                    price_up = bests['best_sell_prices'][0] + 0.03
            if bests['best_buy_prices'].size > 0:
                best_bid = bests['best_buy_prices'][0]
                price_down = (1 - nlevels) * best_bid
                if bests['best_buy_prices'][0] < 0.2:
                    price_down = bests['best_buy_prices'][0] - 0.02
                elif bests['best_buy_prices'][0] < 0.3:
                    price_down = bests['best_buy_prices'][0] - 0.03

        sell_prices_to_remove = [p for p in ob.sell_book.keys() if p > price_up]
        for price in sell_prices_to_remove:
            orders_at_level = ob.sell_book.get(price, {})
            for key in orders_at_level.keys():
                ob.order_map.pop(key, None)
            del ob.sell_book[price]

        buy_prices_to_remove = [p for p in ob.buy_book.keys() if p < price_down]
        for price in buy_prices_to_remove:
            orders_at_level = ob.buy_book.get(price, {})
            for key in orders_at_level.keys():
                ob.order_map.pop(key, None)
            del ob.buy_book[price]

        return ob

    def clear_orderbook(self, ctime=None, nmsg=None):  # F
        """
        Czyści bieżącą instancję OrderBook (in-place),
        usuwając wszystkie zlecenia i resetując metadane (lub nie).
        """
        # Wyczyść struktury L3 i L2
        self.order_map.clear()
        self.buy_book.clear()
        self.sell_book.clear()

        # Zresetuj metadane
        if ctime is not None:
            self.current_time = ctime
        if nmsg is not None:
            self.nmsg = nmsg

    def add_order(self, order):  # A
        key = self._get_key(order)

        if key in self.order_map:
            if self.verbose:
                print(f"Próba dodania istniejącego zlecenia: {key}")
            return

        self.order_map[key] = order
        side = order['side']
        price = order['price']
        book = self.buy_book if side == 1 else self.sell_book

        if price not in book:
            book[price] = {}
        book[price][key] = order

    def del_order(self, order):  # D
        key = self._get_key(order)

        order_to_remove = self.order_map.pop(key, None)
        if order_to_remove is None:
            if self.verbose:
                print(f"Próba usunięcia nieistniejącego zlecenia: {key}")
            return

        side = order_to_remove['side']
        price = order_to_remove['price']
        book = self.buy_book if side == 1 else self.sell_book
        price_level = book.get(price)
        if price_level:
            if key in price_level:
                del price_level[key]
            if not price_level:
                del book[price]

    def mod_order(self, order):  # M
        key = self._get_key(order)

        existing_order = self.order_map.get(key)
        if existing_order is None:
            self.add_order(order)
            return

        # Zlecenie istnieje. Modyfikacja może być:
        # A) Zmiana ceny -> Wymaga usunięcia i dodania
        # B) Zmiana wolumenu/daty -> Modyfikacja w miejscu
        new_price = order.get('price', -1)
        price_changed = (new_price != -1 and new_price != existing_order['price'])

        if price_changed:
            temp_order = deepcopy(existing_order)
            temp_order.update({k: v for k, v in order.items() if v != -1})
            self.del_order(existing_order)  # Usuń stare
            self.add_order(temp_order)  # Dodaj nowe
        else:
            for k, v in order.items():
                if v != -1:
                    existing_order[k] = v

    def retransmit_order(self, order):  # Y
        try:
            self.mod_order(order)
        except ValueError:
            self.add_order(order)

    def get_mid(self):
        bests = self.get_bests()
        if not bests['best_sell_prices'].size or not bests['best_buy_prices'].size:
            return -1.0
        bpask = bests['best_sell_prices'][0]
        bpbid = bests['best_buy_prices'][0]
        mid = 0.5 * (bpask + bpbid)
        return mid

    def get_spread(self):
        bests = self.get_bests()
        if not bests['best_sell_prices'].size or not bests['best_buy_prices'].size:
            return -1.0
        bpask = bests['best_sell_prices'][0]
        bpbid = bests['best_buy_prices'][0]
        spread = bpask - bpbid
        return spread

    def get_wap(self, level=1, mode='separate'):
        bests = self.get_bests(num=level)
        if mode == 'separate':
            assert (len(bests['best_buy_prices']) == len(bests['best_sell_prices']) == level)
            return (bests['best_buy_prices'][level - 1] * bests['best_sell_volumes'][level - 1] +
                    bests['best_sell_prices'][level - 1] * bests['best_buy_volumes'][level - 1]) / \
                (bests['best_buy_volumes'][level - 1] + bests['best_sell_volumes'][level - 1])
        elif mode == 'combine':
            return ((bests['best_buy_prices'] * bests['best_sell_volumes'] +
                     bests['best_sell_prices'] * bests['best_buy_volumes']).sum()) / \
                (bests['best_buy_volumes'] + bests['best_sell_volumes']).sum()
        else:
            raise ValueError('unsupported mode')

    def get_buys_df(self, cum=False):
        """
        Zwraca DataFrame zleceń kupna.
        cum=True: Zwraca zagregowany widok L2 pogrupowany
                  według ('price', 'order_type').
        cum=False: Zwraca pełny widok L3 (listę zleceń).
        """
        if cum:
            agg_data = {}
            for price, price_level_orders in reversed(self.buy_book.items()):
                agg_by_type = {}
                for order in price_level_orders.values():
                    # Użyj 'UNKNOWN' jeśli typ zlecenia jest None lub go brakuje
                    order_type = order.get('order_type') or 'UNKNOWN'
                    if order_type not in agg_by_type:
                        agg_by_type[order_type] = {'sum': 0, 'size': 0}
                    agg_by_type[order_type]['sum'] += order['volume']
                    agg_by_type[order_type]['size'] += 1

                for order_type, stats in agg_by_type.items():
                    agg_data[(price, order_type)] = stats

            if not agg_data:
                idx = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['price', 'order_type'])
                cols = pd.MultiIndex.from_product([['volume'], ['sum', 'size']])
                return pd.DataFrame(columns=cols, index=idx)

            df = pd.DataFrame.from_dict(agg_data, orient='index')
            df.columns = pd.MultiIndex.from_product([['volume'], ['sum', 'size']])
            df.index.names = ['price', 'order_type']
            return df
        else:
            orders = [order for order in self.order_map.values() if order['side'] == 1]
            if not orders:
                return pd.DataFrame(
                    columns=['order_date', 'priority_date', 'order_id', 'price', 'volume', 'order_type'])
            return pd.DataFrame(orders).sort_values(['price', 'priority_date'], ascending=[False, True])

    def get_sells_df(self, cum=False):
        """
        Zwraca DataFrame zleceń sprzedaży.
        cum=True: Zwraca zagregowany widok L2 pogrupowany
                  według ('price', 'order_type').
        cum=False: Zwraca pełny widok L3 (listę zleceń).
        """
        if cum:
            agg_data = {}
            for price, price_level_orders in self.sell_book.items():
                agg_by_type = {}
                for order in price_level_orders.values():
                    order_type = order.get('order_type') or 'UNKNOWN'
                    if order_type not in agg_by_type:
                        agg_by_type[order_type] = {'sum': 0, 'size': 0}
                    agg_by_type[order_type]['sum'] += order['volume']
                    agg_by_type[order_type]['size'] += 1
                for order_type, stats in agg_by_type.items():
                    agg_data[(price, order_type)] = stats

            if not agg_data:
                idx = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['price', 'order_type'])
                cols = pd.MultiIndex.from_product([['volume'], ['sum', 'size']])
                return pd.DataFrame(columns=cols, index=idx)

            df = pd.DataFrame.from_dict(agg_data, orient='index')
            df.columns = pd.MultiIndex.from_product([['volume'], ['sum', 'size']])
            df.index.names = ['price', 'order_type']
            return df
        else:
            orders = [order for order in self.order_map.values() if order['side'] in [2, 5]]
            if not orders:
                return pd.DataFrame(
                    columns=['order_date', 'priority_date', 'order_id', 'price', 'volume', 'order_type'])
            return pd.DataFrame(orders).sort_values(['price', 'priority_date'])

    def display(self, cum=True, out=False):
        dbuys = self.get_buys_df(cum=cum)
        dsells = self.get_sells_df(cum=cum)
        if cum:
            dbuys = dbuys.reset_index().droplevel(1, axis=1)
            dbuys.columns = ['price', 'order_type', 'volume', 'size']
            dsells = dsells.reset_index().droplevel(1, axis=1)
            dsells.columns = ['price', 'order_type', 'volume', 'size']
        dbuys = dbuys[['price', 'volume']]
        dsells = dsells[['price', 'volume']]
        tab = pd.merge(dbuys, dsells, how='outer', on='price', suffixes=('_bid', '_ask')).reindex(
            columns=['volume_bid', 'price', 'volume_ask']).sort_values(by='price', ascending=False)
        width = max(10, len(str(tab.max().max())) + 3)
        stab = tab.to_string(index=False, na_rep='', justify='center', formatters=(
            lambda x: '{:.0f}'.format(x).rjust(width), lambda x: '{:.2f}'.format(x).center(width),
            lambda x: '{:.0f}'.format(x).ljust(width)))
        stab = 'Current time is {}\n'.format(self.current_time) + stab
        if out:
            return stab
        else:
            print(stab)

    def summary(self, out=False):
        dbuys = self.get_buys_df(cum=False)
        dsells = self.get_sells_df(cum=False)
        lask = len(dsells['price'].unique())
        lbid = len(dbuys['price'].unique())
        oask = len(dsells)
        obid = len(dbuys)
        vask = dsells['volume'].sum()
        vbid = dbuys['volume'].sum()

        bests = self.get_bests()
        bpask = bests['best_sell_prices'][0]
        bpbid = bests['best_buy_prices'][0]
        bvask = bests['best_sell_volumes'][0]
        bvbid = bests['best_buy_volumes'][0]

        mid = 0.5 * (bpask + bpbid)
        spread = bpask - bpbid
        wap = self.get_wap()

        s = """Current time is {}

Ask price levels:   {}
Bid price levels:   {}
Total price levels: {}

Ask orders:         {}
Bid orders:         {}
Total orders:       {}

Ask volume:         {}
Bid volume:         {}
Total volume:       {} 

Spread:             {:.2f}
Mid point:          {:.2f}
WAP:                {:.2f}

Best Ask:           {:.2f}
Volume:             {}

Best Bid:           {:.2f}
Volume:             {}
        """.format(self.current_time, lask, lbid, lask + lbid, oask, obid, oask + obid, vask,
                   vbid, vask + vbid, spread, mid, wap, bpask, bvask, bpbid, bvbid)
        if out:
            return s
        else:
            print(s)

    def plot(self, cum=True, bounds=None, ref=None, ax=None, figsize=(10, 10), height=0.1):
        dbuys = self.get_buys_df(cum=cum)
        dsells = self.get_sells_df(cum=cum)
        if cum:
            dbuys = dbuys.reset_index().droplevel(1, axis=1)
            dbuys.columns = ['price', 'order_type', 'volume', 'size']
            dsells = dsells.reset_index().droplevel(1, axis=1)
            dsells.columns = ['price', 'order_type', 'volume', 'size']
        dbuys = dbuys[['price', 'volume']]
        dsells = dsells[['price', 'volume']]
        if ref is None:
            ref = self.get_mid()
        if bounds is not None:
            dbuys = dbuys[(ref * (1 - bounds) <= dbuys['price']) & (dbuys['price'] <= ref * (1 + bounds))]
            dsells = dsells[(ref * (1 - bounds) <= dsells['price']) & (dsells['price'] <= ref * (1 + bounds))]
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.barh(dsells['price'].values, dsells['volume'].values, align='center', height=height)
        ax.barh(dbuys['price'].values, -dbuys['volume'].values, align='center', height=height)
        ax.set_title('Order Book')
        ax.set_title('BID', loc='left')
        ax.set_title('ASK', loc='right')
        a = ax.get_xticks().tolist()
        ax.axvline(x=0, color='k', linewidth=1)
        ax.xaxis.set_major_locator(mticker.FixedLocator(a))
        for i in range(len(a)):
            a[i] = int(abs(a[i]))
        ax.set_xticklabels(a)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
        if ax is None:
            plt.show()

    def plot_sd(self, bounds=None, ref=None, ax=None, figsize=(5, 10)):
        dbuys = self.get_buys_df(cum=True)
        dsells = self.get_sells_df(cum=True)
        dbuys = dbuys.reset_index().droplevel(1, axis=1)
        dbuys.columns = ['price', 'order_type', 'volume', 'size']
        dsells = dsells.reset_index().droplevel(1, axis=1)
        dsells.columns = ['price', 'order_type', 'volume', 'size']
        dbuys = dbuys[['price', 'volume']]
        dsells = dsells[['price', 'volume']]
        dbuys['cum_volume'] = dbuys['volume'].cumsum()
        dsells['cum_volume'] = dsells['volume'].cumsum()
        if ref is None:
            ref = self.get_mid()
        if bounds is not None:
            dbuys = dbuys[(ref * (1 - bounds) <= dbuys['price']) & (dbuys['price'] <= ref * (1 + bounds))]
            dsells = dsells[(ref * (1 - bounds) <= dsells['price']) & (dsells['price'] <= ref * (1 + bounds))]
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.step(dsells['cum_volume'], dsells['price'], where='pre', label='Supply')
        ax.step(dbuys['cum_volume'], dbuys['price'], where='pre', label='Demand')
        ax.set_title('Supply and Demand')
        ax.legend()
        for label in ax.get_xticklabels():
            label.set_rotation(45)
        if ax is None:
            plt.show()
