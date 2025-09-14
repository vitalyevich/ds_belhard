import pandas as pd

class DataProcessing:
    # Инструменты для подсчёта и заполнения пропущенных значений с валидацией входных данных

    def _ensure_df(self, df):
        if not hasattr(df, 'shape') or not hasattr(df, 'columns'):
            raise TypeError('Входные данные должны быть pandas DataFrame')

    def count_missing(self, df):
        # Возвращает Series с количеством пропусков в каждом столбце
        self._ensure_df(df)
        return df.isnull().sum()

    def report_missing(self, df):
        """Возвращает DataFrame-отчёт с количеством и процентом пропусков.
        Если пропусков нет — возвращается пустой DataFrame отчёта.
        """
        self._ensure_df(df)
        if df.shape[0] == 0:
            return pd.DataFrame(columns=['missing_count', 'missing_percent'])
        missing = df.isnull().sum()
        total = len(df)
        report = pd.DataFrame({'missing_count': missing, 'missing_percent': (missing / total) * 100})
        report = report[report['missing_count'] > 0].sort_values('missing_percent', ascending=False)
        return report

    def fill_missing(self, df, method="mean", columns=None, constant_value=None):
        """Заполнение пропусков с проверками.

        method: 'mean', 'median', 'mode', 'constant'
        columns: список столбцов для заполнения (по умолчанию все столбцы)
        constant_value: значение для метода 'constant' (по умолчанию 0)
        """
        self._ensure_df(df)
        if columns is None:
            columns = df.columns.tolist()
        else:
            if not isinstance(columns, (list, tuple)):
                raise TypeError('columns должен быть списком или кортежем имён столбцов')

        allowed = {'mean', 'median', 'mode', 'constant'}
        if method not in allowed:
            raise ValueError(f"Неизвестный метод '{method}'. Выберите один из {allowed}")

        df = df.copy()
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Столбец не найден в DataFrame: {col}")
            if df[col].isnull().sum() == 0:
                continue
            dtype = df[col].dtype
            if method == 'mean' and pd.api.types.is_numeric_dtype(dtype):
                fill_val = df[col].mean()
            elif method == 'median' and pd.api.types.is_numeric_dtype(dtype):
                fill_val = df[col].median()
            elif method == 'mode':
                mode_vals = df[col].mode()
                if len(mode_vals) == 0:
                    fill_val = 0 if pd.api.types.is_numeric_dtype(dtype) else ''
                else:
                    fill_val = mode_vals[0]
            elif method == 'constant':
                fill_val = 0 if constant_value is None else constant_value
            else:
                # запасной вариант — мода
                mode_vals = df[col].mode()
                fill_val = mode_vals[0] if len(mode_vals) > 0 else (0 if pd.api.types.is_numeric_dtype(dtype) else '')

            df[col].fillna(fill_val, inplace=True)
        return df
