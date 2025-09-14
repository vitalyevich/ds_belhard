import matplotlib.pyplot as plt

class Visualization:
    # Утилиты для визуализации с проверкой входных данных

    def __init__(self):
        self.plots = []

    def _ensure_df(self, df):
        if not hasattr(df, 'shape') or not hasattr(df, 'columns'):
            raise TypeError('df должен быть pandas DataFrame')

    def _ensure_columns(self, df, cols):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Не найдено столбцов в DataFrame: {missing}")

    def _sample_df(self, df, sample_size):
        """Возвращает случайную подвыборку, если sample_size указан и меньше длины df."""
        if sample_size and sample_size < len(df):
            return df.sample(n=sample_size, random_state=42)
        return df

    def add_histogram(self, df, column, bins=30, sample_size=None):
        """Построить гистограмму по столбцу. Если нет значимых данных — поднять исключение."""
        self._ensure_df(df)
        self._ensure_columns(df, [column])
        df = self._sample_df(df, sample_size)
        data = df[column].dropna()
        if data.empty:
            raise ValueError(f"Нет ненулевых значений для построения столбца '{column}'")
        plt.figure(figsize=(8,5))
        plt.hist(data, bins=bins)
        plt.title(f'Гистограмма: {column}')
        plt.xlabel(column)
        plt.ylabel('Количество')
        plt.grid(True)
        plt.show()
        self.plots.append(f"histogram_{column}")

    def add_lineplot(self, df, x_col, y_col, sample_size=None):
        """Построить линейный график. Сортирует по x_col, если это возможно."""
        self._ensure_df(df)
        self._ensure_columns(df, [x_col, y_col])
        df = self._sample_df(df, sample_size)
        tmp = df[[x_col, y_col]].dropna()
        if tmp.empty:
            raise ValueError(f"Нет ненулевых пар значений для '{x_col}' и '{y_col}'")
        try:
            tmp = tmp.sort_values(x_col)
        except Exception:
            # если сортировка невозможна — продолжаем без сортировки
            pass
        plt.figure(figsize=(8,5))
        plt.plot(tmp[x_col].values, tmp[y_col].values)
        plt.title(f'Линейный график: {y_col} по {x_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)
        plt.show()
        self.plots.append(f"lineplot_{x_col}_{y_col}")

    def add_scatter(self, df, x_col, y_col, sample_size=None):
        # Построить scatter plot. Проверяет наличие ненулевых пар значений
        self._ensure_df(df)
        self._ensure_columns(df, [x_col, y_col])
        df = self._sample_df(df, sample_size)
        tmp = df[[x_col, y_col]].dropna()
        if tmp.empty:
            raise ValueError(f"Нет ненулевых пар значений для '{x_col}' и '{y_col}'")
        plt.figure(figsize=(8,5))
        plt.scatter(tmp[x_col].values, tmp[y_col].values)
        plt.title(f'Scatter: {x_col} vs {y_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)
        plt.show()
        self.plots.append(f"scatter_{x_col}_{y_col}")

    def remove_plot(self, plot_name):
        if plot_name in self.plots:
            self.plots.remove(plot_name)
            print(f"Визуализация '{plot_name}' удалена.")
        else:
            print('Такой визуализации нет.')
