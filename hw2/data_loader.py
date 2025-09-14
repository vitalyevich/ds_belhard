import pandas as pd

class DataLoader:

    def load_from_csv(self, file_path):
        """Загрузка данных из CSV с обработкой ошибок.
        Параметры:
            file_path (str): путь к CSV-файлу
        Возвращает:
            pandas.DataFrame
        """
        if not isinstance(file_path, str):
            raise TypeError("file_path должен быть строкой — путём к CSV-файлу")
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV файл не найден: {file_path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV файл пуст: {file_path}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Ошибка при разборе CSV: {e}")
        except Exception as e:
            raise RuntimeError(f"Непредвиденная ошибка при загрузке CSV: {e}")

        if not isinstance(df, pd.DataFrame):
            raise RuntimeError("Загруженный объект не является pandas DataFrame")
        return df
