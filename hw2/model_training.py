import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def train_and_evaluate(df, target_column):
    """
    Обучает и оценивает модель RandomForestClassifier на датасете Titanic.
    
    Параметры:
        df (pd.DataFrame): DataFrame с данными.
        target_column (str): Название целевого столбца (например, 'Survived').
        
    Возвращает:
        model (RandomForestClassifier): Обученная модель.
    """
    # Кодируем категориальные признаки
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Разделение на признаки (X) и цель (y)
    X = df_encoded.drop(columns=[target_column])
    y = df_encoded[target_column]

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Данные разделены: {len(X_train)} для обучения, {len(X_test)} для теста.")

    # Обучение модели
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    print("\nОбучаем модель RandomForestClassifier...")
    model.fit(X_train, y_train)
    print("Модель успешно обучена.")

    # Предсказание и оценка
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nТочность (Accuracy) на тестовых данных: {accuracy:.4f}")

    print("\nОтчет о классификации:")
    print(classification_report(y_test, predictions))

    print("\nМатрица ошибок (Confusion Matrix):")
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Не выжил', 'Выжил'], 
                yticklabels=['Не выжил', 'Выжил'])
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.show()

    return model
