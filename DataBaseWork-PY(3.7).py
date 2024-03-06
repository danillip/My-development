# coding=windows-1251
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

#Установим параматер display.max_columns в None, чтобы видеть все столбцы
#pd.set_option('display.max_columns',None)

# Чтение данных из файла CSV
df = pd.read_csv('Dataset_CyberCrime_Sean.csv')

# Удаляем столбец "Others", так как он может быть менее информативным
df.drop(columns=['Others'], inplace=True)

# Создаем новый столбец, содержащий сумму всех видов преступлений для каждого города
df['Total_Crime'] = df.iloc[:, 1:-1].sum(axis=1)

# Нормализуем числовые данные, кроме столбца 'Total_Crime', чтобы сделать их более сопоставимыми
scaler = StandardScaler()
df.iloc[:, 1:-1] = scaler.fit_transform(df.iloc[:, 1:-1])

# Запрос у пользователя количества строк для вывода
num_rows = int(input("Введите количество строк для вывода: "))

# Вывод первых num_rows строк данных
print(df.head(num_rows))

# Выводим описательные статистики (среднее и стандартное отклонение) по всей выборке
print("Описательные статистики по всей выборке:")
print(df.describe())

# Выводим описательные статистики (среднее и стандартное отклонение) по группам (по городам)
#Среднее значение (Mean) |  Стандартное отклонение (Standard Deviation)
print("\nОписательные статистики по группам (по городам):")
print(df.groupby('City').agg({'Total_Crime': ['mean', 'std']}))

# Выводим количество выбросов и асимметрию и эксцесс по всей выборке
print("\nКоличество выбросов и асимметрия и эксцесс:")
print("Количество выбросов:", df.shape[0] - df[(df > df.quantile(0.25)) & (df < df.quantile(0.75))].count())
print("Асимметрия:", df.skew())
print("Эксцесс:", df.kurtosis())

# Выводим вывод на основе среднего значения и стандартного отклонения
print("\nВывод на основе среднего значения и стандартного отклонения:")
for column in df.columns[1:-1]:  # Проходимся по всем числовым столбцам, кроме 'Total_Crime'
    mean = df[column].mean()
    std_dev = df[column].std()
    print(f"Среднее значение {column}: {mean}, Стандартное отклонение {column}: {std_dev}")
    # Можно добавить дополнительные условия для вывода
    if mean > 0 and std_dev > 0.5:
        print(f"Признак {column} имеет смещенное распределение с большим разбросом.")
    elif mean < 0 and std_dev < 0.5:
        print(f"Признак {column} имеет смещенное распределение с небольшим разбросом.")
    else:
        print(f"Признак {column} имеет нормальное распределение.")


##########################################
##########################################
##########################################
# ВАЖНО!
# Когда закрываешь диаграмму, то открывается новая по другим преступлениям(если не скипанули построение)
##########################################
##########################################
##########################################

# Запрос у пользователя на построение диаграмм
build_plots = input("Хотите построить диаграммы рассеяния и столбчатые диаграммы? (да/нет): ")

# Проверяем ответ пользователя и в зависимости от него либо строим, либо пропускаем построение диаграмм
if build_plots.lower() == 'да':
    for column in df.columns[1:-1]:
        # Создаем фигуру и оси (сетку из 2x1 графиков)
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

        # Диаграмма рассеяния
        axes[0].scatter(range(len(df)), df[column], label=column)
        axes[0].set_title(f'Scatter Plot of {column}')
        axes[0].set_xlabel('Index')
        axes[0].set_ylabel(column)
        axes[0].set_xticks(range(len(df)))
        axes[0].set_xticklabels(df['City'], rotation=90)
        axes[0].legend()
        axes[0].grid(True)

        # Столбчатая диаграмма
        axes[1].bar(range(len(df)), df[column], label=column)
        axes[1].set_title(f'Bar Plot of {column}')
        axes[1].set_xlabel('Index')
        axes[1].set_ylabel(column)
        axes[1].set_xticks(range(len(df)))
        axes[1].set_xticklabels(df['City'], rotation=90)
        axes[1].legend()
        axes[1].grid(True)

        # Регулируем расстояние между графиками
        plt.tight_layout()

        # Показываем графики
        plt.show()
elif build_plots.lower() == 'нет':
    print("Построение диаграмм было пропущено.")
else:
    print("Некорректный ввод. Построение диаграмм будет пропущено.")



# Построение box plot для категориальных данных в простонародье усатые ящики, они реально так выглядят
categorical_columns = ['City']  # Список категориальных столбцов, для которых хотим построить box plot

for column in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=column, y='Total_Crime', data=df)
    plt.title(f'Box Plot of Total Crime by {column}')
    plt.xlabel(column)
    plt.ylabel('Total Crime')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Выводы на основе box plot
    print(f"Выводы на основе box plot для категориального признака {column}:")
    for city in df[column].unique():
        city_data = df[df[column] == city]['Total_Crime']
        q1 = city_data.quantile(0.25)
        q3 = city_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = city_data[(city_data < lower_bound) | (city_data > upper_bound)]

        print(f"Город: {city}")
        print(f"25-й квартиль: {q1}")
        print(f"75-й квартиль: {q3}")
        print(f"Межквартильный размах: {iqr}")
        print(f"Границы выбросов: [{lower_bound}, {upper_bound}]")
        print(f"Количество выбросов: {outliers.shape[0]}")
        print()


# Построение таблицы корреляции для числовых признаков
correlation_matrix = df.corr()

# Вывод таблицы корреляции
print("Таблица корреляции:")
print(correlation_matrix)

# Построение тепловой карты корреляции
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title("Тепловая карта корреляции числовых признаков")
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Выбор типа модели и алгоритма (например, линейная регрессия)
model = LinearRegression()

# Удаление строк с пропущенными значениями
df.dropna(inplace=True)

# Разделение данных на признаки (X) и целевую переменную (y)
X = df.drop(['City', 'Total_Crime'], axis=1)  # Исключаем категориальный признак 'City' и целевую переменную
y = df['Total_Crime']

# Разделение данных на train и test выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели на train выборке
model.fit(X_train, y_train)

# Предсказание на train и test выборках
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Вычисление метрики accuracy на train и test выборках
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print("Accuracy на train выборке:", train_accuracy)
print("Accuracy на test выборке:", test_accuracy)

# Анализ весов (коэффициентов) модели для выявления вклада каждого признака
weights = dict(zip(X.columns, model.coef_))
for feature, weight in weights.items():
    print(f"Признак: {feature}, Вес: {weight}")


# Выводим веса признаков и их влияние
print("Веса признаков и их влияние:")
if hasattr(model, 'coef_'):
    coef_shape = model.coef_.shape
    if len(coef_shape) == 1:
        coef = model.coef_
    else:
        coef = model.coef_[0]
    for feature, weight in zip(X_train.columns, coef):
        if weight > 0:
            print(f"{feature}: положительный вклад ({weight})")
        elif weight < 0:
            print(f"{feature}: отрицательный вклад ({weight})")
        else:
            print(f"{feature}: нейтральный вклад (вес близок к нулю)")
else:
    print("Модель не содержит коэффициентов.")
