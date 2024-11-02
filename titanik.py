# Импорт библиотек
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Загрузка данных с Kaggle
data = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Предварительная обработка данных
data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])
data['Embarked'] = label_encoder.fit_transform(data['Embarked'])

# Разделение данных на признаки и целевую переменную
X = data.drop('Survived', axis=1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создание и обучение модели случайного леса
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели на тестовой выборке: {accuracy:.4f}")
print("Отчет по классификации:")
print(classification_report(y_test, y_pred))

# Построение матрицы ошибок
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Матрица ошибок')
plt.xlabel('Предсказанные метки')
plt.ylabel('Истинные метки')
plt.show()

# Визуализация 1: Распределение числовых признаков (Age, Fare)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(data['Age'], bins=30, kde=True)
plt.title('Распределение возраста пассажиров')
plt.xlabel('Возраст')
plt.ylabel('Количество пассажиров')

plt.subplot(1, 2, 2)
sns.histplot(data['Fare'], bins=30, kde=True)
plt.title('Распределение стоимости билетов')
plt.xlabel('Стоимость билета')
plt.ylabel('Количество пассажиров')
plt.tight_layout()
plt.show()

# Визуализация 2: Выживаемость по полу
sns.barplot(x='Sex', y='Survived', data=data)
plt.title('Выживаемость по полу')
plt.xlabel('Пол (0 = female, 1 = male)')
plt.ylabel('Доля выживших')
plt.show()

# Визуализация 3: Выживаемость по классу билета
sns.barplot(x='Pclass', y='Survived', data=data)
plt.title('Выживаемость по классу билета')
plt.xlabel('Класс билета')
plt.ylabel('Доля выживших')
plt.show()

# Визуализация 4: Важность признаков по модели
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title('Важность признаков в модели RandomForest')
plt.xlabel('Важность')
plt.ylabel('Признак')
plt.show()
