# 🚀 Spaceship Titanic — Kaggle Competition

## Описание проекта
Проект по предсказанию транспортировки пассажиров в рамках соревнования Spaceship Titanic на Kaggle.

## Данные
- **train.csv** — обучающая выборка
- **test.csv** — тестовая выборка

## Задача
Предсказать, был ли пассажир транспортирован в другое измерение (колонка `Transported`).

## Результаты
| Модель | Точность |
|--------|----------|
| **LightGBM** | **81.26%** 🏆 |
| SVC | 80.30% |
| LogisticRegression | 79.75% |

## Лучшая модель
- **Алгоритм:** LightGBM
- **Точность:** 81.26%

## Технологии
- Python 3.x
- Pandas, NumPy
- Scikit-learn
- LightGBM
- Matplotlib

## Запуск
```bash
pip install -r requirements.txt
python main.py

Автор:
Abdulla - https://github.com/Arthos761