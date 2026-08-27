import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import Logistic

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 1. Удаляем строковые колонки
X = train.drop(['Transported', 'PassengerId', 'Name', 'Cabin', 'HomePlanet', 'Destination'], axis=1)
y = train['Transported']

# 2. Заполняем булевы колонки
X['CryoSleep'] = X['CryoSleep'].fillna(False).astype(int)
X['VIP'] = X['VIP'].fillna(False).astype(int)

# 3. Заполняем ВСЕ пропуски в числовых колонках
numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

# 4. SelectKBest
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# 5. Названия отобранных признаков
selected_mask = selector.get_support()
selected_features = X.columns[selected_mask]



# 1. Создаём x
x = train[selected_features].copy()

# 2. Заполняем только числовые колонки
numeric_cols_in_x = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in numeric_cols_in_x:
    if col in x.columns:
        x[col] = x[col].fillna(x[col].median())

# 3. Добавляем новые признаки
x['Total_spent'] = train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
x['Has_RoomService'] = (train['RoomService'] > 0).astype(int)
x['Has_FoodCourt'] = (train['FoodCourt'] > 0).astype(int)
x['Has_ShoppingMall'] = (train['ShoppingMall'] > 0).astype(int)
x['Has_Spa'] = (train['Spa'] > 0).astype(int)
x['Has_VRDeck'] = (train['VRDeck'] > 0).astype(int)

# 4. Каюта
x['Deck'] = train['Cabin'].str[0]
x['Side'] = train['Cabin'].str[-1]
x['Cabin_number'] = train['Cabin'].str.extract(r'(\d+)').astype(float)

# 5. Возраст
x['AgeGroup'] = pd.cut(train['Age'], bins=[0, 12, 18, 60, 120], labels=['Child', 'Teen', 'Adult', 'Elderly'])

# 6. Категориальные из train
x['HomePlanet'] = train['HomePlanet']
x['Destination'] = train['Destination']
x['CryoSleep'] = train['CryoSleep'].fillna(False).astype(int)
x['VIP'] = train['VIP'].fillna(False).astype(int)

# 7. Заполняем Total_spent, Cabin_number (если есть NaN)
x['Total_spent'] = x['Total_spent'].fillna(0)
x['Cabin_number'] = x['Cabin_number'].fillna(0)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

numeric_features = [
    'Age',
    'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
    'Total_spent', 'Cabin_number'
]

categorical_features = [
    'HomePlanet', 'CryoSleep', 'Destination', 'VIP',
    'Deck', 'Side', 'AgeGroup',
    'Has_RoomService', 'Has_FoodCourt', 'Has_ShoppingMall', 'Has_Spa', 'Has_VRDeck'
]

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

svc = CalibratedClassifierCV(estimator=SVC(), ensemble=False)


pipeline1 = Pipeline([
    ('preprocessor', preprocessor),
    ('lgb', LGBMClassifier())
])

pipeline2 = Pipeline([
    ('preprocessor', preprocessor),
    ('svc', svc)
])

pipeline3 = Pipeline([
    ('preprocessor', preprocessor),
    ('lin', LogisticRegression())
])

voting = VotingClassifier([
    ('lgb', pipeline1),
    ('svc', pipeline2),
    ('lin', pipeline3)
], voting='soft')

last_pipeline = Pipeline([
    ('voting', voting)
])

pipeline1.fit(x_train, y_train)
pipeline2.fit(x_train, y_train)
pipeline3.fit(x_train, y_train)
last_pipeline.fit(x_train, y_train)

y_pred_lgb = pipeline1.predict(x_test)

y_pred_svc = pipeline2.predict(x_test)

y_pred_lr = pipeline3.predict(x_test)

def get_metrics(y_true, y_pred, name):
    acc = round(accuracy_score(y_true, y_pred), 2) * 100
    prec = round(precision_score(y_true, y_pred, average='binary'), 2) * 100
    rec = round(recall_score(y_true, y_pred, average='binary'), 2) * 100
    f1 = round(f1_score(y_true, y_pred, average='binary'), 2) * 100
    return {'model': name, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

results = [
    get_metrics(y_test, y_pred_lgb, 'LightGBM'),
    get_metrics(y_test, y_pred_svc, 'SVC'),
    get_metrics(y_test, y_pred_lr, 'LogisticRegression')
]

print("\n" + "="*50)
print("ОТЧЁТ ПО МОДЕЛЯМ")
print("="*50)

for r in results:
    print(f"\n{r['model']}:")
    print(f"  Accuracy:  {r['accuracy']:.1f}%")
    print(f"  Precision: {r['precision']:.1f}%")
    print(f"  Recall:    {r['recall']:.1f}%")
    print(f"  F1-score:  {r['f1']:.1f}%")


param_grid_lgb = {
    'lgb__n_estimators': [100, 150, 200],
    'lgb__learning_rate': [0.01, 0.05, 0.1],
    'lgb__num_leaves': [15, 31, 63],
    'lgb__max_depth': [5, 10, 15],
    'lgb__subsample': [0.7, 0.8, 1.0],
    'lgb__reg_alpha': [0.0, 0.1, 0.5],
    'lgb__reg_lambda': [0.0, 0.1, 0.5]
}

param_grid_svc = {
    'svc__estimator__C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    'svc__estimator__gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'svc__estimator__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'svc__estimator__degree': [2, 3, 4],
    'svc__estimator__coef0': [0.0, 0.5, 1.0]
}

param_grid_lr = {
    'lin__C': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    'lin__penalty': ['l1', 'l2', 'elasticnet'],
    'lin__solver': ['liblinear', 'saga', 'lbfgs'],
    'lin__max_iter': [500, 1000, 2000],
    'lin__l1_ratio': [0.1, 0.5, 0.9]    # только для elasticnet
}

random_search_lgb = RandomizedSearchCV(
    pipeline1,
    param_grid_lgb,
    n_iter=30,          # ← только 30 случайных комбинаций
    cv=3,               # ← кросс-валидация
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search_svc = RandomizedSearchCV(
    pipeline2,
    param_grid_svc,
    n_iter=30,          # ← только 30 случайных комбинаций
    cv=3,               # ← кросс-валидация
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search_ln = RandomizedSearchCV(
    pipeline3,
    param_grid_lr,
    n_iter=30,          # ← только 30 случайных комбинаций
    cv=3,               # ← кросс-валидация
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search_lgb.fit(x_train, y_train)
random_search_svc.fit(x_train, y_train)
random_search_ln.fit(x_train, y_train)

print(f"LightGBM: {random_search_lgb.best_score_:.4f} | {random_search_lgb.best_params_}")
print(f"SVC:      {random_search_svc.best_score_:.4f} | {random_search_svc.best_params_}")
print(f"LogReg:   {random_search_ln.best_score_:.4f} | {random_search_ln.best_params_}")


print("\n🔧 ЛУЧШИЕ ПАРАМЕТРЫ (LightGBM):")
print(f"  n_estimators:  {random_search_lgb.best_params_.get('lgb__n_estimators', '—')}")
print(f"  learning_rate: {random_search_lgb.best_params_.get('lgb__learning_rate', '—')}")
print(f"  num_leaves:    {random_search_lgb.best_params_.get('lgb__num_leaves', '—')}")
print(f"  max_depth:     {random_search_lgb.best_params_.get('lgb__max_depth', '—')}")


