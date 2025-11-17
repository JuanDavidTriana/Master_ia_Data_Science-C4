# Pipelines en Machine Learning

## ¿Qué es un Pipeline?

Un **pipeline** (o tubería) en Machine Learning es una secuencia de pasos de procesamiento de datos y modelado que se ejecutan de forma automática y ordenada. Es como una cadena de montaje donde los datos pasan por diferentes etapas de transformación hasta llegar al modelo final.

### Ventajas de usar Pipelines:

1. **Organización**: Todo el flujo de trabajo está en un solo lugar
2. **Reproducibilidad**: Los mismos pasos se aplican siempre en el mismo orden
3. **Prevención de fugas de datos**: Evita que información del conjunto de prueba contamine el entrenamiento
4. **Facilidad de mantenimiento**: Cambios centralizados y fáciles de gestionar
5. **Despliegue simplificado**: Todo el proceso se puede guardar y reutilizar

---

## Estructura de un Pipeline

```
Datos Crudos → Preprocesamiento → Transformación → Modelo → Predicción
```

### Componentes típicos:

1. **Transformadores (Transformers)**: Modifican los datos
   - Imputación de valores nulos
   - Escalado de características
   - Codificación de variables categóricas
   - Selección de características

2. **Estimadores (Estimators)**: Aprenden de los datos
   - Modelos de Machine Learning
   - Algoritmos de clasificación/regresión

---

## Ejemplo 1: Pipeline Básico con Scikit-Learn

### Dataset: Titanic

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Cargar datos
df = pd.read_csv('Titanic-Dataset.csv')

# Seleccionar características numéricas
X = df[['Age', 'Fare', 'SibSp', 'Parch']]
y = df['Survived']

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Crear pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Paso 1: Imputar valores nulos
    ('scaler', StandardScaler()),                    # Paso 2: Escalar datos
    ('classifier', LogisticRegression())             # Paso 3: Modelo
])

# Entrenar el pipeline completo
pipeline.fit(X_train, y_train)

# Predecir
score = pipeline.score(X_test, y_test)
print(f'Accuracy: {score:.2f}')
```

**Resultado**: El pipeline ejecuta automáticamente:
1. Imputa valores nulos con la mediana
2. Escala las características
3. Entrena el modelo de regresión logística

---

## Ejemplo 2: Pipeline con ColumnTransformer

### Manejo de variables numéricas y categóricas simultáneamente

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Definir características numéricas y categóricas
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
categorical_features = ['Sex', 'Embarked', 'Pclass']

# Preparar datos
X = df[numeric_features + categorical_features]
y = df['Survived']

# Crear transformadores para cada tipo de variable
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar transformadores con ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Pipeline completo
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenar
full_pipeline.fit(X_train, y_train)

# Evaluar
score = full_pipeline.score(X_test, y_test)
print(f'Accuracy con Random Forest: {score:.2f}')
```

**¿Qué hace este pipeline?**

```
Variables Numéricas:
  Age, Fare, SibSp, Parch
        ↓
  Imputar con mediana
        ↓
  Escalar (StandardScaler)
        ↓
        └──────────────┐
                       ↓
Variables Categóricas: Combinar → Random Forest → Predicción
  Sex, Embarked, Pclass
        ↓
  Imputar con moda
        ↓
  One-Hot Encoding
        ↓
        ┘
```

---

## Ejemplo 3: Pipeline con Feature Engineering

### Creación de nuevas características

```python
from sklearn.base import BaseEstimator, TransformerMixin

# Crear transformador personalizado
class FeatureCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Crear nuevas características
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
        X['FarePerPerson'] = X['Fare'] / X['FamilySize']
        return X

# Pipeline con feature engineering
pipeline_with_features = Pipeline([
    ('feature_creator', FeatureCreator()),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Entrenar
X = df[['Age', 'Fare', 'SibSp', 'Parch']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline_with_features.fit(X_train, y_train)
score = pipeline_with_features.score(X_test, y_test)
print(f'Accuracy con Feature Engineering: {score:.2f}')
```

---

## Ejemplo 4: Pipeline con Validación Cruzada

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

# Definir pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Preparar datos
X = df[['Age', 'Fare', 'SibSp', 'Parch']].copy()
y = df['Survived']

# Validación cruzada
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f'Accuracy promedio (CV): {scores.mean():.2f} (+/- {scores.std():.2f})')

# Búsqueda de hiperparámetros
param_grid = {
    'imputer__strategy': ['mean', 'median'],
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__solver': ['liblinear', 'lbfgs']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

print(f'Mejores parámetros: {grid_search.best_params_}')
print(f'Mejor score: {grid_search.best_score_:.2f}')
```

**Ventaja**: El pipeline asegura que cada fold de validación cruzada aplique las mismas transformaciones.

---

## Ejemplo 5: Pipeline para Detección de Outliers

```python
from sklearn.preprocessing import RobustScaler
from sklearn.covariance import EllipticEnvelope

# Pipeline para detección de anomalías
outlier_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler()),  # Robusto a outliers
    ('detector', EllipticEnvelope(contamination=0.1))
])

# Entrenar
X = df[['Age', 'Fare']].copy()
outlier_pipeline.fit(X)

# Detectar outliers
predictions = outlier_pipeline.predict(X)
outliers = predictions == -1

print(f'Número de outliers detectados: {outliers.sum()}')
print(f'Porcentaje de outliers: {(outliers.sum() / len(X)) * 100:.2f}%')
```

---

## Ejemplo 6: Pipeline Completo para Producción

```python
import joblib
from datetime import datetime

# Pipeline completo y robusto
production_pipeline = Pipeline([
    ('feature_creator', FeatureCreator()),
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ]), ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'FarePerPerson']),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), ['Sex', 'Embarked'])
        ])),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    ))
])

# Entrenar
X = df[['Age', 'Fare', 'SibSp', 'Parch', 'Sex', 'Embarked']]
y = df['Survived']

production_pipeline.fit(X, y)

# Guardar el pipeline
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'titanic_pipeline_{timestamp}.pkl'
joblib.dump(production_pipeline, filename)
print(f'Pipeline guardado como: {filename}')

# Cargar y usar el pipeline
loaded_pipeline = joblib.load(filename)
new_data = pd.DataFrame({
    'Age': [25],
    'Fare': [30.0],
    'SibSp': [0],
    'Parch': [0],
    'Sex': ['male'],
    'Embarked': ['S']
})

prediction = loaded_pipeline.predict(new_data)
probability = loaded_pipeline.predict_proba(new_data)

print(f'Predicción: {"Sobrevive" if prediction[0] == 1 else "No sobrevive"}')
print(f'Probabilidad: {probability[0][1]:.2%}')
```

---

## Buenas Prácticas al Usar Pipelines

### ✅ Hacer:

1. **Incluir toda la transformación de datos** en el pipeline
2. **Usar nombres descriptivos** para cada paso
3. **Guardar el pipeline completo** para producción
4. **Validar con datos nuevos** después de cargar el pipeline
5. **Documentar cada paso** del pipeline

### ❌ Evitar:

1. **Transformar datos antes del pipeline** (puede causar fuga de datos)
2. **Usar fit_transform en datos de prueba** (solo transform)
3. **Mezclar diferentes versiones** de pipelines en producción
4. **Olvidar manejar valores desconocidos** en variables categóricas

---

## Comparación: Con Pipeline vs Sin Pipeline

### ❌ Sin Pipeline (Propenso a errores):

```python
# Entrenar
X_train_imputed = imputer.fit_transform(X_train)
X_train_scaled = scaler.fit_transform(X_train_imputed)
model.fit(X_train_scaled, y_train)

# Predecir (¿usamos fit_transform o transform?)
X_test_imputed = imputer.transform(X_test)  # ⚠️ Fácil olvidar usar transform
X_test_scaled = scaler.transform(X_test_imputed)
predictions = model.predict(X_test_scaled)
```

### ✅ Con Pipeline (Seguro y limpio):

```python
# Entrenar
pipeline.fit(X_train, y_train)

# Predecir (automáticamente usa transform)
predictions = pipeline.predict(X_test)
```

---

## Ejemplo Avanzado: Pipeline con Selección de Características

```python
from sklearn.feature_selection import SelectKBest, f_classif

# Pipeline con selección automática de características
pipeline_with_selection = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_classif, k=3)),  # Seleccionar las 3 mejores
    ('classifier', LogisticRegression())
])

X = df[['Age', 'Fare', 'SibSp', 'Parch']].copy()
y = df['Survived']

pipeline_with_selection.fit(X, y)

# Ver qué características se seleccionaron
selected_features = pipeline_with_selection.named_steps['selector'].get_support()
feature_names = X.columns
selected_names = feature_names[selected_features]
print(f'Características seleccionadas: {list(selected_names)}')
```

---

## Pipeline para Diferentes Tipos de Problemas

### Clasificación Binaria (Sobrevivió o No):

```python
classification_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])
```

### Regresión (Predecir Edad):

```python
from sklearn.ensemble import RandomForestRegressor

regression_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])
```

### Clustering (Agrupar pasajeros):

```python
from sklearn.cluster import KMeans

clustering_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clusterer', KMeans(n_clusters=3))
])
```

---

## Resumen

Un **Pipeline** es una herramienta fundamental en Machine Learning que:

- 🔄 **Automatiza** el flujo de trabajo
- 🛡️ **Previene errores** comunes
- 📦 **Facilita el despliegue** en producción
- 🔬 **Mejora la reproducibilidad** de experimentos
- 🎯 **Simplifica la validación cruzada** y búsqueda de hiperparámetros

### Estructura básica:

```python
Pipeline([
    ('paso1', Transformador1()),
    ('paso2', Transformador2()),
    ('paso3', Estimador())
])
```

**Recuerda**: Todo lo que se aplica a los datos de entrenamiento debe estar dentro del pipeline para garantizar que se aplique correctamente a los datos de prueba y producción.

---

## Recursos Adicionales

- [Documentación oficial de Scikit-Learn Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [Ejemplos de ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [Crear transformadores personalizados](https://scikit-learn.org/stable/developers/develop.html)
