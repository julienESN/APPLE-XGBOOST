import pandas as pd
from textblob import TextBlob
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import ta
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import yfinance as yf

def add_advanced_features(df, ticker):
    # Ajout du sentiment des nouvelles
    news = yf.Ticker(ticker).news
    sentiment_scores = [TextBlob(n['title']).sentiment.polarity for n in news]
    df['News_Sentiment'] = pd.Series(sentiment_scores).rolling(window=7).mean()
    
    # Ajout de l'indice VIX (indice de volatilité)
    vix = yf.download('^VIX', start=df.index[0], end=df.index[-1])['Close']
    df['VIX'] = vix

    # Ratio prix/bénéfice (P/E)
    ticker_info = yf.Ticker(ticker).info
    df['P/E_Ratio'] = ticker_info.get('forwardPE', None)

    return df


def add_technical_indicators(df):
    # MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    # RSI
    rsi = RSIIndicator(close=df['Close'])
    df['RSI'] = rsi.rsi()

    # Bollinger Bands
    bb = BollingerBands(close=df['Close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()

    return df

# Télécharger les données
ticker = "AAPL"  # Apple Inc. comme exemple
data = yf.download(ticker, start="2020-01-01", end="2024-08-29")
print(data.head())
data = add_technical_indicators(data)
data = add_advanced_features(data, ticker)
print(data.head())

# Préparation des données
data['Returns'] = data['Close'].pct_change()
data['MA5'] = data['Close'].rolling(window=5).mean()
data['MA20'] = data['Close'].rolling(window=20).mean()
data['Target'] = data['Close'].shift(-1)

# Sélection des caractéristiques
features = ['Open', 'High', 'Low', 'Volume', 'Returns', 'MA5', 'MA20', 'MACD', 'RSI', 'BB_High', 'BB_Low']

# Créer un DataFrame avec toutes les colonnes nécessaires
df = data[features + ['Target']]

# Supprimer toutes les lignes contenant des valeurs NaN
df = df.dropna()

# Séparer les features et la cible
X = df[features]
y = df['Target']

# Normalisation des caractéristiques
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Après la normalisation
print("\nDernière ligne des features normalisées :")
print(scaler.inverse_transform(X_scaled[-1].reshape(1, -1)))
# Division des données
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Définition des hyperparamètres à optimiser
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Création du modèle XGBoost
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Recherche par grille avec validation croisée
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Meilleurs paramètres
print("Meilleurs paramètres:", grid_search.best_params_)

# Utilisation du meilleur modèle
best_model = grid_search.best_estimator_
# Avant la prédiction
print("\nDernière ligne des features avant la prédiction :")
print(X.iloc[-1])
# Prédictions et évaluation
predictions = best_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

# Calcul de statistiques supplémentaires
mean_price = y.mean()
std_price = y.std()

print(f"RMSE: {rmse:.2f}")
print(f"Prix moyen de l'action : {mean_price:.2f}")
print(f"Écart-type du prix de l'action : {std_price:.2f}")
print(f"RMSE en pourcentage du prix moyen : {(rmse/mean_price)*100:.2f}%")
print(f"RMSE en pourcentage de l'écart-type : {(rmse/std_price)*100:.2f}%")

# Calcul du R²
r2 = r2_score(y_test, predictions)
print(f"Coefficient de détermination (R²) : {r2:.4f}")

# Calcul du MAE
mae = mean_absolute_error(y_test, predictions)
print(f"Erreur absolue moyenne (MAE) : {mae:.2f}")

# Prédiction pour le jour suivant
last_data = scaler.transform(X.iloc[-1].values.reshape(1, -1))
next_day_prediction = best_model.predict(last_data)
print(f"Prédiction du prix de clôture pour demain : {next_day_prediction[0]:.2f}")
print(f"Dernier prix de clôture connu : {data['Close'].iloc[-1]:.2f}")

# Importance des caractéristiques
feature_importance = best_model.feature_importances_
feature_importance_dict = dict(zip(features, feature_importance))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
print("\nImportance des caractéristiques:")
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")