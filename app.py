import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Define functions for data processing and feature engineering
def build_features(train, store):
    store['StoreType'] = store['StoreType'].astype('category').cat.codes
    store['Assortment'] = store['Assortment'].astype('category').cat.codes
    train["StateHoliday"] = train["StateHoliday"].astype('category').cat.codes

    merged = pd.merge(train, store, on='Store', how='left')
    NaN_replace = 0
    merged.fillna(NaN_replace, inplace=True)

    merged['Year'] = merged.Date.dt.year
    merged['Month'] = merged.Date.dt.month
    merged['Day'] = merged.Date.dt.day
    merged['Week'] = merged.Date.dt.isocalendar().week

    merged['MonthsCompetitionOpen'] = \
        12 * (merged['Year'] - merged['CompetitionOpenSinceYear']) + \
        (merged['Month'] - merged['CompetitionOpenSinceMonth'])
    merged.loc[merged['CompetitionOpenSinceYear'] == NaN_replace, 'MonthsCompetitionOpen'] = NaN_replace

    merged['WeeksPromoOpen'] = \
        12 * (merged['Year'] - merged['Promo2SinceYear']) + \
        (merged['Date'].dt.isocalendar().week - merged['Promo2SinceWeek'])
    merged.loc[merged['Promo2SinceYear'] == NaN_replace, 'WeeksPromoOpen'] = NaN_replace

    toInt = [
        'CompetitionOpenSinceMonth',
        'CompetitionOpenSinceYear',
        'Promo2SinceWeek',
        'Promo2SinceYear',
        'MonthsCompetitionOpen',
        'WeeksPromoOpen'
    ]
    merged[toInt] = merged[toInt].astype(int)

    return merged

# Load data
@st.cache_data
def load_data():
    types = {'StateHoliday': np.dtype(str)}
    train = pd.read_csv("train_v2.csv", parse_dates=[2], nrows=66901, dtype=types)
    store = pd.read_csv("store.csv")
    return train, store

train, store = load_data()

# Data Cleaning
train = train[train['Sales'] > 0]

# Feature Engineering
train['SalesPerCustomer'] = train['Sales'] / train['Customers']
avg_store = train.groupby('Store')[['Sales', 'Customers', 'SalesPerCustomer']].mean()
avg_store.rename(columns=lambda x: 'Avg' + x, inplace=True)
store = pd.merge(avg_store.reset_index(), store, on='Store')

med_store = train.groupby('Store')[['Sales', 'Customers', 'SalesPerCustomer']].median()
med_store.rename(columns=lambda x: 'Med' + x, inplace=True)
store = pd.merge(med_store.reset_index(), store, on='Store')

features = build_features(train, store)

# Sidebar for User Input
st.sidebar.header('User Input Features')

selected_store = st.sidebar.selectbox('Store', sorted(train['Store'].unique()))
selected_day_of_week = st.sidebar.selectbox('Day of Week', sorted(train['DayOfWeek'].unique()))
selected_competition_distance = st.sidebar.slider('Competition Distance (meters)', min_value=0, max_value=10000, step=100, value=1000)
selected_promo = st.sidebar.selectbox('Promo', [0, 1])
selected_promo2 = st.sidebar.selectbox('Promo2', [0, 1])
selected_state_holiday = st.sidebar.selectbox('State Holiday', sorted(train['StateHoliday'].unique()))
selected_store_type = st.sidebar.selectbox('Store Type', sorted(store['StoreType'].unique()))
selected_assortment = st.sidebar.selectbox('Assortment', sorted(store['Assortment'].unique()))
selected_date = st.sidebar.date_input('Select Date')

# Filter data based on user input
filtered_data = features[
    (features['Store'] == selected_store) &
    (features['DayOfWeek'] == selected_day_of_week)
]

# Display the filtered data
st.write(f'Data for Store {selected_store} on Day {selected_day_of_week}')
st.write(filtered_data)

# Data Visualization
def plot_scatter(data, x, y, hue=None):
    plt.figure(figsize=(10, 6))
    if hue:
        sb.scatterplot(data=data, x=x, y=y, hue=hue)
    else:
        sb.scatterplot(data=data, x=x, y=y)
    st.pyplot(plt)

def plot_sales_distribution(data):
    plt.figure(figsize=(10, 6))
    sb.histplot(data['Sales'], kde=True)
    plt.title('Sales Distribution')
    plt.xlabel('Sales')
    plt.ylabel('Frequency')
    st.pyplot(plt)

plot_scatter(filtered_data, 'Customers', 'Sales', 'StateHoliday')
plot_sales_distribution(filtered_data)

# Model Training and Evaluation
X = [
    'Store',
    'Customers',
    'CompetitionDistance',
    'Promo',
    'Promo2',
    'StateHoliday',
    'StoreType',
    'Assortment',
    'AvgSales',
    'AvgCustomers',
    'AvgSalesPerCustomer',
    'MedSales',
    'MedCustomers',
    'MedSalesPerCustomer',
    'DayOfWeek',
    'Week',
    'Day',
    'Month',
    'Year',
    'CompetitionOpenSinceMonth',
    'CompetitionOpenSinceYear',
    'Promo2SinceWeek',
    'Promo2SinceYear',
]

X_train, X_test, y_train, y_test = train_test_split(
    features[X], features['Sales'], test_size=0.15, random_state=10)

# Train RandomForest model
randomForest = RandomForestRegressor(n_estimators=25, n_jobs=-1, verbose=1)
randomForest.fit(X_train, y_train)

# Display feature importance
def plot_feature_importance(model, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    st.pyplot(plt)

plot_feature_importance(randomForest, X)

# Predict and display results
y_hat = randomForest.predict(X_test)
st.write('Random Forest Predictions')
st.write(pd.DataFrame({'Actual': y_test, 'Predicted': y_hat}).head(10))

# Model Performance Metrics for RandomForest
rf_rmse = np.sqrt(mean_squared_error(y_test, y_hat))
rf_r2 = r2_score(y_test, y_hat)
st.write('Random Forest Model Performance')
st.write(f'Root Mean Squared Error: {rf_rmse}')
st.write(f'R-squared: {rf_r2}')

# XGBoost model training
xgboost_tree = xgb.XGBRegressor(
    n_jobs=-1,
    n_estimators=1000,
    eta=0.1,
    max_depth=2,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='exact',
    reg_alpha=0.05,
    silent=0,
    random_state=1023
)
xgboost_tree.fit(X_train, np.log1p(y_train))

# Display XGBoost feature importance
def plot_xgb_feature_importance(model, features):
    importance = model.feature_importances_
    indices = np.argsort(importance)
    plt.figure(figsize=(12, 8))
    plt.title('XGBoost Feature Importances')
    plt.barh(range(len(indices)), importance[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    st.pyplot(plt)

plot_xgb_feature_importance(xgboost_tree, X)

# Predict and display results for XGBoost
xgb_predictions = np.expm1(xgboost_tree.predict(X_test))
st.write('XGBoost Predictions')
st.write(pd.DataFrame({'Actual': y_test, 'Predicted': xgb_predictions}).head(10))

# Model Performance Metrics for XGBoost
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
xgb_r2 = r2_score(y_test, xgb_predictions)
st.write('XGBoost Model Performance')
st.write(f'Root Mean Squared Error: {xgb_rmse}')
st.write(f'R-squared: {xgb_r2}')

# Final Prediction Based on User Input
user_input = pd.DataFrame({
    'Store': [selected_store],
    'DayOfWeek': [selected_day_of_week],
    'CompetitionDistance': [selected_competition_distance],
    'Promo': [selected_promo],
    'Promo2': [selected_promo2],
    'StateHoliday': [selected_state_holiday],
    'StoreType': [selected_store_type],
    'Assortment': [selected_assortment],
    'Year': [selected_date.year],
    'Month': [selected_date.month],
    'Day': [selected_date.day],
    'Week': [pd.to_datetime(selected_date).isocalendar().week]
})

# Adding dummy 'Customers' column to user_input
user_input['Customers'] = 0

# Filling missing columns in user_input based on store data
store_data = store[store['Store'] == selected_store]
for col in ['AvgSales', 'AvgCustomers', 'AvgSalesPerCustomer', 'MedSales', 'MedCustomers', 'MedSalesPerCustomer']:
    user_input[col] = store_data[col].values[0]

# Ensure all necessary columns are present in user_input
if 'CompetitionOpenSinceMonth' not in user_input.columns:
    user_input['CompetitionOpenSinceMonth'] = store_data['CompetitionOpenSinceMonth'].values[0]
if 'CompetitionOpenSinceYear' not in user_input.columns:
    user_input['CompetitionOpenSinceYear'] = store_data['CompetitionOpenSinceYear'].values[0]
if 'Promo2SinceWeek' not in user_input.columns:
    user_input['Promo2SinceWeek'] = store_data['Promo2SinceWeek'].values[0]
if 'Promo2SinceYear' not in user_input.columns:
    user_input['Promo2SinceYear'] = store_data['Promo2SinceYear'].values[0]

# Handle NaN values in user_input before type casting
user_input.fillna(0, inplace=True)

user_input['MonthsCompetitionOpen'] = 12 * (user_input['Year'] - user_input['CompetitionOpenSinceYear']) + (user_input['Month'] - user_input['CompetitionOpenSinceMonth'])
user_input['WeeksPromoOpen'] = 12 * (user_input['Year'] - user_input['Promo2SinceYear']) + (user_input['Week'] - user_input['Promo2SinceWeek'])

# Predict sales for the user input using RandomForest
user_pred_rf = randomForest.predict(user_input[X])

# Convert user_input to the format expected by XGBoost model
user_input_xgb = user_input[X].copy()
user_input_xgb['StateHoliday'] = user_input_xgb['StateHoliday'].astype('category').cat.codes

# Predict with XGBoost
user_pred_xgb = np.expm1(xgboost_tree.predict(user_input_xgb))

st.write(f'Predicted Sales (Random Forest): ${user_pred_rf[0]:.2f}')
st.write(f'Predicted Sales (XGBoost): ${user_pred_xgb[0]:.2f}')

# Download filtered data button
st.download_button(
    label="Download Filtered Data",
    data=filtered_data.to_csv(index=False).encode('utf-8'),
    file_name='filtered_data.csv',
    mime='text/csv',
)
