#Course Project
#Biomedical Data Science
#Rowan, Benjamin J.
#April 24, 2026

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model as lm
import sklearn as skl
import statsmodels.formula.api as smf
import statsmodels as sm
import streamlit as st

#Load Predictors Data
oil_data = pd.read_csv(r"Oil_Data.csv")
oil_data = oil_data.drop('U.S', axis = 1)


gold_data = pd.read_csv(r"Gold_Data.csv")
gold_data = gold_data.drop(['Price', 'Open', 'High', 'Low', 'Vol.'], axis = 1)
gold_data = gold_data.iloc[::-1].reset_index(drop = True)


unemployment_data = pd.read_csv(r"Unemployment_Data.csv")
unemployment_temp = gold_data.copy()
for i in range(12):
    for e in range(20):
        unemployment_temp.iloc[i+e*12, 1] = unemployment_data.iloc[e, i+1]

unemployment_temp.rename(columns = {'Change %': 'Unemployment_Rate'}, inplace = True)
unemployment_temp["Change %"] = unemployment_temp["Unemployment_Rate"]
for i in range(len(unemployment_temp)-1):
    unemployment_temp.loc[i, "Change %"] = 100*(unemployment_temp["Unemployment_Rate"].iloc[i+1] - unemployment_temp["Unemployment_Rate"].iloc[i]) / unemployment_temp["Unemployment_Rate"].iloc[i]


effr_data = pd.read_csv(r"EFFR_Data.csv")
effr_data = effr_data.drop(['FEDFUNDS'], axis = 1)

#Load SP500 Data
sp_data = pd.read_csv(r"SP_Data.csv")

sp_data = sp_data.drop(['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %'], axis = 1)
sp_data = sp_data.iloc[::-1].reset_index(drop = True)

sp_next_month = sp_data.copy()
sp_next_month["SP_Next_Month"] = sp_next_month["Change"].shift(-1)

#Compile to a single df
final_df = oil_data.copy()
final_df.rename(columns = {'% Change': 'Oil_Change'}, inplace = True)
final_df['Gold_Change'] = gold_data['Change %']
final_df['Unemployment_Change'] = unemployment_temp['Change %']
final_df['EFFR'] = effr_data['% Change']
final_df['SP_Change'] = sp_data['Change']
final_df['SP_Next'] = sp_next_month['SP_Next_Month']

#Heading and description in app
st.markdown("# SP500 Monthly Change Prediction Model")
st.markdown("Use the prediction data below and linear regression to predict the following month's SP500 price change.")
st.markdown("Data ranges from January, 2000 to December, 2019")

#Create training df
train_fraction = float(st.text_input("Enter training fraction (how much of the data is used for training).", 0.75))
train_months = train_fraction * len(final_df)
train_months = int(train_months)
train_df = final_df.iloc[0:train_months, :].copy()

#Create checkboxes to allow the user to select which predictors are active
st.write(f"Select predictors (independent variables to use in regression model)")
use_oil = st.checkbox("Oil", value = True)
use_gold = st.checkbox("Gold", value = True)
use_unemployment = st.checkbox("Unemployment Rate", value = True)
use_current_sp = st.checkbox("Current SP Change", value = True)
use_effr = st.checkbox("Effective Federal Funds Rate", value = True)

predictors = []
if use_oil:
    predictors.append('Oil_Change')
if use_gold:
    predictors.append('Gold_Change')
if use_unemployment:
    predictors.append('Unemployment_Change')
if use_current_sp:
    predictors.append('SP_Change')
if use_effr:
    predictors.append('EFFR')

#Create a graph displaying SP500 as well as predictors
fig, ax = plt.subplots()
train_df.plot(x = 'Date', y = ['SP_Change'] + predictors, kind = 'line', ax = ax  )
ax.set_title("Training Data")
ax.set_ylabel("Monthly Change (%)")
st.pyplot(fig)

#Fit model using predictors and regression
results = smf.ols('SP_Next ~ ' + ' + '.join(predictors), data = train_df).fit()

#create the test df
test_df = final_df.iloc[train_months:, :].copy()

#Manipulate our df by adding predicted cases and error
test_df['predicted_SP_Change'] = results.predict(test_df)
test_df['error'] = (test_df['SP_Next'] - test_df['predicted_SP_Change'])

#add a column which answers whether predicted change is correct in direction
test_df['correct_direction'] = np.where((test_df['SP_Next'] > 0) & (test_df['predicted_SP_Change'] > 0), 1, np.where((test_df['SP_Next'] < 0) & (test_df['predicted_SP_Change'] < 0), 1, 0))

#Find ratio of correct directional predictions
number_of_correct = test_df['correct_direction'].sum()
correct_direction_ratio = number_of_correct/len(test_df)
st.write(f"Proportion of correct directional predictions: {correct_direction_ratio:.2%}")

#Find and display mean error
error_mean = test_df['error'].mean()
st.write(f"Mean Error (%): {error_mean:.2f}")


#Finally, display predicted vs actual results
fig, ax = plt.subplots()
test_df.plot(x = 'Date', y = ['SP_Change', 'predicted_SP_Change'], kind = 'line', ax = ax  )
ax.set_title("Predicted vs Actual SP Monthly Change")
ax.set_ylabel("Monthly Change (%)")
st.pyplot(fig)