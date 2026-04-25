# **Course Project**

Rowan, Benjamin J. / Biomedical Data Science / 24 April, 2026

### Summary

Predicting any portion of the stock market is extremely elusive yet highly desirable by professional brokers and amateur day-traders alike. In this project, I used linear regression of multiple market indices to predict SP-500 gains or losses for the following month. 

Independent variables include oil price change, gold price change, unemployment rate change, EFFR (Effective Federal Fund Rate) Change, and current month SP-500 change.

By selecting various combinations of predictors, users can readily see which indices provide meaningful prediction of market performance. The project displays a graph of predicted versus actual monthly changes, mean error, and a ratio of correct guesses at a directional level.

### Goals

1. **Data Manipulation**: Practice data science skills for manipulation including: uploading market marker data as CSVs to panda dataframes, rearranging and cleaning these dataframes for use, and combining these to a single, useable dataframe for regression analysis 

2. **Linear Regression**: Use linear regression to find the correlation coefficients of each predictor as it relates to the dependent variable (SP-500 price change)

3. **User Interactivity via Streamlit**: Enable the user to interact with the regression. The user can change the training fraction to allow for more or less training time. Additionally, the user can change which predictors are used in the regression, enabling visibility of which predictor combination is the most accurate.

### Results
This project was extremely useful in developing python and streamlit skills. I realized my vision of having a market predictor with multiple options for inputs. Further development of this app could include additional predictors (including other stocks and ETFs) and hypothetical ROI given shares sold/bought for each directional prediction. 

*Note:* Unfortunately, while the coding was a success, the prediction model was not. It is extremely difficult to predict the market based on such large indicators. The p-value for all of my predictors except one (gold price) was significantly higher than 0.05, indicating statistical insignificance.


