# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:05:54 2021

@author: daniel.le
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')


def data_loader(path):
    df = pd.read_csv(path)

    '''Data Prep'''

    df['arrival_date_month'] = pd.to_datetime(df.arrival_date_month, format='%B').dt.month
    df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].map(str) + '/' + df['arrival_date_month'].map(str) + '/' + df['arrival_date_day_of_month'].map(str))
    df['arrival_day'] = df['arrival_date'].dt.day_name()
    df['length_of_stay'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']
    df['departure_date'] = df['arrival_date'] + pd.TimedeltaIndex(df['length_of_stay'], unit='D')
    df['departure_day'] = df['departure_date'].dt.day_name()
    df['booking_date'] = df['arrival_date'] - pd.TimedeltaIndex(df['lead_time'], unit='D')
    df['booking_day'] = df['booking_date'].dt.day_name()
    df['revenue'] = df['adr'] * df['length_of_stay']
    df['occupants'] = df['adults'] + df['children'].fillna(value=0) + df['babies']

    cancel_city = df.loc[df['hotel'] == 'City Hotel', ['is_canceled', 'lead_time', 'length_of_stay', 'occupants', 'distribution_channel','is_repeated_guest',
                                                       'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes', 'deposit_type',
                                                       'days_in_waiting_list', 'customer_type', 'adr', 'required_car_parking_spaces', 'total_of_special_requests']]

    return cancel_city


'''Predict cancellation based on lead time, number of nights, number of staying guests, distribution channel ...'''


# Data Exploration
def data_explore(cancel_city):
    sns.countplot(x='is_canceled', data=cancel_city, palette='hls')
    plt.close()

    count_no_cancel = len(cancel_city[cancel_city['is_canceled'] == 0])
    count_cancel = len(cancel_city[cancel_city['is_canceled'] == 1])
    pct_no_cancel = count_no_cancel/(count_cancel+count_no_cancel)
    pct_cancel = count_cancel/(count_cancel+count_no_cancel)

    explore_cancel = cancel_city.groupby('is_canceled').mean()
    explore_distribution = cancel_city.groupby('distribution_channel').mean()
    explore_deposit = cancel_city.groupby('deposit_type').mean()
    explore_customer = cancel_city.groupby('customer_type').mean()

    return pct_cancel, pct_no_cancel


# Visualizations
def data_viz(cancel_city):
    table = pd.crosstab(cancel_city.customer_type, cancel_city.is_canceled)
    table.div(table.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True)
    plt.title('Stacked Bar Chart of Cancellation and Customer')
    plt.xlabel('Customer Type')
    plt.ylabel('Proportion of Cancellations')
    plt.close()
    

# Create Dummy Variables
def data_transform(cancel_city):
    cat_vars = ['distribution_channel', 'deposit_type', 'customer_type']
    for var in cat_vars:
        cat_list = 'var' + '_' + var
        cat_list = pd.get_dummies(cancel_city[var], prefix=var)
        data = cancel_city.join(cat_list)
        cancel_city = data
    data_vars = cancel_city.columns.values.tolist()
    to_keep = [i for i in data_vars if i not in cat_vars]
    city_cancellation = cancel_city[to_keep]

    return city_cancellation


def logistic_regression(city_cancellation):
    # Train Test Split
    X = city_cancellation.loc[:, city_cancellation.columns != 'is_canceled']
    Y = city_cancellation.loc[:, city_cancellation.columns == 'is_canceled']
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=13)
    
    # Recursive Feature Elimination - recursively choose best performing features
    final_vars = city_cancellation.columns.values.tolist()
    Y = ['is_canceled']
    X = [i for i in final_vars if i not in Y]
    
    logreg = LogisticRegression(max_iter=5000)
    
    rfe = RFE(logreg, 10)
    rfe = rfe.fit(X_train, Y_train.values.ravel())
    # print(rfe.support_)
    # print(rfe.ranking_)
    
    cols = ['is_repeated_guest', 'previous_cancellations', 'required_car_parking_spaces', 'distribution_channel_Corporate', 
            'distribution_channel_Direct', 'deposit_type_No Deposit', 'deposit_type_Non Refund', 'customer_type_Transient']
    X = X_train[cols]
    Y = Y_train['is_canceled']
    
    # Implementing the model
    logit_model = sm.Logit(Y, X)
    result = logit_model.fit(method='bfgs')
    
    # Fit Regression Model & Predict Test Set
    logreg.fit(X_train, Y_train)
    y_pred = logreg.predict(X_test)
    accuracy = logreg.score(X_test, Y_test)
    
    # Model Evaluation
    matrix = confusion_matrix(Y_test, y_pred)
    # print(classification_report(Y_test, y_pred))
    
    # ROC Curve
    logit_roc_auc = roc_auc_score(Y_test, logreg.predict(X_test))
    fpr, tpr, thresholds = roc_curve(Y_test, logreg.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' %logit_roc_auc)
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    return accuracy, matrix


def knn(city_cancellation):
    # Define dependent and independent variables
    Y = city_cancellation['is_canceled'].to_numpy()
    X = city_cancellation.iloc[:, 1:].to_numpy()

    _, X_test, _, Y_test = train_test_split(X, Y, test_size=0.33, random_state=13)
    
    # Define model with default parameters
    model = KNeighborsClassifier()
    
    # Define grid value to search
    grid = {'n_neighbors': np.arange(1,25)}
    
    # Define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # Define the grid search procedure
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')
    
    # Execute grid search
    grid_result = grid_search.fit(X, Y)
    
    # Show best score and config
    accuracy = grid_result.best_score_
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    y_pred = grid_result.best_estimator_.predict(X_test)
    matrix = confusion_matrix(Y_test, y_pred)
    
    # Show all computed scores
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, std, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, std, param))

    return accuracy, matrix


if __name__ == '__main__':
    data_path = 'D:/Project/datasets/hotel_bookings.csv'

    data = data_loader(data_path)
    cancel_data = data_transform(data)

    log_acc, log_matrix = logistic_regression(cancel_data)
    knn_acc, knn_matrix = knn(cancel_data)

    print(log_acc, knn_acc)
    print(log_matrix)
    print(knn_matrix)
