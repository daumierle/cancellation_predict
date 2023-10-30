import pandas as pd
import warnings
import joblib

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings('ignore')


def data_loader(path):
    df = pd.read_csv(path)

    '''Data Prep'''

    df['arrival_date_month'] = pd.to_datetime(df.arrival_date_month, format='%B').dt.month
    df['arrival_date'] = pd.to_datetime(
        df['arrival_date_year'].map(str) + '/' + df['arrival_date_month'].map(str) + '/' + df[
            'arrival_date_day_of_month'].map(str))
    df['arrival_day'] = df['arrival_date'].dt.day_name()
    df['length_of_stay'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']
    df['departure_date'] = df['arrival_date'] + pd.TimedeltaIndex(df['length_of_stay'], unit='D')
    df['departure_day'] = df['departure_date'].dt.day_name()
    df['booking_date'] = df['arrival_date'] - pd.TimedeltaIndex(df['lead_time'], unit='D')
    df['booking_day'] = df['booking_date'].dt.day_name()
    df['revenue'] = df['adr'] * df['length_of_stay']
    df['occupants'] = df['adults'] + df['children'].fillna(value=0) + df['babies']

    cancel_city = df.loc[
        df['hotel'] == 'City Hotel', ['is_canceled', 'lead_time', 'length_of_stay', 'occupants', 'distribution_channel',
                                      'is_repeated_guest',
                                      'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes',
                                      'deposit_type',
                                      'days_in_waiting_list', 'customer_type', 'adr', 'required_car_parking_spaces',
                                      'total_of_special_requests']]

    # Create Dummy Variables
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


def gbtree_grid_search(city_cancellation):
    # Define dependent and independent variables
    Y = city_cancellation['is_canceled'].to_numpy()
    X = city_cancellation.iloc[:, 1:].to_numpy()

    # Define model with default hyperparameters
    model = GradientBoostingClassifier()

    # Define the grid of values to search
    grid = dict()
    grid['n_estimators'] = [10, 50, 100, 500]
    grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
    grid['subsample'] = [0.5, 0.7, 1.0]
    grid['max_depth'] = [3, 7, 9]

    # Define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    # Define the grid search procedure
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')

    # Execute the grid search
    grid_result = grid_search.fit(X, Y)

    # Show the best score and configuration
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    # Show all scores that were evaluated
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, std, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, std, param))

    joblib.dump(grid_result.best_params_, 'best_gbtree.pkl')


if __name__ == "__main__":
    data_path = 'D:/Project/datasets/hotel_bookings.csv'
    data = data_loader(data_path)

    search = True

    if search:
        gbtree_grid_search(data)
    else:
        # Train Test Split
        X = data.loc[:, data.columns != 'is_canceled']
        Y = data.loc[:, data.columns == 'is_canceled']

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=13)

        gbtree_params = joblib.load('best_gbtree.pkl')

        model = GradientBoostingClassifier()
        model.set_params(**gbtree_params)

        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        accuracy = model.score(X_test, Y_test)
        matrix = confusion_matrix(Y_test, Y_pred)

        print(accuracy)
        print(matrix)
        print(classification_report(Y_test, Y_pred))