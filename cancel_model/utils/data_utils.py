import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from typing import List
from pathlib import Path
from cancel_model import __version__ as _version
from cancel_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def load_dataset(file_name):
    df = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"), na_values="NULL")

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

    return cancel_city


def save_pipeline(pipeline_to_persist):
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """
    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(file_name: str):
    """Load a persisted pipeline."""
    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(files_to_keep: List[str]):
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


# Data Exploration
def data_explore(cancel_city):
    sns.countplot(x='is_canceled', data=cancel_city, palette='hls')
    plt.close()

    count_no_cancel = len(cancel_city[cancel_city['is_canceled'] == 0])
    count_cancel = len(cancel_city[cancel_city['is_canceled'] == 1])
    pct_no_cancel = count_no_cancel / (count_cancel + count_no_cancel)
    pct_cancel = count_cancel / (count_cancel + count_no_cancel)

    explore_cancel = cancel_city.groupby('is_canceled').mean()
    explore_distribution = cancel_city.groupby('distribution_channel').mean()
    explore_deposit = cancel_city.groupby('deposit_type').mean()
    explore_customer = cancel_city.groupby('customer_type').mean()

    return pct_cancel, pct_no_cancel


# Visualizations
def data_viz(cancel_city):
    table = pd.crosstab(cancel_city.customer_type, cancel_city.is_canceled)
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Stacked Bar Chart of Cancellation and Customer')
    plt.xlabel('Customer Type')
    plt.ylabel('Proportion of Cancellations')
    plt.close()


# Create Dummy Variables
def data_transform(cancel_city):
    cat_vars = ['distribution_channel', 'deposit_type', 'customer_type']
    for var in cat_vars:
        # cat_list = 'var' + '_' + var
        cat_list = pd.get_dummies(cancel_city[var], prefix=var, dtype=float)
        data = cancel_city.join(cat_list)
        cancel_city = data
    data_vars = cancel_city.columns.values.tolist()
    to_keep = [i for i in data_vars if i not in cat_vars]
    city_cancellation = cancel_city[to_keep]

    return city_cancellation


def data_transform_v2(cancel_city):
    cat_vars = ['distribution_channel', 'deposit_type', 'customer_type']
    for var in cat_vars:
        unique_vals = list()
        for val in cancel_city[var]:
            if val not in unique_vals:
                unique_vals.append(val)
        mapping = {val: i + 1 for i, val in enumerate(unique_vals)}
        cancel_city[var] = cancel_city[var].map(mapping)
    return cancel_city
