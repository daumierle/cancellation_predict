from config.core import config
from utils.data_utils import load_dataset, data_transform_v2, save_pipeline
from utils.features import feature_selection

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def train_logistic_regression():
    data = load_dataset(config.app_config.training_data_file)
    data = data_transform_v2(data)

    # divide train and test
    X = data.loc[:, data.columns != config.cancel_model_config.target]
    Y = data.loc[:, data.columns == config.cancel_model_config.target]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=config.cancel_model_config.test_size,
                                                        random_state=config.cancel_model_config.random_state)

    # perform feature selection
    feat_cols = feature_selection(X_train, y_train)
    X_train = X_train[feat_cols]
    y_train = y_train[config.cancel_model_config.target]

    # fit model
    cancel_pipeline = LogisticRegression(random_state=config.cancel_model_config.random_state)
    cancel_pipeline.fit(X_train, y_train)

    # persist trained model
    save_pipeline(pipeline_to_persist=cancel_pipeline)


if __name__ == "__main__":
    train_logistic_regression()
