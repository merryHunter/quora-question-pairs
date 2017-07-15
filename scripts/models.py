import xgboost as xgb

# TODO: add neural network training
#


def train_xgb(X_train, X_valid, y_train, y_valid, params, rounds):
    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_val = xgb.DMatrix(X_valid, label=y_valid)

    watchlist  = [(xg_train,'train'), (xg_val,'eval')]
    return xgb.train(params, xg_train, rounds, watchlist, early_stopping_rounds=50, verbose_eval=50)


def predict_xgb(clr, X_test):
    return clr.predict(xgb.DMatrix(X_test))

