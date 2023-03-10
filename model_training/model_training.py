from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

RF_DISCRETIZER_N_BINS = 9
RF_DISCRETIZER_ENCODE = "ordinal"
RF_CLF_N_ESTIMATORS = 130
RF_CLF_CRITERION = "entropy"
RF_CLF_MAX_DEPTH = None
RF_CLF_BOOTSTRAP = False

XGB_DISCRETIZER_N_BINS = 9
XGB_DISCRETIZER_ENCODE = "ordinal"
XGB_CLF_ETA = 0.3
XGB_CLF_MIN_CHILD_WEIGHT = 9
XGB_CLF_MAX_DEPTH = 7

def train_model(X_train, y_train, model_algorithm):
    """
    Receives the sets for training and a selector for the model algorithm (currently only XGBoost and RandomForest).
    Returns the pipeline containing the trained model. 
    """
    match model_algorithm:
        case "XGBoost":
            discretizer = KBinsDiscretizer(
                strategy="kmeans",
                n_bins=XGB_DISCRETIZER_N_BINS,
                encode=XGB_DISCRETIZER_ENCODE
            )
            classifier = XGBClassifier(
                objective="multi:softmax",
                num_class=6,
                eta=XGB_CLF_ETA,
                min_weight_child=XGB_CLF_MIN_CHILD_WEIGHT,
                max_depth=XGB_CLF_MAX_DEPTH
            )
            
        case "RandomForest":
            discretizer = KBinsDiscretizer(
                strategy="kmeans",
                n_bins=RF_DISCRETIZER_N_BINS,
                encode=RF_DISCRETIZER_ENCODE
            )
            classifier = RandomForestClassifier(
                n_estimators=RF_CLF_N_ESTIMATORS,
                criterion=RF_CLF_CRITERION,
                max_depth=RF_CLF_MAX_DEPTH,
                bootstrap=RF_CLF_BOOTSTRAP
            )

    pipeline = Pipeline([
        ('discretizer', discretizer),
        ('clf', classifier)
    ])

    pipeline.fit(X_train, y_train.values.ravel())
    return pipeline


