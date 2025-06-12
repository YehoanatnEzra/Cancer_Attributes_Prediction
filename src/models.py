from sklearn.linear_model import LogisticRegression, RidgeClassifier, ElasticNet, LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier


def get_model(task: str, model_name: str, **kwargs):
    """
    Returns a model for the given task and model name.

    Args:
        task: 'classification' or 'regression'
        model_name: name of the model (e.g., 'logistic', 'rf', 'knn', etc.)
        **kwargs: optional parameters to override defaults

    Returns:
        A scikit-learn model instance
    """
    if task == "classification":
        model_map = {
            "logistic": LogisticRegression(max_iter=1000),
            "knn": KNeighborsClassifier(),
            "decision_tree": DecisionTreeClassifier(),
            "rf": RandomForestClassifier(n_estimators=200, random_state=1),
            "ridge": RidgeClassifier()
        }
        base_model = model_map.get(model_name)
        if base_model is None:
            raise ValueError(f"Unknown classification model: {model_name}")
        # Update parameters if any
        base_model.set_params(**kwargs)
        return MultiOutputClassifier(base_model)

    elif task == "regression":
        model_map = {
            "linear": LinearRegression(),
            "sgd": SGDRegressor(max_iter=1000),
            "boost": GradientBoostingRegressor(),
            "elasticnet": ElasticNet()
        }
        model = model_map.get(model_name)
        if model is None:
            raise ValueError(f"Unknown regression model: {model_name}")
        model.set_params(**kwargs)
        return model

    else:
        raise ValueError("Task must be 'classification' or 'regression'")
