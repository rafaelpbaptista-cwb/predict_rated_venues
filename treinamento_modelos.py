"""
Módulo de treinamento e avaliação de modelos de Machine Learning.

Fornece funções para treinar RandomForest, XGBoost, MLP e um VotingClassifier,
além de validar os modelos utilizando a métrica F1-Score ponderada.
"""

import pandas as pd
import logging
import pickle
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer


def treinar_random_forest(
    df_treino: pd.DataFrame, salvar_modelo: bool = True
) -> tuple[float, RandomForestClassifier]:
    """
    Treina um modelo Random Forest Classifier utilizando GridSearchCV.

    Parameters
    ----------
    df_treino : pd.DataFrame
        DataFrame contendo as features e a variável alvo na última coluna.
    salvar_modelo : bool, opcional
        Indica se o modelo final deve ser salvo no disco em formato pickle,
        por padrão True.

    Returns
    -------
    tuple[float, RandomForestClassifier]
        Tupla contendo o melhor score de F1 obtido no GridSearch e o modelo treinado.
    """
    param_grid_completo = {
        "criterion": ["gini"],
        "n_estimators": [50, 100],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "class_weight": [None, "balanced"],
        "random_state": [42],
    }

    param_grid_minimo = {
        "n_estimators": [10, 50],
        "max_depth": [None, 10, 20],
        "criterion": ["gini", "entropy"],
        "min_samples_split": [2, 10],
        "class_weight": [None, "balanced"],
        "random_state": [42],
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(),
        param_grid_completo,
        cv=5,
        scoring=make_scorer(f1_score, average="weighted"),
        verbose=0,
    )

    features = df_treino.iloc[:, :-1]
    target = df_treino.iloc[:, -1]

    grid_search.fit(features, target)

    logging.info("Score RandomForestClassifier: %s", grid_search.best_score_)
    logging.info("Melhores parâmetros: %s", grid_search.best_params_)

    model_final = RandomForestClassifier(**grid_search.best_params_)
    model_final.fit(features, target)

    if salvar_modelo:
        with open("data/models/random_forest_model.pkl", "wb") as file:
            pickle.dump(model_final, file)

    return (
        grid_search.best_score_,
        model_final,
    )


def treinar_xgboost(
    df_treino: pd.DataFrame, salvar_modelo: bool = True
) -> tuple[float, XGBClassifier]:
    """
    Treina um modelo XGBoost Classifier utilizando GridSearchCV.

    Parameters
    ----------
    df_treino : pd.DataFrame
        DataFrame contendo as features e a variável alvo na última coluna.
    salvar_modelo : bool, opcional
        Indica se o modelo final deve ser salvo no disco em formato pickle,
        por padrão True.

    Returns
    -------
    tuple[float, XGBClassifier]
        Tupla contendo o melhor score de F1 obtido no GridSearch e o modelo treinado.
    """
    param_grid_completo = {
        "learning_rate": [0.05, 0.1],
        "n_estimators": [50, 100],
        "max_depth": [4, 6],
        "min_child_weight": [1, 3],
        "gamma": [0, 0.1],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [0.8, 0.9],
        "scale_pos_weight": [1, 2],
        "random_state": [42],
        "use_label_encoder": [False],
        "eval_metric": ["logloss"],
    }

    param_grid_minimo = {
        "learning_rate": [0.1],
        "n_estimators": [50, 100],
        "max_depth": [3, 6],
        "random_state": [42],
        "use_label_encoder": [False],
        "eval_metric": ["logloss"],
    }

    grid_search = GridSearchCV(
        XGBClassifier(),
        param_grid_completo,
        cv=5,
        scoring=make_scorer(f1_score, average="weighted"),
        verbose=0,
    )

    features = df_treino.iloc[:, :-1]
    target = df_treino.iloc[:, -1]

    grid_search.fit(features, target)

    logging.info("Score XGBClassifier: %s", grid_search.best_score_)
    logging.info("Melhores parâmetros: %s", grid_search.best_params_)

    model_final = XGBClassifier(**grid_search.best_params_)
    model_final.fit(features, target)

    if salvar_modelo:
        with open("data/models/xgboost_model.pkl", "wb") as file:
            pickle.dump(model_final, file)

    return (
        grid_search.best_score_,
        model_final,
    )


def treinar_mlp(
    df_treino: pd.DataFrame, salvar_modelo: bool = True
) -> tuple[float, MLPClassifier]:
    """
    Treina um modelo Multi-Layer Perceptron (MLP) utilizando GridSearchCV.

    Parameters
    ----------
    df_treino : pd.DataFrame
        DataFrame contendo as features e a variável alvo na última coluna.
    salvar_modelo : bool, opcional
        Indica se o modelo final deve ser salvo no disco em formato pickle,
        por padrão True.

    Returns
    -------
    tuple[float, MLPClassifier]
        Tupla contendo o melhor score de F1 obtido no GridSearch e o modelo treinado.
    """
    param_grid_completo = {
        "hidden_layer_sizes": [(100,), (50, 50), (100, 100)],
        "activation": ["relu"],
        "solver": ["sgd", "adam"],
        "alpha": [0.0001, 0.001],
        "learning_rate": ["constant", "adaptive"],
        "max_iter": [300, 600],
        "random_state": [42],
    }

    param_grid_minimo = {
        "hidden_layer_sizes": [(50,), (100,)],
        "activation": ["relu"],
        "solver": ["adam"],
        "alpha": [0.0001, 0.001],
        "max_iter": [200, 500],
        "random_state": [42],
    }

    grid_search = GridSearchCV(
        MLPClassifier(),
        param_grid_completo,
        cv=5,
        scoring=make_scorer(f1_score, average="weighted"),
        verbose=0,
    )

    features = df_treino.iloc[:, :-1]
    target = df_treino.iloc[:, -1]

    grid_search.fit(features, target)

    logging.info("Score MLPClassifier: %s", grid_search.best_score_)
    logging.info("Melhores parâmetros: %s", grid_search.best_params_)

    model_final = MLPClassifier(**grid_search.best_params_)
    model_final.fit(features, target)

    if salvar_modelo:
        with open("data/models/mlp_model.pkl", "wb") as file:
            pickle.dump(model_final, file)

    return (
        grid_search.best_score_,
        model_final,
    )


def treinar_voting_classifier(
    df_treino: pd.DataFrame, salvar_modelo: bool = True
) -> tuple[float, VotingClassifier]:
    """
    Treina um Voting Classifier combinando os modelos RandomForest, XGBoost e MLP.

    A combinação é feita utilizando hard voting com os modelos previamente
    treinados e salvos em disco.

    Parameters
    ----------
    df_treino : pd.DataFrame
        DataFrame contendo as features e a variável alvo na última coluna.
    salvar_modelo : bool, opcional
        Indica se o modelo final deve ser salvo no disco em formato pickle,
        por padrão True.

    Returns
    -------
    tuple[float, VotingClassifier]
        Tupla contendo o score de F1 obtido no conjunto de treino e o modelo de ensemble treinado.
    """
    # Carregando os modelos pré-treinados
    with open("data/models/random_forest_model.pkl", "rb") as file:
        rf_model = pickle.load(file)

    with open("data/models/xgboost_model.pkl", "rb") as file:
        xgb_model = pickle.load(file)

    with open("data/models/mlp_model.pkl", "rb") as file:
        mlp_model = pickle.load(file)

    # Criando o VotingClassifier com os três modelos
    voting_classifier = VotingClassifier(
        estimators=[("rf", rf_model), ("xgb", xgb_model), ("mlp", mlp_model)],
        voting="hard",
    )

    features = df_treino.iloc[:, :-1]
    target = df_treino.iloc[:, -1]

    # Treinando o VotingClassifier
    voting_classifier.fit(features, target)

    # Calculando o F1-Score usando o próprio conjunto de treino
    predictions = voting_classifier.predict(features)
    f1 = f1_score(target, predictions, average="weighted")

    logging.info("F1-Score do VotingClassifier: %s", f1)

    if salvar_modelo:
        with open("data/models/voting_classifier_model.pkl", "wb") as file:
            pickle.dump(voting_classifier, file)

    return (
        f1,
        voting_classifier,
    )


def validar_modelo(modelo, df_validacao: pd.DataFrame) -> float:
    """
    Avalia a performance de um modelo preditivo num conjunto de validação.

    Parameters
    ----------
    modelo : estimator
        Modelo treinado (ex: RandomForestClassifier, VotingClassifier) que possui o método `predict`.
    df_validacao : pd.DataFrame
        DataFrame contendo as features de validação e a variável alvo na última coluna.

    Returns
    -------
    float
        O valor da métrica F1-Score ponderada (weighted) para as predições.
    """
    features = df_validacao.iloc[:, :-1]
    target = df_validacao.iloc[:, -1]

    previsoes = modelo.predict(features)

    f1_valor = f1_score(target, previsoes, average="weighted")

    logging.info("F1-score dataset validação: %s", f1_valor)

    return f1_valor
