"""
Módulo de engenharia de atributos e processamento de dados brutos.

Responsável pelo fluxo completo de ingestão e transformação dos dados,
incluindo tratamentos categóricos (PCA), geoespaciais (DBSCAN) e de texto
(Embeddings NLP).
"""

import pickle
import re
import json
import pandas as pd
import logging
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
from pandas import json_normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tqdm.pandas()
SEM_VALOR = "SEM VALOR"


def carregar_dados(
    carregar_train: bool = True,
    tratar_dados: bool = True,
    div_dataset_validacao: bool = True,
) -> tuple[pd.DataFrame]:
    """
    Carrega, trata e opcionalmente divide os dados de avaliações.

    Parameters
    ----------
    carregar_train : bool, opcional
        Se True, carrega o dataset de treinamento. Se False, carrega o dataset de teste.
        Padrão é True.

    tratar_dados : bool, opcional
        Se True, aplica o pipeline de tratamento nos dados carregados. Padrão é True.

    div_dataset_validacao : bool, opcional
        Se True e tratar_dados for True, divide os dados tratados em treino e validação
        (80/20) salvando os arquivos no disco. Padrão é True.

    Returns
    -------
    tuple[pd.DataFrame]
        Tupla contendo os DataFrames carregados (e possivelmente divididos).
    """
    logger.info("Carregando dados")

    if carregar_train:
        df = pd.read_csv(r"data\X_trainToronto.csv")
    else:
        df = pd.read_csv(r"data\X_testToronto.csv")

    if tratar_dados:
        df = _tratar_df(df, carregar_train)

        if div_dataset_validacao:
            limite_treino = int(len(df) * 0.8)
            df_treino = df[:limite_treino]
            df_validacao = df[limite_treino:]

            df_treino.to_csv("data/treinamento/df_treino.csv")
            df_validacao.to_csv("data/treinamento/df_validacao.csv")

            return df_treino, df_validacao

    return df


def _df_treino_teste_concat():
    """
    Carrega e concatena os dados de treino e teste originais para geração de
    dicionários/vetores globais.

    Returns
    -------
    pd.DataFrame
        DataFrame contendo a concatenação dos dados brutos de treino e teste.
    """
    df_treino = pd.read_csv(r"data\X_trainToronto.csv")
    df_teste = pd.read_csv(r"data\X_testToronto.csv")

    df = pd.concat([df_treino, df_teste])
    df["categories"] = df["categories"].fillna(SEM_VALOR)

    return df


def _tratar_categorias_selecao_pca(
    df: pd.DataFrame, carregar_train: bool
) -> pd.DataFrame:
    """
    Aplica Análise de Componentes Principais (PCA) nas categorias dos negócios.

    Extrai features das categorias convertendo-as em dummies e reduzindo
    a dimensionalidade para 3 componentes principais.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame em tratamento.

    carregar_train : bool
        Indica se estamos operando nos dados de treino (treina o modelo PCA)
        ou teste (carrega o modelo existente e aplica).

    Returns
    -------
    pd.DataFrame
        DataFrame acrescido das 3 componentes PCA (`cat_pca_0`, `cat_pca_1`,
        `cat_pca_2`).
    """
    logger.info("Seleção PCA para categorias")

    df_aux = _df_treino_teste_concat()

    df_aux["categories_list"] = df_aux["categories"].apply(lambda x: x.split(", "))

    df_dummies = pd.get_dummies(
        df_aux.explode("categories_list"), columns=["categories_list"]
    )

    lista_colunas = [col for col in df_dummies.columns if "categories_list" in col]
    lista_colunas.extend(["business_id"])

    df_dummies = df_dummies[lista_colunas]
    df_dummies_group = df_dummies.groupby(df_dummies["business_id"]).max()

    qtdade_componentes_pca = 3
    if carregar_train:
        pca = PCA(n_components=qtdade_componentes_pca)
        df_pca = pd.DataFrame(pca.fit_transform(df_dummies_group))

        with open("data/treinamento/pca_model.pkl", "wb") as file:
            pickle.dump(pca, file)
    else:
        with open("data/treinamento/pca_model.pkl", "rb") as file:
            pca = pickle.load(file)

        df_pca = pd.DataFrame(pca.transform(df_dummies_group))

    df_pca.index = df_dummies_group.index
    df_pca.columns = [f"cat_pca_{i}" for i in range(qtdade_componentes_pca)]

    return pd.merge(df, df_pca, left_index=True, right_index=True, how="inner")


def _tratar_df(df: pd.DataFrame, carregar_train: bool) -> pd.DataFrame:
    """
    Executa a pipeline de tratamenta do DataFrame dos dados a serem avaliados.

    Orquestra chamadas a todos os métodos privados de transformação:
    (categorias vazias, PCA de categorias, embedding NLP, DBSCAN, etc).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame base a ser tratado.

    carregar_train : bool
        Indica o contexto de execução (treinamento vs teste) para salvar ou carregar
        modelos auxiliares de transformação.

    Returns
    -------
    pd.DataFrame
        DataFrame completamente tratado e pronto para consumo pelo modelo.
    """
    df = df.set_index("business_id")

    df = _tratar_linhas_categoria_vazia(df)
    df = _tratar_categorias_selecao_pca(df, carregar_train)
    _tratar_categorias_nao_populares(df, carregar_train)
    _tratar_categorias_populares(df, carregar_train)
    _tratar_coluna_categoria_embedding(df)
    df = _tratar_attributes_selecao_pca(df, carregar_train)
    df = _tratar_agrupamento_dbscan_latitude_longitude(df, carregar_train)
    df = _tratar_agrupamento_dbscan_latitude_longitude(
        df, carregar_train, agr_populares=False, percent_tratamento=0.40
    )
    df = _tratamento_review(df, carregar_train)
    df = _apagar_ordenar_colunas(df)

    return _padronizar_dados_df(df, carregar_train)


def _tratar_agrupamento_dbscan_latitude_longitude(
    df: pd.DataFrame,
    carregar_train: bool,
    agr_populares: bool = True,
    percent_tratamento: float = 0.15,
) -> pd.DataFrame:
    """
    Realiza agrupamento geoespacial dos estabelecimentos usando o algoritmo DBSCAN.

    Cria features booleanas representando clusters densos de negócios populares ou
    não populares, baseado nas coordenadas de latitude e longitude.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame em tratamento.
    carregar_train : bool
        Se True, ajusta o modelo DBSCAN nos dados atuais. Se False, carrega os clusters salvos.
    agr_populares : bool, opcional
        Se True, agrupa os estabelecimentos com mais reviews. Se False, foca nos de menos reviews.
        Padrão é True.
    percent_tratamento : float, opcional
        Percentual de estabelecimentos (ordenados por review) que participarão
        do processo de agrupamento. Padrão é 0.15.

    Returns
    -------
    pd.DataFrame
        DataFrame enriquecido com variáveis dummies dos clusters geoespaciais encontrados.
    """
    df_treino_teste = _df_treino_teste_concat()

    if agr_populares:
        nome_coluna_agrupamento = "agr_lat_log_popular"
        nome_arquivo = "df_mais_reviews_popular.csv"
    else:
        nome_coluna_agrupamento = "agr_lat_log_nao_popular"
        nome_arquivo = "df_mais_reviews_nao_popular.csv"

    if carregar_train:
        df_treino_teste.set_index("business_id", inplace=True)

        df_mais_reviews = df_treino_teste.sort_values(
            "review_count", ascending=(not agr_populares)
        ).iloc[: int(len(df_treino_teste) * percent_tratamento), :]

        dbscan_config_popular = DBSCAN(eps=0.02, min_samples=5)
        dbscan_results_popular = dbscan_config_popular.fit(
            df_mais_reviews[["latitude", "longitude"]]
        )

        df_mais_reviews[nome_coluna_agrupamento] = dbscan_results_popular.labels_
        df_mais_reviews.loc[
            df_mais_reviews[nome_coluna_agrupamento] == -1, nome_coluna_agrupamento
        ] = np.nan

        df_mais_reviews.to_csv(f"data/treinamento/{nome_arquivo}")
    else:
        df_mais_reviews = pd.read_csv(f"data/treinamento/{nome_arquivo}")
        df_mais_reviews.set_index("business_id", inplace=True)

    df_merged = pd.merge(
        df,
        pd.get_dummies(
            df_mais_reviews[nome_coluna_agrupamento],
            columns=[nome_coluna_agrupamento],
            prefix=nome_coluna_agrupamento,
        ),
        left_index=True,
        right_index=True,
        how="left",
    )

    for coluna in [
        coluna for coluna in df_merged.columns if nome_coluna_agrupamento in coluna
    ]:
        df_merged[coluna] = df_merged[coluna].fillna(0.0).astype(float)

    return df_merged


def _tratar_attributes_selecao_pca(
    df: pd.DataFrame, carregar_train: bool
) -> pd.DataFrame:
    """
    Realiza o parse de atributos JSON e aplica PCA para redução de dimensionalidade.

    Converte a string JSON da coluna 'attributes' para features dummies e reduz
    a 3 componentes principais.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame em tratamento.
    carregar_train : bool
        Se True, ajusta e salva o modelo PCA. Se False, carrega e aplica.

    Returns
    -------
    pd.DataFrame
        DataFrame acrescido das 3 componentes PCA (`attr_pca_0`, `attr_pca_1`, `attr_pca_2`).
    """
    df_treino_teste = _df_treino_teste_concat()

    def parse_json(attributes_str):
        if attributes_str:
            try:
                attributes_aux = re.sub(r"(?<!\\)'", '"', attributes_str)
                attributes_aux = re.sub(r'\bu"', '"', attributes_aux)
                attributes_aux = re.sub(r"u\'", '"', attributes_aux)
                attributes_aux = (
                    attributes_aux.replace('""', '"')
                    .replace("'", '"')
                    .replace('"{"', '{"')
                    .replace('}"', "}")
                )

                return json.loads(attributes_aux)
            except json.JSONDecodeError as e:
                return {}
        else:
            return {}

    df_treino_teste["attributes"] = df_treino_teste["attributes"].fillna("")
    df_treino_teste["attributes_dict"] = df_treino_teste["attributes"].apply(parse_json)

    attributes_df = json_normalize(df_treino_teste["attributes_dict"])
    attributes_df["RestaurantsPriceRange2"] = attributes_df[
        "RestaurantsPriceRange2"
    ].fillna(0)
    attributes_df = attributes_df.fillna("False").astype(str)
    attributes_df = pd.get_dummies(attributes_df, dtype=float)

    qtdade_componentes_pca = 3

    if carregar_train:
        pca = PCA(n_components=qtdade_componentes_pca)
        df_pca = pd.DataFrame(pca.fit_transform(attributes_df))

        with open("data/treinamento/pca_model_attr.pkl", "wb") as file:
            pickle.dump(pca, file)
    else:
        with open("data/treinamento/pca_model_attr.pkl", "rb") as file:
            pca = pickle.load(file)

        df_pca = pd.DataFrame(pca.transform(attributes_df))

    df_treino_teste.set_index("business_id", inplace=True)
    df_pca.index = df_treino_teste.index
    df_pca.columns = [f"attr_pca_{i}" for i in range(qtdade_componentes_pca)]

    return pd.merge(df, df_pca, left_index=True, right_index=True, how="inner")


def _tratamento_review(df: pd.DataFrame, carregar_train: bool) -> pd.DataFrame:
    """
    Calcula a média do sentimento das avaliações (reviews) dos estabelecimentos.

    Utiliza o SentimentIntensityAnalyzer do NLTK para extrair a polaridade composta
    e agrupa a média por negócio.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame base para o merge.
    carregar_train : bool
        Indica qual arquivo de reviews carregar (treino ou teste).

    Returns
    -------
    pd.DataFrame
        DataFrame enriquecido com a feature contínua 'avg_sentimento'.
    """
    logger.info("Tratando reviews")

    if carregar_train:
        df_reviews = pd.read_csv(r"data\reviewsTrainToronto.csv")
    else:
        df_reviews = pd.read_csv(r"data\reviewsTestToronto.csv")

    sentiment = SentimentIntensityAnalyzer()
    df_reviews["sentimento"] = df_reviews["text"].progress_apply(
        lambda x: sentiment.polarity_scores(x)["compound"]
    )

    df_reviews_agrupado = df_reviews.groupby("business_id").agg({"sentimento": "mean"})
    df_reviews_agrupado.columns = [
        "avg_sentimento",
    ]

    return pd.merge(df, df_reviews_agrupado, left_index=True, right_index=True)


def _tratar_categorias_nao_populares(df: pd.DataFrame, carregar_train: bool):
    """
    Calcula a similaridade de cosseno para categorias associadas a más avaliações.

    Usa Word Embeddings do SpaCy para identificar o quão similar a categoria atual
    é em relação às categorias que historicamente não são populares. Adiciona
    uma nova feature `categories_cossine`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame em tratamento. A feature é adicionada in-place.

    carregar_train : bool
        Se True, gera e salva a lista de vetores de categorias mal avaliadas.
        Se False, carrega a lista existente.
    """
    logger.info("Tratando categorias não populares usando similaridade do cosseno")
    nlp = spacy.load("en_core_web_md")

    if carregar_train:
        df_porcent = df.groupby(["categories", "destaque"]).size().unstack(fill_value=0)

        df_porcent["porcentagem"] = df_porcent[0] / (df_porcent[0] + df_porcent[1])
        df_porcent = df_porcent[
            (df_porcent[0] > 10) & (df_porcent[1] > 10)
        ].sort_values("porcentagem", ascending=False)

        lista_cat_mau_aval = [
            nlp(cat).vector.reshape(1, -1)
            for cat in df_porcent[df_porcent["porcentagem"] > 0.7].index.to_list()
        ]

        with open("data/treinamento/lista_cat_mau_aval.pkl", "wb") as file:
            pickle.dump(lista_cat_mau_aval, file)
    else:
        with open("data/treinamento/lista_cat_mau_aval.pkl", "rb") as file:
            lista_cat_mau_aval = pickle.load(file)

    if lista_cat_mau_aval:
        bad_matrix = np.vstack(
            [v.reshape(1, -1) if v.ndim == 1 else v for v in lista_cat_mau_aval]
        )
    else:
        bad_matrix = None

    cat_cache = {}

    def calculate_max_cosine_similarity(categories_str):
        if bad_matrix is None or not categories_str:
            return 0.0
        if categories_str not in cat_cache:
            vec = nlp(categories_str).vector.reshape(1, -1)
            cat_cache[categories_str] = float(cosine_similarity(vec, bad_matrix).max())
        return cat_cache[categories_str]

    df["categories_cossine"] = df["categories"].progress_apply(
        calculate_max_cosine_similarity
    )


def _tratar_coluna_categoria_embedding(df: pd.DataFrame) -> None:
    """
    Cria uma representação vetorial (embedding) média do texto das categorias.

    Utiliza o modelo pré-treinado do SpaCy (en_core_web_md) para gerar uma feature
    multidimensional agregada baseada na string original de categorias.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame em tratamento. A feature `categories_embedding` é adicionada in-place.
    """
    logger.info("Acrescentando coluna embedding para representar as categorias")

    nlp = spacy.load("en_core_web_md")

    cat_emb_cache = {}

    def mean_embedding(text):
        if text not in cat_emb_cache:
            vec = nlp(text).vector
            cat_emb_cache[text] = float(np.mean(vec)) if len(vec) > 0 else 0.0
        return cat_emb_cache[text]

    df["categories_embedding"] = df["categories"].progress_apply(mean_embedding)


def _tratar_linhas_categoria_vazia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preenche valores ausentes na coluna 'categories'.

    Substitui NaN pela constante indicadora predefinida (SEM_VALOR).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame em tratamento.

    Returns
    -------
    pd.DataFrame
        DataFrame com as categorias vazias preenchidas.
    """
    logger.info("Tratando linhas com categoria vazia")

    df["categories"] = df["categories"].fillna(SEM_VALOR)

    return df


def _tratar_categorias_populares(df: pd.DataFrame, carregar_train: bool):
    """
    Identifica se o estabelecimento possui alguma categoria considerada popular (destaque).

    Adiciona uma feature booleana in-place `popular_categories`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame em tratamento.
    carregar_train : bool
        Se True, minera as categorias populares e salva a lista. Se False, carrega a lista.
    """
    logger.info("Adicionando coluna categoria popular")

    if carregar_train:
        df_cat = df.copy()

        df_cat["categories_list"] = df_cat["categories"].apply(lambda x: x.split(", "))

        df_cat = df_cat.explode("categories_list")

        contagem_destaque = (
            df_cat.groupby(["categories_list", "destaque"]).size().unstack(fill_value=0)
        )

        lista_categorias = contagem_destaque[
            contagem_destaque[0] < contagem_destaque[1]
        ].index.to_list()

        with open("data/treinamento/lista_categorias_populares.pkl", "wb") as file:
            pickle.dump(lista_categorias, file)
    else:
        with open("data/treinamento/lista_categorias_populares.pkl", "rb") as file:
            lista_categorias = pickle.load(file)

    def contains_word(category):
        words = category.split(", ")
        return any(word in lista_categorias for word in words)

    df["popular_categories"] = df["categories"].apply(contains_word).astype(float)


def _apagar_ordenar_colunas(df):
    """
    Remove colunas originais que já foram transformadas e ordena as restantes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame após todas as transformações de features.

    Returns
    -------
    pd.DataFrame
        DataFrame enxuto, com colunas redundantes descartadas e a target no final.
    """
    df = df.drop(
        columns=[
            "name",
            "address",
            "postal_code",
            "attributes",
            "categories",
            "hours",
            "loc",
            "latitude",
            "longitude",
        ],
    )

    colunas = [col for col in df.columns if col != "destaque"] + ["destaque"]
    return df.reindex(columns=colunas)


def _padronizar_dados_df(df: pd.DataFrame, carregar_train: bool) -> pd.DataFrame:
    """
    Aplica a padronização Z-score nas features numéricas do dataset.

    Utiliza StandardScaler do scikit-learn para garantir que os dados tenham média 0
    e variância 1, desconsiderando a coluna alvo de destaque.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo as features finais antes da padronização.
    carregar_train : bool
        Se True, treina e salva o scaler. Se False, carrega e aplica aos novos dados.

    Returns
    -------
    pd.DataFrame
        DataFrame com as variáveis independentes padronizadas.
    """
    logger.info("Padronizar dados")

    df_aux = df.drop(columns="destaque").astype(float)

    if carregar_train:
        scaler = StandardScaler()
        df_aux = pd.DataFrame(scaler.fit_transform(df_aux))
        with open("data/treinamento/scaler.pkl", "wb") as file:
            pickle.dump(scaler, file)
    else:
        with open("data/treinamento/scaler.pkl", "rb") as file:
            scaler = pickle.load(file)
        df_aux = pd.DataFrame(scaler.transform(df_aux))

    df_aux.index = df.index
    df_aux["destaque"] = df["destaque"]
    df_aux.columns = df.columns

    return df_aux
