"""
Classificador de Spam com Machine Learning — Suporte Multilíngue (EN / PT-BR)
Datasets:
  - English:    SMS Spam Collection (UCI ML Repository)
  - Português:  SMS Spam Multilingual Collection (HuggingFace — dbarbedillo)
Modelos: Naive Bayes, Regressão Logística, SVM Linear
"""

import os
import io
import zipfile
import re
import string
import unicodedata

import requests
import nltk
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from langdetect import detect, LangDetectException, DetectorFactory
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, RSLPStemmer

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

# Seed fixo para resultados determinísticos na detecção de idioma
DetectorFactory.seed = 0

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DATASET_EN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATASET_EN_PATH = os.path.join("data", "SMSSpamCollection")
DATASET_HF_NAME = "dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset"

DATA_DIR = "data"
MODEL_DIR = "models"
PLOTS_DIR = "plots"
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Recursos NLTK
# ---------------------------------------------------------------------------

def _download_nltk_resources():
    resources = [
        ("stopwords", "corpora"),
        ("punkt", "tokenizers"),
        ("punkt_tab", "tokenizers"),
        ("rslp", "stemmers"),
    ]
    for name, kind in resources:
        try:
            nltk.data.find(f"{kind}/{name}")
        except LookupError:
            nltk.download(name, quiet=True)


_download_nltk_resources()


# ---------------------------------------------------------------------------
# Configuração por idioma
# ---------------------------------------------------------------------------

LANG_CONFIG = {
    "en": {
        "label": "English",
        "stopwords": set(stopwords.words("english")),
        "stemmer": PorterStemmer(),
        "model_file": "model_en.joblib",
    },
    "pt": {
        "label": "Português (PT-BR)",
        "stopwords": set(stopwords.words("portuguese")),
        "stemmer": RSLPStemmer(),
        "model_file": "model_pt.joblib",
    },
}


# ---------------------------------------------------------------------------
# 1. Aquisição dos dados
# ---------------------------------------------------------------------------

def load_data_en() -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATASET_EN_PATH):
        print("[dataset-en] Baixando SMS Spam Collection...")
        resp = requests.get(DATASET_EN_URL, timeout=30)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            z.extractall(DATA_DIR)
        print(f"[dataset-en] Salvo em: {DATASET_EN_PATH}")
    else:
        print("[dataset-en] Arquivo já existe, pulando download.")

    df = pd.read_csv(
        DATASET_EN_PATH, sep="\t", header=None,
        names=["label", "text"], encoding="latin-1",
    )
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def _pt_augmentation_data() -> pd.DataFrame:
    """Exemplos nativos de spam/ham PT-BR para compensar limitações do dataset traduzido."""
    samples = [
        # Golpes de prêmio
        ("spam", "VOCÊ GANHOU 5000 REAIS!!! APENAS PREENCHA ESSE FORMULÁRIO PARA SACAR"),
        ("spam", "PARABÉNS!!! Você foi sorteado e ganhou R$10.000! Clique para resgatar AGORA"),
        ("spam", "Você ganhou um iPhone 15! Só preencher o formulário em: link.xyz"),
        ("spam", "SORTEIO: Você foi selecionado para ganhar R$2.000! Responda SIM para participar"),
        ("spam", "Parabéns! Vc ganhou um premio especial. Acesse agora antes que expire!!!"),
        ("spam", "VOCÊ FOI SORTEADO!!! Retire seu prêmio de R$3.500 pelo link abaixo"),
        ("spam", "Ganhou! Seu nome foi sorteado no nosso concurso. Saque em: bit.ly/xxx"),
        # Golpes de banco / CPF
        ("spam", "Seu CPF foi SUSPENSO pela Receita Federal. Regularize AGORA: 0800-XXX-XXXX"),
        ("spam", "CONTA BLOQUEADA! Acesse urgente para desbloquear sua conta: link-falso.com"),
        ("spam", "ALERTA: Sua conta PIX foi suspensa. Clique aqui para reativar em 24h"),
        ("spam", "Caixa Econômica: Seu benefício de R$1.200 está disponível. Saque já: link"),
        ("spam", "URGENTE: Transação suspeita detectada. Confirme seus dados pelo link"),
        ("spam", "SEU CPF ESTÁ IRREGULAR! Acesse agora e regularize antes do bloqueio definitivo"),
        # Golpes de crédito / empréstimo
        ("spam", "Empréstimo de até R$50.000 APROVADO! Sem consulta ao SPC. Ligue já"),
        ("spam", "CRÉDITO PRÉ-APROVADO R$30.000! Taxa 0% nos primeiros 3 meses. Responda SIM"),
        ("spam", "Seu limite foi aumentado para R$15.000! Acesse agora e aproveite"),
        ("spam", "DINHEIRO RÁPIDO! Empréstimo de R$5.000 sem burocracia. Clique e contrate"),
        # Golpes de renda / trabalho
        ("spam", "Vagas URGENTES! Ganhe até R$5000/mês trabalhando em casa. Cadastre-se: link"),
        ("spam", "Ganhe R$300 por dia trabalhando pelo celular. Sem experiência. ENTRE AGORA"),
        ("spam", "OPORTUNIDADE! Trabalhe de casa e ganhe R$2.000/semana. Acesse já"),
        # Outros golpes comuns
        ("spam", "Emagreça 10kg em 15 dias! Produto MILAGROSO com 70% OFF. Peça já"),
        ("spam", "ÚLTIMA CHANCE!!! Promoção relâmpago acaba em 1h. Garanta pelo link: bit.ly/xxx"),
        ("spam", "Você tem uma mensagem não lida! Clique para ver: http://mensagem-secreta.xyz"),
        ("spam", "ATENÇÃO: O WhatsApp vai cobrar amanhã. Encaminhe para 10 amigos para manter grátis"),
        ("spam", "Notícia URGENTE sobre você! Veja antes que seja removido: link suspeito"),
        ("spam", "PROMOÇÃO IMPERDÍVEL!!! Só hoje, 80% de desconto. Compre agora: link"),
        # Ham nativo PT-BR
        ("ham", "Oi, tudo bem? Que horas você chega em casa hoje?"),
        ("ham", "Bom dia! A reunião de amanhã foi cancelada, ok?"),
        ("ham", "Pode me mandar o endereço do restaurante? Vou chegar às 19h"),
        ("ham", "Feliz aniversário! Muitas felicidades pra você"),
        ("ham", "Lembra de trazer o guarda-chuva, está prevendo chuva hoje"),
        ("ham", "Chegou o pedido? Já faz 3 dias que enviei"),
        ("ham", "Obrigado pela ajuda de ontem, você me salvou!"),
        ("ham", "Consegui o ingresso para o show! Vamos juntos no sábado?"),
        ("ham", "O médico confirmou a consulta para segunda às 14h"),
        ("ham", "Boa noite! Conseguiu resolver o problema do computador?"),
        ("ham", "Vou chegar uns 10 minutos atrasado, pode esperar?"),
        ("ham", "Não se esqueça da festa da empresa na sexta"),
    ]
    rows = [
        {"text": text, "label": label, "label_num": 1 if label == "spam" else 0}
        for label, text in samples
    ]
    return pd.DataFrame(rows)


def load_data_pt() -> pd.DataFrame:
    print("[dataset-pt] Carregando dataset PT via HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset(DATASET_HF_NAME, split="train")
    df = ds.to_pandas()[["text_pt", "labels"]].rename(
        columns={"text_pt": "text", "labels": "label"}
    )
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    aug = _pt_augmentation_data()
    df = pd.concat([df, aug], ignore_index=True)
    print(f"[dataset-pt] {len(df)} mensagens carregadas ({len(aug)} exemplos nativos PT-BR incluídos).")
    return df


def load_data(lang: str) -> pd.DataFrame:
    return load_data_en() if lang == "en" else load_data_pt()


# ---------------------------------------------------------------------------
# 2. Pré-processamento
# ---------------------------------------------------------------------------

def preprocess(text: str, lang: str = "en") -> str:
    """Limpa e normaliza texto usando stemmer e stopwords do idioma indicado."""
    cfg = LANG_CONFIG[lang]
    stemmer = cfg["stemmer"]
    stop_words = cfg["stopwords"]

    # Captura sinais estruturais antes de normalizar (são descartados pela remoção de pontuação)
    excl_count = text.count("!")
    letters = [c for c in text if c.isalpha()]
    caps_ratio = sum(1 for c in letters if c.isupper()) / max(len(letters), 1)

    text = text.lower()
    if lang == "pt":
        # Preserva "R$" como token "reais" antes de remover pontuação
        text = re.sub(r"r\$", " reais ", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Injeta tokens de sinal para que o TF-IDF aprenda esses padrões
    if excl_count >= 2:
        text += " exclamacao"
    if caps_ratio > 0.4:
        text += " maiusculas"

    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# 3. Definição dos modelos
# ---------------------------------------------------------------------------

def build_pipelines() -> dict:
    tfidf_params = dict(max_features=10_000, ngram_range=(1, 2), sublinear_tf=True)
    return {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
        "Regressão Logística": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "SVM Linear": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", LinearSVC(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
    }


# ---------------------------------------------------------------------------
# 4. Treinamento e avaliação
# ---------------------------------------------------------------------------

def evaluate_pipeline(pipeline, X_train, X_test, y_train, y_test, name: str):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    print(f"\n{'='*50}\nModelo: {name}\n{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
    return metrics, y_pred


def cross_validate_models(pipelines: dict, X, y) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, pipeline in pipelines.items():
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1", n_jobs=-1)
        rows.append({"Modelo": name, "F1 Médio": scores.mean(), "Desvio Padrão": scores.std()})
        print(f"[CV] {name}: F1 = {scores.mean():.4f} ± {scores.std():.4f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Visualizações
# ---------------------------------------------------------------------------

def plot_confusion_matrices(results: list, y_test, lang: str):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, y_pred) in zip(axes, results):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
        ax.set_title(f"Matriz de Confusão\n{name}")
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"confusion_matrices_{lang}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[plots] Matrizes de confusão salvas em: {path}")


def plot_metrics_comparison(metrics_list: list, lang: str):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    label = LANG_CONFIG[lang]["label"]
    df = pd.DataFrame(metrics_list).set_index("model")
    df = df[["accuracy", "precision", "recall", "f1"]]
    df.columns = ["Acurácia", "Precisão", "Recall", "F1-Score"]
    ax = df.plot(kind="bar", figsize=(10, 5), edgecolor="black", rot=0)
    ax.set_ylim(0.8, 1.02)
    ax.set_title(f"Comparação de Métricas — {label}")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=7)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"metrics_comparison_{lang}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[plots] Comparação de métricas salva em: {path}")


# ---------------------------------------------------------------------------
# 6. Persistência
# ---------------------------------------------------------------------------

_model_cache: dict = {}


def save_model(pipeline, lang: str) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, LANG_CONFIG[lang]["model_file"])
    joblib.dump(pipeline, path)
    _model_cache[lang] = pipeline
    print(f"[model] Modelo {lang.upper()} salvo em: {path}")
    return path


def load_model(lang: str = "en"):
    if lang in _model_cache:
        return _model_cache[lang]
    path = os.path.join(MODEL_DIR, LANG_CONFIG[lang]["model_file"])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Modelo '{lang}' não encontrado em '{path}'. Execute spam_classifier.py primeiro."
        )
    model = joblib.load(path)
    _model_cache[lang] = model
    return model


# ---------------------------------------------------------------------------
# 7. Inferência
# ---------------------------------------------------------------------------

_PT_CHARS = set("ãõçâêôàáéíóú")  # caracteres exclusivos/comuns do português, ausentes no inglês

# Palavras-chave típicas de spam PT-BR (normalizadas, sem acentos)
_PT_SPAM_KEYWORDS = frozenset({
    "ganhou", "ganhe", "premio", "sorteado", "sorteio",
    "sacar", "saque", "resgate", "resgatar", "beneficio",
    "formulario", "aprovado", "preaprovado", "cadastre",
    "cpf", "bloqueado", "suspenso", "desbloqueie",
    "urgente", "urgencia", "encaminhe", "clique aqui",
    "acesse agora", "responda sim", "gratis", "gratuito",
    "promocao", "desconto", "emprestimo", "credito",
    "sem consulta", "taxa zero", "ganhe dinheiro",
    "trabalhando em casa", "reais", "preencha",
})


def _normalize_pt(text: str) -> str:
    """Remove acentos e converte para minúsculas para comparação de palavras-chave."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()


def _heuristic_spam_pt(text: str) -> bool:
    """Detecta padrões de spam PT-BR típicos via heurística léxico-estrutural.

    Combina densidade de palavras-chave com sinais estruturais (caixa alta,
    exclamações, links) que o modelo ML pode não ter aprendido por limitação
    do dataset de treino (tradução automática do inglês).
    """
    normalized = _normalize_pt(text)
    keyword_hits = sum(1 for kw in _PT_SPAM_KEYWORDS if kw in normalized)

    letters = [c for c in text if c.isalpha()]
    caps_ratio = sum(1 for c in letters if c.isupper()) / max(len(letters), 1)
    excl_count = text.count("!")
    has_link = bool(re.search(r"http|bit\.ly|www\.|\.com|\.xyz|link", normalized))

    # 2+ keywords + (caixa alta excessiva OU muitas exclamações OU link suspeito)
    if keyword_hits >= 2 and (caps_ratio > 0.4 or excl_count >= 2 or has_link):
        return True
    # Caixa alta muito alta + qualquer keyword de spam
    if caps_ratio > 0.6 and keyword_hits >= 1:
        return True
    return False


def _detect_lang(text: str) -> str:
    """Retorna 'pt' ou 'en'.
    Prioridade: presença de acentos PT → força 'pt', evitando falhas do langdetect em textos curtos.
    """
    if any(c in _PT_CHARS for c in text.lower()):
        return "pt"
    if len(text.split()) < 3:
        return "en"
    try:
        detected = detect(text)
        return "pt" if detected == "pt" else "en"
    except LangDetectException:
        return "en"


def predict_auto(texts: list[str]) -> list[dict]:
    """Detecta idioma de cada texto e classifica com o modelo correspondente.

    Para PT-BR, aplica heurística léxico-estrutural como fallback quando o ML
    classifica como ham mas padrões típicos de golpe brasileiro são detectados.
    """
    results = []
    for text in texts:
        lang = _detect_lang(text)
        pipeline = load_model(lang)
        processed = preprocess(text, lang)
        pred = pipeline.predict([processed])[0]

        if lang == "pt" and pred != 1 and _heuristic_spam_pt(text):
            pred = 1

        results.append({
            "text": text,
            "lang": lang,
            "prediction": "spam" if pred == 1 else "ham",
        })
    return results


def predict(pipeline, texts: list[str], lang: str = "en") -> list[str]:
    """Classifica textos com um pipeline e idioma explícitos."""
    processed = [preprocess(t, lang) for t in texts]
    preds = pipeline.predict(processed)
    return ["spam" if p == 1 else "ham" for p in preds]


# ---------------------------------------------------------------------------
# 8. Pipeline de treinamento por idioma
# ---------------------------------------------------------------------------

def train_language_model(lang: str):
    cfg = LANG_CONFIG[lang]
    print(f"\n{'#'*60}")
    print(f"  Treinando modelo: {cfg['label']}")
    print(f"{'#'*60}")

    df = load_data(lang)
    print(f"[info] Total: {len(df)} | Spam: {df['label_num'].sum()} | Ham: {(1 - df['label_num']).sum()}")

    print("[preprocess] Aplicando pré-processamento...")
    df["text_clean"] = df["text"].apply(lambda t: preprocess(t, lang))

    X, y = df["text_clean"], df["label_num"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[split] Treino: {len(X_train)} | Teste: {len(X_test)}")

    print("\n=== Validação Cruzada (5-fold, F1) ===")
    pipelines = build_pipelines()
    cross_validate_models(pipelines, X, y)

    print("\n=== Avaliação no Conjunto de Teste ===")
    metrics_list, cm_results = [], []
    for name, pipeline in pipelines.items():
        metrics, y_pred = evaluate_pipeline(pipeline, X_train, X_test, y_train, y_test, name)
        metrics_list.append(metrics)
        cm_results.append((name, y_pred))

    plot_confusion_matrices(cm_results, y_test, lang)
    plot_metrics_comparison(metrics_list, lang)

    best = max(metrics_list, key=lambda m: m["f1"])
    best_pipeline = pipelines[best["model"]]
    print(f"\n[model] Melhor modelo ({lang.upper()}): {best['model']} (F1={best['f1']:.4f})")
    save_model(best_pipeline, lang)
    return best_pipeline


# ---------------------------------------------------------------------------
# 9. Pipeline principal
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("  Classificador de Spam — EN + PT-BR")
    print("=" * 60)

    train_language_model("en")
    train_language_model("pt")

    print("\n=== Demo: detecção automática de idioma ===")
    examples = [
        "Congratulations! You've won a FREE iPhone. Click here to claim your prize NOW!",
        "Hey, are we still meeting for lunch tomorrow?",
        "Win £1000 cash! Text WIN to 80085. Terms apply.",
        "PARABÉNS! Você foi sorteado e ganhou R$5.000! Clique aqui para resgatar.",
        "Oi, tudo bem? Vai no treino hoje à noite?",
        "Seu CPF foi suspenso. Ligue agora para regularizar: 0800-XXX-XXXX",
    ]
    results = predict_auto(examples)
    for r in results:
        label = "SPAM" if r["prediction"] == "spam" else "HAM "
        lang_tag = r["lang"].upper()
        print(f"  [{label}][{lang_tag}] {r['text'][:65]}{'...' if len(r['text']) > 65 else ''}")

    print("\n[done] Treinamento concluído. Modelos salvos na pasta 'models/'.")
    print("        EN → models/model_en.joblib")
    print("        PT → models/model_pt.joblib")


if __name__ == "__main__":
    run()
