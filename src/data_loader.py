"""Descarga, normalización y carga del dataset UCI 'Default of Credit Card Clients'.

Fuente: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
Cita:   Yeh, I. (2009). Default of Credit Card Clients [Dataset].
        UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
Licencia: CC BY 4.0

Los datos se descargan una sola vez y se guardan en data/raw/credit_default.csv.
Ese CSV se ignora en Git: el repositorio queda ligero y reproducible.

Uso:
    from src.data_loader import load_credit_default
    X, y = load_credit_default()
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "credit_default.csv"
UCI_ID = 350
TARGET = "default"

# ucimlrepo devuelve columnas X1..X23; las renombramos al nombre del paper.
FEATURE_RENAME = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6",
}


def _normalize(X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas, elimina el ID y deja la etiqueta en ``default``."""
    X = X.copy()
    y = y.copy()

    X = X.rename(columns={k: v for k, v in FEATURE_RENAME.items() if k in X.columns})
    X = X.drop(columns=[c for c in ("ID", "id", "Id") if c in X.columns])

    target_col = y.columns[0]
    y = y.rename(columns={target_col: TARGET}).astype("int8")

    df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    return df


def _download_from_ucirepo() -> pd.DataFrame:
    """Vía principal: paquete oficial ucimlrepo."""
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=UCI_ID)
    return _normalize(dataset.data.features, dataset.data.targets)


def _download_from_xls() -> pd.DataFrame:
    """Vía alternativa: ZIP oficial con el .xls original (si ucimlrepo falla)."""
    import io
    import zipfile

    import requests

    url = (
        "https://archive.ics.uci.edu/static/public/350/"
        "default+of+credit+card+clients.zip"
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        xls_name = next(n for n in archive.namelist() if n.lower().endswith((".xls", ".xlsx")))
        raw = archive.read(xls_name)

    # El .xls de UCI tiene una fila de título antes de los encabezados.
    df = pd.read_excel(io.BytesIO(raw), header=1)
    df.columns = [str(c).strip() for c in df.columns]

    # La etiqueta suele llamarse 'Y' o 'default payment next month'.
    target_col = next(
        (c for c in df.columns if c.lower() in {"y", "default"} or "default payment" in c.lower()),
        df.columns[-1],
    )
    return _normalize(df.drop(columns=[target_col]), df[[target_col]])


def load_credit_default(force_download: bool = False, save: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve ``(X, y)`` con columnas con nombre legible y etiqueta ``default``.

    Parameters
    ----------
    force_download:
        Si es True, vuelve a descargar aunque ya exista el CSV local.
    save:
        Si es True, guarda el CSV normalizado en ``data/raw/``.
    """
    if RAW_CSV.exists() and not force_download:
        df = pd.read_csv(RAW_CSV)
    else:
        try:
            df = _download_from_ucirepo()
        except Exception as exc:  # pragma: no cover - depende de la red
            print(f"[data_loader] ucimlrepo falló ({exc}); usando el .xls oficial...")
            df = _download_from_xls()
        if save:
            RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(RAW_CSV, index=False)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype("int8")
    return X, y


def describe(X: pd.DataFrame, y: pd.Series) -> None:
    """Imprime un resumen rápido para la primera celda del EDA."""
    print(f"Forma de X: {X.shape} | forma de y: {y.shape}")
    print(f" Variables: {len(X.columns)} -> {', '.join(X.columns)}")
    print("\nDistribución de la clase objetivo:")
    print(y.value_counts().rename(index={0: "pago (0)", 1: "impago (1)"}).to_string())
    print(f"\nImpago: {y.mean():.2%}")
    print(f"Valores faltantes: {int(X.isna().sum().sum())}")
    print("\nTipos:")
    print(X.dtypes.value_counts().to_string())


if __name__ == "__main__":
    X, y = load_credit_default()
    describe(X, y)
    print(f"\nCSV local: {RAW_CSV.relative_to(ROOT)}")
