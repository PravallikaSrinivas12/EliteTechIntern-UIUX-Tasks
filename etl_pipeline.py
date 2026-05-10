"""
=============================================================
  AUTOMATED ETL DATA PIPELINE
  Tools: Pandas, Scikit-learn, NumPy
  Stages: Extract → Preprocess → Transform → Load
=============================================================
"""

import os
import time
import logging
import warnings
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime

# Scikit-learn imports
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, LabelEncoder,
    OneHotEncoder, RobustScaler
)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  LOGGING SETUP
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("etl_pipeline.log"),
    ]
)
logger = logging.getLogger("ETLPipeline")


# ═══════════════════════════════════════════════════════════
#  STAGE 1 — EXTRACT
# ═══════════════════════════════════════════════════════════

class DataExtractor:
    """
    Handles ingestion from CSV, Excel, JSON, Parquet,
    or in-memory DataFrames. Easily extensible to
    database sources (SQLAlchemy) or REST APIs.
    """

    SUPPORTED_FORMATS = {".csv", ".xlsx", ".xls", ".json", ".parquet"}

    def __init__(self, source):
        """
        Args:
            source: file path (str/Path) OR a pandas DataFrame
                    (for in-memory / programmatic use).
        """
        self.source = source
        self.df: pd.DataFrame | None = None
        self.metadata: dict = {}

    def extract(self) -> pd.DataFrame:
        start = time.perf_counter()
        logger.info("━" * 55)
        logger.info("STAGE 1 — EXTRACT")

        if isinstance(self.source, pd.DataFrame):
            self.df = self.source.copy()
            logger.info("Source: in-memory DataFrame")
        else:
            path = Path(self.source)
            if not path.exists():
                raise FileNotFoundError(f"Source not found: {path}")
            ext = path.suffix.lower()
            if ext not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported format '{ext}'. Supported: {self.SUPPORTED_FORMATS}")

            readers = {
                ".csv":     lambda p: pd.read_csv(p),
                ".json":    lambda p: pd.read_json(p),
                ".parquet": lambda p: pd.read_parquet(p),
                ".xlsx":    lambda p: pd.read_excel(p),
                ".xls":     lambda p: pd.read_excel(p),
            }
            self.df = readers[ext](path)
            logger.info(f"Source file: {path.name}  ({path.stat().st_size / 1024:.1f} KB)")

        elapsed = time.perf_counter() - start
        self._capture_metadata()
        logger.info(f"Extracted  {self.df.shape[0]:,} rows × {self.df.shape[1]} columns  [{elapsed:.3f}s]")
        return self.df

    def _capture_metadata(self):
        self.metadata = {
            "rows":        len(self.df),
            "columns":     list(self.df.columns),
            "dtypes":      self.df.dtypes.astype(str).to_dict(),
            "missing_pct": (self.df.isnull().mean() * 100).round(2).to_dict(),
            "extracted_at": datetime.now().isoformat(),
        }

    def summary(self):
        if self.df is None:
            raise RuntimeError("Call extract() first.")
        print("\n── Extraction Summary ──────────────────────────")
        print(f"  Shape   : {self.df.shape}")
        print(f"  Columns : {self.metadata['columns']}")
        print("\n  Missing values (%):")
        for col, pct in self.metadata["missing_pct"].items():
            bar = "█" * int(pct / 5)
            flag = " ⚠" if pct > 10 else ""
            print(f"    {col:<25} {pct:5.1f}%  {bar}{flag}")
        print()


# ═══════════════════════════════════════════════════════════
#  CUSTOM SKLEARN TRANSFORMERS
# ═══════════════════════════════════════════════════════════

class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip numerical outliers to [Q1 - k*IQR, Q3 + k*IQR]."""

    def __init__(self, k: float = 1.5):
        self.k = k

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for i, col in enumerate(X.columns):
            X[col] = X[col].clip(lower=self.lower_.iloc[i], upper=self.upper_.iloc[i])
        return X.values


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Expands datetime columns into year / month / day /
    day-of-week / hour integer features.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        result = []
        for col in (X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)).columns:
            s = pd.to_datetime(X[col], errors="coerce")
            result += [
                s.dt.year.rename(f"{col}_year"),
                s.dt.month.rename(f"{col}_month"),
                s.dt.day.rename(f"{col}_day"),
                s.dt.dayofweek.rename(f"{col}_dow"),
                s.dt.hour.rename(f"{col}_hour"),
            ]
        return pd.concat(result, axis=1).values


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drop columns with missing-value rate above a threshold."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.cols_to_drop_: list = []

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        self.cols_to_drop_ = df.columns[df.isnull().mean() > self.threshold].tolist()
        if self.cols_to_drop_:
            logger.info(f"  ColumnDropper will drop: {self.cols_to_drop_}")
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(X).copy()
        return df.drop(columns=self.cols_to_drop_, errors="ignore").values


# ═══════════════════════════════════════════════════════════
#  STAGE 2 — PREPROCESS & TRANSFORM
# ═══════════════════════════════════════════════════════════

class DataPreprocessor:
    """
    Builds a sklearn ColumnTransformer pipeline that handles:
      • Numeric columns  → impute → clip outliers → scale
      • Categorical cols → impute → one-hot encode
      • Date columns     → extract temporal features
    """

    def __init__(
        self,
        target_col: str | None = None,
        scale_method: str = "standard",   # "standard" | "minmax" | "robust"
        outlier_k: float = 1.5,
        missing_threshold: float = 0.5,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.target_col       = target_col
        self.scale_method     = scale_method
        self.outlier_k        = outlier_k
        self.missing_threshold = missing_threshold
        self.test_size        = test_size
        self.random_state     = random_state

        self.pipeline_: Pipeline | None = None
        self.feature_names_: list = []

    # ── column detection ──────────────────────────────────
    @staticmethod
    def _detect_columns(df: pd.DataFrame, target: str | None):
        """Auto-detect numeric, categorical, and datetime columns."""
        cols = [c for c in df.columns if c != target]
        num  = df[cols].select_dtypes(include=["number"]).columns.tolist()
        cat  = df[cols].select_dtypes(include=["object", "category"]).columns.tolist()
        dt   = df[cols].select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

        # Heuristic: string cols that look like dates
        for c in cat[:]:
            try:
                pd.to_datetime(df[c], errors="raise")
                dt.append(c)
                cat.remove(c)
            except Exception:
                pass

        return num, cat, dt

    # ── scaler factory ────────────────────────────────────
    def _scaler(self):
        return {
            "standard": StandardScaler(),
            "minmax":   MinMaxScaler(),
            "robust":   RobustScaler(),
        }.get(self.scale_method, StandardScaler())

    # ── build pipeline ────────────────────────────────────
    def build(self, df: pd.DataFrame):
        logger.info("━" * 55)
        logger.info("STAGE 2 — PREPROCESS & TRANSFORM")

        num_cols, cat_cols, dt_cols = self._detect_columns(df, self.target_col)
        logger.info(f"  Numeric     : {num_cols}")
        logger.info(f"  Categorical : {cat_cols}")
        logger.info(f"  Datetime    : {dt_cols}")

        transformers = []

        # Numeric pipeline
        if num_cols:
            num_pipe = Pipeline([
                ("impute",  SimpleImputer(strategy="median")),
                ("clip",    OutlierClipper(k=self.outlier_k)),
                ("scale",   self._scaler()),
            ])
            transformers.append(("numeric", num_pipe, num_cols))

        # Categorical pipeline
        if cat_cols:
            cat_pipe = Pipeline([
                ("impute",  SimpleImputer(strategy="most_frequent")),
                ("encode",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("categorical", cat_pipe, cat_cols))

        # Datetime pipeline
        if dt_cols:
            dt_pipe = Pipeline([
                ("extract", DateFeatureExtractor()),
                ("scale",   self._scaler()),
            ])
            transformers.append(("datetime", dt_pipe, dt_cols))

        if not transformers:
            raise ValueError("No transformable columns found in the DataFrame.")

        self.pipeline_ = Pipeline([
            ("transform", ColumnTransformer(transformers=transformers, remainder="drop"))
        ])

        logger.info(f"  Scale method: {self.scale_method}")
        return self

    # ── fit / transform ───────────────────────────────────
    def fit_transform_split(self, df: pd.DataFrame):
        if self.pipeline_ is None:
            self.build(df)

        # Separate features and target
        X = df.drop(columns=[self.target_col]) if self.target_col else df
        y = df[self.target_col]                if self.target_col else None

        start = time.perf_counter()
        X_transformed = self.pipeline_.fit_transform(X)
        elapsed = time.perf_counter() - start

        logger.info(f"  Transformed shape: {X_transformed.shape}  [{elapsed:.3f}s]")

        # Split
        if y is not None and self.test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X_transformed, y,
                test_size=self.test_size,
                random_state=self.random_state
            )
            logger.info(f"  Train / Test split: {len(X_train):,} / {len(X_test):,} rows")
            return X_train, X_test, y_train, y_test

        return X_transformed, None, y, None

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply fitted pipeline to new data (inference time)."""
        if self.pipeline_ is None:
            raise RuntimeError("Call fit_transform_split() first.")
        X = df.drop(columns=[self.target_col], errors="ignore") if self.target_col else df
        return self.pipeline_.transform(X)


# ═══════════════════════════════════════════════════════════
#  STAGE 3 — LOAD
# ═══════════════════════════════════════════════════════════

class DataLoader:
    """
    Saves processed data to disk in one or more formats.
    Supported: CSV, Parquet, Excel.
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        X_train: np.ndarray,
        X_test:  np.ndarray | None,
        y_train: pd.Series  | None,
        y_test:  pd.Series  | None,
        formats: list[str]  = ("csv",),
    ):
        logger.info("━" * 55)
        logger.info("STAGE 3 — LOAD")
        saved_files = []

        datasets = {"train": (X_train, y_train), "test": (X_test, y_test)}

        for split, (X, y) in datasets.items():
            if X is None:
                continue

            df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
            if y is not None:
                df.insert(0, "target", y.values)

            for fmt in formats:
                fname = self.output_dir / f"{split}.{fmt}"
                if fmt == "csv":
                    df.to_csv(fname, index=False)
                elif fmt == "parquet":
                    df.to_parquet(fname, index=False)
                elif fmt in ("xlsx", "excel"):
                    df.to_excel(fname, index=False)
                else:
                    logger.warning(f"  Unknown format '{fmt}' — skipped.")
                    continue
                saved_files.append(fname)
                logger.info(f"  Saved → {fname}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")

        logger.info(f"  Output directory: {self.output_dir.resolve()}")
        return saved_files


# ═══════════════════════════════════════════════════════════
#  ORCHESTRATOR  (the main ETL runner)
# ═══════════════════════════════════════════════════════════

class ETLPipeline:
    """
    High-level orchestrator that chains:
        DataExtractor → DataPreprocessor → DataLoader
    """

    def __init__(
        self,
        source,
        target_col:        str | None = None,
        scale_method:      str        = "standard",
        outlier_k:         float      = 1.5,
        missing_threshold: float      = 0.5,
        test_size:         float      = 0.2,
        random_state:      int        = 42,
        output_dir:        str        = "output",
        output_formats:    list[str]  = ("csv",),
    ):
        self.extractor    = DataExtractor(source)
        self.preprocessor = DataPreprocessor(
            target_col        = target_col,
            scale_method      = scale_method,
            outlier_k         = outlier_k,
            missing_threshold = missing_threshold,
            test_size         = test_size,
            random_state      = random_state,
        )
        self.loader = DataLoader(output_dir)
        self.output_formats = list(output_formats)
        self._results: dict = {}

    def run(self):
        wall_start = time.perf_counter()
        logger.info("╔" + "═" * 53 + "╗")
        logger.info("║         AUTOMATED ETL PIPELINE  STARTED          ║")
        logger.info("╚" + "═" * 53 + "╝")

        # ── Extract ───────────────────────────────────────
        df = self.extractor.extract()
        self.extractor.summary()

        # ── Preprocess / Transform ────────────────────────
        X_train, X_test, y_train, y_test = self.preprocessor.fit_transform_split(df)

        # ── Load ──────────────────────────────────────────
        saved = self.loader.load(
            X_train, X_test, y_train, y_test,
            formats=self.output_formats,
        )

        elapsed = time.perf_counter() - wall_start
        logger.info("━" * 55)
        logger.info(f"✔  PIPELINE COMPLETE in {elapsed:.3f}s")
        logger.info("━" * 55)

        self._results = {
            "X_train": X_train,
            "X_test":  X_test,
            "y_train": y_train,
            "y_test":  y_test,
            "saved_files":  saved,
            "pipeline":     self.preprocessor.pipeline_,
            "elapsed_sec":  round(elapsed, 3),
        }
        return self._results

    @property
    def results(self):
        return self._results


# ═══════════════════════════════════════════════════════════
#  DEMO  (runs automatically when executed directly)
# ═══════════════════════════════════════════════════════════

def _generate_demo_data(n: int = 1_000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic mixed dataset for demonstration."""
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "age":        rng.integers(18, 80, n).astype(float),
        "income":     rng.normal(50_000, 15_000, n),
        "score":      rng.uniform(0, 100, n),
        "tenure":     rng.integers(0, 40, n).astype(float),
        "city":       rng.choice(["Mumbai", "Delhi", "Pune", "Hyderabad", "Chennai"], n),
        "department": rng.choice(["Sales", "Tech", "HR", "Finance"], n),
        "join_date":  pd.date_range("2010-01-01", periods=n, freq="8h")[:n],
        "target":     rng.integers(0, 2, n),   # binary classification target
    })

    # Inject missingness
    for col, rate in [("age", 0.05), ("income", 0.08), ("city", 0.04)]:
        mask = rng.random(n) < rate
        df.loc[mask, col] = np.nan

    # Inject outliers
    df.loc[rng.choice(n, 10, replace=False), "income"] *= 10

    return df


if __name__ == "__main__":
    print("\n" + "═" * 57)
    print("  AUTOMATED ETL DATA PIPELINE  —  DEMO RUN")
    print("═" * 57)

    # ── Create synthetic data ─────────────────────────────
    demo_df = _generate_demo_data(n=2_000)
    print(f"\nGenerated synthetic dataset: {demo_df.shape}")
    print(demo_df.head(3).to_string())

    # ── Run the full pipeline ─────────────────────────────
    etl = ETLPipeline(
        source         = demo_df,
        target_col     = "target",
        scale_method   = "robust",       # robust to outliers
        outlier_k      = 1.5,
        test_size      = 0.2,
        random_state   = 42,
        output_dir     = "etl_output",
        output_formats = ["csv", "parquet"],
    )

    results = etl.run()

    # ── Summary ───────────────────────────────────────────
    print("\n── Final Results ───────────────────────────────")
    print(f"  X_train : {results['X_train'].shape}")
    if results['X_test'] is not None:
        print(f"  X_test  : {results['X_test'].shape}")
    print(f"  Files   : {[str(f) for f in results['saved_files']]}")
    print(f"  Runtime : {results['elapsed_sec']}s")
    print("\nTo use on your own data:")
    print("  etl = ETLPipeline('path/to/data.csv', target_col='your_target')")
    print("  results = etl.run()\n")
