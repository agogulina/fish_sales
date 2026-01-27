import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PATH_SALES = "sales.csv"
PATH_PROMO_SALES = "promo_sales.csv"
PATH_PROMO_FINAL = "promo_final.csv"

COL_SKU = "Головное СКЮ Артикул"
COL_CLIENT = "УПП__Группа клиентов"
COL_DATE = "Дата"
COL_QTY = "Кол_шт"

COL_WEEK_START = "Дата начала недели"
COL_WEEK_NUM = "Неделя"
COL_PROMO_PLAN = "_промо план шт"
COL_PROMO_DISC = "_промо %скидки"

OPTIONAL_CAT_COLS = [
    "Головное СКЮ ТИП",
    "Категория",
    "Группа new",
    "Головное СКЮ Наименование",
]

RANDOM_STATE = 42


def print_section(title: str):
    print(title)


def to_month_start(dt_series: pd.Series) -> pd.Series:
    """перевод даты в первый день месяца"""
    dt = pd.to_datetime(dt_series, dayfirst=True, errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp()


def safe_numeric(s: pd.Series) -> pd.Series:
    """преобразование строк с запятыми/пробелами в числовые значения"""
    if s.dtype == "O":
        s = s.astype(str).str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# Build monthly panel
def build_panel_monthly(path_sales: str, path_promo_sales: str, path_promo_final: str) -> pd.DataFrame:
    sales = pd.read_csv(path_sales)
    need_sales = [COL_SKU, COL_CLIENT, COL_DATE, COL_QTY] + [c for c in OPTIONAL_CAT_COLS if c in sales.columns]
    sales = sales[need_sales].copy()

    sales[COL_DATE] = to_month_start(sales[COL_DATE])
    sales[COL_QTY] = safe_numeric(sales[COL_QTY]).fillna(0.0)

    agg_dict = {COL_QTY: "sum"}
    for c in OPTIONAL_CAT_COLS:
        if c in sales.columns:
            agg_dict[c] = lambda x: x.dropna().astype(str).mode().iloc[0] if len(x.dropna()) else np.nan

    sales_m = (sales
               .groupby([COL_SKU, COL_CLIENT, COL_DATE], as_index=False)
               .agg(agg_dict))
    sales_m = sales_m.rename(columns={COL_QTY: "total_qty", COL_DATE: "month"})

    # promo_sales
    promo_sales = pd.read_csv(path_promo_sales)
    need_ps = [COL_SKU, COL_CLIENT, COL_DATE, COL_QTY] + [c for c in OPTIONAL_CAT_COLS if c in promo_sales.columns]
    promo_sales = promo_sales[need_ps].copy()

    promo_sales[COL_DATE] = to_month_start(promo_sales[COL_DATE])
    promo_sales[COL_QTY] = safe_numeric(promo_sales[COL_QTY]).fillna(0.0)

    agg_dict_ps = {COL_QTY: "sum"}
    for c in OPTIONAL_CAT_COLS:
        if c in promo_sales.columns:
            agg_dict_ps[c] = lambda x: x.dropna().astype(str).mode().iloc[0] if len(x.dropna()) else np.nan

    promo_sales_m = (promo_sales
                     .groupby([COL_SKU, COL_CLIENT, COL_DATE], as_index=False)
                     .agg(agg_dict_ps))
    promo_sales_m = promo_sales_m.rename(columns={COL_QTY: "promo_fact_qty", COL_DATE: "month"})

    # promo_final
    promo_final = pd.read_csv(path_promo_final)
    need_pf = [COL_SKU, COL_CLIENT, COL_WEEK_START, COL_WEEK_NUM, COL_PROMO_PLAN, COL_PROMO_DISC]
    promo_final = promo_final[need_pf].copy()

    promo_final[COL_WEEK_START] = pd.to_datetime(promo_final[COL_WEEK_START], dayfirst=True, errors="coerce")
    promo_final["month"] = promo_final[COL_WEEK_START].dt.to_period("M").dt.to_timestamp()

    promo_final[COL_PROMO_PLAN] = safe_numeric(promo_final[COL_PROMO_PLAN]).fillna(0.0)
    promo_final[COL_PROMO_DISC] = safe_numeric(promo_final[COL_PROMO_DISC]).fillna(0.0)
    promo_final[COL_WEEK_NUM] = safe_numeric(promo_final[COL_WEEK_NUM])

    promo_plan_m = (
        promo_final
        .groupby([COL_SKU, COL_CLIENT, "month"], as_index=False)
        .agg(
            promo_plan_qty=(COL_PROMO_PLAN, "sum"),
            promo_discount_mean=(COL_PROMO_DISC, "mean"),
            promo_discount_max=(COL_PROMO_DISC, "max"),
            promo_weeks=(COL_WEEK_NUM, "nunique"),
        )
    )

    panel = sales_m.merge(promo_sales_m, on=[COL_SKU, COL_CLIENT, "month"], how="left", suffixes=("", "_ps"))
    panel = panel.merge(promo_plan_m, on=[COL_SKU, COL_CLIENT, "month"], how="left")

    # fill NaNs
    for c in ["promo_fact_qty", "promo_plan_qty", "promo_discount_mean", "promo_discount_max", "promo_weeks"]:
        if c in panel.columns:
            panel[c] = panel[c].fillna(0.0)

    # promo share
    panel["promo_fact_share"] = np.where(panel["total_qty"] > 0, panel["promo_fact_qty"] / panel["total_qty"], 0.0)

    # ensure types
    panel[COL_SKU] = panel[COL_SKU].astype(str)
    panel[COL_CLIENT] = panel[COL_CLIENT].astype(str)
    panel["month"] = pd.to_datetime(panel["month"])

    return panel


# создаем признаки
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["month"].dt.year.astype(int)
    out["month_num"] = out["month"].dt.month.astype(int)

    out["m_sin"] = np.sin(2 * np.pi * out["month_num"] / 12.0)
    out["m_cos"] = np.cos(2 * np.pi * out["month_num"] / 12.0)

    out = out.sort_values([COL_SKU, COL_CLIENT, "month"])
    grp = out.groupby([COL_SKU, COL_CLIENT], sort=False)

    out["lag_total_1"] = grp["total_qty"].shift(1)
    out["lag_total_3"] = grp["total_qty"].shift(3)
    out["lag_promo_fact_1"] = grp["promo_fact_qty"].shift(1)
    out["lag_disc_max_1"] = grp["promo_discount_max"].shift(1)
    out["lag_plan_1"] = grp["promo_plan_qty"].shift(1)

    lag_cols = ["lag_total_1", "lag_total_3", "lag_promo_fact_1", "lag_disc_max_1", "lag_plan_1"]
    out[lag_cols] = out[lag_cols].fillna(0.0)

    return out


# подготовка данных
def make_preprocessors(cat_features, num_features):
    # пропуски заменяются медианой
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    # заполняем пропуски самым частым значением
    # применяем One-Hot Encoding
    cat_pipe_sparse = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))  # sparse
    ])

    # Dense One-Hot Encoding
    try:
        ohe_dense = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe_dense = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe_dense = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe_dense)
    ])

    preprocess_sparse = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_features),
            ("cat", cat_pipe_sparse, cat_features),
        ],
        remainder="drop"
    )

    preprocess_dense = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_features),
            ("cat", cat_pipe_dense, cat_features),
        ],
        remainder="drop"
    )

    return preprocess_sparse, preprocess_dense


# Train/Test split

def time_split(df: pd.DataFrame, test_months: int = 6):
    months = np.sort(df["month"].unique())
    if len(months) <= test_months:
        raise ValueError(f"Not enough months for split: total={len(months)}, test={test_months}")
    test_set_months = months[-test_months:]
    train = df[~df["month"].isin(test_set_months)].copy()
    test = df[df["month"].isin(test_set_months)].copy()
    return train, test, test_set_months


# эксперимент
def run_experiment(panel_ml: pd.DataFrame, target: str = "total_qty", test_months: int = 6):
    # split
    train_df, test_df, test_month_list = time_split(panel_ml, test_months=test_months)

    print("Test months:", [str(pd.Timestamp(m).date()) for m in test_month_list])
    print("Train rows:", len(train_df), "Test rows:", len(test_df))

    base_cat = [COL_SKU, COL_CLIENT]
    for c in OPTIONAL_CAT_COLS:
        if c in panel_ml.columns:
            base_cat.append(c)

    num_features = [
        "promo_plan_qty", "promo_fact_qty",
        "promo_discount_mean", "promo_discount_max",
        "promo_weeks", "promo_fact_share",
        "year", "month_num", "m_sin", "m_cos",
        "lag_total_1", "lag_total_3", "lag_promo_fact_1", "lag_disc_max_1", "lag_plan_1"
    ]
    num_features = [c for c in num_features if c in panel_ml.columns]

    cat_features = [c for c in base_cat if c in panel_ml.columns]

    # Признаки для обучения
    X_train = train_df[cat_features + num_features].copy()
    y_train = safe_numeric(train_df[target]).fillna(0.0).values

    X_test = test_df[cat_features + num_features].copy()
    y_test = safe_numeric(test_df[target]).fillna(0.0).values

    preprocess_sparse, preprocess_dense = make_preprocessors(cat_features, num_features)

    # Модели, работающие с разреженными данными
    models_sparse_ok = {
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.2, random_state=RANDOM_STATE, max_iter=5000),
    }
    # Модели, требующие плотные данные
    models_need_dense = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, max_depth=None
        ),
        "HistGB": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    results = []
    fitted_models = {}

    # sparse models
    for name, model in models_sparse_ok.items():
        pipe = Pipeline([("prep", preprocess_sparse), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        results.append({
            "model": name,
            "RMSE": rmse(y_test, pred),
            "MAE": float(mean_absolute_error(y_test, pred)),
            "R2": float(r2_score(y_test, pred)),
        })
        fitted_models[name] = pipe
        print(f"{name}: RMSE={results[-1]['RMSE']:.3f}  MAE={results[-1]['MAE']:.3f}  R2={results[-1]['R2']:.3f}")

    # dense models
    for name, model in models_need_dense.items():
        pipe = Pipeline([("prep", preprocess_dense), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        results.append({
            "model": name,
            "RMSE": rmse(y_test, pred),
            "MAE": float(mean_absolute_error(y_test, pred)),
            "R2": float(r2_score(y_test, pred)),
        })
        fitted_models[name] = pipe
        print(f"{name}: RMSE={results[-1]['RMSE']:.3f}  MAE={results[-1]['MAE']:.3f}  R2={results[-1]['R2']:.3f}")

    res_df = pd.DataFrame(results).sort_values("RMSE")
    return res_df, fitted_models, (train_df, test_df), (cat_features, num_features)


# Обратная задача подбор скидки для достижения +N% к продажам (пока тестирую)

def required_discount_for_uplift(
        model_pipe: Pipeline,
        row_features: pd.Series,
        uplift_pct: float = 10.0,
        discount_col: str = "promo_discount_max",
        disc_min: float = 0.0,
        disc_max: float = 60.0,
        step: float = 0.5,
):
    """
    Для одного SKU × клиента × месяца подбирается такое значение скидки,
    при котором прогноз продаж увеличивается на uplift_pct процентов.
    """
    # baseline prediction
    base_row = row_features.copy()
    base_pred = float(model_pipe.predict(pd.DataFrame([base_row]))[0])

    target_pred = base_pred * (1.0 + uplift_pct / 100.0)

    # grid search
    best_disc = None
    best_pred = None
    best_gap = np.inf

    disc_values = np.arange(disc_min, disc_max + 1e-9, step)
    for d in disc_values:
        test_row = row_features.copy()
        test_row[discount_col] = d
        pred = float(model_pipe.predict(pd.DataFrame([test_row]))[0])
        gap = abs(pred - target_pred)
        if gap < best_gap:
            best_gap = gap
            best_disc = d
            best_pred = pred

    return {
        "baseline_pred": base_pred,
        "target_pred": target_pred,
        "best_discount": best_disc,
        "pred_at_best_discount": best_pred,
        "abs_gap": best_gap,
    }


if __name__ == "__main__":
    panel = build_panel_monthly(PATH_SALES, PATH_PROMO_SALES, PATH_PROMO_FINAL)
    panel = panel.dropna(subset=["month"]).copy()
    print("panel shape:", panel.shape)
    print(panel.head(3))

    panel_ml = add_time_features(panel)
    print("panel_ml shape:", panel_ml.shape)
    print(panel_ml[[COL_SKU, COL_CLIENT, "month", "total_qty", "promo_plan_qty", "promo_discount_max"]].head(3))

    # Drop rows without target (optional)
    panel_ml["total_qty"] = safe_numeric(panel_ml["total_qty"]).fillna(0.0)

    print_section("3) Train/Test split + Models: прогноз total_qty")
    res_total, fitted_total, (train_df, test_df), (cat_f, num_f) = run_experiment(
        panel_ml, target="total_qty", test_months=6
    )
    print("\nLeaderboard (sorted by RMSE):")
    print(res_total)

    # pick best model
    best_model_name = res_total.iloc[0]["model"]
    best_pipe = fitted_total[best_model_name]
    print_section(f"Best model: {best_model_name}")

    print_section("какая скидка нужна для +N% к total_qty")

    demo = test_df.sort_values("total_qty", ascending=False).head(1).copy()
    demo_row = demo.iloc[0]

    feature_cols = cat_f + num_f
    row_features = demo_row[feature_cols].copy()

    uplift_pct = 10.0  # N%
    inv = required_discount_for_uplift(
        best_pipe,
        row_features=row_features,
        uplift_pct=uplift_pct,
        discount_col="promo_discount_max",
        disc_min=0.0,
        disc_max=60.0,
        step=0.5,
    )

    print("Demo entity:")
    print("SKU:", demo_row[COL_SKU], " Client:", demo_row[COL_CLIENT], " Month:", str(demo_row["month"].date()))
    print(f"Current discount_max: {float(demo_row['promo_discount_max']):.1f}%")
    print(f"Baseline predicted total_qty: {inv['baseline_pred']:.2f}")
    print(f"Target (+{uplift_pct:.1f}%): {inv['target_pred']:.2f}")
    print(f"Best discount to reach target (grid): {inv['best_discount']:.1f}%")
    print(f"Predicted at best discount: {inv['pred_at_best_discount']:.2f} (gap={inv['abs_gap']:.2f})")