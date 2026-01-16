# -*- coding: utf-8 -*-
"""
Анализ промо-акций и скидок по продажам рыбной продукции
Входные файлы:
  sales.csv        : все продажи (месячные)
  promo_final.csv  : план промо (недели)
  promo_sales.csv  : факт промо (месячные)

Выход:
  папка ./eda_outputs/
    графики .png
    таблицы .csv
    panel_sku_client_month.csv (витрина SKU-клиент-месяц)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


SALES_PATH = Path("sales.csv")
PROMO_FINAL_PATH = Path("promo_final.csv")
PROMO_SALES_PATH = Path("promo_sales.csv")

OUT_DIR = Path("eda_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_CLIENTS_N = 15

def must_have_cols(df: pd.DataFrame, cols: list, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Не найдены колонки: {missing}\nЕсть колонки: {df.columns.tolist()}")

def parse_date(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    return df

def to_num(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df

def save_plot(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

# load
sales = pd.read_csv(SALES_PATH)
promo_final = pd.read_csv(PROMO_FINAL_PATH)
promo_sales = pd.read_csv(PROMO_SALES_PATH)


must_have_cols(sales,
               ["Головное СКЮ Артикул", "УПП__Группа клиентов", "Дата", "Кол_шт"],
               "sales")
must_have_cols(promo_sales,
               ["Головное СКЮ Артикул", "УПП__Группа клиентов", "Дата", "Кол_шт"],
               "promo_sales")
must_have_cols(promo_final,
               ["Головное СКЮ Артикул", "УПП__Группа клиентов", "_промо план шт", "_промо %скидки", "Неделя", "Дата начала недели"],
               "promo_final")


# даты
sales = parse_date(sales, "Дата")
promo_sales = parse_date(promo_sales, "Дата")
promo_final["Дата начала недели"] = pd.to_datetime(promo_final["Дата начала недели"], dayfirst=True, errors="coerce")


sales = to_num(sales, "Кол_шт")
promo_sales = to_num(promo_sales, "Кол_шт")
promo_final = to_num(promo_final, "_промо план шт")
promo_final = to_num(promo_final, "_промо %скидки")
promo_final = to_num(promo_final, "Неделя")

# фильтр хвоста: скидка 100% и план=0 (исключаем из анализа)
promo_final = promo_final[~((promo_final["_промо %скидки"] >= 100) & (promo_final["_промо план шт"] == 0))].copy()

# ключ месяца
sales["Месяц"] = sales["Дата"].dt.to_period("M").dt.to_timestamp()
promo_sales["Месяц"] = promo_sales["Дата"].dt.to_period("M").dt.to_timestamp()
promo_final["Месяц"] = promo_final["Дата начала недели"].dt.to_period("M").dt.to_timestamp()

# все продажи
sales_m = (
    sales.groupby(["Головное СКЮ Артикул", "УПП__Группа клиентов", "Месяц"], as_index=False)
         .agg(total_qty=("Кол_шт", "sum"))
)

# факт промо
promo_fact_m = (
    promo_sales.groupby(["Головное СКЮ Артикул", "УПП__Группа клиентов", "Месяц"], as_index=False)
              .agg(promo_fact_qty=("Кол_шт", "sum"))
)

# план промо: неделя -> месяц
promo_plan_m = (
    promo_final.groupby(["Головное СКЮ Артикул", "УПП__Группа клиентов", "Месяц"], as_index=False)
              .agg(
                  promo_plan_qty=("_промо план шт", "sum"),
                  promo_discount_mean=("_промо %скидки", "mean"),
                  promo_discount_max=("_промо %скидки", "max"),
                  promo_weeks=("Неделя", "nunique"),
              )
)

# объединяем в панель
panel = (sales_m
         .merge(promo_plan_m, on=["Головное СКЮ Артикул", "УПП__Группа клиентов", "Месяц"], how="left")
         .merge(promo_fact_m, on=["Головное СКЮ Артикул", "УПП__Группа клиентов", "Месяц"], how="left"))

for c in ["promo_plan_qty", "promo_discount_mean", "promo_discount_max", "promo_weeks", "promo_fact_qty"]:
    panel[c] = panel[c].fillna(0.0)

panel["promo_flag"] = (
    (panel["promo_discount_max"] > 0) |
    (panel["promo_plan_qty"] > 0) |
    (panel["promo_fact_qty"] > 0)
)

# =========================
# 5) EDA TABLES
# =========================
datasets_stats = pd.DataFrame([
    {
        "dataset": "sales",
        "rows": len(sales),
        "unique_sku": sales["Головное СКЮ Артикул"].nunique(),
        "unique_clients": sales["УПП__Группа клиентов"].nunique(),
        "min_date": sales["Дата"].min(),
        "max_date": sales["Дата"].max(),
        "qty_sum": float(sales["Кол_шт"].sum())
    },
    {
        "dataset": "promo_final (filtered)",
        "rows": len(promo_final),
        "unique_sku": promo_final["Головное СКЮ Артикул"].nunique(),
        "unique_clients": promo_final["УПП__Группа клиентов"].nunique(),
        "min_date": promo_final["Дата начала недели"].min(),
        "max_date": promo_final["Дата начала недели"].max(),
        "qty_sum": float(promo_final["_промо план шт"].sum())
    },
    {
        "dataset": "promo_sales",
        "rows": len(promo_sales),
        "unique_sku": promo_sales["Головное СКЮ Артикул"].nunique(),
        "unique_clients": promo_sales["УПП__Группа клиентов"].nunique(),
        "min_date": promo_sales["Дата"].min(),
        "max_date": promo_sales["Дата"].max(),
        "qty_sum": float(promo_sales["Кол_шт"].sum())
    },
])

monthly = (
    panel.groupby("Месяц", as_index=False)
         .agg(
             total_qty=("total_qty", "sum"),
             promo_plan_qty=("promo_plan_qty", "sum"),
             promo_fact_qty=("promo_fact_qty", "sum"),
             promo_obs=("promo_flag", "sum"),
         )
)
monthly["promo_fact_share"] = np.where(
    monthly["total_qty"] > 0,
    monthly["promo_fact_qty"] / monthly["total_qty"],
    0.0
)

# сравнение промо vs нет
promo_qty = panel.loc[panel["promo_flag"], "total_qty"]
nonpromo_qty = panel.loc[~panel["promo_flag"], "total_qty"]
promo_vs_nonpromo = pd.DataFrame({
    "segment": ["Без промо", "Промо-месяцы"],
    "n": [len(nonpromo_qty), len(promo_qty)],
    "mean_total_qty": [float(nonpromo_qty.mean()), float(promo_qty.mean())],
    "median_total_qty": [float(nonpromo_qty.median()), float(promo_qty.median())],
    "p75_total_qty": [float(nonpromo_qty.quantile(0.75)), float(promo_qty.quantile(0.75))],
})

# распределение скидок
discount_series = promo_plan_m.loc[promo_plan_m["promo_discount_max"] > 0, "promo_discount_max"]

discount_desc = pd.DataFrame([{
    "count": float(discount_series.count()),
    "mean": float(discount_series.mean()) if len(discount_series) else 0.0,
    "std": float(discount_series.std()) if len(discount_series) else 0.0,
    "min": float(discount_series.min()) if len(discount_series) else 0.0,
    "p25": float(discount_series.quantile(0.25)) if len(discount_series) else 0.0,
    "median": float(discount_series.median()) if len(discount_series) else 0.0,
    "p75": float(discount_series.quantile(0.75)) if len(discount_series) else 0.0,
    "max": float(discount_series.max()) if len(discount_series) else 0.0,
}])

# uplift
baseline = (
    panel[~panel["promo_flag"]]
    .groupby(["Головное СКЮ Артикул", "УПП__Группа клиентов"], as_index=False)
    .agg(baseline_qty=("total_qty", "mean"),
         n_nonpromo=("total_qty", "size"))
)

promo_with_base = panel[panel["promo_flag"]].merge(
    baseline, on=["Головное СКЮ Артикул", "УПП__Группа клиентов"], how="left"
)
promo_with_base["baseline_qty"] = promo_with_base["baseline_qty"].fillna(0.0)
promo_with_base["uplift_ratio"] = np.where(
    promo_with_base["baseline_qty"] > 0,
    promo_with_base["total_qty"] / promo_with_base["baseline_qty"],
    np.nan
)

bins = [-0.1, 0, 10, 15, 20, 25, 30, 40, 1000]
labels = ["0", "(0;10]", "(10;15]", "(15;20]", "(20;25]", "(25;30]", "(30;40]", "40+"]
promo_with_base["disc_bucket"] = pd.cut(promo_with_base["promo_discount_max"], bins=bins, labels=labels)

uplift_by_disc = (
    promo_with_base[promo_with_base["uplift_ratio"].notna()]
    .groupby("disc_bucket", as_index=False)
    .agg(
        n=("uplift_ratio", "size"),
        uplift_mean=("uplift_ratio", "mean"),
        uplift_median=("uplift_ratio", "median"),
        total_qty_mean=("total_qty", "mean"),
        promo_fact_mean=("promo_fact_qty", "mean"),
        plan_mean=("promo_plan_qty", "mean"),
    )
)

# клиенты
client_summary = (
    panel.groupby("УПП__Группа клиентов", as_index=False)
         .agg(
             total_qty=("total_qty", "sum"),
             promo_months=("promo_flag", "sum"),
             months=("promo_flag", "size"),
             promo_fact_qty=("promo_fact_qty", "sum"),
             promo_plan_qty=("promo_plan_qty", "sum"),
             avg_discount_in_promo=("promo_discount_max",
                                   lambda x: x[x > 0].mean() if (x > 0).any() else 0.0),
         )
)
client_summary["promo_month_share"] = client_summary["promo_months"] / client_summary["months"]
client_summary = client_summary.sort_values("total_qty", ascending=False)

# Time series total vs promo plan/fact
plt.figure()
plt.plot(monthly["Месяц"], monthly["total_qty"], label="Все продажи, Кол-шт")
plt.plot(monthly["Месяц"], monthly["promo_fact_qty"], label="Факт промо, Кол-шт")
plt.plot(monthly["Месяц"], monthly["promo_plan_qty"], label="План промо, Кол-шт")
plt.title("Динамика продаж: общий объём vs промо (план/факт)")
plt.xlabel("Месяц")
plt.ylabel("Кол-шт")
plt.legend()
plt.xticks(rotation=45)
save_plot(OUT_DIR / "01_timeseries_total_vs_promo.png")

# Promo share
plt.figure()
plt.plot(monthly["Месяц"], monthly["promo_fact_share"])
plt.title("Доля промо-продаж в общем объёме (факт)")
plt.xlabel("Месяц")
plt.ylabel("Доля")
plt.xticks(rotation=45)
save_plot(OUT_DIR / "02_promo_share.png")

# Discount distribution
plt.figure()
plt.hist(discount_series.dropna(), bins=30)
plt.title("Распределение максимальной скидки в промо (SKU-клиент-месяц)")
plt.xlabel("Скидка, %")
plt.ylabel("Количество наблюдений")
save_plot(OUT_DIR / "03_discount_distribution.png")

# Boxplot promo vs nonpromo
plt.figure()
plt.boxplot([nonpromo_qty, promo_qty], labels=["Без промо", "Промо-месяцы"])
plt.title("Месячные продажи (total_qty): промо vs без промо")
plt.ylabel("Кол-шт")
save_plot(OUT_DIR / "04_box_total_qty_promo_vs_nonpromo.png")

# Uplift by discount bucket (median)
plt.figure()
plt.bar(uplift_by_disc["disc_bucket"].astype(str), uplift_by_disc["uplift_median"])
plt.title("Медианный uplift (промо/база) по корзинам скидки")
plt.xlabel("Скидка, % (макс. в месяце)")
plt.ylabel("Uplift ratio (median)")
plt.xticks(rotation=45, ha="right")
save_plot(OUT_DIR / "05_uplift_by_discount_bucket.png")

# Top clients
plt.figure()
plt.bar(client_summary.head(TOP_CLIENTS_N)["УПП__Группа клиентов"],
        client_summary.head(TOP_CLIENTS_N)["total_qty"])
plt.title(f"Топ-{TOP_CLIENTS_N} клиентов по объёму продаж (Кол-шт)")
plt.xlabel("Клиент")
plt.ylabel("Кол-шт")
plt.xticks(rotation=45, ha="right")
save_plot(OUT_DIR / "06_top_clients_total_qty.png")

datasets_stats.to_csv(OUT_DIR / "tables_01_datasets_stats.csv", index=False)
monthly.to_csv(OUT_DIR / "tables_02_monthly_series.csv", index=False)
promo_vs_nonpromo.to_csv(OUT_DIR / "tables_03_promo_vs_nonpromo_compare.csv", index=False)
discount_desc.to_csv(OUT_DIR / "tables_04_discount_stats.csv", index=False)
uplift_by_disc.to_csv(OUT_DIR / "tables_05_uplift_by_discount_bucket.csv", index=False)
client_summary.to_csv(OUT_DIR / "tables_06_client_summary.csv", index=False)
panel.to_csv(OUT_DIR / "panel_sku_client_month.csv", index=False)

print(f"Папка с результатами: {OUT_DIR.resolve()}")
print("\nГрафики (*.png):")
for p in sorted(OUT_DIR.glob("*.png")):
    print(" -", p.name)
print("\nТаблицы (*.csv):")
for p in sorted(OUT_DIR.glob("tables_*.csv")):
    print(" -", p.name)
print("\nПанель:")
print("panel_sku_client_month.csv")
