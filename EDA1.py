import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


SALES_PATH = "sales.csv"
PROMO_FINAL_PATH = "promo_final.csv"
PROMO_SALES_PATH = "promo_sales.csv"

TOP_CLIENTS_N = 15
CAP_Q_FOR_BOXPLOT = 0.99


def must_have_cols(df: pd.DataFrame, cols: list, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{name}] Не найдены колонки: {missing}\n"
            f"Доступные колонки: {df.columns.tolist()}"
        )

def parse_date(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    return df

def to_num(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df

def show_df(title: str, df: pd.DataFrame, n: int = 10):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    display(df.head(n))

def show_series_stats(title: str, s: pd.Series):
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)
    display(s.describe())


sales = pd.read_csv(SALES_PATH)
promo_final = pd.read_csv(PROMO_FINAL_PATH)
promo_sales = pd.read_csv(PROMO_SALES_PATH)

must_have_cols(sales, ["Головное СКЮ Артикул", "УПП__Группа клиентов", "Дата", "Кол_шт"], "sales")
must_have_cols(promo_sales, ["Головное СКЮ Артикул", "УПП__Группа клиентов", "Дата", "Кол_шт"], "promo_sales")
must_have_cols(
    promo_final,
    ["Головное СКЮ Артикул", "УПП__Группа клиентов", "_промо план шт", "_промо %скидки", "Неделя", "Дата начала недели"],
    "promo_final"
)

# предобработка
sales = parse_date(sales, "Дата")
promo_sales = parse_date(promo_sales, "Дата")
promo_final["Дата начала недели"] = pd.to_datetime(promo_final["Дата начала недели"], dayfirst=True, errors="coerce")

sales = to_num(sales, "Кол_шт")
promo_sales = to_num(promo_sales, "Кол_шт")
promo_final = to_num(promo_final, "_промо план шт")
promo_final = to_num(promo_final, "_промо %скидки")
promo_final = to_num(promo_final, "Неделя")

# ключ месяца
sales["Месяц"] = sales["Дата"].dt.to_period("M").dt.to_timestamp()
promo_sales["Месяц"] = promo_sales["Дата"].dt.to_period("M").dt.to_timestamp()
promo_final["Месяц"] = promo_final["Дата начала недели"].dt.to_period("M").dt.to_timestamp()

# фильтр хвоста promo_final
before_rows = len(promo_final)
promo_final = promo_final[~((promo_final["_промо %скидки"] >= 100) & (promo_final["_промо план шт"] == 0))].copy()
after_rows = len(promo_final)
print(f"\nУдалено строк (хвост promo_final: скидка>=100% и план=0): {before_rows - after_rows}")

# метрики
total_sales_qty = float(sales["Кол_шт"].sum())
promo_sales_qty = float(promo_sales["Кол_шт"].sum())
promo_share_total = promo_sales_qty / total_sales_qty if total_sales_qty > 0 else np.nan


print("Метрики по объему продаж")
print(f"Общий объём продаж (sales), Кол-шт: {total_sales_qty:,.0f}")
print(f"Общий объём промо-продаж (promo_sales), Кол-шт: {promo_sales_qty:,.0f}")
print(f"Доля промо-продаж в общем объёме: {promo_share_total:.4%}")

# проверка периодов
sales_min_date = sales["Дата"].min()
sales_max_date = sales["Дата"].max()
promo_min_date = promo_sales["Дата"].min()
promo_max_date = promo_sales["Дата"].max()

print("проверка периодов")
print(f"Sales:       {sales_min_date.date()}  —  {sales_max_date.date()}")
print(f"Promo_sales: {promo_min_date.date()}  —  {promo_max_date.date()}")

common_start = max(sales_min_date, promo_min_date)
common_end = min(sales_max_date, promo_max_date)
print(f"Общий период (пересечение): {common_start.date()} — {common_end.date()}")

# PROMO PLAN: НЕДЕЛИ => МЕСЯЦ
promo_plan_m = (
    promo_final
    .groupby(["Головное СКЮ Артикул", "УПП__Группа клиентов", "Месяц"], as_index=False)
    .agg(
        promo_plan_qty=("_промо план шт", "sum"),         # суммарный план за месяц
        promo_discount_mean=("_промо %скидки", "mean"),   # средняя скидка по неделям в месяце
        promo_discount_max=("_промо %скидки", "max"),     # максимальная скидка в месяце
        promo_weeks=("Неделя", "nunique")                 # сколько недель было промо в месяце
    )
)

# ОБЪЕДИНЕНИЕ В ПАНЕЛЬ
sales_m = (
    sales.groupby(["Головное СКЮ Артикул", "УПП__Группа клиентов", "Месяц"], as_index=False)
         .agg(total_qty=("Кол_шт", "sum"))
)

promo_fact_m = (
    promo_sales.groupby(["Головное СКЮ Артикул", "УПП__Группа клиентов", "Месяц"], as_index=False)
              .agg(promo_fact_qty=("Кол_шт", "sum"))
)

panel = (
    sales_m
    .merge(promo_plan_m, on=["Головное СКЮ Артикул", "УПП__Группа клиентов", "Месяц"], how="left")
    .merge(promo_fact_m, on=["Головное СКЮ Артикул", "УПП__Группа клиентов", "Месяц"], how="left")
)

for c in ["promo_plan_qty", "promo_discount_mean", "promo_discount_max", "promo_weeks", "promo_fact_qty"]:
    panel[c] = panel[c].fillna(0.0)

panel["promo_flag"] = (
    (panel["promo_discount_max"] > 0) |
    (panel["promo_plan_qty"] > 0) |
    (panel["promo_fact_qty"] > 0)
)

show_df("Панель SKU × Клиент × Месяц (пример)", panel, n=10)

# СТАТИСТИКА ПО НАБОРАМ
datasets_stats = pd.DataFrame([
    {
        "dataset": "sales",
        "rows": len(sales),
        "unique_sku": sales["Головное СКЮ Артикул"].nunique(),
        "unique_clients": sales["УПП__Группа клиентов"].nunique(),
        "min_date": sales["Дата"].min(),
        "max_date": sales["Дата"].max(),
        "qty_sum": sales["Кол_шт"].sum(),
    },
    {
        "dataset": "promo_final (filtered)",
        "rows": len(promo_final),
        "unique_sku": promo_final["Головное СКЮ Артикул"].nunique(),
        "unique_clients": promo_final["УПП__Группа клиентов"].nunique(),
        "min_date": promo_final["Дата начала недели"].min(),
        "max_date": promo_final["Дата начала недели"].max(),
        "qty_sum": promo_final["_промо план шт"].sum(),
    },
    {
        "dataset": "promo_sales",
        "rows": len(promo_sales),
        "unique_sku": promo_sales["Головное СКЮ Артикул"].nunique(),
        "unique_clients": promo_sales["УПП__Группа клиентов"].nunique(),
        "min_date": promo_sales["Дата"].min(),
        "max_date": promo_sales["Дата"].max(),
        "qty_sum": promo_sales["Кол_шт"].sum(),
    },
])
show_df("Базовая статистика по исходным датасетам", datasets_stats, n=10)


monthly = (
    panel.groupby("Месяц", as_index=False)
         .agg(
             total_qty=("total_qty", "sum"),
             promo_plan_qty=("promo_plan_qty", "sum"),
             promo_fact_qty=("promo_fact_qty", "sum"),
         )
)

monthly["promo_fact_share"] = np.where(
    monthly["total_qty"] > 0,
    monthly["promo_fact_qty"] / monthly["total_qty"],
    0.0
)

show_df("Месячная динамика (итоги по всем SKU и клиентам)", monthly, n=15)

# Статистика доли промо
show_series_stats("Статистика доли промо в общем объёме (promo_fact_share)", monthly["promo_fact_share"])

# График 1: общий объём
plt.figure()
plt.plot(monthly["Месяц"], monthly["total_qty"])
plt.title("Динамика общего объёма продаж (total_qty)")
plt.xlabel("Месяц")
plt.ylabel("Кол-шт")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# График 2: промо план vs факт
plt.figure()
plt.plot(monthly["Месяц"], monthly["promo_plan_qty"], label="План промо")
plt.plot(monthly["Месяц"], monthly["promo_fact_qty"], label="Факт промо")
plt.title("Динамика промо-продаж: план vs факт")
plt.xlabel("Месяц")
plt.ylabel("Кол-шт")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# График 3: доля промо
plt.figure()
plt.plot(monthly["Месяц"], monthly["promo_fact_share"])
plt.title("Доля промо-продаж в общем объёме (факт)")
plt.xlabel("Месяц")
plt.ylabel("Доля")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# СКИДКИ
discount_series = promo_plan_m.loc[promo_plan_m["promo_discount_max"] > 0, "promo_discount_max"]
show_series_stats("Скидка (max) в промо: describe()", discount_series)

# сколько наблюдений > 40% / > 50%
if len(discount_series) > 0:
    share_40 = (discount_series > 40).mean()
    share_50 = (discount_series > 50).mean()
    print(f"\nДоля SKU-клиент-месяц со скидкой > 40%: {share_40:.4f}")
    print(f"Доля SKU-клиент-месяц со скидкой > 50%: {share_50:.4f}")

plt.figure()
plt.hist(discount_series, bins=30)
plt.title("Распределение максимальной скидки в промо (SKU-клиент-месяц)")
plt.xlabel("Скидка, %")
plt.ylabel("Количество наблюдений")
plt.tight_layout()
plt.show()

# ПРОМО vs БЕЗ ПРОМО: таблица + boxplot

promo_qty = panel.loc[panel["promo_flag"], "total_qty"]
nonpromo_qty = panel.loc[~panel["promo_flag"], "total_qty"]

promo_vs_nonpromo = pd.DataFrame({
    "segment": ["Без промо", "Промо-месяцы"],
    "n": [len(nonpromo_qty), len(promo_qty)],
    "mean_total_qty": [nonpromo_qty.mean(), promo_qty.mean()],
    "median_total_qty": [nonpromo_qty.median(), promo_qty.median()],
    "p75_total_qty": [nonpromo_qty.quantile(0.75), promo_qty.quantile(0.75)],
})
show_df("Сравнение total_qty: промо vs без промо", promo_vs_nonpromo, n=10)

# ограничиваем хвост для визуализации
cap = panel["total_qty"].quantile(CAP_Q_FOR_BOXPLOT)
promo_qty_cap = promo_qty[promo_qty <= cap]
nonpromo_qty_cap = nonpromo_qty[nonpromo_qty <= cap]

print(f"\nДля boxplot ограничили хвост на {int(CAP_Q_FOR_BOXPLOT*100)}-м перцентиле: cap={cap:.0f}")

plt.figure()
plt.boxplot([nonpromo_qty_cap, promo_qty_cap], labels=["Без промо", "Промо-месяцы"])
plt.title(f"Boxplot total_qty (обрезано по {int(CAP_Q_FOR_BOXPLOT*100)}-му перцентилю)")
plt.ylabel("Кол-шт")
plt.tight_layout()
plt.show()

# альтернативно: boxplot в лог-шкале
plt.figure()
plt.boxplot([nonpromo_qty, promo_qty], labels=["Без промо", "Промо-месяцы"])
plt.yscale("log")
plt.title("Boxplot total_qty (лог-шкала, без обрезки)")
plt.ylabel("Кол-шт (log)")
plt.tight_layout()
plt.show()

# UPLIFT (промо/база) по корзинам скидки
baseline = (
    panel[~panel["promo_flag"]]
    .groupby(["Головное СКЮ Артикул", "УПП__Группа клиентов"], as_index=False)
    .agg(
        baseline_qty=("total_qty", "mean"),
        n_nonpromo=("total_qty", "size")
    )
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

# корзины скидки
bins = [-0.1, 0, 10, 15, 20, 25, 30, 40, 1000]
labels = ["0", "(0;10]", "(10;15]", "(15;20]", "(20;25]", "(25;30]", "(30;40]", "40+"]
promo_with_base["disc_bucket"] = pd.cut(promo_with_base["promo_discount_max"], bins=bins, labels=labels)

# observed=False чтобы убрать FutureWarning
uplift_by_disc = (
    promo_with_base[promo_with_base["uplift_ratio"].notna()]
    .groupby("disc_bucket", as_index=False, observed=False)
    .agg(
        n=("uplift_ratio", "size"),
        uplift_mean=("uplift_ratio", "mean"),
        uplift_median=("uplift_ratio", "median"),
        total_qty_mean=("total_qty", "mean"),
        promo_fact_mean=("promo_fact_qty", "mean"),
        plan_mean=("promo_plan_qty", "mean"),
    )
)

show_df("Uplift по корзинам скидки (промо/база)", uplift_by_disc, n=20)

plt.figure()
plt.bar(uplift_by_disc["disc_bucket"].astype(str), uplift_by_disc["uplift_median"])
plt.title("Медианный uplift (промо/база) по корзинам скидки")
plt.xlabel("Скидка, % (максимум в месяце)")
plt.ylabel("Uplift ratio (median)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# СКИДКА vs ФАКТ ПРОМО + корреляции
promo_disc = promo_with_base[promo_with_base["promo_discount_max"] > 0].copy()

plt.figure()
plt.scatter(promo_disc["promo_discount_max"], promo_disc["promo_fact_qty"], s=8)
plt.title("Скидка (%) vs факт промо-продаж (Кол-шт)")
plt.xlabel("Макс. скидка, %")
plt.ylabel("Факт промо, Кол-шт")
plt.tight_layout()
plt.show()

# Корреляции
corr_cols = ["promo_discount_max", "promo_fact_qty", "promo_plan_qty", "total_qty"]
corr_spearman = promo_disc[corr_cols].corr(method="spearman")
show_df("Корреляции Spearman (устойчивы к выбросам)", corr_spearman, n=20)

# АНАЛИЗ ПО КЛИЕНТАМ
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

show_df(f"Клиенты: объём и промо-активность (топ-{TOP_CLIENTS_N})", client_summary, n=TOP_CLIENTS_N)

plt.figure()
plt.bar(client_summary.head(TOP_CLIENTS_N)["УПП__Группа клиентов"],
        client_summary.head(TOP_CLIENTS_N)["total_qty"])
plt.title(f"Топ-{TOP_CLIENTS_N} клиентов по объёму продаж (Кол-шт)")
plt.xlabel("Клиент")
plt.ylabel("Кол-шт")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()



#КЛИЕНТЫ С САМОЙ СИЛЬНОЙ ЗАВИСИМОСТЬЮ СКИДКА ↑ => ПРОДАЖИ ↑


MIN_OBS = 300  # минимальное число наблюдений на клиента, чтобы вывод был устойчивее

df_dep = panel[(panel["promo_discount_max"] > 0)].copy()

# убираем строки, где факт промо отсутствует
df_dep["y"] = df_dep["promo_fact_qty"]
df_dep["x"] = df_dep["promo_discount_max"]

# Spearman корреляция по каждому клиенту
spearman_rows = []
for client, g in df_dep.groupby("УПП__Группа клиентов"):
    g = g[["x", "y"]].dropna()
    n = len(g)
    if n < MIN_OBS:
        continue

    # Spearman (ранговая корреляция)
    corr = g["x"].corr(g["y"], method="spearman")

    spearman_rows.append({
        "УПП__Группа клиентов": client,
        "n_obs": n,
        "spearman_corr(x=скидка, y=промо_факт)": corr,
        "avg_discount": g["x"].mean(),
        "avg_promo_fact_qty": g["y"].mean(),
        "median_promo_fact_qty": g["y"].median(),
    })

client_dep_spearman = pd.DataFrame(spearman_rows).sort_values(
    "spearman_corr(x=скидка, y=промо_факт)", ascending=False
)

show_df(
    f"ТОП клиентов: сильная положительная связь (Spearman), MIN_OBS={MIN_OBS}",
    client_dep_spearman,
    n=20
)

# 2) Линейный наклон: log1p(промо_факт) ~ скидка
# slope > 0 => с ростом скидки растут продажи
slope_rows = []
for client, g in df_dep.groupby("УПП__Группа клиентов"):
    g = g[["x", "y"]].dropna()
    n = len(g)
    if n < MIN_OBS:
        continue

    x = g["x"].values.astype(float)
    y = np.log1p(g["y"].values.astype(float))  # лог, чтобы хвосты меньше ломали оценку

    # простая линейная аппроксимация y = a*x + b
    # np.polyfit возвращает [a, b]
    a, b = np.polyfit(x, y, 1)

    slope_rows.append({
        "УПП__Группа клиентов": client,
        "n_obs": n,
        "slope_log1p(promo_fact)_per_1pct_discount": a,
        "avg_discount": g["x"].mean(),
        "avg_promo_fact_qty": g["y"].mean(),
    })

client_dep_slope = pd.DataFrame(slope_rows).sort_values(
    "slope_log1p(promo_fact)_per_1pct_discount", ascending=False
)

show_df(
    f"ТОП клиентов: максимальный наклон (log1p), MIN_OBS={MIN_OBS}",
    client_dep_slope,
    n=20
)

# Визуализация ТОП-10 по Spearman
top10 = client_dep_spearman.head(10).copy()

plt.figure()
plt.bar(top10["УПП__Группа клиентов"], top10["spearman_corr(x=скидка, y=промо_факт)"])
plt.title("ТОП-10 клиентов по силе зависимости: скидка ↑ → промо-продажи ↑ (Spearman)")
plt.xlabel("Клиент")
plt.ylabel("Spearman corr")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

print("\nПояснение интерпретации:")
print("- Spearman > 0: чем выше скидка, тем выше промо-продажи (тенденция).")
print("- Spearman ~ 0: явной зависимости нет.")
print("- Spearman < 0: скидка растёт, а продажи не растут/падают (возможна другая стратегия или шум).")
print("- slope (log1p) > 0: подтверждает рост продаж при росте скидки, но в лог-шкале (устойчивее к хвостам).")

# ПОИСК КЛИЕНТОВ, ГДЕ СКИДКА ↑ => ПРОДАЖИ ↑ (по total_qty в промо-месяцы)


MIN_OBS = 200          # минимум промо-наблюдений на клиента
MIN_DISCOUNT_STD = 2.0 # скидка должна реально варьироваться
TOP_K = 5              # сколько топ-клиентов показать и построить графики

# Берём только промо-месяцы со скидкой > 0
df_effect = panel[(panel["promo_discount_max"] > 0)].copy()

df_effect = df_effect[df_effect["total_qty"] > 0].copy()

rows = []
for client, g in df_effect.groupby("УПП__Группа клиентов"):
    n = len(g)
    if n < MIN_OBS:
        continue

    disc_std = g["promo_discount_max"].std()
    if pd.isna(disc_std) or disc_std < MIN_DISCOUNT_STD:
        # если скидка почти всегда одинаковая — зависимость не оценить
        continue

    # Spearman: устойчива к выбросам и нелинейности
    spearman = g["promo_discount_max"].corr(g["total_qty"], method="spearman")

    # Наклон: log1p(total_qty) ~ скидка (интерпретируемо)
    x = g["promo_discount_max"].astype(float).values
    y = np.log1p(g["total_qty"].astype(float).values)
    slope, intercept = np.polyfit(x, y, 1)

    total_sum = g["total_qty"].sum()
    promo_fact_sum = g["promo_fact_qty"].sum()
    promo_fact_share = (promo_fact_sum / total_sum) if total_sum > 0 else np.nan

    rows.append({
        "Клиент": client,
        "n_obs": n,
        "discount_std": disc_std,
        "spearman(discount,total_qty)": spearman,
        "slope_log1p(total_qty)_per_1pct": slope,
        "avg_discount": g["promo_discount_max"].mean(),
        "median_total_qty": g["total_qty"].median(),
        "promo_fact_share_in_total": promo_fact_share,
    })

client_discount_effect = pd.DataFrame(rows).sort_values(
    ["spearman(discount,total_qty)", "slope_log1p(total_qty)_per_1pct"],
    ascending=False
)

show_df("ТОП клиентов по зависимости: скидка ↑ → total_qty ↑ (промо-месяцы)", client_discount_effect, n=20)

# Берём ТОП-K клиентов
top_clients = client_discount_effect.head(TOP_K)["Клиент"].tolist()
print("\nТОП клиентов для графиков:", top_clients)

# ГРАФИКИ ДЛЯ ТОП КЛИЕНТОВ
bins = [-0.1, 10, 15, 20, 25, 30, 40, 1000]
labels = ["<=10", "10-15", "15-20", "20-25", "25-30", "30-40", "40+"]

for client in top_clients:
    g = df_effect[df_effect["УПП__Группа клиентов"] == client].copy()

    # scatter: скидка vs total_qty
    plt.figure()
    plt.scatter(g["promo_discount_max"], g["total_qty"], s=10)
    plt.title(f"{client}: Скидка (%) vs Все продажи total_qty (промо-месяцы)")
    plt.xlabel("Макс. скидка, %")
    plt.ylabel("Все продажи, Кол-шт (total_qty)")
    plt.tight_layout()
    plt.show()

    # тренд (log)
    plt.figure()
    x = g["promo_discount_max"].astype(float).values
    y = np.log1p(g["total_qty"].astype(float).values)
    plt.scatter(x, y, s=10)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    plt.plot(xs, slope * xs + intercept)
    plt.title(f"{client}: log1p(total_qty) ~ скидка, slope={slope:.4f}")
    plt.xlabel("Макс. скидка, %")
    plt.ylabel("log1p(total_qty)")
    plt.tight_layout()
    plt.show()

    # по корзинам скидок: median(total_qty)
    g["disc_bucket"] = pd.cut(g["promo_discount_max"], bins=bins, labels=labels)
    bucket = (g.groupby("disc_bucket", observed=False)
                .agg(n=("total_qty","size"),
                     median_total_qty=("total_qty","median"),
                     mean_total_qty=("total_qty","mean"))
                .reset_index())

    show_df(f"{client}: статистика продаж по корзинам скидок", bucket, n=20)

    plt.figure()
    plt.bar(bucket["disc_bucket"].astype(str), bucket["median_total_qty"])
    plt.title(f"{client}: median(total_qty) по корзинам скидок")
    plt.xlabel("Скидка, %")
    plt.ylabel("median total_qty (Кол-шт)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Диагностика: доля promo_fact в total по клиенту
    total_sum = g["total_qty"].sum()
    promo_fact_sum = g["promo_fact_qty"].sum()
    promo_fact_share = (promo_fact_sum / total_sum) if total_sum > 0 else np.nan
    print(f"{client}: доля promo_fact_qty в total_qty = {promo_fact_share:.3%} (диагностика)")

# ROI-ПРОКСИ И ОПТИМАЛЬНАЯ СКИДКА

TARGET_CLIENTS = [
    "7Шагов",
    "Дикси",
    "АШАН",
    "ОКЕЙ",
    "Гиперглобус",
]

MIN_BASE_OBS = 6   # минимум месяцев без промо для baseline
MIN_PROMO_OBS = 8  # минимум промо-месяцев в корзине

# BASELINE: средние продажи без промо

baseline = (
    panel[~panel["promo_flag"]]
    .groupby(
        ["Головное СКЮ Артикул", "УПП__Группа клиентов"],
        as_index=False
    )
    .agg(
        baseline_qty=("total_qty", "mean"),
        n_nonpromo=("total_qty", "size")
    )
)

baseline = baseline[baseline["n_nonpromo"] >= MIN_BASE_OBS]

#  ПРОМО + UPLIFT
promo = panel[
    (panel["promo_flag"]) &
    (panel["promo_discount_max"] > 0)
].merge(
    baseline,
    on=["Головное СКЮ Артикул", "УПП__Группа клиентов"],
    how="left"
)

promo = promo[promo["baseline_qty"].notna()].copy()

# uplift (если отрицательный — считаем 0)
promo["uplift_qty"] = promo["total_qty"] - promo["baseline_qty"]
promo.loc[promo["uplift_qty"] < 0, "uplift_qty"] = 0.0

# ROI-ПРОКСИ

promo["discount_frac"] = promo["promo_discount_max"] / 100.0

promo["roi_proxy"] = np.where(
    (promo["baseline_qty"] > 0) & (promo["discount_frac"] > 0),
    promo["uplift_qty"] / (promo["baseline_qty"] * promo["discount_frac"]),
    np.nan
)

# КОРЗИНЫ СКИДОК
bins = [-0.1, 10, 15, 20, 25, 30, 40, 1000]
labels = ["<=10", "10-15", "15-20", "20-25", "25-30", "30-40", "40+"]

promo["disc_bucket"] = pd.cut(
    promo["promo_discount_max"],
    bins=bins,
    labels=labels
)

# ROI ПО КОРЗИНАМ И КЛИЕНТАМ

roi_input = promo[promo["УПП__Группа клиентов"].isin(TARGET_CLIENTS)].copy()

#выбрасываем строки, которые не попали ни в одну корзину
roi_input = roi_input[roi_input["disc_bucket"].notna()].copy()

# убираем inf/NaN в roi_proxy
roi_input["roi_proxy"] = pd.to_numeric(roi_input["roi_proxy"], errors="coerce")
roi_input = roi_input.replace([np.inf, -np.inf], np.nan)
roi_input = roi_input[roi_input["roi_proxy"].notna()].copy()

roi_summary = (
    roi_input
    .groupby(["УПП__Группа клиентов", "disc_bucket"], as_index=True, observed=True)
    .agg(
        n=("roi_proxy", "size"),
        roi_median=("roi_proxy", "median"),
        roi_mean=("roi_proxy", "mean"),
        uplift_median=("uplift_qty", "median"),
        baseline_median=("baseline_qty", "median"),
    )
    .reset_index()
)

# фильтр корзин с малым числом наблюдений
roi_summary = roi_summary[roi_summary["n"] >= MIN_PROMO_OBS].copy()


#  ВЫВОД ТАБЛИЦ И ГРАФИКОВ
for client in TARGET_CLIENTS:
    df_c = roi_summary[roi_summary["УПП__Группа клиентов"] == client]

    if df_c.empty:
        print(f"\n{client}: недостаточно данных для оценки ROI")
        continue

    print(f"{client}: ROI-прокси по корзинам скидок")

    display(df_c.sort_values("roi_median", ascending=False))

    # График: median ROI
    plt.figure()
    plt.bar(df_c["disc_bucket"].astype(str), df_c["roi_median"])
    plt.title(f"{client}: median ROI-proxy по корзинам скидок")
    plt.xlabel("Скидка, %")
    plt.ylabel("ROI-proxy (median)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# ОПТИМАЛЬНАЯ СКИДКА
optimal_discount = (
    roi_summary
    .sort_values(
        ["УПП__Группа клиентов", "roi_median"],
        ascending=[True, False]
    )
    .groupby("УПП__Группа клиентов", as_index=False)
    .first()
)

print("ОПТИМАЛЬНАЯ СКИДКА ПО ROI (ПО КЛИЕНТАМ)")

display(optimal_discount)


# СЕЗОННОСТЬ И ТРЕНДЫ
monthly = monthly.sort_values("Месяц").copy()

# добавим признаки календаря
monthly["year"] = monthly["Месяц"].dt.year
monthly["month"] = monthly["Месяц"].dt.month
monthly["month_name"] = monthly["Месяц"].dt.strftime("%b")  # Jan, Feb, ...
month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

for col, title in [
    ("total_qty", "Продажи (total_qty): динамика и тренд"),
    ("promo_plan_qty", "Промо-план (promo_plan_qty): динамика и тренд"),
    ("promo_fact_qty", "Промо-факт (promo_fact_qty): динамика и тренд"),
]:
    tmp = monthly[["Месяц", col]].copy()
    tmp["ma_3"] = tmp[col].rolling(3, min_periods=1).mean()
    tmp["ma_12"] = tmp[col].rolling(12, min_periods=3).mean()

    plt.figure()
    plt.plot(tmp["Месяц"], tmp[col], label="факт")
    plt.plot(tmp["Месяц"], tmp["ma_3"], label="MA(3)")
    plt.plot(tmp["Месяц"], tmp["ma_12"], label="MA(12)")
    plt.title(title)
    plt.xlabel("Месяц")
    plt.ylabel("Кол-шт")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

season = (
    monthly.groupby("month_name", as_index=False)
    .agg(
        total_mean=("total_qty", "mean"),
        plan_mean=("promo_plan_qty", "mean"),
        fact_mean=("promo_fact_qty", "mean"),
        n=("total_qty", "size"),
    )
)

season["month_name"] = pd.Categorical(season["month_name"], categories=month_order, ordered=True)
season = season.sort_values("month_name")

def plot_season(ycol, title):
    plt.figure()
    plt.bar(season["month_name"].astype(str), season[ycol])
    plt.title(title)
    plt.xlabel("Месяц года")
    plt.ylabel("Среднее, Кол-шт")
    plt.tight_layout()
    plt.show()

plot_season("total_mean", "Сезонность продаж: средние total_qty по месяцам года")
plot_season("plan_mean", "Сезонность промо-плана: средние promo_plan_qty по месяцам года")
plot_season("fact_mean", "Сезонность промо-факта: средние promo_fact_qty по месяцам года")

print("СЕЗОННОСТЬ: таблица средних по месяцам года (чем сильнее 'волны', тем выше сезонность)")

display(season)

def season_strength(series_means: pd.Series) -> float:
    m = series_means.mean()
    s = series_means.std()
    return (s / m) if m != 0 else np.nan  # CV: чем выше, тем сильнее сезонность

total_cv = season_strength(season["total_mean"])
plan_cv = season_strength(season["plan_mean"])
fact_cv = season_strength(season["fact_mean"])

print("ОЦЕНКА СИЛЫ СЕЗОННОСТИ (CV = std/mean по сезонным средним)")

print(f"total_qty  CV: {total_cv:.3f}")
print(f"promo_plan CV: {plan_cv:.3f}")
print(f"promo_fact CV: {fact_cv:.3f}")

print("\nИнтерпретация CV (примерно):")
print("- < 0.10: сезонность слабая/почти нет")
print("- 0.10–0.25: сезонность заметная")
print("- > 0.25: сезонность сильная")

if "promo_fact_share" in monthly.columns:
    season_share = (
        monthly.groupby("month_name", as_index=False)
        .agg(share_mean=("promo_fact_share", "mean"))
    )
    season_share["month_name"] = pd.Categorical(season_share["month_name"], categories=month_order, ordered=True)
    season_share = season_share.sort_values("month_name")

    plt.figure()
    plt.bar(season_share["month_name"].astype(str), season_share["share_mean"])
    plt.title("Сезонность доли промо: средняя promo_fact_share по месяцам года")
    plt.xlabel("Месяц года")
    plt.ylabel("Доля промо")
    plt.tight_layout()
    plt.show()