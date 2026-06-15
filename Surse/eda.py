import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")

def eda(df, subset_name):
    # a)
    print(f"Valori Lipsa - {subset_name}")
    nr_nulls = df.isnull().sum()
    percent_nulls = (df.isnull().sum() / len(df)) * 100
    df_nulls = pd.DataFrame({"Numar": nr_nulls, "Procent": percent_nulls})
    print(df_nulls)
    print("Strategie aplicata in data.py: media pentru si filtrare pentru categorie")

    # b)
    print(f"Statistici descriptive - Variabile numerice - {subset_name}")
    print(df.describe())

    print(f"Statistici descriptive - Variabile categorice - {subset_name}")
    print(df.describe(include=["object"]))

    # c)
    print(f"histograme si barplots pentru {subset_name}")
    # histogram
    cols_dist = ["duration_sector_1", "duration_sector_2", "duration_sector_3", "i1_speed", "i2_speed"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Distributia caracteristicilor numerice {subset_name}",fontsize=16)

    axes = axes.flatten()
    for i, col in enumerate(cols_dist):
        sns.histplot(df[col], kde=True, ax=axes[i], color="skyblue")
        axes[i].set_title(f"Distributie {col}")
    
    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig(f"distributie_numerice_{subset_name}.png")
    plt.close()

    # barplot
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x="compound", palette="Set1", order=df["compound"].value_counts().index)
    plt.title(f"Frecventa tipului de pneu - Compound {subset_name}")
    plt.xlabel("Tip Anvelopa")
    plt.ylabel("Numar Tururi")
    plt.tight_layout()
    plt.savefig(f"countplot_categoric_{subset_name}.png")
    plt.close()

    # d)
    print(f"Analiza outlieri - Regula IQR pentru lap_duration - {subset_name}")
    Q1 = df["lap_duration"].quantile(0.25)
    Q3 = df["lap_duration"].quantile(0.75)
    IQR = Q3 - Q1
    inf_lim = Q1 - 1.5 * IQR
    sup_lim = Q3 + 1.5 * IQR

    outlieri = df[(df["lap_duration"] < inf_lim) | (df["lap_duration"] > sup_lim)]
    print(f"Interval IQR valid: [{inf_lim:.3f}, {sup_lim:.3f} secunde]")
    print(f"Numarul de outlieri detectati in lap_duration: {len(outlieri)} dintr-un total de {len(df)} randuri")

    # boxplot
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df["lap_duration"], color="salmon")
    plt.title(f"Boxplot - Detectare outlieri pentru lap_duration {subset_name}")
    plt.xlabel("Timp Tur (sec)")
    plt.tight_layout()
    plt.savefig(f"boxplot_outliers_{subset_name}.png")
    plt.close()

    # e)
    print(f"Generare matrice de corelatie - {subset_name}")
    plt.figure(figsize=(9, 7))
    df_num = df[["lap_number", "stint_number", "duration_sector_1", "duration_sector_2", "duration_sector_3", "i1_speed", "i2_speed", "lap_duration"]]
    corr_mat = df_num.corr()
    sns.heatmap(corr_mat, annot=True, cmap="coolwarm", fmt=".2f", square=True, cbar_kws={"label": "Coeficient Corelatie"})
    plt.title(f"Matricea de corelatie {subset_name}")
    plt.tight_layout()
    plt.savefig(f"heatmap_corelatie_{subset_name}.png")
    plt.close()

    # f)
    print(f"Generare relatii cu variabila tinta - {subset_name}")
    plt.figure(figsize=(8, 5))
    df_viz = df[df["lap_duration"] < 110]
    sns.scatterplot(data=df_viz, x="duration_sector_2", y="lap_duration", hue="compound", alpha=0.7)
    plt.title(f"Relatia dintre timpul pe sectorul 2 si timpul pe tur {subset_name}")
    plt.xlabel("Durata sector 2 (sec)")
    plt.ylabel("Timp total (sec)")
    plt.legend(title='Tip pneu')
    plt.tight_layout()
    plt.savefig(f"scatter_target_{subset_name}.png")
    plt.close()

df_train = pd.read_csv("../train.csv")
df_test = pd.read_csv("../test.csv")

eda(df_train, "train")
eda(df_test, "test")

