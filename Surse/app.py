import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import gradio as io
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv("../train.csv")
df_test = pd.read_csv("../test.csv")

# Eliminate outliers
init_len = len(df_train)
df_train = df_train[df_train["lap_duration"] < 115]
df_test = df_test[df_test["lap_duration"] < 115]
print(f"{init_len - len(df_train)} eliminated from train data")

# Preprocessing
df_train["is_train"] = 1
df_test["is_train"] = 0
df_combined = pd.concat([df_train, df_test])

# "compound" col into num
df_combined = pd.get_dummies(df_combined, columns=["compound"], drop_first=True)

df_train = df_combined[df_combined["is_train"] == 1].drop(columns=["is_train"])
df_test = df_combined[df_combined["is_train"] == 0].drop(columns=["is_train"])

stress_col = ["driver_name", "driver_number", "lap_duration", "duration_sector_3"]

x_train = df_train.drop(columns=stress_col)
y_train = df_train["lap_duration"]

x_test = df_test.drop(columns=stress_col)
y_test = df_test["lap_duration"]

# Random Forest
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)

model_rf.fit(x_train, y_train)

# Predictions on test data
pred = model_rf.predict(x_test)

# Metrics
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)

print(f"R_squared (R2): {r2:.4f}")
print(f"RMSE: {rmse:.4f} secunde")

df_comp = pd.DataFrame({
    "Timp real (sec)": y_test.values[:5],
    "Timp Prezis (sec)": pred[:5],
    "Eroare absoluta (sec)": abs(y_test.values[:5] - pred[:5])
})

print(df_comp.round(3))


sns.set_theme(style="whitegrid")
plt.figure(figsize=(7, 7))

sns.scatterplot(x=y_test, y=pred, alpha=0.6, color="royalblue", label="Tururi reale vs. Predicții")

lim_min = min(y_test.min(), pred.min())
lim_max = max(y_test.max(), pred.max())
plt.plot([lim_min, lim_max], [lim_min, lim_max], color="crimson", linestyle="--", linewidth=2, label="Predicție Ideală (Eroare 0)")

plt.title("Analiza Performantei: Timpi Reali vs. Timpi Prezisi", fontsize=14, fontweight="bold")
plt.xlabel("Timp Real pe Tur (secunde)", fontsize=12)
plt.ylabel("Timp Prezis de Model (secunde)", fontsize=12)
plt.legend(loc="upper left")
plt.tight_layout()

plt.savefig("grafic_performanta_rf.png", dpi=300)
plt.show()


# Interfata grafica
col_ord = x_train.columns.tolist()
def predict_lap_time(lap_number, stint_number, sec1, sec2, i1_spd, i2_spd, tyre_type):
    input_data = {
        "lap_number": [int(lap_number)],
        "stint_number": [int(stint_number)],
        "duration_sector_1": [float(sec1)],
        "duration_sector_2": [float(sec2)],
        "i1_speed": [float(i1_spd)],
        "i2_speed": [float(i2_spd)],
        "compound_MEDIUM": [1 if tyre_type == "MEDIUM" else 0],
        "compound_SOFT": [1 if tyre_type == "SOFT" else 0]
    }

    df_input = pd.DataFrame(input_data)[col_ord]
    predicted_time = model_rf.predict(df_input)[0]
    
    return f"Timp estimat pe tur: {predicted_time:.3f} secunde"

interface = io.Interface(
    fn=predict_lap_time,
    inputs=[
        io.Number(label="Numar Tur", value=10),
        io.Number(label="Numar Stint", value=1),
        io.Slider(minimum=25.0, maximum=45.0, value=32.0, label="Sector 1 (sec)"),
        io.Slider(minimum=35.0, maximum=60.0, value=45.0, label="Sector 2 (sec)"),
        io.Number(label="Viteza i1 (km/h)", value=270),
        io.Number(label="Viteza i2 (km/h)", value=250),
        io.Dropdown(choices=["HARD", "MEDIUM", "SOFT"], value="SOFT", label="Tip Pneu")
    ],
    outputs=io.Textbox(label="Rezultat Predictie"),
    title="F1 Lap Time Prediction",
    description="Modificati valorile pentru a vedea cum se comporta modelul antrenat",
    theme="soft"
)

interface.launch(share=False)
