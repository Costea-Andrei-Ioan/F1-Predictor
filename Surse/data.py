import requests
import pandas as pd
from sklearn.model_selection import train_test_split


# Collect data form OpenF1 API
# These sessions include trainings, califications and the race for a larger volume of data
session_keys = [9158, 9159, 9160, 9161, 9162]
df_laps_list = []
df_stints_list = []

for key in session_keys:
    url_laps = f"https://api.openf1.org/v1/laps?session_key={key}"
    url_stints = f"https://api.openf1.org/v1/stints?session_key={key}"
    try:
        # Laps
        resp_laps = requests.get(url_laps)
        if resp_laps.status_code == 200 and len(resp_laps.json()) > 0:
            df_laps_list.append(pd.DataFrame(resp_laps.json()))
        
        # Stints
        resp_stints = requests.get(url_stints)
        if resp_stints.status_code == 200 and len(resp_stints.json()) > 0:
            df_stints_list.append(pd.DataFrame(resp_stints.json()))

    except Exception as e:
        print(f"error: {e}")

# Combine the lists
df_laps = pd.concat(df_laps_list, ignore_index=True)
df_stints = pd.concat(df_stints_list, ignore_index=True)

# Data filter, cleaning and completion
df_laps = df_laps.dropna(subset=["lap_duration", "lap_number"]).copy()
df_stints["lap_end"] = df_stints["lap_end"].fillna(999)

# Combine the 2 data sources
df_laps = df_laps.sort_values(by="lap_number")
df_stints = df_stints.sort_values(by="lap_start")

df = pd.merge_asof(
    df_laps,
    df_stints[["session_key", "driver_number", "lap_start", "stint_number", "compound"]],
    left_on="lap_number",
    right_on="lap_start",
    by=["session_key", "driver_number"],
    direction="backward"
)

# Dict for mapping car number to the name of the pilot
driver_map = {
    1: "VER", 11: "PER", 44: "HAM", 63: "RUS", 16: "LEC", 55: "SAI", 
    4: "NOR", 81: "PIA", 14: "ALO", 18: "STR", 10: "GAS", 31: "OCO",
    23: "ALB", 2: "SAR", 22: "TSU", 3: "RIC", 77: "BOT", 24: "ZHO",
    20: "MAG", 27: "HUL"
}

df["driver_name"] = df["driver_number"].map(driver_map)
df["driver_name"] = df["driver_name"].fillna("RESERVE")

# Selecting the cols
selected_cols = [
    "driver_number",
    "driver_name",
    "lap_number",
    "stint_number",
    "compound",
    "duration_sector_1",
    "duration_sector_2",
    "duration_sector_3",
    "i1_speed",
    "i2_speed",
    "lap_duration"
]

df_final = df[selected_cols].copy()

df_final = df_final.dropna(subset=["stint_number", "compound"]).copy()
df_final["stint_number"] = df_final["stint_number"].astype(int)

# Treating the eventual non-existing values
col_nulls = ["duration_sector_1", "duration_sector_2", "duration_sector_3", "i1_speed", "i2_speed"]
for col in col_nulls:
    df_final[col] = df_final[col].fillna(df_final[col].mean())

print(df_final.head())
print("size: ", df_final.shape)

# train_set + test_set
train_df, test_df = train_test_split(df_final, test_size=0.3, random_state=42)

print(f"train size: {len(train_df)}")
print(f"test size: {len(test_df)}")

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)
