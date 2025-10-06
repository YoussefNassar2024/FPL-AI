import pandas as pd
import json
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ---------- STEP 1: Load Data ----------
bootstrap = json.load(open("data/bootstrap-static.json"))
elements = pd.DataFrame(bootstrap["elements"])
fixtures = pd.read_json("data/gw_fixtures.json")

# Optional: add player history
try:
    history = pd.read_json("data/player_history.json")
except:
    history = pd.DataFrame()

# ---------- STEP 2: Basic Feature Engineering ----------
# We’ll build simple features for testing
df = elements[[
    "id", "team", "element_type", "now_cost", "form",
    "total_points", "points_per_game", "minutes",
    "selected_by_percent", "value_form", "value_season"
]].copy()

# Convert types
for col in ["form", "points_per_game", "selected_by_percent", "value_form", "value_season"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df["now_cost"] = df["now_cost"] / 10  # convert to £m
df["target"] = df["total_points"]  # example label (we’ll predict total points)

# ---------- STEP 3: Split ----------
X = df.drop(columns=["id", "target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- STEP 4: Train model ----------
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_test, label=y_test)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "verbose": -1,
}

model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    num_boost_round=2000,
    early_stopping_rounds=50,
    verbose_eval=100
)

# ---------- STEP 5: Evaluate ----------
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
print(f"\n✅ RMSE: {rmse:.3f}")

# ---------- STEP 6: Show feature importance ----------
importance = pd.DataFrame({
    "feature": model.feature_name(),
    "importance": model.feature_importance()
}).sort_values("importance", ascending=False)
print("\nTop features:\n", importance)

# Optional: save model
model.save_model("fpl_test_model.txt")
print("\nModel saved as fpl_test_model.txt ✅")
