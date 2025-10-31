import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer

#DATA FRAME CREATION AND DATA BASE CLEANING

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Path to CSV files
data_archive = r"D:\Python Programs\Programs\F1 Prediction AI\archive"


# List all CSV files in the archive
all_files = os.listdir(data_archive)


# Read all CSV files into a dictionary
data_frame = {}
for file in all_files:
    sheet_name = os.path.splitext(file)[0]
    data_frame[sheet_name] = pd.read_csv(os.path.join(data_archive, file), na_values="\\N")


# Merge data frames: results + races + drivers + constructors
race_results = pd.merge(data_frame["results"], data_frame["races"], on="raceId", how="left")


race_results = pd.merge(race_results, data_frame["drivers"], on="driverId", how="left")


race_results = pd.merge(race_results, data_frame["constructors"], on="constructorId", how="left")


driver_standings = data_frame["driver_standings"][['raceId','driverId','points','position']].rename(
    columns={'points':'driverPoints','position':'driverRank'}
)

race_results = pd.merge(
    race_results, 
    driver_standings, 
    on=['raceId','driverId'], 
    how='left'
)


constructor_standings = data_frame["constructor_standings"][['raceId','constructorId','points','position']].rename(
    columns={'points':'constructorPoints','position':'constructorRank'}
)

race_results = pd.merge(
    race_results, 
    constructor_standings, 
    on=['raceId','constructorId'], 
    how='left'
)


# Columns to keep for ML
columns_to_keep = [
    'raceId',
    'year',
    'driverId',
    'constructorId',
    'grid',
    'laps',
    'fastestLapTime',
    'position',
    'driverPoints',
    'driverRank',
    'constructorPoints',
    'constructorRank',
    'forename',
    'surname'
]


race_results_clean = race_results[columns_to_keep].copy()


# Convert fastestLapTime from "M:SS.sss" string to total seconds
def lap_time_to_seconds(lap_time):
    if pd.isna(lap_time):
        return None
    minutes, seconds = lap_time.split(":")
    return float(minutes) * 60 + float(seconds)


race_results_clean["fastestLapTimeSeconds"] = race_results_clean["fastestLapTime"].apply(lap_time_to_seconds)


# Assign DNFs consistent worst position, lap time, and handle points/ranks
def assign_dnf(df):
    df = df.copy()
    
    # Handle finishers for position and lap time
    finishers = df[~df['position'].isna()]
    if not finishers.empty:
        worst_position = finishers['position'].max() + 1
        worst_lap_time = finishers['fastestLapTimeSeconds'].max() + 1  # +1 second for DNF
        last_driver_rank = finishers['driverRank'].max(skipna=True)
        last_constructor_rank = finishers['constructorRank'].max(skipna=True)
    else:
        worst_position = 1
        worst_lap_time = 1.0
        last_driver_rank = 0
        last_constructor_rank = 0

    # Assign DNFs for position and lap time
    dnf_mask = df['position'].isna()
    df.loc[dnf_mask, 'position'] = worst_position
    df.loc[dnf_mask, 'fastestLapTimeSeconds'] = worst_lap_time

    # Fill NaN points with 0
    df['driverPoints'] = df['driverPoints'].fillna(0)
    df['constructorPoints'] = df['constructorPoints'].fillna(0)

    # Assign ranks for NaNs as one after the last finisher
    df['driverRank'] = df['driverRank'].fillna(last_driver_rank + 1)
    df['constructorRank'] = df['constructorRank'].fillna(last_constructor_rank + 1)
    
    return df


# Apply per race
race_results_clean = race_results_clean.groupby('raceId', group_keys=False).apply(assign_dnf)


# Drop the original string column
race_results_clean = race_results_clean.drop(columns=["fastestLapTime"])





# Show top rows
print(race_results_clean.head(20))



#MODEL SETUP


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------


#varibales the model will use you predict Y
x=race_results_clean[["raceId", "driverId", "grid", "laps", "fastestLapTimeSeconds", "constructorId", "driverPoints", "driverRank", "constructorPoints", "constructorRank"]]

# One-hot encode driverId and constructorId
x = pd.get_dummies(x, columns=['driverId', 'constructorId'], prefix=['driver', 'constructor'])

#We are interested in predicting the final position
y=race_results_clean["position"]


def within_one_accuracy(y_true, y_pred):
    return ((abs(y_true - y_pred) <= 1).sum()) / len(y_true)

custom_scorer = make_scorer(within_one_accuracy, greater_is_better=True)



#Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)



#Initialize and train the Random Forest Regressor

# Random Forest
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

param_dist = {
    'n_estimators': [100, 300, 500, 700],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,  # 50 random combinations
    scoring=custom_scorer,  # uses your ±1 position metric
    cv=5,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# Fit search on training data
rf_search.fit(X_train, y_train)

# Best model
best_rf = rf_search.best_estimator_
print("Best params:", rf_search.best_params_)




#Make predictions on the test set
y_pred = best_rf.predict(X_test)




#Evaluate model performance


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
within_1 = np.mean(np.abs(y_test - y_pred) <= 1)

print(f"RMSE: {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")
print(f"Within ±1 position accuracy: {within_1:.2f} ({within_1*100:.1f}%)")