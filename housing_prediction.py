import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Put Housing.csv in the same folder as this script.
housing = pd.read_csv("Housing.csv")
housing.columns = housing.columns.str.strip().str.lower()

# Keep the columns needed for this project.
columns = ["area", "bedrooms", "price"]
housing = housing[columns].apply(pd.to_numeric, errors="coerce").dropna()

# Use the same training and test rows for both models.
X = housing[["area", "bedrooms"]]
y = housing["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "Area only": ["area"],
    "Area and bedrooms": ["area", "bedrooms"],
}

for name, features in models.items():
    model = LinearRegression()
    model.fit(X_train[features], y_train)
    predictions = model.predict(X_test[features])

    print(f"\n{name}")
    print(f"Test MSE: {mean_squared_error(y_test, predictions):,.2f}")
    print(f"Test R²: {r2_score(y_test, predictions):.3f}")
