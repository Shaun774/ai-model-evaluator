import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from google import genai

def run_pipeline():
    print("--- 1. Generating Realistic Non-Linear Dataset ---")
    np.random.seed(42)
    n_samples = 1000

    square_feet = np.random.randint(800, 4500, size=n_samples)
    bedrooms = np.random.choice([2, 3, 4, 5], size=n_samples, p=[0.1, 0.4, 0.4, 0.1])
    bathrooms = bedrooms * 0.5 + np.random.choice([0, 0.5], size=n_samples)

    # Injecting non-linear scaling and heavy market noise
    base_price = 40000
    price = (
        base_price 
        + (square_feet ** 1.15) * 35 
        + (bedrooms * 18000) 
        + (bathrooms * 12000) 
        + np.random.normal(0, 70000, size=n_samples)
    )

    df = pd.DataFrame({
        'square_feet': square_feet,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'price': price
    })

    print(f"Dataset successfully created with {n_samples} samples.")

    print("\n--- 2. Training Model & Calculating Metrics ---")
    X = df[['square_feet', 'bedrooms', 'bathrooms']]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    metrics_summary = f"""
Model Evaluation Metrics (Out-of-Sample Test Set):
- Mean Absolute Error (MAE): ${mae:,.2f}
- Root Mean Squared Error (RMSE): ${rmse:,.2f}
- R-squared (R² Score): {r2:.4f}
"""
    print(metrics_summary)

    print("\n--- 3. Querying Google Gemini API for Automated Expert Audit ---")
    client = genai.Client()

    prompt = f"""
Act as a Principal Data Scientist. Evaluate the following regression model performance metrics for a real-world housing price prediction task. 
Note that the underlying data features non-linear price growth and significant market noise. Provide a concise expert review on model fit, potential limitations of using a linear model on non-linear data, and strategic recommendations for improvement:
{metrics_summary}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    print("\n[Gemini Expert Evaluation Report]")
    print(response.text)

if __name__ == "__main__":
    run_pipeline()
