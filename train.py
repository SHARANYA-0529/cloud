import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings

warnings.filterwarnings('ignore')


def create_sample_data():
    """Create sample Airbnb data for demonstration"""
    np.random.seed(42)
    n_samples = 1000

    # Generate realistic Airbnb features
    data = {
        'minimum_nights': np.random.randint(1, 30, n_samples),
        'number_of_reviews': np.random.poisson(15, n_samples),
        'availability_365': np.random.randint(0, 365, n_samples),
        'room_type': np.random.choice(['Entire home', 'Private room', 'Shared room'],
                                      n_samples, p=[0.6, 0.3, 0.1]),
        'neighbourhood_group': np.random.choice(['Manhattan', 'Brooklyn', 'Queens', 'Bronx'],
                                                 n_samples, p=[0.4, 0.3, 0.2, 0.1])
    }

    df = pd.DataFrame(data)

    # Create realistic price based on features
    base_price = 50

    # Room type impact
    room_type_multiplier = df['room_type'].map({
        'Entire home': 1.5,
        'Private room': 1.0,
        'Shared room': 0.7
    })

    # Neighbourhood impact
    neighbourhood_multiplier = df['neighbourhood_group'].map({
        'Manhattan': 2.0,
        'Brooklyn': 1.3,
        'Queens': 1.0,
        'Bronx': 0.8
    })

    # Calculate price with some randomness
    df['price'] = (base_price * room_type_multiplier * neighbourhood_multiplier *
                   (1 + df['number_of_reviews'] * 0.01) *
                   (1 + (365 - df['availability_365']) * 0.001) +
                   np.random.normal(0, 20, n_samples))

    # Ensure positive prices
    df['price'] = np.clip(df['price'], 30, 500)

    return df


def preprocess_data(df):
    """Preprocess the data for model training"""
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['room_type', 'neighbourhood_group'],
                                prefix=['room', 'neighbourhood'])

    # Ensure all boolean columns are int
    for col in df_encoded.columns:
        if df_encoded[col].dtype == bool:
            df_encoded[col] = df_encoded[col].astype(int)

    return df_encoded


def train_model():
    """Train the Airbnb price prediction model"""
    print("🏠 Creating sample Airbnb dataset...")
    df = create_sample_data()
    print(f"📊 Dataset created with {len(df)} listings")
    print(f"💰 Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")

    # Preprocess data
    df_processed = preprocess_data(df)

    # Separate features and target
    X = df_processed.drop('price', axis=1)
    y = df_processed['price']
    print(f"🎯 Features: {list(X.columns)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"📚 Training set: {len(X_train)} samples")
    print(f"🧪 Test set: {len(X_test)} samples")

    # Train model
    print("🤖 Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    print("\n📈 Model Performance:")
    print(f"Training MAE: ${train_mae:.2f}")
    print(f"Test MAE: ${test_mae:.2f}")
    print(f"Training R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n🔍 Top 5 Most Important Features:")
    print(feature_importance.head().to_string(index=False))

    # Save model and feature columns
    model_data = {
        'model': model,
        'feature_columns': list(X.columns),
        'performance': {
            'test_mae': test_mae,
            'test_r2': test_r2
        }
    }
    joblib.dump(model_data, 'model.pkl')
    print("\n✅ Model saved as model.pkl")
    return model_data


if __name__ == "__main__":
    train_model()