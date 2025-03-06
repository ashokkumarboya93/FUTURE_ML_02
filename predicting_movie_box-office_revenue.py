import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to preprocess data
def preprocess_data(df):
    """Cleans and prepares the dataset for training."""
    
    # Select relevant columns and remove missing values
    df = df[['genres', 'budget', 'revenue', 'production_companies']].dropna()

    # Ensure revenue is greater than zero to avoid log errors
    df = df[df['revenue'] > 0]
    
    # Convert budget and revenue to log scale
    df['log_budget'] = np.log1p(df['budget'])
    df['log_revenue'] = np.log1p(df['revenue'])

    # Encode genres using mean target encoding
    genre_means = df.groupby('genres')['log_revenue'].mean()
    
    # Handle multi-genre movies by taking the average encoding of all genres in a movie
    def encode_genres(genres):
        genre_list = genres.split(',')
        valid_genres = [genre_means[genre] for genre in genre_list if genre in genre_means]
        return np.mean(valid_genres) if valid_genres else genre_means.mean()
    
    df['genre_encoded'] = df['genres'].apply(encode_genres)
    
    # Encode production companies
    prod_means = df.groupby('production_companies')['log_revenue'].mean()
    df['prod_encoded'] = df['production_companies'].map(prod_means).fillna(prod_means.mean())

    # Drop unnecessary columns
    df.drop(columns=['genres', 'budget', 'revenue', 'production_companies'], inplace=True)

    return df, genre_means, prod_means

# Function to train and evaluate the model
def train_model(df):
    """Trains a linear regression model and evaluates performance."""
    
    df, genre_means, prod_means = preprocess_data(df)

    # Split data into features (X) and target variable (y)
    X = df.drop(columns=['log_revenue'])
    y = df['log_revenue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model & encodings
    joblib.dump(model, 'box_office_model.pkl')
    joblib.dump(genre_means, 'genre_means.pkl')
    joblib.dump(prod_means, 'prod_means.pkl')

    # Predictions
    y_pred = model.predict(X_test)

    # Model Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Feature importance
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)

    # Performance Report
    report = f"""
    🎬 Movie Box Office Revenue Prediction Model 📊
    ------------------------------------------------
    Model Performance:
    - Mean Absolute Error (MAE): {mae:.2f}
    - Root Mean Squared Error (RMSE): {rmse:.2f}
    - R-Squared (R2): {r2:.2f}

    🔥 Key Influencing Factors:
    {feature_importance.to_string(index=False)}
    """
    print(report)

    # Plot feature importance
    plt.figure(figsize=(10, 5))
    sns.barplot(x=feature_importance['Coefficient'], y=feature_importance['Feature'])
    plt.title('Feature Importance in Predicting Box Office Revenue')
    plt.show()

    return model

# Function to predict revenue for a new movie
def predict_revenue(genres, budget, production_company):
    """Predicts the box office revenue of a new movie."""

    # Load the trained model & encodings
    model = joblib.load('box_office_model.pkl')
    genre_means = joblib.load('genre_means.pkl')
    prod_means = joblib.load('prod_means.pkl')

    # Handle multi-genre input
    genre_list = genres.split(',')
    valid_genres = [genre_means[genre] for genre in genre_list if genre in genre_means]
    genre_encoded = np.mean(valid_genres) if valid_genres else genre_means.mean()

    # Encode production company
    prod_encoded = prod_means.get(production_company, prod_means.mean())

    # Convert budget to log scale
    log_budget = np.log1p(budget)

    # Prepare input data
    input_data = pd.DataFrame({'log_budget': [log_budget], 'genre_encoded': [genre_encoded], 'prod_encoded': [prod_encoded]})

    # Predict log revenue and convert back to normal scale
    log_revenue_pred = model.predict(input_data)[0]
    revenue_pred = np.expm1(log_revenue_pred)  # Convert log revenue back to actual revenue

    print(f"🎥 Estimated Box Office Revenue: ${revenue_pred:,.2f}")
    return revenue_pred

# Example usage
if __name__ == "__main__":
    # Load dataset
    df_path = "/content/drive/MyDrive/TMDB_movie_dataset_v11.csv"
    df = pd.read_csv(df_path)

    # Train the model
    train_model(df)

    # Predict revenue for a new movie
    genres = input("Enter movie genres (comma-separated): ")
    budget = float(input("Enter movie budget ($): "))
    production_company = input("Enter production company: ")
    predict_revenue(genres, budget, production_company)
