import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from tabulate import tabulate

# Load dataset
data = pd.read_csv('/content/drive/MyDrive/TMDB_movie_dataset_v11.csv')

# Convert DataFrame to a solid table format
print(tabulate(data.head(), headers='keys', tablefmt='grid'))
# Function to preprocess data
def preprocess_data(df):
    # Drop duplicate rows if any
    df.drop_duplicates(inplace=True)

    # Fill missing values
    df['title'].fillna('Unknown', inplace=True)
    df['release_date'].fillna('1900-01-01', inplace=True)  # Default date for missing values
    df['overview'].fillna('No overview available', inplace=True)
    df['tagline'].fillna('No tagline', inplace=True)
    df['genres'].fillna('[]', inplace=True)
    df['production_companies'].fillna('[]', inplace=True)
    df['production_countries'].fillna('[]', inplace=True)
    df['spoken_languages'].fillna('[]', inplace=True)
    df['keywords'].fillna('[]', inplace=True)
    df['homepage'].fillna('No homepage', inplace=True)
    df['imdb_id'].fillna('No ID', inplace=True)
    df['backdrop_path'].fillna('No image', inplace=True)
    df['poster_path'].fillna('No image', inplace=True)

    # Convert release_date to datetime format
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    # Convert categorical columns to category type
    categorical_cols = ['status', 'original_language', 'adult']
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    # Drop columns with excessive missing values
    threshold = 0.3  # Drop columns with more than 70% missing values
    df = df.dropna(thresh=int(threshold * len(df)), axis=1)

    return df

# Example usage
df = pd.read_csv('/content/drive/MyDrive/TMDB_movie_dataset_v11.csv')
df = preprocess_data(df)

# Save cleaned data
df.to_csv('/content/drive/MyDrive/TMDB_movie_dataset_cleaned.csv', index=False)

print("✅ Preprocessed dataset saved as 'TMDB_movie_dataset_cleaned.csv'")
print(df.info())
