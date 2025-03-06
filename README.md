🎬 Movie Box Office Revenue Prediction

📌 Project Overview

This project builds a regression model to predict the box office revenue of movies based on key factors such as genre, production company, and budget. The model helps estimate a movie’s financial performance before release.

📊 Skills Gained

Regression Modeling

Feature Engineering

Data Preprocessing

🛠 Tools & Technologies

Python (for data processing & modeling)

Scikit-learn (for regression analysis)

Matplotlib & Seaborn (for data visualization)

📁 Dataset

We use the TMDB Movie Dataset for training and evaluation.
🔗 Download Dataset

🚀 Model Performance

🎬 Movie Box Office Revenue Prediction Model 📊
------------------------------------------------
Model Performance:
- Mean Absolute Error (MAE): 0.34
- Root Mean Squared Error (RMSE): 0.91
- R-Squared (R2): 0.96

🔥 Key Influencing Factors:
      Feature  Coefficient
 prod_encoded     0.985554
   log_budget     0.015574
genre_encoded    -0.002644

🎯 Example Prediction

Input:

Enter movie genres (comma-separated): action,drama,comedy
Enter movie budget ($): 5000000
Enter production company: Walt Disney Pictures

Output:

🎥 Estimated Box Office Revenue: $12,015,360.88

📌 How to Run the Project

Clone the repository:

git clone https://github.com/your-username/Movie_Box_Office_Prediction.git
cd Movie_Box_Office_Prediction

Install dependencies:

pip install -r requirements.txt

Train the model:

python train_model.py

Predict box office revenue:

python predict.py

📝 Contribution

Feel free to open issues or submit pull requests if you have improvements!

📩 Contact

For any questions, reach out via GitHub Issues or email: your-email@example.com

