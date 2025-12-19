🎯 AI-Powered Career Guidance & Personalized Learning Platform

This system helps users discover suitable career paths, get relevant course recommendations, and follow personalized learning roadmaps based on their interests, skills, and quiz responses.
It uses machine learning models like Random Forest, TF-IDF with Cosine Similarity, KNN, Decision Tree, and Random Forest pipelines to provide accurate and meaningful guidance.

🚀 Features

Predict suitable career paths

Recommend relevant learning courses

Generate personalized learning paths

Uses real-world Kaggle datasets + synthetic datasets

Structured ML workflow and evaluation

🧰 Tech Stack

Python

Scikit-learn, Pandas, NumPy

Random Forest, Decision Tree, KNN

TF-IDF + Cosine Similarity

Kaggle Datasets & Custom Data

📂 Project Structure
├── career_prediction.ipynb        # Career classification model
├── course_recommendation.ipynb    # Course recommender (TF-IDF + KNN)
├── quiz_learning_path.ipynb       # Quiz-based learning roadmap
├── datasets/                      # Real + synthetic datasets
├── models/                        # Trained ML models
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation

⚙️ How to Run Locally
1️⃣ Clone the Repository
git clone https://github.com/your-username/AI-Career-Guidance.git
cd AI-Career-Guidance

2️⃣ Create a Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Project

Open the notebooks in Jupyter Notebook / VS Code and execute the cells.

🧠 Model Overview

Random Forest → Career Prediction

TF-IDF + Cosine Similarity → Content-based Course Recommendation

KNN → Interest-based Recommendation

Decision Tree & Random Forest → Quiz Evaluation + Learning Path Generation

🔮 Future Improvements

Add roadmap support for more domains

Improve accuracy with larger datasets

Full web deployment

Real-time adaptive recommendations
