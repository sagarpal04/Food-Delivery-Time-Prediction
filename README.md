%%writefile README.md

# Food Delivery Time Prediction

![Delivery Time Prediction App](https://raw.githubusercontent.com/sagarpal04/Food-Delivery-Time-Prediction/main/screenshot.png) <!-- Replace with a screenshot of your app if you have one -->

## Project Description

This project aims to predict food delivery times based on various factors such as distance, weather, traffic, time of day, vehicle type, preparation time, and courier experience. A Random Forest Regressor model, optimized using RandomizedSearchCV, is used for accurate predictions. The solution includes data preprocessing, exploratory data analysis, model training, and deployment as a Streamlit web application.

## Live Application

You can access the live Streamlit application here: [Food Delivery Time Prediction](https://food-delivery-time-prediction-fast.streamlit.app/)

## Features

- **Data Preprocessing**: Handling of missing values and encoding of categorical features.
- **Exploratory Data Analysis (EDA)**: Visualizations to understand data distributions and relationships.
- **Model Training**: Implementation of Decision Tree and Random Forest Regressor models.
- **Hyperparameter Tuning**: Optimization of the Random Forest model using RandomizedSearchCV.
- **Model Evaluation**: Assessment of model performance using MAE, MSE, RMSE, and R2 score.
- **Streamlit Web Application**: An interactive user interface to get real-time delivery time predictions.

## Technologies Used

- Python
- Pandas (for data manipulation)
- NumPy (for numerical operations)
- Scikit-learn (for machine learning models and preprocessing)
- Matplotlib (for data visualization)
- Seaborn (for data visualization)
- Streamlit (for web application deployment)
- Pickle (for saving and loading models)

## Getting Started

To run this project locally, follow these steps:

### Prerequisites

Make sure you have Python installed (preferably Python 3.8+).

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/sagarpal04/Food-Delivery-Time-Prediction.git
    cd Food-Delivery-Time-Prediction
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

    *If you don't have a `requirements.txt` file, you can create one by running `pip freeze > requirements.txt` after installing all dependencies, or manually install them:* 
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn streamlit
    ```

### Running the Streamlit Application

1.  **Ensure you have the necessary model files:**
    Make sure `label_encoder.pkl` and `best_random_forest_model.pkl` are present in the root directory of your project. These files are generated after running the model training notebook.

2.  **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

    This command will open the Streamlit application in your default web browser at `http://localhost:8501`.
