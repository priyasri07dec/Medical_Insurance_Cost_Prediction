# Medical_Insurance_Cost_Prediction

![App UI](images/web_pg_screenshot.png)

#### Project Overview
This project focuses on predicting individual medical insurance charges based on patient demographic and health related information using Machine Learning regression algorithms. The objective is to build multiple regression models, compare their performance, detect overfitting, and deploy the best performing model using streamlit.
The project includes:
* Data preprocessing
* Exploratory Data Analysis(EDA)
* Outlier handling
* Feature engineering
* Multiple regression model building
* Model evaluation
* Hyperparameter tuning
* Streamlit deployment

#### Problem Statement
Medical insurance companies often estimate insurance premiums based on several factors such as age, smoking habits, number of dependents and residential region.
The goal of this project is to predict medical insurance costs accurately using machine learning models and identify the best model based on predictive performance and generalization capability.

#### Dataset Information
The dataset contains the following features:
* age
* sex
* bmi
* children
* smoker
* region
* charges

#### Technologies Used
* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit

#### Machine Learning Workflow
##### 1. Data Loading & Exploration
* Imported dataset using Pandas
* Performed dataset inspection using:
  * .head()
  * .info()
  * .describe()
 
##### 2. Exploratory Data Analysis(EDA)
  Performed:
  * Distribution analysis
  * Correlation analysis
  * Outlier visualization
  * Skewness analysis
 
##### Key Insights
* Insurance charges showed positive skewness.
* Smoking status significantly impacts insurance costs.
* Age and BMI showed positive correlation with charges.
* Outliers were mainly observed in the charges column.I

##### 3. Missing Value & Outlier Treatment
* No missing values were found in the dataset.
* Outliers were detected using boxplots and treated using the IQR method.

##### 4. Feature Engineering & Preprocessing
###### Categorical Encoding
Applied:

pd.get_dummies(drop_first = True)

###### Feature Scaling
Applied:

StandardScaler()
For scaling numerical features.

#### Models Used
The following regression algorithms wese trained and evaluated:
* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* Support Vector Regressor (SVR)
* K-Nearest Neighbors Regressor (KNN)

#### Evaluation Metrics
The models were evaluated using:
* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* R² Score
* Adjusted R² Score

Overfitting was analyzed by comparing training and testing performance.

#### Final Model Selection
Selected Model : Linear Regression
* Reason for Selection
* Lowest RMSE
* Highest R² Score
* Good generalization capability
* No significant overfitting
* Easy interpretability
* Lightweight and efficient for deployment

#### Hyperparameter Tuning
Hyperparameter tuning techniques such as:

* GridSearchCV
* RandomizedSearchCV
can be further applied to ensemble models like:
* Random Forest
* Gradient Boosting
to optimize performance.

#### Streamlet Web Application

https://priya-srivastava-medicalinsurancecostprediction.streamlit.app/

A Streamlit application was developed to allow users to:
* Input patient details
* Predict insurance charges instantly

User Inputs:
* Age
* Gender
* BMI
* Number of Children
* Smoking Status
* Region

#### Conclusion
This project successfully developed a machine learning-based insurance cost prediction system capable of estimating medical insurance charges. Multiple regression models were evaluated, and Linear Regression was selected as the final model due to its strong performance, interpretability, and generalization capability.

The project demonstrates an end-to-end machine learning workflow from data preprocessing to deployment using Streamlit.


