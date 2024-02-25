## Bank-Loan-Approval-Prediction
This repository contains code for predicting bank loan approval status based on various features of the applicants. The dataset includes information about customers to whom a loan was granted or denied.

**Description of Dataset**
<br>The dataset consists of the following columns:

**Loan_ID:** Unique identifier for each loan application. 
**<br>Gender:** Gender of the applicant.
**<br>Married:** Marital status of the applicant.
**<br>Dependents:** Number of dependents of the applicant.
**<br>Education:** Education level of the applicant.
**<br>Self_Employed:** Whether the applicant is self-employed or not.
**<br>ApplicantIncome:** Income of the applicant.
**<br>CoapplicantIncome:** Income of the co-applicant.
**<br>LoanAmount:** Amount of the loan applied for.
**<br>Loan_Amount_Term:** Term of the loan in months.
**<br>Credit_History:** Credit history of the applicant.
**<br>Property_Area:** Area where the property is located.
**<br>Loan_Status:** Status of the loan application (1 for approved, 0 for denied).<br>
**<br>Data Preprocessing and Cleaning**
<br>The dataset underwent preprocessing and cleaning to handle missing values and ensure consistency in data types. Missing values were imputed, and categorical variables were appropriately handled.

**Exploratory Data Analysis (EDA)**
<br>EDA was performed to gain insights into the distribution and relationships among different variables. Visualizations such as bar plots and histograms were created to understand the distribution of loan status, education level, property area, gender, credit history, and other factors.

**Modeling**
<br>Two types of models were built:

**1. Logistic Regression Model:** Used **'glm'** function to build a logistic regression model to predict loan approval status. The model was trained on the training dataset and evaluated on the test dataset.<br>
**2. Decision Trees:** Employed the **'rpart'** package to construct a decision tree model for loan approval prediction. The decision tree was visualized to interpret the decision-making process.

**<br>Neural Network Model**
<br>A neural network model was implemented using the **'neuralnet'** package to predict loan approval status based on features such as coapplicant income, credit history, and applicant income. The model's accuracy was evaluated on the test dataset.

**Conclusion**
<br>This project demonstrates the process of predicting bank loan approval status using machine learning techniques. By analyzing various features of loan applicants, financial institutions can make informed decisions about granting or denying loans.
