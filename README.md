

# 💰 Predicting Loan Default Using Machine Learning 

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

![IMG_0074](https://github.com/PATRICK079/LOAN_DEFAULT_PREDICTION/assets/157173680/c2344f49-9cc3-472d-97ad-90cbcbc7b6b6)


# Introduction

Lending a loan to a person could be a very critical  decision for financial institutions given that  many persons who were loaned never got to pay back their loans, and this could be a real situation for most financial institutions   which could lead to liquidation. It is imperative for financial institutions to understand the behavior of their  customer before they take action to lend money to  them; having a clearer definition for the purposes of the requested loan is also a great factor to put into considerations.  I have harnessed the full capability of Machine Learning In this project to make this critical decision making process very seamless for financial institutions to make the best decision  as to if a customer deserves a loan or not. 

 Primarily, the objectives of this project are:
1. Build a classification model to predict clients who are likely to default on their loan or not
2. Give recommendations to financial instutions  on the important features to consider wheen a client is  applying for a loan.
3. Deploy the final classification model to API and streamlit web application 

# Data Dictionary 

  The dataset encompasses of over 14 inputs and a target with over 200+ observations(points). These include:
  
●   LoanID -->  for a unique observation

●   Age: The age of the loan applicant. Age can influence the likelihood of default, as it may correlate with financial stability and earning potential.

● Income: The annual income of the loan applicant. Higher income usually indicates a greater ability to repay loans.

● LoanAmount: The total amount of the loan requested by the applicant. Larger loans might be more difficult to repay, influencing default risk.

● CreditScore: A numerical expression based on the applicant’s credit history, representing their creditworthiness. Higher credit scores generally indicate lower default risk.

● MonthsEmployed: The number of months the applicant has been employed. Longer employment durations often indicate job stability, which can reduce default risk.

● NumCreditLines: The number of credit lines (e.g., credit cards, loans) the applicant has open. This can reflect the applicant's credit utilization and management skills.

● InterestRate: The interest rate charged on the loan. Higher interest rates increase the cost of borrowing, which might affect the applicant’s ability to repay.

● LoanTerm: The duration of the loan (usually in months or years). Longer loan terms can mean smaller monthly payments but can also lead to more interest paid over time.

● DTIRatio (Debt-to-Income Ratio): The ratio of the applicant’s total monthly debt payments to their gross monthly income. A higher DTI ratio suggests a higher debt burden,       potentially increasing default risk.

● Education: The educational level of the applicant (e.g., high school, bachelor's degree, etc.). Higher education levels can be associated with higher earning potential and job stability.

● EmploymentType: The type of employment (e.g., full-time, part-time, self-employed). Different employment types can indicate varying levels of income stability.

● MaritalStatus: The marital status of the applicant (e.g., single, married, divorced). Marital status can affect financial stability and responsibilities.

● HasMortgage: Indicates whether the applicant currently has a mortgage. Existing mortgages can increase the applicant’s debt obligations.

● HasDependents: Indicates whether the applicant has dependents (e.g., children). More dependents can increase financial responsibilities and expenses.

● LoanPurpose: The purpose of the loan (e.g., home purchase, debt consolidation, education). Different purposes can influence the risk associated with the loan.

● HasCoSigner: Indicates whether the applicant has a co-signer for the loan. A co-signer can reduce default risk by providing additional assurance of repayment.

● Default: The target variable indicating whether the applicant has defaulted on the loan (binary variable: 0 for no default, 1 for default).

# Tools used

I utilized the following tools and technologies for this project

●  Python

●  Flask API

●  Heroku

●  Git

●  Streamlit

 # Data Exploration

   I dove into the dataset by gaining insights into the various features and their distributions, i understood the relationships between different vairables and their potential 
   impact on loan default. 
  
  1. I started off by exploring with countplot on the  target to visualize if i have an imbalanced class or not given that  this is somewhat an issue when it comes to classificiation problem which possible solution could either be using a SAMPLING technique(OVER SAMPLING/ UNDERSAMPLING) or have a class_weight = 'balanced'   

  2. I also did a pairplot with some continous features ( numeric) with hue as my target to visualize if i would have  a clearer seperation between my default.

  3. I  did more exploration using the bar plot on some numerical features, Also had a data exploration on my categorical features visualizing ratio of default and no default and so many more data visualizations were done including correlation matrix and heatmap.

      kindly check this link for  the  complete data exploration https://github.com/PATRICK079/LOAN_DEFAULT_PREDICTION/blob/main/LOAN_DEFAULT_PREDICTION%20PROJECT.ipynb

  # Data Pre-processing

  I started the data preprocessing stage by 

 1. Finding missing value  using isna().sum()
    
 2. Making sure the right data type was asigned to each feature using info()

 3. i checked for duplicated values using duplicated().sum()

 4. Checked for outliers using  the box plot on the continous features  or using the descritive statistical summary 

 5. Train_test_split the dataset

 6. Encoded my categorical featurs using get_dummies both on train and test dataset. Dataset conpasses more of nominal categorical features. 

 7. I performed feature scaling using MinMax scaler. I fit only on Train and transform train and test.

 kindly check this link for  the  complete data proprocessing  https://github.com/PATRICK079/LOAN_DEFAULT_PREDICTION/blob/main/LOAN_DEFAULT_PREDICTION%20PROJECT.ipynb

# Model Development 

 To solve this project I built lots of  machine learning models looking for the best model possible for my predictions and afterwards made comparisons of  each model complexity and generalization.  The following models were utilized in this project 
  1. voting classifier,
  2.  Baggging classifier,
  3.  Adaboost classifier,
  4.  Random forest classifier,
  5.  Gradient boosting classifier,
  6.  XGboost classifier,
  7.  GaussianNB classifer,
  8.  BernoulliNB classifier,
  9.  MultinomialNB classifier,
  10. Multi-layer Perceptron classifier,
  11. Logit Regression,
  12. Tensorflow sequential,
  13.  DecisionTreeClassifier'

   Based on model complexity and generalization; logistic Regression seems to capture my attention having a better generalization for both classes. Sequential, MultinomialNB classifier,BernoulliNB classifier and GaussianNB classifer seem very good too for this project. However i saved and  deployed the LOGIT model due to its high interpretability and high recall that minimizes defaulters.

 kindly check this link for  the  complete model development and metric scores 
 
 https://github.com/PATRICK079/LOAN_DEFAULT_PREDICTION/blob/main/LOAN_DEFAULT_PREDICTION%20PROJECT.ipynb


##  Api and Streamlit Development













 
