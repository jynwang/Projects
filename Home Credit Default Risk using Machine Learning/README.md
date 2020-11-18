STA208_FinalProject
==============================
Home Credit Default Risk using Machine Learning
-----------------------------

## Members
Yahui Li(917797914)\
Jieyun Wang(9177814831)\
Siyao Wang(917856208)\
Ruichen Xu(917858772)

## Abstract

In this project, we use logistic regression, l2-penalized logistic regression and SVM to do binary classification on Home Credit default dataset. We use ROC curve and PR curve to evaluate the performance of difference models. When we optimize the logistic and Ridge regression function, the error need to be minimized, and we apply "Newton Sketch", which performs an approximate Newton step using random projected or subsampled Hessian. 

## Notebook

Main file with plots and conclusions.

## Data Source

The data we use is from this <a href="https://www.kaggle.com/c/home-credit-default-risk/data">Kaggle competition</a>. There are 8 data files in total, with an additional “HomeCredit_columns_description.csv” served as the description of columns in data files. But we will use 3 of them.
The main table we will be using is “application_train.csv”, which has the information about clients at the time of each application. The other two tables we will be using are “bureau.csv” and “previous_application.csv”, which could be joint with “application_train.csv” using “SK_ID_CURR”

## Data

Since we choose a wide range of sketch dimension(m), it takes time to run all of them each time. Thus, we save the result in csv files for later use. Here is a summary of all the data files:
<table>
<tr><td width="300px">File Name</td><td width="500px">Description</td></tr>
<tr><td width="300px">data_join.csv(not uploaded)</td><td width="500px">Data after joining three tables</td></tr>
<tr><td width="300px">data_clean.csv</td><td width="500px">Data after dropping the cases with NA in CB_MAX_AMT_OVERDUE and COMMONAREA_MEDI'</td></tr>
<tr><td width="300px">SN_None_*.csv</td><td width="500px">Stored results for Logistic Regression</td></tr>
<tr><td width="300px">SN_Ridge_*_*.csv</td><td width="500px">Stored results for using different lambda for l2-penalized Logistic Regression</td></tr>
</table>

## Code

The files in code folder are used for:
<table>
<tr><td width="300px">File Name</td><td width="500px">Description</td></tr>
<tr><td width="300px">data_munging_join.py</td><td width="500px">Joining three tables and store result in csv file</td></tr>
<tr><td width="300px">data_munging_clean.py</td><td width="500px">Dropping the cases with NA in CB_MAX_AMT_OVERDUE and COMMONAREA_MEDI' and store result in csv file</td></tr>
<tr><td width="300px">data_train_test.py</td><td width="500px">Train-test split, imputation and standardization</td></tr>
<tr><td width="300px">SketchedNewtonLogistic.py</td><td width="500px">Implement Newton Sketch method</td></tr>
<tr><td width="300px">ComparisonResult.py</td><td width="500px">Call Newton Sketch method and store result in csv file</td></tr>
</table>



