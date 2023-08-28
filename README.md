# Credit-Risk-Analysis

Data 245 - Machine Learning Fall, 2022

Dhruv Jain, Edward Montoya, Nghi Nguyen, Tam Huynh

**Contents**

1. [**Abstract](#_page2_x72.00_y72.00) **[3**](#_page2_x72.00_y72.00)**
1. [**Introduction](#_page3_x72.00_y72.00) **[5**](#_page3_x72.00_y72.00)**
1. **Problem Statement 6**
1. [Project Background](#_page4_x72.00_y182.60) [6](#_page4_x72.00_y182.60)
1. [Literature Survey](#_page4_x72.00_y486.18) [6](#_page4_x72.00_y486.18)
4. [**Project Management - CRISP DM](#_page6_x72.00_y485.96) **[8**](#_page6_x72.00_y485.96)**
4. [**Exploratory Data Analysis](#_page8_x72.00_y541.16) **[10**](#_page8_x72.00_y541.16)**
1. [Data Exploration](#_page8_x72.00_y583.03) [10](#_page8_x72.00_y583.03)
1. [Data Cleaning](#_page15_x72.00_y463.57) [17](#_page15_x72.00_y463.57)
   1. [Handling Missing Values](#_page15_x72.00_y491.17) [17](#_page15_x72.00_y491.17)
   1. [Handling Messy Data](#_page16_x72.00_y436.66) [18](#_page16_x72.00_y436.66)
   1. [Changing Data Types](#_page17_x72.00_y379.24) [19](#_page17_x72.00_y379.24)
   1. [Feature Engineering](#_page18_x72.00_y72.00) [19](#_page18_x72.00_y72.00)
1. [Data Preprocessing](#_page18_x72.00_y631.71) [20](#_page18_x72.00_y631.71)
6. [**Model Selection](#_page20_x72.00_y350.73) **[22** ](#_page20_x72.00_y350.73)[Logistic Regression](#_page20_x72.00_y502.99)** [22 ](#_page20_x72.00_y502.99)[K Nearest Neighbors](#_page21_x72.00_y72.00) [22 ](#_page21_x72.00_y72.00)[Random Forest](#_page21_x72.00_y237.59) [22 ](#_page21_x72.00_y237.59)[XGBoost](#_page21_x72.00_y430.77) [23 ](#_page21_x72.00_y430.77)[Ensemble Voting Classifier](#_page21_x72.00_y623.95) [23](#_page21_x72.00_y623.95)
6. [**Model Development](#_page22_x72.00_y265.18) **[23**](#_page22_x72.00_y265.18)**
1. [Model Preparation](#_page22_x72.00_y307.05) [23](#_page22_x72.00_y307.05)
1. [Feature Selection](#_page24_x72.00_y630.32) [26](#_page24_x72.00_y630.32)
8. [**Model Evaluation](#_page26_x72.00_y99.60) **[27**](#_page26_x72.00_y99.60)**
   1. [Metrics (F1-macro, Accuracy, AUC, ROC curve)](#_page26_x72.00_y141.46) [27](#_page26_x72.00_y141.46)
   1. [Logistic Regression](#_page26_x72.00_y555.43) [28](#_page26_x72.00_y555.43)
   1. [KNN](#_page28_x72.00_y394.09) [30](#_page28_x72.00_y394.09)
   1. [Random Forest](#_page30_x72.00_y72.00) [31](#_page30_x72.00_y72.00)
   1. [XGBoost](#_page31_x72.00_y627.22) [33](#_page31_x72.00_y627.22)
   1. [Voting Classifier](#_page33_x72.00_y338.89) [34](#_page33_x72.00_y338.89)
   1. [Comparing All Models](#_page35_x72.00_y72.00) [36](#_page35_x72.00_y72.00)
   1. [Best Model Tuned](#_page35_x72.00_y548.12) [37](#_page35_x72.00_y548.12)
9. [**Deployment](#_page37_x72.00_y127.20) **[38**](#_page37_x72.00_y127.20)**
9. [**Conclusion](#_page37_x72.00_y334.65) **[39** ](#_page37_x72.00_y334.65)[References](#_page38_x72.00_y72.00) [4](#_page38_x72.00_y72.00)1**
1. **Abstract**

<a name="_page2_x72.00_y72.00"></a>Over the last few decades, financial management has been a huge issue for individuals

globally, especially when the global financial crisis occurred in 2008. Credit risk assessment is crucial in helping financial institutions define their policies and standards. Through appropriate ruling, only individuals with low credit risk are lent big loans, which minimizes the possibility of financial breakdown in the future. By applying machine learning, this project aims to classify people’s credit card status from input information on their personal information, including salary, spending habits, and loans. This paper examines four different machine learning algorithms: Logistic Regression, [K-nearest neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm), eXtreme Gradient Boosting, and Random Forest. This research bridges previous papers by applying an Ensemble Voting Classifier technique to the performance of the mentioned models to build a robust credit score classification classifier. The oversampling method Synthetic Minority Oversampling Technique (SMOTE) and the feature selection technique Recursive Feature Elimination, Cross-Validated (RFECV) are used to improve the efficiency of the predictions. A comparative analysis is performed between the naive models, models with the application of SMOTE, and models with the employment of SMOTE and RFECV. Random Forest model with sampling and feature selection techniques outperforms the other with 81.25% accuracy after performing hyper-parameter tuning with Random Search.

2. **Introduction**

<a name="_page3_x72.00_y72.00"></a>Credit is the ability of an individual to borrow money or goods or services and pay back

at a later time, and credit cards are one of the most popular forms of credit. Undeniably, credit plays an important role in pushing up the economy in the U.S. as people tend to spend more when they use credit cards than when they use cash. However, if financial organizations don’t take credit risk assessment seriously, the economy can bubble and explode. The target of the project is to find the best machine learning method for credit risk modeling using a basic bank details dataset that contains a vast array of credit-related information. The model classifies credit scores into three categories: 0 stands for poor, 1 stands for standard, and 2 stands for good. This will help financial institutions determine qualified candidates for any specific loans.

The objectives of the project are shown as follows:

- Conduct data preparation (e.g., data cleaning, data preprocessing) to clean and transform the dataset.
- Conduct data exploration to understand the dataset and observe any patterns or possible relationships among the variables.
- Implement a variety of machine learning algorithms, including Logistic Regression, K-Nearest Neighbor, Random Forest, XGBoosting, and Ensemble Voting Classifier.
- Implement all algorithms again with the synthetic minority oversampling technique (SMOTE) function. The SMOTE function will re-distribute the training and testing dataset with balanced portions of target categories.
- Perform feature selection technique Recursive Feature Elimination, Cross-Validated (RFECV) and implement all algorithms one last time
- Evaluate and compare different models which are:
- pure (no SMOTE function and feature selection)
- applied the SMOTE function
- applied the SMOTE function and the RFECV feature selection technique.
- Deploy the models
3. **Problem Statement**
1. **Project<a name="_page4_x72.00_y182.60"></a> Background**

In 2008, a financial crisis was triggered in the U.S. that was also known as the Great Recession. The cause of the crisis is believed to be the collapse of the subprime mortgage market, meaning individuals with below-average credit scores took out mortgages and did not have the ability to pay the loans afterward. The situation occurred due to poor risk assessment as financial organizations and institutions kept lending money to high-risk-credit individuals.

The dataset for this project consists of 100,000 records of credit-related information. With this dataset, we are able to find out the best machine learning algorithm for the credit risk modeling problem. Ensemble Voting Classifier and SMOTE function are some highlights of the project as no research papers we have read use the Ensemble Voting Classifier or combine the Ensemble Voting Classifier and SMOTE.

2. **Literature<a name="_page4_x72.00_y486.18"></a> Survey**

Gahlaut et al. (2017) propose data mining models including a Decision Tree, Random Forest, Neural Network, Support Vector Machine, Adaptive Boosting Model, and Linear Regression to determine customers’ ability to pay back their credit loans by classifying them as “Good credit” or “Bad credit.” Among the models, Random Forest returns the best result with high accuracy even when its running time is quite high. The authors use a dataset from the UCI machine learning data repository, which contains a variety of features, such as demographic characteristics (e.g., race, age, occupation), credit, rate, etc.

In the research by Machado and Karray (2022), hybrid algorithms are compared with the six individual algorithms, Adaboost, Gradient Boosting, Decision Tree, Random Forest, Support Vector Machine, and Artificial Neural Network in predicting commercial customers’ credit scores of a large bank in the U.S. Hybrid algorithms combine supervised and unsupervised machine learning models; for example, first, it uses k-Means to cluster the data then applies Adaboost to obtain the prediction. The result shows that hybrid models outperform individual models in terms of mean absolute error, explained variance, and mean squared error.

Laborda and Ryoo (2021) work on four different machine learning algorithms - Random Forest, K-nearest neighbor, Logistic Regression, and Support Vector Machine to find out the best model for the credit score classification problem. The authors offer three different feature selection methods - one filter method that uses the Chi-squared test and correlation analysis and two wrapper methods that use forward and backward stepwise selection. Their research paper shows that forward stepwise selection outperforms the other two methods, especially when it collaborates with the Random Forest model.

Moscato et al.(2021) focus on P2P credit risk modelings. P2P is a financial platform that allows people to lend their money to others without going through a bank, and the models try to predict if the borrowers have the ability to pay back the P2P loan. A benchmarking study is proposed and tested by combining various machine learning algorithms and sampling methodologies and performance comparison XAI tools. Algorithms applied include Logistic Regression, Random Forest, and Multi-layer perceptron, and sampling approaches consist of Random Under Sampling (RUS), IHT, Random Over Sampling (ROS), SMOTE, ADASYN, SMOTE-TOKEN, and SMOTE-EN. In conclusion, Logistics Regression is the best approach for over-sampling technology ROS, Logistic Regression is the best algorithm for hybrid sampling methods SMOTE-TOKEN, and Random Forest is the best for under-sampling methodology.

In the work of Singh (2017), twenty-five major classification techniques of individuals and ensembles are examined to determine the best approaches for the credit score modeling problem. Some of the algorithms are Multilayer perceptron, Logistic, Random Forest, Bagging, Artificial neural networks, and so forth. For individuals, Multilayer perceptron has the best performance in terms of AUC, and Random Forest is the runner-up with ROC 0.178. For ensembles, Random Forest and Bagging form the best combination with the highest accuracy score among all ensembles.

Trivedi (2020) implements five machine learning classifiers (Random Forest, Support Vector Machine, Decision Tree, Bayesian, and Naïve Bayes) and three feature selection methods (Gain-Ratio, Information Gain, and Chi-Square) to identify the best approach for the credit score classification model. In general, Random Forest is the best algorithm with 93% accuracy, and Chi-Square appears to be the most informative feature selection for all machine learning models. A disadvantage of Random Forest is the slightly high running time.

4. **Project<a name="_page6_x72.00_y485.96"></a> Management - CRISP DM**

The project management follows the CRoss Industry Standard Process for Data Mining,

also known as CRISP-DM. Figure 1 shows six phases of the process method.

**Figure 1**

*CRISP-DM Process WorkFlow*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 001](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/3c14cb86-faca-4438-89e1-964d7d420e81)


The first phase is Business Understanding. It is vital to have a credit card classification system to classify a credit card status. Classifying credit cards based on a customer’s income and spending habits, loans, and other conditions can help prevent economic issues. After understanding the business goal, literature surveys are being done to understand the potential model approaches that can be used to solve this problem.

Data Understanding first needs to collect the data based on the scope of credit risk         analysis. Data is first collected through the Kaggle dataset, then checked on source and quality. After that, appropriate exploratory data analysis was done to better understand the data, potential outliers, categorical and numerical variables, and the target features.

Next is Data Preparation. After understanding data types, values, and distribution, appropriate data cleaning and preprocessing have been considered for processing the data. Numerical normalization and categorical encoding are also applied for the possibility of model prediction and better performance. Furthermore, the preprocessed data is also split into training, testing, and validation sets. Finally, feature selection has been applied to get the best feature for the model.

The fourth step, which is Model development, is where data has been trained and tested on four different machine learning models. Ensemble voting classification, which is our unique technique, has also been applied to try to improve the performance of the project.

After the modeling phase comes the Evaluation. Based on different metrics, such as accuracy, precision, recall, and F1-score, each model is evaluated for accuracy and complexity performance to see which meets the business objectives the most. Hyper-parameter tuning is also being done by using Random Grid Search.

The last phase is deployment. After reviewing the whole project, the code and implementation are wrapped up and uploaded to GitHub for public access.

5. **Exploratory<a name="_page8_x72.00_y541.16"></a> Data Analysis**
1. **Data<a name="_page8_x72.00_y583.03"></a> Exploration**

The data, Credit score classification, was originally collected by Paris (2022) on Kaggle. Looking at the data information, the dataset consists of 100,000 rows and 28 columns. There are 20 categorical features and 8 numerical features. The target feature is ‘Credit\_Score,’ which has

an object type. There are eight columns with null values. These columns will be taken into consideration in the next steps. Figure 2 shows the overall description of the dataset. **Figure 2**

*Credit Score Data Types and Null Values Information*

Figure 3 shows a part of the data for sample inspection.

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 002](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/09aaecfe-d204-4105-8f5d-3084cd51bb55)

**Figure 3**

*Credit Score Data Sample*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 003](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/39ac995e-8073-4ca5-87de-2500142c3de3)


There are three values for the target Credit\_Score feature, which are ‘Poor’, ‘Standard,’ and ‘ Good.’ The distribution of the target is shown as a donut pie chart in figure 4 below. Clearly, there is an imbalance among the three types, with ‘Poor’ credit status accounting for 28998 rows, ‘Standard’ credit accounts for 53174 rows, and the rest is ‘Good’ credit with 17828 records. Being aware of this target imbalance, there will be an idea to apply target resampling techniques to solve this problem.

**Figure 4**

*Credit Score Target Feature Distribution*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 004](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/4b18e102-4977-4039-be05-23960e7c9d97)


Figure 5 shows how outliers can be detected through a boxplot for each class of the target feature and potentially the correlation between the target and the ‘Monthly\_Inhand\_Salary’ feature. From the plot, ‘Poor’ targets more outliers in the high salary range, then comes ‘Standard’ status. ‘Good’ credit status people do not have any outliers in a high salary. In this case, we can see the possible positive correlation between a high monthly salary and good credit status.

**Figure 5**

*Monthly Inhand Salary Boxplot by Credit Score Status*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 005](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/1e7417d5-c86d-438d-9449-48402d3dff20)


Figure 6 shows the number of credit cards each customer has for each credit card status. There are a lot of outliers going on in this particular feature, and the preprocessing step will take care of it. However, by looking at the distribution, ‘Good’ status credit cards tend to have fewer outliers than the other two classes.

**Figure 6**

*Number of Credit Cards Boxplot by Credit Score Status*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 006](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/003af86f-13fd-47fe-a098-ca1c0abb5251)


Figure 7 shows the distribution of the number of days delayed after the due date by each of the credit card class statuses. It is taken from the figure that ‘Good’ status people are less likely to delay their payment after 30 days. The better the credit status is the less number of days the payment is delayed.

**Figure 7**

*Number of Days Delayed From Due Date by Credit Status*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 007](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/a6668669-ec51-4495-9d05-2351fda9eabf)


Figure 8 shows the distribution of equated monthly installments by credit status in a strip plot. As inferred from the figure, ‘good’ credit card status has the least outliers, and ‘poor’ has fewer outliers than ‘standard’. It can be inferred that ‘good’ credit might have the least loan to pay every month, and ‘standard’ credit has the most.

**Figure 8**

*Equated Monthly Installment by Credit Status*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 008](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/60dca242-910b-472b-83fe-92b2e9c53ee1)


When comparing Figure 5, 6, 7, and 8, we can see that their range value differs from one another, with ‘Monthly\_Inhand\_Salary’ having the highest value of around 14,000 while ‘Delay\_from\_due\_date’ has the highest value of around 1,400, ‘Num\_Credit\_Card’ max value is around 70, and ‘Total\_EMI\_per\_month’ highest value is around 80,000. Inspecting three numerical features raises the idea that there is a need for normalizing the numerical variables so that different features can be weighted similarly without changing the differences in values within each feature.

After understanding some numerical features, we also try to understand the categorical features. Figure 9 shows some sample values for credit card history age. Notice that it is in years and months, which brings the idea to convert that into months values only, eventually converting to a numerical variable and normalizing it.

**Figure 9**

*Credit\_History\_Age Sample Values*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 009](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/3b3c2d54-0d71-4bea-b724-044fd882bbe5)


Similarly, when understanding the number of delayed payments for each customer, we see that the data integrity wasn’t being ensured, as shown in Figure 10. There are ‘\_’ characters for many values; therefore, we will consider cleaning them, converting them into numerical features, and normalizing them.

**Figure 10**

*Num\_of\_Delayed\_Payment Sample Values*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 010](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/d6093d80-8463-442a-89bf-4d1a2ea2d47f)

Figure 11 shows the unique values in the ‘Occupation’ feature. Notice that there are odd values that need to be cleaned, like the above variable; however, after cleaning, Occupation should still be kept as a categorical feature. In this case, an appropriate label encoding will be applied in the pre-processing step so that Occupation will be converted into machine-readable values for better performance results.

**Figure 11**

*Occupation Feature Values*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 011](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/979bfb98-3bd2-42a6-846f-0e0a71f23a19)


2. **Data<a name="_page15_x72.00_y463.57"></a> Cleaning**
1. **Handling<a name="_page15_x72.00_y491.17"></a> Missing Values**

The first step while cleaning data is to look for missing values. This dataset had quite a few missing values that had to be dealt with. Figure 12 shows the percentage of missing values in a particular feature.

**Figure 12**

*Percentage Of Missing Values*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 012](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/a85a8558-cb9e-4a1a-a6cf-ff83a15d3bed)

These missing values were handled in the following ways:

- Name - mode imputation based on Customer\_ID
- Monthly\_Inhand\_Salary - mode imputation based on Customer\_ID
- Num\_of\_Delayed\_Payments - mode imputation based on Customer\_ID
- Num\_Credit\_Inquiries - mode imputation based on Customer\_ID
- Credit\_History\_Age- mode imputation based on Customer\_ID
- Amount\_invested\_monthly- mode imputation based on Customer\_ID
- Monthly\_Balance- mode imputation based on Customer\_ID
2. **Handling<a name="_page16_x72.00_y436.66"></a> Messy Data**

The dataset also had data that was not needed in that particular field. These kinds of data create a noise in the model development phase and cannot be handled by the models very well. It is important to find such data and clean it so that it can be used by the machine learning model. We found these noisy data by going through the unique values of those fields. We found that many numerical features had ‘\_’ in the end the necessary data. We cleaned them by removing those discrepancies. SSN feature had values such as ‘#F%$D@\*&8’. Such values were replaced with the mode of that particular customer ID. A sample of these messy data can be seen in Figure 13.

**Figure 13**

*Sample Of Messy Data*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 013](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/68e58b62-d3b6-47f3-8bf7-501afb610144)


3. **Changing<a name="_page17_x72.00_y379.24"></a> Data Types**

Since there was noisy data in the numerical features, the datatypes of those features were not in the correct format. Once the noisy data was treated, the data types were changed to the correct type so that the machine understands the data correctly. Figure 14 shows the code that was used to correct these data types.

**Figure 14**

*Code Implementation To Change Data Types*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 014](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/6b3d016a-04a9-400e-a870-fed59dedfc3a)

4. **Feature<a name="_page18_x72.00_y72.00"></a> Engineering**

The Credit\_History\_Age feature had the values in string which made it difficult to gain insights from it. The sample values can be seen in Figure 15. A function was written to convert these into the number of months. The split function was used to divide the values of the string. The number of years and the number of months were assigned to temporary variables year and month. Then, the year was multiplied by 12 and added to the month. This gave us the total number of months and this was returned back to the field. Figure 16 shows how the field looks after this transformation.

**Figure 15**

*Credit\_History\_Age Before Transformation\*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 015](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/cac07342-cffa-444d-b076-26e021aef988)

**Figure 16**

*Credit\_History\_Age After Transformation*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 016](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/e5d12611-6de6-468b-942d-9ba03a60cbd8)


3. **Data<a name="_page18_x72.00_y631.71"></a> Preprocessing**

After exploring the data, there is a need to preprocess some numerical and categorical features so that the model can perform better. We first need to map our target feature, ‘Credit\_Score’, into numerical variables. ‘Poor’ credit card is mapped as 0, ‘Good’ as 1, and ‘Standard’ as 2. Figure 17 shows the target feature after encoding.

**Figure 17**

*Credit\_Score Target After Encoding*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 017](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/1d3f9d5b-6dc9-4583-8160-b0625fceef67)


Next, ‘Occupation’ is a nominal categorical variable, meaning there is no actual order for the values to follow under. Therefore, dummy encoding has been applied for this pre-processing step. After being encoded, each unique value of the Occupation feature will be transformed into a new column with binary values. For example, if a customer is a Doctor, the column Occupation\_Doctor will have a value of 1, and the rest will have values of 0. Figure 18 shows a new column after Occupation encoding with 15 more columns being added to the data frame. **Figure 18**

*Post Dummy Encoding for Occupation - New Columns*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 018](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/b37c999d-c026-46a5-91da-67560f4bb3b0)


Finally, for the numerical variables, MinMax Scaling will be applied to normalize different ranges among the features. Basically, MinMax Scaling will scale each of the declared variables into the range [0,1], with 0 being the minimum value of that feature and 1 being the maximum. Figure 19 shows part of the dataframe after applying normalization.

**Figure 19**

*Post Normalization For Numerical Variables*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 019](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/ecb08768-8705-44bb-bc7c-a001e4725ee5)


6. **Model<a name="_page20_x72.00_y350.73"></a> Selection**

There are a high number of machine learning algorithms available to solve all sorts of

problems. For this project we choose Logistic Regression (LR), K Nearest Neighbors (KNN), Random Forest (RF), XGBoost and Ensemble Voting Classifier. These algorithms were chosen based on the literature survey that was conducted.

<a name="_page20_x72.00_y502.99"></a>**Logistic Regression**

Logistic Regression is classification based supervised machine learning algorithm. Since it is an easy to implement and simple to use algorithm, it was used as the baseline model for our project. A baseline model is used to set a benchmark for the project to get an idea how well the preprocessing and transformation steps were done for the model development. It generally works well on binary classification problems but can be extended into multiclass classification as well. It uses the concept of sigmoid function to classify instances. A certain threshold is set, based on which the instances are classified into their classes.

<a name="_page21_x72.00_y72.00"></a>**K Nearest Neighbors**

Another simple yet efficient classification algorithm is KNN. The way a KNN works is that a training subset of pattern vectors from all the classes are provided together with a collection of sample prototypes for that class. The class label is chosen using a majority rule after finding the k closest neighbors of an unknown vector among all the prototype vectors. The value of k should be odd in order to prevent ties on class overlap regions (Laaksonen & Oja, 1996).

<a name="_page21_x72.00_y237.59"></a>**Random Forest**

Random Forest is an ensemble based machine learning algorithm that solves both classification and regression problems. The way a RF works is that multiple decision trees are created and are trained parallel to each other. Each tree gives out its own prediction and the final prediction is the majority output of all those trees. Since the prediction is based on majority, the performance is much better than an individual tree. This makes this model highly used in the world of machine learning.

<a name="_page21_x72.00_y430.77"></a>**XGBoost**

XGBoost is another ensemble machine learning algorithm which is based on gradient boosted trees. In recent years it has become highly popular. The major advantages of this algorithm are that it is a parallelized algorithm and also is optimized well. The way an XGBoost works is that it runs multiple trees one after the other. These trees are weak and hence called base learners. Collectively these trees make up a strong learner which gives out predictions.

XGBoost works very well on sparse and large datasets.

<a name="_page21_x72.00_y623.95"></a>**Ensemble Voting Classifier**

Ensemble Voting Classifier is nothing but a combination of two or more models           individually in an ensemble. With an increase in the volumes of data, the traditional algorithms cannot handle such volumes. Hence, voting classifiers come into the picture. It helps improve model performance by using the voting mechanism. The way it works is that different predictions will be obtained by each classifier. Using a voting mechanism where the majority will be from the winning class will yield the best prediction. The benefit of using ensembles of various classifiers is that no two of them will make the same error (Leon et al., 2017). There are a number of voting techniques such as soft vote and hard vote which can be used according to the use case.

7. **Model<a name="_page22_x72.00_y265.18"></a> Development**
1. **Model<a name="_page22_x72.00_y307.05"></a> Preparation**

The model development process begins by reading the cleaned preprocessed dataset into a pandas dataframe. Since the dataset is large at 100,000 rows, we decided to perform a train/validate/test split using Sklearn’s model selection library. This allowed the 100,000 rows to be split into 70,000 rows for training, 15,000 for validation, and 15,000 for testing. We considered doing K Fold cross-validation, but we felt that our dataset was large enough that we would be okay with using just the train test split. Typically Kfold is used for projects that do not have as much training data and need to use as much training data as possible. We believed that 70,000 rows were more than adequate for training our models. The figure below shows our data split.

**Figure 20**

*Train, Validate, and Test Diagram*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 020](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/87049a7b-f825-42bd-b5ae-fff4be8dbda3)


After completing the split, the training data was oversampled, which can be seen in figure

22\. Based on the literature review, we knew our project would need to balance the target classes. If not, we risk oversampling the majority class and inducing bias into our models (Zhu & Lin, 2017). For this reason, we applied the Synthetic Minority Oversampling Technique (SMOTE). Implementing SMOTE allowed us to balance the target classes evenly. The figure below shows our target classes before applying SMOTE.

**Figure 21**

*Target Classes Before SMOTE*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 021](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/d52afc7f-ddb0-489d-9fdb-8761e5c5b780)


The figure above clearly shows the imbalance of target classes. The distribution breaks down as 0-20,193 instances, 1-12,506 instances, and 2-37,301 instances. The application of SMOTE oversampled all of the non-majority classes to the same amount as the majority class. Meaning that all of the classes are synthetically oversampled to 37,301 instances each. The figure below shows the target class distribution after SMOTE is applied.

**Figure 22**

*Target Class After SMOTE*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 022](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/08ff72af-3e24-48e6-ad8b-f8aff356a2c0)


By applying SMOTE, all of the target classes are now synthetically the same amount. This means that the training dataset has grown from 70,000 rows to 111,903 rows. This should equate to better performance for the classifier on every target class.

2. **Feature<a name="_page24_x72.00_y630.32"></a> Selection**

When we first approached this project, we should have considered implementing a form of feature selection. Upon completing the initial process of validating the models, we identified that some of our models were overfitting. The extent of overfitting will be discussed in detail in section 8 of this paper. To remedy this problem, we decided to implement Recursive Feature Elimination Cross-Validation (RFECV). RFECV functions by removing features with low importance and re-fitting the model until it finds the least amount of features with the best performance (Kuhn & Johnson, 2018, p. 494). For our project, RFECV started with 50 features and determined that 10 features were the optimal amount for feature selection purposes. The figure below shows the plot for RFECV and how 10 features was the best performance based on the metric F1 with the minimal amount of features.

**Figure 23**

*RFECV Plot*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 023](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/44f02185-4ec8-48e8-92f6-909c7a71183a)


The result of the selected features with the highest importance can be seen in figure (). **Figure 24**

*View of Features Selected From RFECV*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 024](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/4e9cabed-f1c5-442c-94bb-7d98a1eec0e3)


8. **Model<a name="_page26_x72.00_y99.60"></a> Evaluation**
1. **Metrics<a name="_page26_x72.00_y141.46"></a> (*F1-macro, Accuracy, AUC, ROC curve)***

The metrics used to evaluate the models were accuracy, AUC score, ROC curve, and F1-macro. The primary metric was F1-macro. This is due to the fact that our project was an imbalanced multi-class classification problem, and this meant that using a metric like accuracy as the primary means of evaluating performance would be flawed. Accuracy would not be able to correctly identify if a single target class was being under-represented and the model was suffering from bias. Instead, the metric F1-macro calculated the average F1 score of all individual target classes (Géron, 2019, p. 160). This means that if a single target class is underperforming, then the F1-macro score will be sensitive enough to represent that issue by its scoring adequately.

Accuracy was mainly used in the initial phase of model building. When evaluating the performance of models on training data versus the validation data, accuracy allowed for the ability to quickly identify if the model was displaying symptoms of possibly overfitting.

Finally, the ROC curve was implemented as a means to compare how the model was performing visually.

2. **Logistic<a name="_page26_x72.00_y555.43"></a> Regression**

The first model implemented was Logistic Regression. Due to the simplicity of this model, it acted as our pseudo-baseline model. A model which we could compare against for a general idea of what a simplistic model would perform like. Table 1 below shows the initial performance of Logistic Regression on the training and validation data. The metric accuracy demonstrated that the model was not having issues with overfitting because there was not much difference between the two.

**Table 1**

*Logistic Regression Initial Model*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 025](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/6bec20e5-9a11-470e-b035-66bde74bb13c)


After the initial validation, the model had SMOTE, and SMOTE plus RFECV applied. The model benefited from applying SMOTE as the F1-macro score went up, which is our primary metric to focus upon. The addition of RFECV caused the model to perform worse, which makes sense because it was not having issues with overfitting. The best-performing Logistic Regression model was just the implementation of SMOTE. Table 2 below shows the performance of Logistic Regression.

**Table 2**

*Logistic Regression with SMOTE and SMOTE + RFECV*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 026](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/c1b5d330-713a-4002-a42e-ad4c0a4a9ae7)


The plot below depicts the ROC curve for best performing Logistic Regression, which was with SMOTE.

**Figure 25**

*Logistic Regression with SMOTE, ROC curve*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 027](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/5d19912d-7c63-40e9-b52d-be7903ef9eef)

3. **KNN**

<a name="_page28_x72.00_y394.09"></a>During the initial validation phase for KNN, we identified that the model was having issues with overfitting. The table below shows that the accuracy of the training data was significantly higher than the accuracy of the validation data. This meant that the implementation of RFECV would potentially help curtail this issue.

**Table 3**

*KNN Initial Model*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 028](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/fb1351b7-2e1a-4a81-98a0-fa67d9b1b72a)


KNN is known to suffer from dimensionality issues, which is the probable cause of its overfitting (Kelleher et al., 2015, p. 281). Table 4 below shows how the implementation of SMOTE helped slightly improve the performance of KNN, and the addition of RFECV feature selection dramatically improved its performance. RFECV mitigated the overfitting issue by training the model with just 10 features and allowed for the model to perform optimally.

**Table 4**

*KNN with SMOTE and SMOTE + RFECV*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 029](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/ecdb01d8-a201-49f6-854a-497f005c4833)


The plot below depicts the ROC curve for the best performing KNN, which was with SMOTE + RFECV.

**Figure 26**

*KNN with SMOTE + RFECV, ROC curve*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 030](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/326b0786-686d-4680-b0d9-c9958edb4c26)


4. **Random<a name="_page30_x72.00_y72.00"></a> Forest**

Random Forest is a machine learning technique that is known to not typically overfit due to its “extra randomness when growing trees” (Géron, 2019, p. 261). During our initial validation phase, we identified that our model was overfitting. Table 5 below shows the performance differences between training and validation data. Similar to standard Decision Trees, it is possible that the Random Forest model was growing without restrictions which means the trees grew too deep and overfit the training data (Géron, 2019, p. 246).

**Table 5**

*Random Forest Initial Model*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 031](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/f098c387-4abd-49b5-8985-9b6912ca2303)


Applying SMOTE and SMOTE + RFECV improved the model performance slightly. Out-of-the-box Random Forest is a very capable technique, and it is not surprising that the addition of SMOTE and RFECV offered minimal improvements. Perhaps, it is due to how Random Forest works. Random Forest operates by randomly sampling data, and features in an aggregated form, which is like a pseudo-feature selection (Géron, 2019, p. 262). Plus, the model does this sampling with replacement, which means that it can potentially train on more instances of an under-represented target class, which is similar to the benefits of SMOTE (Vanderplas & VanderPlas, 2016, p.444). Table 6 below shows the performance of Random Forest with SMOTE and SMOTE + RFECV.

**Table 6**

*Random Forest with SMOTE and SMOTE + RFECV*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 032](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/3fedbb71-b7f4-4dfa-a47d-62478fcc258b)


The best-performing model ended up being Random Forest with SMOTE + RFECV, but the performance gain over the initial model was minimal. The plot below depicts the ROC curve for Random Forest with SMOTE + RFECV.

**Figure 27**

*Random Forest with SMOTE + RFECV, ROC curve*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 033](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/c2329c82-9cab-402c-af3d-541825d4af55)


5. **XGBoost**

<a name="_page31_x72.00_y627.22"></a>The initial validation phase for XGBoost demonstrated that the model was not exhibiting overfitting issues. Table 7 below shows the model performance on training and validation data.

**Table 7**

*XGBoost Initial Model*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 034](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/bdfcb08b-43bb-4d76-a504-3f5dc552a3c9)


The implementation of SMOTE produced some measurable benefits for XGBoost, but RFECV feature selection reduced the model performance. As stated above, this is because the model was not having issues with overfitting. Table 8 below shows the performance of XGBoost with SMOTE and SMOTE + RFECV.

**Table 8**

*XGBoost with SMOTE and SMOTE + RFECV*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 035](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/1ba611af-b650-42e1-84c1-f87989f18ded)


The best-performing model for XGBoost was XGBoost with SMOTE. The ROC curve plot for this model can be seen in figure 28 below.

**Figure 28**

*XGBoost with SMOTE, ROC curve*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 036](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/063bc953-0e56-4e37-8852-e4c7778956aa)


6. **Voting<a name="_page33_x72.00_y338.89"></a> Classifier**

Finally, we wanted to implement a novel classifier that had yet to be done before for this particular multi-class classification problem. The ensemble Voting classifier we constructed consisted of our KNN, Random Forest, and XGboost models. This Voting classifier functioned by aggregating the predictions of all three models to make a classification (Géron, 2019, p. 254). For our model implementation, we decided to use soft voting, which yielded higher performance than hard voting. During the initial validation phase, the Voting classifier struggled with overfitting, which is expected because two of the three models that made up the Voting classifier were dealing with overfitting issues (KNN, Random Forest). This performance can be seen in the table below.

**Table 9**

*Voting Classifier Initial Model*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 037](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/f2e5efe6-b729-4bbe-9b1f-1bd4ae034f7d)


Implementing SMOTE and SMOTE + RFECV improved the overall performance of the Voting classifier. The SMOTE + RFECV model was the best-performing classifier out of all the models created in this report.

**Table 10**

*Voting Classifier with SMOTE and SMOTE + RFECV*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 038](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/dbbf4675-c9ad-4cf3-a0d5-50c743c309ea)


The plot below visualizes the ROC curve for the Voting classifier **Figure 29**

*Voting with SMOTE + RFECV, ROC curve*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 039](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/0735cce0-8a09-45a0-8ba1-46a3646073b1)


7. **Comparing<a name="_page35_x72.00_y72.00"></a> All Models**

All of the best-performing models from each machine learning technique were compared amongst one another using the validation data. The ensemble Voting classifier was the best overall model. For deployment, we decided to use the next best-performing model, Random Forest. We were concerned that the Voting classifier would be too computationally expensive for any real-world deployment, and that a single classifier would be more efficient. Plus, the performance difference between the Voting classifier and Random Forest were minimal. Table 11 below shows the performance metrics for all of the models evaluated on validation data.

**Table 11**

*Comparing All of the Models*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 040](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/ce3efb9d-3e4e-440a-9fb2-c2f15e495fec)


8. **Best<a name="_page35_x72.00_y548.12"></a> Model Tuned**

Random Forest SMOTE + RFECV was our best-performing single model on validation data that we wanted to hyper-parameter tune. We believed that with the tuning, we could gain some performance improvements. For this, we decided to implement a Randomized Search. This approach is computationally friendly compared to a traditional Grid Search (Géron, 2019, p. 420).

We set up the search parameters as shown in figure 30 below. **Figure 30**

*Randomized Search Parameters*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 041](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/846ae6d3-48f0-4a62-a075-c27a3f89dc95)


The result from the Randomized Search provided the following hyper-parameters as being the best parameters.

**Figure 31**

*Randomized Search Best Parameters*


![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 042](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/19881fc7-f098-40c5-be14-f7fce289aa75)

The Random Forest model with SMOTE + RFECV was re-trained with new parameters and tested on the validation data, demonstrating performance gains over the non-tuned model.

Finally, the non-tuned and tuned Random Forest models were evaluated on Test data. The table below shows their performance.

**Table 12**

*Un-tuned Random Forest Compared to Tuned Random Forest on Test Data*

![Aspose Words 4605fa2d-c8f8-4a27-acb0-98773c326984 043](https://github.com/jaymonty/Credit-Risk-Analysis/assets/18198506/75c2d403-f63f-40be-8165-31d68d36e618)


The tuned Random Forest performed superior with a performance increase on all metrics. We felt that this tuned model was ready for deployment.

9. **Deployment**

<a name="_page37_x72.00_y127.20"></a>All the work done and the code implementation has been uploaded to GitHub.

10. **Conclusion**

<a name="_page37_x72.00_y334.65"></a>In conclusion, this project provides the best prediction of credit card status based on

collected data features. After conducting essential cleaning and preprocessing steps, the data is utilized for modeling. SMOTE is implemented to mitigate target class imbalance, and RFECV feature selection is implemented to assist with overfitting issues. Random Forest and ensemble Voting Classifier yield the best predictions, with F1-macro scores of 80.10% and 80.38% on validation data, respectively. Considering the trade-off between computational expense and performance for potential future deployment, the project group decided to select the most appropriate modeling algorithm, Random Forest, for fine-tuning. This led to Random Forest being tuned with a randomized search, resulting in the best overall model that was capable of successfully classifying with an F1-macro score of 80.48% on test data.

<a name="_page38_x72.00_y72.00"></a>**References**

F. Leon, S. A. Floria and C. Bădică, "Evaluating the effect of voting methods on ensemble-based classification," 2017 *IEEE International Conference on INnovations in Intelligent SysTems and Applications (INISTA)*, 2017, pp. 1-6, doi: 10.1109/INISTA.2017.8001122.

Gahlaut, A., Tushar, & Singh, P. K. (2017). Prediction analysis of risky credit using data mining

classification models. *2017 8th International Conference on Computing, Communication and Networking Technologies (ICCCNT)*. https://doi.org/10.1109/icccnt.2017.8203982

Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow:

Concepts, Tools, and Techniques to Build Intelligent Systems (2nd ed.). O’Reilly Media.

Kelleher, J. D., Namee, B. M., & D’Arcy, A. (2015). Fundamentals of Machine Learning for

Predictive Data Analytics: Algorithms, Worked Examples, and Case Studies. Amsterdam University Press.

Kuhn, M., & Johnson, K. (2018). Applied Predictive Modeling. Springer Publishing.

J. Laaksonen and E. Oja, "Classification with learning k-nearest neighbors," Proceedings of International Conference on Neural Networks (ICNN'96), 1996, pp. 1480-1483 vol.3, doi: 10.1109/ICNN.1996.549118.

Laborda, J., & Ryoo, S. (2021). Feature selection in a credit scoring model. *Mathematics*, *9*(7),

746\. https://doi.org/10.3390/math9070746

Machado, M. R., & Karray, S. (2022). Assessing credit risk of commercial customers using

hybrid machine learning algorithms. Expert Systems with Applications, 200, 116889. https://doi.org/10.1016/j.eswa.2022.116889

Moscato, V., Picariello, A., & Sperlí, G. (2021). A benchmark of machine learning approaches

for Credit Score Prediction. *Expert Systems with Applications*, *165*, 113986. https://doi.org/10.1016/j.eswa.2020.113986

Paris, R. (2022). Credit score classification [Data set]. *Kaggle*.

https://www.kaggle.com/datasets/parisrohan/credit-score-classification

Singh, P. (2017). Comparative study of individual and ensemble methods of classification for

credit scoring. *2017 International Conference on Inventive Computing and Informatics (ICICI)*. https://doi.org/10.1109/icici.2017.8365282

Trivedi, S. K. (2020). A study on credit scoring modeling with different feature selection and

machine learning approaches. *Technology in Society*, *63*, 101413. https://doi.org/10.1016/j.techsoc.2020.101413

Vanderplas, J., & VanderPlas, J. (2016). Python Data Science Handbook: Essential Tools for

Working with Data. Van Duuren Media.

What is XGBoost? (n.d.). NVIDIA Data Science Glossary.

https://www.nvidia.com/en-us/glossary/data-science/xgboost/

Zhu, & Lin. (2017, December). Synthetic minority oversampling technique for multiclass

imbalance problems. ScienceDirect. https://www.sciencedirect.com/science/article/pii/S0031320317302947?via=ihub
