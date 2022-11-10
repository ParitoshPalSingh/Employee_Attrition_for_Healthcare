# Employee_Attrition_for_Healthcare




Many of the causes of healthcare worker attrition exist because of the the stressful nature of the healthcare work industry and how over the years it has become more stressful. Numerous employees have to work long hours unlike other industries where there is a limit on how many hours employees have to work. This excessive work often leads to high burnout. Employee attrition in healthcare is a pertinent issue and it exacerbates due to the limited supply of workers in the healthcare industry. Since the healthcare space is understaffed and many healthcare employees are overworked, the quality of care and the speed to care are often ignored and negatively impacted.

Dataset Description:

The dataset contains 35 columns in total and it only has 1676 data entries.

This dataset contains employee and company data useful for supervised ML and analytics. Attrition - whether an employee left or not - is included and is used as the target variable. The data is synthetic and based on the IBM Watson dataset for attrition. Employee roles and departments were changed to reflect the healthcare domain. Also, known outcomes for some employees were changed to help increase the performance of ML models.

About the Dataset Features and values :- 

EmployeeID                  : This column contains the Unique Employee ID for all the employees
Age                         : This column contains the age of the employees
Attrition (Target)          : Binary output suggesting whether an employee left or not. This column is also unbalanced with majority class being no and minority being yes.
BusinessTravel              : This column classifies the amount of travel the employees do i.e., rarely, frequently or no travel.
DailyRate                   : The salary the employees earn per day
Department                  : Type of department the employees work under like Maternity, cardiology, neurology
DistanceFromHome            : How far they live from the office
Education                   : This column gives the level of education each employee has achieved
EducationField              : This column consists of type of Education Field like, Life Sciences, Medical, Marketing, etc.
EmployeeCount               : This column tells about the employee count
EnvironmentSatisfaction     : This gives the level of environment satisfaction for the employees
Gender                      : This column gives the gender of the employee Male or Female
HourlyRate                  : This column gives the hourly earnings by the people
JobInvolvement              : This column gives the level of involvement in job
JobLevel                    : This column represents the level of job for each of the employees
JobRole                     : This column gives the job role like Nurse, Admin, etc for all the employees
JobSatisfaction             : This gives the level of job satisfaction as rated by the employees
MaritalStatus               : This gives the martial status of people if they are married, single or divorced
MonthlyIncome               : This column gives the monthly earning income by the employees
MonthlyRate                 : This column gives the monthly rate that the employees earn
NumCompaniesWorked          : This column gives information about how many different companies did the employees worked for
Over18                      : Binary output suggesting if employees are above 18 years old or not
OverTime                    : Binary output suggesting if the employee has done overtime or not
PercentSalaryHike           : This column gives the total salary hike in percentage given to each of the employees
PerformanceRating           : Gives the performance rating for the employees
RelationshipSatisfaction    : Level of relationship satisfaction as rated by the employees
StandardHours               : The number of standard hours each individual employee is working
Shift                       : Number of shifts the employee undertake
TotalWorkingYears           : Number of years the employees have been working in total
TrainingTimesLastYear       : The amount of times each employee underwent training
WorkLifeBalance             : Ratings for work life balance as given by the employees
YearsAtCompany              : Number of years the employees have been working at the company
YearsInCurrentRole          : Number of years the employees have been working in their current role/position
YearsSinceLastPromotion     : Number of years since their last promotion
YearsWithCurrManager        : Number of years the employees have been working with their current manager 


The Attrition dataset for healthcare can be found here!

Dataset and Data type for each of the columns:

The dataset has 35 columns and Attrition is the target variable that I will be trying to predict using the other 34 columns	To check whether any of the columns have null values that need to be dealt with, I have used the info() in pandas and this gives us a brief overview of how many non null columns the dataset has.




Data Preprocessing and Cleaning:

Data preprocessing is the process of transforming raw data into an understandable and desired format. It is one of the most important steps in Machine Learning as we cannot work directly with raw data to produce positive and insightful predictions. It is essential to check the quality of the data before applying any machine learning algorithm. 80% of the time is mostly spent on polishing the data to a certain extent, which can then be used to apply Machine Learning algorithms on. The data needs to be free from unwanted noise and outliers to make sure that the models focus on learning the generalized data and not the noise, thereby making the predictions much more effective and correct.

There are many techniques we can follow but I have done data preprocessing using the following steps:

Identify and Remove or update any and all Null and N/A values
Identify the outliers for each of the numeric based columns
Deal with outliers that are identified using quartiles and interquartile range
Remove any columns that have only one value throughout the dataset




Since, the data is already free form n/a and null values, I will move to the second and third step in which I have to identify any and all outliers for all the columns and then deal with them by either dropping or replacing them with a certain value. 



Box Plots

A box plot tells us, more or less, about the distribution of the data. It gives a sense of how much the data is actually spread about, what’s its range, and about its skewness. 

Box Plot has lines away from the box and these lines are maximum and minimum from the dataset.

minimum is the minimum value in the dataset, and maximum is the maximum value in the dataset.

So the difference between the two tells us about the range of dataset.

The median is the median (or center point), also called second quartile, of the data (resulting from the fact that the data is ordered).
Q1 is the first quartile of the data, i.e., to say 25% of the data lies between minimum and Q1.
Q3 is the third quartile of the data, i.e., to say 75% of the data lies between minimum and Q3.
The first and the third quartiles, Q1 and Q3, lies at -0.675σ and +0.675σ from the mean, respectively.
The difference between Q3 and Q1 is called the Inter-Quartile Range or IQR


The box plot above shows the outliers that are seen for the column Total Working Years. The points above the max whisker of the box plot are all considered as outliers. These outliers need to be removed or dealt with as these outliers can have detrimental effect on the performance of various machine learning algorithms. 	The box plot above shows the outliers that are seen for the column Monthly Income. There are many points above the max whisker of the box plot. This shows that the monthly income of people varies a lot and there are many instances where people are being paid more than 99 percentile of employees. 


To deal with the outliers, we have to check the quartile range and then set up both upper and lower limits outside which all data points are considered as outliers. For this it is important to know what IQR is and how it is used to set a limit for identifying and removing the outliers

When is a point considered as outlier is dependent on its position in the distribution
About 68.26% of the whole data lies within one standard deviation (<σ) of the mean (μ)
About 95.44% of the whole data lies within two standard deviations (2σ) of the mean (μ)
About 99.72% of the whole data lies within three standard deviations (<3σ) of the mean (μ)
And the rest 0.28% of the whole data lies outside three standard deviations (>3σ) of the mean (μ). This part of the data is considered as outliers.
 
Why scaling 1.5 times is used? 
(When scale is taken as 1.5, then according to IQR Method any data which lies beyond 2.7σ from the mean (μ), on either side, shall be considered as outlier. And this decision range is the closest to what Gaussian Distribution tells us)
To get exactly 3σ, we need to take the scale = 1.7


To deal with the outliers, I have selected to remove all outliers that have their values outside of a desired range. It is important to make sure that I don't drop a lot of data points as the dataset only has 1600+ rows in total. Hence, I have tailored the range of acceptable points such that the extreme values are dropped but not all the outliers are removed straight away. 	After handling all the outlier values, the updated number of entries are shown. In the dataset before preprocessing there were 1676 data points and it has dropped to 1545 after carrying out data preprocessing.




Correlation 

Correlation shows the strength of a relationship between two variables and is expressed numerically by the correlation coefficient. The correlation coefficient's values range between -1.0 and 1.0.

A perfect positive correlation means that the correlation coefficient is exactly 1. This implies that as one feature value moves, either up or down, the other feature value moves in tandem, in the same direction.

A perfect negative correlation means that two assets move in opposite directions, while a zero correlation implies no linear relationship at all.


There are many columns that have correlation amongst each other like:

Years at company is highly correlated with years in current role, years with current manager and years since last promotion
Percent Salary Hike is highly correlated with the Performance Rating meaning higher the rating, higher the chance of getting a better salary hike
Job level and Monthly income are highly correlated, showing that the higher job levels have a better monthly income than the lower job levels
Since StandardHours and EmployeeCount have a constant value, they have no correlation at all with any of the other columns/features.




Models and Experimentation:

The dataset is used to implement three classification ensemble learning based models which are Random Forest, eXtreme Gradient Boosting (XGB) and Categorical Boosting (CatBoost). The experiment is carried out in 4 phases where each phase has a different approach towards reaching the solution. The four phases are as follows:

Phase 1: In phase one, there is no feature selection, no over sampling and no outliers are removed. The dataset is normalized and then encoded to be used with the different Machine Learning Models.
Phase 2: In phase two, there is no feature selection, no over sampling. However the outliers are removed and then the dataset is normalized & encoded to be used with the different Machine Learning Models.
Phase 3: In phase three, there is no over sampling. However, limited features are selected based on the correlation values and all the outliers are detected and removed. The dataset is then normalized & encoded to be used with the different Machine Learning Models.
Phase 4: In phase four, a limited number of features are selected based on the correlation values and all the outliers are detected and removed. Moreover, the training set is oversampled as there is an imbalance in the data. The dataset is normalized and encoded to be used with the different Machine Learning Models.




Bagging:

A Bagging classifier is an ensemble technique that fits base classifiers each on random subsets of the original dataset and then aggregates their individual predictions (either by voting or by averaging) to form a final prediction. Each base classifier is trained in parallel with a training set which is generated by randomly drawing, with replacement, N examples from the original training dataset, where N is the size of the original training set. The training set for each of the base classifiers is independent of each other. Bagging reduces overfitting (variance) by averaging or voting, however, this leads to an increase in bias, which is compensated by the reduction in variance though




Random Forest:

The random forest algorithm is an extension of the bagging method as it utilizes both bagging and feature randomness to create an uncorrelated forest of decision trees. Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction. Feature randomness, also known as feature bagging.

Boosting:

Boosting is an ensemble modelling, technique that attempts to build a strong classifier from the number of weak classifiers. It is done by building a model by using weak models in series. Firstly, a model is built from the training data. Then the second model is built which tries to correct the errors present in the first model. This procedure is continued and models are added until either the complete training data set is predicted correctly or the maximum number of models are added.



eXtreme Gradient Boosting (XGB): 

XGBoost is an implementation of Gradient Boosted decision trees. In XGB, decision trees are created in sequential form. Weights are assigned to all the independent variables which are then fed into the decision tree which predicts results. The weight of variables predicted wrong by the tree is increased and these variables are then fed to the second decision tree. These individual classifiers/predictors then ensemble to give a strong and more precise model. It can work on regression, classification, ranking, and user-defined prediction problems.

Categorical Boost (CatBoost): 

CatBoost is based on gradient boosted decision trees. During training, a set of decision trees is built consecutively. Each successive tree is built with reduced loss compared to the previous trees. The number of trees is controlled by the starting parameters. To prevent overfitting, there is an option to use the overfitting detector. When it is triggered, trees stop being built. The overall performance for CatBoost is better than other gradient boosting techniques and additionally, the categorical boosting is better than XGB when dealing with categorical features and is faster than XGBoost in most cases.







Performance & Results:

Accuracy:

It is defined as total correctly classified example divided by the total number of classified examples. This metric is very important when error in predicting all class is equally important. Here False positive is most important to address than False negative. 

Accuracy = TP+ TN /(TP+FP+TN+FN)     

OR       

Accuracy = Correct Predictions / Total Predictions



F1-Score:

F1 score is a weighted average of precision and recall. As we know in precision and in recall there is false positive and false negative so it also consider both of them. F1 score is usually more useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall.

F1 Score = 2*(Recall * Precision) / (Recall + Precision)


The key differences between the F1-score and the accuracy,

Accuracy is used when the True Positives and True negatives are more important while F1-score is used when the False Negatives and False Positives are crucial
Accuracy can be used when the class distribution is similar while F1-score is a better metric when there are imbalanced classes.



Conclusion: 

The best model was CatBoost as this model achieved best F1-Scores throughout the different phases as compared to both Random Forest and the Extreme Gradient Boosting. Since, the dataset used was highly imbalanced, the performance of the better model is decided on the basis of F1-score and not accuracy. Throughout all the phases, CatBoost performed the best as it not only has the highest F1-Scores, but also the performance for all the different phases was nearly identical when using CatBoost as compared to Random Forest and XGBoost where the variance in performance was greater for all the different phases.   

Employee attrition is a growing issue in healthcare spaces. Issues with long hours, low pay, and low supply in the workforce contribute to the high burnout rate of healthcare workers. While some employees work to foster better work life balance and work environments, using data analytics can help identify employees at high risk of leaving and even take preventative measures. Having insights into which factors contribute to attrition can aid employers in taking these preventative measures.


