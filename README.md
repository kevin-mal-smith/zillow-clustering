# How Many Log Errors Does it Take to Predict Home Value?
---
by Kevin smith 8/9/2022

## Project Goal
---
The goal of this project is to develop a model that utilizes clustering and log error and outperforms the baseline prediction, and develop recommendations for ways that the model can be improved and deployed. 

This goal will be accomplished utilizing the following steps:

* Planning
* Acqusition
* Prep
* Exploration
* Clustering
* Feature Engineering
* Modeling
* Delivery

### Steps For Reproduction
---
1. You will need an <mark>env.py</mark> file that contains the hostname, username and password of the mySQL database that contains the <mark>tzillow</mark> database. Store that env file locally in the repository.
2. Clone my repo (including the <mark>explore.py</mark> , <mark>model.py</mark> & <mark>wrangle.py</mark>files.
3. The libraries used are pandas, numpy, scipy, matplotlib, seaborn, and sklearn.
4. You should now be able to run the <mark>zillow_final_cluster.ipynb</mark> file.

## Planning
---
Their are two essential parts to any good plan. Identify your **Goals**, and the necessary **Steps** to get there. 

### Goals:
1. Identify variables driving housing prices.
2. Develop a model to make value predicitons based on those variable. 
3. Deliver actionable takeaways

### Steps:
1. Initial hypothesis
2. Acquire and cache the dataset
3. Clean, prep, and split the data to prevent data leakage
4. Do some preliminary exploration of the data (including visualiztions and statistical analyses)
5. Trim dataset of variables that are not statistically significant
6. Determine which machine learning model perfoms the best
7. Utilize the best model on the test dataset
8. Create a final report notebook with streamlined code optimized for a technical audience



## Data Library
---
| **Variable Name** | **Explanation** | **Values** |
| :---: | :---: | :---: |
| parcelid | a unique identification number | Nueric value |
| logerror | a metric of how much the zestimate missed by | Numeric value |
| bed | The number of bedrooms in the house | Numeric value |
| bath | The number of bathrooms in the house | Numeric value |
| sq_feet | The total area inside the home | Numeric value |
| latitude | Measure of location North/South | Numeric value|
| longitude | Measure of location East/West | Numeric value |
| lot_size | lot size in square feet | Numeric value |
| city_id | a unique code corresponding to city | Numeric value |
| zip | unique 5 digit code for regions utilized by the post office| Numeric vaue |
| year | what year the house was built | date |
| home_value | the cost of the structure itself in $USD | Numeric value |
| land_value | the cost of the land itself in $USD | Numeric value |
| la | a binary value for whether or not the home is in LA county | 1=Yes, 0=No |
| orange| a binary value for whether or not the home is in Orange county | 1=Yes, 0=No |
| ventura | a binary value for whether or not the home is in Ventura county | 1=Yes, 0=No |
| age | year built subtracted from current year | Numeric value |
| acres | measure of the size of the lot the home was built on | Numeric value |
| land_cost/sqf | price of land per square foot | Numeric value |
| home_cost/sqf | price of the home per square foot | Numeric value |
| tax_rate | the tax percentage on the home | Numeric |
| bed_bath_ratio | the number of bathrooms compared to number of bedrooms | Numeric value|



## Initial Hypothesis
--- 
The initial hypothesis can be based on a gut instinct or the first question that comes to mind when encountering a dataset.

|**Initial hypothesis number** |**hypothesis** |
| :---: | :---: |
|Initial hypothesis 1 | Latitude and Longitude have a non-linear relationship with the price of land per square foot|
|Initial hypothesis 2 | The size of the lot has a non-linear relationship with size of the home and costof the home per square foot |

## Acquire and Prep
---
Utitlize the functions imported from the <mark>wrangle.py</mark> to create a DataFrame with pandas.

These functions will also cache the data to reduce execution time in the future should we need to create the DataFrame again.

In this step we will utilize the functions in the <mark>wrangle.py</mark> file to get our data ready for exploration. 

This means that we will be looking for columns that may be dropped because they are duplicates, and either dropping or filling any rows that contain blanks depending on the number of blank rows there are.

This also means that we will be splitting the data into 3 separate DataFrames in order to prevent data leakage corrupting our exploration and modeling phases.


## Exploration
---
This is the fun part! this is where we get to ask questions, form hypotheses based on the answers to those questions and use our skills as data scientist to evaluate those hypotheses!

For example, in the zillow dataset I asked "Does square footage drive up value?" and unsurprisingly the answer was generally yes. This lead me to the hypothesis that square footage would have a dependent relationship with tax value, which hypothesis testing confirmed. However I was able to find another variable that did a better job of predicting value.

## Clustering
---
I created 3 clusters and utilized them as features for the models.

|**Cluster** |**Elements** |
| :---: | :---: |
| area_home_cost | scaled_sq_feet, scaled_acres, scaled_home_cost |
| location | scaled_latitude, scaled_longitude, city_id |
| age_bed_cost | age, bed_bath_ratio, scaled_home_cost/sqf |

## Feature Engineering
---
I didnt try to reinvent the wheel here. I used Sklearn's Kbest function to find the most important variables for our model.

Our clusters were the bottom 3.


## Modeling
---
Here we determine the best model to use for predicting value. I ran the data through 3 different regression algorithms to determine which would perform the best

The linear and polynomial models both had the same results. likely due to the fact that the polynomial utilizes a linear regressor in its modeling.

all 3 models performed better than baseline, with liner/polynomial tied for best performance

## Delivery
---
Here we will complete the goal of the project by delivering actionable suggestions to improve value projections based on our identification of contributing factors. 

Since the model on the dataset as a whole perfomed comparably to the models split by county, I suggest **not** clustering as it yields no ipact on the models performance.

My second suggestion would be predict where log error will be poistive or negative and then implement a non-linear regressor to make predictions about how positive or negative log error will be. 
