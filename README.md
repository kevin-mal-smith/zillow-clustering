# What Is My House Worth?
---
by Kevin smith 7/26/2022

## Project Goal
---
The goal of this project is to develop a home price estimation model that performs better than the baseline prediction, and develop recommendations for ways that the model can be improved and deployed. 

This goal will be accomplished utilizing the following steps:

* Planning
* Acqusition
* Prep
* Exploration
* Feature Engineering
* Modeling
* Delivery

### Steps For Reproduction
---
1. You will need an <mark>env.py</mark> file that contains the hostname, username and password of the mySQL database that contains the <mark>telco_churn</mark> database. Store that env file locally in the repository.
2. Clone my repo (including the <mark>acquire.py</mark> , <mark>prepare.py</mark> & <mark>wrangle.py</mark>files.
3. The libraries used are pandas, numpy, scipy, matplotlib, seaborn, and sklearn.
4. You should now be able to run the <mark>zillow_final_report.ipynb</mark> file.

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
4. Do some preliminary exploration of the data (including visualiztions and statistical analyses)*
5. Trim dataset of variables that are not statistically significant
6. Determine which machine learning model perfoms the best
7. Utilize the best model on the test dataset
8. Create a final report notebook with streamlined code optimized for a technical audience

*at least 4 visualizations and 2 statistical analyses

## Data Library
---
| **Variable Name** | **Explanation** | **Values** |
| :---: | :---: | :---: |
| bedrooms | The number of bedrooms in the house | Numeric value |
| bathrooms | The number of bathrooms in the house | Numeric value |
| quality | a numeric score based on quality on construction | Numeric value|
| sq_feet | The total area inside the home | Numeric value |
| pool | Whether or not the house has a pool | Yes=1/No=0|
| tax_value| The taxable value of the home in $USD | Numeric |
| yearbuilt | The year in which the home was originally built | Year |
| fips | A unique code specific to the county in which the home is located| Numeric |



## Initial Hypothesis
--- 
The initial hypothesis can be based on a gut instinct or the first question that comes to mind when encountering a dataset.

|**Initial hypothesis number** |**hypothesis** |
| :---: | :---: |
|Initial hypothesis 1 | Square footage drives up home value|
|Initial hypothesis 2 | Age drives down home value|

## Acquire and Cache
---
Utitlize the functions imported from the <mark>acquire.py</mark> to create a DataFrame with pandas.

These functions will also cache the data to reduce execution time in the future should we need to create the DataFrame again.

## Prep
--- 
In this step we will utilize the functions in the <mark>wrangle.py</mark> file to get our data ready for exploration. 

This means that we will be looking for columns that may be dropped because they are duplicates, and either dropping or filling any rows that contain blanks depending on the number of blank rows there are.

This also means that we will be splitting the data into 3 separate DataFrames in order to prevent data leakage corrupting our exploration and modeling phases.


## Exploration
---
This is the fun part! this is where we get to ask questions, form hypotheses based on the answers to those questions and use our skills as data scientist to evaluate those hypotheses!

For example, in the zillow dataset I asked "Does square footage drive up value?" and unsurprisingly the answer was generally yes. This lead me to the hypothesis that square footage would have a dependent relationship with tax value, which hypothesis testing confirmed. However I was able to find another variable that did a better job of predicting value.

## Feature Engineering
---
I didnt try to reinvent the wheel here. I used Sklearn's RFE function to find the 4 most important variables for each county.


## Modeling
---
Here we determine the best model to use for predicting value. I ran the data through 4 different regression algorithms to determine which would perform the best

The polynomial model worked best on larger data sets like the LA homes, or the dataframe as a whole without splitting into counties

## Delivery
---
Here we will complete the goal of the project by delivering actionable suggestions to improve value projections based on our identification of contributing factors. 

Since the model on the dataset as a whole perfomed comparably to the models split by county, I suggest **not** splitting the data unless working with more observations.

my second suggestion is to continue to search for more variables that can drive tax value. Purchase history for individuals houses would be the first place i would look to improve the data, and in turn improve the model. 
