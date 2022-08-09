import matplotlib.pyplot as plt
from datetime import date
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import wrangle

import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import RobustScaler, MinMaxScaler, QuantileTransformer,PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge


def baseline(y_train_scaled,y_validate_scaled, y_test_scaled):
     y_train_scaled['baseline'] = y_train_scaled.logerror.mean()
     y_validate_scaled['baseline'] = y_train_scaled.logerror.mean()
     y_test_scaled['baseline'] = y_train_scaled.logerror.mean()


     rmse_train_baseline = mean_squared_error(y_train_scaled.logerror, y_train_scaled.baseline)**(1/2)
     rmse_validate_baseline = mean_squared_error(y_validate_scaled.logerror, y_validate_scaled.baseline)**(1/2)
     rmse_test_baseline = mean_squared_error(y_test_scaled.logerror, y_test_scaled.baseline)**(1/2)

     print("RMSE using Mean\nTrain/In-Sample: ", rmse_train_baseline, 
      "\nValidate/Out-of-Sample: ", rmse_validate_baseline,
      "\nTest/Out-of-Sample: ", rmse_test_baseline)

def linear(x_train_scaled,y_train_scaled, y_validate_scaled,x_validate_scaled,x_test_scaled, y_test_scaled):
     lm = LinearRegression()
     lm.fit(x_train_scaled, y_train_scaled.logerror)

     y_train_scaled['pred'] = lm.predict(x_train_scaled)

     rmse_train = mean_squared_error(y_train_scaled.logerror, y_train_scaled.pred)**(1/2)

     y_validate_scaled['pred'] = lm.predict(x_validate_scaled)

     rmse_validate = mean_squared_error(y_validate_scaled.logerror, y_validate_scaled.pred)**(1/2)

     y_test_scaled['pred'] = lm.predict(x_test_scaled)

     rmse_test = mean_squared_error(y_test_scaled.logerror, y_test_scaled.pred)**(1/2)

     print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate,
      "\nTest/ Out-of-Sample: ", rmse_test)

     baseline_total_rmse=0.16690221961201632+0.1714646972037276+0.1965547402621701
     model_total_rmse = rmse_train+rmse_validate+rmse_test

     final_pct= (baseline_total_rmse - model_total_rmse)/baseline_total_rmse
     print(final_pct)
     print(f'The linear model performed {final_pct:.2%} better than the baseline when predicting log error.')


def kernel(x_train_scaled,y_train_scaled, y_validate_scaled,x_validate_scaled):
     kr = KernelRidge(alpha=1)
     kr.fit(x_train_scaled, y_train_scaled.logerror)

     y_train_scaled['pred'] = kr.predict(x_train_scaled)

     rmse_train = mean_squared_error(y_train_scaled.logerror, y_train_scaled.pred)**(1/2)

     y_validate_scaled['pred'] = kr.predict(x_validate_scaled)

     rmse_validate = mean_squared_error(y_validate_scaled.logerror, y_validate_scaled.pred)**(1/2)


     print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)

def lasso(x_train_scaled,y_train_scaled, y_validate_scaled,x_validate_scaled):
     lars = LassoLars(alpha=1.0)
     lars.fit(x_train_scaled, y_train_scaled.logerror)

     y_train_scaled['pred'] = lars.predict(x_train_scaled)

     rmse_train = mean_squared_error(y_train_scaled.logerror, y_train_scaled.pred)**(1/2)

     y_validate_scaled['pred'] = lars.predict(x_validate_scaled)

     rmse_validate = mean_squared_error(y_validate_scaled.logerror, y_validate_scaled.pred)**(1/2)


     print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)


def poly(x_train_scaled,y_train_scaled, y_validate_scaled,x_validate_scaled, x_test_scaled, y_test_scaled):
     pf = PolynomialFeatures(degree=3) 
     x_train_degree2 = pf.fit_transform(x_train_scaled)
     x_validate_degree2 = pf.transform(x_validate_scaled)

     

     lm2 = LinearRegression()
     lm2.fit(x_train_scaled, y_train_scaled.logerror)

     y_train_scaled['pred'] = lm2.predict(x_train_scaled)

     rmse_train = mean_squared_error(y_train_scaled.logerror, y_train_scaled.pred)**(1/2)

     y_validate_scaled['pred'] = lm2.predict(x_validate_scaled)

     rmse_validate = mean_squared_error(y_validate_scaled.logerror, y_validate_scaled.pred)**(1/2)

     y_test_scaled['pred'] = lm2.predict(x_test_scaled)

     rmse_test = mean_squared_error(y_test_scaled.logerror, y_test_scaled.pred)**(1/2)


     print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate,
      "\nTest/ Out-of-Sample: ", rmse_test)

     baseline_total_rmse=0.16690221961201632+0.1714646972037276 +0.1965547402621701
     model_total_rmse = rmse_train+rmse_validate+rmse_test

     final_pct= (baseline_total_rmse - model_total_rmse)/baseline_total_rmse
     print(final_pct)
     print(f'The polynomial model performed {final_pct:.2%} better than the baseline when predicting log error.')
