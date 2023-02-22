

The aim of the project is to build a machine learning model that can accurately predict the worth of a house. This can aid both the buyers and sellers of houses in making a well informed decision. The dataset contains . Notebook contains code that covers profiling, data cleansing, feature engineering, exploratory data analysis, data visualization, model training, and hyper-parameter tuning. We find the best prediction model among all the regression models in the ML pipeline.


# Home-Valuation
## Problem Statement & Background
To build a machine learning (regression) model which is able to accurately predict the worth of a home. <br>
<br>
Home Valuation model can be a very important tool for both the seller and the buyer as it can aid them in making well informed decision. For sellers, it may help them to determine the average price at which they should put their home for sale while for buyers, it may help them find out the right average price to purchase the home.
<br>

## Exploratory Data Analysis & Visualizations
Collected data from a hackathon. It has data of around 50K houses across different Indian cities. Done some data cleaning and features extraction. Some of the key attributes are - 1. Area 2. Year_built 3. BHK_NO 4. Furnishing 5. RERA_Approved. Here are few data visualizations such as correlation heatmaps, scatter plots, pie charts, etc. 
<br>
<br>
<p>                                                                                                                      
<img src="https://github.com/thota-sasanth/Home-Valuation/blob/main/lat_long.png" width="480" height="380" align="right">
<img src="https://github.com/thota-sasanth/Home-Valuation/blob/main/heatmap.png" width="480" height="380" align="left"> <br>
</p>  <br>
<br>




<br><br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

<br>
                                                                                                          
<br>
  
  <img src="https://github.com/thota-sasanth/Home-Valuation/blob/main/piecharts.png" width="1200" height="200">
  <img src="https://github.com/thota-sasanth/Home-Valuation/blob/main/sctter_plot.png" width="800" height="500">
  <img src="https://github.com/thota-sasanth/Home-Valuation/blob/main/bhkno.png" width="600" height="400">

<br>

## Models
Applied the following regression algorithms - Lasso, Ridge, Enet, Random forest and Gradient boosting. These algorithms are also used to deal with the problems like overfitting. Created Pipelines for each regression algorithm with different hyperparameters. 
<br>

## Results
Metrics such as CV (cross validation) score, R2_score and MAE are used to evaluate performance of each model. The Gradient boosting model predicted home prices with highest CV score.
<br>
<br>

