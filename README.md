
# Stock-Price-Predictor

## Introduction:
Machine learning and deep learning have found their place in the financial institutions for their power in predicting time series data with high degrees of accuracy and the research is still going on to make the models better. This project was done in order to complete the academic project of Data Mining.

## Project workflow
The workflow for this project is essentially in these steps:

* Linear Regression
* KNN Regression
* Decision Tree Regression
* SVM Regression
* **RNN LSTM**
* Overall Conclusion

## Data Used

Yahoo Finances api is used with the help of pandas-datareader to fetch the historical data of a particular stock.

![Data Used](https://user-images.githubusercontent.com/68889070/123813577-a8d37080-d912-11eb-9b8e-0bf0f476e31b.png)

## Regression Models
Various regression models are implemented to predict the stock price, but alas!, they all failed miserably.

Let's take a look at their results one by one:

### Linear Regression
![Linear Regression Model Graph](https://user-images.githubusercontent.com/68889070/123814226-27301280-d913-11eb-953d-da0b7c73938f.png)
### K-NN Regression
![k-nn Regression Model Graph](https://user-images.githubusercontent.com/68889070/123814305-3616c500-d913-11eb-895f-3da1d0b7f264.png)
### Decision Tree Regression
![Decision Tree Regression Model Graph](https://user-images.githubusercontent.com/68889070/123814373-43cc4a80-d913-11eb-8ba6-423cde373323.png)
### SVM Regression
![SVM Regression Model Graph](https://user-images.githubusercontent.com/68889070/123814453-521a6680-d913-11eb-9939-185eb361332f.png)

## RNN LSTM
This model performed very well. It surpassed all the regression models in terms of accuracy.

#### Summary
![LSTM Model Summary](https://user-images.githubusercontent.com/68889070/123815083-d371f900-d913-11eb-8958-dd54f60320e8.png)

#### Prediction 
![LSTM Graph](https://user-images.githubusercontent.com/68889070/123815347-09af7880-d914-11eb-96f2-994f30f416b6.png)

## Conclusion:

* Working with time series datasets is not that easy, since most of machine learning algorithms are not smart to detect the overfit on the Date column during the model training.

* Thus, LSTM is a good choice to be used especially when dealing with this kind of problems.
