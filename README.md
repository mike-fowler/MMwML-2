# MMwML-2

Stock Price Prediction

This program takes its training data from the first 23 days of August and then uses 3 different regression models to predict the stock price at the end of the month. The 3 different models were:

1.  Simple Linear Regression
2.  Support Vector Regression using 2nd order Polynomial kernel
3.  Support Vector Regression using 'Radial Basis Function' kernel

The chart displayed shows the actual price over the course of August and the 3 models' respective projections.


Results

_____________________________________________________________________
Price predictions made for the end of the month on the 23rd of August
_____________________________________________________________________
The Actual price at the end of August:    $167.5100

The svr_rbf predicted price:              $139.3750

The svr_poly predicted price:             $188.7270

The lin_reg predicted price:              $168.3800
_____________________________________________________________________


Conclusion

In this particular set of circumstances, the linear model is vastly superior to the other more sophisticated models, but this is just an illustration and not representitive of real world situations.
