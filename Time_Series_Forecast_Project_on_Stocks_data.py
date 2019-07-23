from sklearn.externals import joblib
import sys, numpy as np
import pandas as pd

model = joblib.load(open('DecisionTreeClassified_29Apr.obj', 'rb'))
#proprocess = joblib.load(open('minMaxTransform_29Apr.obj', 'rb'))

dfStocks = pd.read_csv(sys.argv[1])
dfStocksXformed = dfStocks.copy()

# Preprocessing remaining
dfStocksXformed['UpDaysChange'] = np.round(dfStocksXformed['upperc5']/dfStocksXformed['upperc20'], 3)
dfStocksXformed['Price3dChange'] = dfStocksXformed['dpmin']/dfStocksXformed['dpmax']
dfStocksXformed['PriceChange'] = np.round(dfStocksXformed['close']/dfStocksXformed['preClose'], 3)
dfStocksXformed['ChangeSMA13vs26'] = dfStocksXformed['sma_13']/dfStocksXformed['sma_26']

# Impute values for Inf
dfStocksXformed.loc[dfStocksXformed['UpDaysChange'] == np.inf, 'UpDaysChange']=20
dfStocksXformed.loc[dfStocksXformed['Price3dChange'] == np.inf, 'Price3dChange']=20
dfStocksXformed.loc[dfStocksXformed['PriceChange'] == np.inf, 'PriceChange']=20
dfStocksXformed.loc[dfStocksXformed['ChangeSMA13vs26'] == np.inf, 'ChangeSMA13vs26']=20

#Significant Variables
impCols = ['UpDaysChange', 'ChangeSMA13vs26', 'Price3dChange', 'PriceChange', 'macd']

temp = model.predict(dfStocksXformed[impCols])
dfStocksXformed['Predicted'] = pd.Series(temp+1)
dfStocksXformed.to_csv('Output_DTC.csv')