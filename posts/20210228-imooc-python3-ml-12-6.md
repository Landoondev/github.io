# 12-6 决策树解决回归问题

```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
from sklearn import datasets

boston = datasets.load_boston()
X = boston.data
y = boston.target
```



```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
```

## Decision Tree Regressor


```python
from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor()
dt_reg.fit(X_train, y_train)
```




    DecisionTreeRegressor()




```python
dt_reg.score(X_test, y_test)
```




    0.5962687054930247




```python
dt_reg.score(X_train, y_train) # 过拟合了
```




    1.0


```python

```

