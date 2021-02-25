# 10-4 F1 Score

精准率和召回率根据使用场景进行权衡。

- 有的时候我们注重精准率，如股票预测。
- 有的时候，我们注重召回率，如病人诊断。我们期望将所有有病的患者，都能预测出来。这种情况下，准确率低一些没有关系，即一些人没有病，算法错误的预测他们患病，这种情况下，让他们进行进一步的确诊就好。

## F1 Score

兼顾 Precision 和 Recall：F1 Score。

F1 Score 是 Precision 和 Recall 的调和平均值。

$$\frac{1}{F_1} = \frac{1}{2} (\frac{1}{precision} + \frac{1}{recall})$$

得到 F1 Score ：

$$F_1 = \frac{2 \cdot precision \cdot recall}{precision + recall}$$

```python
import numpy as np
```


```python
def f1_score(precision, recall):
    try:
        return 2 * precision * recall / (precision + recall)
    except:
        return 0.0
```


```python
precision = 0.5
recall = 0.5
f1_score(precision, recall)
```




    0.5




```python
precision = 0.1
recall = 0.9
f1_score(precision, recall)
```




    0.18000000000000002




```python
precision = 0.0
recall = 1
f1_score(precision, recall)
```




    0.0




```python
from sklearn import datasets
digits = datasets.load_digits()
X = digits.data
y = digits.target

# 使得手写数字数据集极度偏斜
y = digits.target.copy()
y[digits.target==9] = 1
y[digits.target!=9] = 0
```


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
```


```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)
```

    /Users/landonglei/anaconda3/envs/imooc-ml/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)





    0.9755555555555555




```python
y_predict = log_reg.predict(X_test)
```


```python
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

precision_score(y_test, y_predict)
```




    0.9473684210526315




```python
recall_score(y_test, y_predict)
```




    0.8




```python
from sklearn.metrics import f1_score

f1_score(y_test, y_predict)
```




    0.8674698795180723

