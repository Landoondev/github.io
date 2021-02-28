# 13-5 随机森林和 Extra-Trees

决策树在节点划分上，在随机的特征子集上寻找最优划分特征。

## 随机森林

随机森林集成了决策树和 bagging 方法。

```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, random_state=666, oob_score=True, n_jobs=-1)
rf_clf.fit(X, y)
rf_clf.oob_score_ # 0.892
```



```python
rf_clf2 = RandomForestClassifier(n_estimators=500,
                                 max_leaf_nodes=16,
                                 random_state=666, oob_score=True, n_jobs=-1)
rf_clf2.fit(X, y)
rf_clf2.oob_score_ # 0.906
```

决策树在节点划分上，使用**随机的特征**和**随机的阈值**。

## Extra-Trees

Extra-Trees 提供额外的随机性，抑制过拟合，但增大了 bias。

```python
from sklearn.ensemble import ExtraTreesClassifier

et_clf = ExtraTreesClassifier(n_estimators=500, bootstrap=True, random_state=666)
et_clf.fit(X, y)
```

## 集成学习解决回归问题

```python
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
```

