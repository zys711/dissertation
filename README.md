# Dissertation Code Implementation of Yisu Zong
This repository contains:
1. hospital closure prediction using multiple hospital characteristics variables (hosp_pred_q.ipynb)
2. hospital closure prediction using quality of care variables (hosp_pred_hosp.ipynb)
3. combined hospital closure prediction (combine.ipynb)
4. helper functions for tuning and plot (helper_function.py)
5. test of helper functions (test_helper_function.py)

Instructions
------------
Requirements
====

To sucessfully run files in this repository, the following packages should be installed first:
```
numpy==1.20.2
pandas==1.0.5
seaborn==0.11.0
xgboost==1.4.2
skopt==0.9.dev0
sklearn==0.24.2
pyodbc==4.0.0
pip==21.2.4
imblearn==0.7.0
```

Specially, you may have to:
```
pip install xgboost
pip install pyodbc
pip install imblearn
```

```skopt``` should be installed the newest version in any .ipynb file first:
```
!pip install git
!git clone https://github.com/scikit-optimize/scikit-optimize.git
cd scikit-optimize
!pip install -r requirements.txt
```

Test
====
Tests run as:
```
pytest -p no:warnings
```
