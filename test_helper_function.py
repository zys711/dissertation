from helper_function import bayes_tuning,grid_tuning
import pytest
from numpy import array
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from warnings import filterwarnings

filterwarnings("ignore")

X_train=array([[1,1],[2,2],[3,3],[4,4],[-1,-1],[-2,-2],[-3,-3],[-4,-4]])
y_train=array([0,0,0,0,1,1,1,1])
X_test=array([[10,10],[-10,-10]])
y_test=array([0,1])
stratified_kfold=StratifiedKFold(n_splits=2)
oversampling=None
classifier=SVC()
search_spaces={'classifier__C':[1,2]}

@pytest.mark.filterwarnings("ignore")
def test_bayes_tuning():
    cv,test,list_=bayes_tuning(X_train, y_train,X_test, y_test,stratified_kfold,oversampling,classifier,search_spaces)
    assert cv==1
    assert test==1
    assert list_==[1,1]

@pytest.mark.filterwarnings("ignore")
def test_grid_tuning():
    cv,test,list_=grid_tuning(X_train, y_train,X_test, y_test,stratified_kfold,oversampling,classifier,search_spaces)
    assert cv==1
    assert test==1
    assert list_==[1,1]