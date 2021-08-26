import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV


def bayes_tuning(X_train, y_train,X_test, y_test,stratified_kfold,oversampling,classifier,search_spaces):
    '''wrapper of hyperparameter tuning using bayesian optimisation. 

    Input
    ----------
    X_train : numpy.ndarray, size=(m,n)
    y_train : numpy.ndarray, size=(m,)
    X_test : numpy.ndarray, size=(k,n)
    y_test : numpy.ndarray, size=(k,)
    stratified_kfold : function of KFold() or StratifiedKFold()
    oversampling : oversampling function, for example SMOTE(), ADASYN(). if it's None, then no oversampling strategy is used 
    classifier : Machine learning classifier function, for example XGBClassifer()
    search_spaces : dict, region of hyperparameter. 
        for oversampling, the key and value in this dict should be:'oversampling__<hyperparameter name>': <range of this parameter>
        for classifier, the key and value in this dict should be:'classifier__<hyperparameter name>': <range of this parameter>

    Output
    ----------
    cv_score: float, the cv score of the best estimator
    test_score: float, the test score of the best estimator
    list_best_cv: list, list of the best estimator's validation score on each fold
    '''
    pipeline = imbpipeline(steps = [['oversampling', oversampling],['classifier', classifier]])
    bayes_search = BayesSearchCV(
        estimator = pipeline,
        search_spaces=search_spaces,    
        scoring='f1',
        cv=stratified_kfold,
        n_jobs=-1)
    bayes_search.fit(X_train, y_train)
    cv_score= bayes_search.best_score_
    test_score= bayes_search.score(X_test, y_test)
    best_estimator=pd.DataFrame(bayes_search.cv_results_)[pd.DataFrame(bayes_search.cv_results_)['rank_test_score']==1]
    orderd_dict_best_params=best_estimator.iloc[0]['params']
    list_best_cv=[]
    for col in best_estimator:
        if 'test_score' in col:
            if 'split' in col:
                list_best_cv.append(float(best_estimator.iloc[0:1][col]))
    print(f'Classifier: {str(classifier)}\nOver-sampling strategy: {str(oversampling)}\nCross-validation score: {cv_score}\nTest score: {test_score}\nBest parameters: {str(dict(orderd_dict_best_params))}\n')
    return cv_score,test_score,list_best_cv

def grid_tuning(X_train, y_train,X_test, y_test,stratified_kfold, oversampling,classifier,search_spaces):
    '''wrapper of hyperparameter tuning using grid search. 
    
    Input
    ----------
    X_train : numpy.ndarray, size=(m,n)
    y_train : numpy.ndarray, size=(m,)
    X_test : numpy.ndarray, size=(k,n)
    y_test : numpy.ndarray, size=(k,)
    stratified_kfold : function of KFold() or StratifiedKFold()
    oversampling : oversampling function, for example SMOTE(), ADASYN(). if it's None, then no oversampling strategy is used 
    classifier : Machine learning classifier function, for example XGBClassifer()
    search_spaces : dict, region of hyperparameter. 
        for oversampling, the key and value in this dict should be:'oversampling__<hyperparameter name>': <range of this parameter>
        for classifier, the key and value in this dict should be:'classifier__<hyperparameter name>': <range of this parameter>

    Output
    ----------
    cv_score: float, the cv score of the best estimator
    test_score: float, the test score of the best estimator
    list_best_cv: list, list of the best estimator's validation score on each fold
    '''
    pipeline = imbpipeline(steps = [['oversampling', oversampling],['classifier', classifier]])
    grid_search = GridSearchCV(
        estimator = pipeline,
        param_grid=search_spaces,    
        scoring='f1',
        cv=stratified_kfold,
        n_jobs=-1)
    grid_search.fit(X_train, y_train)
    cv_score= grid_search.best_score_
    test_score= grid_search.score(X_test, y_test)
    best_estimator=pd.DataFrame(grid_search.cv_results_)[pd.DataFrame(grid_search.cv_results_)['rank_test_score']==1]
    orderd_dict_best_params=best_estimator.iloc[0]['params']
    list_best_cv=[]
    for col in best_estimator[best_estimator.index==0]:
        if 'test_score' in col:
            if 'split' in col:
                list_best_cv.append(float(best_estimator.iloc[0:1][col]))
    print(f'Classifier: {str(classifier)}\nOver-sampling strategy: {str(oversampling)}\nCross-validation score: {cv_score}\nTest score: {test_score}\nBest parameters: {str(dict(orderd_dict_best_params))}\n')
    return cv_score,test_score,list_best_cv

def plot_models_comp(models_comparsion):
    '''
    plot of model comparison of cv score
    '''
    models_comparsion_barplot=models_comparsion.melt(id_vars='Model').rename(columns={'variable':'Oversampling','value':'f1-score'})
    fig,_=plt.subplots(figsize=(10, 10))
    sns.barplot(x="Model", y='f1-score',hue='Oversampling',data=models_comparsion_barplot,palette="rocket")
    plt.xlabel('Model',labelpad=10,fontsize='large',fontweight='bold')
    plt.ylabel('F1-score',labelpad=10,fontsize='large',fontweight='bold')
    plt.legend(title='Over-sampling',title_fontsize='large')
    plt.title('')
    plt.show()

def plot_cv_vs_test(cv_score,test_score):
    '''
    plot comparison of cv score and test score of models with best over-sampling
    '''
    test_model, test_score_ = zip(*test_score.items())
    fig,_=plt.subplots(figsize=(10, 10))
    sns.boxplot(data=cv_score,palette='light:#5A9')
    sns.swarmplot(data=cv_score,color='.12',size=8)
    plt.xlabel('Model',labelpad=10,fontsize='large',fontweight='bold')
    plt.ylabel('F1-score',labelpad=10,fontsize='large',fontweight='bold')
    plt.plot(test_model, test_score_,'*',color='r',markersize=14,label='test score')
    plt.legend(loc='upper left',fontsize=12)
    plt.show()