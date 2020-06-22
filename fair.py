import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as rc


def DP_accuracy (y_hat,A):
  # Demographic Parity
  A , y_hat = np.array(A).reshape(-1) , y_hat.reshape(-1)
  sum_A0 = 0
  sum_A1 = 0
  n_A1 = np.count_nonzero(A)
  n_A0 = len(A) - np.count_nonzero(A)
  for i in range (len(y_hat)):
    sum_A0 += y_hat[i]*(1-A[i])
    sum_A1 += y_hat[i]*A[i]
    accuracy= abs(sum_A0/n_A0 - sum_A1/n_A1)
  return accuracy

def p_rule(y_pred, z_values):
  # Disparate Impact
  y_z_1 =np.array([i for i,j in zip(y_pred,z_values) if i==1 and j==1])
  y_z_0 = np.array([i for i,j in zip(y_pred,z_values) if i==1 and j==0])
  odds = y_z_1.sum() / y_z_0.sum()
  return  np.min([odds, 1/odds]) 

def split(y_pred,y_test,z_test): 
  # spliting into protected and unprotected groups
  pred0 =np.array([i for i,j,k in zip(y_pred,y_test,z_test) if k==0])
  pred1 = np.array([i for i,j,k in zip(y_pred,y_test,z_test) if k==1])
  y_test0 =np.array([j for i,j,k in zip(y_pred,y_test,z_test) if k==0])
  y_test1 = np.array([j for i,j,k in zip(y_pred,y_test,z_test) if k==1])
  return [pred0,y_test0,pred1,y_test1]

def classreport(pred,y_test):
  # classification report
  fp = sum(np.logical_and(y_test == 0, pred == 1)) 
  fn = sum(np.logical_and(y_test == 1, pred == 0)) 
  tp = sum(np.logical_and(y_test == 1, pred == 1)) 
  tn = sum(np.logical_and(y_test== 0, pred == 0))

  fpr = float(fp) / float(fp + tn)
  fnr = float(fn) / float(fn + tp)
  tpr = float(tp) / float(tp + fn)
  tnr = float(tn) / float(tn + fp)
  npv = float(tn) / float(tn + fn)
  ppv = float(tp) / float(tp + fp)
  acc = (tp + tn) / (tp + tn + fp + fn)
  di= {'False Positive': fp,
       'False Negative': fn,
       'True Positive': tp,
       'True Negative': tn,
       'False Positive rate': fpr,
       'False Negative rate': fnr,
       'True Positive rate': tpr,
       'True Negative rate': tnr,
       'npv': npv,
       'ppv': ppv,
      }
  return di

def Get_fair_metric(model,y_pred,y_test,Z_test):
  # Fairness report
  # Split into groups
  pre1,y_test1 = split(y_pred,y_test,Z_test)[0],split(y_pred,y_test,Z_test)[1]
  pre2,y_test2 = split(y_pred,y_test,Z_test)[2],split(y_pred,y_test,Z_test)[3]
  # Get classification report
  a=classreport(pre1,y_test1)
  b=classreport(pre2,y_test2)
  # Calculate demographic Parity
  dp= DP_accuracy (y_pred,Z_test)
  # disparate impact
  dI=p_rule(y_pred, Z_test)
  # ROC accuracy
  aoc = rc(y_test, y_pred)
  # Equalised Opportunity
  Equalised_opportunity= a['True Positive rate']- b['True Positive rate']
  # Equalised Opportunity
  Equalised_odds = a['False Positive rate']- b['False Positive rate']
  # Equalised Odds
  Treatmentequal= (b['False Positive']/b['False Negative'])/(a['False Positive']/a['False Negative'])
  # Predictive equality
  Predictiveequal= a['True Negative rate']- b['True Negative rate']
  # Conditional Use Accuracy
  condUseAcc= (a['ppv']+a['npv'])- (b['ppv']+b['npv'])
  # Overall Use Accuracy
  OVerUseAcc= (a['True Positive rate']+a['True Negative rate'])- (b['True Positive rate']+b['True Negative rate'])
  D={'model': model,'Demographic Parity':dp, 'Disparate Impact':dI, 'Equalised_opportunity': Equalised_opportunity,
     'Equalised_odds': Equalised_odds, 'Treatment equality': Treatmentequal,
     'Predictive equality' : Predictiveequal, 'Conditional Use Accuracy': condUseAcc, 
     'Overall Use Accuracy': OVerUseAcc, 'ROC Accuracy': aoc}
  return D