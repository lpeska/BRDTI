"""
Combined approach for aggregation of raw DTI prediction methods
- the approach works with existing raw results, so you neeed to run the raw methods before the combined approach
- four aggregation methods are available, those are average, max, min and probabilistic sum (S-norm from fuzzy theory; S(a,b) = a+b - a*b

Parameters:
    aggregation = aggregation method
    dataset = evaluated dataset
    cvType = evaluated cross-validation type
    methods = considered raw methods

"""
from __future__ import division
import os
import csv
import numpy as np  
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
from functions import normalized_discounted_cummulative_gain
from functions import write_metric_vector_to_file

class CombinedApproach(object):

    def __init__(self,args):

        self.aggregation = args["aggregation"]
        self.dataset = args["dataset"]
        self.cvType = args["cvType"]
        
        self.methods = ["blmnii","wnngip","netlaprls","cmf"]#,"brdti"

    
    def evaluate(self):        
        #collect results of all methods
        i = 0
        with open(os.path.join('output','rawResults', self.methods[0]+"_res_"+str(self.cvType)+"_"+self.dataset+".csv"), 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar='"')
            m_results = np.array(list(reader))
            n_data = len(m_results)-1
            drugs = m_results[2:,0] 
            targets = m_results[2:,1]
            labels = m_results[2:,2] 
        
        pred = np.zeros([n_data,len(self.methods)+3], dtype='S20')
        pred[:-1,0] = drugs 
        pred[:-1,1] = targets 
        pred[:-1,2] = labels 

        i = 3
        for m in self.methods:
            with open(os.path.join('output','rawResults', m+"_res_"+str(self.cvType)+"_"+self.dataset+".csv"), 'rb') as csvfile:
                reader = csv.reader(csvfile, delimiter=';', quotechar='"')
                m_results = np.array(list(reader)) 
            
            predictions = m_results[2:,3]  
            pred[:-1,i] = predictions
            i = i+1 
        #print pred   
        auc_arr, aupr_arr, ndcg_arr, ndcg_inv_arr = [],[],[],[]
        test_data,test_label,scores = [],[],[]

        for row in pred:
            if row[0] == "":
                #end of last CV, produce results
                aupr, auc, ndcg, ndcg_inv = self.evaluation(np.array(test_data),test_label,scores)
                aupr_arr += [aupr]
                auc_arr += [auc]                
                ndcg_arr += [ndcg]
                ndcg_inv_arr += [ndcg_inv]
                test_data,test_label,scores = [],[],[]
                #print aupr, auc, ndcg, ndcg_inv
            else:
                test_data += [(int(row[0]),int(row[1]))]
                test_label += [int(float(row[2]))]
                scores += [self.aggregate_scores(row[3:])]
        
        output_dir = 'output'         
        write_metric_vector_to_file(auc_arr, os.path.join(output_dir, "combined_auc_cvs"+str(self.cvType)+"_"+self.dataset+".txt"))
        write_metric_vector_to_file(aupr_arr, os.path.join(output_dir, "combined_aupr_cvs"+str(self.cvType)+"_"+self.dataset+".txt"))            
        write_metric_vector_to_file(ndcg_arr, os.path.join(output_dir, "combined_ndcg_cvs"+str(self.cvType)+"_"+self.dataset+".txt"))    
                 
        print np.mean(auc_arr),np.mean(aupr_arr),np.mean(ndcg_arr),np.mean(ndcg_inv_arr)
        
    def aggregate_scores(self, scores):
        fscores = [float(s) for s in scores]
        if self.aggregation == "avg":
            return np.mean(fscores)
        elif self.aggregation == "max":
            return np.amax(fscores)
        elif self.aggregation == "min":
            return np.amin(fscores)
        elif self.aggregation == "softmax":
            bounded_fscores = [np.amax([0, np.amin([1,s]) ]) for s in fscores]
            sc = 0
            for bsc in bounded_fscores:
                sc = (sc+bsc) - (sc*bsc)            
            return sc
        


    def evaluation(self, test_data, test_label,scores):       
        
        x, y = test_data[:, 0], test_data[:, 1]
        test_data_T = np.column_stack((y,x))
        
        ndcg = normalized_discounted_cummulative_gain(test_data, np.array(test_label), np.array(scores))
        ndcg_inv = normalized_discounted_cummulative_gain(test_data_T, np.array(test_label), np.array(scores))
        prec, rec, thr = precision_recall_curve(test_label, scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, scores)
        auc_val = auc(fpr, tpr)
        
        return aupr_val, auc_val, ndcg, ndcg_inv
    
    

        
    
    def __str__(self):
        return "Model: BPRCA, factors:%s, learningRate:%s,  max_iters:%s, lambda_bias:%s, lambda_u:%s, lambda_i:%s, lambda_ca_u:%s, lambda_ca_i:%s, simple_predict:%s" % (self.D, self.learning_rate, self.max_iters, self.bias_regularization, self.user_regularization,  self.positive_item_regularization, self.user_cb_alignment_regularization, self.item_cb_alignment_regularization, self.simple_predict)
    



if __name__ == '__main__':
    agg = ["avg","max","softmax","min"]
    datasets = ["gpcr","ic","nr","e"]
    cvs = [1,2,3]
    for a in agg:
        for d in datasets:
            for c in cvs:
                print a,d,str(c)
                
                args = {'aggregation':a, 'dataset':d,'cvType' : c }
                c = CombinedApproach(args)     
                c.evaluate()
    
    
    
    