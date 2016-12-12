import os
import csv
from functions import *
import scipy.stats as st
import numpy as np

with open("results_sign_novelDTI_ndcg_AVG.csv", "w") as resFile:
    top_k_size = 10
    resFile.write("\n")
    resFile.write("method;nDCG;t_nDCG;p_nDCG\n" )
    dt = ["gpcr","ic", "nr", "e"]#
    met = [ "blmnii", "wnngip", "netlaprls", "cmf","knn_bprcasq"] #,"knn_bprcasq"
    max_ndcg = 0    
    v_max_ndcg = np.ones(50)  
    for cp in met: #get maximal values for each evaluation metric throughout the evaluated methods   
        vec_ndcg = []
        for dataset in dt: 
       
            
            with open(os.path.join('output','newDTI',cp+'_'+dataset+'_'+str(top_k_size)+'_drug_stats.csv'), 'rb') as csvfile:
                reader = csv.reader(csvfile, delimiter=';', quotechar='"')
                res = np.array(list(reader) ) 
                v_ndcg = res[:,3]                  
                v_ndcg = [s for s in v_ndcg if s != "nan" and s != "ndcg"]
                v_ndcg = [float(d) for d in v_ndcg]
            
            if cp == "knn_bprcasq":
                cp = "knn_inv_bprcasq"
                
            with open(os.path.join('output','newDTI',cp+'_'+dataset+'_'+str(top_k_size)+'_target_stats.csv'), 'rb') as csvfile:
                    reader = csv.reader(csvfile, delimiter=';', quotechar='"')
                    res = np.array(list(reader) ) 
                    v_ndcg2 = res[:,3]                  
                    v_ndcg2 = [s for s in v_ndcg2 if s != "nan" and s != "ndcg"]
                    v_ndcg2 = [float(d) for d in v_ndcg2]    
            vec_ndcg = vec_ndcg + v_ndcg +  v_ndcg2 
            
    avg_ndcg = np.mean(vec_ndcg)
    if avg_ndcg > max_ndcg:
        max_ndcg = avg_ndcg
        v_max_ndcg = vec_ndcg[:]

                   
    for cp in met:  #calculate stat. sign. of other methods vs. the best one 
        cp_ndcg = []
        for dataset in dt: 
            with open(os.path.join('output','newDTI',cp+'_'+dataset+'_'+str(top_k_size)+'_drug_stats.csv'), 'rb') as csvfile:
                reader = csv.reader(csvfile, delimiter=';', quotechar='"')
                res = np.array(list(reader) ) 
                v_ndcg = res[:,3]                  
                v_ndcg = [s for s in v_ndcg if s != "nan" and s != "ndcg"]
                v_ndcg = [float(d) for d in v_ndcg]
                
            if cp == "knn_bprcasq":
                cp = "knn_inv_bprcasq"
                
            with open(os.path.join('output','newDTI',cp+'_'+dataset+'_'+str(top_k_size)+'_target_stats.csv'), 'rb') as csvfile:
                    reader = csv.reader(csvfile, delimiter=';', quotechar='"')
                    res = np.array(list(reader) ) 
                    v_ndcg2 = res[:,3]                  
                    v_ndcg2 = [s for s in v_ndcg2 if s != "nan" and s != "ndcg"]
                    v_ndcg2 = [float(d) for d in v_ndcg2]  
                    
            cp_ndcg = cp_ndcg + v_ndcg +  v_ndcg2  
            
        x1, y1 = st.ttest_rel(v_max_ndcg, cp_ndcg)
        resFile.write(cp+";%.6f;%.9f;%.9f\n" % (np.mean(cp_ndcg), x1, y1/2.0) )
        print dataset,cp, np.mean(cp_ndcg), x1, y1
    

with open("results_sign_novelDTI_recall_AVG.csv", "w") as resFile:
    top_k_size = 10
    resFile.write("\n")
    resFile.write("method;recall;t_recall;p_recall\n" )
    dt = ["gpcr","ic", "nr", "e"]#
    met = [ "blmnii", "wnngip", "netlaprls", "cmf","knn_bprcasq"] #,"knn_bprcasq"
    max_ndcg = 0    
    v_max_ndcg = np.ones(50)  
    for cp in met: #get maximal values for each evaluation metric throughout the evaluated methods   
        vec_ndcg = []
        for dataset in dt: 
       
            
            with open(os.path.join('output','newDTI',cp+'_'+dataset+'_'+str(top_k_size)+'_drug_stats.csv'), 'rb') as csvfile:
                reader = csv.reader(csvfile, delimiter=';', quotechar='"')
                res = np.array(list(reader) ) 
                v_ndcg = res[:,4]                  
                v_ndcg = [s for s in v_ndcg if s != "nan" and s != "recall"]
                v_ndcg = [float(d) for d in v_ndcg]
            
            if cp == "knn_bprcasq":
                cp = "knn_inv_bprcasq"
                
            with open(os.path.join('output','newDTI',cp+'_'+dataset+'_'+str(top_k_size)+'_target_stats.csv'), 'rb') as csvfile:
                    reader = csv.reader(csvfile, delimiter=';', quotechar='"')
                    res = np.array(list(reader) ) 
                    v_ndcg2 = res[:,4]                  
                    v_ndcg2 = [s for s in v_ndcg2 if s != "nan" and s != "recall"]
                    v_ndcg2 = [float(d) for d in v_ndcg2]    
            vec_ndcg = vec_ndcg + v_ndcg +  v_ndcg2 
            
    avg_ndcg = np.mean(vec_ndcg)
    if avg_ndcg > max_ndcg:
        max_ndcg = avg_ndcg
        v_max_ndcg = vec_ndcg[:]

                   
    for cp in met:  #calculate stat. sign. of other methods vs. the best one 
        cp_ndcg = []
        for dataset in dt: 
            with open(os.path.join('output','newDTI',cp+'_'+dataset+'_'+str(top_k_size)+'_drug_stats.csv'), 'rb') as csvfile:
                reader = csv.reader(csvfile, delimiter=';', quotechar='"')
                res = np.array(list(reader) ) 
                v_ndcg = res[:,4]                  
                v_ndcg = [s for s in v_ndcg if s != "nan" and s != "recall"]
                v_ndcg = [float(d) for d in v_ndcg]
                
            if cp == "knn_bprcasq":
                cp = "knn_inv_bprcasq"
                
            with open(os.path.join('output','newDTI',cp+'_'+dataset+'_'+str(top_k_size)+'_target_stats.csv'), 'rb') as csvfile:
                    reader = csv.reader(csvfile, delimiter=';', quotechar='"')
                    res = np.array(list(reader) ) 
                    v_ndcg2 = res[:,4]                  
                    v_ndcg2 = [s for s in v_ndcg2 if s != "nan" and s != "recall"]
                    v_ndcg2 = [float(d) for d in v_ndcg2]  
                    
            cp_ndcg = cp_ndcg + v_ndcg +  v_ndcg2  
            
        x1, y1 = st.ttest_rel(v_max_ndcg, cp_ndcg)
        resFile.write(cp+";%.6f;%.9f;%.9f\n" % (np.mean(cp_ndcg), x1, y1/2.0) )
        print dataset,cp, np.mean(cp_ndcg), x1, y1
                