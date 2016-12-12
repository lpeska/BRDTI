import os
import csv
from functions import *
import scipy.stats as st
import numpy as np

with open("results_sign_novelDTI_drugs_ndcg.csv", "w") as resFile:
        top_k_size = 10
        resFile.write("\n")
        resFile.write("dataset;method;nDCG;t_nDCG;p_nDCG\n" )
        dt = ["gpcr","ic", "nr", "e"]#
        met = [ "blmnii", "wnngip", "netlaprls", "cmf","knn_bprcasq"] #,"knn_bprcasq"
        for dataset in dt: 
            resFile.write("\n")
            max_ndcg = 0    
            v_max_ndcg = np.ones(50)            
            for cp in met: #get maximal values for each evaluation metric throughout the evaluated methods
                with open(os.path.join('output','newDTI',cp+'_'+dataset+'_'+str(top_k_size)+'_drug_stats.csv'), 'rb') as csvfile:
                    reader = csv.reader(csvfile, delimiter=';', quotechar='"')
                    res = np.array(list(reader) ) 
                    v_ndcg = res[:,3]                  
                    v_ndcg = [s for s in v_ndcg if s != "nan" and s != "ndcg"]
                    v_ndcg = [float(d) for d in v_ndcg]
                
                
                avg_ndcg = np.mean(v_ndcg)
                if avg_ndcg > max_ndcg:
                    max_ndcg = avg_ndcg
                    v_max_ndcg = v_ndcg[:]

                   
            for cp in met:  #calculate stat. sign. of other methods vs. the best one 
                with open(os.path.join('output','newDTI',cp+'_'+dataset+'_'+str(top_k_size)+'_drug_stats.csv'), 'rb') as csvfile:
                    reader = csv.reader(csvfile, delimiter=';', quotechar='"')
                    res = np.array(list(reader) ) 
                    cp_ndcg = res[:,3]                  
                    cp_ndcg = [s for s in cp_ndcg if s != "nan" and s != "ndcg"]
                    cp_ndcg = [float(d) for d in cp_ndcg]
            
                x1, y1 = st.ttest_rel(v_max_ndcg, cp_ndcg)
                resFile.write(dataset+";"+cp+";%.6f;%.9f;%.9f\n" % (np.mean(cp_ndcg), x1, y1/2.0) )
                print dataset,cp, np.mean(cp_ndcg), x1, y1
            print ""
            

with open("results_sign_novelDTI_targets_ndcg.csv", "w") as resFile:
        top_k_size = 10
        resFile.write("\n")
        resFile.write("dataset;method;nDCG;t_nDCG;p_nDCG\n" )
        dt = ["gpcr","ic", "nr", "e"]#
        met = [ "blmnii", "wnngip", "netlaprls", "cmf","knn_inv_bprcasq"] #,"knn_bprcasq"
        for dataset in dt: 
            resFile.write("\n")
            max_ndcg = 0    
            v_max_ndcg = np.ones(50)            
            for cp in met: #get maximal values for each evaluation metric throughout the evaluated methods
                with open(os.path.join('output','newDTI',cp+'_'+dataset+'_'+str(top_k_size)+'_target_stats.csv'), 'rb') as csvfile:
                    reader = csv.reader(csvfile, delimiter=';', quotechar='"')
                    res = np.array(list(reader) ) 
                    v_ndcg = res[:,3]                  
                    v_ndcg = [s for s in v_ndcg if s != "nan" and s != "ndcg"]
                    v_ndcg = [float(d) for d in v_ndcg]
                
                
                avg_ndcg = np.mean(v_ndcg)
                if avg_ndcg > max_ndcg:
                    max_ndcg = avg_ndcg
                    v_max_ndcg = v_ndcg[:]

                   
            for cp in met:  #calculate stat. sign. of other methods vs. the best one 
                with open(os.path.join('output','newDTI',cp+'_'+dataset+'_'+str(top_k_size)+'_target_stats.csv'), 'rb') as csvfile:
                    reader = csv.reader(csvfile, delimiter=';', quotechar='"')
                    res = np.array(list(reader) ) 
                    cp_ndcg = res[:,3]                  
                    cp_ndcg = [s for s in cp_ndcg if s != "nan" and s != "ndcg"]
                    cp_ndcg = [float(d) for d in cp_ndcg]
            
                x1, y1 = st.ttest_rel(v_max_ndcg, cp_ndcg)
                resFile.write(dataset+";"+cp+";%.6f;%.9f;%.9f\n" % (np.mean(cp_ndcg), x1, y1/2.0) )
                print dataset,cp, np.mean(cp_ndcg), x1, y1
            print ""
            

with open("results_sign_novelDTI_drugs_recall.csv", "w") as resFile:
        top_k_size = 10
        resFile.write("\n")
        resFile.write("dataset;method;recall;t_recall;p_recall\n" )
        dt = ["gpcr","ic", "nr", "e"]#
        met = [ "blmnii", "wnngip", "netlaprls", "cmf","knn_bprcasq"] #,"knn_bprcasq"
        for dataset in dt: 
            resFile.write("\n")
            max_ndcg = 0    
            v_max_ndcg = np.ones(50)            
            for cp in met: #get maximal values for each evaluation metric throughout the evaluated methods
                with open(os.path.join('output','newDTI',cp+'_'+dataset+'_'+str(top_k_size)+'_drug_stats.csv'), 'rb') as csvfile:
                    reader = csv.reader(csvfile, delimiter=';', quotechar='"')
                    res = np.array(list(reader) ) 
                    v_ndcg = res[:,4]                  
                    v_ndcg = [s for s in v_ndcg if s != "nan" and s != "recall"]
                    v_ndcg = [float(d) for d in v_ndcg]
                
                
                avg_ndcg = np.mean(v_ndcg)
                if avg_ndcg > max_ndcg:
                    max_ndcg = avg_ndcg
                    v_max_ndcg = v_ndcg[:]

                   
            for cp in met:  #calculate stat. sign. of other methods vs. the best one 
                with open(os.path.join('output','newDTI',cp+'_'+dataset+'_'+str(top_k_size)+'_drug_stats.csv'), 'rb') as csvfile:
                    reader = csv.reader(csvfile, delimiter=';', quotechar='"')
                    res = np.array(list(reader) ) 
                    cp_ndcg = res[:,4]                  
                    cp_ndcg = [s for s in cp_ndcg if s != "nan" and s != "recall"]
                    cp_ndcg = [float(d) for d in cp_ndcg]
            
                x1, y1 = st.ttest_rel(v_max_ndcg, cp_ndcg)
                resFile.write(dataset+";"+cp+";%.6f;%.9f;%.9f\n" % (np.mean(cp_ndcg), x1, y1/2.0) )
                print dataset,cp, np.mean(cp_ndcg), x1, y1
            print ""
            

with open("results_sign_novelDTI_targets_recall.csv", "w") as resFile:
        top_k_size = 10
        resFile.write("\n")
        resFile.write("dataset;method;recall;t_recall;p_recall\n" )
        dt = ["gpcr","ic", "nr", "e"]#
        met = [ "blmnii", "wnngip", "netlaprls", "cmf","knn_inv_bprcasq"] #,"knn_bprcasq"
        for dataset in dt: 
            resFile.write("\n")
            max_ndcg = 0    
            v_max_ndcg = np.ones(50)            
            for cp in met: #get maximal values for each evaluation metric throughout the evaluated methods
                with open(os.path.join('output','newDTI',cp+'_'+dataset+'_'+str(top_k_size)+'_target_stats.csv'), 'rb') as csvfile:
                    reader = csv.reader(csvfile, delimiter=';', quotechar='"')
                    res = np.array(list(reader) ) 
                    v_ndcg = res[:,4]                  
                    v_ndcg = [s for s in v_ndcg if s != "nan" and s != "recall"]
                    v_ndcg = [float(d) for d in v_ndcg]
                
                
                avg_ndcg = np.mean(v_ndcg)
                if avg_ndcg > max_ndcg:
                    max_ndcg = avg_ndcg
                    v_max_ndcg = v_ndcg[:]

                   
            for cp in met:  #calculate stat. sign. of other methods vs. the best one 
                with open(os.path.join('output','newDTI',cp+'_'+dataset+'_'+str(top_k_size)+'_target_stats.csv'), 'rb') as csvfile:
                    reader = csv.reader(csvfile, delimiter=';', quotechar='"')
                    res = np.array(list(reader) ) 
                    cp_ndcg = res[:,4]                  
                    cp_ndcg = [s for s in cp_ndcg if s != "nan" and s != "recall"]
                    cp_ndcg = [float(d) for d in cp_ndcg]
            
                x1, y1 = st.ttest_rel(v_max_ndcg, cp_ndcg)
                resFile.write(dataset+";"+cp+";%.6f;%.9f;%.9f\n" % (np.mean(cp_ndcg), x1, y1/2.0) )
                print dataset,cp, np.mean(cp_ndcg), x1, y1
            print ""
            