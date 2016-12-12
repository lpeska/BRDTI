
from functions import *
import scipy.stats as st
import numpy as np
with open("results_sign_avg.csv", "w") as resFile:
    resFile.write("\n")
    resFile.write("method;AUC;AUPR;per_drug nDCG;per_target nDCG;t_AUC;p_AUC;t_AUPR;p_AUPR;t_pd_nDCG;p_pd_nDCG;t_pt_nDCG;p_pt_nDCG\n" )
    dt = ["gpcr","ic", "nr", "e"]
    met = [ "blmnii", "wnngip", "netlaprls", "cmf","knn_bprcasq","knn_inv_bprcasq"]
    resFile.write("\n")
    max_auc = 0
    max_aupr = 0
    max_ndcg = 0
    max_ndcg_inv = 0    
    v_max_auc = np.ones(60)
    v_max_aupr = np.ones(60)
    v_max_ndcg = np.ones(60)
    v_max_ndcg_inv = np.ones(60)
    
    for cp in met:
        v_auc, v_aupr, v_ndcg, v_ndcg_inv  = [],[],[],[]
        for cv in ["1", "2", "3"]:                          
            for dataset in dt: 
             #get maximal values for each evaluation metric throughout the evaluated methods
                v_auc += load_metric_vector("output/"+cp+"_auc_cvs"+cv+"_"+dataset+".txt").tolist()
                v_aupr += load_metric_vector("output/"+cp+"_aupr_cvs"+cv+"_"+dataset+".txt").tolist()
                if cp !="knn_inv_bprcasq":
                    v_ndcg += load_metric_vector("output/"+cp+"_ndcg_cvs"+cv+"_"+dataset+".txt").tolist()
                if cp !="knn_bprcasq":
                    v_ndcg_inv += load_metric_vector("output/"+cp+"_ndcg_inv_cvs"+cv+"_"+dataset+".txt").tolist()
                
        avg_auc = np.mean(np.asarray(v_auc))
        if avg_auc > max_auc:
            max_auc = avg_auc
            v_max_auc = np.asarray(v_auc)
            
        avg_aupr = np.mean(np.asarray(v_aupr))
        if avg_aupr > max_aupr:
            max_aupr = avg_aupr
            v_max_aupr = np.asarray(v_aupr)
            
        avg_ndcg = np.mean(np.asarray(v_ndcg))
        if avg_ndcg > max_ndcg:
            max_ndcg = avg_ndcg
            v_max_ndcg = np.asarray(v_ndcg)
            
        avg_ndcg_inv = np.mean(np.asarray(v_ndcg_inv))
        if avg_ndcg_inv > max_ndcg_inv:
            max_ndcg_inv = avg_ndcg_inv
            v_max_ndcg_inv = np.asarray(v_ndcg_inv)
                    
    for cp in met:  #calculate stat. sign. of other methods vs. the best one
        cp_auc, cp_aupr, cp_ndcg, cp_ndcg_inv  = [],[],[],[]
        
        for cv in ["1", "2", "3"]:                               
            for dataset in dt: 
                cp_auc += load_metric_vector("output/"+cp+"_auc_cvs"+cv+"_"+dataset+".txt").tolist()
                cp_aupr += load_metric_vector("output/"+cp+"_aupr_cvs"+cv+"_"+dataset+".txt").tolist()
                cp_ndcg += load_metric_vector("output/"+cp+"_ndcg_cvs"+cv+"_"+dataset+".txt").tolist()
                cp_ndcg_inv += load_metric_vector("output/"+cp+"_ndcg_inv_cvs"+cv+"_"+dataset+".txt").tolist()
                
        print(v_max_auc.shape, np.asarray(cp_auc).shape) 
        print(v_max_aupr.shape, np.asarray(cp_aupr).shape)  
        print(v_max_ndcg.shape, np.asarray(cp_ndcg).shape)  
        print(v_max_ndcg_inv.shape, np.asarray(cp_ndcg_inv).shape)  
        
        x1, y1 = st.ttest_rel(v_max_auc, np.asarray(cp_auc))
        x2, y2 = st.ttest_rel(v_max_aupr, np.asarray(cp_aupr))
        x3, y3 = st.ttest_rel(v_max_ndcg, np.asarray(cp_ndcg))
        x4, y4 = st.ttest_rel(v_max_ndcg_inv, np.asarray(cp_ndcg_inv))
        resFile.write(cp+";%.6f;%.6f;%.6f;%.6f;%.9f;%.9f;%.9f;%.9f;%.9f;%.9f;%.9f;%.9f\n" % (np.mean(cp_auc), np.mean(cp_aupr), np.mean(cp_ndcg), np.mean(cp_ndcg_inv), x1, y1/2.0, x2, y2/2.0, x3, y3/2.0, x4, y4/2.0) )
        print cp, np.mean(cp_auc), np.mean(cp_aupr), np.mean(cp_ndcg), np.mean(cp_ndcg_inv), x1, y1, x2, y2, x3, y3, x4, y4
    print ""
