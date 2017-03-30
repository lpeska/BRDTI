
from functions import *
import scipy.stats as st
import numpy as np
with open("results_sign.csv", "w") as resFile:
    for cv in ["1"]: #"1", "2", "3"
        print "CVS:"+cv
        resFile.write("\n")
        resFile.write("CVS;dataset;method;AUC;AUPR;per_drug nDCG;t_AUC;p_AUC;t_AUPR;p_AUPR;t_pd_nDCG;p_pd_nDCG\n" )
        dt = ["gpcr","ic", "nr", "e" , "e","metz"]#"gpcr","ic", "nr", "e" , "e"
        
        met = ["blmnii", "wnngip", "netlaprls", "cmf",  "brdti"] 
        for dataset in dt: 
            resFile.write("\n")
            max_auc = 0
            max_aupr = 0
            max_ndcg = 0  
            v_max_auc = np.ones(50)
            v_max_aupr = np.ones(50)
            v_max_ndcg = np.ones(50)
            
            for cp in met: #get maximal values for each evaluation metric throughout the evaluated methods
                v_auc = load_metric_vector("output/"+cp+"_auc_cvs"+cv+"_"+dataset+".txt")
                v_aupr = load_metric_vector("output/"+cp+"_aupr_cvs"+cv+"_"+dataset+".txt")
                v_ndcg = load_metric_vector("output/"+cp+"_ndcg_cvs"+cv+"_"+dataset+".txt")

                avg_auc = np.mean(v_auc)
                if avg_auc > max_auc:
                    max_auc = avg_auc
                    v_max_auc = v_auc[:]
                avg_aupr = np.mean(v_aupr)
                if avg_aupr > max_aupr:
                    max_aupr = avg_aupr
                    v_max_aupr = v_aupr[:]
                avg_ndcg = np.mean(v_ndcg)
                if avg_ndcg > max_ndcg:
                    max_ndcg = avg_ndcg
                    v_max_ndcg = v_ndcg[:]

            for cp in met:  #calculate stat. sign. of other methods vs. the best one
                cp_auc = load_metric_vector("output/"+cp+"_auc_cvs"+cv+"_"+dataset+".txt")
                cp_aupr = load_metric_vector("output/"+cp+"_aupr_cvs"+cv+"_"+dataset+".txt")
                cp_ndcg = load_metric_vector("output/"+cp+"_ndcg_cvs"+cv+"_"+dataset+".txt")
                x1, y1 = st.ttest_rel(v_max_auc, cp_auc)
                x2, y2 = st.ttest_rel(v_max_aupr, cp_aupr)
                x3, y3 = st.ttest_rel(v_max_ndcg, cp_ndcg)
                
                resFile.write("CVS:"+cv+";"+dataset+";"+cp+";%.6f;%.6f;%.6f;%.9f;%.9f;%.9f;%.9f;%.9f;%.9f\n" % (np.mean(cp_auc), np.mean(cp_aupr), np.mean(cp_ndcg), x1, y1/2.0, x2, y2/2.0, x3, y3/2.0) )
                print dataset, cp, np.mean(cp_auc), np.mean(cp_aupr), np.mean(cp_ndcg), x1, y1, x2, y2, x3, y3
            print ""
