
import os
import sys
import time
import getopt
import cv_eval
from functions import *
from netlaprls import NetLapRLS
from blmnii import BLMNII
from wnngip import WNNGIP
from cmf import CMF
from brdti import BRDTI

from eval_new_DTI_prediction import *

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "m:d:f:c:s:o:n:p", ["method=", "dataset=", "data-dir=", "cvs=", "specify-arg=", "method-options=", "predict-num=", "output-dir=", ])
    except getopt.GetoptError:
        sys.exit()

    data_dir = 'data'
    output_dir = 'output'
    cvs, sp_arg, model_settings, predict_num = 1, 1, [], 0

    seeds = [7771, 8367, 22, 1812, 4659]
    seedsOptPar = [156]
    # seeds = np.random.choice(10000, 5, replace=False)
    for opt, arg in opts:
        if opt == "--method":
            method = arg
        if opt == "--dataset":
            dataset = arg
        if opt == "--data-dir":
            data_dir = arg
        if opt == "--output-dir":
            output_dir = arg
        if opt == "--cvs":
            cvs = int(arg)
        if opt == "--specify-arg":
            sp_arg = int(arg)
        if opt == "--method-options":
            model_settings = [s.split('=') for s in str(arg).split()]
        if opt == "--predict-num":
            predict_num = int(arg)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.isdir(os.path.join(output_dir,"optPar")):
        os.makedirs(os.path.join(output_dir,"optPar"))    
        
    # default parameters for each methods
    if (method == 'brdti') | (method == 'inv_brdti') :
        args = {
            'D':100,
            'learning_rate':0.1,
            'max_iters' : 100,   
            'simple_predict' :False, 
            'bias_regularization':1,                 
            'global_regularization':10**(-2),  
            "cbSim": "knn",
            'cb_alignment_regularization_user' :1,                 
            'cb_alignment_regularization_item' :1}

    if method == 'netlaprls':
        args = {'gamma_d': 10, 'gamma_t': 10, 'beta_d': 1e-5, 'beta_t': 1e-5}
    if method == 'blmnii':
        args = {'alpha': 0.7, 'gamma': 1.0, 'sigma': 1.0, 'avg': False}
    if method == 'wnngip':
        args = {'T': 0.8, 'sigma': 1.0, 'alpha': 0.8}
    if method == 'cmf':
        args = {'K': 100, 'lambda_l': 0.5, 'lambda_d': 0.125, 'lambda_t': 0.125, 'max_iter': 100}
     
    #print(model_settings)    
    for key, val in model_settings:
        args[key] = float(eval(val))

    intMat, drugMat, targetMat = load_data_from_file(dataset, os.path.join(data_dir, 'datasets'))
    drug_names, target_names = get_drugs_targets_names(dataset, os.path.join(data_dir, 'datasets'))
    
    invert = 0    
    if (method == 'inv_brdti')  : 
        invert = 1
        
    if predict_num == 0:
        if cvs == 1:  # CV setting CVS1
            X, D, T, cv = intMat, drugMat, targetMat, 1             
                
        if cvs == 2:  # CV setting CVS2
            X, D, T, cv = intMat, drugMat, targetMat, 0
                
        if cvs == 3:  # CV setting CVS3
            X, D, T, cv = intMat.T, targetMat, drugMat, 0 
        

            
        cv_data = cross_validation(X, seeds, cv, invert)
        cv_data_optimize_params = cross_validation(X, seedsOptPar, cv, invert, num=5)

        
    if sp_arg == 0 and predict_num == 0:
        if (method == 'brdti'):
            cv_eval.brdti_cv_eval(method, dataset,output_dir, cv_data_optimize_params, X, D, T, cvs, args)                             
        if (method == 'inv_brdti'):
            cv_eval.brdti_cv_eval(method, dataset,output_dir, cv_data_optimize_params, X.T, T, D, cvs, args) 
        
        if method == 'netlaprls':
            cv_eval.netlaprls_cv_eval(method, dataset,output_dir, cv_data_optimize_params, X, D, T, cvs, args)
        if method == 'blmnii':
            cv_eval.blmnii_cv_eval(method, dataset,output_dir, cv_data_optimize_params, X, D, T, cvs, args)
        if method == 'wnngip':
            cv_eval.wnngip_cv_eval(method, dataset,output_dir, cv_data_optimize_params, X, D, T, cvs, args)        
        if method == 'cmf':
            cv_eval.cmf_cv_eval(method, dataset,output_dir, cv_data_optimize_params, X, D, T, cvs, args)
    

    if sp_arg == 1 or predict_num > 0:
        tic = time.clock()
        if (method == 'brdti')|(method == 'inv_brdti'):
            model = BRDTI(args)       
        if method == 'netlaprls':
            model = NetLapRLS(gamma_d=args['gamma_d'], gamma_t=args['gamma_t'], beta_d=args['beta_t'], beta_t=args['beta_t'])
        if method == 'blmnii':
            model = BLMNII(alpha=args['alpha'], gamma=args['gamma'], sigma=args['sigma'], avg=args['avg'])
        if method == 'wnngip':
            model = WNNGIP(T=args['T'], sigma=args['sigma'], alpha=args['alpha'])        
        if method == 'cmf':
            model = CMF(K=args['K'], lambda_l=args['lambda_l'], lambda_d=args['lambda_d'], lambda_t=args['lambda_t'], max_iter=args['max_iter'])
        cmd = str(model)
        
        #predict hidden part of the current datasets
        if predict_num == 0:
            print "Dataset:"+dataset+" CVS:"+str(cvs)+"\n"+cmd
            if (method == 'inv_brdti') : 
                aupr_vec, auc_vec, ndcg_inv_vec, ndcg_vec, results = train(model, cv_data, X.T, T, D)
            else:
                aupr_vec, auc_vec, ndcg_vec, ndcg_inv_vec, results = train(model, cv_data, X, D, T)
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
            ndcg_avg, ndcg_conf = mean_confidence_interval(ndcg_vec)
            ndcg_inv_avg, ndcg_inv_conf = mean_confidence_interval(ndcg_inv_vec)
            
            resfile = os.path.join('output','rawResults', method+"_res_"+str(cvs)+"_"+dataset+".csv")
            outd = open(resfile, "w")
            outd.write(('drug;target;true;predict\n'))
            
            for r in results:
                outd.write('%s;%s;%s;%s\n' % (r[0],r[1],r[2],r[3]) )
            
            print "auc:%.6f, aupr: %.6f, ndcg: %.6f, ndcg_inv: %.6f, auc_conf:%.6f, aupr_conf:%.6f, ndcg_conf:%.6f, ndcg_inv_conf:%.6f, Time:%.6f" % (auc_avg, aupr_avg, ndcg_avg, ndcg_inv_avg, auc_conf, aupr_conf, ndcg_conf, ndcg_inv_conf, time.clock()-tic)
            write_metric_vector_to_file(auc_vec, os.path.join(output_dir, method+"_auc_cvs"+str(cvs)+"_"+dataset+".txt"))
            write_metric_vector_to_file(aupr_vec, os.path.join(output_dir, method+"_aupr_cvs"+str(cvs)+"_"+dataset+".txt"))            
            write_metric_vector_to_file(ndcg_vec, os.path.join(output_dir, method+"_ndcg_cvs"+str(cvs)+"_"+dataset+".txt"))
            write_metric_vector_to_file(ndcg_inv_vec, os.path.join(output_dir, method+"_ndcg_inv_cvs"+str(cvs)+"_"+dataset+".txt"))
        
        #predict novel DTIs    
        elif predict_num > 0:
            print "Dataset:"+dataset+"\n"+cmd
            seed = 376
            if invert: #predicting drugs for targets
                model.fix_model(intMat.T, intMat.T, targetMat, drugMat, seed)
                npa = newDTIPrediction()
                x, y = np.where(intMat == 0)
                scores = model.predict_scores(zip(y, x), 1)
                sz = np.array(zip(x,y,scores))    
                
            else: #predicting targets for drugs
                model.fix_model(intMat, intMat, drugMat, targetMat, seed)
                npa = newDTIPrediction()
                x, y = np.where(intMat == 0)
                scores = model.predict_scores(zip(x, y), 1)
                sz = np.array(zip(x,y,scores))
                
            ndcg_d, ndcg_t, recall_d, recall_t = npa.verify_novel_interactions(method, dataset, sz, predict_num, drug_names, target_names)
            
            st_file= os.path.join('output/newDTI', "_".join([dataset,str(predict_num), "stats.csv"]))
            out = open(st_file, "a")
            out.write(('%s;%f;%f;%f;%f\n' % (method,ndcg_d, ndcg_t, recall_d, recall_t)))

            

if __name__ == "__main__":  
      
    """  
    main(['--method=blmnii', '--dataset=davis', '--cvs=1', '--specify-arg=1', '--method-opt=alpha=0.6' ])    
    main(['--method=brdti', '--dataset=gpcr', '--cvs=1', '--specify-arg=0'])
    """    
