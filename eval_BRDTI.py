
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
            'learning_rate':0.05,
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
    main(['--method=knn_bprcasqnb', '--dataset=gpcr', '--cvs=1', '--specify-arg=0'])    
    main(['--method=knn_bprcasqnb', '--dataset=ic', '--cvs=1', '--specify-arg=0'])        
    main(['--method=knn_bprcasqnb', '--dataset=nr', '--cvs=1', '--specify-arg=0'])       
  
    main(['--method=knn_bprcasqo', '--dataset=gpcr', '--cvs=1', '--specify-arg=0'])    
    main(['--method=knn_bprcasqo', '--dataset=ic', '--cvs=1', '--specify-arg=0'])        
    main(['--method=knn_bprcasqo', '--dataset=nr', '--cvs=1', '--specify-arg=0'])  
    
    main(['--method=bprcasq', '--dataset=gpcr', '--cvs=1', '--specify-arg=0'])    
    main(['--method=bprcasq', '--dataset=ic', '--cvs=1', '--specify-arg=0'])        
    main(['--method=bprcasq', '--dataset=nr', '--cvs=1', '--specify-arg=0'])  
    
    main(['--method=knn_bprcaest', '--dataset=gpcr', '--cvs=1', '--specify-arg=0'])    
    main(['--method=knn_bprcaest', '--dataset=ic', '--cvs=1', '--specify-arg=0'])        
    main(['--method=knn_bprcaest', '--dataset=nr', '--cvs=1', '--specify-arg=0'])    
    
    main(['--method=bpr', '--dataset=gpcr', '--cvs=1', '--specify-arg=0'])    
    main(['--method=bpr', '--dataset=ic', '--cvs=1', '--specify-arg=0'])        
    main(['--method=bpr', '--dataset=nr', '--cvs=1', '--specify-arg=0']) 
    
    
    main(['--method=bpr', '--dataset=gpcr', '--cvs=1', '--method-opt=D=100 global_regularization=0.05 cb_alignment_regularization_user=0.0 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])    
    main(['--method=bpr', '--dataset=ic', '--cvs=1', '--method-opt=D=100 global_regularization=0.05 cb_alignment_regularization_user=0.0 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])       
    main(['--method=bpr', '--dataset=nr', '--cvs=1', '--method-opt=D=100 global_regularization=0.05 cb_alignment_regularization_user=0.0 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])      
  
    main(['--method=knn_bprcaest', '--dataset=gpcr', '--cvs=1', '--method-opt=D=100 global_regularization=0.05 cb_alignment_regularization_user=0.0 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])   
    main(['--method=knn_bprcaest', '--dataset=ic', '--cvs=1', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.0 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])     
    main(['--method=knn_bprcaest', '--dataset=nr', '--cvs=1', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.0 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
           
    main(['--method=knn_bprcasqnb', '--dataset=gpcr', '--cvs=1', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=0 cb_alignment_regularization_item=1'])   
    main(['--method=knn_bprcasqnb', '--dataset=ic', '--cvs=1', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=0 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprcasqnb', '--dataset=nr', '--cvs=1', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=0 cb_alignment_regularization_item=1'])    
  
    main(['--method=knn_bprcasqo', '--dataset=gpcr', '--cvs=1', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.9 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])    
    main(['--method=knn_bprcasqo', '--dataset=ic', '--cvs=1', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])     
    main(['--method=knn_bprcasqo', '--dataset=nr', '--cvs=1', '--method-opt=D=100 global_regularization=0.05 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    
    main(['--method=bprcasq', '--dataset=gpcr', '--cvs=1', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.9 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])   
    main(['--method=bprcasq', '--dataset=ic', '--cvs=1', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.9 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])       
    main(['--method=bprcasq', '--dataset=nr', '--cvs=1', '--method-opt=D=100 global_regularization=0.05 cb_alignment_regularization_user=0.1 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])  
    

    
     
    
    main(['--method=bpr', '--dataset=e', '--cvs=1', '--specify-arg=0'])  
    
    main(['--method=knn_bprcaest', '--dataset=e', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_bprca', '--dataset=e', '--cvs=1', '--specify-arg=0'])
    main(['--method=bprcasq', '--dataset=e', '--cvs=1', '--specify-arg=0'])
    
    main(['--method=bpr', '--dataset=e', '--cvs=1', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1'])

    main(['--method=knn_bprcasqnb', '--dataset=e', '--cvs=1', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.9 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprcasqo', '--dataset=e', '--cvs=1', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.9 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])

    #######################################predicting novel DTIs##########################
    
    main(['--method=cmf', '--dataset=gpcr', '--predict-num=5', '--method-opt=lambda_d=0.125 lambda_l=0.5 lambda_t=0.03125 K=100'])
    main(['--method=cmf', '--dataset=ic', '--predict-num=5', '--method-opt=lambda_d=0.0625 lambda_l=1 lambda_t=0.125 K=100'])
    main(['--method=cmf', '--dataset=nr', '--predict-num=5', '--method-opt=lambda_d=0.125 lambda_l=0.25 lambda_t=0.03125 K=100'])
    
    
    
    main(['--method=blmnii', '--dataset=gpcr', '--predict-num=20', '--method-opt=alpha=1.0'])
    main(['--method=wnngip', '--dataset=gpcr', '--predict-num=20', '--method-opt=alpha=0.9 T=0.2'])
    main(['--method=netlaprls', '--dataset=gpcr', '--predict-num=20', '--method-opt=beta_d=10**(-6) beta_t=10**(-6) gamma_d=100 gamma_t=100'])
    
    main(['--method=knn_bprcasq', '--dataset=gpcr', '--predict-num=20', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprcasq', '--dataset=gpcr', '--predict-num=20', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=nrlmf', '--dataset=gpcr', '--predict-num=20', '--method-opt=alpha=2 lambda_d=0.5 lambda_t=0.5 num_factors=100 beta=0.5 theta=0.5'])
        
    main(['--method=blmnii', '--dataset=ic', '--predict-num=20', '--method-opt=alpha=0.8'])
    main(['--method=wnngip', '--dataset=ic', '--predict-num=20', '--method-opt=alpha=0.9 T=0.3'])
    main(['--method=netlaprls', '--dataset=ic', '--predict-num=20', '--method-opt=beta_d=10**(-6) gamma_d=10 beta_t=10**(-6) gamma_t=10'])
    
    main(['--method=knn_bprcasq', '--dataset=ic', '--predict-num=20', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprcasq', '--dataset=ic', '--predict-num=20', '--method-opt=D=50 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=nrlmf', '--dataset=ic', '--predict-num=20','--method-opt=alpha=1 lambda_d=1 lambda_t=1 num_factors=100 beta=2**(-1) theta=2**(-1)'])
    
    main(['--method=blmnii', '--dataset=nr', '--predict-num=15', '--method-opt=alpha=0.9'])
    main(['--method=wnngip', '--dataset=nr', '--predict-num=15', '--method-opt=alpha=0.8 T=0.5'])
    main(['--method=netlaprls', '--dataset=nr', '--predict-num=15', '--method-opt=beta_d=10**(-1) gamma_d=10 beta_t=10**(-1) gamma_t=10'])
    
    main(['--method=knn_bprcasq', '--dataset=nr', '--predict-num=15', '--method-opt=D=100 global_regularization=0.05 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprcasq', '--dataset=nr', '--predict-num=15', '--method-opt=D=100 global_regularization=0.05 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=nrlmf', '--dataset=nr', '--predict-num=15','--method-opt=alpha=2 lambda_d=0.125 lambda_t=0.125 num_factors=100 beta=2**(-1) theta=2**(-2)'])
    
    main(['--method=blmnii', '--dataset=e', '--predict-num=20', '--method-opt=alpha=1.0'])
    main(['--method=wnngip', '--dataset=e', '--predict-num=20', '--method-opt=alpha=0.9 T=0.3'])
    main(['--method=netlaprls', '--dataset=e', '--predict-num=20', '--method-opt=beta_d=10**(-3) gamma_d=10 beta_t=10**(-3) gamma_t=10'])
    
    main(['--method=knn_bprcasq', '--dataset=e', '--predict-num=20', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprcasq', '--dataset=e', '--predict-num=20', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=nrlmf', '--dataset=e', '--predict-num=20', '--method-opt=alpha=2 lambda_d=1 lambda_t=1 num_factors=100 beta=2**(-1) theta=2**(-2)'])
    main(['--method=cmf', '--dataset=e', '--predict-num=5', '--method-opt=lambda_d=0.0625 lambda_l=1 lambda_t=0.25 K=100'])
    
    
    main(['--method=blmnii', '--dataset=nr', '--predict-num=10', '--method-opt=alpha=0.9'])
    main(['--method=wnngip', '--dataset=nr', '--predict-num=10', '--method-opt=alpha=0.8 T=0.5'])
    main(['--method=netlaprls', '--dataset=nr', '--predict-num=10', '--method-opt=beta_d=10**(-1) gamma_d=10 beta_t=10**(-1) gamma_t=10'])
    main(['--method=cmf', '--dataset=nr', '--predict-num=10', '--method-opt=lambda_d=0.125 lambda_l=0.25 lambda_t=0.03125 K=100'])
    main(['--method=knn_bprcasq', '--dataset=nr', '--predict-num=10', '--method-opt=D=100 global_regularization=0.05 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprcasq', '--dataset=nr', '--predict-num=10', '--method-opt=D=100 global_regularization=0.05 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])        

    main(['--method=blmnii', '--dataset=gpcr', '--predict-num=10', '--method-opt=alpha=1.0'])
    main(['--method=wnngip', '--dataset=gpcr', '--predict-num=10', '--method-opt=alpha=0.9 T=0.2'])
    main(['--method=netlaprls', '--dataset=gpcr', '--predict-num=10', '--method-opt=beta_d=10**(-6) beta_t=10**(-6) gamma_d=100 gamma_t=100'])
    main(['--method=cmf', '--dataset=gpcr', '--predict-num=10', '--method-opt=lambda_d=0.125 lambda_l=0.5 lambda_t=0.03125 K=100'])
    main(['--method=knn_bprcasq', '--dataset=gpcr', '--predict-num=10', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprcasq', '--dataset=gpcr', '--predict-num=10', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
        
    main(['--method=blmnii', '--dataset=ic', '--predict-num=10', '--method-opt=alpha=0.8'])
    main(['--method=wnngip', '--dataset=ic', '--predict-num=10', '--method-opt=alpha=0.9 T=0.3'])
    main(['--method=netlaprls', '--dataset=ic', '--predict-num=10', '--method-opt=beta_d=10**(-6) gamma_d=10 beta_t=10**(-6) gamma_t=10'])
    main(['--method=cmf', '--dataset=ic', '--predict-num=10', '--method-opt=lambda_d=0.0625 lambda_l=1 lambda_t=0.125 K=100'])
    main(['--method=knn_bprcasq', '--dataset=ic', '--predict-num=10', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprcasq', '--dataset=ic', '--predict-num=10', '--method-opt=D=50 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
     
    main(['--method=blmnii', '--dataset=e', '--predict-num=10', '--method-opt=alpha=1.0'])
    main(['--method=wnngip', '--dataset=e', '--predict-num=10', '--method-opt=alpha=0.9 T=0.3'])
    main(['--method=netlaprls', '--dataset=e', '--predict-num=10', '--method-opt=beta_d=10**(-3) gamma_d=10 beta_t=10**(-3) gamma_t=10'])
    main(['--method=cmf', '--dataset=e', '--predict-num=10', '--method-opt=lambda_d=0.0625 lambda_l=1 lambda_t=0.25 K=100'])
    main(['--method=knn_bprcasq', '--dataset=e', '--predict-num=10', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprcasq', '--dataset=e', '--predict-num=10', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
       
    
    main(['--method=blmnii', '--dataset=ic', '--predict-num=5', '--method-opt=alpha=0.8'])    
    main(['--method=blmnii', '--dataset=nr', '--predict-num=5', '--method-opt=alpha=0.9'])    
    main(['--method=blmnii', '--dataset=e', '--predict-num=5', '--method-opt=alpha=1.0'])

    #######################################end of predicting novel DTIs###################
    
    
    
    
    main(['--method=knn_bprca', '--dataset=gpcr', '--cvs=1', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1 cb_alignment_regularization_user=0.5 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprca', '--dataset=gpcr', '--cvs=2', '--method-opt=D=50 learning_rate=0.1 global_regularization=0.01 bias_regularization=1 cb_alignment_regularization_user=0.5 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprca', '--dataset=gpcr', '--cvs=3', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1 cb_alignment_regularization_user=0.1 cb_alignment_regularization_item=1'])
    
    main(['--method=knn_bprca', '--dataset=ic', '--cvs=1', '--method-opt=D=100 global_regularization=0.05 cb_alignment_regularization_user=0.1 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprca', '--dataset=ic', '--cvs=2', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1 cb_alignment_regularization_user=0.01 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprca', '--dataset=ic', '--cvs=3', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1 cb_alignment_regularization_user=0.1 cb_alignment_regularization_item=1'])            
        
    main(['--method=knn_bprca', '--dataset=nr', '--cvs=1', '--method-opt=D=100 global_regularization=0.05 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprca', '--dataset=nr', '--cvs=2', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.9 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprca', '--dataset=nr', '--cvs=3', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    
    main(['--method=knn_bprca', '--dataset=e', '--cvs=1', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.1 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprca', '--dataset=e', '--cvs=2', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.01 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprca', '--dataset=e', '--cvs=3', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.1 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    
    
    main(['--method=knn_inv_bprca', '--dataset=gpcr', '--cvs=1', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1 cb_alignment_regularization_user=0.1 cb_alignment_regularization_item=1'])
    #main(['--method=knn_inv_bprca', '--dataset=gpcr', '--cvs=2', '--specify-arg=0'])
    #main(['--method=knn_inv_bprca', '--dataset=gpcr', '--cvs=3', '--specify-arg=0'])
   
    main(['--method=knn_inv_bprca', '--dataset=ic', '--cvs=1', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1 cb_alignment_regularization_user=0.1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprca', '--dataset=ic', '--cvs=2', '--method-opt=D=50 learning_rate=0.1 global_regularization=0.001 bias_regularization=1 cb_alignment_regularization_user=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprca', '--dataset=ic', '--cvs=3', '--method-opt=D=50 learning_rate=0.1 global_regularization=0.001 bias_regularization=1 cb_alignment_regularization_user=0.01 cb_alignment_regularization_item=1'])
      
    main(['--method=knn_inv_bprca', '--dataset=nr', '--cvs=1', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.1 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprca', '--dataset=nr', '--cvs=2', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=1 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprca', '--dataset=nr', '--cvs=3', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.01 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    
    main(['--method=knn_inv_bprca', '--dataset=e', '--cvs=1', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.1 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprca', '--dataset=e', '--cvs=2', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.1 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprca', '--dataset=e', '--cvs=3', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.1 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    
    
    
    


    ###########################blmnii##################################
    main(['--method=blmnii', '--dataset=gpcr', '--cvs=1', '--method-opt=alpha=1.0'])
    main(['--method=blmnii', '--dataset=gpcr', '--cvs=2', '--method-opt=alpha=1.0'])
    main(['--method=blmnii', '--dataset=gpcr', '--cvs=3', '--method-opt=alpha=1.0'])
    
    main(['--method=blmnii', '--dataset=ic', '--cvs=1', '--method-opt=alpha=0.8'])
    main(['--method=blmnii', '--dataset=ic', '--cvs=2', '--method-opt=alpha=1.0'])
    main(['--method=blmnii', '--dataset=ic', '--cvs=3', '--method-opt=alpha=1.0'])
    
    main(['--method=blmnii', '--dataset=nr', '--cvs=1', '--method-opt=alpha=0.9'])
    main(['--method=blmnii', '--dataset=nr', '--cvs=2', '--method-opt=alpha=1.0'])
    main(['--method=blmnii', '--dataset=nr', '--cvs=3', '--method-opt=alpha=1.0'])
    
    main(['--method=blmnii', '--dataset=e', '--cvs=1', '--method-opt=alpha=1.0'])
    main(['--method=blmnii', '--dataset=e', '--cvs=2', '--method-opt=alpha=1.0'])
    main(['--method=blmnii', '--dataset=e', '--cvs=3', '--method-opt=alpha=1.0'])
   
    
    ###########################wnngip##################################
    main(['--method=wnngip', '--dataset=gpcr', '--cvs=1', '--method-opt=alpha=0.9 T=0.2'])
    main(['--method=wnngip', '--dataset=gpcr', '--cvs=2', '--method-opt=alpha=1.0 T=0.7'])
    main(['--method=wnngip', '--dataset=gpcr', '--cvs=3', '--method-opt=alpha=1.0 T=0.5'])
    
    main(['--method=wnngip', '--dataset=ic', '--cvs=1', '--method-opt=alpha=0.9 T=0.3'])
    main(['--method=wnngip', '--dataset=ic', '--cvs=2', '--method-opt=alpha=0.9 T=0.6'])
    main(['--method=wnngip', '--dataset=ic', '--cvs=3', '--method-opt=alpha=0.9 T=0.6'])
    
    main(['--method=wnngip', '--dataset=nr', '--cvs=1', '--method-opt=alpha=0.8 T=0.5'])
    main(['--method=wnngip', '--dataset=nr', '--cvs=2', '--method-opt=alpha=0.9 T=0.3'])
    main(['--method=wnngip', '--dataset=nr', '--cvs=3', '--method-opt=alpha=0.8 T=0.6'])
    
    main(['--method=wnngip', '--dataset=e', '--cvs=1', '--method-opt=alpha=0.9 T=0.3'])
    main(['--method=wnngip', '--dataset=e', '--cvs=2', '--method-opt=alpha=1.0 T=0.7'])
    main(['--method=wnngip', '--dataset=e', '--cvs=3', '--method-opt=alpha=1.0 T=0.5'])       
  

     
    ###########################netlaprls##################################
    main(['--method=netlaprls', '--dataset=gpcr', '--cvs=1', '--method-opt=beta_d=10**(-6) beta_t=10**(-6) gamma_d=100 gamma_t=100'])
    main(['--method=netlaprls', '--dataset=gpcr', '--cvs=2', '--method-opt=beta_d=10**(-3) gamma_d=10**2 beta_t=10**(-3) gamma_t=10**2'])
    main(['--method=netlaprls', '--dataset=gpcr', '--cvs=3', '--method-opt=beta_d=10**(-3) gamma_d=10**(1) beta_t=10**(-3) gamma_t=10**1'])
    
    main(['--method=netlaprls', '--dataset=ic', '--cvs=1', '--method-opt=beta_d=10**(-6) gamma_d=10 beta_t=10**(-6) gamma_t=10'])
    main(['--method=netlaprls', '--dataset=ic', '--cvs=2', '--method-opt=beta_d=10**(-1) gamma_d=10 beta_t=10**(-1) gamma_t=10'])
    main(['--method=netlaprls', '--dataset=ic', '--cvs=3', '--method-opt=beta_d=10**(-3) gamma_d=10**(0) beta_t=10**(-3) gamma_t=10**0'])
    
    main(['--method=netlaprls', '--dataset=nr', '--cvs=1', '--method-opt=beta_d=10**(-1) gamma_d=10 beta_t=10**(-1) gamma_t=10'])
    main(['--method=netlaprls', '--dataset=nr', '--cvs=2', '--method-opt=beta_d=10**(0) gamma_d=10**2 beta_t=10**(0) gamma_t=10**2'])
    main(['--method=netlaprls', '--dataset=nr', '--cvs=3', '--method-opt=beta_d=10**(-6) gamma_d=0.1 beta_t=10**(-6) gamma_t=0.1'])
    
    main(['--method=netlaprls', '--dataset=e', '--cvs=1', '--method-opt=beta_d=10**(-3) gamma_d=10 beta_t=10**(-3) gamma_t=10'])
    main(['--method=netlaprls', '--dataset=e', '--cvs=2', '--method-opt=beta_d=10**(-6) gamma_d=10**2 beta_t=10**(-6) gamma_t=10**2'])
    main(['--method=netlaprls', '--dataset=e', '--cvs=3', '--method-opt=beta_d=10**(-4) gamma_d=10 beta_t=10**(-4) gamma_t=10'])           
    
    
    ###########################cmf - second version of parameters##################################
    main(['--method=cmf', '--dataset=gpcr', '--cvs=1', '--method-opt=lambda_d=0.125 lambda_l=0.5 lambda_t=0.03125 K=100'])
    main(['--method=cmf', '--dataset=gpcr', '--cvs=2', '--method-opt=lambda_d=0.25 lambda_l=1 lambda_t=0.03125 K=100'])
    main(['--method=cmf', '--dataset=gpcr', '--cvs=3', '--method-opt=lambda_d=0.25 lambda_l=1 lambda_t=0.03125 K=100'])
    
    main(['--method=cmf', '--dataset=ic', '--cvs=1', '--method-opt=lambda_d=0.0625 lambda_l=1 lambda_t=0.125 K=100'])
    main(['--method=cmf', '--dataset=ic', '--cvs=2', '--method-opt=lambda_d=0.0625 lambda_l=0.5 lambda_t=0.03125 K=100'])
    main(['--method=cmf', '--dataset=ic', '--cvs=3', '--method-opt=lambda_d=0.0625 lambda_l=1 lambda_t=0.03125 K=100'])
    
    main(['--method=cmf', '--dataset=nr', '--cvs=1', '--method-opt=lambda_d=0.125 lambda_l=0.25 lambda_t=0.03125 K=100'])
    main(['--method=cmf', '--dataset=nr', '--cvs=2', '--method-opt=lambda_d=0.125 lambda_l=0.5 lambda_t=0.03125 K=100'])
    main(['--method=cmf', '--dataset=nr', '--cvs=3', '--method-opt=lambda_d=0.25 lambda_l=0.5 lambda_t=0.03125 K=100'])
    
    main(['--method=cmf', '--dataset=e', '--cvs=1', '--method-opt=lambda_d=0.0625 lambda_l=1 lambda_t=0.25 K=100'])
    main(['--method=cmf', '--dataset=e', '--cvs=2', '--method-opt=lambda_d=0.0625 lambda_l=0.5 lambda_t=0.03125 K=100'])
    main(['--method=cmf', '--dataset=e', '--cvs=3', '--method-opt=lambda_d=0.0625 lambda_l=0.5 lambda_t=0.03125 K=100'])      
          
    """"""
     
    
   
    ###########################nrlmf################################## {'y': 1, 'x': 0, 'x': 0, 'r': 100, 'z': -1, 't': -1}
    
    main(['--method=nrlmf', '--dataset=e', '--cvs=1', '--method-opt=alpha=2 lambda_d=1 lambda_t=1 num_factors=100 beta=2**(-1) theta=2**(-2)'])
    main(['--method=nrlmf', '--dataset=e', '--cvs=2', '--method-opt=alpha=2**(1) lambda_d=2**(-2) lambda_t=2**(-2) num_factors=100 beta=2**(-1) theta=2**(-2)'])
    main(['--method=nrlmf', '--dataset=e', '--cvs=3', '--method-opt=alpha=2**(0) lambda_d=2**(-4) lambda_t=2**(-4) num_factors=100 beta=2**(-1) theta=2**(-1)']) 
    
    main(['--method=nrlmf', '--dataset=gpcr', '--cvs=1', '--method-opt=alpha=2 lambda_d=0.5 lambda_t=0.5 num_factors=100 beta=0.5 theta=0.5'])
    main(['--method=nrlmf', '--dataset=gpcr', '--cvs=2', '--method-opt=alpha=0.5 lambda_d=0.0625 lambda_t=0.0625 num_factors=100 beta=0.5 theta=0.25'])
    main(['--method=nrlmf', '--dataset=gpcr', '--cvs=3', '--method-opt=alpha=0.0625 lambda_d=2**(-3) lambda_t=2**(-3) num_factors=100 beta=2**(-3) theta=2**(-2)'])
    
    main(['--method=nrlmf', '--dataset=ic', '--cvs=1', '--method-opt=alpha=1 lambda_d=1 lambda_t=1 num_factors=100 beta=2**(-1) theta=2**(-1)'])
    main(['--method=nrlmf', '--dataset=ic', '--cvs=2', '--method-opt=alpha=2 lambda_d=0.0625 lambda_t=0.0625 num_factors=100 beta=2**(-2) theta=2**(-1)'])
    main(['--method=nrlmf', '--dataset=ic', '--cvs=3', '--method-opt=alpha=2**(-2) lambda_d=0.5 lambda_t=0.5 num_factors=100 beta=2**(-1) theta=2**(-1)'])
    
    main(['--method=nrlmf', '--dataset=nr', '--cvs=1', '--method-opt=alpha=2 lambda_d=0.125 lambda_t=0.125 num_factors=100 beta=2**(-1) theta=2**(-2)'])
    main(['--method=nrlmf', '--dataset=nr', '--cvs=2', '--method-opt=alpha=2**(-3) lambda_d=2**(-2) lambda_t=2**(-2) num_factors=100 beta=2**(-3) theta=2**(-2)'])
    main(['--method=nrlmf', '--dataset=nr', '--cvs=3', '--method-opt=alpha=2**(-4) lambda_d=0.5 lambda_t=0.5 num_factors=100 beta=0.5 theta=0.5'])
    
    
   
    ######################################################## KNN_INV BPRCA########################
    main(['--method=knn_inv_bprca', '--dataset=gpcr', '--cvs=1', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1 cb_alignment_regularization_user=0.5 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprca', '--dataset=gpcr', '--cvs=2', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1 cb_alignment_regularization_user=0.5 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprca', '--dataset=gpcr', '--cvs=3', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1 cb_alignment_regularization_user=0.01 cb_alignment_regularization_item=1'])
    
    main(['--method=knn_inv_bprca', '--dataset=ic', '--cvs=1', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1 cb_alignment_regularization_user=0.5 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprca', '--dataset=ic', '--cvs=2', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1 cb_alignment_regularization_user=0.1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprca', '--dataset=ic', '--cvs=3', '--method-opt=D=50 learning_rate=0.1 global_regularization=0.01 bias_regularization=1 cb_alignment_regularization_user=0.01 cb_alignment_regularization_item=1'])
        
    main(['--method=knn_inv_bprca', '--dataset=nr', '--cvs=1', '--method-opt=D=100 global_regularization=0.05 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprca', '--dataset=nr', '--cvs=2', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprca', '--dataset=nr', '--cvs=3', '--method-opt=D=100 global_regularization=0.05 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])

    
    
    ###########################blmnii##################################
    main(['--method=blmnii', '--dataset=gpcr', '--cvs=1', '--specify-arg=0'])
    main(['--method=blmnii', '--dataset=gpcr', '--cvs=2', '--specify-arg=0'])
    main(['--method=blmnii', '--dataset=gpcr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=blmnii', '--dataset=ic', '--cvs=1', '--specify-arg=0'])
    main(['--method=blmnii', '--dataset=ic', '--cvs=2', '--specify-arg=0'])
    main(['--method=blmnii', '--dataset=ic', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=blmnii', '--dataset=nr', '--cvs=1', '--specify-arg=0'])
    main(['--method=blmnii', '--dataset=nr', '--cvs=2', '--specify-arg=0'])
    main(['--method=blmnii', '--dataset=nr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=blmnii', '--dataset=e', '--cvs=1', '--specify-arg=0'])
    main(['--method=blmnii', '--dataset=e', '--cvs=2', '--specify-arg=0'])
    main(['--method=blmnii', '--dataset=e', '--cvs=3', '--specify-arg=0'])
   
    
    ###########################wnngip##################################
    main(['--method=wnngip', '--dataset=gpcr', '--cvs=1', '--specify-arg=0'])
    main(['--method=wnngip', '--dataset=gpcr', '--cvs=2', '--specify-arg=0'])
    main(['--method=wnngip', '--dataset=gpcr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=wnngip', '--dataset=ic', '--cvs=1', '--specify-arg=0'])
    main(['--method=wnngip', '--dataset=ic', '--cvs=2', '--specify-arg=0'])
    main(['--method=wnngip', '--dataset=ic', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=wnngip', '--dataset=nr', '--cvs=1', '--specify-arg=0'])
    main(['--method=wnngip', '--dataset=nr', '--cvs=2', '--specify-arg=0'])
    main(['--method=wnngip', '--dataset=nr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=wnngip', '--dataset=e', '--cvs=1', '--specify-arg=0'])
    main(['--method=wnngip', '--dataset=e', '--cvs=2', '--specify-arg=0'])
    main(['--method=wnngip', '--dataset=e', '--cvs=3', '--specify-arg=0'])    
  
     
    ###########################netlaprls##################################
    main(['--method=netlaprls', '--dataset=gpcr', '--cvs=1', '--specify-arg=0'])
    main(['--method=netlaprls', '--dataset=gpcr', '--cvs=2', '--specify-arg=0'])
    main(['--method=netlaprls', '--dataset=gpcr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=netlaprls', '--dataset=ic', '--cvs=1', '--specify-arg=0'])
    main(['--method=netlaprls', '--dataset=ic', '--cvs=2', '--specify-arg=0'])
    main(['--method=netlaprls', '--dataset=ic', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=netlaprls', '--dataset=nr', '--cvs=1', '--specify-arg=0'])
    main(['--method=netlaprls', '--dataset=nr', '--cvs=2', '--specify-arg=0'])
    main(['--method=netlaprls', '--dataset=nr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=netlaprls', '--dataset=e', '--cvs=1', '--specify-arg=0'])
    main(['--method=netlaprls', '--dataset=e', '--cvs=2', '--specify-arg=0'])
    main(['--method=netlaprls', '--dataset=e', '--cvs=3', '--specify-arg=0'])      
    
    
    ###########################cmf##################################
    
    main(['--method=cmf', '--dataset=nr', '--cvs=1', '--specify-arg=0'])
    main(['--method=cmf', '--dataset=nr', '--cvs=2', '--specify-arg=0'])
    main(['--method=cmf', '--dataset=nr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=cmf', '--dataset=gpcr', '--cvs=1', '--specify-arg=0'])
    main(['--method=cmf', '--dataset=gpcr', '--cvs=2', '--specify-arg=0'])
    main(['--method=cmf', '--dataset=gpcr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=cmf', '--dataset=ic', '--cvs=1', '--specify-arg=0'])
    main(['--method=cmf', '--dataset=ic', '--cvs=2', '--specify-arg=0'])
    main(['--method=cmf', '--dataset=ic', '--cvs=3', '--specify-arg=0'])    
    
    main(['--method=cmf', '--dataset=e', '--cvs=1', '--specify-arg=0'])
    main(['--method=cmf', '--dataset=e', '--cvs=2', '--specify-arg=0'])
    main(['--method=cmf', '--dataset=e', '--cvs=3', '--specify-arg=0'])      
 
   
    ###########################nrlmf##################################
    
    main(['--method=nrlmf', '--dataset=gpcr', '--cvs=1', '--specify-arg=0'])
    main(['--method=nrlmf', '--dataset=gpcr', '--cvs=2', '--specify-arg=0'])
    main(['--method=nrlmf', '--dataset=gpcr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=nrlmf', '--dataset=ic', '--cvs=1', '--specify-arg=0'])
    main(['--method=nrlmf', '--dataset=ic', '--cvs=2', '--specify-arg=0'])
    main(['--method=nrlmf', '--dataset=ic', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=nrlmf', '--dataset=nr', '--cvs=1', '--specify-arg=0'])
    main(['--method=nrlmf', '--dataset=nr', '--cvs=2', '--specify-arg=0'])
    main(['--method=nrlmf', '--dataset=nr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=nrlmf', '--dataset=e', '--cvs=1', '--specify-arg=0'])
    main(['--method=nrlmf', '--dataset=e', '--cvs=2', '--specify-arg=0'])
    main(['--method=nrlmf', '--dataset=e', '--cvs=3', '--specify-arg=0']) 
    
    ###########################bpr_ca_squared_CA##################################
    
    main(['--method=knn_inv_bprcasq', '--dataset=e', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_inv_bprcasq', '--dataset=e', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_inv_bprcasq', '--dataset=e', '--cvs=3', '--specify-arg=0'])        
    
    main(['--method=knn_bprcasq', '--dataset=e', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_bprcasq', '--dataset=e', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_bprcasq', '--dataset=e', '--cvs=3', '--specify-arg=0']) 
     
    main(['--method=knn_inv_bprcasq', '--dataset=e', '--cvs=3', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1 cb_alignment_regularization_user=0.05 cb_alignment_regularization_item=1']) 
    main(['--method=knn_bprcasq', '--dataset=e', '--cvs=3', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1 cb_alignment_regularization_user=0.05 cb_alignment_regularization_item=1'])

     
    main(['--method=knn_inv_bprca', '--dataset=gpcr', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_inv_bprca', '--dataset=gpcr', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_inv_bprca', '--dataset=gpcr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=knn_inv_bprca', '--dataset=ic', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_inv_bprca', '--dataset=ic', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_inv_bprca', '--dataset=ic', '--cvs=3', '--specify-arg=0'])
        
    main(['--method=knn_inv_bprca', '--dataset=nr', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_inv_bprca', '--dataset=nr', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_inv_bprca', '--dataset=nr', '--cvs=3', '--specify-arg=0'])
    
    
    main(['--method=knn_bprca', '--dataset=gpcr', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_bprca', '--dataset=gpcr', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_bprca', '--dataset=gpcr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=knn_bprca', '--dataset=ic', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_bprca', '--dataset=ic', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_bprca', '--dataset=ic', '--cvs=3', '--specify-arg=0'])
        
    main(['--method=knn_bprca', '--dataset=nr', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_bprca', '--dataset=nr', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_bprca', '--dataset=nr', '--cvs=3', '--specify-arg=0'])
    
    
    main(['--method=knn_inv_bprca', '--dataset=e', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_inv_bprca', '--dataset=e', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_inv_bprca', '--dataset=e', '--cvs=3', '--specify-arg=0'])        
    
    main(['--method=knn_bprca', '--dataset=e', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_bprca', '--dataset=e', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_bprca', '--dataset=e', '--cvs=3', '--specify-arg=0'])
    
   
  
   
    
    main(['--method=bprca', '--dataset=gpcr', '--cvs=1', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1 cb_alignment_regularization_user=1 cb_alignment_regularization_item=1'])
    main(['--method=bprca', '--dataset=gpcr', '--cvs=2', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.001 bias_regularization=1 cb_alignment_regularization_user=0.01 cb_alignment_regularization_item=1'])
    main(['--method=bprca', '--dataset=gpcr', '--cvs=3', '--method-opt=D=50 learning_rate=0.1 global_regularization=0.001 bias_regularization=1 cb_alignment_regularization_user=1 cb_alignment_regularization_item=1'])
    
    main(['--method=bprca', '--dataset=ic', '--cvs=1', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1 cb_alignment_regularization_user=0.1 cb_alignment_regularization_item=1'])
    main(['--method=bprca', '--dataset=ic', '--cvs=2', '--method-opt=D=50 learning_rate=0.1 global_regularization=0.01 bias_regularization=1 cb_alignment_regularization_user=0.01 cb_alignment_regularization_item=1'])
    main(['--method=bprca', '--dataset=ic', '--cvs=3', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.001 bias_regularization=1 cb_alignment_regularization_user=1 cb_alignment_regularization_item=1'])
        
    main(['--method=bprca', '--dataset=nr', '--cvs=1', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1 cb_alignment_regularization_user=1 cb_alignment_regularization_item=1'])
    main(['--method=bprca', '--dataset=nr', '--cvs=2', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.001 bias_regularization=1 cb_alignment_regularization_user=1 cb_alignment_regularization_item=1'])
    main(['--method=bprca', '--dataset=nr', '--cvs=3', '--method-opt=D=50 learning_rate=0.1 global_regularization=0.01 bias_regularization=1 cb_alignment_regularization_user=0.01 cb_alignment_regularization_item=1'])
    
    main(['--method=bprca', '--dataset=e', '--cvs=1', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1 cb_alignment_regularization_user=1 cb_alignment_regularization_item=1'])
    main(['--method=bprca', '--dataset=e', '--cvs=2', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.001 bias_regularization=1 cb_alignment_regularization_user=0.01 cb_alignment_regularization_item=1'])
    main(['--method=bprca', '--dataset=e', '--cvs=3', '--method-opt=D=50 learning_rate=0.1 global_regularization=0.001 bias_regularization=1 cb_alignment_regularization_user=1 cb_alignment_regularization_item=1'])
    
    main(['--method=bprcaest', '--dataset=gpcr', '--cvs=1', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1'])
    main(['--method=bprcaest', '--dataset=gpcr', '--cvs=2', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.001 bias_regularization=1'])
    main(['--method=bprcaest', '--dataset=gpcr', '--cvs=3', '--method-opt=D=50 learning_rate=0.1 global_regularization=0.001 bias_regularization=1'])
    
    main(['--method=bprcaest', '--dataset=ic', '--cvs=1', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1'])
    main(['--method=bprcaest', '--dataset=ic', '--cvs=2', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.001 bias_regularization=1'])
    main(['--method=bprcaest', '--dataset=ic', '--cvs=3', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.001 bias_regularization=1'])
        
    main(['--method=bprcaest', '--dataset=nr', '--cvs=1', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1'])
    main(['--method=bprcaest', '--dataset=nr', '--cvs=2', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1'])
    main(['--method=bprcaest', '--dataset=nr', '--cvs=3', '--method-opt=D=50 learning_rate=0.1 global_regularization=0.01 bias_regularization=1'])
    
    main(['--method=bprcaest', '--dataset=e', '--cvs=1', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1'])
    main(['--method=bprcaest', '--dataset=e', '--cvs=2', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.001 bias_regularization=1'])
    main(['--method=bprcaest', '--dataset=e', '--cvs=3', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1'])


    main(['--method=bpr', '--dataset=gpcr', '--cvs=1', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1'])
    main(['--method=bpr', '--dataset=gpcr', '--cvs=2', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.1 bias_regularization=1'])
    main(['--method=bpr', '--dataset=gpcr', '--cvs=3', '--method-opt=D=50 learning_rate=0.1 global_regularization=0.01 bias_regularization=1'])
    
    main(['--method=bpr', '--dataset=ic', '--cvs=1', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1'])
    main(['--method=bpr', '--dataset=ic', '--cvs=2', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.3 bias_regularization=1'])
    main(['--method=bpr', '--dataset=ic', '--cvs=3', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.3 bias_regularization=1'])
        
    main(['--method=bpr', '--dataset=nr', '--cvs=1', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1'])
    main(['--method=bpr', '--dataset=nr', '--cvs=2', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.3 bias_regularization=1'])
    main(['--method=bpr', '--dataset=nr', '--cvs=3', '--method-opt=D=50 learning_rate=0.1 global_regularization=0.1 bias_regularization=1'])
    
    main(['--method=bpr', '--dataset=e', '--cvs=1', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.01 bias_regularization=1'])
    main(['--method=bpr', '--dataset=e', '--cvs=2', '--method-opt=D=50 learning_rate=0.1 global_regularization=0.3 bias_regularization=1'])
    main(['--method=bpr', '--dataset=e', '--cvs=3', '--method-opt=D=100 learning_rate=0.1 global_regularization=0.3 bias_regularization=1'])
    
    
    main(['--method=knn_bprca', '--dataset=nr', '--cvs=1', '--method-opt=D=100 global_regularization=0.04 cb_alignment_regularization_user=0.1 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprca', '--dataset=nr', '--cvs=2', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.6 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprca', '--dataset=nr', '--cvs=3', '--method-opt=D=100 global_regularization=0.04 cb_alignment_regularization_user=0.4 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])

    main(['--method=knn_bprca', '--dataset=gpcr', '--cvs=1', '--method-opt=D=100 global_regularization=0.02 cb_alignment_regularization_user=0.4 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprca', '--dataset=gpcr', '--cvs=2', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.2 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprca', '--dataset=gpcr', '--cvs=3', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.1 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    
    
    main(['--method=knn_bprcafbp', '--dataset=nr', '--cvs=1', '--method-opt=D=100 global_regularization=0.05 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprcafbp', '--dataset=nr', '--cvs=2', '--method-opt=D=50 global_regularization=0.01 cb_alignment_regularization_user=0.1 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprcafbp', '--dataset=nr', '--cvs=3', '--method-opt=D=100 global_regularization=0.05 cb_alignment_regularization_user=0.1 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])

    main(['--method=knn_bprcafbp', '--dataset=gpcr', '--cvs=1', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprcafbp', '--dataset=gpcr', '--cvs=2', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprcafbp', '--dataset=gpcr', '--cvs=3', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.05 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])



    
    main(['--method=knn_bprcasq', '--dataset=gpcr', '--cvs=1', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprcasq', '--dataset=gpcr', '--cvs=2', '--method-opt=D=50 global_regularization=0.01 cb_alignment_regularization_user=0.1 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprcasq', '--dataset=gpcr', '--cvs=3', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.05 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
        
    main(['--method=knn_bprcasq', '--dataset=ic', '--cvs=1', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprcasq', '--dataset=ic', '--cvs=2', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.05 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprcasq', '--dataset=ic', '--cvs=3', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.1 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])        
        
    main(['--method=knn_inv_bprcasq', '--dataset=gpcr', '--cvs=1', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprcasq', '--dataset=gpcr', '--cvs=2', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprcasq', '--dataset=gpcr', '--cvs=3', '--method-opt=D=50 global_regularization=0.01 cb_alignment_regularization_user=0.05 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
      
    main(['--method=knn_inv_bprcasq', '--dataset=ic', '--cvs=1', '--method-opt=D=50 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprcasq', '--dataset=ic', '--cvs=2', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.1 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprcasq', '--dataset=ic', '--cvs=3', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.05 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])        
                   
    main(['--method=knn_bprcasq', '--dataset=nr', '--cvs=1', '--method-opt=D=100 global_regularization=0.05 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprcasq', '--dataset=nr', '--cvs=2', '--method-opt=D=50 global_regularization=0.01 cb_alignment_regularization_user=0.05 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprcasq', '--dataset=nr', '--cvs=3', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.05 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
         
    main(['--method=knn_inv_bprcasq', '--dataset=nr', '--cvs=1', '--method-opt=D=100 global_regularization=0.05 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprcasq', '--dataset=nr', '--cvs=2', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=1.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprcasq', '--dataset=nr', '--cvs=3', '--method-opt=D=100 global_regularization=0.05 cb_alignment_regularization_user=0.05 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
   
   
    main(['--method=knn_bprcasq', '--dataset=e', '--cvs=1', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprcasq', '--dataset=e', '--cvs=2', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.05 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_bprcasq', '--dataset=e', '--cvs=3', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.05 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])        
    """    
    main(['--method=knn_inv_bprcasq', '--dataset=e', '--cvs=1', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprcasq', '--dataset=e', '--cvs=2', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    main(['--method=knn_inv_bprcasq', '--dataset=e', '--cvs=3', '--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.05 learning_rate=0.1 bias_regularization=1 cb_alignment_regularization_item=1'])
    """  
   
    main(['--method=knn_bprcasq', '--dataset=nr', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_bprcasq', '--dataset=nr', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_bprcasq', '--dataset=nr', '--cvs=3', '--specify-arg=0'])
            
    main(['--method=knn_inv_bprcasq', '--dataset=nr', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_inv_bprcasq', '--dataset=nr', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_inv_bprcasq', '--dataset=nr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=knn_bprcasq', '--dataset=gpcr', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_bprcasq', '--dataset=gpcr', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_bprcasq', '--dataset=gpcr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=knn_bprcasq', '--dataset=ic', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_bprcasq', '--dataset=ic', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_bprcasq', '--dataset=ic', '--cvs=3', '--specify-arg=0'])
        
    main(['--method=knn_inv_bprcasq', '--dataset=gpcr', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_inv_bprcasq', '--dataset=gpcr', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_inv_bprcasq', '--dataset=gpcr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=knn_inv_bprcasq', '--dataset=ic', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_inv_bprcasq', '--dataset=ic', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_inv_bprcasq', '--dataset=ic', '--cvs=3', '--specify-arg=0'])

     
    main(['--method=knn_bprca', '--dataset=e', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_bprca', '--dataset=e', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_bprca', '--dataset=e', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=knn_bprcaest', '--dataset=gpcr', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_bprcaest', '--dataset=gpcr', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_bprcaest', '--dataset=gpcr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=knn_bprcaest', '--dataset=ic', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_bprcaest', '--dataset=ic', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_bprcaest', '--dataset=ic', '--cvs=3', '--specify-arg=0'])
        
    main(['--method=knn_bprcaest', '--dataset=nr', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_bprcaest', '--dataset=nr', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_bprcaest', '--dataset=nr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=knn_bprcaest', '--dataset=e', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_bprcaest', '--dataset=e', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_bprcaest', '--dataset=e', '--cvs=3', '--specify-arg=0'])

    main(['--method=rankalsca', '--dataset=gpcr', '--cvs=1', '--specify-arg=0'])
    main(['--method=rankalsca', '--dataset=gpcr', '--cvs=2', '--specify-arg=0'])
    main(['--method=rankalsca', '--dataset=gpcr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=rankalsca', '--dataset=ic', '--cvs=1', '--specify-arg=0'])
    main(['--method=rankalsca', '--dataset=ic', '--cvs=2', '--specify-arg=0'])
    main(['--method=rankalsca', '--dataset=ic', '--cvs=3', '--specify-arg=0'])
        
    main(['--method=rankalsca', '--dataset=nr', '--cvs=1', '--specify-arg=0'])
    main(['--method=rankalsca', '--dataset=nr', '--cvs=2', '--specify-arg=0'])
    main(['--method=rankalsca', '--dataset=nr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=rankalsca', '--dataset=e', '--cvs=1', '--specify-arg=0'])
    main(['--method=rankalsca', '--dataset=e', '--cvs=2', '--specify-arg=0'])
    main(['--method=rankalsca', '--dataset=e', '--cvs=3', '--specify-arg=0'])  

    
    main(['--method=inv_bprca', '--dataset=gpcr', '--cvs=1', '--specify-arg=0'])
    main(['--method=inv_bprca', '--dataset=gpcr', '--cvs=2', '--specify-arg=0'])
    main(['--method=inv_bprca', '--dataset=gpcr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=inv_bprca', '--dataset=ic', '--cvs=1', '--specify-arg=0'])
    main(['--method=inv_bprca', '--dataset=ic', '--cvs=2', '--specify-arg=0'])
    main(['--method=inv_bprca', '--dataset=ic', '--cvs=3', '--specify-arg=0'])
        
    main(['--method=inv_bprca', '--dataset=nr', '--cvs=1', '--specify-arg=0'])
    main(['--method=inv_bprca', '--dataset=nr', '--cvs=2', '--specify-arg=0'])
    main(['--method=inv_bprca', '--dataset=nr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=inv_bprca', '--dataset=e', '--cvs=1', '--specify-arg=0'])
    main(['--method=inv_bprca', '--dataset=e', '--cvs=2', '--specify-arg=0'])
    main(['--method=inv_bprca', '--dataset=e', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=inv_bprcaest', '--dataset=gpcr', '--cvs=1', '--specify-arg=0'])
    main(['--method=inv_bprcaest', '--dataset=gpcr', '--cvs=2', '--specify-arg=0'])
    main(['--method=inv_bprcaest', '--dataset=gpcr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=inv_bprcaest', '--dataset=ic', '--cvs=1', '--specify-arg=0'])
    main(['--method=inv_bprcaest', '--dataset=ic', '--cvs=2', '--specify-arg=0'])
    main(['--method=inv_bprcaest', '--dataset=ic', '--cvs=3', '--specify-arg=0'])
        
    main(['--method=inv_bprcaest', '--dataset=nr', '--cvs=1', '--specify-arg=0'])
    main(['--method=inv_bprcaest', '--dataset=nr', '--cvs=2', '--specify-arg=0'])
    main(['--method=inv_bprcaest', '--dataset=nr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=inv_bprcaest', '--dataset=e', '--cvs=1', '--specify-arg=0'])
    main(['--method=inv_bprcaest', '--dataset=e', '--cvs=2', '--specify-arg=0'])
    main(['--method=inv_bprcaest', '--dataset=e', '--cvs=3', '--specify-arg=0'])


    main(['--method=inv_bpr', '--dataset=gpcr', '--cvs=1', '--specify-arg=0'])
    main(['--method=inv_bpr', '--dataset=gpcr', '--cvs=2', '--specify-arg=0'])
    main(['--method=inv_bpr', '--dataset=gpcr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=inv_bpr', '--dataset=ic', '--cvs=1', '--specify-arg=0'])
    main(['--method=inv_bpr', '--dataset=ic', '--cvs=2', '--specify-arg=0'])
    main(['--method=inv_bpr', '--dataset=ic', '--cvs=3', '--specify-arg=0'])
        
    main(['--method=inv_bpr', '--dataset=nr', '--cvs=1', '--specify-arg=0'])
    main(['--method=inv_bpr', '--dataset=nr', '--cvs=2', '--specify-arg=0'])
    main(['--method=inv_bpr', '--dataset=nr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=inv_bpr', '--dataset=e', '--cvs=1', '--specify-arg=0'])
    main(['--method=inv_bpr', '--dataset=e', '--cvs=2', '--specify-arg=0'])
    main(['--method=inv_bpr', '--dataset=e', '--cvs=3', '--specify-arg=0'])
    
    
    main(['--method=rankals', '--dataset=gpcr', '--cvs=1', '--specify-arg=0'])
    main(['--method=rankals', '--dataset=gpcr', '--cvs=2', '--specify-arg=0'])
    main(['--method=rankals', '--dataset=gpcr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=rankals', '--dataset=ic', '--cvs=1', '--specify-arg=0'])
    main(['--method=rankals', '--dataset=ic', '--cvs=2', '--specify-arg=0'])
    main(['--method=rankals', '--dataset=ic', '--cvs=3', '--specify-arg=0'])
        
    main(['--method=rankals', '--dataset=nr', '--cvs=1', '--specify-arg=0'])
    main(['--method=rankals', '--dataset=nr', '--cvs=2', '--specify-arg=0'])
    main(['--method=rankals', '--dataset=nr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=rankals', '--dataset=e', '--cvs=1', '--specify-arg=0'])
    main(['--method=rankals', '--dataset=e', '--cvs=2', '--specify-arg=0'])
    main(['--method=rankals', '--dataset=e', '--cvs=3', '--specify-arg=0'])    
    
    
    main(['--method=bprca', '--dataset=gpcr', '--cvs=2', '--specify-arg=0'])
    
    main(['--method=knn_inv_bprca', '--dataset=gpcr', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_inv_bprca', '--dataset=gpcr', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_inv_bprca', '--dataset=gpcr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=knn_inv_bprca', '--dataset=ic', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_inv_bprca', '--dataset=ic', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_inv_bprca', '--dataset=ic', '--cvs=3', '--specify-arg=0'])
        
    main(['--method=knn_inv_bprca', '--dataset=nr', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_inv_bprca', '--dataset=nr', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_inv_bprca', '--dataset=nr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=knn_inv_bprca', '--dataset=e', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_inv_bprca', '--dataset=e', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_inv_bprca', '--dataset=e', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=knn_inv_bprcaest', '--dataset=gpcr', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_inv_bprcaest', '--dataset=gpcr', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_inv_bprcaest', '--dataset=gpcr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=knn_inv_bprcaest', '--dataset=ic', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_inv_bprcaest', '--dataset=ic', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_inv_bprcaest', '--dataset=ic', '--cvs=3', '--specify-arg=0'])
        
    main(['--method=knn_inv_bprcaest', '--dataset=nr', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_inv_bprcaest', '--dataset=nr', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_inv_bprcaest', '--dataset=nr', '--cvs=3', '--specify-arg=0'])
    
    main(['--method=knn_inv_bprcaest', '--dataset=e', '--cvs=1', '--specify-arg=0'])
    main(['--method=knn_inv_bprcaest', '--dataset=e', '--cvs=2', '--specify-arg=0'])
    main(['--method=knn_inv_bprcaest', '--dataset=e', '--cvs=3', '--specify-arg=0'])
    
    
    d = np.random.rand(5000,5000)
    e = np.random.rand(5000,100)
    i = 50
    l = 0.0001
 
    t = time.clock()
    for j in xrange(5000):
        
        f=d[i,:].reshape(-1,1).T
        g = np.dot(f, e)  
        
        e[i,:] += (l * np.asarray(g).reshape(-1))
        
    print(e[i,:])
    print(time.clock()-t) 
    
    
    t = time.clock()
    for j in xrange(5000):
        
        f=d[i,:].reshape(-1,1).T
        g = np.dot(f, e)  
        e[i,:] += l * g.reshape(-1)
        
        
    print(e[i,:])
    print(time.clock()-t)  
    
    t = time.clock()
    for j in xrange(5000):
        
        f=d[i,:].reshape(-1,1).T
        g = np.dot(f, e)  
        e[i,:] += l * g.flatten()
        
        
    print(e[i,:])
    print(time.clock()-t)  
    """   