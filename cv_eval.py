
import time
from functions import *
from netlaprls import NetLapRLS
from blmnii import BLMNII
from wnngip import WNNGIP
from cmf import CMF
from brdti import BRDTI





def blmnii_cv_eval(method, dataset,output_dir, cv_data, X, D, T, cvs, para):
    max_metric, metric_opt, optArg  = 0, [], []
    for x in np.arange(0, 1.1, 0.1):
        tic = time.clock()
        model = BLMNII(alpha=x, avg=False)
        cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
        print cmd
        aupr_vec, auc_vec, ndcg_vec , ndcg_inv_vec, results = train(model, cv_data, X, D, T)                        
        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        ndcg_avg, ndcg_conf = mean_confidence_interval(ndcg_vec)
        ndcg_inv_avg, ndcg_inv_conf = mean_confidence_interval(ndcg_inv_vec)
        with open(os.path.join(output_dir,"optPar", "proc_"+dataset+"_"+str(cvs)+"_"+method+".txt"), "a") as procFile:
            procFile.write(str(model)+": ")
            procFile.write("auc:%.6f, aupr: %.6f,ndcg: %.6f,ndcg_inv: %.6f, Time:%.6f\n" % (auc_avg, aupr_avg, ndcg_avg, ndcg_inv_avg, time.clock()-tic))

        print "auc:%.6f, aupr: %.6f,ndcg: %.6f,ndcg_inv: %.6f, Time:%.6f\n" % (auc_avg, aupr_avg, ndcg_avg, ndcg_inv_avg, time.clock()-tic)
        metric = ndcg_inv_avg + ndcg_avg
        if metric > max_metric:
            max_metric = metric
            metric_opt= [cmd, auc_avg, aupr_avg, ndcg_avg, ndcg_inv_avg ]
            optArg = {"alpha":x}  
            #each time a better solution is found, the params are stored
            with open(os.path.join(output_dir,"optPar", "res_"+dataset+"_"+str(cvs)+"_"+method+".txt"), "w") as resFile:
                resFile.write(str(optArg)+"\n"+str(metric_opt))
    
    cmd = "Optimal parameter setting:\n%s\n" % metric_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, ndcg:%.6f, ndcg_inv:%.6f\n" % (metric_opt[1], metric_opt[2], metric_opt[3], metric_opt[4])
    print cmd


def wnngip_cv_eval(method, dataset,output_dir, cv_data, X, D, T, cvs, para):   
    max_metric, metric_opt, optArg  = 0, [], []
    for x in np.arange(0.1, 1.1, 0.1):
        for y in np.arange(0.0, 1.1, 0.1):
            tic = time.clock()
            model = WNNGIP(T=x, sigma=1, alpha=y)
            cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
            print cmd
            aupr_vec, auc_vec, ndcg_vec , ndcg_inv_vec, results = train(model, cv_data, X, D, T)                        
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
            ndcg_avg, ndcg_conf = mean_confidence_interval(ndcg_vec)
            ndcg_inv_avg, ndcg_inv_conf = mean_confidence_interval(ndcg_inv_vec)
            with open(os.path.join(output_dir,"optPar", "proc_"+dataset+"_"+str(cvs)+"_"+method+".txt"), "a") as procFile:
                procFile.write(str(model)+": ")
                procFile.write("auc:%.6f, aupr: %.6f,ndcg: %.6f,ndcg_inv: %.6f, Time:%.6f\n" % (auc_avg, aupr_avg, ndcg_avg, ndcg_inv_avg, time.clock()-tic))

            print "auc:%.6f, aupr: %.6f,ndcg: %.6f,ndcg_inv: %.6f, Time:%.6f\n" % (auc_avg, aupr_avg, ndcg_avg, ndcg_inv_avg, time.clock()-tic)
            metric = ndcg_inv_avg + ndcg_avg
            if metric > max_metric:
                max_metric = metric
                metric_opt= [cmd, auc_avg, aupr_avg, ndcg_avg, ndcg_inv_avg ]
                optArg = {"x":x, "y":y}   
                #each time a better solution is found, the params are stored
                with open(os.path.join(output_dir,"optPar", "res_"+dataset+"_"+str(cvs)+"_"+method+".txt"), "w") as resFile:
                    resFile.write(str(optArg)+"\n"+str(metric_opt))
    
    cmd = "Optimal parameter setting:\n%s\n" % metric_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, ndcg:%.6f, ndcg_inv:%.6f\n" % (metric_opt[1], metric_opt[2], metric_opt[3], metric_opt[4])
    print cmd


def netlaprls_cv_eval(method, dataset,output_dir, cv_data, X, D, T, cvs, para): 
    max_metric, metric_opt, optArg  = 0, [], []
    for x in np.arange(-6, 3):  
        for y in np.arange(-6, 3):  
            tic = time.clock()
            model = NetLapRLS(gamma_d=10**(x), gamma_t=10**(x), beta_d=10**(y), beta_t=10**(y))
            cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
            print cmd
            aupr_vec, auc_vec, ndcg_vec , ndcg_inv_vec, results = train(model, cv_data, X, D, T)                        
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
            ndcg_avg, ndcg_conf = mean_confidence_interval(ndcg_vec)
            ndcg_inv_avg, ndcg_inv_conf = mean_confidence_interval(ndcg_inv_vec)
            with open(os.path.join(output_dir,"optPar", "proc_"+dataset+"_"+str(cvs)+"_"+method+".txt"), "a") as procFile:
                procFile.write(str(model)+": ")
                procFile.write("auc:%.6f, aupr: %.6f,ndcg: %.6f,ndcg_inv: %.6f, Time:%.6f\n" % (auc_avg, aupr_avg, ndcg_avg, ndcg_inv_avg, time.clock()-tic))

            print "auc:%.6f, aupr: %.6f,ndcg: %.6f,ndcg_inv: %.6f, Time:%.6f\n" % (auc_avg, aupr_avg, ndcg_avg, ndcg_inv_avg, time.clock()-tic)
            metric = ndcg_inv_avg + ndcg_avg
            if metric > max_metric:
                max_metric = metric
                metric_opt= [cmd, auc_avg, aupr_avg, ndcg_avg, ndcg_inv_avg ]
                optArg = {"x":x, "y":y}         
                #each time a better solution is found, the params are stored
                with open(os.path.join(output_dir,"optPar", "res_"+dataset+"_"+str(cvs)+"_"+method+".txt"), "w") as resFile:
                    resFile.write(str(optArg)+"\n"+str(metric_opt))
    
    cmd = "Optimal parameter setting:\n%s\n" % metric_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, ndcg:%.6f, ndcg_inv:%.6f\n" % (metric_opt[1], metric_opt[2], metric_opt[3], metric_opt[4])
    print cmd
    
    
def cmf_cv_eval(method, dataset,output_dir, cv_data, X, D, T, cvs, para):
    max_metric, metric_opt, optArg  = 0, [], []
    for d in [100]:
        for x in np.arange(-2, 1):
            for y in np.arange(-5, -1):
                for z in np.arange(-5, -1):
                    tic = time.clock()
                    model = CMF(K=d, lambda_l=2**(x), lambda_d=2**(y), lambda_t=2**(z), max_iter=100)
                    cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
                    print cmd
                    aupr_vec, auc_vec, ndcg_vec , ndcg_inv_vec, results = train(model, cv_data, X, D, T)                        
                    aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                    auc_avg, auc_conf = mean_confidence_interval(auc_vec)
                    ndcg_avg, ndcg_conf = mean_confidence_interval(ndcg_vec)
                    ndcg_inv_avg, ndcg_inv_conf = mean_confidence_interval(ndcg_inv_vec)
                    with open(os.path.join(output_dir,"optPar", "proc_"+dataset+"_"+str(cvs)+"_"+method+".txt"), "a") as procFile:
                        procFile.write(str(model)+": ")
                        procFile.write("auc:%.6f, aupr: %.6f,ndcg: %.6f,ndcg_inv: %.6f, Time:%.6f\n" % (auc_avg, aupr_avg, ndcg_avg, ndcg_inv_avg, time.clock()-tic))

                    print "auc:%.6f, aupr: %.6f,ndcg: %.6f,ndcg_inv: %.6f, Time:%.6f\n" % (auc_avg, aupr_avg, ndcg_avg, ndcg_inv_avg, time.clock()-tic)
                    metric = ndcg_inv_avg + ndcg_avg
                    if metric > max_metric:
                        max_metric = metric
                        metric_opt= [cmd, auc_avg, aupr_avg, ndcg_avg, ndcg_inv_avg ]
                        optArg = {"d":d, "x":x, "y":y, "z":z}   
                        #each time a better solution is found, the params are stored
                        with open(os.path.join(output_dir,"optPar", "res_"+dataset+"_"+str(cvs)+"_"+method+".txt"), "w") as resFile:
                            resFile.write(str(optArg)+"\n"+str(metric_opt))
    
    cmd = "Optimal parameter setting:\n%s\n" % metric_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, ndcg:%.6f, ndcg_inv:%.6f\n" % (metric_opt[1], metric_opt[2], metric_opt[3], metric_opt[4])
    print cmd


def brdti_cv_eval(method, dataset,output_dir, cv_data, X, D, T, cvs, para):   
    max_metric, metric_opt, optArg  = 0, [], []
    for d in [50, 100]:
        for lr in [0.1]:
            for glR in [ 0.01, 0.05, 0.1, 0.3]:
                for bR in [1]:
                    for caRU in [  0.05, 0.1, 0.5, 0.9]:#, 1.5
                        for caRI in [1]:                    
                            tic = time.clock()
                            ar = {
                                'D':d,
                                'learning_rate':lr,
                                'max_iters' : 100,                 
                                'bias_regularization':bR,   
                                'simple_predict' :False,  
                                'global_regularization':glR,   
                                "cbSim" : para["cbSim"],
                                'cb_alignment_regularization_user' :caRU,                 
                                'cb_alignment_regularization_item' :caRI}
                            
                                
                            model = BRDTI(ar)                        
                            cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
                            print cmd

                            aupr_vec, auc_vec, ndcg_vec , ndcg_inv_vec, results = train(model, cv_data, X, D, T)                        
                            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
                            ndcg_avg, ndcg_conf = mean_confidence_interval(ndcg_vec)
                            ndcg_inv_avg, ndcg_inv_conf = mean_confidence_interval(ndcg_inv_vec)
                            with open(os.path.join(output_dir,"optPar", "proc_"+dataset+"_"+str(cvs)+"_"+method+".txt"), "a") as procFile:
                                procFile.write(str(model)+": ")
                                procFile.write("auc:%.6f, aupr: %.6f,ndcg: %.6f,ndcg_inv: %.6f, Time:%.6f\n" % (auc_avg, aupr_avg, ndcg_avg, ndcg_inv_avg, time.clock()-tic))

                            print "auc:%.6f, aupr: %.6f,ndcg: %.6f,ndcg_inv: %.6f, Time:%.6f\n" % (auc_avg, aupr_avg, ndcg_avg, ndcg_inv_avg, time.clock()-tic)
                            metric = ndcg_inv_avg + ndcg_avg
                            if metric > max_metric:
                                max_metric = metric
                                metric_opt= [cmd, auc_avg, aupr_avg, ndcg_avg, ndcg_inv_avg ]
                                optArg = {"d":d, "lr":lr, "glR":glR, "caRU":caRU, "caRI":caRI}
                                #each time a better solution is found, the params are stored
                                with open(os.path.join(output_dir,"optPar", "res_"+dataset+"_"+str(cvs)+"_"+method+".txt"), "w") as resFile:
                                    resFile.write(str(optArg)+"\n"+str(metric_opt))
    
    cmd = "Optimal parameter setting:\n%s\n" % metric_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, ndcg:%.6f, ndcg_inv:%.6f\n" % (metric_opt[1], metric_opt[2], metric_opt[3], metric_opt[4])
    print cmd
    
