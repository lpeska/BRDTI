
import os
import csv
import numpy as np
import rank_metrics as rank
from functions import *

class newDTIPrediction:
    def __init__(self):        
        with open(os.path.join('data','novelDrugsKEGG.csv'), 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar='"')
            kg = np.array(list(reader))            
            t = kg[np.arange(1,kg.shape[0]),0]
            d = kg[np.arange(1,kg.shape[0]),1]
            self.kegg = zip(d,t)
        
        with open(os.path.join('data','novelDrugsDrugBank.csv'), 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar='"')
            db = np.array(list(reader))            
            t = db[np.arange(1,db.shape[0]),1]
            d = db[np.arange(1,db.shape[0]),0]
            self.drugBank = zip(d,t)  
        
        with open(os.path.join('data','novelDrugsMatador.csv'), 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar='"')
            mt = np.array(list(reader))            
            t = mt[np.arange(1,mt.shape[0]),1]
            d = mt[np.arange(1,mt.shape[0]),0]
            self.matador = zip(d,t)  
                   
                    
        #print(self.kegg)
        #print(self.drugBank)
        #print(self.matador)
        
    def analyse_new_known_interactions(self): 
        self.allData =   self.kegg + self.drugBank + self.matador
        for dataset in ["gpcr","ic","nr","e"]:
            drug_names, target_names = get_drugs_targets_names(dataset, os.path.join("data", 'datasets'))
            new_interactions = set([s for s in self.allData if any(s[0] in d for d in drug_names) and any(s[1] in t.replace("hsa","hsa:") for t in target_names)])
            print(len(self.allData),len(drug_names),len(target_names))
            print(dataset)
            print(len(new_interactions))
            #print(new_interactions)
            
    def verify_novel_interactions(self, method, dataset, sz, predict_num, drug_names, target_names):    
        drugs = np.unique(sz[:,0])
        targets = np.unique(sz[:,1])
        self.drugs_ndcg = []
        self.targets_ndcg = []
        self.drugs_recall = []
        self.targets_recall = []
        
        new_dti_drugs = os.path.join('output/newDTI', "_".join([method, dataset,str(predict_num), "drugs_new_dti.csv"]))
        out_dti_d = open(new_dti_drugs, "w")
        out_dti_d.write(('drug;target;score;hit;kegg_hit;drugBank_hit;matador_hit\n'))

        new_dti_targets = os.path.join('output/newDTI', "_".join([method, dataset,str(predict_num), "targets_new_dti.csv"]))
        out_dti_t = open(new_dti_targets, "w")
        out_dti_t.write(('drug;target;score;hit;kegg_hit;drugBank_hit;matador_hit\n'))

        drug_stats = os.path.join('output/newDTI', "_".join([method, dataset,str(predict_num), "drug_stats.csv"]))
        outd = open(drug_stats, "w")
        outd.write(('drug;hits;possible_hits;ndcg;recall;total_known_targets\n'))

        target_stats = os.path.join('output/newDTI', "_".join([method, dataset,str(predict_num), "target_stats.csv"]))
        outt = open(target_stats, "w")
        outt.write(('target;hits;possible_hits;ndcg;recall;total_known_targets\n'))
        
        self.allData = self.kegg + self.drugBank + self.matador
        self.dataset_new_interactions = set([s for s in self.allData if any(s[0] in d for d in drug_names) and any(s[1] in t.replace("hsa","hsa:") for t in target_names)])

        
        for d in drugs:
            dti_score = sz[sz[:,0] == d]
            dti_score = dti_score[dti_score[:,2].argsort()[::-1]]
            pred_dti = [(drug_names[int(dti_score[i,0])], target_names[int(dti_score[i,1])], dti_score[i,2]) for i in np.arange(0, predict_num)]
            self.novel_prediction_analysis(pred_dti, drug_names[int(d)], "NA")                
            out_dti_d.write(''.join('%s;%s;%f;%i;%i;%i;%i \n' % x for x in self.eval_dti_pairs))
            outd.write(''.join('%s;%i;%i;%f;%f;%i \n' % self.eval_drugs))                        
        print("finish: per-drug evaluation, ndcg: %f recall: %f " % (np.nanmean(self.drugs_ndcg), np.nanmean(self.drugs_recall)) )  
       
        
        
        for t in targets:
            dti_score = sz[sz[:,1] == t]            
            dti_score = dti_score[dti_score[:,2].argsort()[::-1]]
            pred_dti = [(drug_names[int(dti_score[i,0])], target_names[int(dti_score[i,1])], dti_score[i,2]) for i in np.arange(0, predict_num)]
            self.novel_prediction_analysis(pred_dti, "NA", target_names[int(t)])      
            out_dti_t.write(''.join('%s;%s;%f;%i;%i;%i;%i \n' % x for x in self.eval_dti_pairs))
            outt.write(''.join('%s;%i;%i;%f;%f;%i \n' % self.eval_targets))                        
        print("finish: per-target evaluation,  ndcg: %f recall: %f " % ( np.nanmean(self.targets_ndcg), np.nanmean(self.targets_recall)) ) 
        
        return (np.nanmean(self.drugs_ndcg),np.nanmean(self.targets_ndcg),np.nanmean(self.drugs_recall),np.nanmean(self.targets_recall))
    
    def novel_prediction_analysis(self,dti_pairs, drug, target):   
        eval_dti_pairs = []
        hit_list = []
        for num in xrange(len(dti_pairs)):
            kg_hit, db_hit, mt_hit, hit = 0,0,0,0
            d, t, score = dti_pairs[num]
            dtp = (d,t.replace("hsa","hsa:"))
            #print(dtp)
            if dtp in self.kegg:
                kg_hit = 1
            if dtp in self.drugBank:
                db_hit = 1
            if dtp in self.matador:
                mt_hit = 1
            hit = max(kg_hit,db_hit,mt_hit)
            eval_dti_pairs.append((d,t,score,hit,kg_hit,db_hit,mt_hit))
            hit_list.append(hit)
            
        self.eval_dti_pairs = eval_dti_pairs    
        if drug != "NA":
            kt_set = set([dti for dti in self.dataset_new_interactions if dti[0] == drug])
            total_known_DTI = len(kt_set)
            ndcg_d = self.ndcg(hit_list,total_known_DTI)
            self.drugs_ndcg.append(ndcg_d)
            recall_d = sum(hit_list)/float(total_known_DTI) if total_known_DTI > 0 else float('nan')
            self.drugs_recall.append(recall_d)
            self.eval_drugs = (drug,sum(hit_list),min(total_known_DTI,len(hit_list)),ndcg_d,recall_d,total_known_DTI)

        if target != "NA":
            target = target.replace("hsa","hsa:")
            kt_set = set([dti for dti in self.dataset_new_interactions if dti[1] == target])
            total_known_DTI = len(kt_set)
            ndcg_t = self.ndcg(hit_list,total_known_DTI)
            self.targets_ndcg.append(ndcg_t)
            recall_t = sum(hit_list)/float(total_known_DTI) if total_known_DTI > 0 else float('nan')
            self.targets_recall.append(recall_t)
            self.eval_targets = (target,sum(hit_list),min(total_known_DTI,len(hit_list)),ndcg_t,recall_t,total_known_DTI)
             
    def ndcg(self,hit_list,total_known_DTI):
        if total_known_DTI == 0:
            return float('nan')
        else:
            if total_known_DTI >= len(hit_list):
                ideal_list = [1 for number in xrange(len(hit_list))]
            else:
                ideal_list =[1 for number in xrange(total_known_DTI)]+[ 0  for number in xrange(len(hit_list)-total_known_DTI)]
            return rank.dcg_at_k(hit_list,len(hit_list),1)/rank.dcg_at_k(ideal_list,len(hit_list),1)


if __name__ == "__main__":
    d = new_pairs()
    d.analyse_new_known_interactions()
    
    

