
import os, sys, json, csv, random, argparse
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module2_model.model import build_model
from module3_training.train import (
    BrainTumorDataset, CLASS_NAMES, CLASS_TO_IDX,
    NUM_CLASSES, HARMONIZED_DIR, MODELS_DIR, RANDOM_SEED
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "outputs", "results")
KAGGLE_NOTUMOR_TEST_DIR = os.path.join(PROJECT_ROOT, "outputs_v1_224", "deduplicated", "kaggle", "test")
USE_TTA = False

PROTOCOLS = {
    "merged": {"checkpoint": os.path.join(MODELS_DIR,"merged","best_model.pth"),
               "test_dir": os.path.join(HARMONIZED_DIR,"merged_baseline","test"),
               "description": "Merged Baseline", "lodo_round2": False},
    "lodo1":  {"checkpoint": os.path.join(MODELS_DIR,"lodo1","best_model.pth"),
               "test_dir": os.path.join(HARMONIZED_DIR,"lodo_round1","test"),
               "description": "LODO Round 1 (Test: Kaggle)", "lodo_round2": False},
    "lodo2":  {"checkpoint": os.path.join(MODELS_DIR,"lodo2","best_model.pth"),
               "test_dir": os.path.join(HARMONIZED_DIR,"lodo_round2","test"),
               "description": "LODO Round 2 (Test: Figshare)", "lodo_round2": True},
    "lodo3":  {"checkpoint": os.path.join(MODELS_DIR,"lodo3","best_model.pth"),
               "test_dir": os.path.join(HARMONIZED_DIR,"lodo_round3","test"),
               "description": "LODO Round 3 (Test: Mendeley)", "lodo_round2": False},
}

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    img_size = ckpt.get('img_size', 224)
    print(f'  Loaded: {path}\n  img_size={img_size}px, epoch={ckpt.get("epoch","?")}, val_acc={ckpt.get("val_acc",0):.4f}')
    model = build_model(num_classes=NUM_CLASSES, pretrained=False, img_size=img_size)
    model.load_state_dict(ckpt['model_state'])
    return model.to(device).eval(), ckpt, img_size

@torch.no_grad()
def run_inference(model, loader, device, present_classes):
    """Standard single-pass inference."""
    model.eval()
    vit_preds, cnn_preds, vit_probs_list, true_list = [], [], [], []
    for images, labels in tqdm(loader, desc='  inference', leave=False):
        images = images.to(device)
        out = model(images, return_attention=False)
        vit_probs_list.append(F.softmax(out['vit_logits'], dim=-1).cpu().numpy())
        vit_preds.append(out['vit_logits'].argmax(1).cpu().numpy())
        cnn_preds.append(out['cnn_logits'].argmax(1).cpu().numpy())
        true_list.append(labels.numpy())
    return {
        'vit_preds': np.concatenate(vit_preds), 'cnn_preds': np.concatenate(cnn_preds),
        'vit_probs': np.concatenate(vit_probs_list), 'true_labels': np.concatenate(true_list),
        'present_classes': present_classes,
    }

@torch.no_grad()
def run_inference_tta(model, loader, device, present_classes):
    """
    Test-Time Augmentation: 5 views (original, h-flip, v-flip, +10°, -10°).
    Averages softmax outputs across all views for ViT head.
    CNN head uses original view only (TTA applies to ViT predictions).
    Expected improvement: +1-2pp accuracy without any retraining.
    """
    model.eval()
    print('  Running TTA (5 views: original + h-flip + v-flip + rot+10 + rot-10)...')
    all_views_probs = []
    cnn_preds_list, true_list = [], []

    for images, labels in tqdm(loader, desc='  TTA', leave=False):
        images = images.to(device)
        batch_probs = []

        out = model(images, return_attention=False)
        batch_probs.append(F.softmax(out['vit_logits'], dim=-1).cpu())
        cnn_preds_list.append(out['cnn_logits'].argmax(1).cpu().numpy())
        true_list.append(labels.numpy())

        out = model(torch.flip(images, dims=[3]), return_attention=False)
        batch_probs.append(F.softmax(out['vit_logits'], dim=-1).cpu())

        out = model(torch.flip(images, dims=[2]), return_attention=False)
        batch_probs.append(F.softmax(out['vit_logits'], dim=-1).cpu())

        rotated_p = torch.stack([TF.rotate(img, 10) for img in images])
        out = model(rotated_p, return_attention=False)
        batch_probs.append(F.softmax(out['vit_logits'], dim=-1).cpu())

        rotated_m = torch.stack([TF.rotate(img, -10) for img in images])
        out = model(rotated_m, return_attention=False)
        batch_probs.append(F.softmax(out['vit_logits'], dim=-1).cpu())

        avg_probs = torch.stack(batch_probs, dim=0).mean(dim=0)
        all_views_probs.append(avg_probs.numpy())

    avg_probs = np.concatenate(all_views_probs, axis=0)
    return {
        'vit_preds': avg_probs.argmax(axis=1),
        'cnn_preds': np.concatenate(cnn_preds_list),
        'vit_probs': avg_probs,
        'true_labels': np.concatenate(true_list),
        'present_classes': present_classes,
    }

def compute_metrics(results, present_classes):
    y_true = results['true_labels']; vit_preds = results['vit_preds']
    cnn_preds = results['cnn_preds']; vit_probs = results['vit_probs']
    present_idx = [CLASS_TO_IDX[c] for c in present_classes]
    vit_acc = accuracy_score(y_true, vit_preds)
    per_class = {}
    for cls in present_classes:
        idx = CLASS_TO_IDX[cls]
        tp=int(np.sum((y_true==idx)&(vit_preds==idx))); fp=int(np.sum((y_true!=idx)&(vit_preds==idx)))
        fn=int(np.sum((y_true==idx)&(vit_preds!=idx))); tn=int(np.sum((y_true!=idx)&(vit_preds!=idx)))
        prec=tp/(tp+fp) if (tp+fp)>0 else 0; rec=tp/(tp+fn) if (tp+fn)>0 else 0
        f1=2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        try: auc_val=roc_auc_score((y_true==idx).astype(int), vit_probs[:,idx])
        except: auc_val=float('nan')
        per_class[cls]={'precision':round(prec,4),'recall':round(rec,4),'f1':round(f1,4),'auc':round(auc_val,4),
                        'support':int(np.sum(y_true==idx)),'tp':tp,'fp':fp,'fn':fn,'tn':tn}
    macro_f1=np.mean([per_class[c]['f1'] for c in present_classes])
    valid_aucs=[per_class[c]['auc'] for c in present_classes if not np.isnan(per_class[c]['auc'])]
    macro_auc=float(np.mean(valid_aucs)) if valid_aucs else float('nan')
    cm=confusion_matrix(y_true,vit_preds,labels=present_idx)
    cnn_acc=accuracy_score(y_true,cnn_preds)
    cnn_f1=f1_score(y_true,cnn_preds,labels=present_idx,average='macro',zero_division=0)
    cnn_cm=confusion_matrix(y_true,cnn_preds,labels=present_idx)
    return {
        'vit':{'accuracy':round(float(vit_acc),4),'macro_precision':round(float(np.mean([per_class[c]['precision'] for c in present_classes])),4),
               'macro_recall':round(float(np.mean([per_class[c]['recall'] for c in present_classes])),4),
               'macro_f1':round(float(macro_f1),4),'macro_auc':round(float(macro_auc),4),
               'per_class':per_class,'confusion_matrix':cm.tolist()},
        'cnn':{'accuracy':round(float(cnn_acc),4),'macro_f1':round(float(cnn_f1),4),'confusion_matrix':cnn_cm.tolist()},
        'n_samples':int(len(y_true)),'present_classes':present_classes,
    }

def plot_confusion_matrix(cm, labels, path, title):
    fig,axes=plt.subplots(1,2,figsize=(14,6)); fig.suptitle(title,fontsize=13,fontweight='bold')
    for ax,(mat,norm,sub) in zip(axes,[(cm,False,'Absolute'),(cm.astype(float)/(cm.sum(axis=1,keepdims=True)+1e-8),True,'Normalised')]):
        im=ax.imshow(mat,interpolation='nearest',cmap='Blues' if norm else 'YlOrRd'); plt.colorbar(im,ax=ax,shrink=0.8)
        ax.set_title(sub);ax.set_xlabel('Predicted');ax.set_ylabel('True')
        cap=[c.capitalize() for c in labels]; ax.set_xticks(range(len(labels)));ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(cap,rotation=30,ha='right',fontsize=9);ax.set_yticklabels(cap,fontsize=9)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v=mat[i,j];t=f'{v:.2f}' if norm else f'{int(v)}'
                ax.text(j,i,t,ha='center',va='center',fontsize=9,color='white' if v>mat.max()*0.6 else 'black',fontweight='bold')
    plt.tight_layout();plt.savefig(path,dpi=150,bbox_inches='tight');plt.close()

def plot_roc_curves(results, present_classes, path, title):
    y=results['true_labels'];p=results['vit_probs'];colors=['#2196F3','#F44336','#4CAF50','#FF9800']
    fig,ax=plt.subplots(figsize=(8,7));aucs=[]
    for i,cls in enumerate(present_classes):
        idx=CLASS_TO_IDX[cls];yb=(y==idx).astype(int)
        try:
            fpr,tpr,_=roc_curve(yb,p[:,idx]);a=auc(fpr,tpr);aucs.append(a)
            ax.plot(fpr,tpr,color=colors[i%4],lw=2,label=f'{cls.capitalize()} (AUC={a:.3f})')
        except:pass
    ax.plot([0,1],[0,1],'k--',lw=1);ax.set_xlim([0,1]);ax.set_ylim([0,1.05])
    ax.set_xlabel('FPR');ax.set_ylabel('TPR');ax.set_title(title,fontweight='bold')
    if aucs:ax.text(0.6,0.12,f'Macro AUC={np.mean(aucs):.3f}',fontsize=11,fontweight='bold',bbox=dict(boxstyle='round',facecolor='lightyellow',alpha=0.8))
    ax.legend(loc='lower right');ax.grid(alpha=0.3);plt.tight_layout();plt.savefig(path,dpi=150,bbox_inches='tight');plt.close()

def plot_comparison(all_metrics, path):
    protocols=list(all_metrics.keys());acc=[all_metrics[p]['vit']['accuracy']*100 for p in protocols]
    f1=[all_metrics[p]['vit']['macro_f1']*100 for p in protocols]
    x=np.arange(len(protocols));w=0.35;labs=['Merged','LODO R1','LODO R2','LODO R3']
    fig,ax=plt.subplots(figsize=(12,7))
    ax.bar(x-w/2,acc,w,label='Accuracy',color='#2196F3',alpha=0.85)
    ax.bar(x+w/2,f1,w,label='Macro F1',color='#4CAF50',alpha=0.85)
    tta_str=' (TTA)' if USE_TTA else ''
    ax.set_title(f'v2 Performance{tta_str}',fontweight='bold');ax.set_xticks(x);ax.set_xticklabels(labs)
    ax.set_ylim([min(acc+f1)-5,102]);ax.legend();ax.grid(axis='y',alpha=0.3)
    plt.tight_layout();plt.savefig(path,dpi=150,bbox_inches='tight');plt.close()

def save_report(protocol, metrics, path, img_size):
    m=metrics['vit'];c=metrics['cnn'];pcs=metrics['present_classes']
    tta_str=' + TTA' if USE_TTA else ''
    lines=['='*68,f'  EVALUATION — {protocol.upper()} (v2: {img_size}px{tta_str})','='*68,'',
           f'  Test: {metrics["n_samples"]} images | Classes: {pcs}','',
           f'  ViT Accuracy:  {m["accuracy"]*100:.2f}%',f'  ViT Macro F1:  {m["macro_f1"]*100:.2f}%',
           f'  ViT Macro AUC: {m["macro_auc"]*100:.2f}%','',
           f'  {"Class":<14} {"P":>6} {"R":>6} {"F1":>6} {"AUC":>6} {"N":>5}','  '+'-'*45]
    for cls in pcs:
        pc=m['per_class'][cls]
        lines.append(f'  {cls:<14} {pc["precision"]*100:>5.1f}% {pc["recall"]*100:>5.1f}% {pc["f1"]*100:>5.1f}% {pc["auc"]*100:>5.1f}% {pc["support"]:>5}')
    lines+=['',f'  CNN Accuracy: {c["accuracy"]*100:.2f}% | CNN F1: {c["macro_f1"]*100:.2f}%',
            f'  ViT-CNN gap: {(m["accuracy"]-c["accuracy"])*100:+.2f}pp','='*68]
    report='\n'.join(lines)
    with open(path,'w',encoding='utf-8') as f:f.write(report)
    print(report)

def save_csv(all_metrics, path):
    rows=[]
    for p,m in all_metrics.items():
        v=m['vit'];c=m['cnn']
        row={'protocol':p,'n':m['n_samples'],'vit_acc':v['accuracy'],'vit_f1':v['macro_f1'],
             'vit_auc':v['macro_auc'],'cnn_acc':c['accuracy'],'gap':round(v['accuracy']-c['accuracy'],4)}
        rows.append(row)
    if rows:
        with open(path,'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

@torch.no_grad()
def eval_notumor_lodo2(model, device, img_size):
    nd=os.path.join(KAGGLE_NOTUMOR_TEST_DIR,"notumor")
    if not os.path.isdir(nd):
        alt=os.path.join(PROJECT_ROOT,"outputs_v1_224","deduplicated","kaggle","test")
        nd=os.path.join(alt,"notumor") if os.path.isdir(os.path.join(alt,"notumor")) else None
    if not nd:print('  [SKIP] No notumor dir');return {}
    tf=transforms.Compose([transforms.Resize((img_size,img_size)),transforms.ToTensor(),
                            transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    ds=BrainTumorDataset(os.path.dirname(nd),transform=tf,classes=['notumor'])
    if len(ds)==0:return {}
    loader=DataLoader(ds,batch_size=32,shuffle=False,num_workers=4,pin_memory=True)
    ni=CLASS_TO_IDX['notumor'];tp=fn=0
    for imgs,_ in loader:
        for p in model(imgs.to(device))['vit_logits'].argmax(1).cpu().numpy():
            if p==ni:tp+=1
            else:fn+=1
    tot=tp+fn;acc=tp/tot if tot>0 else 0
    print(f'  No Tumor: {tp}/{tot} ({acc*100:.2f}%)')
    return {'n_samples':tot,'accuracy':round(acc,4)}

def evaluate_protocol(protocol, device):
    cfg=PROTOCOLS[protocol];out_dir=os.path.join(RESULTS_DIR,protocol);os.makedirs(out_dir,exist_ok=True)
    tta_str=' + TTA' if USE_TTA else ''
    print(f'\n{"="*68}\n  EVAL: {protocol.upper()}{tta_str}\n{"="*68}')
    model,ckpt,img_size=load_checkpoint(cfg['checkpoint'],device)
    tf=transforms.Compose([transforms.Resize((img_size,img_size)),transforms.ToTensor(),
                            transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    test_ds=BrainTumorDataset(cfg['test_dir'],transform=tf)
    present=list(test_ds.class_counts.keys())
    print(f'  Test: {len(test_ds)} images, classes: {present}')
    loader=DataLoader(test_ds,batch_size=32,shuffle=False,num_workers=4,pin_memory=True)

    if USE_TTA:
        results=run_inference_tta(model,loader,device,present)
    else:
        results=run_inference(model,loader,device,present)

    metrics=compute_metrics(results,present)
    metrics.update({'protocol':protocol,'description':cfg['description'],'img_size':img_size,'tta':USE_TTA})

    if cfg['lodo_round2'] and 'meningioma' in present:
        rng=np.random.default_rng(42);idx_m=CLASS_TO_IDX['meningioma'];f1s=[]
        for _ in range(1000):
            si=rng.choice(len(results['true_labels']),size=len(results['true_labels']),replace=True)
            yt=results['true_labels'][si];yp=results['vit_preds'][si]
            tp=np.sum((yt==idx_m)&(yp==idx_m));fp=np.sum((yt!=idx_m)&(yp==idx_m));fn=np.sum((yt==idx_m)&(yp!=idx_m))
            p=tp/(tp+fp+1e-9);r=tp/(tp+fn+1e-9);f1s.append(2*p*r/(p+r+1e-9))
        metrics['bootstrap_ci_meningioma']={'f1_lo':round(float(np.percentile(f1s,2.5)),4),'f1_hi':round(float(np.percentile(f1s,97.5)),4)}

    if cfg['lodo_round2']:
        nt=eval_notumor_lodo2(model,device,img_size)
        if nt:metrics['notumor_separate_eval']=nt

    with open(os.path.join(out_dir,'test_metrics.json'),'w') as f:json.dump(metrics,f,indent=2)
    cm_v=np.array(metrics['vit']['confusion_matrix'])
    plot_confusion_matrix(cm_v,present,os.path.join(out_dir,'confusion_matrix.png'),f'{protocol.upper()} ViT{tta_str}')
    cm_c=np.array(metrics['cnn']['confusion_matrix'])
    plot_confusion_matrix(cm_c,present,os.path.join(out_dir,'confusion_matrix_cnn.png'),f'{protocol.upper()} CNN')
    plot_roc_curves(results,present,os.path.join(out_dir,'roc_curves.png'),f'ROC {protocol.upper()}{tta_str}')
    save_report(protocol,metrics,os.path.join(out_dir,'test_metrics_summary.txt'),img_size)
    v=metrics['vit']
    print(f'\n  {protocol.upper()}: Acc={v["accuracy"]*100:.2f}% F1={v["macro_f1"]*100:.2f}% AUC={v["macro_auc"]*100:.2f}%')
    return metrics

def main():
    global USE_TTA
    parser=argparse.ArgumentParser(description='Module 4 — Evaluate (v2.1: TTA support)')
    parser.add_argument('--protocol',default='all',choices=['all','merged','lodo1','lodo2','lodo3'])
    parser.add_argument('--tta',action='store_true',help='Enable Test-Time Augmentation (5 views)')
    args=parser.parse_args()
    USE_TTA=args.tta

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n  Device: {device}')
    if device.type=='cuda':print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  TTA: {"ON (5 views)" if USE_TTA else "OFF"}')

    protocols=['merged','lodo1','lodo2','lodo3'] if args.protocol=='all' else [args.protocol]
    os.makedirs(RESULTS_DIR,exist_ok=True)

    all_m={}
    for p in protocols:
        try:all_m[p]=evaluate_protocol(p,device)
        except Exception as e:
            import traceback;traceback.print_exc();print(f'  [ERROR] {p}: {e}')

    if len(all_m)>1:
        save_csv(all_m,os.path.join(RESULTS_DIR,'all_protocols_comparison.csv'))
        plot_comparison(all_m,os.path.join(RESULTS_DIR,'all_protocols_comparison.png'))

    tta_str=' + TTA' if USE_TTA else ''
    print(f'\n{"="*68}\n  EVAL COMPLETE{tta_str}\n{"="*68}')
    print(f'  {"Proto":<12} {"ViT Acc":>8} {"ViT F1":>8} {"CNN Acc":>8} {"Gap":>6}')
    print('  '+'-'*45)
    for p,m in all_m.items():
        v=m['vit'];c=m['cnn']
        print(f'  {p.upper():<12} {v["accuracy"]*100:>7.2f}% {v["macro_f1"]*100:>7.2f}% {c["accuracy"]*100:>7.2f}% {(v["accuracy"]-c["accuracy"])*100:>+5.2f}')
    print('='*68)
