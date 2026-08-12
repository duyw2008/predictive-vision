#!/usr/bin/env python3
"""C: 增容量 + shift augment few-shot"""
import sys, numpy as np
sys.path.insert(0, '.')
from train_multiscale import ColdEye, load_mnist

def augment(img):
    return [img] + [np.roll(np.roll(img,dy,0),dx,1) for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]]

np.random.seed(42)
X_tr,y_tr,X_te,y_te=load_mnist()
mask_04=y_tr<=4
te_idx=np.where(y_te>=5)[0][:500]

for label,ng,nc in [('v3 100+50',100,50),('x2 200+100',200,100)]:
    print(f'\n{label} (0-4, 30K, 5ep)...', flush=True)
    m=ColdEye(eye_specs=[{'type':'global','n':ng},{'type':'patch','ps':16,'st':8,'n':nc}])
    m.init_templates(X_tr[mask_04][:5000])
    m.train(X_tr[mask_04],y_tr[mask_04],epochs=5,n_train=30000,contrast_aug=True)

    for n_shot in [1,3,5]:
        m.memory.clear()
        for cls in [5,6,7,8,9]:
            idxs=np.where(y_tr==cls)[0]
            chosen=np.random.choice(idxs,min(n_shot,len(idxs)),replace=False)
            for idx in chosen:
                for img in augment(X_tr[idx]):
                    m.memory.append((m._activate_one(img),cls))

        best,best_k=0,0
        for k in [1,3,5]:
            correct=0
            for i in range(500):
                act=m._activate_one(X_te[te_idx[i]])
                scores=[(np.dot(mv,act)/(np.linalg.norm(mv)*np.linalg.norm(act)+1e-8),int(ml))
                        for mv,ml in m.memory]
                scores.sort(key=lambda x:x[0],reverse=True)
                votes=np.bincount([l for _,l in scores[:k]],minlength=10)
                if votes.argmax()==y_te[te_idx[i]]: correct+=1
            acc=correct/500
            if acc>best: best,best_k=acc,k
        print(f'  {n_shot}-shot: {best:.1%} (k={best_k})')

print('\n=== DONE ===')
