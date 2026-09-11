import torch
import torch.nn.functional as F
import cv2
import numpy as np
from .bbox import *

def detect(net, img, device):
    img = img - np.array([104,117,123])
    img = img.transpose(2,0,1).reshape((1,)+img.transpose(2,0,1).shape)
    if 'cuda' in device:
        torch.backends.cudnn.benchmark=True
    img=torch.from_numpy(img).float().to(device)
    with torch.no_grad():
        olist=net(img)
    bboxlist=[]
    for i in range(len(olist)//2):
        olist[i*2]=F.softmax(olist[i*2],dim=1)
    olist=[o.data.cpu() for o in olist]
    for i in range(len(olist)//2):
        ocls,oreg=olist[i*2],olist[i*2+1]
        stride=2**(i+2)
        poss=zip(*np.where(ocls[:,1,:,:]>0.05))
        for _,hindex,windex in poss:
            axc,ayc=stride/2+windex*stride,stride/2+hindex*stride
            score=ocls[0,1,hindex,windex]
            loc=oreg[0,:,hindex,windex].contiguous().view(1,4)
            priors=torch.Tensor([[axc,ayc,stride*4,stride*4]])
            box=decode(loc,priors,[0.1,0.2])
            x1,y1,x2,y2=box[0]
            bboxlist.append([x1,y1,x2,y2,score])
    bboxlist=np.array(bboxlist)
    if len(bboxlist)==0:
        bboxlist=np.zeros((1,5))
    return bboxlist

def batch_detect(net, imgs, device):
    imgs=imgs-np.array([104,117,123])
    imgs=imgs.transpose(0,3,1,2)
    if 'cuda' in device:
        torch.backends.cudnn.benchmark=True
    imgs=torch.from_numpy(imgs).float().to(device)
    BB=imgs.size(0)
    with torch.no_grad():
        olist=net(imgs)
    bboxlist=[]
    for i in range(len(olist)//2):
        olist[i*2]=F.softmax(olist[i*2],dim=1)
    olist=[o.cpu() for o in olist]
    for i in range(len(olist)//2):
        ocls,oreg=olist[i*2],olist[i*2+1]
        stride=2**(i+2)
        poss=zip(*np.where(ocls[:,1,:,:]>0.05))
        for _,hindex,windex in poss:
            axc,ayc=stride/2+windex*stride,stride/2+hindex*stride
            score=ocls[:,1,hindex,windex]
            loc=oreg[:,:,hindex,windex].contiguous().view(BB,1,4)
            priors=torch.Tensor([[axc,ayc,stride*4,stride*4]]).view(1,1,4)
            box=batch_decode(loc,priors,[0.1,0.2])[:,0]
            bboxlist.append(torch.cat([box,score.unsqueeze(1)],1).cpu().numpy())
    bboxlist=np.array(bboxlist)
    if len(bboxlist)==0:
        bboxlist=np.zeros((1,BB,5))
    return bboxlist

def flip_detect(net,img,device):
    img=cv2.flip(img,1)
    b=detect(net,img,device)
    bboxlist=np.zeros(b.shape)
    bboxlist[:,0]=img.shape[1]-b[:,2]
    bboxlist[:,1]=b[:,1]
    bboxlist[:,2]=img.shape[1]-b[:,0]
    bboxlist[:,3]=b[:,3]
    bboxlist[:,4]=b[:,4]
    return bboxlist

def pts_to_bb(pts):
    min_x,min_y=np.min(pts,axis=0); max_x,max_y=np.max(pts,axis=0)
    return np.array([min_x,min_y,max_x,max_y])
