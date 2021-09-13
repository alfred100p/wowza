import tensorflow_hub as hub
import system
import tensorflow as tf
import time
import numpy as np


pat='wowza/'

thresh = sys.argv[0]
#getting model
module_handle ="https://tfhub.dev/tensorflow/efficientdet/d7/1"
detector = hub.load(module_handle)

def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img

def run_detector(detector, path,threshold):
  img = load_img(path)
  converted_img  = tf.image.convert_image_dtype(img, tf.uint8)[tf.newaxis, ...]
  start_time = time.time()
  result = detector(converted_img)
  end_time = time.time()
  result = {key:value.numpy() for key,value in result.items()}
  n=result['detection_scores']
  n2=len(n[n>threshold])
  #print("Found %d objects." % len(result["num_detections"]))
  #print("Inference time: ", end_time-start_time)

  #image_with_boxes = draw_boxes(       img.numpy(), 
  #print(result["detection_boxes"])#,      result["detection_class_entities"], result["detection_scores"])
  r=[]
  #get center of box
  for el in result['detection_boxes'].reshape(int(result['num_detections'][0]),4):
    a=(el[0]+el[2])/2
    b=(el[1]+el[3])/2
    r+=[a,b]
  
  
  return result['detection_classes'],result['detection_scores'],r,n2

#storing detection data from model
def get_info(img,threshold=0.1,itr=0):
  '''
    img: image file name
    threshold: detection score threshold
    itr: index of iteration
  '''
  id=img[-11:-4]
  c=np.array([100])
  b=np.array([200])
  s=np.array([100])
  

  

  i=0
  c,s,b,n2=run_detector(detector,im_path+img,threshold)
  n3=n2*2#apply detecion_score threshold
  c=np.append(c[0,:n2],np.zeros([100-n2])-1)#assigning 0 to all below threshold
  b=np.append(b[:n3],np.zeros([200-n3])-1)
  s=np.append(s[0,:n2],np.zeros([100-n2])-1)
  if itr%100==0:#printing to check if errors are present like last time
    print(itr)
    print(c)
  np.save(dir+'c/'+id+'.npy',c)
  np.save(dir+'b/'+id+'.npy',b)
  np.save(dir+'s/'+id+'.npy',s)
  return 1
splits=['train','val','test']
i=0
l=[]
for split in splits:
  dir=path+split[:2]+'vec/'
  os.mkdir(pat+split[:2]+vec+str(thresh))
  os.mkdir(pat+split[:2]+vec+str(thresh)+'/C')
  os.mkdir(pat+split[:2]+vec+str(thresh)+'/B')
  os.mkdir(pat+split[:2]+vec+str(thresh)+'/S')
  im_path='images/MS-COCO-2014/'+split+'2014/'
  for el in os.listdir(im_path):
    get_info(img=el,threshold=thresh,itr=i)
    
    i+=1
   
import glob
file_paths = glob.glob(os.path.join(path+'trvec'+str(thresh)+'/C/', '*.npy'))
C=np.ndarray([len(file_paths),100])
i=0
l=[]
for el in file_paths:
  C[i]=np.load(el)
  l+=[el[-11:-4]]
  i+=1
  if i%100==0:
    print(i)
C=np.ndarray([len(file_paths),100])
S=np.ndarray([len(file_paths),100])
B=np.ndarray([len(file_paths),200])
i=0
for el in f:
  el=el[-11:-4]
  C[i]=np.load(path+'trvec'+str(thresh)+'/C/'+el+'.npy')
  B[i]=np.load(path+'trvec'+str(thresh)+'/B/'+el+'.npy')
  S[i]=np.load(path+'trvec'+str(thresh)+'/S/'+el+'.npy')
  i+=1
  if i%100==0:
    print(i)
u=np.unique(C)
dicti=dict(zip(np.unique(C),np.concatenate([[-1],range(len(np.unique(C))-1)])))
np.save(path+'trvec'+str(thresh)+'C2.npy',C)
np.save(path+'trvec'+str(thresh)+'B2.npy',B)
np.save(path+'trvec'+str(thresh)+'S2.npy',S)
c_d=[]
for el in C:
  f=[]
  for e in el:
    f+=[dicti[e]]
  c_d+=[f]
nC=len(np.unique(C))-1

b=B
s=S
i=0
tab1=np.zeros([nC,nC,4])
tab2=np.zeros([nC],dtype=np.int32)
asq=0
while i<len(C):
  k=0
  f=[]
  while k<99:
    if c_d[i][k] in f:
      k+=1
      
    else:
      j=k+1
      f+=[c_d[i][k]]
      f2=f
      while j<100:
        if c_d[i][j] in f2:
          j+=1
          
        else:
          f2+=[c_d[i][j]]
          a=0
          #detecting which quadrant and choosing part of feature vector
          if b[i][2*k]>b[i][2*j]:
            a+=2
          if b[i][2*k+1]>b[i][2*j+1]:
            a+=1  
          tab1[c_d[i][k],c_d[i][j],a]+=s[i][k]*s[i][j]
          tab1[c_d[i][j],c_d[i][k],3-a]+=s[i][k]*s[i][j]
          tab2[c_d[i][k]]+=1
          
          j+=1
      k+=1
  if i%100==0:
    print(i)
  i+=1
  asq+=len(f)
asq

cat=np.zeros([12])

cat[0]=tab2[0]
cat[1]=np.sum(tab2[1:9])
cat[2]=np.sum(tab2[9:14])
cat[3]=np.sum(tab2[14:24])
cat[4]=np.sum(tab2[24:29])
cat[5]=np.sum(tab2[29:39])
cat[6]=np.sum(tab2[39:46])
cat[7]=np.sum(tab2[46:56])
cat[8]=np.sum(tab2[56:62])
cat[9]=np.sum(tab2[62:69])
cat[10]=np.sum(tab2[69:74])
cat[11]=np.sum(tab2[74:80])

ncat=np.zeros([12,80,4])
ncat[0]=tab1[0]
ncat[1]=np.sum(tab1[1:9],axis=0)
ncat[2]=np.sum(tab1[9:14],axis=0)
ncat[3]=np.sum(tab1[14:24],axis=0)
ncat[4]=np.sum(tab1[24:29],axis=0)
ncat[5]=np.sum(tab1[29:39],axis=0)
ncat[6]=np.sum(tab1[39:46],axis=0)
ncat[7]=np.sum(tab1[46:56],axis=0)
ncat[8]=np.sum(tab1[56:62],axis=0)
ncat[9]=np.sum(tab1[62:69],axis=0)
ncat[10]=np.sum(tab1[69:74],axis=0)
ncat[11]=np.sum(tab1[74:80],axis=0)

ncat1=np.empty(ncat.shape)
i=0
for el in ncat:
  j=0
  for e in el:
    t=np.sum(e)
    if t==0:
      ncat1[i][j]=[0,0,0,0]
    else:
      ncat1[i][j]=e/t
    j+=1
  i+=1
  
  
ncat2=np.zeros([12,80*4])
ncat2[0]=tab1[0].reshape([-1,320])
ncat2[1]=np.sum(tab1[1:9].reshape([-1,320]),axis=0)
ncat2[2]=np.sum(tab1[9:14].reshape([-1,320]),axis=0)
ncat2[3]=np.sum(tab1[14:24].reshape([-1,320]),axis=0)
ncat2[4]=np.sum(tab1[24:29].reshape([-1,320]),axis=0)
ncat2[5]=np.sum(tab1[29:39].reshape([-1,320]),axis=0)
ncat2[6]=np.sum(tab1[39:46].reshape([-1,320]),axis=0)
ncat2[7]=np.sum(tab1[46:56].reshape([-1,320]),axis=0)
ncat2[8]=np.sum(tab1[56:62].reshape([-1,320]),axis=0)
ncat2[9]=np.sum(tab1[62:69].reshape([-1,320]),axis=0)
ncat2[10]=np.sum(tab1[69:74].reshape([-1,320]),axis=0)
ncat2[11]=np.sum(tab1[74:80].reshape([-1,320]),axis=0)  


nncat1=np.empty([12,nC*4])
i=0
for el in ncat1:
  j=0
  f=[]
  for e in el:
    f=np.concatenate([f,e])
    j+=1
  t=np.sum(f)
  if t==0:
    nncat1[i]=np.zeros([nC*4])
  else:
    nncat1[i]=f/t
  
  i+=1
  
  
  tot=np.sum(tab2)
  
  ntab2=tab2/tot
  
  
  ntab1=np.empty(tab1.shape)
i=0
for el in tab1:
  j=0
  for e in el:
    t=np.sum(e)
    if t==0:
      ntab1[i][j]=[0,0,0,0]
    else:
      ntab1[i][j]=e/t
    j+=1
  i+=1
  
  nntab1=np.empty([nC,nC*4])
i=0
for el in tab1:
  j=0
  f=[]
  for e in el:
    f=np.concatenate([f,e])
    j+=1
  t=np.sum(f)
  if t==0:
    nntab1[i]=np.zeros([nC*4])
  else:
    nntab1[i]=f/t
  
  i+=1
  
np.save(path+'ftab1.npy',tab1)#co occurence matrix
np.save(path+'ftab2.npy',tab2)#occurence count
np.save(path+'fnntab1.npy',nntab1)#normalised per object[320]
np.save(path+'fntab1.npy',ntab1)#normalised per 2 object[80,4]
np.save(path+'fntab2.npy',ntab2)#normalised occurence
np.save(path+'fC1.npy',C)#class
np.save(path+'fB1.npy',B)#boxes
np.save(path+'fS1.npy',S)#scores
np.save(path+'fcat.npy',cat)#category wise occurence
np.save(path+'fncat1.npy',ncat1)#category ntab1
np.save(path+'fnncat1.npy',nncat1)#category nntab1
