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
np.save(path+'C2.npy',C)
np.save(path+'B2.npy',B)
np.save(path+'S2.npy',S)
