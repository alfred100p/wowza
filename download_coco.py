import os
import tensorflow as tf
import numpy as np


splits=['train','val','test']
img_ids={
    'train':[],
    'val':[],
    'test':[]
}
for split in splits:
  print(split)
  image_folder = 'images/MS-COCO-2014/'+split+'2014/'
  if not os.path.exists(image_folder):
    image_zip = tf.keras.utils.get_file(split+'2014.zip',
                                        cache_subdir=os.path.abspath('images/MS-COCO-2014/'),
                                        origin = 'http://images.cocodataset.org/zips/'+split+'2014.zip',
                                        extract = True)
    PATH = os.path.dirname(image_zip) + image_folder
    os.remove(image_zip)
  else:
    PATH = os.path.abspath('images/MS-COCO-2014/') + image_folder
  
  img_list=os.listdir('images/MS-COCO-2014/'+split+'2014/')
  
  for el in img_list:
    img_ids[split]+=[el[:-4]]
img_ids=np.array(img_ids,dtype=str)
##################################Look if below line reqd, i.e. if we need list of IDS
np.save('images/MS-COCO-2014/COCO_2014_img_ids.npy', img_ids)
