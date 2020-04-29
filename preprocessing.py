import cv2
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
import os
from progressBar import printProgressBar

class imageSet():
    def __init__(self,labeled,path,targetShape):
        self.name = path.split('chest_xray')[-1]
        print('Unpacking',self.name)
        self.path = path
        self.labeled = labeled
        self.X,self.y = self.extractArray(path,targetShape)
    def extractArray(self,path,targetShape):
        X_direct = []
        y = []
        l = len(glob(path+'/*'))
        printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
        for i,im_path in enumerate(glob(path+'/*')):
            printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
            init = np.float32(cv2.imread(im_path,0))
            resized = cv2.resize(init,targetShape,interpolation=cv2.INTER_LINEAR)
            X_direct.append(resized)
            if self.labeled:
                if 'NORMAL' in path:
                   y.append(0)
                elif 'bacteria' in path:
                    y.append(1)
                else:
                    y.append(2)
        X_direct = np.stack(X_direct,axis=0)
        return X_direct,np.asarray(y)
    
    def saveVersion(self,set_name):
        working_dir = os.path.dirname(os.path.realpath(__file__))
        set_folder = os.path.join(working_dir,'PreprocessedNumpy',set_name)
        try:
            os.mkdir(set_folder)
        except OSError as error:
            print(error)
        destination_path = os.path.join(set_folder,self.name)
        destination_path_y = destination_path+'_target'
        np.save(destination_path,self.X)
        if self.labeled:
            np.save(destination_path_y,self.y)