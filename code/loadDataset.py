######load montezuma dataset
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split
import pickle
import random
import os
import scipy
from keras.utils import to_categorical


#remove the background (mean) from the images and save them in X_normalized/

def normalizeDataset(X):

  sfondo = np.mean(X, axis =0)

  #sfondo = np.flip(sfondo, 2)
  print(sfondo.shape)
  
  cv2.imshow('sfondo', sfondo)
  cv2.waitKey(0)
  #os.chdir("/home/elena/montezuma-revenge/X_normalized/") 

  for i in range(X.shape[0]):
    img = X[i]
    #img = np.flip(img, 2)
    #cv2.imshow('img', img)
    #cv2.waitKey(0)    
    img -= sfondo
    #cv2.imshow('img senza sfondo', img)
    #cv2.waitKey(0)
    for i in range(img.shape[0]):
      for j in range(img.shape[1]):
        for c in range(img.shape[2]):
          if img[i, j, c]:
            img[i,j,c] = 0
    assert not((img > [1,1,1]).any())


    scipy.misc.imsave("X_normalized/"+str(i)+".png", img)
    #cv2.imwrite("{}.png".format(i), img) 


def loadMontezumaDataset(filename ='', test_size = 0, channels = 3, latoIm = 50, samplesA = 0, samplesB = 0, Xdir ='X/'):
  if filename != '':
    file = open(filename)
    lines = file.readlines()
    file.close()
  
    data = [line.split('\t') for line in lines]
    data = [[int(d[0]), int(d[1])] for d in data]

    if samplesB != 0:
      data = data[samplesA:samplesB]

  else:
    data = [[i, -1] for i in range(samplesA, samplesB)]

  print("############ samplesA = {}".format(samplesA))
  print("############ samplesB = {}".format(samplesB))


  samples = len(data)
  X = np.zeros((samples, latoIm, latoIm, channels), dtype = int)
  y = np.zeros((samples,), dtype = int )

  print("...LOADED DATASET...") 
  print("X.shape", X.shape)
  print("y.shape", y.shape)

  #dirImages = "home/elena/montezuma-revenge/X/"

  for (i,d) in enumerate(data):
    #if(i%100 == 0):
    #  print(i)
    xi = d[0]
    yi = d[1]
    if channels == 1:
      img = load_img(Xdir+str(xi)+".png",  color_mode = "grayscale")
    else:
      img = load_img(Xdir+str(xi)+".png")
    if(latoIm != 50):
      img = img.resize((latoIm, latoIm))
    #if i == 0:
    #  img.show()
    X[i] = img_to_array(img)
    y[i] = yi

  X = X.astype('float32') / 255

  if test_size != 0:
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_size)
  else:
    X_train = X
    y_train = y
    X_test = np.array([])
    y_test = np.array([])
 

  print("...SPLITTED DATASET...\ntest_size = ", test_size) 
  print("X_train.shape ", X_train.shape)
  print("y_train.shape ", y_train.shape)
  print("X_test.shape ", X_test.shape)
  print("y_test.shape ", y_test.shape)


  with open('datasetMontezuma.pickle', 'wb') as f:
    pickle.dump([X_train, y_train], f)

  return X_train, X_test, y_train, y_test


def loadDatasetTransitions(filename, test_size = 0, channels = 3, latoIm = 50, Xdir ='dataset/images/'):
    file = open(filename)
    lines = file.readlines()
    file.close()
  
    data = [line.split('\t') for line in lines]
    data = [[int(d[0]), int(d[1]), int(d[2])] for d in data]
    samples = len(data)
    #X = np.zeros((samples, latoIm, latoIm, channels), dtype = int)
    #X_prime = np.zeros((samples, latoIm, latoIm, channels), dtype = int )
    #actions = np.zeros((samples, ), dtype = int)
    X = np.zeros( (1, latoIm, latoIm, channels), dtype = int)
    X_prime = np.zeros((1, latoIm, latoIm, channels), dtype = int )
    actions = []
    for (i,d) in enumerate(data):
      #if(i%100 == 0):
      #  print(i)
      xi = d[0]
      ai = d[1]
      x_primei = d[2]
      if channels == 1:
        img = load_img(Xdir+str(xi)+".png",  color_mode = "grayscale")
        img_prime = load_img(Xdir+str(x_primei)+".png",  color_mode = "grayscale")
      else:
        img = load_img(Xdir+str(xi)+".png")
        img_prime = load_img(Xdir+str(x_primei)+".png")
      if(latoIm != 50):
        img = img.resize((latoIm, latoIm))
        img_prime = img_prime.resize((latoIm, latoIm))
      #if i == 0:
      #  img.show()
      xi = img_to_array(img)
      #print(xi.shape)
      if (xi != np.zeros((latoIm, latoIm, channels), dtype = int)).any():
        X = np.append(X, xi.reshape((1, latoIm,latoIm,channels)), axis = 0)
        actions.append( ai)
        X_prime = np.append(X_prime,  img_to_array(img_prime).reshape((1, latoIm, latoIm, channels)), axis = 0)

      #X[i] = img_to_array(img)
      #actions[i] = ai
      #X_prime[i] = img_to_array(img_prime)
    X = X[1:]
    X_prime = X_prime[1:]
    X = X.astype('float32') / 255
    X_prime = X_prime.astype('float32') / 255
    actions = np.array(actions)
    actions =to_categorical(actions)
    '''
    if test_size != 0:
      X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_size)
    else:
      X_train = X
      y_train = y
      X_test = np.array([])
      y_test = np.array([])
 
    '''
    print("...SPLITTED DATASET...\ntest_size = ", test_size) 
    print("X.shape ", X.shape)
    print("X_prime.shape ", X_prime.shape)
    print("actions.shape ", actions.shape)
    
    with open('datasetMontezumaTransitions.pickle', 'wb') as f:
      pickle.dump([X, actions, X_prime], f)
   
    #imgp = array_to_img(X[0])
    #imgp.show()
    #imgp = array_to_img(X_prime[0])
    #imgp.show()


    return X, actions, X_prime

def loadDatasetLL(what):
   if what == "states":
     filename = "datasetLLstates.pickle"
   else:
     filename = "datasetLLactions.pickle"  
   
   infile = open(filename,'rb')
   data = pickle.load(infile)
   infile.close()

   if what == "actions":
     data = np.array(data)
     data =to_categorical(data)
   return data


#x_train, _ , y_train, _ = loadMontezumaDataset(channels= 1, latoIm = 52, samplesB = 1915)
#normalizeDataset(x_train)

#X, actions, X_prime = loadDatasetTransitions("dataset/transitions.txt")
def loadDatasetLLTransitions():
   states = loadDatasetLL("states")
   actions = loadDatasetLL("actions")
   X = states[:-1]
   X_prime = states[1:]
   actions = actions[1:]
   return X, actions, X_prime

#X, actions, X_prime = loadDatasetLLTransitions()

#print(X.shape)
#print(actions.shape)
#print(X_prime.shape)

