import numpy as np
import math
from keras.models import model_from_json
from scipy.ndimage import maximum_filter
import cv2
from random import randint as rand
import matplotlib.pyplot as plt
import klepto.archives as klepto


h_pic=256
w_pic=256
h_heat=64
w_heat=64

def load_model(path):
    # Import model
    with open(path+"model.json","r") as file:
      loaded_model_json = file.read()
    #Import weights
    mymodel=model_from_json(loaded_model_json)
    mymodel.load_weights(path+"weight.h5")
    return mymodel


def argmax_(img):
    coordinates = np.unravel_index(np.argmax(img, axis=None), img.shape)
    return coordinates


def euclidean_dist(pred, gt_img):
    pred=non_max_suppression(pred)
    img_x, img_y = argmax_(pred)
    gt_img_x, gt_img_y = argmax_(gt_img)
    return math.sqrt((float(img_x - gt_img_x))**2 + (float(img_y - gt_img_y))**2)


def accuracy_pred(predictions, gt_maps,accurate_preds):
    # threshold = 4 because it's the radius of a circle drawn around the joint
    threshold = 4
    for preds in range(predictions.shape[3]):
        if euclidean_dist(predictions[0,:,:,preds], gt_maps[preds]) <= threshold:
            accurate_preds[preds]+=1
    return accurate_preds
    
    
def PCK(path_GT,path_pred):
    prediction_set = klepto.dir_archive(path_pred,cached=False)
    prediction_set.load()

    gt_maps = klepto.dir_archive(path_GT,cached=False)
    gt_maps.load()

    accuracy=[0]*16
    for name in prediction_set.keys():
        accuracy=accuracy_pred(prediction_set[name], gt_maps[name]['joints'],accuracy)
    return np.array(accuracy)/len(prediction_set)


def non_max_suppression(heatmap, windowSize=3, threshold=1e-3):
    
    # clear value less than threshold
    indices_under_thres = heatmap < threshold
    heatmap[indices_under_thres] = 0

    return heatmap * (heatmap == maximum_filter(heatmap, footprint=np.ones((windowSize, windowSize))))



def prediction(path_GT,path_pred,mymodel):
    prediction=klepto.dir_archive(path_pred,{},cached=False)

    archive= klepto.dir_archive(path_GT,cached=False)
    archive.load()

    for name in archive.keys():
        img=archive[name]['img'].reshape(1,w_pic,h_pic,3)
        predict_heat=mymodel.predict(img/255)
        prediction[name]=predict_heat


def rescale(coords):
    ''' transform coordinates into 256x256 space '''
    x=(w_pic*coords[1])//w_heat
    y=(h_pic*coords[0])//h_heat
    return (x,y)


def rescale_joint_coords(heatmap):
    for joint_matrix in heatmap:
        x,y = rescale(argmax_(joint_matrix))
        yield x,y

def draw_skeleton(img, heatmap,MPII=True):
    if MPII:
        lines = [(0,1),(1,2),(2,6),(6,3),(3,4),(4,5),(6,7),(7,8),(8,9),(10,11),(11,12),(12,7),(7,13),(13,14),(14,15)]
    else:
        lines = [(0,1),(1,2),(3,4),(4,5),(6,7),(7,8),(8,9),(9,10),(10,11),(2,8),(3,9),(12,13)]
    coords = dict(enumerate(list(rescale_joint_coords(heatmap))))
    for points in lines:
        if coords[points[0]]==(0,0) or coords[points[1]]==(0,0): continue
        else:
            cv2.line(img, coords[points[0]], coords[points[1]], (rand(0,255),rand(0,255),rand(0,255)), thickness=2, lineType=8)
    plt.imshow(img)

