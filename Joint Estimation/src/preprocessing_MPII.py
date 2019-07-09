import json
import cv2
from scipy.ndimage import rotate   
import numpy as np
from PIL import Image
import random
from scipy.ndimage.filters import gaussian_filter
import klepto.archives as klepto

random.seed(123)
archive_train=klepto.dir_archive('../dataset/MPII/Train',{},cached=False)
archive_test=klepto.dir_archive('../dataset/MPII/Test',{},cached=False)

# Size of heatmap and pictures
h_pic=256
w_pic=256
h_heat=64
w_heat=64

def read_data(json_file_source):
# =============================================================================
#     Read Json file
# =============================================================================
    with open(json_file_source) as f:
        all_data= json.load(f)
    return(all_data)


def padder(im, side_size, upper_left_x, upper_left_y, joint_positions ,center):
# =============================================================================
#     Crop and pad image
# =============================================================================
    
    #set the size of the padding equal to the size of the bounding box's side
    rr=cv2.copyMakeBorder(im,side_size,side_size,side_size,side_size,cv2.BORDER_CONSTANT,value=[0,0,0])

    # Calculate the new position of the joints
    new_coordinates=[]
    for j in joint_positions:
        if j[0]!=0 and j[1]!=0:     
            new_coordinates.append((int(j[0]-upper_left_x),int( j[1]-upper_left_y),j[2]))
        else:
            new_coordinates.append((0,0,j[2]))
    rr=rr[upper_left_y+side_size:upper_left_y+2*side_size, upper_left_x+side_size:upper_left_x+2*side_size]
    return (rr, new_coordinates, center)


def img_annonate_writer(dict_pic, json_data):
# =============================================================================
#     Take info of a picture
# =============================================================================

    imm= np.array(Image.open('../dataset/MPII/pictures/'+dict_pic['img_paths']))

    joint_positions= dict_pic['joint_self']
    
    center= dict_pic['objpos']

    #coordinates and side length of bounding box
    side= dict_pic['scale_provided'] *200 *1.25    
    upper_left_x= int(center[0] - side/2)
    upper_left_y= int(center[1] +15 - side/2)
    
    return [imm,int(side), upper_left_x, upper_left_y, joint_positions, center]
        

def rot(image, joints):
# =============================================================================
#     Rotate a picture and calculate the new position of the joints
# =============================================================================

    angle=random.randint(-50,50)
    im_rot = rotate(image,angle) 
    
    # new joints coordinates
    org_center = (np.array(image.shape[:2][::-1])-1)//2
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)//2
    new=[]
    angle = np.deg2rad(angle)
    
    # Calculate the new position of the joints    
    for joint in joints:
        if joint[0]!=0 and joint[1]!=0:
            org=joint-org_center
            new.append([int(org[0]*np.cos(angle) + org[1]*np.sin(angle)+rot_center[0]),
              int(-org[0]*np.sin(angle) +org[1]*np.cos(angle)+rot_center[1])])
        else: 
            new.append([0,0])

    return im_rot, np.array(new),rot_center



def augmented_img(img,joints,center,dict_pic,name):
# =============================================================================
#     Augment a picture
# =============================================================================
    for num in range(random.randint(1,2)):
        name+=1
        img_ag,joints_ag,center_ag=rot(img,joints)
        
        # Random gaussian filter
        if bool(random.getrandbits(1)):
            img_ag = gaussian_filter(img_ag,random.randint(1,4))
        heatmap=converting(joints_ag,img_ag.shape[:2])
        
        # Resize picture (256,256)
        pic=Image.fromarray(img_ag)
        pic=pic.resize((h_pic,w_pic), Image.ANTIALIAS)
        archive_train[str(name)+'a.jpg']={'img':np.array(pic),'joints':heatmap}

    return name
        
        
        
def converting(joints_coord,img_shape):
# =============================================================================
#     Calculate the new poistion of the joints after resizing
# =============================================================================
    my_dict_joints=[]

    ratio=h_heat/np.array([img_shape[1],img_shape[0]])
    new_joints=joints_coord*ratio
    for joint in new_joints:

        arr=np.zeros((h_heat,w_heat))
        
        # Set the information of unknown joints as equal to zero
        if joint[0]>63 or joint[1]>63 or joint[0]<0 or joint[1]<0:
            joint=[0,0]

        if joint[0]==0 and joint[1]==0:
            arr=arr
        else:
            arr[int(joint[1]),int(joint[0])]=1.
            
        # Gaussian peak on the joint
        arr=gaussian_filter(arr,1)
        my_dict_joints.append(arr)
    return my_dict_joints
    

def preprocess(data,dict_pic,name,flag=True):
# =============================================================================
#     Image preprocessing
# =============================================================================
    if flag:
        archive_=archive_train
    else:
        archive_=archive_test
    img,coordinates,center=padder(*img_annonate_writer(dict_pic, data))
    heatmap=converting(np.array(coordinates)[:,0:2],img.shape[:2])
    pic=Image.fromarray(img)
    pic=pic.resize((h_pic,w_pic), Image.ANTIALIAS)
    archive_[str(name)+'.jpg']={'img':np.array(pic),'joints':heatmap}

    return (img,coordinates,center) if flag else None

        
def split_train_test(data):
# =============================================================================
#     Split train and test
# =============================================================================
    name=0
    for dict_pic in data:
        name+=1
        if dict_pic['isValidation']==1.0:
            preprocess(data,dict_pic,name,False)
        else:
            img,coordinates,center=preprocess(data,dict_pic,name)
            name=augmented_img(img,np.array(coordinates)[:,0:2],center, dict_pic,name)
