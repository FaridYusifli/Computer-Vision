import model
from keras.optimizers import RMSprop
import numpy as np
import klepto.archives as klepto
import random

random.seed(123)

# Size of heatmap and pictures
h_pic=256
w_pic=256
h_heat=64
w_heat=64

        

def train_data_generator(path,batch_size, inres=(h_pic,w_pic) , outres= (h_heat,w_heat)):
# =============================================================================
#     Create data generator 
# =============================================================================
    archive_train = klepto.dir_archive(path,cached=False)
    archive_train.load()
    all_images = np.array(list(archive_train.keys()))
    size = len(all_images)

    while True:
        
        # take random images
        names = np.random.permutation(list(archive_train.keys()))
        num_of_batches= size// batch_size 
        
        for im in range(num_of_batches): 
            gt_stack = np.zeros(shape=(batch_size, outres[0], outres[1],nOutput))
            img_stack= np.zeros(shape= (batch_size, inres[0], inres[1], 3))

            selected_photo_names = names[im * batch_size : (im+1)* batch_size]

            for j in range(len(selected_photo_names)):
                
                gt_stack[j,:,:,:] = np.transpose(np.array(archive_train[selected_photo_names[j]]['joints']),(1,2,0))
                img_stack[j,:,:,:] = archive_train[selected_photo_names[j]]['img']/255.
               
            yield(img_stack, gt_stack)



def train_model(path_data,path_model,njoints,batch_size=6,optimizer=RMSprop(lr=5e-4),loss='mean_squared_error',metrics=['accuracy'],epochs=200,step_epochs=800):
# =============================================================================
#     Train the model
# =============================================================================
    global nOutput
    nOutput=njoints
    
    mymodel= model.hg_train(nOutput)
    data_gen= train_data_generator(path_data,batch_size)
    mymodel.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    history=mymodel.fit_generator(data_gen,step_epochs, epochs)
    model_json = mymodel.to_json()

    with open(path_model+"model.json","w") as json_file:
        json_file.write(model_json)

    mymodel.save_weights(path_model+"weight.h5")
    return history