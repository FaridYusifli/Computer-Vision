import preprocessing_MPII as prep_MPII
import preprocessing_UP14 as prep_UP14
from trainer import train_model
import evaluation as eval_


h_pic=256
w_pic=256


dataset ='MPII'

if dataset=='MPII':
    path_data_Train='../dataset/MPII/Train'
    path_data_Test='../dataset/MPII/Test'

    path_model='../model/MPII/'
    path_pred_Train='../results/MPII/Train'
    path_pred_Test='../results/MPII/Test'

    num_joints=16
    # Load GT
    data=prep_MPII.read_data('../labels/MPII/mpii_annotations.json')
    
    # Preprocess images and augment them
    prep_MPII.split_train_test(data)

elif dataset== 'UP14':
    path_data_Train='../dataset/UP14/Train'
    path_data_Test='../dataset/UP14/Test'

    path_model='../model/UP14/'
    path_pred_Train='../results/UP14/Train'
    path_pred_Test='../results/UP14/Test'
    num_joints=14

    # Load GT
    train_name,test_name=prep_UP14.read_name('../dataset/UP14/pictures')
    
    # Preprocess images and augment them
    prep_UP14.create_data(train_name)
    prep_UP14.create_data(test_name,False)


#Train the model
history=train_model(path_data_Train,path_model,njoints=num_joints)

#load model
mymodel=eval_.load_model(path_model)

# Prediction Train
eval_.prediction(path_data_Train,path_pred_Train,mymodel)   

# Prediction Test
eval_.prediction(path_data_Test,path_pred_Test,mymodel)   



train_pck=eval_.PCK(path_data_Train,path_pred_Train)

test_pck=eval_.PCK(path_data_Test,path_pred_Test)



#eval_.draw_skeleton((images_train[20]*255).astype('uint8'),prediction_train[20])



