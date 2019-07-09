import json
with open('mpii_annotations.json') as json_file: 
    mpii = json.load(json_file)
#mpii_copy = mpii.copy()
with open('mpii_annotations.json') as json_file: 
    mpii_copy = json.load(json_file)
    
for i in range(len(mpii)):
    mpii[i]['joint_self'] = mpii[i]['joint_self'][:6]+mpii[i]['joint_self'][8:]
    mpii[i]['joint_self'] = mpii[i]['joint_self'][:6]+mpii[i]['joint_self'][12:]+mpii[i]['joint_self'][6:12]
    if (mpii[i]['numOtherPeople']==1.0):
        mpii[i]['joint_others'] = mpii[i]['joint_others'][:6]+mpii[i]['joint_others'][8:]
        mpii[i]['joint_others'] = mpii[i]['joint_others'][:6]+mpii[i]['joint_others'][12:]+mpii[i]['joint_others'][6:12]

    elif (mpii[i]['numOtherPeople']>=2.0):
        for j in range(len(mpii[i]['joint_others'])):
            mpii[i]['joint_others'][j] = mpii[i]['joint_others'][j][:6]+mpii[i]['joint_others'][j][8:]
            mpii[i]['joint_others'][j] = mpii[i]['joint_others'][j][:6]+mpii[i]['joint_others'][j][12:]\
            +mpii[i]['joint_others'][j][6:12]
            
with open("new_mpii.json", "w") as write_file:
    json.dump(mpii, write_file)