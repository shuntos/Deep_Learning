import scipy.io
import pandas as pd

dataset_path = "../datasets"
dataset_location = dataset_path+"/RAP"+"/RAP_dataset"
datasets_path_annotation = dataset_path+'/RAP/'+'RAP_annotation/RAP_annotation.mat'
mat = scipy.io.loadmat(datasets_path_annotation)


def exctract_lables():
    columns = []
    data = mat["RAP_annotation"][0][0][3]
    for row in data: 
        label = row[0][0]
        columns.append(label)
    return columns

column = exctract_lables()
column.insert(0, "img_path")
dataset_df = pd.DataFrame(columns=column)


for i in range(len(mat["RAP_annotation"][0][0][1])):
    img_location = dataset_location+'/'+ mat["RAP_annotation"][0][0][5][i][0][0]
    # img = cv2.resize(cv2.imread(img_location, cv2.IMREAD_COLOR),(width,height), interpolation = cv2.INTER_CUBIC)
    print(i)
    img_attr = mat["RAP_annotation"][0][0][1][i].tolist()
    img_attr.insert(0, img_location)
    dataset_df.loc[i] = img_attr




dataset_df.to_csv("../datasets/RAP/data.csv")