import numpy as np
import pandas as pd 
import os
import csv
from sklearn import metrics
from data.cnews_loader import read_category

label_names = [
    "location_traffic_convenience", 
    "location_distance_from_business_district", 
    "location_easy_to_find", 
    "service_wait_time", 
    "service_waiters_attitude", 
    "service_parking_convenience", 
    "service_serving_speed", 
    "price_level", 
    "price_cost_effective", 
    "price_discount", 
    "environment_decoration", 
    "environment_noise", 
    "environment_space", 
    "environment_cleaness", 
    "dish_portion", 
    "dish_taste", 
    "dish_look", 
    "dish_recommendation", 
    "others_overall_experience", 
    "others_willing_to_consume_again"
]

base_dir = 'data/pred/'
# test_raw_dir = 'data/raw/test/sentiment_analysis_testa.csv'
test_raw_dir = 'data/raw/val/sentiment_analysis_validationset.csv'
file_type = 'preds_val_'

categories, cat_to_id = read_category()   

def ensemble_labels():
    for i, file in enumerate(label_names):
        file_name = file_type + file + '.npy'
        file_path = os.path.join(base_dir, file_name)

        col = np.load(file_path)
        col = col[:, np.newaxis]

        if i == 0:
            labels = col
        else:
            labels = np.concatenate((labels, col), axis= 1)

    return np.array(labels)

def load_data_from_csv(file_name, header=0, encoding="utf-8"):
    data_df = pd.read_csv(file_name, header=header, encoding=encoding)
    return data_df

def write_pred(test_name):
    test_data_df = load_data_from_csv(test_raw_dir)
    columns = test_data_df.columns.tolist()
    labels = ensemble_labels()
    
    categories, cat_to_id = read_category()    
    categories = np.array(categories)
    print(categories)

    for i, column in enumerate(columns[2:]):
        test_data_df[column] = categories[labels[:, i]]

    test_data_df.to_csv(test_name, encoding="utf_8_sig", index=False)

def evaluete():
    y_preds = ensemble_labels()
    data_df = pd.read_csv(test_raw_dir, header=0, encoding='utf8')
    score = 0

    for i, label_name in enumerate(label_names):
        y_test = list(data_df[label_name])
        y_test_cls = [cat_to_id[str(y)] for y in y_test]
        y_pred_cls = y_preds[:, i]

        print(label_name)
        print("Precision, Recall and F1-Score...")
        report = metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories)
        print(report)

        # # 混淆矩阵
        # print("Confusion Matrix...")
        # cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
        # print(cm)

        class_score = metrics.f1_score(y_test_cls, y_pred_cls, average='macro')
        print(label_name, ' : ', class_score)
        score += class_score
        print('#############################################################')
    
    score = score / len(label_names)
    print("f1_score", score)

    
if __name__ == "__main__":
    test_name = os.path.join(base_dir, 'senti_0.2.csv')
    evaluete()
    # write_pred(test_name)
