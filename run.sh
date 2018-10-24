#!/bin/bash
if [ ! -d data/raw/train ]; then
    cd data/raw
    # wget 'https://www.dropbox.com/s/0u7c59sj5l5codk/ai_challenger_fsauor2018_testa_20180816.zip?dl=1' -O test.zip
    # wget 'https://www.dropbox.com/s/dojhfh1vkvzmcph/ai_challenger_fsauor2018_trainingset_20180816.zip?dl=1' -O train.zip
    # wget 'https://www.dropbox.com/s/zq0zmwvdwtigdq5/ai_challenger_fsauor2018_validationset_20180816.zip?dl=1' -O val.zip
    unzip train.zip
    unzip test.zip
    unzip val.zip
    mv ai_challenger_sentiment_analysis_trainingset_20180816 train
    mv ai_challenger_sentiment_analysis_testa_20180816 test
    mv ai_challenger_sentiment_analysis_validationset_20180816 val
    cd ../..
    python helper/data_group.py
    python helper/preprocess_data.py
fi

for label in location_traffic_convenience location_distance_from_business_district location_easy_to_find service_wait_time service_waiters_attitude service_parking_convenience service_serving_speed price_level price_cost_effective price_discount environment_decoration environment_noise environment_space environment_cleaness dish_portion dish_taste dish_look dish_recommendation others_overall_experience others_willing_to_consume_again
do
    python run_cnn.py train $label
    python run_cnn.py test $label
done

for label in location_traffic_convenience location_distance_from_business_district location_easy_to_find service_wait_time service_waiters_attitude service_parking_convenience service_serving_speed price_level price_cost_effective price_discount environment_decoration environment_noise environment_space environment_cleaness dish_portion dish_taste dish_look dish_recommendation others_overall_experience others_willing_to_consume_again
do
    python pred_val.py $label
done

python ensemble.py