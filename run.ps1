
# data process
# cd .\data\raw
# mv ai_challenger_sentiment_analysis_trainingset_20180816 train
# mv ai_challenger_sentiment_analysis_testa_20180816 test
# mv ai_challenger_sentiment_analysis_validationset_20180816 val
# cd ../..
# python helper/data_group.py
# python helper/preprocess_data.py

# python run_cnn.py train location_traffic_convenience
# python run_cnn.py test location_traffic_convenience
python pred_val.py location_traffic_convenience