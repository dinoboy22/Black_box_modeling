
CAT_BOOST_RANDOM_STATE = 110
CAT_BOOST_THRESHOLD = 0.5

LEARNING_RATE=1.0
NUM_EPOCHS=10

BASELINE_MODEL = {
    # 'train_csv_file': 'train.csv', 
    'train_csv_file': 'train_pt.csv', 
    # 'test_csv_file': 'test.csv', # 'test_pt.csv',
    'test_csv_file': 'test_pt.csv',
    'num_input': 11,
    'num_output': 1,
    'layers': [32, 32, 8],
    'dropout': 0.1,
    'loss_fn': 'MSELoss',
    'num_epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.0001,
    'num_workers': 4,
}
