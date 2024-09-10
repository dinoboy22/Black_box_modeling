
CAT_BOOST_RANDOM_STATE = 110
CAT_BOOST_THRESHOLD = 0.5

LEARNING_RATE=1.0
NUM_EPOCHS=10

BASELINE_MODEL = {
    'train_csv_file': 'train.csv', # 'train_pt.csv', 'train_pt_excl_y.csv', 
    'test_csv_file': 'test.csv',
    'num_input': 11,
    'num_output': 1,
    # 'layers': [32, 32, 8],
    'layers': [8, 8, 8],
    'dropout': 0.0,
    'loss_fn': 'MSELoss',
    'num_epochs': 100,
    'batch_size': 1,
    'learning_rate': 0.001,
    'num_workers': 4,
    'label': 'baseline',
}

BASELINE_WIDE_MODEL = {
    'train_csv_file': 'train.csv', 
    'test_csv_file': 'test.csv',
    'num_input': 11,
    'num_output': 1,
    # 'layers': [512, 32, 8],
    'layers': [1024, 32, 8],
    'dropout': 0.0,
    'loss_fn': 'MSELoss',
    'num_epochs': 100,
    'batch_size': 16,
    'learning_rate': 0.001,
    'num_workers': 4,
    'label': 'baseline-wide',
}
