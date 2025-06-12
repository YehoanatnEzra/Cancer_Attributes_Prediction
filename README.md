# iml-hack-oncology

Welcome to the oncology challenge in IML hackathon 2025!
Follow these instructions to setup your environment, ideally on a virtual environment.


## Data setup 

1. **Install requirements:**
```
pip install -r requirements.txt
```
2. **Download data:** follow instructions in [data/README.md](data/README.md)

3. **Switch to the [src](src) folder:**
```
cd ./src
```

4. **Run the following commands to make sure everything installed properly**:
```
python evaluate_part_0.py --gold=../data/train_test_splits/train.labels.0.csv  --pred=../data/train_test_splits/train.labels.0.csv
python evaluate_part_1.py --gold=../data/train_test_splits/train.labels.1.csv  --pred=../data/train_test_splits/train.labels.1.csv
```

These two last steps should show perfect scores (F1 = 1.0, MSE = 0.0)

## You're good to go

If this works, you're set to go!

