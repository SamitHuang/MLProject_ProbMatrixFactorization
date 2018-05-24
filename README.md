This is a intruction for running the code for project 3.

After downloading the submission, please put data.npz into “src” folder. Or you may need to sepecify the data path in the command line.

1) To run all experiments (on dense data and Sparse data):
    $ python PMF.py 
    It will run 100 iterations for each grid search CV followed by a full evaluation, which takes about 6 minutes in my laptop (MBP).  

    You may want to qucikly watch the grid search CV progress and results:
    $ python PMF.py --max_iter=10 --verbose=3
    
    Brief manual for PMF.py: 
    --verbose: 3 for showing detail score during grid search CV. Defalut value 1, for only showing final tuning results.
    --max_iter: 10 for setting the max iteration to be 10, for a quick running. Default: 100
    --path: data path. Default: ./data.npz
    --tune: control whether to perform grid search tuning. if 0, previously tuned optimal hyper-params will be loaded. if 1, grid search cv will performed to pick the best hyper-parmas. Default: 1

2) To train and test the PMF model without grid search CV and with the build-in optimal hyper-params :
    $ python PMF.py --tune=0 --max_iter=100

    It will save a lot of time, which takes only 1 second in my laptop.
