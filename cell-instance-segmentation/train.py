import os
import data
import modules

################### HYPERPARAMS ###################
<<<<<<< HEAD
dataset_path = 'datasets/kidney_training'
MAX_ITERATIONS = 100000
output_dir = 'models/512'
=======
dataset_path = 'datasets/cells_train_256'
MAX_ITERATIONS = 100000
output_dir = 'models/256'
>>>>>>> 772f13cbb3671e604caa5a673c7dc35da0d9a4b6
###################################################

def main():
    os.makedirs(output_dir, exist_ok=True)

    model = modules.BrightfieldPredictor()
    model.cfg.OUTPUT_DIR = output_dir
    model.train(dataset_path, max_iterations=MAX_ITERATIONS)

if __name__ == '__main__':
    main()
