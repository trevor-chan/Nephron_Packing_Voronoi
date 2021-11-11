import os
import data
import modules

################### HYPERPARAMS ###################
dataset_path = 'datasets/cells_train_256'
MAX_ITERATIONS = 100000
output_dir = 'models/256'
###################################################

def main():
    os.makedirs(output_dir, exist_ok=True)

    model = modules.BrightfieldPredictor()
    model.cfg.OUTPUT_DIR = output_dir
    model.train(dataset_path, max_iterations=MAX_ITERATIONS)

if __name__ == '__main__':
    main()
