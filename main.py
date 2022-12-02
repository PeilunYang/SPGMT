from Trainer import GTrajST_Trainer, GTrajsim_Trainer
import torch
import os

if __name__ == '__main__':
    # load model
    load_model_name = None
    load_optimizer_name = None

    # initilize SPGMT
    GTrajSim = GTrajsim_Trainer()
    # train SPGMT
    GTrajSim.Spa_train(load_model=load_model_name, load_optimizer=load_optimizer_name)
    # test SPGMT
    # GTrajSim.Spa_eval(load_model=load_model_name)
