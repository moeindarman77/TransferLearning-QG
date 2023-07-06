# --------------------------- Importing Libraries ---------------------------
print("--------------------------- Importing Libraries ---------------------------")
from ray.air import session
from torch.utils.data import DataLoader, Subset
from scipy.io import savemat
from utils.utils import MyDataset, plot_loss, NCDataset
from utils.corr2 import corr2
from utils.count_parameters import count_parameters
from utils.cc_energy_transfer_2d_fhit import cc_energy_transfer_2d_fhit
from utils.comparison_contour_plot import comparison_contour_plot
from utils.create_contour_plot import create_contour_plot
from utils.post_proccess import post_proccess
from prettytable import PrettyTable
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import json
import gc
import netCDF4 as nc

################################################################################################################################
                        # --------------------------- CNN Architecture ---------------------------#
################################################################################################################################
print("--------------------------- Setting CNN Architecture ---------------------------")
#Creating a CNN class
class ConvNeuralNet(nn.Module):

        # Determine what layers and their order in CNN object
        def __init__(self, num_classes):

                super(ConvNeuralNet, self).__init__()

                # Input Layer
                self.conv_layer1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=5, padding="same")
                
                # Hidden layers
                self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                self.conv_layer5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                self.conv_layer6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                self.conv_layer7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                # self.conv_layer8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                # self.conv_layer9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                # self.conv_layer10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                
                # Output layer
                self.conv_layer11 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=5, padding="same")
                
                # Activation function
                self.relu1 = nn.ReLU()
                
        def forward(self, x):

                out1 = self.relu1(self.conv_layer1(x)) # Layer1 (Input Layer)    
                
                ## Hidden Layers
                out2 = self.relu1(self.conv_layer2(out1)) #Layer2
            
                out3 = self.relu1(self.conv_layer3(out2)) #Layer3

                out4 = self.relu1(self.conv_layer4(out3)) #Layer4

                out5 = self.relu1(self.conv_layer5(out4)) #Layer5

                out6 = self.relu1(self.conv_layer6(out5)) #Layer6

                out7 = self.relu1(self.conv_layer7(out6)) #Layer7

                # out8 = self.relu1(self.conv_layer8(out7)) #Layer8

                # out9 = self.relu1(self.conv_layer9(out8)) #Layer9

                # out10 = self.relu1(self.conv_layer10(out9)) #Layer10
                
                ####  !!! Do not forgety to chage teh output layer when changing the number of hidden layers !!! ####

                output = self.conv_layer11(out7) #Layer11 (Output Layer) 
                return output, out1, out2, out3, out4, out5, out6, out7


################################################################################################################################
                        # --------------------------- Setting up Hyper-Parmeters for CNN --------------------------- #
################################################################################################################################
print("--------------------------- Setting up Hyper-Parmeters for CNN ---------------------------")
#Define relevant parameters for Training
config_default = {
          "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
          "batch_size_train": 8, 
          "learning_rate": 5e-5,
          "num_classes": 1,
          "num_epochs": 5,
          "data_fraction": 0.1, # Fraction of data to be trained 0<x<1 [Whole number of data is 87*2500 =  217500]
          "split_ratio": 0.9, # Ratio of train and test data

         }

################################################################################################################################
                        # --------------------------- Training the CNN --------------------------- #    
################################################################################################################################

def train_model(config=config_default):

    # --------------------------- Initialization ---------------------------
    print("--------------------------- Training Initialization ---------------------------")

    # 0. Model Initialization
    model = ConvNeuralNet(config["num_classes"]).to(config["device"])

    print('**** Number of Trainable Parameters in BNN ****')
    count_parameters(model)

    # 0.1. Set loss funtion with criterion
    criterion = nn.MSELoss()

    # 0.2 Set the optimizer (Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr= config['learning_rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad = True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-9, last_epoch=- 1, verbose=True)
    
    # --------------------------- Loading DATA ---------------------------
    print("--------------------------- Loading DATA ---------------------------")

    # 1. Loading the base model (Optional)[For Transfer Learning purpose]
    '''    
    print("---Loading Base model---")
    base_model_path = '/home/exouser/tl/10layer/base_model.pt' # PATH
    base_model = torch.load(base_model_path) # Loading Model
    model.load_state_dict(base_model['model_state_dict']) # Loading model State_dict
    #optimizer.load_state_dict(base_model['optimizer_state_dict']) # Optimizer Loading
    '''

    # 2. Loading Data
    # 2.1 Loading training data set
    x_train_path = '/media/volume/sdc/data_qg/3jet/'
    y_train_path = '/media/volume/sdc/data_qg/3jet/'

    # 2.2 Create a dataset
    nc_dataset = NCDataset(input_directory=x_train_path, 
                        output_directory=y_train_path, 
                        variable_names={'input': ['u', 'v'], 'output': ['q']},
                        data_fraction = config['data_fraction'],
                        normalize_input=True,
                        normalize_output=True,)
    
    # 2.3 Splitting the data to train and test and validation
    split_ratio = config['split_ratio']
    train_size = int(split_ratio * len(nc_dataset))
    test_size = len(nc_dataset) - train_size

    # 2.4 Split test to test and validation
    split_ratio_test_validation = 0.5
    test_size = int(split_ratio_test_validation * test_size)
    validation_size = len(nc_dataset) - train_size - test_size

    train_dataset, test_dataset, validation_dataset = torch.utils.data.random_split(nc_dataset, [train_size, test_size, validation_size])

    # 2.5 Use the dataset with a DataLoader
    data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    data_loader_validation = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=True)

    ################################################################################################################################ 
                                # --------------------------- Training Part ---------------------------#
    ################################################################################################################################
    loss_epoch = []
    loss_test_epoch = []
    cc_epoch = []
    weights = []
    best_test_loss = float('inf')
    print("--------------------------- Training Part ---------------------------")

    # 3. ------------------------------------------------------ Training Loop ------------------------------------------------------
    table = PrettyTable(['Epoch', 'Total Epoch', 'Training Loss', 'Test Loss', 'CC_tau11', 'CC_tau12', 'CC_tau22'])
    table.align = 'r'
    # print('Epoch  TotalEpoch    Training Loss   Test Loss   CC_tau11    CC_tau12    CC_tau22 \n ---------------------------------------------------------------------------------')
    
    for epoch in range(config['num_epochs']):

        # ------------------------------------------------------ Training Data Loop ------------------------------------------------------
        train_loss_total = 0
        for i, data_train in enumerate(data_loader_train):

            # Get batch of data
            inputs_train, labels_train = data_train[0].reshape(-1, 4, 64, 64), data_train[1].reshape(-1, 2, 64, 64)
        
            # Move tensors to the configured device with right data type
            inputs_train = inputs_train.to(device=config['device'], dtype = torch.float32)
            labels_train = labels_train.to(device=config['device'], dtype = torch.float32)

            # Forward pass train 
            model_output,_,_,_,_,_,_,_ = model(inputs_train)
            loss = criterion(model_output, labels_train)

            # Sum the loss over the batch
            train_loss_total =+ loss.item()
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print the progress in files
            print(f"Train Batch[{i+1}/{len(data_loader_train)}]", end='\r')
        # Average the loss over the batch
        train_loss_avg = train_loss_total/len(data_loader_train)
        del inputs_train, labels_train, model_output
        
        # ------------------------------------------------------ Test Loop ------------------------------------------------------
        test_loss_total = 0
        cc_q1_test_total = 0
        cc_q2_test_total = 0
        for i, data_test in enumerate(data_loader_test):
            
            #Get batch of data
            inputs_test, labels_test = data_test[0].reshape(-1, 4, 64, 64), data_test[1].reshape(-1, 2, 64, 64)
            
            # Move tensors to the configured device
            inputs_test = inputs_test.to(device=config['device'], dtype = torch.float32)
            labels_test = labels_test.to(device=config['device'], dtype = torch.float32)

            # Evaluating test loss and CC
            model_output_test,_,_,_,_,_,_,_ = model(inputs_test)
            loss_test = criterion(model_output_test, labels_test)
            cc_q1_test = corr2(model_output_test[:,0], labels_test[:, 0])
            cc_q2_test = corr2(model_output_test[:,1], labels_test[:, 1])

            # Sum the loss over the batches
            test_loss_total =+ loss_test.item()
            cc_q1_total =+ cc_q1_test.item()
            cc_q2_total =+ cc_q2_test.item()

            # Print the progress in files
            print(f"Test Batch[{i+1}/{len(data_loader_test)}]", end='\r')
            
        # Average the loss over the batch
        test_loss_avg = test_loss_total/len(data_loader_test)
        cc_q1_avg = cc_q1_total/len(data_loader_test)
        cc_q2_avg = cc_q2_total/len(data_loader_test)
        del inputs_test, labels_test, model_output_test
            
        # # Saving Model Checkpoint
        # if (epoch + 1 ) % 100 == 0:
            
        #     # Save the model checkpoint XXXX
        #     modelname = "checkpoint_epoch" + str(epoch+1) + ".pt"
        #     torch.save({
        #                 'model_state_dict': model.state_dict(),
        #                 'epoch' : epoch,
        #                 'optimizer_state_dict' : optimizer.state_dict(),
        #                 'loss' : loss,
        #                 'loss_test' : loss_test,
        #                 },modelname)
        #     print("Checkpoint {} has been saved!".format(epoch+1))
            
        if test_loss_avg < best_test_loss:
            best_test_loss = test_loss_avg
            # Save the model checkpoint XXXX
            modelname = "BestModelBasedOnTestLoss.pt"
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch' : epoch,
                        'optimizer_state_dict' : optimizer.state_dict(),
                        'loss' : loss,
                        'loss_test' : loss_test,
                        },modelname)
            print("Checkpoint {} has been saved as the BEST model so far!".format(epoch+1))
        
        # Storing Loss, loss_test, cc histroy
        loss_epoch.append(train_loss_avg)
        loss_test_epoch.append(test_loss_avg)
        print(f"Epoch[{epoch+1}/{config['num_epochs']}] TrainingLoss:{loss_epoch:10.3e} TestLoss:{loss_test_epoch[-1]:10.3e} CC_q1:{cc_q1_avg:10.3e} CC_q2:{cc_q2_avg:10.3e}")
        
        # scheduler.step()

        # Clear the GPU memory cache and all variables stored on GPU
        torch.cuda.empty_cache()
        delete = gc.collect()
        print('Garbage collector: collected {} objects.'.format(delete))

    # End of Epoch 

    # Saving weights for the training
    for i in range(7):
        layer_name = "conv_layer" + str(i+1)
        weights.append(getattr(model,layer_name).weight.data.cpu().numpy())

    # Save final model
    modelname = "final_model_after_training.pt"
    torch.save({
                'model_state_dict': model.state_dict(),
                'epoch' : epoch,
                'optimizer_state_dict' : optimizer.state_dict(),
                'loss' : loss,
                'loss_test' : loss_test,
                },modelname)

    print("---------------------------Trained model has been saved successfully!---------------------------")

    ################################################################################################################################
                                    # --------------------------- Inference Part ---------------------------
    ################################################################################################################################
    print("--------------------------- Inference Part ---------------------------")

    '''
    # This is just for the evaluation (Optional)
    base_model_path = 'model.pt' # PATH
    base_model = torch.load(base_model_path) # Loading Model
    model.load_state_dict(base_model['model_state_dict']) # Loading model State_dict
    #optimizer.load_state_dict(base_model['optimizer_state_dict']) # Optimizer Loading
    '''

    data_loader_test = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialzing the inference Initializng the
    prediction = torch.tensor([], dtype = torch.float32)
    true = torch.tensor([], dtype = torch.float32)
    inputs = torch.tensor([], dtype = torch.float32)
    best_model_path = 'best_model.pt' # PATH
    best_model = torch.load(best_model_path) # Loading Model
    model.load_state_dict(best_model['model_state_dict']) # Loading model State_dict
    
    # Inference Loop on validation data
    for i, data_val in enumerate(data_loader_validation):
        inputs_val, labels_val = data_val[0].reshape(-1, 4, 64, 64), data_val[1].reshape(-1, 2, 64, 64)

        # Move tensors to the configured device
        inputs_val = inputs_val.to(device=config['device'], dtype = torch.float32)

        # Forward pass
        model_output_val,_,_,_,_,_,_,_ = model(inputs_val)

        # Denormalizing Validation data [input and output]
        # inputs_val_normalized = denormalize_input(inputs_val)
        # labels_val_normalized = denormalize_output(labels_val)
        # model_output_val_normalized = denormalize_output(model_output_val)

        # Detach from GPU
        model_output_val_normalized = model_output_val_normalized.detach().cpu()
        labels_val_normalized = labels_val_normalized.detach().cpu()
        inputs_val_normalized = inputs_val_normalized.detach().cpu()

        # Storing the data
        prediction = torch.cat((prediction, model_output_val_normalized), 0)
        true = torch.cat((true, labels_val_normalized), 0)
        inputs = torch.cat((inputs, inputs_val_normalized), 0)

        # Calculating the CC
        cc_q1 = corr2(model_output_val_normalized[:,0,:,:], labels_val_normalized[:,0,:,:])
        cc_q2 = corr2(model_output_val_normalized[:,1,:,:], labels_val_normalized[:,1,:,:])

        print(f"Test Batch[{i+1}/{len(data_loader_validation)}]", end='\r')
        del inputs_test, label_test, model_output_test, inputs_test_normalized, label_test_normalized, model_output_test_normalized

        # Calculating the average CC

################################################################################################################################
    # ------------------------------------------------------ Saving the outputs  ------------------------------------------------------ 
################################################################################################################################
    print("--------------------------- Saving the outputs ---------------------------")

    plot_loss(loss_epoch, loss_test_epoch, save_fig=True, fig_name='loss.pdf')

    # # Dumping the output to txt file
    # file_name = 'metrics_lr' + str(config['learning_rate']) + '_bs' + str(config['batch_size_train']) + '.txt'
    # with open(file_name, "w") as f:
    #     json.dump(ccs, f)

    # Save the output in .nc file
    ## Saving the output data to .mat file
    output = {
                "prediction":prediction.numpy(), 
                "true":true.numpy(),
                "inputs": inputs.numpy(),
                "weights":weights,
                "train_loss":loss_epoch,
                "test_loss":loss_test_epoch,
                }
    
    # Save the output in the .nc file
    
    # filename = "Data.mat"
    # savemat(filename, output)

    
    # Create a new .nc file
    rootgrp = nc.Dataset("output.nc", "w", format="NETCDF4")

    # Define the dimensions of the data
    num_of_samples = rootgrp.createDimension("num_of_samples", prediction.shape[0])
    
    input_channel = rootgrp.createDimension("input_channel", inputs.shape[1])
    output_channel = rootgrp.createDimension("output_channel", prediction.shape[1])

    y_dim = rootgrp.createDimension("y_dim", prediction.shape[2])
    x_dim = rootgrp.createDimension("x_dim", prediction.shape[3])

    num_epochs = rootgrp.createDimension("num_epochs", config['num_epochs'])

    # Create the variables
    variables = {}
    for key, value in output.items():

        if key in ["inputs"]:
            variables[key] = rootgrp.createVariable(key, "f4", ("num_of_samples", "input_channel", "y_dim", "x_dim"))

        elif key in ["prediciton", "true"]:
             variables[key] = rootgrp.createVariable(key, "f4", ("num_of_samples", "output_channel", "y_dim", "x_dim"))

        elif key in ["train_loss", "test_loss"]:
            variables[key] = rootgrp.createVariable(key,"f4", ("num_epochs",))
        # else:
        #     variables[key] = value

    # Close the .nc file
    rootgrp.close()

    print("--------------------------- Model has been trained succesfully! ---------------------------")
    # return metric

output = train_model(config_default)