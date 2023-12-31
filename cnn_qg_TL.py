# --------------------------- Importing Libraries ---------------------------
print("--------------------------- Importing Libraries ---------------------------")
import numpy as np
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

                out1_before = self.conv_layer1(x) # Layer1 (Input Layer)
                out1_after = self.relu1(out1_before) # Layer1 (Input Layer)    
                
                ## Hidden Layers
                out2_before = self.conv_layer2(out1_after) #Layer2
                out2_after = self.relu1(out2_before) #Layer2
            
                out3_before = self.conv_layer3(out2_after) #Layer3
                out3_after = self.relu1(out3_before) #Layer3

                out4_before = self.conv_layer4(out3_after) #Layer4
                out4_after = self.relu1(out4_before) #Layer4

                out5_before = self.conv_layer5(out4_after) #Layer5
                out5_after = self.relu1(out5_before) #Layer5

                out6_before = self.conv_layer6(out5_after) #Layer6
                out6_after = self.relu1(out6_before) #Layer6

                out7_before = self.conv_layer7(out6_after) #Layer7
                out7_after = self.relu1(out7_before) #Layer7

                # out8 = self.relu1(self.conv_layer8(out7)) #Layer8

                # out9 = self.relu1(self.conv_layer9(out8)) #Layer9

                # out10 = self.relu1(self.conv_layer10(out9)) #Layer10
                
                ####  !!! Do not forget to change teh output layer when changing the number of hidden layers !!! ####

                output = self.conv_layer11(out7_after) #Layer11 (Output Layer) 
                mid_ouptut = {'out1_before': out1_before, 'out1_after': out1_after, 'out2_before': out2_before, 'out2_after': out2_after, 'out3_before': out3_before, 'out3_after': out3_after, 'out4_before': out4_before, 'out4_after': out4_after, 'out5_before': out5_before, 'out5_after': out5_after, 'out6_before': out6_before, 'out6_after': out6_after, 'out7_before': out7_before, 'out7_after': out7_after}
                return output, mid_ouptut
        # def store_activation(self, epoch, num_epochs, activation, mid_output, model_output):
        #     # List of keys
        #     keys = ['out1_before', 'out1_after', 'out2_before', 'out2_after',
        #             'out3_before', 'out3_after', 'out4_before', 'out4_after',
        #             'out5_before', 'out5_after', 'out6_before', 'out6_after',
        #             'out7_before', 'out7_after', 'output']

        #     for key in keys:
        #         # If the key is not present in the dictionary, initialize it
        #         if key not in activation:
        #             if key != 'output':
        #                 shape = mid_output[key].shape
        #             else:  # for 'output'
        #                 shape = model_output.shape
        #             activation[key] = np.zeros((num_epochs,) + shape)

        #         # Assign values
        #         if key != 'output':
        #             activation[key][epoch] = mid_output[key].detach().cpu().numpy()
        #         else:  # for 'output'
        #             activation[key][epoch] = model_output.detach().cpu().numpy()
        #     return activation
        def store_activation(self, activation, mid_output, model_output):
            # List of keys
            keys = ['out1_before', 'out1_after', 'out2_before', 'out2_after',
                    'out3_before', 'out3_after', 'out4_before', 'out4_after',
                    'out5_before', 'out5_after', 'out6_before', 'out6_after',
                    'out7_before', 'out7_after', 'output']

            for key in keys:
                # If the key is not present in the dictionary, initialize it
                if key not in activation:
                    if key != 'output':
                        shape = mid_output[key].shape
                    else:  # for 'output'
                        shape = model_output.shape
                    activation[key] = np.zeros(shape)

                # Assign values
                if key != 'output':
                    activation[key] = mid_output[key].detach().cpu().numpy()
                else:  # for 'output'
                    activation[key] = model_output.detach().cpu().numpy()
            return activation

# Define a function to calculate the rRMSE loss
class rRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y)) / torch.sqrt(torch.mean(y**2))

################################################################################################################################
                        # --------------------------- Setting up Hyper-Parmeters for CNN --------------------------- #
################################################################################################################################
print("--------------------------- Setting up Hyper-Parmeters for CNN ---------------------------")
#Define relevant parameters for Training
config_default = {
          "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
          "batch_size_train": 8, 
          "learning_rate": 1e-4,
          "num_classes": 1,
          "num_epochs": 40,
          "data_fraction": 0.7, # Fraction of data to be trained 0<x<1 [Whole number of data is 87*2500 =  217500] [For 0.23 * 0.8 * 217000 = 39,928 I get almost 40,000 samples Ashesh trained with.]
          "train_fraction": 0.9, # Ratio of train and test data
          "val_fraction":0.05, # Ratio of validation data
          "test_fraction":0.05, # Ratio of test data
          "skip_first_snap": 80, # Number of snapshots to skip from the beginning of each file
          "base_model_path": "BNN.pt"
         }

################################################################################################################################
                        # --------------------------- Training the CNN --------------------------- #    
################################################################################################################################

def train_model(config=config_default):

    # --------------------------- Initialization ---------------------------
    print("--------------------------- Training Initialization ---------------------------")

    # 0. Model Initialization
    model = ConvNeuralNet(config["num_classes"]).to(config["device"])
    model.double()  # Convert to float64

    print('**** Number of Trainable Parameters in BNN ****')
    count_parameters(model)

    # 0.1. Set loss funtion with criterion
    criterion = nn.MSELoss()
    criterion_rmse = rRMSELoss() # The first argument should be the "predicted value" and the second argument should be the "true value"

    # 0.2 Set the optimizer (Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr= config['learning_rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad = True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-9, last_epoch=- 1, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    
    
    # --------------------------- Loading DATA ---------------------------
    print("--------------------------- Loading DATA ---------------------------")
    # 1. Loading the base model (Optional)[For Transfer Learning purpose]
     
    print("---Loading Base model---")
    base_model_path = config['base_model_path'] # PATH
    base_model = torch.load(base_model_path) # Loading Model
    model.load_state_dict(base_model['model_state_dict']) # Loading model State_dict
    #optimizer.load_state_dict(base_model['optimizer_state_dict']) # Optimizer Loading
    

    # Freezing and unfreezing layers
    model.conv_layer1.weight.requires_grad = False
    model.conv_layer1.bias.requires_grad = False

    model.conv_layer2.weight.requires_grad = False
    model.conv_layer2.bias.requires_grad = False

    model.conv_layer3.weight.requires_grad = False
    model.conv_layer3.bias.requires_grad = False

    model.conv_layer4.weight.requires_grad = False
    model.conv_layer4.bias.requires_grad = False

    model.conv_layer5.weight.requires_grad = False
    model.conv_layer5.bias.requires_grad = False

    model.conv_layer6.weight.requires_grad = False
    model.conv_layer6.bias.requires_grad = False

    model.conv_layer7.weight.requires_grad = False
    model.conv_layer7.bias.requires_grad = False

    model.conv_layer11.weight.requires_grad = True
    model.conv_layer11.bias.requires_grad = True


    # 2. Loading Data
    # 2.1 Loading training data set
    x_train_path = '/media/volume/sdc/data_qg/3jet/'
    y_train_path = '/media/volume/sdc/data_qg/3jet/'

    # 2.2 Create a dataset
    nc_dataset = NCDataset(input_directory=x_train_path, 
                        output_directory=y_train_path, 
                        variable_names={'input': ['u', 'v'], 'output': ['q_forcing_advection']},
                        data_fraction = config['data_fraction'],
                        skip_first_snap = config['skip_first_snap']
                        )
    
    # 2.3 Define the size of each split
    train_fraction = config['train_fraction']  # 80% of the data for training
    val_fraction = config['val_fraction']  # 10% of the data for validation
    test_fraction = config['test_fraction']  # 10% of the data for testing

    # 2.4 Calculate the number of samples for each split
    num_samples = len(nc_dataset)
    train_end = int(num_samples * train_fraction)
    val_end = train_end + int(num_samples * val_fraction)

    # 2.5 Split the dataset
    train_dataset = torch.utils.data.Subset(nc_dataset, range(0, train_end))
    validation_dataset = torch.utils.data.Subset(nc_dataset, range(train_end, val_end))
    test_dataset = torch.utils.data.Subset(nc_dataset, range(val_end, num_samples))

    # 2.6 Calculate the mean and standard deviation of different sets
    input_mean_train, input_std_train, output_mean_train, output_std_train = train_dataset.dataset.get_means_and_stds(indices=train_dataset.indices)
    # input_mean, input_std, output_mean, output_std = train_dataset.dataset.get_means_and_stds(train_dataset.indices)
    # input_mean_test, input_std_test, output_mean_test, output_std_test = test_dataset.dataset.get_means_and_stds(test_dataset.indices)
    # input_mean_val, input_std_val, output_mean_val, output_std_val = validation_dataset.dataset.get_means_and_stds(validation_dataset.indices)

################################################################################################################################
    ## IMPORTANT NOTE ABOUT DATALOADER: The dataloader is written in a way that if you pass the mean and std of the data set, it will normalize the data based on that. 
    # IF you want the normalize to be off, you can pass None for the mean and std. The default is None. 
    # # Be careful: when you pass the mean and std of the training set, the data would be always normalized, unless you pass None for the mean and std.
################################################################################################################################

    # Get the mean std of the training set
    train_dataset.dataset.input_mean = input_mean_train
    train_dataset.dataset.input_std = input_std_train
    train_dataset.dataset.output_mean = output_mean_train
    train_dataset.dataset.output_std = output_std_train
    
    # # Get the mean std of the test set
    test_dataset.dataset.input_mean = input_mean_train
    test_dataset.dataset.input_std = input_std_train
    test_dataset.dataset.output_mean = output_mean_train
    test_dataset.dataset.output_std = output_std_train

    validation_dataset.dataset.input_mean = input_mean_train
    validation_dataset.dataset.input_std = input_std_train
    validation_dataset.dataset.output_mean = output_mean_train
    validation_dataset.dataset.output_std = output_std_train

    # 2.8 Use the dataset with a DataLoader
    data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    data_loader_validation = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)
    data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    ################################################################################################################################ 
                                # --------------------------- Training Part ---------------------------#
    ################################################################################################################################
    loss_epoch = []
    loss_test_epoch = []
    cc_epoch = []
    weights = []
    best_test_loss = float('inf')
    activations = {}
    print("--------------------------- Training Part ---------------------------")

    # 3. ------------------------------------------------------ Training Loop ------------------------------------------------------
    table = PrettyTable(['Epoch', 'Total Epoch', 'Training Loss', 'Test Loss', 'CC_tau11', 'CC_tau12', 'CC_tau22'])
    table.align = 'r'
    # print('Epoch  TotalEpoch    Training Loss   Test Loss   CC_tau11    CC_tau12    CC_tau22 \n ---------------------------------------------------------------------------------')
    
    for epoch in range(config['num_epochs']):
        # ------------------------------------------------------ Training Data Loop ------------------------------------------------------
        train_loss_total = 0
        train_rmse_q1_total = 0
        train_rmse_q2_total = 0
        for i, data_train in enumerate(data_loader_train):

            # Get batch of data
            inputs_train, labels_train = data_train[0].reshape(-1, 4, 64, 64), data_train[1].reshape(-1, 2, 64, 64)

            # Move tensors to the configured device with right data type
            inputs_train = inputs_train.to(device=config['device'], dtype = torch.float64)
            labels_train = labels_train.to(device=config['device'], dtype = torch.float64)

            # Forward pass train 
            model_output, _ = model(inputs_train)

            # Calculate the loss, RMSE 
            loss = criterion(model_output, labels_train)
            rmse_q1_train_ = criterion_rmse(model_output[:,0], labels_train[:,0]) # The first argument should be the "predicted value" and the second argument should be the "true value"
            rmse_q2_train_ = criterion_rmse(model_output[:,1], labels_train[:,1]) # The first argument should be the "predicted value" and the second argument should be the "true value"

            # Sum the loss over the batch
            train_loss_total =+ loss.item()
            train_rmse_q1_total =+ rmse_q1_train_.item()
            train_rmse_q2_total =+ rmse_q2_train_.item()
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print the progress in files
            print(f"Train Batch[{i+1}/{len(data_loader_train)}]", end='\r')
        # Average the loss over the batch
        train_loss_avg = train_loss_total/len(data_loader_train)
        train_rmse_q1_avg = train_rmse_q1_total/len(data_loader_train)
        train_rmse_q2_avg = train_rmse_q2_total/len(data_loader_train)
        # Save the activattion
        # activations = model.store_activation(epoch, config['num_epochs'], activations, mid_output, model_output)
        del inputs_train, labels_train, model_output
        
        # ------------------------------------------------------ Test Loop ------------------------------------------------------
        test_loss_total = 0
        test_rmse_q1_total = 0
        test_rmse_q2_total = 0
        cc_q1_total = 0
        cc_q2_total = 0
        scale = 1e0
        for i, data_test in enumerate(data_loader_test):
            
            #Get batch of data
            inputs_test, labels_test = data_test[0].reshape(-1, 4, 64, 64), data_test[1].reshape(-1, 2, 64, 64)
            
            # Move tensors to the configured device
            inputs_test = inputs_test.to(device=config['device'], dtype = torch.float64)
            labels_test = labels_test.to(device=config['device'], dtype = torch.float64)

            # Forward pass test
            model_output_test,_ = model(inputs_test)
            
            # Evaluating test loss and CC, rRMSE
            loss_test = criterion(model_output_test, labels_test)
            rmse_q1_test = criterion_rmse(model_output_test[:,0], labels_test[:,0]) # The first argument should be the "predicted value" and the second argument should be the "true value"
            rmse_q2_test = criterion_rmse(model_output_test[:,1], labels_test[:,1]) # The first argument should be the "predicted value" and the second argument should be the "true value"
    
            # cc_q1_test = corr2(model_output_test[:,0]* output_std_train[None, 0, None, None].cuda() + output_mean_train[None, 0, None, None].cuda(), labels_test[:,0]* output_std_train[None, 0, None, None].cuda() + output_mean_train[None, 0, None, None].cuda())
            # cc_q2_test = corr2(model_output_test[:,1]* output_std_train[None, 1, None, None].cuda() + output_mean_train[None, 1, None, None].cuda(), labels_test[:,1]* output_std_train[None, 1, None, None].cuda() + output_mean_train[None, 1, None, None].cuda())
            cc_q1_test = corr2(model_output_test[:,0], labels_test[:,0])
            cc_q2_test = corr2(model_output_test[:,1], labels_test[:,1])

            # Sum the loss over the batches
            test_loss_total =+ loss_test.item()
            test_rmse_q1_total =+ rmse_q1_test.item()
            test_rmse_q2_total =+ rmse_q2_test.item()
            cc_q1_total =+ cc_q1_test
            cc_q2_total =+ cc_q2_test

            # Print the progress in files
            print(f"Test Batch[{i+1}/{len(data_loader_test)}]", end='\r')
            
        # Average the loss over the batch
        test_loss_avg = test_loss_total/len(data_loader_test)
        test_rmse_q1_avg = test_rmse_q1_total/len(data_loader_test)
        test_rmse_q2_avg = test_rmse_q2_total/len(data_loader_test)
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
            bestmodelname = "BestModelBasedOnTestLoss.pt"
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch' : epoch,
                        'optimizer_state_dict' : optimizer.state_dict(),
                        'loss' : loss,
                        'loss_test' : loss_test,
                        },bestmodelname)
            print("Checkpoint {} has been saved as the BEST model so far!".format(epoch+1))
        
        # Storing Loss, loss_test, cc histroy
        loss_epoch.append(train_loss_avg)
        loss_test_epoch.append(test_loss_avg)
        print(f"Epoch[{epoch+1}/{config['num_epochs']}] TrainingLoss:{loss_epoch[-1]:10.3e} TestLoss:{loss_test_epoch[-1]:10.3e} CC_q1:{cc_q1_avg:10.3e} CC_q2:{cc_q2_avg:10.3e} rRMSE_q1:{test_rmse_q1_avg:10.3e} rRMSE_q2:{test_rmse_q2_avg:10.3e}")
        
        # scheduler.step()
        scheduler.step(test_loss_avg)

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
    finalmodelname = "final_model_after_training.pt"
    torch.save({
                'model_state_dict': model.state_dict(),
                'epoch' : epoch,
                'optimizer_state_dict' : optimizer.state_dict(),
                'loss' : loss,
                'loss_test' : loss_test,
                },finalmodelname)

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

    # Initialzing the inference Initializng the
    prediction = torch.tensor([], dtype = torch.float64)
    true = torch.tensor([], dtype = torch.float64)
    inputs = torch.tensor([], dtype = torch.float64)
    best_model_path = bestmodelname # PATH of the best model: This can be changed "finalmodelname"
    best_model = torch.load(best_model_path) # Loading Model
    model.load_state_dict(best_model['model_state_dict']) # Loading model State_dict
    
    # Inference Loop on validation data
    val_loss_total = 0
    val_rmse_q1_total = 0
    val_rmse_q2_total = 0
    cc_q1_total = 0
    cc_q2_total = 0

    for i, data_val in enumerate(data_loader_validation):
        inputs_val, labels_val = data_val[0].reshape(-1, 4, 64, 64), data_val[1].reshape(-1, 2, 64, 64)

        # Move tensors to the configured device
        inputs_val = inputs_val.to(device=config['device'], dtype = torch.float64)
        labels_val = labels_val.to(device=config['device'], dtype = torch.float64)

        # Forward pass
        model_output_val,mid_output_val = model(inputs_val)

        # Calculate the loss and other offline metrics
        loss_val = criterion(model_output_val, labels_val)
        rmse_q1_val = criterion_rmse(model_output_val[:,0], labels_val[:,0]) # The first argument should be the "predicted value" and the second argument should be the "true value"
        rmse_q2_val = criterion_rmse(model_output_val[:,1], labels_val[:,1]) # The first argument should be the "predicted value" and the second argument should be the "true value"

        cc_q1 = corr2(model_output_val[:,0]*scale, labels_val[:,0]*scale)
        cc_q2 = corr2(model_output_val[:,1]*scale, labels_val[:,1]*scale)

        # Sum the loss over the batches
        val_loss_total =+ loss_val.item()
        val_rmse_q1_total =+ rmse_q1_val.item()
        val_rmse_q2_total =+ rmse_q2_val.item()
        cc_q1_total =+ cc_q1
        cc_q2_total =+ cc_q2

        # Denormalizing Validation data [input and output]
        inputs_val_denormalized = inputs_val * input_std_train[None, :, None, None].cuda() + input_mean_train[None, :, None, None].cuda()
        labels_val_denormalized = labels_val * output_std_train[None, :, None, None].cuda() + output_mean_train[None, :, None, None].cuda()
        model_output_val_denormalized = model_output_val * output_std_train[None, :, None, None].cuda() + output_mean_train[None, :, None, None].cuda()
        
        # Storing the data
        inputs = torch.cat((inputs, inputs_val_denormalized.detach().cpu()), 0)
        true = torch.cat((true, labels_val_denormalized.detach().cpu()), 0)
        prediction = torch.cat((prediction, model_output_val_denormalized.detach().cpu()), 0)
        
    
        print(f"Test Batch[{i+1}/{len(data_loader_validation)}]", end='\r')
    activations = model.store_activation(activations, mid_output_val, model_output_val)
    del inputs_val, labels_val, model_output_val
    
    # Calculating the average CC
    val_loss_avg = val_loss_total/len(data_loader_validation)
    val_rmse_q1_avg = val_rmse_q1_total/len(data_loader_validation)
    val_rmse_q2_avg = val_rmse_q2_total/len(data_loader_validation)
    cc_q1_val_avg = cc_q1_total/len(data_loader_validation)
    cc_q2_val_avg = cc_q2_total/len(data_loader_validation)

################################################################################################################################
    # ------------------------------------------------------ Saving the outputs  ------------------------------------------------------ 
################################################################################################################################
    print("--------------------------- Saving the outputs ---------------------------")
    import pickle
    plot_loss(loss_epoch, loss_test_epoch, save_fig=True, fig_name='loss.pdf')

    # Dumping the output to txt file
    # file_name = 'metrics_lr' + str(config['learning_rate']) + '_bs' + str(config['batch_size_train']) + '.txt'
    offline_metrics = {"best_test_loss":best_test_loss,
                        "validation_loss":val_loss_avg,
                        "validation_rmse_q1":val_rmse_q1_avg,
                        "validation_rmse_q2":val_rmse_q2_avg,
                        "validation_cc_q1":cc_q1_val_avg,
                        "validation_cc_q2":cc_q2_val_avg,
                        "training_loss":loss_epoch,
                        "test_loss":loss_test_epoch,
    }
    file_name = "metrics.txt"
    with open(file_name, "w") as f:
        json.dump(offline_metrics, f)
        print("Metrics saved in file: ", file_name)

    # Saving the trianing statistics
    training_mean_std = {"input_mean_train":input_mean_train.tolist(),
                        "input_std_train":input_std_train.tolist(),
                        "output_mean_train":output_mean_train.tolist(),
                        "output_std_train":output_std_train.tolist(),}
    file_name = "training_mean_std.txt"
    with open(file_name, "w") as f:
        json.dump(training_mean_std, f)
        print("Training mean and std saved in file: ", file_name)

    # Saving activations
    file_name = "activation_dict.pkl"
    with open(file_name, 'wb') as f:
        pickle.dump(activations, f)
        print("Activations saved in file: ", file_name)

    # Saving the weights
    file_name = "weights_dict.pkl"
    with open(file_name, 'wb') as f:
        pickle.dump(weights, f)
        print("Weights saved in file: ", file_name)
    
    # Save the output in .nc file
    ## Saving the output data to .mat file
    output = {
                "prediction":prediction.numpy(), 
                "true":true.numpy(),
                "inputs": inputs.numpy(),
                }
    
    # ------------------------------------------------------ Save .nc file ------------------------------------------------------
    # Create a new .nc file
    rootgrp = nc.Dataset("output.nc", "w", format="NETCDF4")

    # Define the dimensions of the data
    num_of_samples = rootgrp.createDimension("num_of_samples", prediction.shape[0])
    input_channel = rootgrp.createDimension("input_channel", inputs.shape[1])
    output_channel = rootgrp.createDimension("output_channel", prediction.shape[1])
    y_dim = rootgrp.createDimension("y_dim", prediction.shape[2])
    x_dim = rootgrp.createDimension("x_dim", prediction.shape[3])

    # Create the variables
    variables = {}
    for key, value in output.items():

        if key in ["inputs"]:
            variables[key] = rootgrp.createVariable(key, "f4", ("num_of_samples", "input_channel", "y_dim", "x_dim"))
            variables[key][:] = value  # write the data to the file

        elif key in ["prediction", "true"]:
             variables[key] = rootgrp.createVariable(key, "f4", ("num_of_samples", "output_channel", "y_dim", "x_dim"))
             variables[key][:] = value  # write the data to the file

    # Close the .nc file
    rootgrp.close()


    print("--------------------------- Model has been trained succesfully! ---------------------------")
    # return metric

output = train_model(config_default)