import numpy as np
# from matplotlib import pyplot as plt
# %matplotlib inline

import pyqg
from pyqg import diagnostic_tools as tools

import pyfftw

import torch
import torch.nn as nn
from pyqg import Parameterization
import json 

class CNNparametrization(Parameterization):
    def __init__(self, model_type, model_path, stat_path=None):
        super().__init__()
        self.model = self.init_model(model_type, model_path)
        
        # Load .txt
        with open(stat_path) as json_file:
            data = json.load(json_file)
            
            self.input_mean = torch.tensor(data['input_mean_train']).reshape(4, 1, 1)
            self.input_std = torch.tensor(data['input_std_train']).reshape(4, 1, 1)
            self.output_mean = torch.tensor(data['output_mean_train']).reshape(2, 1, 1)
            self.output_std = torch.tensor(data['output_std_train']).reshape(2, 1, 1)

    def __call__(self, m):
        # Here you will need to transform your model `m` into the appropriate input tensor for the CNN
        # As I don't know the exact structure of your model and the expected input for the CNN, 
        # I'm using a placeholder function `prepare_input(m)`. You will need to implement this function.
        input_tensor = self.prepare_input(m) 
        # Then you can use the evaluate_model function to get the output of the CNN
        output = self.evaluate_model(self.model, input_tensor)
        return output
    
    def prepare_input(self, m):
        # get the model variables 'u' and 'v'
        u = m.u
        v = m.v
        # Ensure that 'u' and 'v' are numpy arrays
        if not isinstance(u, np.ndarray):
            u = np.array(u)
        if not isinstance(v, np.ndarray):
            v = np.array(v)
        # concatenate 'u' and 'v' across the first axis
        combined = np.concatenate((u, v), axis=0)
        # Convert the combined array to a torch tensor
        input_tensor = torch.from_numpy(combined)
        # Ensure the tensor is of the right type
        input_tensor = input_tensor.float()
        return input_tensor

    def evaluate_model(self, model, input_tensor):
        # Set device to GPU if available, otherwise use CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Move the input tensor to the specified device
        input_tensor = input_tensor.to(device=device)

        # Normalize the input tensor    
        input_tensor = (input_tensor - self.input_mean.to(device)) / self.input_std.to(device)
        
        # Pass the input tensor through the model
        output, _ = model(input_tensor)
        
        print(output.shape)
        # Denormalize the output tensor
        output = output * self.output_std.to(device) + self.output_mean.to(device)
        # Move the output tensor back to the CPU and numpy
        output = output.cpu().detach().numpy()
        
        # Ensure 'output' is of type float64
        output = output.astype(np.float64)

        return output

    def init_model(self, model_type, model_path):
    # Define the ConvNeuralNet classes for each model type
# -------------------------------------------------- Deep ------------------------------------------------------------------------------------------------------------------------
        if model_type == 'deep':
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
# -------------------------------------------------- Shallow ------------------------------------------------------------------------------------------------------------------------
        elif model_type == 'shallow':
            class ConvNeuralNet(nn.Module):

                # Determine what layers and their order in CNN object
                def __init__(self, num_classes):

                        super(ConvNeuralNet, self).__init__()
                        # Input Layer
                        self.conv_layer1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=5, padding="same")
                        
                        # Hidden layers
                        self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                        # self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                        # self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                        # self.conv_layer5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                        # self.conv_layer6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                        # self.conv_layer7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
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
                    
                        # out3_before = self.conv_layer3(out2_after) #Layer3
                        # out3_after = self.relu1(out3_before) #Layer3

                        # out4_before = self.conv_layer4(out3_after) #Layer4
                        # out4_after = self.relu1(out4_before) #Layer4

                        # out5_before = self.conv_layer5(out4_after) #Layer5
                        # out5_after = self.relu1(out5_before) #Layer5

                        # out6_before = self.conv_layer6(out5_after) #Layer6
                        # out6_after = self.relu1(out6_before) #Layer6

                        # out7_before = self.conv_layer7(out6_after) #Layer7
                        # out7_after = self.relu1(out7_before) #Layer7

                        # out8 = self.relu1(self.conv_layer8(out7)) #Layer8

                        # out9 = self.relu1(self.conv_layer9(out8)) #Layer9

                        # out10 = self.relu1(self.conv_layer10(out9)) #Layer10
                        
                        ####  !!! Do not forget to change teh output layer when changing the number of hidden layers !!! ####

                        output = self.conv_layer11(out2_after) #Layer11 (Output Layer) 
                        mid_ouptut = {'out1_before': out1_before, 'out1_after': out1_after, 'out2_before': out2_before, 'out2_after': out2_after, }
                        return output, mid_ouptut
        else:
            print('Invalid model type. Please choose from "deep" or "shallow"')

        # Set device to GPU if available, otherwise use CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model instance and move it to the specified device
        model = ConvNeuralNet(1).to(device=device)

        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Extract the model_state_dict from the checkpoint
        model_state_dict = checkpoint['model_state_dict']

        # Load the state dictionary into the model
        model.load_state_dict(model_state_dict)

        # Move the model to the specified device
        model.to(device)

        # Set the model to evaluation mode
        model.eval()

        return model

    @property
    def parameterization_type(self):
        return "q_parameterization"

import xarray as xr
import numpy as np
import pyqg

def drop_vars(ds):
    '''
    Drop complex variables 
    and convert to float32
    '''
    for key,var in ds.variables.items():
        if var.dtype == np.float64:
            ds[key] = var.astype(np.float32)
        elif var.dtype == np.complex128:
            ds = ds.drop_vars(key)
    for key in ['dqdt', 'ufull', 'vfull']:
        if key in ds.keys():
            ds = ds.drop_vars([key])
    if 'p' in ds.keys():
        ds = ds.rename({'p': 'psi'}) # Change for conventional name
    
    if ds['time'].attrs['units'] != 'days':
        ds['time'] = ds.time.values / 86400
        ds['time'].attrs['units'] = 'days'
    
    return ds

def concat_in_time(datasets):
    '''
    Concatenation of snapshots in time:
    - Concatenate everything
    - Store averaged statistics
    - Discard complex vars
    - Reduce precision
    '''
    from time import time
    # Concatenate datasets along the time dimension
    tt = time()
    ds = xr.concat(datasets, dim='time')

    # Spectral statistics are taken from the last 
    # snapshot because it is time-averaged
    for key in datasets[-1].keys():
        if 'k' in datasets[-1][key].dims:
            ds[key] = datasets[-1][key].isel(time=-1)
        
    ds = drop_vars(ds)

    return ds

def run_simulation(pyqg_params, parameterization=None, sampling_freq=24*60*60.):
    '''
    pyqg_params - only str-type parameters
    parameterization - pyqg parameterization class
    q_init - initial conditiona for PV, numpy array nlev*ny*nx
    '''
    ds = None
    pyqg_params['tmax'] = float(pyqg_params['tmax'])
    if parameterization is None:
        m = pyqg.QGModel(**pyqg_params)
    else:
        params = pyqg_params.copy()
        params['parameterization'] = parameterization
        m = pyqg.QGModel(**params, )
        
    for t in m.run_with_snapshots(tsnapstart=0, tsnapint=sampling_freq):
        
        _ds = drop_vars(m.to_dataset()).copy(deep=True)
        if ds is None:
            ds = _ds
        else:
            ds = concat_in_time([ds, _ds])
    ds.attrs['pyqg_params'] = str(pyqg_params)
    return ds

def run_parameterized_model(model, case, base_kwargs, sampling_freq=24*60*60.):
  parameterization = model 
  pyqg_params = {**case , **base_kwargs}
  ds = run_simulation(pyqg_params, parameterization, sampling_freq=sampling_freq)
  return ds

## ------------------------------------------------ RUNS ------------------------------------------------##

year = 24*60*60*360.
day = 24*60*60.
hr = 60*60.

base_kwargs = dict(tmax=10*year, dt=1*hr, twrite=10000, tavestart=5*year, taveint=1*day)


bnn = { "beta": 1.5e-11, #[m^-1 s^-1]
       "delta": 0.25,
       "rek": 5.78e-7, #[s^-1]
       "rd": 15e3, # [m]
}



case2 = { "beta": 5e-12,
            "delta": 0.05,
            "rek": 3.5e-8,
            "rd": 15e3,
}

case3 = { "beta": 5e-12,
            "delta": 0.05,
            "rek": 3.5e-8,
            "rd": 20e3,
}
case4 = { "beta": 1e-11,
       "delta": 0.1,
       "rek": 7e-8, #[s^-1]
       "rd": 15e3,
}

# base_kwargs['tmax'] = 20*year
sims = {}
# sims['case2'] = {
#     'label': 'Case2',
#     'cnn_path':'results/test_case2_with_better_data/BestModelBasedOnTestLoss.pt',
#     'bnn_path':'results/BNN/BestModelBasedOnTestLoss.pt',
#     'tl_path':'results/bnn_case2_10p/final_model_after_training.pt',
#     'shallow_path':'results/shallow_1HL_Case2/BestModelBasedOnTestLoss.pt',

#     'cnn_stat_path': 'results/test_case2_with_better_data/training_mean_std.txt',
#     'bnn_stat_path': 'results/BNN/training_mean_std.txt',
#     'tl_stat_path': 'results/bnn_case2_10p/training_mean_std.txt',
#     'shallow_stat_path': 'results/shallow_Case2/training_mean_std.txt',

#     'case': case2,
# }
# sims['case3'] = {
#     'label': 'Case3',
#     'cnn_path':'results/Case3/BestModelBasedOnTestLoss.pt',
#     'bnn_path':'results/BNN/BestModelBasedOnTestLoss.pt',
#     'tl_path':'results/bnn_case3_10p/final_model_after_training.pt',
#     'shallow_path':'results/shallow_1HL_Case3/BestModelBasedOnTestLoss.pt',

#     'cnn_stat_path': 'results/Case3/training_mean_std.txt',
#     'bnn_stat_path': 'results/BNN/training_mean_std.txt',
#     'tl_stat_path': 'results/bnn_case3_10p/training_mean_std.txt',
#     'shallow_stat_path': 'results/shallow_Case3/training_mean_std.txt',

#     'case': case3,
# }
sims['case4'] = {
    'label': 'case4',
    'cnn_path':'results/Case4/BestModelBasedOnTestLoss.pt',
    'bnn_path':'results/BNN/BestModelBasedOnTestLoss.pt',
    'tl_path':'results/bnn_case4_10p/final_model_after_training.pt',
    'shallow_path':'results/shallow_1HL_Case4/BestModelBasedOnTestLoss.pt',

    'cnn_stat_path': 'results/Case4/training_mean_std.txt',
    'bnn_stat_path': 'results/BNN/training_mean_std.txt',
    'tl_stat_path': 'results/bnn_case4_10p/training_mean_std.txt',
    'shallow_stat_path': 'results/shallow_Case4/training_mean_std.txt',

    'case': case4,
}
# sims['bnn'] = {
#     'label': 'bnn_test_smaller_std',
#     'cnn_path':'results/BNN/BestModelBasedOnTestLoss.pt',
#     'bnn_path':'results/BNN/BestModelBasedOnTestLoss.pt',
#     'tl_path':'results/BNN/final_model_after_training.pt',
#     'shallow_path':'results/BNN/BestModelBasedOnTestLoss.pt',

#     'cnn_stat_path': 'results/BNN/training_mean_std.txt',
#     'bnn_stat_path': 'results/BNN/training_mean_std.txt',
#     'tl_stat_path': 'results/BNN/training_mean_std.txt',
#     'shallow_stat_path': 'results/BNN/training_mean_std.txt',

#     'case': bnn,
# }
for key, value in sims.items():
    label = value['label']
    cnn_path = value['cnn_path']
    bnn_path = value['bnn_path']
    tl_path = value['tl_path']
    shallow_path = value['shallow_path']
    cnn_stat_path = value['cnn_stat_path']
    bnn_stat_path = value['bnn_stat_path']
    tl_stat_path = value['tl_stat_path']
    shallow_stat_path = value['shallow_stat_path']
    case = value['case']
    
    # lores = run_parameterized_model(model=None, case={'nx':64, **case}, base_kwargs=base_kwargs, sampling_freq=24*60*60.)
    # lores.to_netcdf(f'{label}_lores.nc')
    # print("lores done")

    # hires = run_parameterized_model(model=None, case={'nx':256, **case}, base_kwargs=base_kwargs, sampling_freq=24*60*60.)
    # hires.to_netcdf(f'{label}_hires.nc')
    # print("hires done")

    cnn_param = CNNparametrization('deep', cnn_path, cnn_stat_path)
    print(cnn_param)
    cnn = run_parameterized_model(model=cnn_param, case={'nx':64, **case}, base_kwargs=base_kwargs, sampling_freq=24*60*60.)
    cnn.to_netcdf(f'{label}_cnn.nc')
    print("cnn done")

    # bnn_param = CNNparametrization('deep', bnn_path, bnn_stat_path)
    # bnn = run_parameterized_model(model=bnn_param, case={'nx':64, **case}, base_kwargs=base_kwargs, sampling_freq=24*60*60.)
    # bnn.to_netcdf(f'{label}_bnn.nc')
    # print("bnn done")

    # tl_param = CNNparametrization('deep', tl_path, tl_stat_path)
    # tl = run_parameterized_model(model=tl_param, case={'nx':64, **case}, base_kwargs=base_kwargs, sampling_freq=24*60*60.)
    # tl.to_netcdf(f'{label}_tl.nc')
    # print("tl done")

    # shallow_param = CNNparametrization('shallow', shallow_path, shallow_stat_path)
    # shallow = run_parameterized_model(model=shallow_param, case={'nx':64, **case}, base_kwargs=base_kwargs, sampling_freq=24*60*60.)
    # shallow.to_netcdf(f'{label}_shallow.nc')
    # print("shallow done")

