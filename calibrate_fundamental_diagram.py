# -*- coding: utf-8 -*-
# Citation: Cheng, Q., Liu, Z., Lin, Y., Zhou, X., 2021. An s-shaped three-parameter (S3) traffic stream model with consistent car following relationship. Under review with Transportation Research Part B.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fundamental_diagram_model import fundamental_diagram as fd
from fundamental_diagram_model import estimated_value, theoretical_value
from scipy.optimize import minimize, Bounds
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rc('font',family='Times New Roman')
plt.rcParams['mathtext.fontset']='stix'

class solve:

    def __init__(self, data):
        
        self.speed = np.array(data.Speed)
        self.density = np.array(data.Density)
        self.flow = np.array(data.Flow)
        self.init_model_dict()
        
    def init_model_dict(self):
        
        self.model = fd()
        self.estimated_value = estimated_value()
        self.theoretical_value = theoretical_value()
        
        self.model_dict = {"S3":self.model.S3,
                           "Greenshields":self.model.Greenshields,
                           "Greenberg":self.model.Greenberg,
                           "Underwood":self.model.Underwood,
                           "NF":self.model.NF,
                           "GHR_M1":self.model.GHR_M1,
                           "GHR_M2":self.model.GHR_M2,
                           "GHR_M3":self.model.GHR_M3,
                           "KK":self.model.KK,
                           "Jayakrishnan":self.model.Jayakrishnan,
                           "Van_Aerde":self.model.Van_Aerde,
                           "MacNicholas":self.model.MacNicholas,
                           "Wang_3PL":self.model.Wang_3PL,
                           "Wang_4PL":self.model.Wang_4PL,
                           "Wang_5PL":self.model.Wang_5PL,
                           "Ni":self.model.Ni,
                           "S3_joint_estimation":self.model.S3_joint_estimation,
                           "Greenshields_joint_estimation":self.model.Greenshields_joint_estimation,
                           "Greenberg_joint_estimation":self.model.Greenberg_joint_estimation,
                           "Underwood_joint_estimation":self.model.Underwood_joint_estimation,
                           "NF_joint_estimation":self.model.NF_joint_estimation,
                           "GHR_M1_joint_estimation":self.model.GHR_M1_joint_estimation,
                           "GHR_M2_joint_estimation":self.model.GHR_M2_joint_estimation,
                           "GHR_M3_joint_estimation":self.model.GHR_M3_joint_estimation,
                           "KK_joint_estimation":self.model.KK_joint_estimation,
                           "Jayakrishnan_joint_estimation":self.model.Jayakrishnan_joint_estimation,
                           "Van_Aerde_joint_estimation":self.model.Van_Aerde_joint_estimation,
                           "MacNicholas_joint_estimation":self.model.MacNicholas_joint_estimation,
                           "Wang_3PL_joint_estimation":self.model.Wang_3PL_joint_estimation,
                           "Wang_4PL_joint_estimation":self.model.Wang_4PL_joint_estimation,
                           "Wang_5PL_joint_estimation":self.model.Wang_5PL_joint_estimation,
                           "Ni_joint_estimation":self.model.Ni_joint_estimation
                           }
        
        self.estimated_value_dict = {"S3":self.estimated_value.S3,
                                     "Greenshields":self.estimated_value.Greenshields,
                                     "Greenberg":self.estimated_value.Greenberg,
                                     "Underwood":self.estimated_value.Underwood,
                                     "NF":self.estimated_value.NF,
                                     "GHR_M1":self.estimated_value.GHR_M1,
                                     "GHR_M2":self.estimated_value.GHR_M2,
                                     "GHR_M3":self.estimated_value.GHR_M3,
                                     "KK":self.estimated_value.KK,
                                     "Jayakrishnan":self.estimated_value.Jayakrishnan,
                                     "Van_Aerde":self.estimated_value.Van_Aerde,
                                     "MacNicholas":self.estimated_value.MacNicholas,
                                     "Wang_3PL":self.estimated_value.Wang_3PL,
                                     "Wang_4PL":self.estimated_value.Wang_4PL,
                                     "Wang_5PL":self.estimated_value.Wang_5PL,
                                     "Ni":self.estimated_value.Ni,
                                     "S3_joint_estimation":self.estimated_value.S3_joint_estimation,
                                     "Greenshields_joint_estimation":self.estimated_value.Greenshields_joint_estimation,
                                     "Greenberg_joint_estimation":self.estimated_value.Greenberg_joint_estimation,
                                     "Underwood_joint_estimation":self.estimated_value.Underwood_joint_estimation,
                                     "NF_joint_estimation":self.estimated_value.NF_joint_estimation,
                                     "GHR_M1_joint_estimation":self.estimated_value.GHR_M1_joint_estimation,
                                     "GHR_M2_joint_estimation":self.estimated_value.GHR_M2_joint_estimation,
                                     "GHR_M3_joint_estimation":self.estimated_value.GHR_M3_joint_estimation,
                                     "KK_joint_estimation":self.estimated_value.KK_joint_estimation,
                                     "Jayakrishnan_joint_estimation":self.estimated_value.Jayakrishnan_joint_estimation,
                                     "Van_Aerde_joint_estimation":self.estimated_value.Van_Aerde_joint_estimation,
                                     "MacNicholas_joint_estimation":self.estimated_value.MacNicholas_joint_estimation,
                                     "Wang_3PL_joint_estimation":self.estimated_value.Wang_3PL_joint_estimation,
                                     "Wang_4PL_joint_estimation":self.estimated_value.Wang_4PL_joint_estimation,
                                     "Wang_5PL_joint_estimation":self.estimated_value.Wang_5PL_joint_estimation,
                                     "Ni_joint_estimation":self.estimated_value.Ni_joint_estimation,
                                       }
        
        self.theoretical_value_dict = {"S3":self.theoretical_value.S3,
                                       "Greenshields":self.theoretical_value.Greenshields,
                                       "Greenberg":self.theoretical_value.Greenberg,
                                       "Underwood":self.theoretical_value.Underwood,
                                       "NF":self.theoretical_value.NF,
                                       "GHR_M1":self.theoretical_value.GHR_M1,
                                       "GHR_M2":self.theoretical_value.GHR_M2,
                                       "GHR_M3":self.theoretical_value.GHR_M3,
                                       "KK":self.theoretical_value.KK,
                                       "Jayakrishnan":self.theoretical_value.Jayakrishnan,
                                       "Van_Aerde":self.theoretical_value.Van_Aerde,
                                       "MacNicholas":self.theoretical_value.MacNicholas,
                                       "Wang_3PL":self.theoretical_value.Wang_3PL,
                                       "Wang_4PL":self.theoretical_value.Wang_4PL,
                                       "Wang_5PL":self.theoretical_value.Wang_5PL,
                                       "Ni":self.theoretical_value.Ni,
                                       "S3_joint_estimation":self.theoretical_value.S3_joint_estimation,
                                       "Greenshields_joint_estimation":self.theoretical_value.Greenshields_joint_estimation,
                                       "Greenberg_joint_estimation":self.theoretical_value.Greenberg_joint_estimation,
                                       "Underwood_joint_estimation":self.theoretical_value.Underwood_joint_estimation,
                                       "NF_joint_estimation":self.theoretical_value.NF_joint_estimation,
                                       "GHR_M1_joint_estimation":self.theoretical_value.GHR_M1_joint_estimation,
                                       "GHR_M2_joint_estimation":self.theoretical_value.GHR_M2_joint_estimation,
                                       "GHR_M3_joint_estimation":self.theoretical_value.GHR_M3_joint_estimation,
                                       "KK_joint_estimation":self.theoretical_value.KK_joint_estimation,
                                       "Jayakrishnan_joint_estimation":self.theoretical_value.Jayakrishnan_joint_estimation,
                                       "Van_Aerde_joint_estimation":self.theoretical_value.Van_Aerde_joint_estimation,
                                       "MacNicholas_joint_estimation":self.theoretical_value.MacNicholas_joint_estimation,
                                       "Wang_3PL_joint_estimation":self.theoretical_value.Wang_3PL_joint_estimation,
                                       "Wang_4PL_joint_estimation":self.theoretical_value.Wang_4PL_joint_estimation,
                                       "Wang_5PL_joint_estimation":self.theoretical_value.Wang_5PL_joint_estimation,
                                       "Ni_joint_estimation":self.theoretical_value.Ni_joint_estimation,
                                       }
        
        self.bounds = {"S3": Bounds([60, 20, 1], [80, 60, 10]),
                       "Greenshields":Bounds([60, 120], [80, 200]),
                       "Greenberg":Bounds([20, 140], [70, 180]),
                       "Underwood":Bounds([60, 20], [80, 60]),
                       "NF":Bounds([60, 140, 0], [80, 200, 5000]),
                       "GHR_M1":Bounds([60, 20], [80, 60]),
                       "GHR_M2":Bounds([60, 133, 0], [80, 200, 10]),
                       "GHR_M3":Bounds([60, 133, 0.0001], [80, 200, 10]),
                       "KK":Bounds([60, 20, 0, 0, 0], [80, 60, np.inf, np.inf, np.inf]),
                       "Jayakrishnan":Bounds([60, 0, 133, 0], [80, 10, 200, np.inf]),
                       "Van_Aerde":Bounds([60, 20, 133, 1400], [80, 70, 200, 2000]),
                       "MacNicholas":Bounds([60, 140, 1, 0], [80, 200, 10, np.inf]),
                       "Wang_3PL":Bounds([60, 20, 1], [80, 60, 20]),
                       "Wang_4PL":Bounds([60, 0, 20, 1], [80, 10, 60, 20]),
                       "Wang_5PL":Bounds([60, 0, 20, 1, 0], [80, 10, 60, 20, 1]),
                       "Ni":Bounds([60, -1, 0, 0], [80, 0, 3, 10]),
                       "S3_joint_estimation": Bounds([60, 20, 1], [80, 60, 10]),
                       "Greenshields_joint_estimation": Bounds([60, 120], [80, 200]),
                       "Greenberg_joint_estimation":Bounds([20, 140], [70, 180]),
                       "Underwood_joint_estimation":Bounds([60, 20], [80, 60]),
                       "NF_joint_estimation":Bounds([60, 140, 0], [80, 200, 5000]),
                       "GHR_M1_joint_estimation":Bounds([60, 20], [80, 60]),
                       "GHR_M2_joint_estimation":Bounds([60, 133, 0], [80, 200, 10]),
                       "GHR_M3_joint_estimation":Bounds([60, 133, 0.0001], [80, 200, 10]),
                       "KK_joint_estimation":Bounds([60, 20, 0, 0, 0], [80, 60, np.inf, np.inf, np.inf]),
                       "Jayakrishnan_joint_estimation":Bounds([60, 0, 133, 0], [80, 10, 200, np.inf]),
                       "Van_Aerde_joint_estimation":Bounds([60, 20, 133, 1400], [80, 70, 200, 2000]),
                       "MacNicholas_joint_estimation":Bounds([60, 140, 1, 0], [80, 200, 10, np.inf]),
                       "Wang_3PL_joint_estimation":Bounds([60, 20, 0], [80, 60, 20]),
                       "Wang_4PL_joint_estimation":Bounds([60, 0, 20, 0], [80, 10, 60, 20]),
                       "Wang_5PL_joint_estimation":Bounds([60, 0, 20, 0, 0], [80, 10, 60, 20, 1]),
                       "Ni_joint_estimation":Bounds([60, -1, 0, 0], [80, 0, 3, 10])
                       }
        
    def getSolution(self, model_str, init_solu):
        # calibration
        objective = self.model_dict[model_str]
        bound = self.bounds[model_str]
        solution = minimize(objective, init_solu, args=(self.flow, self.density, self.speed), method='trust-constr', bounds=bound)
        return solution.x
    
    def getEstimatedValue(self, model_str, parameters):
        # Claculate the estimated flow/speed/density based on the fundamental diagram model.
        # Based on these estimated values, we can further calculate the metrics, such as MRE, MSE, RMSE, etc.
        estimated_values = self.estimated_value_dict[model_str]
        if model_str in ["Van_Aerde", "Ni", "Van_Aerde_joint_estimation", "Ni_joint_estimation"]:
            estimated_density = estimated_values(parameters, self.flow, self.density, self.speed)[0]
            estimated_flow = estimated_values(parameters, self.flow, self.density, self.speed)[1]
            return estimated_density, estimated_flow
        elif model_str in ["S3","Greenshields","Greenberg","Underwood","NF","GHR_M1","GHR_M2","GHR_M3","KK","Jayakrishnan","MacNicholas","Wang_3PL","Wang_4PL","Wang_5PL","S3_joint_estimation","Greenshields_joint_estimation","Greenberg_joint_estimation","Underwood_joint_estimation","NF_joint_estimation","GHR_M1_joint_estimation","GHR_M2_joint_estimation","GHR_M3_joint_estimation","KK_joint_estimation","Jayakrishnan_joint_estimation","MacNicholas_joint_estimation","Wang_3PL_joint_estimation","Wang_4PL_joint_estimation","Wang_5PL_joint_estimation"]:
            estimated_speed = estimated_values(parameters, self.flow, self.density, self.speed)[0]
            estimated_flow = estimated_values(parameters, self.flow, self.density, self.speed)[1]
            return estimated_speed, estimated_flow
        else:
            print('Please define the fundamental diagram model before calibration.')
    
    def plot_fd(self, model_str, para, para_S3):
        
        dirs = './Figures' 
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        
        self.k = np.linspace(0.000001,140,70)
        self.v = np.linspace(0.000001,90,70)
        theoretical_values = self.theoretical_value_dict[model_str]
        theoretical_values_S3 = self.theoretical_value_dict["S3"]
        theoretical_speed_S3 = theoretical_values_S3(para_S3, self.k)[0]
        theoretical_flow_S3 = theoretical_values_S3(para_S3, self.k)[1]
        
        if model_str in ["Van_Aerde", "Ni", "Van_Aerde_joint_estimation", "Ni_joint_estimation"]:
            theoretical_density = theoretical_values(para, self.k)[0]
            theoretical_flow = theoretical_values(para, self.k)[1]
            x1 = theoretical_density
            y1 = theoretical_flow
            x2 = theoretical_density
            y2 = self.v
            x3 = theoretical_flow
            y3 = self.v
            pass 
        elif model_str in ["S3","Greenshields","Greenberg","Underwood","NF","GHR_M1","GHR_M2","GHR_M3","KK","Jayakrishnan","MacNicholas","Wang_3PL","Wang_4PL","Wang_5PL","S3_joint_estimation","Greenshields_joint_estimation","Greenberg_joint_estimation","Underwood_joint_estimation","NF_joint_estimation","GHR_M1_joint_estimation","GHR_M2_joint_estimation","GHR_M3_joint_estimation","KK_joint_estimation","Jayakrishnan_joint_estimation","MacNicholas_joint_estimation","Wang_3PL_joint_estimation","Wang_4PL_joint_estimation","Wang_5PL_joint_estimation"]:
            theoretical_speed = theoretical_values(para, self.k)[0]
            theoretical_flow = theoretical_values(para, self.k)[1]
            x1 = self.k
            y1 = theoretical_flow
            x2 = self.k
            y2 = theoretical_speed
            x3 = theoretical_flow
            y3 = theoretical_speed
            pass
        
        fig = plt.figure(figsize=(7,5))
        plt.scatter(self.density.flatten(), self.flow.flatten(), s = 3, marker='o', c='r', edgecolors='r', label = 'Observation')
        plt.plot(x1, y1, 'y-', linewidth=3, label = model_str)
        plt.plot(self.k, theoretical_flow_S3, 'b--', linewidth=4, label = "S3")
        plt.plot()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Density (veh/km)', fontsize=16)
        plt.ylabel('Flow (veh/h)', fontsize=16)
        plt.xlim((0, 140))
        plt.ylim((0, 2250))
        plt.legend(loc='upper right', fontsize=14)
        plt.title('Flow vs. density', fontsize=20)
        fig.savefig("Figures\\flow vs density_{}.png".format(model_str), dpi=400, bbox_inches='tight')
        
        fig = plt.figure(figsize=(7,5))
        plt.scatter(self.density.flatten(), self.speed.flatten(), s = 3, marker='o', c='r', edgecolors='r', label = 'Observation')
        plt.plot(x2, y2, 'y-', linewidth=3, label = model_str)
        plt.plot(self.k, theoretical_speed_S3, 'b--', linewidth=4, label = "S3")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Density (veh/km)', fontsize=16)
        plt.ylabel('Speed (km/h)', fontsize=16)
        plt.xlim((0, 140))
        plt.ylim((0, 90))
        plt.legend(loc='upper right', fontsize=14)
        plt.title('Speed vs. density', fontsize=20)
        fig.savefig("Figures\\speed vs density_{}.png".format(model_str), dpi=400, bbox_inches='tight')
        
        fig = plt.figure(figsize=(7,5))
        plt.scatter(self.flow.flatten(), self.speed.flatten(), s = 3, marker='o', c='r', edgecolors='r', label = 'Observation')
        plt.plot(x3, y3, 'y-', linewidth=3, label = model_str)
        plt.plot(theoretical_flow_S3, theoretical_speed_S3, 'b--', linewidth=4, label = "S3")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Flow (veh/h)', fontsize=16)
        plt.ylabel('Speed (km/h)', fontsize=16)
        plt.xlim((0,2250))
        plt.ylim((0, 90))
        plt.legend(loc='upper right', fontsize=14)
        plt.title('Speed vs. flow', fontsize=20)
        fig.savefig("Figures\\speed vs flow_{}.png".format(model_str), dpi=400, bbox_inches='tight')
    

if __name__ == '__main__':
    
    # load q-k-v data
    data = pd.read_csv('Data/input_data.csv', header=0)
    
    # Calibrate for different FD models
    x0 = {"S3":[70, 35, 3.6],
          "Greenshields":[70, 140],
          "Greenberg":[35, 140],
          "Underwood":[70, 35],
          "NF":[70, 140, 1],
          "GHR_M1":[70, 35],
          "GHR_M2":[70, 140, 2],
          "GHR_M3":[70, 140, 2],
          "KK":[70, 35, 0.25, 0.06, 3.72e-06],
          "Jayakrishnan":[70, 3, 140, 2],
          "Van_Aerde":[70,35,130,1700],
          "MacNicholas":[70, 140, 3, 2],
          "Wang_3PL":[70, 35, 1],
          "Wang_4PL":[70, 5, 35, 1],
          "Wang_5PL":[70, 5, 35, 1, 1],
          "Ni":[70, -3.48e-6, 1/3600, 7.5e-3],
          "S3_joint_estimation":[70, 35, 3.6],
          "Greenshields_joint_estimation":[70, 140],
          "Greenberg_joint_estimation":[35, 140],
          "Underwood_joint_estimation":[70, 35],
          "NF_joint_estimation":[70, 140, 1],
          "GHR_M1_joint_estimation":[70, 35],
          "GHR_M2_joint_estimation":[70, 140, 2],
          "GHR_M3_joint_estimation":[70, 140, 2],
          "KK_joint_estimation":[70, 35, 0.25, 0.06, 3.72e-06],
          "Jayakrishnan_joint_estimation":[70, 3, 140, 2],
          "Van_Aerde_joint_estimation":[70,35,130,1700],
          "MacNicholas_joint_estimation":[70, 140, 3, 2],
          "Wang_3PL_joint_estimation":[70, 35, 1],
          "Wang_4PL_joint_estimation":[70, 5, 35, 1],
          "Wang_5PL_joint_estimation":[70, 5, 35, 1, 1],
          "Ni_joint_estimation":[70, -3.48e-6, 1/3600, 7.5e-3],
          }
    
    solver = solve(data)
    
    para = {"S3":solver.getSolution("S3", x0['S3']),
            "Greenshields":solver.getSolution("Greenshields", x0['Greenshields']),
            "Greenberg":solver.getSolution("Greenberg", x0['Greenberg']),
            "Underwood":solver.getSolution("Underwood", x0['Underwood']),
            "NF":solver.getSolution("NF", x0['NF']),
            "GHR_M1":solver.getSolution("GHR_M1", x0['GHR_M1']),
            "GHR_M2":solver.getSolution("GHR_M2", x0['GHR_M2']),
            "GHR_M3":solver.getSolution("GHR_M3", x0['GHR_M3']),
            "KK":solver.getSolution("KK", x0['KK']),
            "Jayakrishnan":solver.getSolution("Jayakrishnan", x0['Jayakrishnan']),
            # "Van_Aerde":solver.getSolution("Van_Aerde", x0['Van_Aerde']),
            "MacNicholas":solver.getSolution("MacNicholas", x0['MacNicholas']),
            "Wang_3PL":solver.getSolution("Wang_3PL", x0['Wang_3PL']),
            "Wang_4PL":solver.getSolution("Wang_4PL", x0['Wang_4PL']),
            "Wang_5PL":solver.getSolution("Wang_5PL", x0['Wang_5PL']),
            "S3_joint_estimation":solver.getSolution("S3_joint_estimation", x0['S3_joint_estimation']),
            "Greenshields_joint_estimation":solver.getSolution("Greenshields_joint_estimation", x0['Greenshields_joint_estimation']),
            "Greenberg_joint_estimation":solver.getSolution("Greenberg_joint_estimation", x0['Greenberg_joint_estimation']),
            "Underwood_joint_estimation":solver.getSolution("Underwood_joint_estimation", x0['Underwood_joint_estimation']),
            "NF_joint_estimation":solver.getSolution("NF_joint_estimation", x0['NF_joint_estimation']),
            "GHR_M1_joint_estimation":solver.getSolution("GHR_M1_joint_estimation", x0['GHR_M1_joint_estimation']),
            "GHR_M2_joint_estimation":solver.getSolution("GHR_M2_joint_estimation", x0['GHR_M2_joint_estimation']),
            "GHR_M3_joint_estimation":solver.getSolution("GHR_M3_joint_estimation", x0['GHR_M3_joint_estimation']),
            "KK_joint_estimation":solver.getSolution("KK_joint_estimation", x0['KK_joint_estimation']),
            "Jayakrishnan_joint_estimation":solver.getSolution("Jayakrishnan_joint_estimation", x0['Jayakrishnan_joint_estimation']),
            # "Van_Aerde_joint_estimation":solver.getSolution("Van_Aerde_joint_estimation", x0['Van_Aerde_joint_estimation']),
            "MacNicholas_joint_estimation":solver.getSolution("MacNicholas_joint_estimation", x0['MacNicholas_joint_estimation']),
            "Wang_3PL_joint_estimation":solver.getSolution("Wang_3PL_joint_estimation", x0['Wang_3PL_joint_estimation']),
            "Wang_4PL_joint_estimation":solver.getSolution("Wang_4PL_joint_estimation", x0['Wang_4PL_joint_estimation']),
            "Wang_5PL_joint_estimation":solver.getSolution("Wang_5PL_joint_estimation", x0['Wang_5PL_joint_estimation']),
            }
    
    dirs = './Results' 
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    np.save('Results//Calibrated parameters for different fundamental diagram models.npy', para)
    '''
    print('free flow speed = {} mile/hour, \ncritical density = {} veh/mile, \nfoc = {}'.format(np.round(para[0], 2), np.round(para[1], 2), np.round(para[2], 3)))
    '''
    '''
    # Calculate the estimated flow and speed
    estimated_flow = solver.getEstimatedValue("S3", para)[0]
    estimated_speed = solver.getEstimatedValue("S3", para)[1]
    np.savetxt("Data/output_flow.csv", estimated_flow, delimiter=',')
    np.savetxt("Data/output_speed.csv", estimated_speed, delimiter=',')
    '''
    # Plot the results
    plot_result = {"Greenshields":solver.plot_fd("Greenshields", para['Greenshields'], para['S3']),
                   "Greenberg":solver.plot_fd("Greenberg", para['Greenberg'], para['S3']),
                   "Underwood":solver.plot_fd("Underwood", para['Underwood'], para['S3']),
                   "NF":solver.plot_fd("NF", para['NF'], para['S3']),
                   "GHR_M1":solver.plot_fd("GHR_M1", para['GHR_M1'], para['S3']),
                   "GHR_M2":solver.plot_fd("GHR_M2", para['GHR_M2'], para['S3']),
                   "GHR_M3":solver.plot_fd("GHR_M3", para['GHR_M3'], para['S3']),
                   "KK":solver.plot_fd("KK", para['KK'], para['S3']),
                   "Jayakrishnan":solver.plot_fd("Jayakrishnan", para['Jayakrishnan'], para['S3']),
                   # "Van_Aerde":solver.plot_fd("Van_Aerde", para['Van_Aerde'], para['S3']),
                   "MacNicholas":solver.plot_fd("MacNicholas", para['MacNicholas'], para['S3']),
                   "Wang_3PL":solver.plot_fd("Wang_3PL", para['Wang_3PL'], para['S3']),
                   "Wang_4PL":solver.plot_fd("Wang_4PL", para['Wang_4PL'], para['S3']),
                   "Wang_5PL":solver.plot_fd("Wang_5PL", para['Wang_5PL'], para['S3']),
                   "Greenshields_joint_estimation":solver.plot_fd("Greenshields_joint_estimation", para['Greenshields_joint_estimation'], para['S3_joint_estimation']),
                   "Greenberg_joint_estimation":solver.plot_fd("Greenberg_joint_estimation", para['Greenberg_joint_estimation'], para['S3_joint_estimation']),
                   "Underwood_joint_estimation":solver.plot_fd("Underwood_joint_estimation", para['Underwood_joint_estimation'], para['S3_joint_estimation']),
                   "NF_joint_estimation":solver.plot_fd("NF_joint_estimation", para['NF_joint_estimation'], para['S3_joint_estimation']),
                   "GHR_M1_joint_estimation":solver.plot_fd("GHR_M1_joint_estimation", para['GHR_M1_joint_estimation'], para['S3_joint_estimation']),
                   "GHR_M2_joint_estimation":solver.plot_fd("GHR_M2_joint_estimation", para['GHR_M2_joint_estimation'], para['S3_joint_estimation']),
                   "GHR_M3_joint_estimation":solver.plot_fd("GHR_M3_joint_estimation", para['GHR_M3_joint_estimation'], para['S3_joint_estimation']),
                   "KK_joint_estimation":solver.plot_fd("KK_joint_estimation", para['KK_joint_estimation'], para['S3_joint_estimation']),
                   "Jayakrishnan_joint_estimation":solver.plot_fd("Jayakrishnan_joint_estimation", para['Jayakrishnan_joint_estimation'], para['S3_joint_estimation']),
                   # "Van_Aerde_joint_estimation":solver.plot_fd("Van_Aerde_joint_estimation", para['Van_Aerde_joint_estimation'], para['S3_joint_estimation']),
                   "MacNicholas_joint_estimation":solver.plot_fd("MacNicholas_joint_estimation", para['MacNicholas_joint_estimation'], para['S3_joint_estimation']),
                   "Wang_3PL_joint_estimation":solver.plot_fd("Wang_3PL_joint_estimation", para['Wang_3PL_joint_estimation'], para['S3_joint_estimation']),
                   "Wang_4PL_joint_estimation":solver.plot_fd("Wang_4PL_joint_estimation", para['Wang_4PL_joint_estimation'], para['S3_joint_estimation']),
                   "Wang_5PL_joint_estimation":solver.plot_fd("Wang_5PL_joint_estimation", para['Wang_5PL_joint_estimation'], para['S3_joint_estimation']),
                   }
