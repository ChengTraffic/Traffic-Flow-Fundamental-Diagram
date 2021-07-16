# -*- coding: utf-8 -*-

import numpy as np

class fundamental_diagram():

    def S3(self, beta, *args):
        vf, kc, foc = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf/np.power(1 + np.power((observed_density/kc), foc), 2/foc)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def Greenshields(self, beta, *args):
        vf, k_jam = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1 - observed_density/k_jam)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def Greenberg(self, beta, *args):
        vc, k_jam = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vc*np.log(k_jam/observed_density)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def Underwood(self, beta, *args):
        vf, kc = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*np.exp(-1*observed_density/kc)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def NF(self, beta, *args):
        vf, k_jam, lambda_NF = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1-np.exp(-1*lambda_NF/vf*(1/observed_density - 1/k_jam)))
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def GHR_M1(self, beta, *args):
        vf, kc = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*np.exp(-0.5*np.power(observed_density/kc, 2))
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def GHR_M2(self, beta, *args):
        vf, k_jam, m = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1 - np.power(observed_density/k_jam, m))
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def GHR_M3(self, beta, *args):
        vf, k_jam, m = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*np.power(1 - observed_density/k_jam, m)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def KK(self, beta, *args):
        vf, kc, c1, c2, c3 = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1/(1 + np.exp((observed_density/kc - c1)/c2)) - c3)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def Jayakrishnan(self, beta, *args):
        vf, v_jam, k_jam, m = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = v_jam + (vf - v_jam)*np.power(1 - observed_density/k_jam, m)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def Van_Aerde(self, beta, *args):
        vf, vc, k_jam, max_flow = beta
        c1 = vf/(k_jam*np.power(vc, 2))*(2*vc - vf)
        c2 = vf/(k_jam*np.power(vc, 2))*np.power(vf - vc, 2)
        c3 = 1/max_flow - vf/(k_jam*np.power(vc, 2))
        observed_flow, observed_density, observed_speed = args
        estimated_density = 1/(c1 + c2/(vf-observed_speed) + c3*observed_speed)
        f_obj = np.sum(np.power(estimated_density - observed_density, 2))
        return f_obj

    def MacNicholas(self, beta, *args):
        vf, k_jam, m, c = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(np.power(k_jam, m) - np.power(observed_density, m))/(np.power(k_jam, m) + c*np.power(observed_density, m))
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def Wang_3PL(self, beta, *args):
        vf, kc, theta = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf/(1+np.exp((observed_density-kc)/theta))
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def Wang_4PL(self, beta, *args):
        vf, vb, kc, theta = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vb + (vf-vb)/(1+np.exp((observed_density-kc)/theta))
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def Wang_5PL(self, beta, *args):
        vf, vb, kc, theta1, theta2 = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vb + (vf-vb)/np.power(1+np.exp((observed_density-kc)/theta1), theta2)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def Ni(self, beta, *args):
        vf, gamma, tao, l = beta
        observed_flow, observed_density, observed_speed = args
        estimated_density = 1/((gamma*np.power(observed_speed,2)+tao*observed_speed+l)*(1-np.log(1-observed_speed/vf)))
        f_obj = np.sum(np.power(estimated_density - observed_density, 2))
        return f_obj

    def S3_joint_estimation(self, beta, *args):
        vf, kc, foc = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf/np.power(1 + np.power((observed_density/kc), foc), 2/foc)
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def Greenshields_joint_estimation(self, beta, *args):
        vf, k_jam = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1 - observed_density/k_jam)
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def Greenberg_joint_estimation(self, beta, *args):
        vc, k_jam = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vc*np.log(k_jam/observed_density)
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def Underwood_joint_estimation(self, beta, *args):
        vf, kc = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*np.exp(-1*observed_density/kc)
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def NF_joint_estimation(self, beta, *args):
        vf, k_jam, lambda_NF = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1-np.exp(-1*lambda_NF/vf*(1/observed_density - 1/k_jam)))
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def GHR_M1_joint_estimation(self, beta, *args):
        vf, kc = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*np.exp(-0.5*np.power(observed_density/kc, 2))
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def GHR_M2_joint_estimation(self, beta, *args):
        vf, k_jam, m = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1 - np.power(observed_density/k_jam, m))
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def GHR_M3_joint_estimation(self, beta, *args):
        vf, k_jam, m = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*np.power(1 - observed_density/k_jam, m)
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def KK_joint_estimation(self, beta, *args):
        vf, kc, c1, c2, c3 = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1/(1 + np.exp((observed_density/kc - c1)/c2)) - c3)
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def Jayakrishnan_joint_estimation(self, beta, *args):
        vf, v_jam, k_jam, m = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = v_jam + (vf - v_jam)*np.power(1 - observed_density/k_jam, m)
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def Van_Aerde_joint_estimation(self, beta, *args):
        vf, vc, k_jam, max_flow = beta
        c1 = vf/(k_jam*np.power(vc, 2))*(2*vc - vf)
        c2 = vf/(k_jam*np.power(vc, 2))*np.power(vf - vc, 2)
        c3 = 1/max_flow - vf/(k_jam*np.power(vc, 2))
        observed_flow, observed_density, observed_speed = args
        estimated_density = 1/(c1 + c2/(vf-observed_speed) + c3*observed_speed)
        estimated_flow = observed_speed * estimated_density
        sigma = np.var(observed_density) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_density - observed_density, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def MacNicholas_joint_estimation(self, beta, *args):
        vf, k_jam, m, c = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(np.power(k_jam, m) - np.power(observed_density, m))/(np.power(k_jam, m) + c*np.power(observed_density, m))
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def Wang_3PL_joint_estimation(self, beta, *args):
        vf, kc, theta = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf/(1+np.exp((observed_density-kc)/theta))
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def Wang_4PL_joint_estimation(self, beta, *args):
        vf, vb, kc, theta = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vb + (vf-vb)/(1+np.exp((observed_density-kc)/theta))
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def Wang_5PL_joint_estimation(self, beta, *args):
        vf, vb, kc, theta1, theta2 = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vb + (vf-vb)/np.power(1+np.exp((observed_density-kc)/theta1), theta2)
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def Ni_joint_estimation(self, beta, *args):
        vf, gamma, tao, l = beta
        observed_flow, observed_density, observed_speed = args
        estimated_density = 1/((gamma*np.power(observed_speed,2)+tao*observed_speed+l)*(1-np.log(1-observed_speed/vf)))
        estimated_flow = observed_speed * estimated_density
        sigma = np.var(observed_density) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_density - observed_density, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj


class estimated_value():

    def S3(self, beta, *args):
        vf, kc, foc = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf/np.power(1 + np.power((observed_density/kc), foc), 2/foc)
        estimated_flow = observed_density*vf/np.power(1 + np.power((observed_density/kc), foc), 2/foc)
        return estimated_speed, estimated_flow

    def Greenshields(self, beta, *args):
        vf, k_jam = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1 - observed_density/k_jam)
        estimated_flow = observed_density*vf*(1 - observed_density/k_jam)
        return estimated_speed, estimated_flow

    def Greenberg(self, beta, *args):
        vc, k_jam = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vc*np.log(k_jam/observed_density)
        estimated_flow = observed_density*vc*np.log(k_jam/observed_density)
        return estimated_speed, estimated_flow

    def Underwood(self, beta, *args):
        vf, kc = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*np.exp(-1*observed_density/kc)
        estimated_flow = observed_density*vf*np.exp(-1*observed_density/kc)
        return estimated_speed, estimated_flow

    def NF(self, beta, *args):
        vf, k_jam, lambda_NF = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1-np.exp(-1*lambda_NF/vf*(1/observed_density - 1/k_jam)))
        estimated_flow = observed_density*vf*(1-np.exp(-1*lambda_NF/vf*(1/observed_density - 1/k_jam)))
        return estimated_speed, estimated_flow

    def GHR_M1(self, beta, *args):
        vf, kc = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*np.exp(-0.5*np.power(observed_density/kc, 2))
        estimated_flow = observed_density*vf*np.exp(-0.5*np.power(observed_density/kc, 2))
        return estimated_speed, estimated_flow

    def GHR_M2(self, beta, *args):
        vf, k_jam, m = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1 - np.power(observed_density/k_jam, m))
        estimated_flow = observed_density*vf*(1 - np.power(observed_density/k_jam, m))
        return estimated_speed, estimated_flow

    def GHR_M3(self, beta, *args):
        vf, k_jam, m = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*np.power(1 - observed_density/k_jam, m)
        estimated_flow = observed_density*vf*np.power(1 - observed_density/k_jam, m)
        return estimated_speed, estimated_flow

    def KK(self, beta, *args):
        vf, kc, c1, c2, c3 = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1/(1 + np.exp((observed_density/kc - c1)/c2)) - c3)
        estimated_flow = observed_density*vf*(1/(1 + np.exp((observed_density/kc - c1)/c2)) - c3)
        return estimated_speed, estimated_flow

    def Jayakrishnan(self, beta, *args):
        vf, v_jam, k_jam, m = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = v_jam + (vf - v_jam)*np.power(1 - observed_density/k_jam, m)
        estimated_flow = observed_density*(v_jam + (vf - v_jam)*np.power(1 - observed_density/k_jam, m))
        return estimated_speed, estimated_flow

    def Van_Aerde(self, beta, *args):
        vf, vc, k_jam, max_flow = beta
        c1 = vf/(k_jam*np.power(vc, 2))*(2*vc - vf)
        c2 = vf/(k_jam*np.power(vc, 2))*np.power(vf - vc, 2)
        c3 = 1/max_flow - vf/(k_jam*np.power(vc, 2))
        observed_flow, observed_density, observed_speed = args
        estimated_density = 1/(c1 + c2/(vf-observed_speed) + c3*observed_speed)
        estimated_flow = observed_speed/(c1 + c2/(vf-observed_speed) + c3*observed_speed)
        return estimated_density, estimated_flow

    def MacNicholas(self, beta, *args):
        vf, k_jam, m, c = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(np.power(k_jam, m) - np.power(observed_density, m))/(np.power(k_jam, m) + c*np.power(observed_density, m))
        estimated_flow = observed_density*vf*(np.power(k_jam, m) - np.power(observed_density, m))/(np.power(k_jam, m) + c*np.power(observed_density, m))
        return estimated_speed, estimated_flow

    def Wang_3PL(self, beta, *args):
        vf, kc, theta = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf/(1+np.exp((observed_density-kc)/theta))
        estimated_flow = observed_density*vf/(1+np.exp((observed_density-kc)/theta))
        return estimated_speed, estimated_flow

    def Wang_4PL(self, beta, *args):
        vf, vb, kc, theta = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vb + (vf-vb)/(1+np.exp((observed_density-kc)/theta))
        estimated_flow = observed_density*(vb + (vf-vb)/(1+np.exp((observed_density-kc)/theta)))
        return estimated_speed, estimated_flow

    def Wang_5PL(self, beta, *args):
        vf, vb, kc, theta1, theta2 = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vb + (vf-vb)/np.power(1+np.exp((observed_density-kc)/theta1), theta2)
        estimated_flow = observed_density*(vb + (vf-vb)/np.power(1+np.exp((observed_density-kc)/theta1), theta2))
        return estimated_speed, estimated_flow

    def Ni(self, beta, *args):
        vf, gamma, tao, l = beta
        observed_flow, observed_density, observed_speed = args
        estimated_density = 1/((gamma*np.power(observed_speed,2)+tao*observed_speed+l)*(1-np.log(1-observed_speed/vf)))
        estimated_flow = observed_speed/((gamma*np.power(observed_speed,2)+tao*observed_speed+l)*(1-np.log(1-observed_speed/vf)))
        return estimated_density, estimated_flow

    def S3_joint_estimation(self, beta, *args):
        vf, kc, foc = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf/np.power(1 + np.power((observed_density/kc), foc), 2/foc)
        estimated_flow = observed_density*vf/np.power(1 + np.power((observed_density/kc), foc), 2/foc)
        return estimated_speed, estimated_flow

    def Greenshields_joint_estimation(self, beta, *args):
        vf, k_jam = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1 - observed_density/k_jam)
        estimated_flow = observed_density*vf*(1 - observed_density/k_jam)
        return estimated_speed, estimated_flow

    def Greenberg_joint_estimation(self, beta, *args):
        vc, k_jam = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vc*np.log(k_jam/observed_density)
        estimated_flow = observed_density*vc*np.log(k_jam/observed_density)
        return estimated_speed, estimated_flow

    def Underwood_joint_estimation(self, beta, *args):
        vf, kc = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*np.exp(-1*observed_density/kc)
        estimated_flow = observed_density*vf*np.exp(-1*observed_density/kc)
        return estimated_speed, estimated_flow

    def NF_joint_estimation(self, beta, *args):
        vf, k_jam, lambda_NF = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1-np.exp(-1*lambda_NF/vf*(1/observed_density - 1/k_jam)))
        estimated_flow = observed_density*vf*(1-np.exp(-1*lambda_NF/vf*(1/observed_density - 1/k_jam)))
        return estimated_speed, estimated_flow

    def GHR_M1_joint_estimation(self, beta, *args):
        vf, kc = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*np.exp(-0.5*np.power(observed_density/kc, 2))
        estimated_flow = observed_density*vf*np.exp(-0.5*np.power(observed_density/kc, 2))
        return estimated_speed, estimated_flow

    def GHR_M2_joint_estimation(self, beta, *args):
        vf, k_jam, m = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1 - np.power(observed_density/k_jam, m))
        estimated_flow = observed_density*vf*(1 - np.power(observed_density/k_jam, m))
        return estimated_speed, estimated_flow

    def GHR_M3_joint_estimation(self, beta, *args):
        vf, k_jam, m = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*np.power(1 - observed_density/k_jam, m)
        estimated_flow = observed_density*vf*np.power(1 - observed_density/k_jam, m)
        return estimated_speed, estimated_flow

    def KK_joint_estimation(self, beta, *args):
        vf, kc, c1, c2, c3 = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1/(1 + np.exp((observed_density/kc - c1)/c2)) - c3)
        estimated_flow = observed_density*vf*(1/(1 + np.exp((observed_density/kc - c1)/c2)) - c3)
        return estimated_speed, estimated_flow

    def Jayakrishnan_joint_estimation(self, beta, *args):
        vf, v_jam, k_jam, m = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = v_jam + (vf - v_jam)*np.power(1 - observed_density/k_jam, m)
        estimated_flow = observed_density*(v_jam + (vf - v_jam)*np.power(1 - observed_density/k_jam, m))
        return estimated_speed, estimated_flow

    def Van_Aerde_joint_estimation(self, beta, *args):
        vf, vc, k_jam, max_flow = beta
        c1 = vf/(k_jam*np.power(vc, 2))*(2*vc - vf)
        c2 = vf/(k_jam*np.power(vc, 2))*np.power(vf - vc, 2)
        c3 = 1/max_flow - vf/(k_jam*np.power(vc, 2))
        observed_flow, observed_density, observed_speed = args
        estimated_density = 1/(c1 + c2/(vf-observed_speed) + c3*observed_speed)
        estimated_flow = observed_speed/(c1 + c2/(vf-observed_speed) + c3*observed_speed)
        return estimated_density, estimated_flow

    def MacNicholas_joint_estimation(self, beta, *args):
        vf, k_jam, m, c = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(np.power(k_jam, m) - np.power(observed_density, m))/(np.power(k_jam, m) + c*np.power(observed_density, m))
        estimated_flow = observed_density*vf*(np.power(k_jam, m) - np.power(observed_density, m))/(np.power(k_jam, m) + c*np.power(observed_density, m))
        return estimated_speed, estimated_flow

    def Wang_3PL_joint_estimation(self, beta, *args):
        vf, kc, theta = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf/(1+np.exp((observed_density-kc)/theta))
        estimated_flow = observed_density*vf/(1+np.exp((observed_density-kc)/theta))
        return estimated_speed, estimated_flow

    def Wang_4PL_joint_estimation(self, beta, *args):
        vf, vb, kc, theta = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vb + (vf-vb)/(1+np.exp((observed_density-kc)/theta))
        estimated_flow = observed_density*(vb + (vf-vb)/(1+np.exp((observed_density-kc)/theta)))
        return estimated_speed, estimated_flow

    def Wang_5PL_joint_estimation(self, beta, *args):
        vf, vb, kc, theta1, theta2 = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vb + (vf-vb)/np.power(1+np.exp((observed_density-kc)/theta1), theta2)
        estimated_flow = observed_density*(vb + (vf-vb)/np.power(1+np.exp((observed_density-kc)/theta1), theta2))
        return estimated_speed, estimated_flow

    def Ni_joint_estimation(self, beta, *args):
        vf, gamma, tao, l = beta
        observed_flow, observed_density, observed_speed = args
        estimated_density = 1/((gamma*np.power(observed_speed,2)+tao*observed_speed+l)*(1-np.log(1-observed_speed/vf)))
        estimated_flow = observed_speed/((gamma*np.power(observed_speed,2)+tao*observed_speed+l)*(1-np.log(1-observed_speed/vf)))
        return estimated_density, estimated_flow


class theoretical_value():

    def S3(self, beta, density):
        vf, kc, foc = beta
        theoretical_speed = vf/np.power(1 + np.power((density/kc), foc), 2/foc)
        theoretical_flow = density*vf/np.power(1 + np.power((density/kc), foc), 2/foc)
        return theoretical_speed, theoretical_flow

    def Greenshields(self, beta, density):
        vf, k_jam = beta
        theoretical_speed = vf*(1 - density/k_jam)
        theoretical_flow = density*vf*(1 - density/k_jam)
        return theoretical_speed, theoretical_flow

    def Greenberg(self, beta, density):
        vc, k_jam = beta
        theoretical_speed = vc*np.log(k_jam/density)
        theoretical_flow = density*vc*np.log(k_jam/density)
        return theoretical_speed, theoretical_flow

    def Underwood(self, beta, density):
        vf, kc = beta
        theoretical_speed = vf*np.exp(-1*density/kc)
        theoretical_flow = density*vf*np.exp(-1*density/kc)
        return theoretical_speed, theoretical_flow

    def NF(self, beta, density):
        vf, k_jam, lambda_NF = beta
        theoretical_speed = vf*(1-np.exp(-1*lambda_NF/vf*(1/density - 1/k_jam)))
        theoretical_flow = density*vf*(1-np.exp(-1*lambda_NF/vf*(1/density - 1/k_jam)))
        return theoretical_speed, theoretical_flow

    def GHR_M1(self, beta, density):
        vf, kc = beta
        theoretical_speed = vf*np.exp(-0.5*np.power(density/kc, 2))
        theoretical_flow = density*vf*np.exp(-0.5*np.power(density/kc, 2))
        return theoretical_speed, theoretical_flow

    def GHR_M2(self, beta, density):
        vf, k_jam, m = beta
        theoretical_speed = vf*(1 - np.power(density/k_jam, m))
        theoretical_flow = density*vf*(1 - np.power(density/k_jam, m))
        return theoretical_speed, theoretical_flow

    def GHR_M3(self, beta, density):
        vf, k_jam, m = beta
        theoretical_speed = vf*np.sign(1-density/k_jam)*np.abs(1-density/k_jam)**m
        theoretical_flow = density*vf*np.sign(1-density/k_jam)*np.abs(1-density/k_jam)**m
        return theoretical_speed, theoretical_flow

    def KK(self, beta, density):
        vf, kc, c1, c2, c3 = beta
        theoretical_speed = vf*(1/(1 + np.exp((density/kc - c1)/c2)) - c3)
        theoretical_flow = density*vf*(1/(1 + np.exp((density/kc - c1)/c2)) - c3)
        return theoretical_speed, theoretical_flow

    def Jayakrishnan(self, beta, density):
        vf, v_jam, k_jam, m = beta
        theoretical_speed = v_jam + (vf - v_jam)*np.sign(1-density/k_jam)*np.abs(1-density/k_jam)**m
        theoretical_flow = density*(v_jam + (vf - v_jam)*(np.sign(1-density/k_jam)*np.abs(1-density/k_jam)**m))
        return theoretical_speed, theoretical_flow

    def Van_Aerde(self, beta, speed):
        vf, vc, k_jam, max_flow = beta
        c1 = vf/(k_jam*np.power(vc, 2))*(2*vc - vf)
        c2 = vf/(k_jam*np.power(vc, 2))*np.power(vf - vc, 2)
        c3 = 1/max_flow - vf/(k_jam*np.power(vc, 2))
        theoretical_density = 1/(c1 + c2/(vf-speed) + c3*speed)
        theoretical_flow = speed/(c1 + c2/(vf-speed) + c3*speed)
        return theoretical_density, theoretical_flow

    def MacNicholas(self, beta, density):
        vf, k_jam, m, c = beta
        theoretical_speed = vf*(np.power(k_jam, m) - np.power(density, m))/(np.power(k_jam, m) + c*np.power(density, m))
        theoretical_flow = density*vf*(np.power(k_jam, m) - np.power(density, m))/(np.power(k_jam, m) + c*np.power(density, m))
        return theoretical_speed, theoretical_flow

    def Wang_3PL(self, beta, density):
        vf, kc, theta = beta
        theoretical_speed = vf/(1+np.exp((density-kc)/theta))
        theoretical_flow = density*vf/(1+np.exp((density-kc)/theta))
        return theoretical_speed, theoretical_flow

    def Wang_4PL(self, beta, density):
        vf, vb, kc, theta = beta
        theoretical_speed = vb + (vf-vb)/(1+np.exp((density-kc)/theta))
        theoretical_flow = density*(vb + (vf-vb)/(1+np.exp((density-kc)/theta)))
        return theoretical_speed, theoretical_flow

    def Wang_5PL(self, beta, density):
        vf, vb, kc, theta1, theta2 = beta
        theoretical_speed = vb + (vf-vb)/np.power(1+np.exp((density-kc)/theta1), theta2)
        theoretical_flow = density*(vb + (vf-vb)/np.power(1+np.exp((density-kc)/theta1), theta2))
        return theoretical_speed, theoretical_flow

    def Ni(self, beta, speed):
        vf, gamma, tao, l = beta
        theoretical_density = 1/((gamma*np.power(speed,2)+tao*speed+l)*(1-np.log(1-speed/vf)))
        theoretical_flow = speed/((gamma*np.power(speed,2)+tao*speed+l)*(1-np.log(1-speed/vf)))
        return theoretical_density, theoretical_flow

    def S3_joint_estimation(self, beta, density):
        vf, kc, foc = beta
        theoretical_speed = vf/np.power(1 + np.power((density/kc), foc), 2/foc)
        theoretical_flow = density*vf/np.power(1 + np.power((density/kc), foc), 2/foc)
        return theoretical_speed, theoretical_flow

    def Greenshields_joint_estimation(self, beta, density):
        vf, k_jam = beta
        theoretical_speed = vf*(1 - density/k_jam)
        theoretical_flow = density*vf*(1 - density/k_jam)
        return theoretical_speed, theoretical_flow

    def Greenberg_joint_estimation(self, beta, density):
        vc, k_jam = beta
        theoretical_speed = vc*np.log(k_jam/density)
        theoretical_flow = density*vc*np.log(k_jam/density)
        return theoretical_speed, theoretical_flow

    def Underwood_joint_estimation(self, beta, density):
        vf, kc = beta
        theoretical_speed = vf*np.exp(-1*density/kc)
        theoretical_flow = density*vf*np.exp(-1*density/kc)
        return theoretical_speed, theoretical_flow

    def NF_joint_estimation(self, beta, density):
        vf, k_jam, lambda_NF = beta
        theoretical_speed = vf*(1-np.exp(-1*lambda_NF/vf*(1/density - 1/k_jam)))
        theoretical_flow = density*vf*(1-np.exp(-1*lambda_NF/vf*(1/density - 1/k_jam)))
        return theoretical_speed, theoretical_flow

    def GHR_M1_joint_estimation(self, beta, density):
        vf, kc = beta
        theoretical_speed = vf*np.exp(-0.5*np.power(density/kc, 2))
        theoretical_flow = density*vf*np.exp(-0.5*np.power(density/kc, 2))
        return theoretical_speed, theoretical_flow

    def GHR_M2_joint_estimation(self, beta, density):
        vf, k_jam, m = beta
        theoretical_speed = vf*(1 - np.power(density/k_jam, m))
        theoretical_flow = density*vf*(1 - np.power(density/k_jam, m))
        return theoretical_speed, theoretical_flow

    def GHR_M3_joint_estimation(self, beta, density):
        vf, k_jam, m = beta
        theoretical_speed = vf*np.sign(1-density/k_jam)*np.abs(1-density/k_jam)**m
        theoretical_flow = density*vf*np.sign(1-density/k_jam)*np.abs(1-density/k_jam)**m
        return theoretical_speed, theoretical_flow

    def KK_joint_estimation(self, beta, density):
        vf, kc, c1, c2, c3 = beta
        theoretical_speed = vf*(1/(1 + np.exp((density/kc - c1)/c2)) - c3)
        theoretical_flow = density*vf*(1/(1 + np.exp((density/kc - c1)/c2)) - c3)
        return theoretical_speed, theoretical_flow

    def Jayakrishnan_joint_estimation(self, beta, density):
        vf, v_jam, k_jam, m = beta
        theoretical_speed = v_jam + (vf - v_jam)*np.sign(1-density/k_jam)*np.abs(1-density/k_jam)**m
        theoretical_flow = density*(v_jam + (vf - v_jam)*(np.sign(1-density/k_jam)*np.abs(1-density/k_jam)**m))
        return theoretical_speed, theoretical_flow

    def Van_Aerde_joint_estimation(self, beta, speed):
        vf, vc, k_jam, max_flow = beta
        c1 = vf/(k_jam*np.power(vc, 2))*(2*vc - vf)
        c2 = vf/(k_jam*np.power(vc, 2))*np.power(vf - vc, 2)
        c3 = 1/max_flow - vf/(k_jam*np.power(vc, 2))
        theoretical_density = 1/(c1 + c2/(vf-speed) + c3*speed)
        theoretical_flow = speed/(c1 + c2/(vf-speed) + c3*speed)
        return theoretical_density, theoretical_flow

    def MacNicholas_joint_estimation(self, beta, density):
        vf, k_jam, m, c = beta
        theoretical_speed = vf*(np.power(k_jam, m) - np.power(density, m))/(np.power(k_jam, m) + c*np.power(density, m))
        theoretical_flow = density*vf*(np.power(k_jam, m) - np.power(density, m))/(np.power(k_jam, m) + c*np.power(density, m))
        return theoretical_speed, theoretical_flow

    def Wang_3PL_joint_estimation(self, beta, density):
        vf, kc, theta = beta
        theoretical_speed = vf/(1+np.exp((density-kc)/theta))
        theoretical_flow = density*vf/(1+np.exp((density-kc)/theta))
        return theoretical_speed, theoretical_flow

    def Wang_4PL_joint_estimation(self, beta, density):
        vf, vb, kc, theta = beta
        theoretical_speed = vb + (vf-vb)/(1+np.exp((density-kc)/theta))
        theoretical_flow = density*(vb + (vf-vb)/(1+np.exp((density-kc)/theta)))
        return theoretical_speed, theoretical_flow

    def Wang_5PL_joint_estimation(self, beta, density):
        vf, vb, kc, theta1, theta2 = beta
        theoretical_speed = vb + (vf-vb)/np.power(1+np.exp((density-kc)/theta1), theta2)
        theoretical_flow = density*(vb + (vf-vb)/np.power(1+np.exp((density-kc)/theta1), theta2))
        return theoretical_speed, theoretical_flow

    def Ni_joint_estimation(self, beta, speed):
        vf, gamma, tao, l = beta
        theoretical_density = 1/((gamma*np.power(speed,2)+tao*speed+l)*(1-np.log(1-speed/vf)))
        theoretical_flow = speed/((gamma*np.power(speed,2)+tao*speed+l)*(1-np.log(1-speed/vf)))
        return theoretical_density, theoretical_flow
