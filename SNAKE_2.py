import numpy as np
from math import cos, sin, pi, sqrt, atan2, acos
import math
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy import integrate,optimize
from mpl_toolkits.mplot3d import Axes3D
from Surface_2 import surface
import copy
class snake(object):

    def __init__(self, nb=6, P=1.,P_head = .107, ro_seg=0.,r_head=0.42, head_pos=np.array([]), head_config=np.array([]), neck_config=np.array([]),
                 joint_config=np.array([]), antirouli_config=np.array([]),
                 list_of_DH_matrices=None, list_of_DH_matrices_in_R0=None, list_of_DH_matrices_joint=None,
                 list_of_DH_matrices_roulie=None, list_of_DH_matrices_head=None,
                 head_quaternion=None, section=None):
        '''

        :param nb:
        :param P:
        :param ro_seg:
        :param a:
        :param r:
        :param head_pos:
        :param head_config:
        :param joint_config:
        :param antirouli_config:
        :param list_of_DH_matrices:
        :param list_of_DH_matrices_in_R0:
        :param list_of_DH_matrices_joint:
        :param list_of_DH_matrices_roulie:
        :param list_of_DH_matrices_head:
        :param head_quaternion:
        :param section:
        '''
        self.nb = nb
        self.P = P
        self.P_head = P_head
        self.P_neck = 0.10575
        self.P_roll = 0.09
        self.ro_seg = ro_seg
        #self.a = a
        #self.r = r
        self.r_head = r_head
        self.surf = surface(1)
        self.m_seg =0.3941#0.488876#self.ro_seg*(0.032+0.013+0.013+0.023)*pow(self.r_head,2)*pi+self.ro_seg*(self.P-(0.032+0.013+0.013+0.023))*self.surf.section
        self.m_last_seg = 0.3941#0.4
        #self.m_last_seg = self.ro_seg*(0.032+0.013+0.013)*pow(self.r_head,2)*pi + self.ro_seg*(self.P-(0.032+0.013+0.013+0.023))*self.surf.section
        self.m_roll = 0.113
        self.m_neck = 0.147#0.313 #self.ro_seg*self.P_head*pow(self.r_head,2)*pi
        self.m_head = 0.153# 0.247 #self.ro_seg*self.P_head*pow(self.r_head,2)*pi
        self.M = self.m_roll * (self.nb) + self.m_seg * (self.nb-1)+self.m_head+self.m_neck+self.m_last_seg
        #self.L = self.nb * self.P - 0.023 + self.P_neck+self.P_head
        self.head_pos = head_pos
        self.head_config = head_config
        self.neck_config = neck_config
        self.joint_config = joint_config
        self.antirouli_config = antirouli_config
        self.list_of_DH_matrices = list_of_DH_matrices
        self.list_of_DH_matrices_in_R0 = list_of_DH_matrices_in_R0
        self.list_of_DH_matrices_joint = list_of_DH_matrices_joint
        self.list_of_DH_matrices_roulie = list_of_DH_matrices_roulie
        self.list_of_DH_matrices_head = list_of_DH_matrices_head
        self.DH_matrix_neck = None
        self.DH_matrix_neck_in_R0 = None
        self.array_of_wrench = None # an array of dimension 3*(self.nb+1) which save every wrench in R0 of segments
        self.head_quaternion = head_quaternion
        self.list_of_DH_matrices_joint_in_R0 = None
        self.list_of_DH_matrices_roulie_in_R0 = None
        self.section = self.surf.section
        self.volume = (self.section- self.r_head**2*pi)*(self.P - 0.036-0.045) *self.nb+ self.r_head**2*pi*(self.P_head+self.P_neck+self.P*6-0.023)
        #self.volum = self.section * self.L
        self.centre_segment = np.array([0.111154, 0, -0.00279, 1])#np.array([0.111154, 0, -0.00279, 1])
        self.centre_last_seg = np.array([0.111154, 0, -0.00279, 1])#np.array([0.09, 0, -0.00279, 1])
        self.centre_roll = np.array([0., 0., -0.00279, 1])#np.array([0.,0.,-0.005937,1])
        self.centre_head =np.array([0.06224, 0., 0., 1]) #np.array([0.0468, 0., 0., 1])
        self.centre_neck = np.array([0.086, 0., 0., 1])#np.array([0.0552, 0., 0.0019, 1])
        self.g = 9.8
    def creation_a_traj(self,T=10,A=0.1,B=1):
        time = np.linspace(0., T, 100)
        from scipy.optimize import fsolve
        omega = 2*pi/B
        angles = np.zeros((time.shape[0],5))
        for i in range(time.shape[0]):
            x=0
            t = 2*pi/T*time[i]
            y = A*sin(t+0)
            g = np.eye(3)
            X = [x];Y = [y]
            angle_r0 = 0.
            for j in range(5):
                #r = fsolve(self.func, [x, y],args=(t,x,y,omega,A))
                #x_,y_ = r[0],r[1]
                x_,y_ = self.dichotomie_2(x, 1.5, t, x, y, omega, A)
                if x_<x:
                    print('wtf')
                if j == 0:
                    angle = atan2(y_-y,x_-x)
                    angles[i,j] = angle
                    g = np.array([[cos(angle),sin(angle),-cos(angle)*x_-sin(angle)*y_],[-sin(angle),cos(angle), sin(angle)*x_-cos(angle)*y_],[0,0,1]])
                    angle_r0 +=angle
                else:
                    xy_local_rep = np.dot(g,np.array([x_,y_,1]))
                    x_local,y_local = xy_local_rep[0],xy_local_rep[1]
                    angle = atan2(y_local,x_local)
                    angles[i, j] = angle
                    angle_r0+=angle
                    g = np.array([[cos(angle_r0), sin(angle_r0), -cos(angle_r0) * x_ - sin(angle_r0) * y_],
                                  [-sin(angle_r0), cos(angle_r0), sin(angle_r0) * x_ - cos(angle_r0) * y_], [0, 0, 1]])
                X.append(x_);Y.append(y_)
                x = x_; y = y_
        '''
        xx = np.linspace(0,1,100)
        yy = np.array([])
        for k in range(xx.shape[0]):
                yy=np.append(yy,A*sin(t+omega*xx[k]))
        plt.plot(xx,yy)
        plt.plot(X,Y)
        plt.axis('equal')
        '''
        return angles
    def dichotomie_2(self,a,b,t,x0,y0,omega,A):
        f = lambda x:(x-x0)**2+(A*sin(t+omega*x)-y0)**2-self.P**2
        m=0
        while abs(b-a)>1E-5:
            m = (a + b) / 2
            if f(m) * f(a) < 0:
               b = m
            else:
               a = m
        y = A*sin(t+omega*m)
        return m,y

    def func(self,i,t,x0,y0,omega,A):
        a,b=i[0],i[1]
        return [b-A*sin(omega*a+t), (a-x0)**2+(b-y0)**2-self.P**2]

    def find_COM(self):
        if self.list_of_DH_matrices_roulie_in_R0.shape[1] != self.nb * 4:
            AssertionError('the number of segments is wrong, please check it')
        '''
        P = self.m_head * self.centre_head
        head_g_0 = np.eye(4)
        R_t = np.transpose(self.list_of_DH_matrices_head[:3, :3])
        head_g_0[:3, :3] = R_t
        head_g_0[:3, 3] = - np.dot(R_t, self.list_of_DH_matrices_head[:3, 3])
        head_g_neck = np.dot(head_g_0, self.DH_matrix_neck_in_R0)
        P += self.m_neck * np.dot(head_g_neck, self.centre_neck)
        for i in range(self.nb):
            head_g_roulie_i = np.dot(head_g_0, self.list_of_DH_matrices_roulie_in_R0[:, i * 4:i * 4 + 4])
            if i!=self.nb-1:
                P += self.m_seg * np.dot(head_g_roulie_i, self.centre_segment)
            else:
                P += self.m_last_seg * np.dot(head_g_roulie_i, self.centre_segment)

        P = P / (self.M)
        head_g_com = np.eye(4)
        head_g_com[0:3, 3] = P[:3]
        com_g_head = np.eye(4)
        com_g_head[0:3, 3] = -P[:3]
        return head_g_com, com_g_head
        '''
        P = self.m_head * np.dot(self.list_of_DH_matrices_head, self.centre_head)
        P += self.m_neck * np.dot(self.DH_matrix_neck_in_R0, self.centre_neck)
        for i in range(self.nb):
            if i != self.nb - 1:
               P += self.m_seg * np.dot(self.list_of_DH_matrices_joint_in_R0[:, i * 4:i * 4 + 4],
                                                      self.centre_segment)
               P+= self.m_roll * np.dot(self.list_of_DH_matrices_roulie_in_R0[:, i * 4:i * 4 + 4],
                                                       self.centre_roll)
            else:
                P += self.m_last_seg * np.dot(self.list_of_DH_matrices_joint_in_R0[:, i * 4:i * 4 + 4],
                                                           self.centre_last_seg)
                P += self.m_roll * np.dot(self.list_of_DH_matrices_roulie_in_R0[:, i * 4:i * 4 + 4],
                                                       self.centre_roll)
        g0 = self.inverse_configuration(self.list_of_DH_matrices_head)
        P = np.dot(g0,P)/self.M
        return P
    def gravity_ponential_energy(self):
        P = self.m_head * np.dot(self.list_of_DH_matrices_head,self.centre_head)
        P += self.m_neck * np.dot(self.DH_matrix_neck_in_R0, self.centre_neck)
        for i in range(self.nb):
            if i != self.nb-1:
                P += self.m_seg * np.dot(self.list_of_DH_matrices_joint_in_R0[:, i * 4:i * 4 + 4], self.centre_segment)
                P += self.m_roll * np.dot(self.list_of_DH_matrices_roulie_in_R0[:, i * 4:i * 4 + 4], self.centre_roll)
            else:
                P += self.m_last_seg * np.dot(self.list_of_DH_matrices_joint_in_R0[:, i * 4:i * 4 + 4], self.centre_last_seg)
                P += self.m_roll * np.dot(self.list_of_DH_matrices_roulie_in_R0[:, i * 4:i * 4 + 4], self.centre_roll)
        return P[2]

    def gravity_wrench(self):
        if self.list_of_DH_matrices_joint_in_R0.shape[1] != self.nb * 4:
            AssertionError('the number of segments is wrong, pleace check it')
        gravity_wrench = np.empty((6))
        e3 = np.array([0, 0, 1])
        gravity_wrench[:3] = -self.M * self.g * e3
        gravity_torque = self.m_head * np.dot(self.list_of_DH_matrices_head, self.centre_head)
        gravity_torque += self.m_neck * np.dot(self.DH_matrix_neck_in_R0, self.centre_neck)
        for i in range(self.nb):
            if i !=self.nb-1:
                gravity_torque += self.m_seg * np.dot(self.list_of_DH_matrices_joint_in_R0[:, i * 4:i * 4 + 4],
                                                  self.centre_segment)
                gravity_torque += self.m_roll * np.dot(self.list_of_DH_matrices_roulie_in_R0[:, i * 4:i * 4 + 4],
                                                      self.centre_roll)
            else:
                gravity_torque += self.m_last_seg * np.dot(self.list_of_DH_matrices_joint_in_R0[:, i * 4:i * 4 + 4],
                                                      self.centre_last_seg)
                gravity_torque += self.m_roll * np.dot(self.list_of_DH_matrices_roulie_in_R0[:, i * 4:i * 4 + 4],
                                                           self.centre_roll)
        gravity_torque = - self.g * np.cross(gravity_torque[:3], e3)
        gravity_wrench[3:] = gravity_torque
        return gravity_wrench

    def buoyancy_potential_energy(self):
        '''
        :param nb_step1: number of steps for the integration of the volume of segment i
        :return:
        '''
        g_head = self.list_of_DH_matrices_head
        g_neck = self.DH_matrix_neck_in_R0
        g_segment = self.list_of_DH_matrices_roulie_in_R0
        U = self.potential_energy_i(g_head, head=True,neck=False,last_seg=False)
        U += self.potential_energy_i(g_neck, head=False,neck=True,last_seg=False)
        for j in range(0, self.nb):
            g_segment_i = g_segment[:, j * 4:j * 4 + 4]
            if j!=self.nb-1:
                temp = self.potential_energy_i(g_segment_i, False, False, False)
            else:
                temp = self.potential_energy_i(g_segment_i, False, False, True)
            U += temp
        return U

    def potential_energy_i(self, g, head, neck, last_seg):

        if head is False and neck is False:
            temp = np.identity(4)
            temp[0, 3] = -self.P_roll
            g = np.dot(g, temp)

        def PE(x,config_0,circular):
            temp = np.identity(4)
            temp[0, 3] = x
            config = np.dot(config_0, temp)
            Sim, SimQb_2d = self.Immersed_surface_theoric(config,circular)
            SimQb = np.array([0, SimQb_2d[0], SimQb_2d[1], Sim])
            SimQb_in_R0 = np.dot(config, SimQb)
            U = 1000 * SimQb_in_R0[2]
            return U

        U = 0.
        if not last_seg:
            position = [0., 0.045, self.P - 0.036, self.P]
        else:
            position = [0., 0.045, self.P - 0.036, self.P - 0.023]
        if head or neck:
            s1, SimQb_2d = self.Immersed_surface_theoric(g, True)  ##g is the configuration of x = 0
            g2 = np.identity(4)
            if head:
                P = self.P_head
            else:
                P = self.P_neck
            g2[0, 3] = P
            g2 = np.dot(g, g2)
            s2, SimQb_2d2 = self.Immersed_surface_theoric(g2, True)
            if s1 == 0. and s2 == 0.:  # not immersed
                U = 0
            else:
                if s1 > 0. and s2 == 0.:
                    l = self.Dichotomy(g, s1, s2, 0, P, circular=True)
                    interval = [0, l]
                elif s1 == 0. and s2 < 0.:
                    l = self.Dichotomy(g, s1, s2, 0, P, circular=True)
                    interval = [l, P]
                else:
                    interval = [0, P]

                U, e = integrate.quad(PE, interval[0], interval[1], args=(g, True))

        else:
            for i in range(3):
                g1 = np.identity(4)
                g1[0, 3] = position[i]
                g1 = np.dot(g, g1)
                if i != 1:
                    circular = True
                else:
                    circular = False
                s1, SimQb_2d = self.Immersed_surface_theoric(g1, circular)
                g2 = np.identity(4)
                g2[0, 3] = position[i + 1]
                g2 = np.dot(g, g2)
                s2, SimQb_2d2 = self.Immersed_surface_theoric(g2, circular)
                if s1 == 0. and s2 == 0.:  # not immersed
                    U += 0
                else:
                    if s1 > 0. and s2 == 0.:
                        l = self.Dichotomy(g, s1, s2, position[i], position[i + 1], circular=circular)
                        interval = [position[i], l]
                    elif s1 == 0. and s2 < 0.:
                        l = self.Dichotomy(g, s1, s2, position[i], position[i + 1], circular=circular)
                        interval = [l, position[i + 1]]
                    else:
                        interval = [position[i], position[i + 1]]
                    U_i, e = integrate.quad(PE, interval[0], interval[1], args=(g, circular))
                    U += U_i
        return U

    def get_list_buoyancy_wrench(self):
        '''
        :param nb_step1: number of sectors of the segment
        :param nb_step2: number of mesh of one sector
        :param n: in whiche moment the brench is wanted
        :param Trapez: Use the Trapez method?
        :param Sp_rule: Use the Simpson rule?
        :param Gauss: Use the Chebyshev–Gauss quadrature?
        :return: the buoyancy_wrench of n moment
        '''

        def buoyancy_force(x, config_0, circular):
            F = np.zeros((3,x.shape[0]))
            for i in range(x.shape[0]):
                  temp = np.identity(4)
                  temp[0, 3] = x[i]
                  config = np.dot(config_0, temp)
                  Sim, SimQb_2d = self.Immersed_surface_theoric(config, circular)
                  if Sim != 0:
                      SimQb = np.array([0, SimQb_2d[0], SimQb_2d[1], Sim])
                      SimQb_in_R0 = np.dot(config, SimQb)
                      SimQb_in_R0 = SimQb_in_R0[:3]
                      Fi = np.array([Sim,SimQb_in_R0[0],SimQb_in_R0[1]])
                  else:
                      Fi=np.zeros(3)
                  F[:,i]=Fi
            return F
        def integrate_for_one_seg(gi,a,b,circ):
              g1 = np.identity(4)
              g1[0, 3] = a
              g1 = np.dot(gi, g1)
              s1, SimQb_2d = self.Immersed_surface_theoric(g1, circ)
              g2 = np.identity(4)
              g2[0, 3] = b
              g2 = np.dot(gi, g2)
              s2, SimQb_2d2 = self.Immersed_surface_theoric(g2, circ)
              if s1 != 0. or s2 != 0.:
                  e3 = np.array([0, 0, 1])
                  if s1 > 0. and s2 == 0.:
                      l = self.Dichotomy(gi, s1, s2, a,b, circular=circ)
                      interval = [a, l]
                  elif s1 == 0. and s2 < 0.:
                      l = self.Dichotomy(gi, s1, s2, a, b, circular=circ)
                      interval = [l, b]
                  else:
                      interval = [a, b]
                  F, e = integrate.fixed_quad(buoyancy_force, interval[0], interval[1], n=15, args=(gi, circ))
                  Mi = np.array([F[1], F[2], 0.])
                  Mi = np.cross(Mi, e3)
                  buoyancy_wrench = 9800 * np.array([0, 0, F[0], Mi[0], Mi[1], 0])
              else:
                  buoyancy_wrench = np.zeros((6))
              return buoyancy_wrench
        Fb = integrate_for_one_seg(self.list_of_DH_matrices_head,0,self.P_head,True)
        Fb += integrate_for_one_seg(self.DH_matrix_neck_in_R0,0,self.P_neck,True)
        for i in range(self.nb):
            l = self.P - 0.036-0.045
            Fb += integrate_for_one_seg(self.list_of_DH_matrices_roulie_in_R0[:,i*4:i*4+4],-l/2,l/2,False)
            if i == self.nb-1:
                Fb += integrate_for_one_seg(self.list_of_DH_matrices_joint_in_R0[:,i*4:i*4+4],0.,0.045,True)+integrate_for_one_seg(self.list_of_DH_matrices_joint_in_R0[:,i*4:i*4+4],self.P - 0.036, self.P - 0.023,True)
            else:
                Fb += integrate_for_one_seg(self.list_of_DH_matrices_joint_in_R0[:,i*4:i*4+4],0.,0.045,True)+integrate_for_one_seg(self.list_of_DH_matrices_joint_in_R0[:,i*4:i*4+4],self.P - 0.036, self.P,True)

        g_head = self.list_of_DH_matrices_head
        g_neck = self.DH_matrix_neck_in_R0
        g_segment = self.list_of_DH_matrices_roulie_in_R0
        '''
        W = self.buoyancy_wrench_i_segment2(g_head, head=True,neck = False, last_seg = False)
        W += self.buoyancy_wrench_i_segment2(g_neck, head=False,neck=True, last_seg = False)
        for j in range(0, self.nb):
            g_segment_i = g_segment[:, j * 4:j * 4 + 4]
            if j != self.nb-1:
                temp = self.buoyancy_wrench_i_segment2(g=g_segment_i,head=False, neck = False, last_seg = False)
            else:
                temp = self.buoyancy_wrench_i_segment2(g=g_segment_i, head=False, neck = False, last_seg=True)
            W += temp
        print('?',W.reshape((6))-Fb,W.reshape((6)))
        '''
        return Fb.reshape((6,1))

    def buoyancy_wrench_i_segment2(self, g, head=False, neck=False, last_seg=False):
        '''
        :param g: the configuration of segment
        :param nb: number of gauss nodes
        :param head: if the segment is the head?
        :return:
        '''
        if head is False and neck is False:
            temp = np.identity(4)
            temp[0, 3] = -self.P_roll
            g = np.dot(g, temp)

        def buoyancy_force(x,config_0, circular):
            temp = np.identity(4)
            temp[0, 3] = x
            config = np.dot(config_0, temp)
            Sim, SimQb_2d = self.Immersed_surface_theoric(config,circular)
            if Sim != 0:
                Fi =Sim
            else:
                Fi=0
            return Fi
        def buoyancy_torque_x(x,config_0,circular):
            temp = np.identity(4)
            temp[0, 3] = x
            config = np.dot(config_0, temp)
            #e3 = np.array([0, 0, 1])
            Sim, SimQb_2d = self.Immersed_surface_theoric(config,circular)
            if Sim != 0:
                SimQb = np.array([0, SimQb_2d[0], SimQb_2d[1], Sim])
                SimQb_in_R0 = np.dot(config, SimQb)
                SimQb_in_R0 = SimQb_in_R0[:3]
                Mi = SimQb_in_R0[0]
            else:
                Mi=0
            return Mi
        def buoyancy_torque_y(x,config_0,circular):
            temp = np.identity(4)
            temp[0, 3] = x
            config = np.dot(config_0, temp)
            #e3 = np.array([0, 0, 1])
            Sim, SimQb_2d = self.Immersed_surface_theoric(config,circular)
            if Sim != 0:
                SimQb = np.array([0, SimQb_2d[0], SimQb_2d[1], Sim])
                SimQb_in_R0 = np.dot(config, SimQb)
                SimQb_in_R0 = SimQb_in_R0[:3]
                Mi = SimQb_in_R0[1]
            else:
                Mi=0
            return Mi
        buoyancy_wrench = np.zeros((6, 1))
        if not last_seg:
                position = [0, 0.045,self.P-0.036,self.P]
        else:
                position = [0, 0.045, self.P-0.036,self.P-0.023]
        if head is True or neck is True:
                s1, SimQb_2d = self.Immersed_surface_theoric(g,True)  ##g is the configuration of x = 0
                g2 = np.identity(4)
                if head:
                    P = self.P_head
                else:
                    P = self.P_neck
                g2[0, 3] = P
                g2 = np.dot(g, g2)
                s2, SimQb_2d2 = self.Immersed_surface_theoric(g2,True)
                if s1 ==0. and s2 == 0.:  # not immersed
                    buoyancy_wrench = np.zeros((6, 1))
                else:
                    e3 = np.array([0, 0, 1])
                    if s1 > 0. and s2 == 0.:
                        l = self.Dichotomy(g, s1, s2, 0, P,  circular=True)
                        interval = [0, l]
                    elif s1 == 0. and s2 < 0.:
                        l = self.Dichotomy(g, s1, s2, 0, P,  circular=True)
                        interval = [l, P]
                    else:
                        interval = [0, P]

                    Fz, e = integrate.quad(buoyancy_force, interval[0], interval[1],args=(g,True))
                    Mx, e = integrate.quad(buoyancy_torque_x, interval[0], interval[1],args=(g,True))
                    My, e = integrate.quad(buoyancy_torque_y, interval[0], interval[1],args=(g,True))
                    Mi = np.array([Mx, My, 0.])
                    Mi = np.cross(Mi, e3)
                    buoyancy_wrench = 9800 * np.array([[0, 0, Fz, Mi[0], Mi[1], 0]])
                    buoyancy_wrench = np.reshape(buoyancy_wrench, (6, 1))
        else:
            for i in range(3):
                g1 = np.identity(4)
                g1[0, 3] = position[i]
                g1 = np.dot(g, g1)
                if i != 1:
                    circular = True
                else:
                    circular = False
                s1, SimQb_2d = self.Immersed_surface_theoric(g1, circular)
                g2 = np.identity(4)
                g2[0, 3] = position[i+1]
                g2 = np.dot(g, g2)
                s2, SimQb_2d2 = self.Immersed_surface_theoric(g2, circular)
                if s1 == 0. and s2 == 0.:  # not immersed
                    buoyancy_wrench += np.zeros((6, 1))
                else:
                    e3 = np.array([0, 0, 1])
                    if s1 > 0. and s2 == 0.:
                        l = self.Dichotomy(g, s1, s2, position[i], position[i+1], circular=circular)
                        interval = [position[i], l]
                    elif s1 == 0. and s2 < 0.:
                        l = self.Dichotomy(g, s1, s2, position[i], position[i+1], circular=circular)
                        interval = [l, position[i+1]]
                    else:
                        interval = [position[i], position[i+1]]

                    Fz, e = integrate.quad(buoyancy_force, interval[0], interval[1], args=(g, circular))
                    Mx, e = integrate.quad(buoyancy_torque_x, interval[0], interval[1], args=(g, circular))
                    My, e = integrate.quad(buoyancy_torque_y, interval[0], interval[1], args=(g, circular))

                    Mi = np.array([Mx, My, 0.])
                    Mi = np.cross(Mi, e3)
                    buoyancy_wrench_i = 9800 * np.array([[0, 0, Fz, Mi[0], Mi[1], 0]])
                    buoyancy_wrench += np.reshape(buoyancy_wrench_i, (6, 1))
        return buoyancy_wrench

    def R0_wrench_jacobian_matrix_of_body(self, step):
        '''
        :param step_Sp:  nb of steps for the Simpson's method
        :param step:  nb of steps for the partial differential
        :return: the jacobian matrix of wrench in R0 compared with the body configuration
        '''
        grad = np.empty((3, 2*self.nb+1))
        temp1 = copy.copy(self.list_of_DH_matrices_head)
        temp2 = copy.copy(self.list_of_DH_matrices_joint_in_R0);temp22 = copy.copy(self.list_of_DH_matrices_joint)
        temp3 = copy.copy(self.list_of_DH_matrices_roulie_in_R0);temp33 = copy.copy(self.list_of_DH_matrices_roulie)
        temp4 = copy.copy(self.joint_config); temp5 = copy.copy(self.antirouli_config); temp6 = copy.copy(self.neck_config)
        temp7 = copy.copy(self.DH_matrix_neck_in_R0);temp77 = copy.copy(self.DH_matrix_neck)
        #for i in range()
        self.neck_config[1] = self.neck_config[1] + step
        self.get_neck_frame(); self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench = wb + wg.reshape((6, 1))
        new_wrench_forward = new_wrench[2:5, 0]
        self.neck_config[1] = self.neck_config[1] - 2*step
        self.get_neck_frame();
        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench = wb + wg.reshape((6, 1))
        new_wrench_afterward = new_wrench[2:5, 0]
        self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2)
        self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3)
        self.neck_config = copy.copy(temp6); self.DH_matrix_neck_in_R0 = copy.copy(temp7);self.DH_matrix_neck = copy.copy(temp77)
        for j in range(3):
            grad[j, 0] = (new_wrench_forward[j]-new_wrench_afterward[j]) / (2 * step)
        for i in range(self.nb):
            self.joint_config[i] = self.joint_config[i] + step
            self.get_neck_frame()
            self.get_list_joint_frame();self.get_list_roulie_frame(); self.get_list_frame_R02()

            wb = self.get_list_buoyancy_wrench()
            wg = self.gravity_wrench()
            d_W_left = wb + wg.reshape((6, 1))

            #d_W_left = self.delta_W_joint(i, step_Sp)
            self.joint_config[i] = self.joint_config[i] - 2*step
            self.get_neck_frame()
            self.get_list_joint_frame(); self.get_list_roulie_frame(); self.get_list_frame_R02()
            #d_W_right = self.delta_W_joint(i, step_Sp)

            wb = self.get_list_buoyancy_wrench()
            wg = self.gravity_wrench()
            d_W_right= wb + wg.reshape((6, 1))
            self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2);self.list_of_DH_matrices_joint = copy.copy(temp22)
            self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3)
            self.DH_matrix_neck_in_R0 = copy.copy(temp7)
            self.joint_config = copy.copy(temp4)
            d_W = (d_W_left - d_W_right)[2:5, 0]

            for j in range(3):
                grad[j, 1 + 2 * i] = d_W[j]/(2*step)
            self.antirouli_config[i] = self.antirouli_config[i] + step
            self.get_neck_frame(); self.get_list_joint_frame()
            self.get_list_roulie_frame();self.get_list_frame_R02()

            #wb = self.get_list_buoyancy_wrench()
            #wg = self.gravity_wrench()
            #d_W_left = wb + wg.reshape((6, 1))
            d_W_left = self.delta_W_roll(i)
            self.antirouli_config[i] = self.antirouli_config[i]-2*step
            self.get_neck_frame();self.get_list_joint_frame()
            self.get_list_roulie_frame();self.get_list_frame_R02()

            #wb = self.get_list_buoyancy_wrench()
            #wg = self.gravity_wrench()
            #d_W_right = wb + wg.reshape((6, 1))
            d_W_right = self.delta_W_roll(i)
            self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2)
            self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3);self.list_of_DH_matrices_roulie= copy.copy(temp33)
            self.DH_matrix_neck_in_R0 = copy.copy(temp7)
            self.antirouli_config = copy.copy(temp5)
            d_W = (d_W_left - d_W_right)
            for j in range(3):
                grad[j, 1 + 2 * i + 1] = d_W[j] / (2 * step)
        return grad
    def R0_wrench_jacobian_matrix_of_roll(self, step):
        '''
        dW/dq_neck; dW/dq_joint1; dW/dq_roll
        :param step:  nb of steps for the partial differential
        :return: the jacobian matrix of wrench in R0 compared with the body configuration
        '''
        grad = np.empty((3, self.nb+2))
        temp1 = copy.copy(self.list_of_DH_matrices_head)
        temp2 = copy.copy(self.list_of_DH_matrices_joint_in_R0);temp22 = copy.copy(self.list_of_DH_matrices_joint)
        temp3 = copy.copy(self.list_of_DH_matrices_roulie_in_R0);temp33 = copy.copy(self.list_of_DH_matrices_roulie);
        temp4 = copy.copy(self.joint_config); temp5 = copy.copy(self.antirouli_config); temp6 = copy.copy(self.neck_config)
        temp7 = copy.copy(self.DH_matrix_neck_in_R0); temp77 = copy.copy(self.DH_matrix_neck)
        #for i in range()
        self.neck_config[1] = self.neck_config[1] + step
        self.get_neck_frame();  self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench = wb + wg.reshape((6, 1))
        new_wrench_forward = new_wrench[2:5, 0]
        self.neck_config[1] = self.neck_config[1] - 2*step
        self.get_neck_frame();
        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench = wb + wg.reshape((6, 1))
        new_wrench_afterward = new_wrench[2:5, 0]
        self.list_of_DH_matrices_head = copy.copy(temp1)
        self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2)
        self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3)
        self.neck_config = copy.copy(temp6); self.DH_matrix_neck_in_R0 = copy.copy(temp7); self.DH_matrix_neck = copy.copy(temp77)
        for j in range(3):
            grad[j, 0] = (new_wrench_forward[j]-new_wrench_afterward[j]) / (2 * step)
        self.joint_config[0] = self.joint_config[0] + step
        self.get_list_joint_frame();self.get_list_frame_R02()

        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        d_W_left = wb + wg.reshape((6, 1))
        self.joint_config[0] = self.joint_config[0] - 2 * step
        self.get_list_joint_frame();
        self.get_list_frame_R02()
        # d_W_right = self.delta_W_joint(i, step_Sp)

        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        d_W_right = wb + wg.reshape((6, 1))
        self.list_of_DH_matrices_joint = copy.copy(temp22)
        self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2)
        self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3)
        self.DH_matrix_neck_in_R0 = copy.copy(temp7)
        self.joint_config = copy.copy(temp4)
        d_W = (d_W_left - d_W_right)[2:5, 0]

        for j in range(3):
            grad[j, 1] = d_W[j] / (2 * step)
        for i in range(self.nb):
            self.antirouli_config[i] = self.antirouli_config[i] + step
            self.get_list_roulie_frame();self.get_list_frame_R02()
            wb = self.get_list_buoyancy_wrench()
            wg = self.gravity_wrench()
            d_W_left = wb + wg.reshape((6, 1))
            #d_W_left = self.delta_W_roll(i, step_Sp)
            self.antirouli_config[i] = self.antirouli_config[i]-2*step
            self.get_list_roulie_frame();self.get_list_frame_R02()

            wb = self.get_list_buoyancy_wrench()
            wg = self.gravity_wrench()
            d_W_right = wb + wg.reshape((6, 1))
            #d_W_right = self.delta_W_roll(i, step_Sp)
            self.list_of_DH_matrices_roulie = copy.copy(temp33)
            self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3)
            self.antirouli_config = copy.copy(temp5)
            d_W = (d_W_left - d_W_right)[2:5, 0]
            for j in range(3):
                grad[j, 1 + i ] = d_W[j] / (2 * step)
        return grad
    def delta_W_roll(self, i):
        '''
        :param i: number of segment to do the differential
        :param step: nb of steps for the Simpson's method
        :return: the sum of wrench for each segment between i to self.nb

        '''
        e3 = np.array([0, 0, 1])
        g = self.list_of_DH_matrices_roulie_in_R0[:, 4 * i:4 * i + 4]
        if i == self.nb-1:
            last_seg = True; m_seg = self.m_last_seg; center_seg = self.centre_last_seg
        else:
            last_seg = False; m_seg = self.m_seg; center_seg = self.centre_segment
        buoyancy_wrench = self.buoyancy_wrench_i_segment2(g=g, head=False, neck=False, last_seg= last_seg)
        buoyancy_wrench = buoyancy_wrench[2:5, 0]
        gravity_wrench = np.empty((3))
        gravity_wrench[0] = -(self.m_seg+self.m_last_seg) * self.g
        gravity_torque = m_seg * np.dot(self.list_of_DH_matrices_joint_in_R0[:, i * 4:i * 4 + 4],
                                              center_seg)
        gravity_torque += self.m_roll * np.dot(self.list_of_DH_matrices_roulie_in_R0[:, i * 4:i * 4 + 4],
                                               self.centre_roll)
        gravity_torque = - self.g * np.cross(gravity_torque[:3], e3)
        gravity_wrench[1:3] = gravity_torque[:2]
        wrench_diff = buoyancy_wrench + gravity_wrench
        return wrench_diff

    def delta_W_joint(self, i, step):
        '''
        :param i: number of segment to do the differential
        :param step: nb of steps for the Simpson's method
        :return: the sum of wrench for each segment between i to self.nb
        '''
        wrench_diff = np.zeros((3))
        e3 = np.array([0, 0, 1])
        for j in range(i,self.nb):
            g = self.list_of_DH_matrices_roulie_in_R0[:,4*j:4*j+4]
            #g[:3,:3] = np.eye(3)
            buoyancy_wrench = self.buoyancy_wrench_i_segment(g=g, nb_step1=step, Sp_Rule=True,head=False,head_or_neck=False)
            buoyancy_wrench = buoyancy_wrench[2:5, 0]
            gravity_wrench = np.empty((3))
            gravity_wrench[0] = -self.m_seg * self.g
            gravity_torque = self.m_seg * np.dot(self.list_of_DH_matrices_roulie_in_R0[:, j * 4: j * 4 + 4],self.centre_segment)
            gravity_torque = - self.g * np.cross(gravity_torque[:3], e3)
            gravity_wrench[1:3] = gravity_torque[:2]
            '''
            if i == self.nb-1:
                
                print(self.joint_config)
                
                print('#############DH')
                for h in range(self.nb):
                     print(self.list_of_DH_matrices_joint[:,4*h:4*h+4])
                print('##################DH R0')
                for h in range(self.nb):
                    print(self.list_of_DH_matrices_joint_in_R0[:, 4 * h:4 * h + 4])
                
                print('g',g)
                print('buoyancy_wrench',buoyancy_wrench)
                print('gravity_wrench',gravity_wrench)
            '''
            wrench_diff += buoyancy_wrench+gravity_wrench
        return wrench_diff

    ###############################################################################
    ####calculate the D_psi_W with psi the displacement of head expressed in F_head, it's for dynamic snake
    ################################################################################
    def one_derivation_psi_W(self, d_psi,step):
        '''
        :param d_psi: delta_psi {tx, ty,tz, theta x, theta y,theta z} expressed in F_head
        : param step: step of numeric derivation
        :return:
        '''
        temp1 = copy.copy(self.list_of_DH_matrices_head)
        temp2 = copy.copy(self.list_of_DH_matrices_joint_in_R0);
        temp3 = copy.copy(self.list_of_DH_matrices_roulie_in_R0)
        temp4 = copy.copy(self.DH_matrix_neck_in_R0)
        psi_hat = np.zeros((4, 4))
        psi_hat[:3, :3] = self.skew_symetric_matrix(d_psi[3:])
        psi_hat[:3, 3] = d_psi[:3]
        self.list_of_DH_matrices_head = np.dot(self.list_of_DH_matrices_head,expm(psi_hat))
        Ad_gi = self.adjoint_matrix_2(self.list_of_DH_matrices_head)

        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench_left = wb + wg.reshape((6, 1))
        F_left = np.dot(Ad_gi.T, new_wrench_left)
        self.list_of_DH_matrices_head = copy.copy(temp1);
        self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2)
        self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3);
        self.DH_matrix_neck_in_R0 = copy.copy(temp4)

        psi_hat[:3, :3] = self.skew_symetric_matrix(-d_psi[3:])
        psi_hat[:3, 3] = -d_psi[:3]
        self.list_of_DH_matrices_head = np.dot(self.list_of_DH_matrices_head, expm(psi_hat))
        Ad_gi = self.adjoint_matrix_2(self.list_of_DH_matrices_head)
        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench_right = wb + wg.reshape((6, 1))
        F_right = np.dot(Ad_gi.T, new_wrench_right)
        self.list_of_DH_matrices_head = copy.copy(temp1);
        self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2)
        self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3);
        self.DH_matrix_neck_in_R0 = copy.copy(temp4)
        gradient = (F_left-F_right)/(2*step)
        return  gradient
    def R_h_wrench_jacobian_matrix_psi(self, step):
        Jacob = np.zeros((6,6))
        for i in range(6):
            d_psi = np.zeros((6))
            d_psi[i] = step
            grad_i = self.one_derivation_psi_W(d_psi,step)
            Jacob[:,i] = grad_i[:,0]
        return Jacob

    def one_derivation_q_W(self, q_0, dq, step):
        '''
        :param q_0: equilibrium body shape [qn,qf1,qr1,...,]
        : param dq: variation of q_0
        : param step: step of numeric derivation
        :return:
        '''
        temp2 = copy.copy(self.list_of_DH_matrices_joint_in_R0);
        temp3 = copy.copy(self.list_of_DH_matrices_roulie_in_R0)
        temp4 = copy.copy(self.DH_matrix_neck_in_R0)
        q = q_0+dq
        self.neck_config = q[:2]
        for i in range(self.nb):
            self.joint_config[i] = q[2 * i + 2]
            self.antirouli_config[i] = q[2 * i + 3]
        Ad_gi = self.adjoint_matrix_2(self.list_of_DH_matrices_head)
        self.get_neck_frame()
        self.get_list_joint_frame()
        self.get_list_roulie_frame()
        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench_left = wb + wg.reshape((6, 1))
        F_left = np.dot(Ad_gi.T, new_wrench_left)
        self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2)
        self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3);
        self.DH_matrix_neck_in_R0 = copy.copy(temp4)

        q = q_0 - dq
        self.neck_config = q[:2]
        for i in range(self.nb):
            self.joint_config[i] = q[2 * i + 2]
            self.antirouli_config[i] = q[2 * i + 3]
        Ad_gi = self.adjoint_matrix_2(self.list_of_DH_matrices_head)
        self.get_neck_frame()
        self.get_list_joint_frame()
        self.get_list_roulie_frame()
        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench_right = wb + wg.reshape((6, 1))
        F_right = np.dot(Ad_gi.T, new_wrench_right)
        self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2)
        self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3);
        self.DH_matrix_neck_in_R0 = copy.copy(temp4)
        gradient = (F_left - F_right) / (2 * step)

        self.neck_config = q_0[:2]
        for i in range(self.nb):
            self.joint_config[i] = q_0[2 * i + 2]
            self.antirouli_config[i] = q_0[2 * i + 3]
        return gradient

    def R_h_wrench_jacobian_matrix_q(self, step):
        Jacob = np.zeros((6, 2*self.nb+2))
        q0 = np.zeros((2*self.nb+2))
        q0[:2] = self.neck_config
        for i in range(self.nb):
            q0[2 * i + 2] = self.joint_config[i]
            q0[2 * i + 3] = self.antirouli_config[i]
        for i in range(2*self.nb+2):
            d_q = np.zeros((2*self.nb+2))
            d_q[i] = step
            grad_i = self.one_derivation_q_W(q0, d_q, step)
            Jacob[:, i] = grad_i[:,0]
        q0 = np.zeros((2 * self.nb + 2))
        q0[:2] = self.neck_config
        for i in range(self.nb):
            q0[2 * i + 2] = self.joint_config[i]
            q0[2 * i + 3] = self.antirouli_config[i]
        return Jacob
    ###################################################################################
    ###################################################################################


    def R0_wrench_jacobian_matrix(self, pas):
        temp1 = copy.copy(self.list_of_DH_matrices_head)
        temp2 = copy.copy(self.list_of_DH_matrices_joint_in_R0);
        temp3 = copy.copy(self.list_of_DH_matrices_roulie_in_R0)
        temp4 = copy.copy(self.DH_matrix_neck_in_R0)
        #delta_T = np.array([1, 1, 0, 0, 0, 1]) * pas
        gradiant = np.empty((3,3))
        #d_z = pas; d_omega_x = pas; d_omega_y = pas
        d_head_config_z = np.zeros((4, 4)); d_head_config_z[2, 3] = pas
        g_0_new = expm(d_head_config_z)
        self.list_of_DH_matrices_head = np.dot(g_0_new, self.list_of_DH_matrices_head)
        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench_left = wb + wg.reshape((6, 1))
        self.list_of_DH_matrices_head = copy.copy(temp1); self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2)
        self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3); self.DH_matrix_neck_in_R0 = copy.copy(temp4)
        d_head_config_z = np.zeros((4, 4)); d_head_config_z[2, 3] = -pas
        g_0_new = expm(d_head_config_z)
        self.list_of_DH_matrices_head = np.dot(g_0_new, self.list_of_DH_matrices_head)
        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench_right = wb + wg.reshape((6, 1))
        d_Wn = new_wrench_left - new_wrench_right
        for i in range(3):
            gradiant[i, 0] = d_Wn[i+2, 0]/(2*pas)
        self.list_of_DH_matrices_head = copy.copy(temp1); self.list_of_DH_matrices_joint_in_R0=copy.copy(temp2)
        self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3); self.DH_matrix_neck_in_R0 = copy.copy(temp4)
        ########################
        d_head_config_omega_x = np.zeros((4, 4))
        d_head_config_omega_x[:3, :3] = self.skew_symetric_matrix(pas * np.array([1, 0, 0]))
        g_0_new = expm(d_head_config_omega_x)
        self.list_of_DH_matrices_head = np.dot(g_0_new, self.list_of_DH_matrices_head)
        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench_left = wb + wg.reshape((6, 1))
        self.list_of_DH_matrices_head = copy.copy(temp1); self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2)
        self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3); self.DH_matrix_neck_in_R0 = copy.copy(temp4)
        ####
        d_head_config_omega_x = np.zeros((4, 4))
        d_head_config_omega_x[:3, :3] = self.skew_symetric_matrix(2 * pas * np.array([1, 0, 0]))
        g_0_new = expm(d_head_config_omega_x)
        self.list_of_DH_matrices_head = np.dot(g_0_new, self.list_of_DH_matrices_head)
        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench_left_left = wb + wg.reshape((6, 1))
        self.list_of_DH_matrices_head = copy.copy(temp1); self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2)
        self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3); self.DH_matrix_neck_in_R0 = copy.copy(temp4)
        ####
        ####
        d_head_config_omega_x = np.zeros((4, 4))
        d_head_config_omega_x[:3, :3] = self.skew_symetric_matrix(-2 * pas * np.array([1, 0, 0]))
        g_0_new = expm(d_head_config_omega_x)
        self.list_of_DH_matrices_head = np.dot(g_0_new, self.list_of_DH_matrices_head)
        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench_left_right_right = wb + wg.reshape((6, 1))
        self.list_of_DH_matrices_head = copy.copy(temp1); self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2)
        self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3); self.DH_matrix_neck_in_R0 = copy.copy(temp4)
        ####
        ####
        d_head_config_omega_x = np.zeros((4, 4))
        d_head_config_omega_x[:3, :3] = self.skew_symetric_matrix(-pas * np.array([1, 0, 0]))
        g_0_new = expm(d_head_config_omega_x)
        self.list_of_DH_matrices_head = np.dot(g_0_new, self.list_of_DH_matrices_head)
        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench_right = wb + wg.reshape((6, 1))
        d_Wn =  -new_wrench_left_left + 8 * new_wrench_left - 8*new_wrench_right+new_wrench_left_right_right
        for i in range(3):
            gradiant[i, 1] = d_Wn[i+2, 0]/(12*pas)
        self.list_of_DH_matrices_head = copy.copy(temp1); self.list_of_DH_matrices_joint_in_R0=copy.copy(temp2)
        self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3); self.DH_matrix_neck_in_R0 = copy.copy(temp4)
        #######################
        d_head_config_omega_y = np.zeros((4, 4))
        d_head_config_omega_y[:3, :3] = self.skew_symetric_matrix(pas * np.array([0, 1, 0]))
        g_0_new = expm(d_head_config_omega_y)
        self.list_of_DH_matrices_head = np.dot(g_0_new, self.list_of_DH_matrices_head)
        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench_left = wb + wg.reshape((6, 1))
        self.list_of_DH_matrices_head = copy.copy(temp1); self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2)
        self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3); self.DH_matrix_neck_in_R0 = copy.copy(temp4)
        d_head_config_omega_y = np.zeros((4, 4))
        d_head_config_omega_y[:3, :3] = self.skew_symetric_matrix(-pas * np.array([0, 1, 0]))
        g_0_new = expm(d_head_config_omega_y)
        self.list_of_DH_matrices_head = np.dot(g_0_new, self.list_of_DH_matrices_head)
        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench_right = wb + wg.reshape((6, 1))
        d_Wn = new_wrench_left - new_wrench_right
        for i in range(3):
            gradiant[i, 2] = d_Wn[i + 2, 0] / (2*pas)
        self.list_of_DH_matrices_head = copy.copy(temp1); self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2)
        self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3); self.DH_matrix_neck_in_R0 = copy.copy(temp4)
        return gradiant

    def inverse_configuration(self,g):
        g_inv= np.eye(4)
        R_t = np.transpose(g[:3, :3])
        g_inv[:3, :3] = R_t; g_inv[:3, 3] = - np.dot(R_t, g[:3, 3])
        return g_inv

    def adjoint_matrix_2(self,g):
        adj = np.zeros((6,6))
        adj[:3,:3] = adj[3:6, 3:6] = g[:3,:3]
        adj[:3,3:6] = np.dot(self.skew_symetric_matrix(g[:3, 3]), g[:3, :3])
        return adj

    def find_equilibrum_head_config_test(self, init_head_config=None, joint_config=None, rouli_config=None, Newton=bool,accelarate=bool,tau=5E-4):
        if joint_config is not None:
            self.joint_config = joint_config
        if rouli_config is not None:
            self.antirouli_config = rouli_config
        if init_head_config is not None:
            self.list_of_DH_matrices_head = init_head_config
        tn = 0.5
        self.get_neck_frame();self.get_list_joint_frame(); self.get_list_roulie_frame();self.get_list_frame_R02()
        list_head = []
        list_energie = []
        Fw_equi = self.M * self.g
        t = []; X = []; X2 = []
        while True:
            Temp_head = copy.copy(self.list_of_DH_matrices_head)
            print('before',self.list_of_DH_matrices_head)
            list_head.append(self.list_of_DH_matrices_head)
            # condition initial
            wb = self.get_list_buoyancy_wrench()
            wg = self.gravity_wrench()
            w_R0 = wb + wg.reshape((6, 1))
            Adj_g0 = self.adjoint_matrix_2(self.list_of_DH_matrices_head)
            W0_n = np.dot(np.transpose(Adj_g0), w_R0)
            Un = self.gravity_ponential_energy() - self.buoyancy_potential_energy()
            print('energy',Un)
            print('wrench',w_R0, np.linalg.norm(W0_n))
            if np.linalg.norm(W0_n) / Fw_equi < tau:
                # if np.linalg.norm(W_com_np1) / Fw_equi < 5E-4:
                #print(np.linalg.norm(W0_n))
                break

            gradient_Y = self.R0_wrench_jacobian_matrix(1E-5)
            #print('grad',gradient_Y)
            if gradient_Y[:,0].all()==0.:
                gradient_Y=np.eye(3)
                delta_Y = np.dot(-np.linalg.inv(gradient_Y), w_R0[2:5, 0])
            else:
                 delta_Y = np.dot(-np.linalg.inv(gradient_Y), w_R0[2:5, 0])

            #print('new grad', gradient_Y)
            delta_Y = delta_Y / np.linalg.norm(delta_Y)
            Yg_n = np.zeros((6, 1));Yg_n[2:5, 0] = delta_Y
            # move the head
            #print('s', s)
            g0_inv = self.inverse_configuration(self.list_of_DH_matrices_head)
            Adj_go_inv = self.adjoint_matrix_2(g0_inv)
            Y0_n = np.dot(Adj_go_inv, Yg_n)
            Y0_n_hat = np.zeros((4, 4))
            Y0_n_hat[:3, :3] = self.skew_symetric_matrix(Y0_n[3:6, 0]);
            Y0_n_hat[:3, 3] = Y0_n[:3, 0]
            n_g0_np1 = expm(tn * Y0_n_hat)
            for i in range(3):
                n_g0_np1[:3, i] = n_g0_np1[:3, i] / np.linalg.norm(n_g0_np1[:3, i])
            self.list_of_DH_matrices_head = np.dot(self.list_of_DH_matrices_head, n_g0_np1)
            # calculate the new body configuration
            self.get_list_frame_R02()
            Unp1 = self.gravity_ponential_energy() - self.buoyancy_potential_energy( )
            # print(np.linalg.norm(W_com_np1)/Fw_equi)

            list_energie.append(Un)
            X.append(np.linalg.norm(W0_n[2]))
            X2.append(np.linalg.norm(W0_n[2:]))
            c1 = 1E-10
            #print('i', Unp1 - Un)
            #print('test', np.dot(np.transpose(X0_n), W0_n))
            #print('iden', np.dot(np.transpose(Y0_n), W0_n))
            if np.dot(np.transpose(Y0_n), W0_n)<0 or Newton is False:
                print('cao')
                self.list_of_DH_matrices_head = copy.copy(Temp_head)
                self.get_list_frame_R02()
                Xg_n = w_R0/np.linalg.norm(w_R0)
                Adj_g0 = self.adjoint_matrix_2(self.list_of_DH_matrices_head)
                W0_n = np.dot(np.transpose(Adj_g0), w_R0)
                g0_inv = self.inverse_configuration(self.list_of_DH_matrices_head)
                Adj_go_inv = self.adjoint_matrix_2(g0_inv)
                Y0_n = np.dot(Adj_go_inv, Xg_n)
                print('check',np.dot(np.transpose(Y0_n), W0_n))
                Y0_n_hat = np.zeros((4, 4))
                Y0_n_hat[:3, :3] = self.skew_symetric_matrix(Y0_n[3:6, 0]);
                Y0_n_hat[:3, 3] = Y0_n[:3, 0]
                n_g0_np1 = expm(tn * Y0_n_hat)
                print()
                for i in range(3):
                    n_g0_np1[:3, i] = n_g0_np1[:3, i] / np.linalg.norm(n_g0_np1[:3, i])
                self.list_of_DH_matrices_head = np.dot(self.list_of_DH_matrices_head, n_g0_np1)
                # calculate the new body configuration
                self.get_list_frame_R02()
                Unp1 = self.gravity_ponential_energy() - self.buoyancy_potential_energy()
               # print('iden', np.dot(np.transpose(Y0_n), W0_n))
            # print('i',Unp1 < Un)
            cc = 0
            while Unp1 > Un - c1 * tn * np.dot(np.transpose(Y0_n), W0_n):
                cc+=1

                #if cc>5:
                 #   break
                #print('Unp1-Un',Unp1-Un)
                # print('np.dot(np.transpose(X0_n), W0_n)',np.dot(np.transpose(Y0_n), W0_n),- c1 * tn * np.dot(np.transpose(Y0_n), W0_n))
                tn = 0.5 * tn
                # print(Unp1 - Un)
                self.list_of_DH_matrices_head = Temp_head
                n_g0_np1 = expm(tn * Y0_n_hat)
                for i in range(3):
                    n_g0_np1[:3, i] = n_g0_np1[:3, i] / np.linalg.norm(n_g0_np1[:3, i])
                self.list_of_DH_matrices_head = np.dot(self.list_of_DH_matrices_head, n_g0_np1)
                print(self.list_of_DH_matrices_head)
                self.get_list_frame_R02()
                Unp1 = self.gravity_ponential_energy() - self.buoyancy_potential_energy()
                print(Unp1 < Un)
                print('U',Unp1,Un)
                print('Ug',self.gravity_ponential_energy())
                print('tn', tn)
            if Newton is False:
                wb = self.get_list_buoyancy_wrench()
                # wb = self.get_list_buoyancy_wrench(100, 100, False, True, False)
                wg = self.gravity_wrench()
                w_R0 = wb + wg.reshape((6, 1));  # w_R0[0,0] = 0; w_R0[1,0] = 0; w_R0[5,0] = 0
                Adj_g0 = self.adjoint_matrix_2(self.list_of_DH_matrices_head)
                W0_np1_new = np.dot(np.transpose(Adj_g0), w_R0)
                omega = self.acceleration_gradient_descent(-W0_n, -W0_np1_new)
                self.list_of_DH_matrices_head = copy.copy(Temp_head)
                n_g0_np1 = expm(omega * tn * Y0_n_hat)
                for i in range(3):
                    n_g0_np1[:3, i] = n_g0_np1[:3, i] / np.linalg.norm(n_g0_np1[:3, i])
                self.list_of_DH_matrices_head = np.dot(self.list_of_DH_matrices_head, n_g0_np1)
                self.get_list_frame_R02()
            #print('j', Unp1 - Un)
            print('tn',tn)
            t.append(tn)
            H = copy.copy(self.list_of_DH_matrices_head)
            R = Rotation.from_matrix(H[:3, :3])
            angle_euler = R.as_euler('xyz')
            print('angle', angle_euler)
            angle_euler[2] = 0.
            angle_euler = Rotation.from_euler('xyz', angle_euler)
            R = angle_euler.as_matrix()
            self.list_of_DH_matrices_head[:3, :3] = R; self.list_of_DH_matrices_head[0, 3] = 0.; self.list_of_DH_matrices_head[1, 3] = 0.
            #print('tn',tn)
            # calculate the new brench on the original of R0
        return self.list_of_DH_matrices_head, list_head, list_energie, t, X, X2


    def acceleration_gradient_descent(self, gk, gz):
        ak = pow(np.linalg.norm(gk), 2)
        temp = gz - gk; bk = - np.dot(np.transpose(temp), gk)
        return ak / bk

    def find_equilibrum_head_config2(self, init_head_config, joint_config, rouli_config, tau1, tau2, Newton=bool):
        self.joint_config = joint_config
        self.antirouli_config = rouli_config
        self.list_of_DH_matrices_head = init_head_config
        tn = 0.5
        self.get_list_frame_R02()
        head_g_com, com_g_head = self.find_COM()
        Tau = np.eye(6)
        Tau[0, 0] = Tau[1, 1] = Tau[2, 2] = tau1; Tau[3, 3] = Tau[4, 4] = Tau[5, 5] = tau2
        list_head = []
        list_energie = []
        Fw_equi = self.M * self.g
        U = []; X = []
        while True:
            Temp_head = self.list_of_DH_matrices_head
            list_head.append(self.list_of_DH_matrices_head)
            # condition initial
            #wb = self.get_list_buoyancy_wrench(50, 100, False, True, False)
            wb = self.get_list_buoyancy_wrench()
            wg = self.gravity_wrench()
            w_R0 = wb + wg.reshape((6, 1));  # w_R0[0,0] = 0; w_R0[1,0] = 0; w_R0[5,0] = 0
            o_g_com_n = np.dot(self.list_of_DH_matrices_head, head_g_com)
            o_g_com_n_inv = np.eye(4)
            R_t = np.transpose(o_g_com_n[:3, :3])
            o_g_com_n_inv[:3, :3] = R_t; o_g_com_n_inv[:3, 3] = - np.dot(R_t, o_g_com_n[:3, 3])
            Adj = self.adjoint_matrix(o_g_com_n_inv)
            W_com_n = np.dot(Adj, w_R0)
            X.append(np.linalg.norm(W_com_n))
            # print(np.linalg.norm(W_com_np1)/Fw_equi)
            if np.linalg.norm(W_com_n) / Fw_equi < 1E-5:
                # if np.linalg.norm(W_com_np1) / Fw_equi < 5E-4:
                print(np.linalg.norm(W_com_n))
                break
            z_com = o_g_com_n[2, 3]
            Un = z_com * self.M - self.buoyancy_potential_energy(50, True)
            list_energie.append(Un)
            # gradiant of the potential energy; the factor matrix is identity matrix
            Xi_n = np.dot(Tau, W_com_n)
            # move the head
            Xi_n = Xi_n / np.linalg.norm(Xi_n)
            Xi_n_hat = np.zeros((4, 4))
            Xi_n_hat[:3, :3] = self.skew_symetric_matrix(Xi_n[3:6, 0]); Xi_n_hat[:3, 3] = Xi_n[:3, 0]
            n_gcom_np1 = expm(tn * Xi_n_hat)
            for i in range(3):
                n_gcom_np1[:3, i] = n_gcom_np1[:3, i] / np.linalg.norm(n_gcom_np1[:3, i])
            o_g_com_np1 = np.dot(o_g_com_n, n_gcom_np1)
            self.list_of_DH_matrices_head = np.dot(o_g_com_np1, com_g_head)
            # calculate the new body configuration
            self.get_list_frame_R02()
            # calculate the new potential energy
            o_g_com_np1 = np.dot(self.list_of_DH_matrices_head, head_g_com)
            z_com = o_g_com_np1[2, 3]
            Unp1 = z_com * self.M - self.buoyancy_potential_energy(50,True)
            c1 = 1e-4
            while Unp1 > Un - c1 * tn * np.dot(np.transpose(Xi_n), W_com_n):
                tn = 0.8 * tn
                self.list_of_DH_matrices_head = Temp_head
                n_gcom_np1 = expm(tn * Xi_n_hat)
                for i in range(3):
                    n_gcom_np1[:3, i] = n_gcom_np1[:3, i] / np.linalg.norm(n_gcom_np1[:3, i])
                o_g_com_np1 = np.dot(o_g_com_n, n_gcom_np1)
                self.list_of_DH_matrices_head = np.dot(o_g_com_np1, com_g_head)
                # calculate the new body configuration
                self.get_list_frame_R02()
                # calculate the new brench on the original of R0
                # calculate the new potential energy
                o_g_com_np1 = np.dot(self.list_of_DH_matrices_head, head_g_com)
                z_com = o_g_com_np1[2, 3]
                Unp1 = z_com * self.M - self.buoyancy_potential_energy(50,True)
            ##############accelate the decsent gradiant
            wb = self.get_list_buoyancy_wrench()
            #wb = self.get_list_buoyancy_wrench(100, 100, False, True, False)
            wg = self.gravity_wrench()
            w_R0 = wb + wg.reshape((6, 1));  # w_R0[0,0] = 0; w_R0[1,0] = 0; w_R0[5,0] = 0
            o_g_com_np1_inv = np.eye(4)
            R_t = np.transpose(o_g_com_np1[:3, :3])
            o_g_com_np1_inv[:3, :3] = R_t; o_g_com_np1_inv[:3, 3] = - np.dot(R_t, o_g_com_np1[:3, 3])
            Adj_new = self.adjoint_matrix(o_g_com_np1_inv)
            W_com_np1 = np.dot(Adj_new, w_R0)
            Xi_np1 = np.dot(Tau, W_com_np1)
            # move the head
            Xi_np1 = Xi_np1 / np.linalg.norm(Xi_np1)
            omega = self.acceleration_gradient_descent(-Xi_n, -Xi_np1)
            self.list_of_DH_matrices_head = Temp_head
            n_gcom_np1 = expm(omega * tn * Xi_n_hat)
            for i in range(3):
                n_gcom_np1[:3, i] = n_gcom_np1[:3, i] / np.linalg.norm(n_gcom_np1[:3, i])
            o_g_com_np1 = np.dot(o_g_com_n, n_gcom_np1)
            self.list_of_DH_matrices_head = np.dot(o_g_com_np1, com_g_head)
            self.get_list_frame_R02()
            U.append(tn)
        return self.list_of_DH_matrices_head, list_head, list_energie, U, X

    def parallel_vertical_operation(self):
        self.grad = self.R0_wrench_jacobian_matrix_of_body(1E-5)

        u, s, v = np.linalg.svd(self.grad)
        if s.shape[0] == 3:
            if s[2] != 0.:
                #print('dqW', v)
                self.Pparal = np.transpose(v[3:, :])
                #print('Pparal',self.Pparal)
                self.Pvec = np.transpose(v[:3, :])
                # print('Pvec',self.Pvec)
            else:
                print('Ahhhhhhhhhhhhhh')
                self.Pparal = np.transpose(v[2:, :])
                self.Pvec = np.transpose(v[:2, :])
        else:
            self.Pparal = np.transpose(v[2:, :])
            self.Pvec = np.transpose(v[:2, :])
        return self.Pparal, self.Pvec, self.grad

    def transmission_on_nullspace(self,X,lock_roll,residu,last_wrench,fac):
        '''
        :param X: q// = P//*X
        :param lock_roll: if we lock the rotations of rolls
        :return:
        '''
        Fw_equi = self.M * self.g
        DqW = np.dot(self.grad, self.Pvec)
        dQ = np.dot(self.Pparal, X)

        dQ_joint = np.zeros(self.nb)

        for j in range(1,self.nb):
            dQ_joint[j] = dQ[2 * j+1]
        print('after proj', dQ_joint)

        if np.linalg.norm(dQ_joint)<0.0005:
             dQ_joint = residu*fac

        for j in range(1, self.nb):
            dQ[2 * j + 1] = dQ_joint[j]
        #dQ = residu*dQ/np.linalg.norm(dQ)
        #print('q //',dQ)
        self.neck_config[1] += dQ[0]
        for j in range(self.nb):
                self.joint_config[j] += dQ[2 * j+1]
                self.antirouli_config[j] += dQ[2 * j + 2]

        self.correct_angle()
        temp1 = copy.copy(self.list_of_DH_matrices_head); temp2 = copy.copy(self.list_of_DH_matrices_joint_in_R0);  temp3 = copy.copy(self.list_of_DH_matrices_roulie_in_R0)
        temp4 = copy.copy(self.joint_config);  temp5 = copy.copy(self.antirouli_config)
        temp6 = copy.copy(self.neck_config); temp7 = copy.copy(self.DH_matrix_neck)
        self.get_neck_frame()
        self.get_list_joint_frame(); self.get_list_roulie_frame(); self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench = wb + wg.reshape((6, 1))
        if DqW.shape[0] == DqW.shape[1]:
           invDqW = np.linalg.inv(DqW)
        else:
            invDqW = np.linalg.pinv(DqW)
        if not lock_roll:
            q = - np.dot(invDqW, new_wrench[2:5, 0]-last_wrench[2:5, 0])
        else:
            q = - np.dot(invDqW, new_wrench[3:5, 0]-last_wrench[3:5,0])
        W_old = np.linalg.norm(new_wrench) / Fw_equi
        #print('W||',W_old)
        #print('avant',np.linalg.norm(new_wrench) / Fw_equi,np.linalg.norm(new_wrench))
        if W_old > 5E-10:
            dQ = np.dot(self.Pvec, q)
            #print('q vec', dQ)
            self.neck_config[1] += dQ[0]
            for j in range(self.nb):
                    self.joint_config[j] += dQ[2 * j+1]
                    self.antirouli_config[j] += dQ[2 * j + 2]
            self.correct_angle()
            #print('parallel transmision',self.joint_config,self.antirouli_config)
            self.get_neck_frame()
            self.get_list_joint_frame(); self.get_list_roulie_frame(); self.get_list_frame_R02()
            wb = self.get_list_buoyancy_wrench()
            wg = self.gravity_wrench()
            new_wrench = wb + wg.reshape((6, 1))

            q1 = q; q2 = q
            W_retraction = np.linalg.norm(new_wrench) / Fw_equi
            #print('W_retra',W_retraction)
            #print('après',np.linalg.norm(new_wrench) / Fw_equi, np.linalg.norm(new_wrench))
            if W_retraction > 1E-10:
                self.list_of_DH_matrices_head = copy.copy(temp1)
                self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2)
                self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3)
                self.joint_config = copy.copy(temp4)
                self.antirouli_config = copy.copy(temp5)
                self.neck_config = copy.copy(temp6); self.DH_matrix_neck = copy.copy(temp7)
                q1 = 1.05 * q1
                dQ1 = np.dot(self.Pvec, q1)
                self.neck_config[1] += dQ[0]
                for j in range(self.nb):
                    self.joint_config[j] += dQ1[2 * j+1]
                    self.antirouli_config[j] += dQ1[2 * j + 2]
                self.correct_angle()
                self.get_neck_frame()
                self.get_list_joint_frame(); self.get_list_roulie_frame(); self.get_list_frame_R02()
                wb = self.get_list_buoyancy_wrench()
                wg = self.gravity_wrench()
                new_wrench = wb + wg.reshape((6, 1))
                W_retraction_up=np.linalg.norm(new_wrench) / Fw_equi
                #print('W_retraction_up',W_retraction_up)
                list_dQ1 = [dQ1]
                if W_retraction_up < W_retraction:
                    while W_retraction_up < W_retraction:
                        W_retraction = W_retraction_up
                        self.list_of_DH_matrices_head = copy.copy(temp1)
                        self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2)
                        self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3)
                        self.joint_config = copy.copy(temp4)
                        self.antirouli_config = copy.copy(temp5)
                        self.neck_config = copy.copy(temp6);self.DH_matrix_neck_in_R0 = copy.copy(temp7)
                        q1 = 1.1 * q1
                        dQ1 = np.dot(self.Pvec, q1)
                        list_dQ1.append(dQ1)
                        self.neck_config[1] += dQ[0]
                        for j in range(self.nb):
                            self.joint_config[j] += dQ1[2 * j+1]
                            self.antirouli_config[j] += dQ1[2 * j + 2]
                        self.correct_angle()
                        self.get_neck_frame()
                        self.get_list_joint_frame(); self.get_list_roulie_frame(); self.get_list_frame_R02()
                        wb = self.get_list_buoyancy_wrench()
                        wg = self.gravity_wrench()
                        new_wrench = wb + wg.reshape((6, 1))
                        W_retraction_up = np.linalg.norm(new_wrench) / Fw_equi
                        #print('W_retraction_up',W_retraction_up)
                else:
                    self.list_of_DH_matrices_head = copy.copy(temp1)
                    self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2)
                    self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3)
                    self.joint_config = copy.copy(temp4)
                    self.antirouli_config = copy.copy(temp5)
                    self.neck_config = copy.copy(temp6); self.DH_matrix_neck_in_R0 = copy.copy(temp7)
                    q2 = 0.95 * q2
                    dQ2 = np.dot(self.Pvec, q2)
                    list_dQ2 = [dQ2]
                    self.neck_config[1] = dQ2[0]
                    for j in range(self.nb):
                            self.joint_config[j] += dQ2[2 * j+1]
                            self.antirouli_config[j] += dQ2[2 * j + 2]
                    self.correct_angle()
                    #print('vectorial transmision', self.joint_config, self.antirouli_config)
                    self.get_neck_frame()
                    self.get_list_joint_frame(); self.get_list_roulie_frame(); self.get_list_frame_R02()
                    wb = self.get_list_buoyancy_wrench()
                    wg = self.gravity_wrench()
                    new_wrench = wb + wg.reshape((6, 1))
                    W_retraction_down = np.linalg.norm(new_wrench) / Fw_equi
                    #print('W_retraction_down',W_retraction_down)
                    while W_retraction_down < W_retraction:
                        W_retraction = W_retraction_down
                        self.list_of_DH_matrices_head = copy.copy(temp1)
                        self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2)
                        self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3)
                        self.joint_config = copy.copy(temp4)
                        self.antirouli_config = copy.copy(temp5)
                        self.neck_config = copy.copy(temp6); self.DH_matrix_neck_in_R0 = copy.copy(temp7)
                        q2 = 0.9 * q2
                        list_dQ2.append(dQ2)
                        dQ2 = np.dot(self.Pvec, q2)
                        self.neck_config[1] = dQ2[0]
                        for j in range(self.nb):
                            self.joint_config[j] += dQ2[2 * j+1]
                            self.antirouli_config[j] += dQ2[2 * j + 2]
                        self.correct_angle()
                        # print('vectorial transmision', self.joint_config, self.antirouli_config)
                        self.get_neck_frame()
                        self.get_list_joint_frame(); self.get_list_roulie_frame(); self.get_list_frame_R02()
                        wb = self.get_list_buoyancy_wrench()
                        wg = self.gravity_wrench()
                        new_wrench = wb + wg.reshape((6, 1))
                        W_retraction_down = np.linalg.norm(new_wrench) / Fw_equi
                        #print('W_retraction_down',W_retraction_down)
                        #print('W_retraction_down',W_retraction_down,np.linalg.norm(new_wrench))
                        if W_retraction_down-W_retraction > 0 or abs(W_retraction_down-W_retraction) < 1E-8:
                            break
                    #print('len',len(list_dQ2))
                    '''
                    if not len(list_dQ2)<2:
                       dQ2 = list_dQ2[-2]
                    else:
                        dQ2 = list_dQ2[-1]
                    for j in range(self.nb):
                        self.joint_config[j] += dQ2[2 * j]
                        self.antirouli_config[j] += dQ2[2 * j + 1]
                    self.correct_angle()
                    self.get_list_joint_frame(); self.get_list_roulie_frame(); self.get_list_frame_R02()
                    wb = self.get_list_buoyancy_wrench(Sp_rule=True)
                    wg = self.gravity_wrench()
                    new_wrench = wb + wg.reshape((6, 1))
                    '''
                    #print('B', np.linalg.norm(new_wrench) / Fw_equi,np.linalg.norm(new_wrench),)
        else:
            print('ohhhh')
        if W_old < np.linalg.norm(new_wrench) / Fw_equi:
            print('attention!')
            self.list_of_DH_matrices_head = copy.copy(temp1)
            self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2)
            self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3)
            self.joint_config = copy.copy(temp4)
            self.antirouli_config = copy.copy(temp5)
            self.neck_config = copy.copy(temp6); self.DH_matrix_neck_in_R0 = copy.copy(temp7)
        return self.neck_config[1],self.joint_config, self.antirouli_config
    #def local_stifness_optimisation(self):


    def derivation_stiffness(self):
        temp2 = copy.copy(self.list_of_DH_matrices_joint_in_R0);
        temp3 = copy.copy(self.list_of_DH_matrices_roulie_in_R0)
        temp4 = copy.copy(self.joint_config);
        temp5 = copy.copy(self.antirouli_config)
        temp6 = copy.copy(self.neck_config);
        temp7 = copy.copy(self.DH_matrix_neck_in_R0)

        Dq_lambda = np.zeros((self.nb+2))
        for i in range(self.nb+2):
            if i == 0:
                self.neck_config[1] += 1E-5
            elif i == 1:
                self.joint_config[0] += 1E-5
            else:
                self.antirouli_config[i-2] += 1E-5
            self.get_neck_frame()
            self.get_list_joint_frame();
            self.get_list_roulie_frame();
            self.get_list_frame_R02()
            Grad = self.R0_wrench_jacobian_matrix(1E-4)
            w, v = np.linalg.eig(Grad)
            w3_forward = w[2]
            if i == 0:
                self.neck_config[1] -= 2E-5
            elif i == 1:
                self.joint_config[0] -= 2E-5
            else:
                self.antirouli_config[i - 2] -= 2E-5
            self.get_neck_frame()
            self.get_list_joint_frame();
            self.get_list_roulie_frame();
            self.get_list_frame_R02()
            Grad = self.R0_wrench_jacobian_matrix(1E-4)
            w, v = np.linalg.eig(Grad)
            w3_backward = w[2]
            Dq_lambda[i] = (w3_forward-w3_backward)/2E-5
            self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2)
            self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3)
            self.joint_config = copy.copy(temp4)
            self.antirouli_config = copy.copy(temp5)
            self.neck_config = copy.copy(temp6);
            self.DH_matrix_neck_in_R0 = copy.copy(temp7)
        return Dq_lambda

    def transmission_on_nullspace_2(self,X,residu,last_wrench,fac,displace):
        '''
        :param X: q// = P//*X
        :return:
        '''
        dQ = np.dot(self.Pparal, X)
        dQ_joint = np.zeros(self.nb-1)
        for j in range(1,self.nb):
            dQ_joint[j-1] = dQ[2 * j+1]
        print('after proj', dQ_joint)
        turn_back = False

        if displace:#np.linalg.norm(dQ_joint) < 0.005:
             dQ = residu*fac/6
             print('watch out')
             turn_back = True

        self.neck_config[1] += dQ[0]
        for j in range(self.nb):
                self.joint_config[j] += dQ[2 * j+1]
                self.antirouli_config[j] += dQ[2 * j + 2]
        self.correct_angle()
        self.get_neck_frame()
        self.get_list_joint_frame(); self.get_list_roulie_frame(); self.get_list_frame_R02()
        temp1 = copy.copy(self.list_of_DH_matrices_head);
        temp2 = copy.copy(self.list_of_DH_matrices_joint_in_R0);
        temp3 = copy.copy(self.list_of_DH_matrices_roulie_in_R0)
        temp4 = copy.copy(self.joint_config);
        temp5 = copy.copy(self.antirouli_config)
        temp6 = copy.copy(self.neck_config);
        temp7 = copy.copy(self.DH_matrix_neck_in_R0)
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        old_wrench = wb + wg.reshape((6, 1))
        DqW_vec = np.dot(self.grad, self.Pvec)
        if DqW_vec.shape[0] == DqW_vec.shape[1]:
           invDqW_vec = np.linalg.inv(DqW_vec)
        else:
            invDqW_vec = np.linalg.pinv(DqW_vec)
        invDqW_vec = -np.dot(self.Pvec,invDqW_vec)
        dW = old_wrench[2:5, 0]-last_wrench[2:5, 0]
        #q = - np.dot(invDqW, old_wrench[2:5, 0]-last_wrench[2:5, 0])
        if turn_back==False:
            solution = optimize.minimize_scalar(self.retraction, method='brent', args=(dW, invDqW_vec))
            alpha = solution.x
        else:
            alpha = 0
        '''
        invDqW_vec_retraction = np.empty((invDqW_vec.shape[0], invDqW_vec.shape[1]))
        for i in range(3):
            invDqW_vec_retraction[:, i] = alpha[i] * invDqW_vec[:, i]
        '''
        invDqW_vec_retraction = alpha * invDqW_vec
        dQ = np.dot(invDqW_vec_retraction, dW)
        dQ_joint = np.zeros(self.nb - 1)
        for j in range(1, self.nb):
            dQ_joint[j - 1] = dQ[2 * j + 1]
        print('after retraction', dQ_joint)
        self.neck_config[1] += dQ[0]
        for j in range(self.nb):
            self.joint_config[j] += dQ[2 * j + 1]
            self.antirouli_config[j] += dQ[2 * j + 2]
        self.correct_angle()
        self.get_neck_frame()
        self.get_list_joint_frame(); self.get_list_roulie_frame(); self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench = wb + wg.reshape((6, 1))
        if np.linalg.norm(new_wrench) > np.linalg.norm(old_wrench):
            self.list_of_DH_matrices_head = copy.copy(temp1)
            self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2)
            self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3)
            self.joint_config = copy.copy(temp4)
            self.antirouli_config = copy.copy(temp5)
            self.neck_config = copy.copy(temp6);
            self.DH_matrix_neck_in_R0 = copy.copy(temp7)
            print('wtf')
        return self.neck_config[1],self.joint_config, self.antirouli_config, turn_back

    def retraction(self, alpha, dW, invDqW_vec):
        temp2 = copy.copy(self.list_of_DH_matrices_joint_in_R0); temp3 = copy.copy(self.list_of_DH_matrices_roulie_in_R0)
        temp4 = copy.copy(self.joint_config);temp5 = copy.copy(self.antirouli_config)
        temp6 = copy.copy(self.neck_config); temp7 = copy.copy(self.DH_matrix_neck_in_R0)
        temp8 = copy.copy(self.list_of_DH_matrices_joint); temp9 = copy.copy(self.list_of_DH_matrices_roulie);temp10 = copy.copy(self.DH_matrix_neck)
        '''
        invDqW_vec_retraction = np.empty((invDqW_vec.shape[0],invDqW_vec.shape[1]))
        
        for i in range(3):
            invDqW_vec_retraction[:, i] = alpha[i] * invDqW_vec[:, i]
        '''
        invDqW_vec_retraction= alpha * invDqW_vec
        dQ = np.dot(invDqW_vec_retraction,dW)
        self.neck_config[1] += dQ[0]
        for j in range(self.nb):
            self.joint_config[j] += dQ[2 * j + 1]
            self.antirouli_config[j] += dQ[2 * j + 2]
        self.correct_angle()
        self.get_neck_frame()
        self.get_list_joint_frame();
        self.get_list_roulie_frame();
        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench = wb + wg.reshape((6, 1))
        self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2)
        self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3)
        self.joint_config = copy.copy(temp4)
        self.antirouli_config = copy.copy(temp5)
        self.neck_config = copy.copy(temp6); self.DH_matrix_neck_in_R0 = copy.copy(temp7)
        self.list_of_DH_matrices_joint = copy.copy(temp8); self.list_of_DH_matrices_roulie = copy.copy(temp9); self.DH_matrix_neck = copy.copy(temp10)
        return np.linalg.norm(new_wrench)

    def correct_angle(self):
        for j in range(self.nb):
                while self.joint_config[j] > pi:
                    self.joint_config[j] -= 2 * pi
                while self.joint_config[j] < -pi:
                    self.joint_config[j] += 2 * pi
                while self.antirouli_config[j] > pi:
                    self.antirouli_config[j] -= 2 * pi
                while self.antirouli_config[j] < -pi:
                    self.antirouli_config[j] += 2 * pi
    def find_equilibrum_body_config(self,step,step2,goal_traj,lock_roll,tau=[0.97],goal_roll=None,goal_neck=None):
        '''
        :param step: stepsize of the approximation to the goal body shape
        :param goal_traj: goal body shape
        :param lock_roll if we lock the rotatations of rolls
        :return:
        '''
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench = wb + wg.reshape((6, 1))
        W = [np.linalg.norm(new_wrench)]
        config=[]
        gg = []
        neck_conf = np.array([self.neck_config[1]])
        joint_conf = np.array([self.joint_config])
        roll_conf = np.array([self.antirouli_config])
        iteration = len(goal_traj)
        Fw_equi = self.M*self.g
        for i in range(iteration):
            config.append(self.list_of_DH_matrices_joint_in_R0)
            Q_goal_joint = goal_traj[i]
            Q_goal = np.zeros((2*self.nb+1))
            Q_goal[0] = goal_neck[i][0]
            for l in range(self.nb):
                Q_goal[2*l+1] = Q_goal_joint[l]
                if goal_roll is not None:
                    Q_goal[2*l+2] = goal_roll[i][l]
            #print(w1-w2)
            Pparal, Pvec, grad = self.parallel_vertical_operation()
            Qip1_joint = self.joint_config
            Qip1 = np.zeros((2*self.nb+1))
            Qip1[0] = self.neck_config[1]
            for l in range(self.nb):
                Qip1[2*l+1] = Qip1_joint[l]
                Qip1[2 * l + 2] = self.antirouli_config[l]
            #print('attention',np.dot(Pparal,np.transpose(Pparal)))
            print('norm of goal-q',np.linalg.norm(Qip1_joint[1:]-Q_goal_joint[1:]))
            #distance = np.linalg.norm(Qip1 - Q_goal)
            distance_global = np.linalg.norm(Qip1_joint-Q_goal_joint)
            residu=distance_global/step2[i]
            gg.append(np.linalg.norm(Qip1 - Q_goal))
            v = 0
            ############# illustration
            print('residu',10*residu)
            while np.linalg.norm(Qip1_joint[1:]-Q_goal_joint[1:]) >10*residu:
                print('iteration', v)

                Qip1[0] = self.neck_config[1]
                for l in range(self.nb):
                    Qip1[2 * l + 1] = Qip1_joint[l]
                    Qip1[2 * l + 2] = self.antirouli_config[l]
                # print('attention',np.dot(Pparal,np.transpose(Pparal)))
                print('norm of goal-qn', np.linalg.norm(Qip1 - Q_goal))

                if goal_roll is not None:
                    distance =np.linalg.norm(Qip1_joint[1:] - Q_goal_joint[1:])
                else:
                    distance = np.linalg.norm(Qip1 - Q_goal)
                gg.append(distance)
                v += 1
                if not lock_roll:
                    Xip1 = residu* np.dot(np.transpose(Pparal), (Q_goal - Qip1)/np.linalg.norm(Q_goal - Qip1))#distance * step[i] * np.dot(np.transpose(Pparal), (Q_goal - Qip1)/np.linalg.norm(Q_goal - Qip1))
                else:
                    Xip1 = distance * step[i] * np.dot(np.transpose(Pparal),(Q_goal_joint - Qip1_joint) / np.linalg.norm(Q_goal_joint - Qip1_joint))
                Xip1_joint = np.zeros((self.nb))
                for c in range(self.nb):
                    Xip1_joint[c] = Q_goal_joint[c] - Qip1_joint[c]
                print('before proj',Xip1_joint)
                Qip1_neck,Qip1_joint, Qip1_roll = self.transmission_on_nullspace(Xip1,lock_roll,
                                            residu,new_wrench,Xip1_joint/np.linalg.norm(Xip1_joint))
                Qip1 = np.zeros((2 * self.nb+1))
                Qip1[0] =Qip1_neck
                for l in range(self.nb):
                    Qip1[2 * l+1] = Qip1_joint[l]
                    Qip1[2 * l + 2] = Qip1_roll[l]
                if goal_roll is not None:
                    distance_new = np.linalg.norm(Qip1_joint[1:] - Q_goal_joint[1:])
                else:
                    distance_new = np.linalg.norm(Qip1 - Q_goal)
                self.get_neck_frame()
                self.get_list_joint_frame();  self.get_list_roulie_frame(); self.get_list_frame_R02()
                wb = self.get_list_buoyancy_wrench()
                wg = self.gravity_wrench()
                new_wrench = wb + wg.reshape((6, 1))
                print('test1', np.linalg.norm(new_wrench), np.linalg.norm(new_wrench) / Fw_equi)
                Pparal, Pvec, grad = self.parallel_vertical_operation()
                config.append(self.list_of_DH_matrices_joint_in_R0)
                neck_conf = np.r_[neck_conf,np.array([Qip1_neck])]
                joint_conf = np.r_[joint_conf,np.array([Qip1_joint])]
                roll_conf = np.r_[roll_conf, np.array([Qip1_roll])]
                W.append(np.linalg.norm(new_wrench))
                #if step[i]/step_ini>0.035 or i !=0:
                if 0<=distance-distance_new<residu:
                    step[i] = tau[i]*step[i]
                    print('oui')
                elif distance-distance_new>=residu:
                    step[i] = step[i]/tau[i]
                    print('non')
                else:
                    step[i] = step[i] / (2*tau[i])
                    print('ni oui ni non')
                if v > 500 :#and lock_roll is True:
                    break
            #if i != iteration-1:
              #self.find_equilibrum_head_config_test(Newton=True, accelarate=False, tau=1E-3)
            '''
            self.neck_config = Q
            self.joint_config = Q_goal_joint
            self.antirouli_config = np.zeros((self.nb))
            self.get_list_joint_frame(); self.get_list_roulie_frame(); self.get_list_frame_R02()
            gg.append(0)
            joint_conf = np.r_[joint_conf, np.array([self.joint_config])]
            roll_conf = np.r_[roll_conf, np.array([self.antirouli_config])]
            '''
        return W, config, gg, joint_conf, roll_conf, neck_conf
    def find_equilibrum_body_config_2(self,step,step2,goal_traj,tau=[0.97],goal_roll=None,goal_neck=None,limite=0.13):
        '''
        :param step: stepsize of the approximation to the goal body shape
        :param goal_traj: goal body shape
        :return:
        '''
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench = wb + wg.reshape((6, 1))
        W = [np.linalg.norm(new_wrench)]
        config = []
        gg = []
        neck_conf = np.array([self.neck_config[1]])
        joint_conf = np.array([self.joint_config])
        roll_conf = np.array([self.antirouli_config])
        iteration = len(goal_traj)
        Fw_equi = self.M*self.g
        for i in range(iteration):
            config.append(self.list_of_DH_matrices_joint_in_R0)
            Q_goal_joint = goal_traj[i]
            Q_goal = np.zeros((2*self.nb+1))
            Q_goal[0] = goal_neck[i][0]
            for l in range(self.nb):
                Q_goal[2*l+1] = Q_goal_joint[l]
                if goal_roll is not None:
                    Q_goal[2*l+2] = goal_roll[i][l]
            Pparal, Pvec, grad = self.parallel_vertical_operation()
            Qip1_joint = self.joint_config
            Qip1 = np.zeros((2*self.nb+1))
            Qip1[0] = self.neck_config[1]
            for l in range(self.nb):
                Qip1[2*l+1] = Qip1_joint[l]
                Qip1[2 * l + 2] = self.antirouli_config[l]
            print('norm of goal-q',np.linalg.norm(Qip1_joint[1:]-Q_goal_joint[1:]))
            #distance = np.linalg.norm(Qip1 - Q_goal)
            distance_global = np.linalg.norm(Qip1_joint[1:]-Q_goal_joint[1:])
            residu=distance_global/step2[i]
            gg.append(np.linalg.norm(Qip1 - Q_goal))
            v = 0
            ############# illustration
            print('residu',residu)
            b = 0
            while np.linalg.norm(Qip1_joint[1:]-Q_goal_joint[1:])>residu:
                print('iteration', v)
                b+=1
                if b>=400:
                    break
                Qip1[0] = self.neck_config[1]
                for l in range(self.nb):
                    Qip1[2 * l + 1] = Qip1_joint[l]
                    Qip1[2 * l + 2] = self.antirouli_config[l]
                # print('attention',np.dot(Pparal,np.transpose(Pparal)))

                distance =np.linalg.norm(Qip1_joint[1:] - Q_goal_joint[1:])
                print('norm of goal-qn', distance)
                print('look here', np.linalg.norm(goal_roll[0]-self.antirouli_config))
                gg.append(distance)
                v += 1
                Xip1 = step[i]*residu * np.dot(np.transpose(Pparal), (Q_goal - Qip1)/np.linalg.norm(Q_goal - Qip1))#distance*step[i] * np.dot(np.transpose(Pparal), (Q_goal - Qip1)/np.linalg.norm(Q_goal - Qip1))
                Xip1_joint = np.zeros((self.nb-1))
                for c in range(1,self.nb):
                    Xip1_joint[c-1] = Q_goal_joint[c] - Qip1_joint[c]
                print('before proj',Xip1_joint)
                if distance<limite:
                    Qip1_neck, Qip1_joint, Qip1_roll, turn_back = self.transmission_on_nullspace_2(Xip1, residu, new_wrench, ( Q_goal - Qip1) / np.linalg.norm(Q_goal - Qip1),True)
                else:
                    Qip1_neck,Qip1_joint, Qip1_roll,turn_back = self.transmission_on_nullspace_2(Xip1, residu, new_wrench, (Q_goal - Qip1)/np.linalg.norm(Q_goal - Qip1),False)
                Qip1 = np.zeros((2 * self.nb+1))
                Qip1[0] =Qip1_neck
                for l in range(self.nb):
                    Qip1[2 * l + 1] = Qip1_joint[l]
                    Qip1[2 * l + 2] = Qip1_roll[l]
                distance_new = np.linalg.norm(Qip1_joint[1:] - Q_goal_joint[1:])
                self.get_neck_frame()
                self.get_list_joint_frame();  self.get_list_roulie_frame(); self.get_list_frame_R02()
                wb = self.get_list_buoyancy_wrench()
                wg = self.gravity_wrench()
                new_wrench = wb + wg.reshape((6, 1))
                print('test1', np.linalg.norm(new_wrench), np.linalg.norm(new_wrench) / Fw_equi)
                Pparal, Pvec, grad = self.parallel_vertical_operation()
                config.append(self.list_of_DH_matrices_joint_in_R0)
                neck_conf = np.r_[neck_conf,np.array([Qip1_neck])]
                joint_conf = np.r_[joint_conf,np.array([Qip1_joint])]
                roll_conf = np.r_[roll_conf, np.array([Qip1_roll])]
                W.append(np.linalg.norm(new_wrench))

                if not turn_back:
                    if 0<=distance-distance_new<residu/2:
                        step[i] = tau[i]*step[i]
                        print('oui')
                    elif distance-distance_new >= residu/2:
                        step[i] = step[i]/tau[i]
                        print('non')
                    else:
                        step[i] = step[i] / (2*tau[i])
                        print('ni oui ni non')
                else:
                    step[i] = step[i]
                if v > 500 :#and lock_roll is True:
                    break
            neck_conf = np.r_[neck_conf, np.array([goal_neck[i][0]])]
            joint_conf = np.r_[joint_conf, np.array([goal_traj[i]])]
            roll_conf = np.r_[roll_conf, np.array([goal_roll[i]])]
        return W, config, gg, joint_conf, roll_conf, neck_conf
    ##########################
    # optimise test
    ####################################
    def find_equilibrum_body_config_3(self,joint_goal,roll_goal,neck_goal,step):
        Q_goal = np.zeros((2 * self.nb + 1))
        Q_goal[0] = neck_goal[0]
        for l in range(self.nb):
            Q_goal[2 * l + 1] = joint_goal[l]
            Q_goal[2 * l + 2] = roll_goal[l]
        distance_body = joint_goal-self.joint_config
        self.get_neck_frame(); self.get_list_joint_frame(); self.get_list_roulie_frame()
        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        last_wrench = wb + wg.reshape((6, 1))
        list_wrench = [np.linalg.norm(last_wrench)]
        print('distance body',np.linalg.norm(distance_body))
        list_joint = copy.copy(self.joint_config);list_roll = copy.copy(self.antirouli_config);list_neck= copy.copy(self.neck_config[1])
        size = 0
        while np.linalg.norm(distance_body) > 5*step:
            size +=1
            if size>192:
                step = step/2
            if size>300:
                break
            # transport on the null space
            Xnp1 = np.zeros((self.nb*2+1))
            Xnp1[0] = neck_goal[0]-self.neck_config[1]
            for j in range(self.nb):
                Xnp1[2 * j + 1] = joint_goal[j]-self.joint_config[j]
                Xnp1[2 * j + 2] = roll_goal[j]-self.antirouli_config[j]

            if np.linalg.norm(distance_body)<0.138:
                dQ = Xnp1/np.linalg.norm(Xnp1)*step/10
                self.neck_config[1] += dQ[0]
                for j in range(self.nb):
                    self.joint_config[j] += dQ[2 * j + 1]
                    self.antirouli_config[j] += dQ[2 * j + 2]
                self.get_neck_frame(); self.get_list_joint_frame(); self.get_list_roulie_frame()
                self.get_list_frame_R02()
            else:
                Pparal, Pvec, grad = self.parallel_vertical_operation()
                Xnp1 = np.dot(np.transpose(Pparal), Xnp1 )#/ np.linalg.norm(Xnp1))
                Xnp1 = np.dot(Pparal, Xnp1)
                Xnp1 = step * Xnp1/np.linalg.norm(Xnp1)

                self.neck_config[1] += Xnp1[0]
                for j in range(self.nb):
                    self.joint_config[j] += Xnp1[2*j+1]
                    self.antirouli_config[j] += Xnp1[2*j+2]
                self.get_neck_frame();self.get_list_joint_frame();self.get_list_roulie_frame()
                self.get_list_frame_R02()

                # retraction
                wb = self.get_list_buoyancy_wrench()
                wg = self.gravity_wrench()
                new_wrench = wb + wg.reshape((6, 1))
                DqW_vec = np.dot(self.grad, self.Pvec)
                if DqW_vec.shape[0] == DqW_vec.shape[1]:
                    invDqW_vec = np.linalg.inv(DqW_vec)
                else:
                    invDqW_vec = np.linalg.pinv(DqW_vec)
                invDqW_vec = -np.dot(self.Pvec, invDqW_vec)
                dW = new_wrench[2:5, 0] - last_wrench[2:5, 0]
                solution = optimize.minimize_scalar(self.retraction, method='brent', args=(dW, invDqW_vec))
                alpha = solution.x
                invDqW_vec_retraction = alpha * invDqW_vec
                dQ = np.dot(invDqW_vec_retraction, dW)
                self.neck_config[1] += dQ[0]
                for j in range(self.nb):
                    self.joint_config[j] += dQ[2 * j + 1]
                    self.antirouli_config[j] += dQ[2 * j + 2]
                print('alpha',alpha)
                self.correct_angle()
                self.get_neck_frame();self.get_list_joint_frame();self.get_list_roulie_frame()
                self.get_list_frame_R02()
            ##### optimize the roll position
            if size<=192:
                Pparal_roll, Pvec_roll,grad_roll = self.tangent_space_of_roll()
                DqW_vec_roll = np.dot(grad_roll, Pvec_roll)
                if DqW_vec_roll.shape[0] == DqW_vec_roll.shape[1]:
                    invDqW_vec_roll = np.linalg.inv(DqW_vec_roll)
                else:
                    invDqW_vec_roll = np.linalg.pinv(DqW_vec_roll)
                invDqW_vec_roll = -np.dot(Pvec_roll, invDqW_vec_roll)
                #dx = (0.05-0.01)/40
                #bound = optimize.Bounds((-0.01-dx*size) * np.ones((self.nb-1)), (0.01+dx*size) * np.ones((self.nb-1)))
                bound = optimize.Bounds((-0.03) * np.ones((self.nb - 1)),(0.03 ) * np.ones((self.nb - 1)))
                sol_opt = optimize.minimize(self.opt_only_roll, x0=np.zeros(self.nb-1), method='SLSQP', bounds=bound, args=(Pparal_roll, invDqW_vec_roll))
                dp = sol_opt.x
                dq = np.dot(Pparal_roll, dp)
                self.neck_config[1] += dq[0];self.joint_config[0] += dq[1]
                for i in range(self.nb):
                    self.antirouli_config[i] += dq[i + 2]
                self.get_neck_frame(); self.get_list_joint_frame(); self.get_list_roulie_frame(); self.get_list_frame_R02()
                wb = self.get_list_buoyancy_wrench()
                wg = self.gravity_wrench()
                dW = wb + wg.reshape((6, 1))
                solution = optimize.minimize_scalar(self.retraction_roll,  method='brent', args=(dW[2:5, 0], invDqW_vec_roll))
                alpha = solution.x
                print('final',alpha)
                dQ = alpha * np.dot(invDqW_vec_roll, dW[2:5, 0])
                self.neck_config[1] += dQ[0]
                self.joint_config[0] += dQ[1]
                for j in range(self.nb):
                    self.antirouli_config[j] += dQ[j + 2]
                print('result',dQ)
                self.correct_angle()
                print('after look',self.antirouli_config)
                self.get_neck_frame();self.get_list_joint_frame();self.get_list_roulie_frame(); self.get_list_frame_R02()
            wb = self.get_list_buoyancy_wrench()
            wg = self.gravity_wrench()
            last_wrench = wb + wg.reshape((6, 1))
            distance_body = joint_goal - self.joint_config
            print('new wrench', np.linalg.norm(last_wrench))
            print('distance body',np.linalg.norm(distance_body))
            list_wrench.append(np.linalg.norm(last_wrench))
            list_joint=np.r_[list_joint,self.joint_config]#;list_joint = np.reshape(list_joint,(-1,6))
            list_roll=np.r_[list_roll,self.antirouli_config]#;list_roll = np.reshape(list_roll,(-1,6))
            list_neck=np.r_[list_neck,self.neck_config[1]]#;list_neck = np.reshape(list_neck,(-1,2))
        return list_wrench,list_joint,list_roll,list_neck
    def opt_non_linear(self):
        Pparal_roll, Pvec_roll,grad_roll = self.tangent_space_of_roll()
        DqW_vec_roll = np.dot(grad_roll, Pvec_roll)
        if DqW_vec_roll.shape[0] == DqW_vec_roll.shape[1]:
            invDqW_vec_roll = np.linalg.inv(DqW_vec_roll)
        else:
            invDqW_vec_roll = np.linalg.pinv(DqW_vec_roll)
        invDqW_vec_roll = -np.dot(Pvec_roll, invDqW_vec_roll)
        #dx = (0.05-0.01)/40
        #bound = optimize.Bounds((-0.01-dx*size) * np.ones((self.nb-1)), (0.01+dx*size) * np.ones((self.nb-1)))
        bound = optimize.Bounds((-0.6) * np.ones((self.nb - 1)),(0.6) * np.ones((self.nb - 1)))
        sol_opt = optimize.minimize(self.opt_only_roll, x0=np.zeros(self.nb-1), method='SLSQP', bounds=bound, args=(Pparal_roll, invDqW_vec_roll))
        dp = sol_opt.x
        dq = np.dot(Pparal_roll, dp)
        self.neck_config[1] += dq[0];self.joint_config[0] += dq[1]
        for i in range(self.nb):
            self.antirouli_config[i] += dq[i + 2]
        self.get_neck_frame(); self.get_list_joint_frame(); self.get_list_roulie_frame(); self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        dW = wb + wg.reshape((6, 1))
        solution = optimize.minimize_scalar(self.retraction_roll,  method='brent', args=(dW[2:5, 0], invDqW_vec_roll))
        alpha = solution.x
        print('final',alpha)
        dQ = alpha * np.dot(invDqW_vec_roll, dW[2:5, 0])
        self.neck_config[1] += dQ[0]
        self.joint_config[0] += dQ[1]
        for j in range(self.nb):
            self.antirouli_config[j] += dQ[j + 2]
        self.correct_angle()
        self.get_neck_frame();self.get_list_joint_frame();self.get_list_roulie_frame(); self.get_list_frame_R02()

        return [self.neck_config[1],self.joint_config[0],self.antirouli_config]
    def tangent_space_of_roll(self):
        grad = self.R0_wrench_jacobian_matrix_of_roll(1e-5)
        u, s, v = np.linalg.svd(grad)
        Pparal = np.transpose(v[3:, :])
        Pvec = np.transpose(v[:3, :])
        return Pparal, Pvec, grad
    def opt_only_roll(self,dp,Pparal,invDqW_vec):
        temp1 = copy.copy(self.antirouli_config); temp2 = copy.copy(self.list_of_DH_matrices_roulie); temp3 = copy.copy(self.list_of_DH_matrices_roulie_in_R0)
        temp4 = copy.copy(self.neck_config); temp5 = copy.copy(self.DH_matrix_neck); temp6 = copy.copy(self.DH_matrix_neck_in_R0)
        temp7 = copy.copy(self.joint_config); temp8 = copy.copy(self.list_of_DH_matrices_joint); temp9 = copy.copy(self.list_of_DH_matrices_joint_in_R0)
        dq = np.dot(Pparal,dp)
        self.neck_config[1] += dq[0]; self.joint_config[0]+=dq[1]
        for i in range(self.nb):
            self.antirouli_config[i]+=dq[i+2]
        self.get_neck_frame();self.get_list_joint_frame();self.get_list_roulie_frame();self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        dW = wb + wg.reshape((6, 1))
        #solution = optimize.minimize(self.retraction_roll,x0=1, method='COBYLA', args=(dW[2:5, 0], invDqW_vec))
        solution = optimize.minimize_scalar(self.retraction_roll, method='brent', args=(dW[2:5, 0], invDqW_vec))
        alpha = solution.x

        dQ = alpha*np.dot(invDqW_vec, dW[2:5, 0])
        self.neck_config[1] += dQ[0]
        self.joint_config[0] += dQ[1]
        for j in range(self.nb):
            self.antirouli_config[j] += dQ[j + 2]
        self.correct_angle()
        self.get_neck_frame();  self.get_list_joint_frame(); self.get_list_roulie_frame(); self.get_list_frame_R02()

        JACOB = self.R0_wrench_jacobian_matrix(1E-4)
        w, v = np.linalg.eig(JACOB)
        self.antirouli_config=copy.copy(temp1); self.list_of_DH_matrices_roulie=copy.copy(temp2); self.list_of_DH_matrices_roulie_in_R0=copy.copy(temp3)
        self.neck_config=copy.copy(temp4);self.DH_matrix_neck=copy.copy(temp5);self.DH_matrix_neck_in_R0=copy.copy(temp6)
        self.joint_config=copy.copy(temp7);self.list_of_DH_matrices_joint=copy.copy(temp8);self.list_of_DH_matrices_joint_in_R0=copy.copy(temp9)
        print('lambda',w[2])
        return w[2]

    def retraction_roll(self, alpha, dW, invDqW_vec):
        temp2 = copy.copy(self.list_of_DH_matrices_joint_in_R0); temp3 = copy.copy(self.list_of_DH_matrices_roulie_in_R0)
        temp4 = copy.copy(self.joint_config); temp5 = copy.copy(self.antirouli_config)
        temp6 = copy.copy(self.neck_config); temp7 = copy.copy(self.DH_matrix_neck_in_R0)
        temp8 = copy.copy(self.list_of_DH_matrices_joint); temp9 = copy.copy(self.list_of_DH_matrices_roulie); temp10 = copy.copy(self.DH_matrix_neck)

        invDqW_vec_retraction = alpha * invDqW_vec
        dQ = np.dot(invDqW_vec_retraction, dW)
        self.neck_config[1] += dQ[0]
        self.joint_config[0] += dQ[1]
        for j in range(self.nb):
            self.antirouli_config[j] += dQ[j + 2]
        self.correct_angle()
        self.get_neck_frame(); self.get_list_joint_frame(); self.get_list_roulie_frame(); self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench = wb + wg.reshape((6, 1))
        self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2)
        self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3)
        self.joint_config = copy.copy(temp4)
        self.antirouli_config = copy.copy(temp5)
        self.neck_config = copy.copy(temp6); self.DH_matrix_neck_in_R0 = copy.copy(temp7)
        self.list_of_DH_matrices_joint = copy.copy(temp8); self.list_of_DH_matrices_roulie = copy.copy(temp9); self.DH_matrix_neck = copy.copy(temp10)
        return np.linalg.norm(new_wrench)
    ######################################################################
    ##########################
    ###another optimise test
    ############################################################
    def optimise_every_step(self,q,r,n):
        nn=np.array([n[0]]);rr=r[0,:];qq = q[0,:]
        self.neck_config = np.array([0., n[0]])
        self.joint_config = q[0, :]
        self.antirouli_config = r[0, :]
        self.get_neck_frame();
        self.get_list_joint_frame();
        self.get_list_roulie_frame();
        self.get_list_frame_R02()
        JACOB = self.R0_wrench_jacobian_matrix(1E-4)
        w, v = np.linalg.eig(JACOB)
        lamb = np.array([w[2]])
        for index in range(1,q.shape[0]):
            self.neck_config = np.array([0., n[index]])
            self.joint_config = q[index, :]
            self.antirouli_config = r[index, :]
            self.get_neck_frame();
            self.get_list_joint_frame();
            self.get_list_roulie_frame();
            self.get_list_frame_R02()
            JACOB = self.R0_wrench_jacobian_matrix(1E-4)
            w, v = np.linalg.eig(JACOB)
            print(v);print('origin',w[2])

            Pparal_roll, Pvec_roll, grad_roll = self.tangent_space_of_roll()
            DqW_vec_roll = np.dot(grad_roll, Pvec_roll)
            if DqW_vec_roll.shape[0] == DqW_vec_roll.shape[1]:
                invDqW_vec_roll = np.linalg.inv(DqW_vec_roll)
            else:
                invDqW_vec_roll = np.linalg.pinv(DqW_vec_roll)
            invDqW_vec_roll = -np.dot(Pvec_roll, invDqW_vec_roll)
            # dx = (0.05-0.01)/40
            # bound = optimize.Bounds((-0.01-dx*size) * np.ones((self.nb-1)), (0.01+dx*size) * np.ones((self.nb-1)))
            bound = optimize.Bounds((-0.1) * np.ones((self.nb - 1)), (0.1) * np.ones((self.nb - 1)))
            sol_opt = optimize.minimize(self.opt_only_roll, x0=np.zeros(self.nb - 1), method='SLSQP', bounds=bound,
                                        args=(Pparal_roll, invDqW_vec_roll))
            dp = sol_opt.x
            dq = np.dot(Pparal_roll, dp)
            self.neck_config[1] += dq[0];
            self.joint_config[0] += dq[1]
            for i in range(self.nb):
                self.antirouli_config[i] += dq[i + 2]
            self.get_neck_frame();
            self.get_list_joint_frame();
            self.get_list_roulie_frame();
            self.get_list_frame_R02()
            wb = self.get_list_buoyancy_wrench()
            wg = self.gravity_wrench()
            dW = wb + wg.reshape((6, 1))
            solution = optimize.minimize_scalar(self.retraction_roll, method='brent',
                                                args=(dW[2:5, 0], invDqW_vec_roll))
            alpha = solution.x
            print('final', alpha)
            dQ = alpha * np.dot(invDqW_vec_roll, dW[2:5, 0])
            self.neck_config[1] += dQ[0]
            self.joint_config[0] += dQ[1]
            for j in range(self.nb):
                self.antirouli_config[j] += dQ[j + 2]
            print('result', dQ)
            self.correct_angle()
            print('after look', self.antirouli_config)

            self.get_neck_frame();
            self.get_list_joint_frame();
            self.get_list_roulie_frame();
            self.get_list_frame_R02()
            JACOB = self.R0_wrench_jacobian_matrix(1E-4)
            w, v = np.linalg.eig(JACOB)
            print(v);print('origin', w[2])
            nn = np.append(nn,self.neck_config[1]);rr=np.r_[rr,self.antirouli_config];qq=np.r_[qq,self.joint_config];lamb=np.append(lamb,w[2])
        return nn,rr,qq,lamb
    #######################################################
    def find_equilibrum_body_config_5(self,joint_goal,roll_goal,neck_goal,step):
        Q_goal = np.zeros((2 * self.nb + 1))
        Q_goal[0] = neck_goal[0]
        for l in range(self.nb):
            Q_goal[2 * l + 1] = joint_goal[l]
            Q_goal[2 * l + 2] = roll_goal[l]
        distance_body = joint_goal-self.joint_config
        self.get_neck_frame(); self.get_list_joint_frame(); self.get_list_roulie_frame()
        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        last_wrench = wb + wg.reshape((6, 1))
        list_wrench = [np.linalg.norm(last_wrench)]
        print('distance body',np.linalg.norm(distance_body))
        list_joint = copy.copy(self.joint_config);list_roll = copy.copy(self.antirouli_config);list_neck = copy.copy(self.neck_config[1])
        size = 0
        ite_max = 5
        while np.linalg.norm(distance_body) > step:
            size +=1
            if size>500:
                break
            # transport on the null space
            Xnp1 = np.zeros((self.nb*2+1))
            Xnp1[0] = neck_goal[0]-self.neck_config[1]
            for j in range(self.nb):
                Xnp1[2 * j + 1] = joint_goal[j]-self.joint_config[j]
                Xnp1[2 * j + 2] = roll_goal[j]-self.antirouli_config[j]

            if np.linalg.norm(distance_body) < 0.42:
                dQ = Xnp1/np.linalg.norm(Xnp1)*step/5
                self.neck_config[1] += dQ[0]
                for j in range(self.nb):
                    self.joint_config[j] += dQ[2 * j + 1]
                    self.antirouli_config[j] += dQ[2 * j + 2]
                self.get_neck_frame(); self.get_list_joint_frame(); self.get_list_roulie_frame()
                self.get_list_frame_R02()
            else:
                Pparal, Pvec, grad = self.parallel_vertical_operation()
                Xnp1 = step*np.dot(np.transpose(Pparal), Xnp1/np.linalg.norm(Xnp1))#/ np.linalg.norm(Xnp1))
                Gram_base = self.Gram_Schmidt(Xnp1)
                bound = optimize.Bounds((-0.001) * np.ones((Xnp1.shape[0]-1)), (0.001) * np.ones((Xnp1.shape[0]-1)))
                if size<30:
                    ite_max+=1
                else:
                    ite_max=500
                    bound = optimize.Bounds((-0.005) * np.ones((Xnp1.shape[0] - 1)),
                                            (0.005) * np.ones((Xnp1.shape[0] - 1)))
                sol_opt = optimize.minimize(self.fast_approximation, x0=np.zeros(Xnp1.shape[0]-1), method='SLSQP', bounds=bound,
                                            args=(Xnp1,Pparal,Gram_base,last_wrench,joint_goal,roll_goal),options={'maxiter': ite_max})
                fac= sol_opt.x
                for i in range(Gram_base.shape[0]):
                    Xnp1 += fac[i] * Gram_base[i, :]
                Xnp1 = np.dot(Pparal,Xnp1)
                self.neck_config[1] += Xnp1[0]
                for j in range(self.nb):
                    self.joint_config[j] += Xnp1[2*j+1]
                    self.antirouli_config[j] += Xnp1[2*j+2]
                self.get_neck_frame();self.get_list_joint_frame();self.get_list_roulie_frame()
                self.get_list_frame_R02()

                # retraction
                wb = self.get_list_buoyancy_wrench()
                wg = self.gravity_wrench()
                new_wrench = wb + wg.reshape((6, 1))
                DqW_vec = np.dot(self.grad, self.Pvec)
                if DqW_vec.shape[0] == DqW_vec.shape[1]:
                    invDqW_vec = np.linalg.inv(DqW_vec)
                else:
                    invDqW_vec = np.linalg.pinv(DqW_vec)
                invDqW_vec = -np.dot(self.Pvec, invDqW_vec)
                dW = new_wrench[2:5, 0] - last_wrench[2:5, 0]
                solution = optimize.minimize_scalar(self.retraction, method='brent', args=(dW, invDqW_vec))
                alpha = solution.x
                invDqW_vec_retraction = alpha * invDqW_vec
                dQ = np.dot(invDqW_vec_retraction, dW)
                self.neck_config[1] += dQ[0]
                for j in range(self.nb):
                    self.joint_config[j] += dQ[2 * j + 1]
                    self.antirouli_config[j] += dQ[2 * j + 2]
                print('look',self.antirouli_config)
                self.correct_angle()
                self.get_neck_frame();self.get_list_joint_frame();self.get_list_roulie_frame()
                self.get_list_frame_R02()
                #####       opt_roll
                Pparal_roll, Pvec_roll, grad_roll = self.tangent_space_of_roll()
                DqW_vec_roll = np.dot(grad_roll, Pvec_roll)
                if DqW_vec_roll.shape[0] == DqW_vec_roll.shape[1]:
                    invDqW_vec_roll = np.linalg.inv(DqW_vec_roll)
                else:
                    invDqW_vec_roll = np.linalg.pinv(DqW_vec_roll)
                invDqW_vec_roll = -np.dot(Pvec_roll, invDqW_vec_roll)
                # dx = (0.05-0.01)/40
                # bound = optimize.Bounds((-0.01-dx*size) * np.ones((self.nb-1)), (0.01+dx*size) * np.ones((self.nb-1)))
                a = 0.01+size*0.05
                bound = optimize.Bounds((-a) * np.ones((self.nb - 1)), (a) * np.ones((self.nb - 1)))
                sol_opt = optimize.minimize(self.opt_only_roll, x0=np.zeros(self.nb - 1), method='SLSQP', bounds=bound,
                                            args=(Pparal_roll, invDqW_vec_roll))
                dp = sol_opt.x
                dq = np.dot(Pparal_roll, dp)
                self.neck_config[1] += dq[0];
                self.joint_config[0] += dq[1]
                for i in range(self.nb):
                    self.antirouli_config[i] += dq[i + 2]
                self.get_neck_frame();
                self.get_list_joint_frame();
                self.get_list_roulie_frame();
                self.get_list_frame_R02()
                wb = self.get_list_buoyancy_wrench()
                wg = self.gravity_wrench()
                dW = wb + wg.reshape((6, 1))
                solution = optimize.minimize_scalar(self.retraction_roll, method='brent',
                                                    args=(dW[2:5, 0], invDqW_vec_roll))
                alpha = solution.x
                print('final', alpha)
                dQ = alpha * np.dot(invDqW_vec_roll, dW[2:5, 0])
                self.neck_config[1] += dQ[0]
                self.joint_config[0] += dQ[1]
                for j in range(self.nb):
                    self.antirouli_config[j] += dQ[j + 2]
                print('result', dQ)
                self.correct_angle()
                print('after look', self.antirouli_config)
                self.get_neck_frame();
                self.get_list_joint_frame();
                self.get_list_roulie_frame();
                self.get_list_frame_R02()
            wb = self.get_list_buoyancy_wrench()
            wg = self.gravity_wrench()
            last_wrench = wb + wg.reshape((6, 1))
            distance_body = joint_goal - self.joint_config
            print('new wrench', np.linalg.norm(last_wrench))
            print('distance body',np.linalg.norm(distance_body))
            list_wrench.append(np.linalg.norm(last_wrench))
            list_joint=np.r_[list_joint,self.joint_config]#;list_joint = np.reshape(list_joint,(-1,6))
            list_roll=np.r_[list_roll,self.antirouli_config]#;list_roll = np.reshape(list_roll,(-1,6))
            list_neck=np.r_[list_neck,self.neck_config[1]]#;list_neck = np.reshape(list_neck,(-1,2))
        return list_wrench,list_joint,list_roll,list_neck
    def fast_approximation(self,fac,Xnp1,Pparal,Gram_base,last_wrench,joint_goal,roll_goal):
        temp1 = copy.copy(self.antirouli_config);
        temp2 = copy.copy(self.list_of_DH_matrices_roulie);
        temp3 = copy.copy(self.list_of_DH_matrices_roulie_in_R0)
        temp4 = copy.copy(self.neck_config);
        temp5 = copy.copy(self.DH_matrix_neck);
        temp6 = copy.copy(self.DH_matrix_neck_in_R0)
        temp7 = copy.copy(self.joint_config);
        temp8 = copy.copy(self.list_of_DH_matrices_joint);
        temp9 = copy.copy(self.list_of_DH_matrices_joint_in_R0)
        for i in range(Gram_base.shape[0]):
            Xnp1 += fac[i] * Gram_base[i, :]
        Xnp1 = np.dot(Pparal, Xnp1)
        self.neck_config[1] += Xnp1[0]
        for j in range(self.nb):
            self.joint_config[j] += Xnp1[2 * j + 1]
            self.antirouli_config[j] += Xnp1[2 * j + 2]
        self.get_neck_frame();self.get_list_joint_frame();self.get_list_roulie_frame()
        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench = wb + wg.reshape((6, 1))
        DqW_vec = np.dot(self.grad, self.Pvec)
        if DqW_vec.shape[0] == DqW_vec.shape[1]:
            invDqW_vec = np.linalg.inv(DqW_vec)
        else:
            invDqW_vec = np.linalg.pinv(DqW_vec)
        invDqW_vec = -np.dot(self.Pvec, invDqW_vec)
        dW = new_wrench[2:5, 0] - last_wrench[2:5, 0]
        solution = optimize.minimize_scalar(self.retraction, method='brent', args=(dW, invDqW_vec))
        alpha = solution.x
        invDqW_vec_retraction = alpha * invDqW_vec
        dQ = np.dot(invDqW_vec_retraction, dW)
        self.neck_config[1] += dQ[0]
        for j in range(self.nb):
            self.joint_config[j] += dQ[2 * j + 1]
            self.antirouli_config[j] += dQ[2 * j + 2]
        distance = joint_goal[1:]-self.joint_config[1:]
        distance = np.append(distance,roll_goal-self.antirouli_config)
        print('look', self.antirouli_config)
        self.correct_angle()
        self.get_neck_frame();
        self.get_list_joint_frame();
        self.get_list_roulie_frame()
        self.get_list_frame_R02()
        self.antirouli_config = copy.copy(temp1); self.list_of_DH_matrices_roulie = copy.copy(temp2); self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3)
        self.neck_config = copy.copy(temp4); self.DH_matrix_neck = copy.copy(temp5);
        self.DH_matrix_neck_in_R0 = copy.copy(temp6)
        self.joint_config = copy.copy(temp7);
        self.list_of_DH_matrices_joint = copy.copy(temp8);
        self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp9)
        print('dis',np.linalg.norm(distance))
        return np.linalg.norm(distance)
    ############################################################################################
    def find_equilibrum_body_config_4(self,joint_goal,roll_goal,neck_goal,step):
        Q_goal = np.zeros((2 * self.nb + 1))
        Q_goal[0] = neck_goal[0]
        for l in range(self.nb):
            Q_goal[2 * l + 1] = joint_goal[l]
            Q_goal[2 * l + 2] = roll_goal[l]
        distance_body = joint_goal-self.joint_config
        self.get_neck_frame(); self.get_list_joint_frame(); self.get_list_roulie_frame()
        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        last_wrench = wb + wg.reshape((6, 1))
        list_wrench = [np.linalg.norm(last_wrench)]
        print('distance body',np.linalg.norm(distance_body))
        list_joint = copy.copy(self.joint_config);list_roll= copy.copy(self.antirouli_config);list_neck= copy.copy(self.neck_config[1])
        size = 0
        while np.linalg.norm(distance_body) > 2.5*step:
            size += 1

            if size > 200:
                break
            # transport on the null space
            Xnp1 = np.zeros((self.nb*2+1))
            Xnp1[0] = neck_goal[0]-self.neck_config[1]
            for j in range(self.nb):
                Xnp1[2 * j + 1] = joint_goal[j]-self.joint_config[j]
                Xnp1[2 * j + 2] = roll_goal[j]-self.antirouli_config[j]
            Pparal, Pvec, grad = self.parallel_vertical_operation()
            Xnp1 = np.dot(np.transpose(Pparal), Xnp1)#/ np.linalg.norm(Xnp1))
            Gram_base = self.Gram_Schmidt(Xnp1)
            DqW_vec = np.dot(self.grad, self.Pvec)
            if DqW_vec.shape[0] == DqW_vec.shape[1]:
                invDqW_vec = np.linalg.inv(DqW_vec)
            else:
                invDqW_vec = np.linalg.pinv(DqW_vec)
            invDqW_vec = -np.dot(self.Pvec, invDqW_vec)
            bound = optimize.Bounds(-0.03 * np.ones((2*self.nb - 3)), 0.03 * np.ones((2*self.nb - 3)))


            sol_opt = optimize.minimize(self.opt_full_body, x0=np.zeros(2*self.nb - 3), method='SLSQP', bounds=bound,
                                        args=(step, Xnp1, Gram_base, Pparal, invDqW_vec,last_wrench))
            x = sol_opt.x
            for i in range(Gram_base.shape[0]):
                Xnp1 += x[i] * Gram_base[i, :]

            Xnp1 = np.dot(Pparal,Xnp1)
            Xnp1 = step * Xnp1/np.linalg.norm(Xnp1)
            self.neck_config[1] += Xnp1[0]
            for j in range(self.nb):
                self.joint_config[j] += Xnp1[2*j+1]
                self.antirouli_config[j] += Xnp1[2*j+2]
            self.get_neck_frame(); self.get_list_joint_frame(); self.get_list_roulie_frame()
            self.get_list_frame_R02()
            # retraction
            wb = self.get_list_buoyancy_wrench()
            wg = self.gravity_wrench()
            new_wrench = wb + wg.reshape((6, 1))

            dW = new_wrench[2:5, 0] - last_wrench[2:5, 0]

            solution = optimize.minimize_scalar(self.retraction,  method='brent', args=(dW, invDqW_vec))
            alpha = solution.x
            invDqW_vec_retraction = alpha * invDqW_vec
            dQ = np.dot(invDqW_vec_retraction, dW)
            self.neck_config[1] += dQ[0]
            for j in range(self.nb):
                self.joint_config[j] += dQ[2 * j + 1]
                self.antirouli_config[j] += dQ[2 * j + 2]
            self.correct_angle()
            wb = self.get_list_buoyancy_wrench()
            wg = self.gravity_wrench()
            last_wrench = wb + wg.reshape((6, 1))
            distance_body = joint_goal - self.joint_config
            JACOB = self.R0_wrench_jacobian_matrix(1E-4)
            w, v = np.linalg.eig(JACOB)
            print('here',w[2])
            print('new wrench', np.linalg.norm(last_wrench))
            print('distance body', np.linalg.norm(distance_body))
            list_wrench.append(np.linalg.norm(last_wrench))
            list_joint = np.r_[list_joint, self.joint_config]  # ;list_joint = np.reshape(list_joint,(-1,6))
            list_roll = np.r_[list_roll, self.antirouli_config]  # ;list_roll = np.reshape(list_roll,(-1,6))
            list_neck = np.r_[list_neck, self.neck_config[1]]  # ;list_neck = np.reshape(list_neck,(-1,2))

        return list_wrench,list_joint,list_roll,list_neck

    def Gram_Schmidt(self,dp):
        dp = dp/np.linalg.norm(dp)
        dim = dp.shape[0]
        Gram_base = np.zeros((dim,dim))
        Gram_base[0,:] = dp
        for i in range(1,dim):
            Gram_base[i,i-1] = 1
        for i in range(dim):
            v = Gram_base[i,:]
            if i !=0:
                for j in range(0,i):
                    v += np.dot(Gram_base[i,:],Gram_base[j,:])*Gram_base[j,:]
                Gram_base[i, :] = v
        Gram_base = Gram_base[1:,:]
        for i in range(dim-1):
            Gram_base[i:, :] =Gram_base[i:, :]/np.linalg.norm(Gram_base[i:, :])
        return Gram_base

    def opt_full_body(self,fac,step, dp_ini, Gram_base, Pparal, invDqW_vec,last_wrench):
        temp1 = copy.copy(self.antirouli_config); temp2 = copy.copy(self.list_of_DH_matrices_roulie); temp3 = copy.copy(self.list_of_DH_matrices_roulie_in_R0)
        temp4 = copy.copy(self.neck_config); temp5 = copy.copy(self.DH_matrix_neck); temp6 = copy.copy(self.DH_matrix_neck_in_R0)
        temp7 = copy.copy(self.joint_config); temp8 = copy.copy(self.list_of_DH_matrices_joint); temp9 = copy.copy(self.list_of_DH_matrices_joint_in_R0)
        print('fac',fac)
        for i in range(Gram_base.shape[0]):
            dp_ini += fac[i] * Gram_base[i,:]
        dq = np.dot(Pparal, dp_ini)
        dq = step * dq/ np.linalg.norm(dq)
        self.neck_config[1] += dq[0]
        for i in range(self.nb):
            self.joint_config[i] += dq[2 * i + 1]
            self.antirouli_config[i] += dq[2 * i + 2]
        self.get_neck_frame(); self.get_list_joint_frame(); self.get_list_roulie_frame(); self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench = wb + wg.reshape((6, 1))
        dW = new_wrench[2:5, 0] - last_wrench[2:5, 0]

        # solution = optimize.minimize(self.retraction_roll,x0=1, method='COBYLA', args=(dW[2:5, 0], invDqW_vec))
        solution = optimize.minimize_scalar(self.retraction, method='brent', args=(dW, invDqW_vec))
        alpha = solution.x
        print('x', alpha)
        print('w', solution.fun)
        dQ = alpha * np.dot(invDqW_vec, dW)
        self.neck_config[1] += dQ[0]
        for j in range(self.nb):
            self.joint_config[j] += dQ[2 * j + 1]
            self.antirouli_config[j] += dQ[2 * j + 2]
        self.correct_angle()

        self.get_neck_frame(); self.get_list_joint_frame(); self.get_list_roulie_frame(); self.get_list_frame_R02()
        JACOB = self.R0_wrench_jacobian_matrix(1E-4)
        w, v = np.linalg.eig(JACOB)
        print('lambda', w[2])
        self.antirouli_config = copy.copy(temp1); self.list_of_DH_matrices_roulie = copy.copy(temp2); self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3)
        self.neck_config = copy.copy(temp4);
        self.DH_matrix_neck = copy.copy(temp5);
        self.DH_matrix_neck_in_R0 = copy.copy(temp6)
        self.joint_config = copy.copy(temp7);
        self.list_of_DH_matrices_joint = copy.copy(temp8);
        self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp9)
        return w[2]
    def displacement_head_body(self,joint_goal,roll_goal,neck_goal,z_goal,step):
        Q_goal = np.zeros((2 * self.nb + 1))
        Q_goal[0] = neck_goal[0]
        for l in range(self.nb):
            Q_goal[2 * l + 1] = joint_goal[l]
            Q_goal[2 * l + 2] = roll_goal[l]
        distance_body = joint_goal-self.joint_config
        distance_head = (z_goal[2, 3]-self.list_of_DH_matrices_head[2,3])
        self.get_neck_frame();self.get_list_joint_frame();self.get_list_roulie_frame()
        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        last_wrench = wb + wg.reshape((6, 1))
        list_wrench = [np.linalg.norm(last_wrench)]
        dis_head = [distance_head];dis_body = [np.linalg.norm(distance_body)]
        print('distance_head',distance_head)
        print('distance body',np.linalg.norm(distance_body))
        list_head = copy.copy(self.list_of_DH_matrices_head);list_joint = copy.copy(self.joint_config);list_roll= copy.copy(self.antirouli_config);list_neck= copy.copy(self.neck_config)
        while np.linalg.norm(distance_body) > 5*step and abs(distance_head)>step/30:
            '''
            Q_goal = np.zeros((2 * self.nb + 1))
            Q_goal[0] = neck_goal[0]
            for l in range(self.nb):
                Q_goal[2 * l + 1] = joint_goal[l]
                Q_goal[2 * l + 2] = roll_goal[l]            
            distance_body = joint_goal - self.neck_config[1]
            distance_head = abs(self.list_of_DH_matrices_head[2, 3] - z_goal[2, 3])
            '''
            Xnp1 = np.zeros((self.nb*2+1))
            Xnp1[0] = neck_goal[0]-self.neck_config[1]
            for j in range(self.nb):
                Xnp1[2 * j + 1] = joint_goal[j]-self.joint_config[j]
                Xnp1[2 * j + 2] = roll_goal[j]-self.antirouli_config[j]
            Pparal, Pvec, grad = self.parallel_vertical_operation()
            Dpsi_W = self.R0_wrench_jacobian_matrix(1E-4)
            Xnp1 =  np.dot(np.transpose(Pparal), Xnp1 )#/ np.linalg.norm(Xnp1))
            Xnp1 = np.dot(Pparal,Xnp1)
            Xnp1 = step * Xnp1/np.linalg.norm(Xnp1)
            A_g0_q_T = np.dot(np.linalg.pinv(grad),Dpsi_W)
            Xnp1_head = -np.dot(A_g0_q_T,np.array([distance_head/abs(distance_head)*step/50,0,0]))
            self.neck_config[1] += Xnp1[0]
            for j in range(self.nb):
                self.joint_config[j] += Xnp1[2*j+1]
                self.antirouli_config[j] += Xnp1[2*j+2]
            self.get_neck_frame();
            self.get_list_joint_frame();
            self.get_list_roulie_frame()
            self.get_list_frame_R02()
            wb = self.get_list_buoyancy_wrench()
            wg = self.gravity_wrench()
            new_wrench = wb + wg.reshape((6, 1))
            DqW_vec = np.dot(self.grad, self.Pvec)
            if DqW_vec.shape[0] == DqW_vec.shape[1]:
                invDqW_vec = np.linalg.inv(DqW_vec)
            else:
                invDqW_vec = np.linalg.pinv(DqW_vec)
            invDqW_vec = -np.dot(self.Pvec, invDqW_vec)
            dW = new_wrench[2:5, 0] - last_wrench[2:5, 0]

            solution = optimize.minimize(self.retraction, x0=1, method='COBYLA', args=(dW, invDqW_vec))
            alpha = solution.x
            invDqW_vec_retraction = alpha * invDqW_vec
            dQ = np.dot(invDqW_vec_retraction, dW)
            self.neck_config[1] += dQ[0]
            for j in range(self.nb):
                self.joint_config[j] += dQ[2 * j + 1]
                self.antirouli_config[j] += dQ[2 * j + 2]
            self.correct_angle()
            if abs(distance_head) > step / 30:
                self.neck_config[1] += Xnp1_head[0]
                for j in range(self.nb):
                    self.joint_config[j] +=Xnp1_head[2*j+1]
                    self.antirouli_config[j] += Xnp1_head[2*j+2]
                self.get_neck_frame()
                self.get_list_joint_frame();
                self.get_list_roulie_frame();
                self.get_list_frame_R02()
                self.find_equilibrum_head_config_test(Newton=True, tau=1E-5)
            wb = self.get_list_buoyancy_wrench()
            wg = self.gravity_wrench()
            last_wrench = wb + wg.reshape((6, 1))
            distance_body = joint_goal - self.joint_config
            distance_head = (z_goal[2, 3]-self.list_of_DH_matrices_head[2, 3] )
            print('distance_head',distance_head)
            print('distance body',np.linalg.norm(distance_body))
            list_wrench.append(np.linalg.norm(last_wrench))
            dis_body.append(distance_body); dis_head.append(np.linalg.norm(distance_head))
            list_head=np.r_[list_head,self.list_of_DH_matrices_head];
            list_joint=np.r_[list_joint,self.joint_config]#;list_joint = np.reshape(list_joint,(-1,6))
            list_roll=np.r_[list_roll,self.antirouli_config]#;list_roll = np.reshape(list_roll,(-1,6))
            list_neck=np.r_[list_neck,self.neck_config]#;list_neck = np.reshape(list_neck,(-1,2))
        return dis_head,dis_body,list_wrench,list_head,list_joint,list_roll,list_neck

    def third_eigenvalue_GA(self,q1,q2,q3,q4,q5,q6,q7,q8):
        self.neck_config[1] = q1
        self.joint_config[0] = q2
        self.antirouli_config[0] = q3
        self.antirouli_config[1] = q4
        self.antirouli_config[2] = q5
        self.antirouli_config[3] = q6
        self.antirouli_config[4] = q7
        self.antirouli_config[5] = q8
        self.get_list_joint_frame(); self.get_neck_frame();self.get_list_roulie_frame(); self.get_list_frame_R02()
        Grad = self.R0_wrench_jacobian_matrix(1E-4)
        w, v = np.linalg.eig(Grad)
        Fitness = w[2] #+ PEN
        print('Fitness',w[2])#,np.linalg.norm(new_wrench) / Fg,PEN,Fitness)
        return Fitness
    def third_eigenvalue_scipy(self,q):
        self.neck_config[1] = q[0]
        self.joint_config[0] = q[1]
        self.antirouli_config[0] = q[2]
        self.antirouli_config[1] = q[3]
        self.antirouli_config[2] = q[4]
        self.antirouli_config[3] = q[5]
        self.antirouli_config[4] = q[6]
        self.antirouli_config[5] = q[7]
        self.get_list_joint_frame(); self.get_neck_frame();self.get_list_roulie_frame(); self.get_list_frame_R02()
        Grad = self.R0_wrench_jacobian_matrix(1E-4)
        w, v = np.linalg.eig(Grad)
        Fitness = w[2] #+ PEN
        print('Fitness',w[2])#,np.linalg.norm(new_wrench) / Fg,PEN,Fitness)
        return Fitness
    def third_eigenvalue(self,q_roll=None):
        if q_roll is not None:
            self.antirouli_config = q_roll
            self.get_list_roulie_frame(); self.get_list_frame_R02()
        Grad = self.R0_wrench_jacobian_matrix(1E-4)
        w, v = np.linalg.eig(Grad)
        return w[2]
    def wrench_norm(self,q1,q2,q3,q4,q5,q6,q7,q8):
        self.antirouli_config = np.array([q3,q4,q5,q6,q7,q8])
        self.neck_config[1] = q1
        self.joint_config[0]= q2
        self.get_list_roulie_frame();self.get_list_joint_frame();self.get_neck_frame();self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench = wb + wg.reshape((6, 1))
        print('W', np.linalg.norm(new_wrench))
        return np.linalg.norm(new_wrench)
    def wrench_norm2(self,q_roll=None):
        if q_roll is not None:
            self.antirouli_config[1] = q_roll
            self.get_list_roulie_frame();
            self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench = wb + wg.reshape((6, 1))
        return np.linalg.norm(new_wrench)


    def find_stable_config_by_GA(self):
        lb = np.array([-0.58719658,  0.73756285])-20 / 180 * pi * np.ones((2));
        #lb[0] = lb[1] = -30 / 180 * pi
        ub = np.array([-0.58719658,  0.73756285])+20 / 180 * pi * np.ones((2));
        #ub[0] = ub[1] = 30 / 180 * pi
        from sko.GA import GA, GA_TSP
        self.ga = GA(func=self.wrench_norm, n_dim=2, size_pop=100, max_iter=500,prob_mut=0.05,
                lb=lb, ub=ub,precision=1e-7)
        best_roll, best_eigenvalue = self.ga.run()

        import pandas as pd
        Y_history = pd.DataFrame(self.ga.all_history_Y)
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
        ax[0].set_xlabel('number of iteration');ax[0].set_ylabel('fitness func of generation')

        Y_history.min(axis=1).cummin().plot(kind='line')
        ax[1].set_xlabel('number of iteration');
        ax[1].set_ylabel('best fitness func of generation')
        plt.show()

        return best_roll, best_eigenvalue
    def find_stable_roll_by_scipy(self):
        from scipy.optimize import differential_evolution
        from scipy.optimize import NonlinearConstraint, Bounds

        # the sum of x[0] and x[1] must be less than 1.9
        constraint1 = lambda q: self.wrench_norm(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7])
        nlc = NonlinearConstraint(self.third_eigenvalue_scipy, -np.inf, 0)
        nlc2 = NonlinearConstraint(constraint1, 0, 1)
        # specify limits using a `Bounds` object.

        bounds =[(-15/180*pi, 15/180*pi), (-15/180*pi, 15/180*pi),(-45/180*pi, 45/180*pi),(-45/180*pi, 45/180*pi),(-45/180*pi, 45/180*pi),(-45/180*pi, 45/180*pi),(-45/180*pi, 45/180*pi),(-45/180*pi, 45/180*pi)]

        result = differential_evolution(self.third_eigenvalue_scipy, bounds, constraints=(nlc,nlc2),maxiter=150,popsize=100)

        return result.x, result.fun
    def find_stable_roll_by_GA(self):
        from sko.GA import GA, GA_TSP

        constraint1 = lambda q:  self.wrench_norm(q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7])-1
        constraint2 = lambda q: self.third_eigenvalue_GA(q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7])
        l = np.array([-0.08142043,0.09238805,-0.6354549 , -0.58434368, -0.60968704 ,-0.67218272,-0.28929532 , 0.14083024])
        '''
        lb = l-np.array([0.05,0.05,0.2,0.2,0.2,0.2,0.2,0.2])
        ub = l + np.array([0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        '''
        lb = -45 / 180 * pi * np.ones((self.nb + 2)); lb[0]=lb[1]=-15/180*pi
        ub=45/180*pi*np.ones((self.nb+2)); ub[0]=ub[1]=15/180*pi

        self.ga = GA(func=self.third_eigenvalue_GA, n_dim=self.nb+2, size_pop=100, max_iter=200,prob_mut=0.05,
                lb=lb, ub=ub,precision=1e-7,
                constraint_ueq=[constraint1,constraint2])
        best_roll, best_eigenvalue = self.ga.run()

        import pandas as pd
        Y_history = pd.DataFrame(self.ga.all_history_Y)
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
        ax[0].set_xlabel('number of iteration');ax[0].set_ylabel('fitness func of generation')

        Y_history.min(axis=1).cummin().plot(kind='line')
        ax[1].set_xlabel('number of iteration');
        ax[1].set_ylabel('best fitness func of generation')
        plt.show()

        return best_roll, best_eigenvalue
    '''
    this part for find equilibrum neck and first joint configuration
    '''
    def neck_wrench(self,q):
        self.neck_config[1] = q[0]
        self.joint_config[0] = q[1]
        self.get_neck_frame();
        self.get_list_joint_frame();
        self.get_list_roulie_frame();
        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench = np.linalg.norm(wb + wg.reshape((6, 1)))
        return np.linalg.norm(new_wrench)

    def find_stable_neck(self):
        bound = optimize.Bounds(-0.5*np.ones((2)),0.5*np.ones((2)))
        solution = optimize.minimize(self.neck_wrench, x0=np.array([0,0]), method='SLSQP',bounds=bound)
        return solution
    '''
    def find_stable_neck(self):
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        new_wrench = np.linalg.norm(wb + wg.reshape((6, 1)))
        alpha = 0.5
        c1 = 1E-2
        while new_wrench > 1E-6:
            temp2 = copy.copy(self.list_of_DH_matrices_joint_in_R0);
            temp3 = copy.copy(self.list_of_DH_matrices_roulie_in_R0)
            temp4 = copy.copy(self.joint_config);
            temp6 = copy.copy(self.neck_config)
            temp7 = copy.copy(self.DH_matrix_neck_in_R0)
            q0 = np.array([self.neck_config[1],self.joint_config[0]])
            Dq = self.neck_grad()
            print('grad',Dq)
            q0 = q0 - alpha*Dq

            self.neck_config[1], self.joint_config[0] = q0[0]%(2*pi),q0[1]%(2*pi)
            print('dq',  self.neck_config[1], self.joint_config[0])
            self.get_list_joint_frame(); self.get_neck_frame()
            self.get_list_frame_R02()
            wb = self.get_list_buoyancy_wrench()
            wg = self.gravity_wrench()
            old_wrench = new_wrench
            new_wrench = np.linalg.norm(wb + wg.reshape((6, 1)))
            while new_wrench-old_wrench>- c1*alpha*np.linalg.norm(Dq):
                self.DH_matrix_neck_in_R0 = copy.copy(temp7);
                self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3)
                self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2);
                self.neck_config = copy.copy(temp6)
                self.joint_config = copy.copy(temp4)
                alpha = alpha*0.8
                #q0 = np.array([self.neck_config[1], self.joint_config[0]])
                #Dq = self.neck_grad()
                q0 = np.array([self.neck_config[1], self.joint_config[0]])
                q0 = q0 - alpha * Dq
                self.neck_config[1], self.joint_config[0] = q0[0]%(2*pi), q0[1]%(2*pi)
                print('dq-arm', self.neck_config[1], self.joint_config[0])
                self.get_list_joint_frame();
                self.get_neck_frame()
                self.get_list_frame_R02()
                wb = self.get_list_buoyancy_wrench()
                wg = self.gravity_wrench()
                new_wrench = np.linalg.norm(wb + wg.reshape((6, 1)))
            if alpha<1E-3:
                alpha = 0.05
            print('get out',new_wrench)
        return q0
    '''
    def neck_grad(self):
        gradient = np.zeros((2))
        temp2 = copy.copy(self.list_of_DH_matrices_joint_in_R0); temp3 = copy.copy(self.list_of_DH_matrices_roulie_in_R0)
        temp4 = copy.copy(self.joint_config);  temp6 = copy.copy(self.neck_config)
        temp7 = copy.copy(self.DH_matrix_neck_in_R0)
        self.neck_config[1] += 0.0001
        self.get_neck_frame();self.get_list_joint_frame();
        self.get_list_roulie_frame(); self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        w_forward = np.linalg.norm(wb + wg.reshape((6, 1)))
        self.neck_config[1] -= 0.0002
        self.get_neck_frame();self.get_list_joint_frame();
        self.get_list_roulie_frame();
        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        w_backward = np.linalg.norm(wb + wg.reshape((6, 1)))
        gradient[0] = (w_forward-w_backward)/0.0002
        self.DH_matrix_neck_in_R0 = copy.copy(temp7);self.list_of_DH_matrices_roulie_in_R0=copy.copy(temp3)
        self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2); self.neck_config=copy.copy(temp6)

        self.joint_config[0]  += 0.0001
        self.get_list_joint_frame();
        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        w_forward = np.linalg.norm(wb + wg.reshape((6, 1)))
        self.joint_config[0] -= 0.0002
        self.get_list_joint_frame();
        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        w_backward = np.linalg.norm(wb + wg.reshape((6, 1)))
        gradient[1] = (w_forward - w_backward) / 0.0002
        self.DH_matrix_neck_in_R0 = copy.copy(temp7);
        self.list_of_DH_matrices_roulie_in_R0 = copy.copy(temp3)
        self.list_of_DH_matrices_joint_in_R0 = copy.copy(temp2);
        self.joint_config = copy.copy(temp4)
        return gradient

    def stiffness_optimisation(self):
        Fw_equi = self.M * self.g
        #Wrench_contraint = optimize.NonlinearConstraint(self.wrench_norm,0,Fw_equi*1E-3)
        #Angle_contraint = optimize.LinearConstraint(np.eye(self.nb),-45/180*pi*np.ones(self.nb),45/180*pi*np.ones(self.nb))
        constraint1 = lambda q: Fw_equi*1E-3 -self.wrench_norm(q)
        constraint2 = lambda q: 45/180*pi*np.ones(self.nb)+np.dot(np.eye(self.nb),q)
        constraint3 = lambda q: 45 / 180 * pi * np.ones(self.nb) - np.dot(np.eye(self.nb), q)
        Contraint_dic = [{'type':'ineq','fun':constraint1},{'type':'ineq','fun':constraint2},{'type':'ineq','fun':constraint3}]
        q_roll = optimize.minimize(self.third_eigenvalue,x0=self.antirouli_config,method='COBYLA',constraints=Contraint_dic)
        return q_roll
    def test_eigenvalue_jacobian_matrix(self,joint_conf,roll_conf,neck_config):
        w1 = []; w2 = []; w3 = []
        w1_ = []; w2_ = []; w3_ = []
        for i in range(joint_conf.shape[0]):
            self.joint_config = joint_conf[i,:]
            self.neck_config[1] = neck_config[i]
            print(self.neck_config[1],neck_config[i])
            self.antirouli_config = roll_conf[i,:]
            self.get_list_joint_frame();
            self.get_neck_frame()
            self.get_list_roulie_frame();
            self.get_list_frame_R02()
            Grad = self.R0_wrench_jacobian_matrix(1E-4)
            w, v = np.linalg.eig(Grad)
            w1_.append(w[0]); w2_.append(w[1]); w3_.append(w[2])
        fig, axs = plt.subplots(3, sharex=True, sharey=False, gridspec_kw={'hspace': 0})
        '''
        fig.suptitle('eigenvalue of jacobian matrix')
        #axs[0].plot(w1)
        axs[0].plot(w1_)
        axs[0].set_xlabel('number of iteration')
        axs[0].set_ylabel('eigenvalue 1')
        #axs[1].plot(w2)
        axs[1].plot(w2_)
        axs[1].set_xlabel('number of iteration')
        axs[1].set_ylabel('eigenvalue 2')
        #axs[2].plot(w3)
        axs[2].plot(w3_)
        axs[2].set_xlabel('number of iteration')
        axs[2].set_ylabel('eigenvalue 3')
        '''
        return w3_

    def illustration_body_shape(self,joint_conf,roll_conf,lock_roll,goal=None,
                                goal_roll=None,list_joint=None,list_roll=None):
        fig, axs = plt.subplots(self.nb, sharex=True, sharey=False, gridspec_kw={'hspace': 0})
        #fig.suptitle(t='Revolute joint angle',y=0.95,fontsize=12)
        '''
        for i in range(self.nb):
            axs[i].plot(joint_conf[:,i])
            axs[i].plot(goal[0][i]*np.ones((joint_conf.shape[0])),label='goal joint angle')
            axs[i].set_xlabel('number of iteration')
            axs[i].set_ylabel(list_joint[i], fontsize=8)
            axs[i].legend(fontsize=5)
        '''
        for i in range(self.nb):
            plt.subplot(6,1,i+1)
            plt.plot(joint_conf[:,i])
            if goal is not None:
                plt.plot(goal[0][i] * np.ones((joint_conf.shape[0])), label='goal joint angle')
            plt.xlabel('number of iterations')
            plt.ylim(-0.8, 0.8)
            plt.ylabel(list_joint[i], fontsize=6)
            plt.legend(fontsize=5)
            plt.yticks(fontsize=6)
        if not lock_roll:
            fig2, axs2 = plt.subplots(self.nb, sharex=True, sharey=True, gridspec_kw={'hspace': 0})

            #fig2.suptitle(t='Rolling angle',y=0.95,fontsize=12)
            for i in range(self.nb):
                plt.subplot(6, 1, i + 1)
                plt.plot(roll_conf[:, i])
                if goal_roll is not None:
                       plt.plot(goal_roll[0][i] * np.ones((roll_conf.shape[0])), label='goal rolling angle')
                plt.xlabel('number of iterations')
                plt.ylabel(list_roll[i], fontsize=6)
                plt.legend(fontsize=5)

                plt.ylim(-1,1)
                plt.yticks(fontsize=6)
            '''
            for i in range(self.nb):
                axs2[i].plot(roll_conf[:, i])
                axs2[i].plot(goal_roll[0][i] * np.ones((joint_conf.shape[0])), label='goal rolling angle')
                axs2[i].set_xlabel('number of iteration')
                axs2[i].set_ylabel(list_roll[i],fontsize=8)
                axs2[i].legend(fontsize=5)
            '''
        '''
        axs[1].plot(joint_conf[:,1])
        axs[2].plot(joint_conf[:,2])
        axs[3].plot(joint_conf[:, 3])
        axs[4].plot(joint_conf[:, 4])
        axs[5].plot(joint_conf[:, 5])
        '''

    def trace_body_manifold(self,config_body, n):
        '''
        :param config_body: list of joints configurations
        :return:
        '''
        fig = plt.figure()
        ax = Axes3D(fig)

        for s in range(int(len(config_body)/n)):
            i = s*n
            config_i = config_body[i]
            X=[self.list_of_DH_matrices_head[0,3]];Y = [self.list_of_DH_matrices_head[1,3]];Z=[self.list_of_DH_matrices_head[2,3]]

            for j in range(self.nb):
                g = config_i[:,4*j:4*j+4]
                X.append(g[0,3]);Y.append(g[1,3]);Z.append(g[2,3])
                if j == self.nb-1:
                    temp = np.eye(4); temp[0, 3] = self.P
                    gs = np.dot(g,temp)
                    X.append(gs[0, 3]);
                    Y.append(gs[1, 3]);
                    Z.append(gs[2, 3])
            if i == 0:

                ax.plot(X, Y, Z, label='initial body shape', marker='o')
            elif i == 1:
                ax.plot(X,Y,Z,label='second body shape', marker='v')

            elif i == len(config_body)-1:
                ax.plot(X, Y, Z, label='last body shape',marker='*')
            else:
                ax.plot(X, Y, Z)
        ax.set_xlim(0, 10)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-1, 1)
        ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
        ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
        ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
        ax.legend()
        #plt.show()
    def adjoint_matrix(self, g):
        Adj = np.zeros((6, 6))
        Adj[:3, :3] = g[:3, :3];
        Adj[3:6, 3:6] = g[:3, :3]
        P_hat = self.skew_symetric_matrix(g[:3, 3])
        Adj[3:6, :3] = np.dot(P_hat, g[:3, :3])
        return Adj



    def DH_matrix(self, r=None, beta=None,c=None, p=np.empty((3))):
        '''
        :param r: euler rotation angle around axe z
        :param beta: euler rotation angle around axe x
        :param p: position of frame
        :return: transformation matrix between two frames
        '''
        Rot = np.zeros((3,3))
        if r != None and beta == None:
            Rot = np.array([[cos(r), -sin(r), 0.],
                            [sin(r), cos(r), 0.],
                            [0., 0., 1.]])
        elif r == None and beta != None:
            Rot = np.array([[cos(beta), 0, sin(beta)],
                            [0., 1., 0.],
                            [-sin(beta),0, cos(beta)]])
        elif r != None and beta != None:
            temp1 = np.array([[cos(r), -sin(r), 0], [sin(r), cos(r), 0], [0, 0, 1]])
            temp2 = np.array([[cos(beta), 0, sin(beta)], [0., 1., 0.], [-sin(beta),0, cos(beta)]])
            Rot = np.dot(temp1, temp2)
        elif c != None:
            Rot = np.array([[1., 0, 0],
                            [0., cos(c), -sin(c)],
                            [0., sin(c), cos(c)]])
        else:
            AssertionError('Please input the angle of rotation')
        DH_M = np.zeros((4, 4))
        DH_M[0:3, 0:3] = Rot
        DH_M[0:3, 3] = p
        DH_M[3, :] = np.array([0, 0, 0, 1])
        return DH_M

    def skew_symetric_matrix(self, M=np.array([])):
        if not len(M.shape) == 1:
            raise AssertionError('it is not a vector')
        dimension = M.shape[0]
        if dimension == 3:
            M1 = M[0]
            M2 = M[1]
            M3 = M[2]
            Mx = np.array([[0, -M3, M2],
                           [M3, 0, -M1],
                           [-M2, M1, 0]])
        else:
            raise AssertionError('it must be a 3*1 vector')
        return Mx

    def get_neck_frame(self):
        if not self.neck_config.shape[0] == 2:
            raise AssertionError('There must be 2 degrees of freedom ')
        self.DH_matrix_neck = self.DH_matrix(r=self.neck_config[0],beta=self.neck_config[1],p=np.array([self.P_head, 0, 0]))
        return self.DH_matrix_neck

    def get_list_joint_frame(self):  ## list of DH matrices of joint segments { T12, T23, ..., Tn-1n}
        if not self.joint_config.shape[0] == self.nb:
            raise AssertionError('wrong dimension of joint configuration')
        self.list_of_DH_matrices_joint = np.empty((4, 4 * self.nb))
        self.list_of_DH_matrices_joint[:, 0:4] = self.DH_matrix(beta=self.joint_config[0], p=np.array([self.P_neck, 0, 0]))
        for j in range(1, self.nb):
            temp1 = self.DH_matrix(r=self.joint_config[j], p=np.array([self.P, 0, 0]))
            self.list_of_DH_matrices_joint[:, 4 * j:4 * j + 4] = temp1
        return self.list_of_DH_matrices_joint

    def get_list_roulie_frame(self):  ## list of DH matrices of joint segments { T22', T33', ..., Tnn'}
        if not self.antirouli_config.shape[0] == self.nb:
            raise AssertionError('wrong dimension of antiroulie configuration')
        self.list_of_DH_matrices_roulie = np.empty((4, 4 * self.nb))
        for j in range(0, self.nb):
            temp1 = self.DH_matrix(c=self.antirouli_config[j], p=np.array([self.P_roll, 0, 0]))
            self.list_of_DH_matrices_roulie[:, 4 * j:4 * j + 4] = temp1
        return self.list_of_DH_matrices_roulie

    def get_list_frame_R02(self):
        ## cette fonction calcule les matrices DH T0,i
        if self.list_of_DH_matrices_joint is None:
            self.get_list_joint_frame()
        else:
            self.list_of_DH_matrices_joint = self.list_of_DH_matrices_joint
        if self.list_of_DH_matrices_roulie is None:
            self.get_list_roulie_frame()
        else:
            self.list_of_DH_matrices_roulie = self.list_of_DH_matrices_roulie
        if self.DH_matrix_neck is None:
            self.get_neck_frame()
        else:
            self.DH_matrix_neck = self.DH_matrix_neck
        self.list_of_DH_matrices_joint_in_R0 = np.empty((4, 4 * self.nb))
        self.list_of_DH_matrices_roulie_in_R0 = np.empty((4, 4 * self.nb))
        T01 = np.dot(self.list_of_DH_matrices_head,self.DH_matrix_neck)
        self.DH_matrix_neck_in_R0 = T01
        T12 = self.list_of_DH_matrices_joint[:, 0:4]
        T02 = np.dot(T01, T12)
        self.list_of_DH_matrices_joint_in_R0[:, 0:4] = T02
        T22rouli = self.list_of_DH_matrices_roulie[:, 0:4]
        T02rouli = np.dot(T02, T22rouli)
        self.list_of_DH_matrices_roulie_in_R0[:, 0:4] = T02rouli
        for j in range(1, self.nb):
            Tjjp1 = self.list_of_DH_matrices_joint[:, 4 * j:4 * j + 4]
            T0j = self.list_of_DH_matrices_joint_in_R0[:, 4 * j - 4:4 * j]
            T0jp1 = np.dot(T0j, Tjjp1)
            self.list_of_DH_matrices_joint_in_R0[:, 4 * j:4 * j + 4] = T0jp1
            Tjp1jp1_rouli = self.list_of_DH_matrices_roulie[:, 4 * j:4 * j + 4]
            T0jp1_rouli = np.dot(T0jp1, Tjp1jp1_rouli)
            self.list_of_DH_matrices_roulie_in_R0[:, 4 * j:4 * j + 4] = T0jp1_rouli
        return self.DH_matrix_neck_in_R0,self.list_of_DH_matrices_joint_in_R0, self.list_of_DH_matrices_roulie_in_R0

    def get_list_frame_R0(self, dt):
        ## cette fonction calcule les matrices DH T0,i
        if not self.head_pos.shape[0] == self.joint_config.shape[0] == self.antirouli_config.shape[0]:
            print(self.head_pos.shape[0], self.joint_config.shape[0], self.antirouli_config.shape[0])
            raise AssertionError('number of samples incorrect!!!')
        if self.list_of_DH_matrices_joint is None:
            self.get_list_joint_frame()
        else:
            self.list_of_DH_matrices_joint = self.list_of_DH_matrices_joint
        if self.list_of_DH_matrices_roulie is None:
            self.get_list_roulie_frame()
        else:
            self.list_of_DH_matrices_roulie = self.list_of_DH_matrices_roulie
        nb_sample = self.joint_config.shape[0]
        self.list_of_DH_matrices_in_R0 = np.empty((4, 4 * (2 * self.nb + 1), nb_sample))
        self.list_of_DH_matrices_joint_in_R0 = np.empty((4, 4 * self.nb, nb_sample))
        self.list_of_DH_matrices_roulie_in_R0 = np.empty((4, 4 * self.nb, nb_sample))
        for i in range(0, nb_sample):
            T01 = self.list_of_DH_matrices_head[:, :, i]
            T12 = self.list_of_DH_matrices_joint[:, 0:4, i]
            T02 = np.dot(T01, T12)
            self.list_of_DH_matrices_joint_in_R0[:, 0:4, i] = T02
            T22rouli = self.list_of_DH_matrices_roulie[:, 0:4, i]
            T02rouli = np.dot(T02, T22rouli)
            self.list_of_DH_matrices_roulie_in_R0[:, 0:4, i] = T02rouli
            for j in range(1, self.nb):
                Tjjp1 = self.list_of_DH_matrices_joint[:, 4 * j:4 * j + 4, i]
                T0j = self.list_of_DH_matrices_joint_in_R0[:, 4 * j - 4:4 * j, i]
                T0jp1 = np.dot(T0j, Tjjp1)
                self.list_of_DH_matrices_joint_in_R0[:, 4 * j:4 * j + 4, i] = T0jp1
                Tjp1jp1_rouli = self.list_of_DH_matrices_roulie[:, 4 * j:4 * j + 4, i]
                T0jp1_rouli = np.dot(T0jp1, Tjp1jp1_rouli)
                self.list_of_DH_matrices_roulie_in_R0[:, 4 * j:4 * j + 4, i] = T0jp1_rouli
        return self.list_of_DH_matrices_head, self.list_of_DH_matrices_joint_in_R0, \
               self.list_of_DH_matrices_roulie_in_R0


    def Dichotomy(self, g, s_g, s_d,lg,ld,circular):
        ds = abs(s_g - s_d)
        if s_d == 0:
            s_d = -self.section / 200000
        else:
            s_g = -self.section / 200000
        while ds > self.section / 100000:
            l = (lg + ld) / 2
            temp = np.identity(4)
            temp[0, 3] = l
            g_center = np.dot(g, temp)
            s_center, Sim = self.Immersed_surface_theoric(g_center,circular)
            if s_center == 0:
                s_center = -self.section / 200000
            if s_center * s_g <= 0:  # différent à gauche
                ld = (ld + l) / 2
                temp = np.identity(4)
                temp[0, 3] = ld
                g_d = np.dot(g, temp)
                s_d, Sim = self.Immersed_surface_theoric(g_d,circular)
                if s_d == 0:
                    s_d = - self.section / 200000
                ds = abs(s_g - s_d)
            elif s_center * s_d <= 0:
                lg = (lg + l) / 2
                temp = np.identity(4)
                temp[0, 3] = lg
                g_g = np.dot(g, temp)
                s_g, Sim = self.Immersed_surface_theoric(g_g,circular)
                if s_g == 0:
                    s_g = -self.section / 200000
                ds = abs(s_g - s_d)
            else:
                lg = l
                break
        return lg

    def buoyancy_wrench_i_segment(self, g=None, nb_step1=100, nb_step2=100, head=False, trapese=False, Sp_Rule=False,head_or_neck = False):
        '''
        :param g: the frame of segment i expressed in the world frame R0,
        g is placed on the barycenter of segment i
        :param nb_step1: number of steps for the integration of the volume of segment i
        :param head: if we calculate the head's buoyancy_wrench
        :param trapez: if we use the trapez integration
        :param Sp_Rule: if we use the Simpson's rule
        :return:
        '''
        if head:
            dL = self.P_head / nb_step1
        else:
            dL = self.P/nb_step1
        if not head_or_neck:
            temp = np.identity(4)
            temp[0, 3] = -self.P / 2
            g = np.dot(g, temp)
        list_Sim = np.array([])
        for i in range(0, nb_step1 + 1):
            g0i = np.identity(4)
            g0i[0, 3] = dL * i
            gi = np.dot(g, g0i)
            Sim, SimQb_2d = self.Immersed_surface_theoric(gi,head_or_neck)
            if Sim != 0:
                SimQb = np.array([0, SimQb_2d[0], SimQb_2d[1], Sim])
                SimQb_in_R0 = np.dot(gi, SimQb)
                list_Sim = np.append(list_Sim, np.array([i, Sim, SimQb_in_R0[0], SimQb_in_R0[1], SimQb_in_R0[2]]))
        if list_Sim.shape[0] == 0:
            buoyancy_wrench = np.zeros((6, 1))
        else:
            list_Sim = np.reshape(list_Sim, (-1, 5))
            Mi = np.zeros(3)  ##Vecteur de l'effort et du couple
            Fi = np.zeros(3)
            e3 = np.array([0, 0, 1])
            ###Euler
            if trapese is False and Sp_Rule is False:
                for i in range(0, list_Sim.shape[0] - 1):
                    Sim = list_Sim[i, 1]
                    Fi = np.array([0, 0, 9800 * Sim])
                    SimQb_in_R0 = list_Sim[i, 2:]
                    Mi = 9800 * dL * SimQb_in_R0 + Mi
            #####trapese
            elif trapese is True:
                for i in range(0, list_Sim.shape[0] - 1):
                    Sim = (list_Sim[i, 1] + list_Sim[i + 1, 1]) / 2  ##Sim = (Sim(x+dL)+Sim(x))/2
                    Fi = np.array([0, 0, 9800 * Sim * dL]) + Fi
                    SimQb_in_R0 = (list_Sim[i, 2:] + list_Sim[i + 1, 2:]) / 2
                    Mi = 9800 * dL * SimQb_in_R0 + Mi
            #####simpson
            else:
                left_simple = (list_Sim.shape[0]) % 2
                nb_int = (list_Sim.shape[0] - 1) // 2  ## number of interval of Simpson's Rule
                for i in range(0, nb_int):
                    Sim_a = list_Sim[i * 2, 1]; SimQb_a = list_Sim[i * 2, 2:]
                    Sim_mid = list_Sim[i * 2 + 1, 1]; SimQb_mid = list_Sim[i * 2 + 1, 2:]
                    Sim_b = list_Sim[i * 2 + 2, 1]; SimQb_b = list_Sim[i * 2 + 2, 2:]
                    Sim_esteemed = Sim_a + 4 * Sim_mid + Sim_b
                    Fi = np.array([0, 0, 9800 * Sim_esteemed * dL / 3]) + Fi
                    SimQb_esteemed = SimQb_a + 4 * SimQb_mid + SimQb_b
                    Mi = 9800 * SimQb_esteemed * dL / 3 + Mi

                if not left_simple == 1:
                    Sim = list_Sim[-2, 1]
                    Fi = np.array([0, 0, 9800 * Sim * dL]) + Fi
                    SimQb_in_R0 = list_Sim[-2, 2:]
                    Mi = 9800 * dL * SimQb_in_R0 + Mi

            Mi = np.cross(Mi, e3)
            buoyancy_wrench = np.array([Fi, Mi]).reshape(6, 1)
        return buoyancy_wrench
    def Immersed_surface_theoric(self, g,circular):
        if circular:
            R33 = g[2, 2]
            R32 = g[2, 1]
            P3 = g[2, 3]
            a = b = sqrt(self.r_head)
            r0 = self.r_head
            sin_omega = -R32 #* sqrt(a / b)
            cos_omega = -R33 #* sqrt(b / a)
            omega = atan2(cos_omega, sin_omega)
            h = P3 / sqrt(pow(cos_omega, 2) + pow(sin_omega, 2))
            if h >= r0:
                Sim = 0
                Sim_rB = np.array([0., 0.])
            elif abs(h) < r0:
                temp0 = 1 - pow(h / r0, 2)
                temp1 = acos(h / r0)
                temp2 = (h / r0) * sqrt(temp0)
                Sim = pow(r0, 2) * (temp1 - temp2)
                Sim_rB = ((2 * pow(r0, 3)) / 3) * pow(temp0, 1.5)
                Sim_rB = Sim_rB * np.array([cos(omega), sin(omega)])
            else:
                Sim = self.r_head**2*pi
                Sim_rB = np.array([0., 0.])
        else:
            Sim,Sim_rB = self.surf.Immersed_surface_theoric(g)
        return Sim, Sim_rB

    def draw_error_config_head(self, data=[], data2=None, legend=[], legend2=[], title=''):
        data_ref = data[0]
        plt.title(title)
        for i in range(1, len(data)):
            data_i = data[i]
            Err = np.empty(data_i.shape[2] - 1)
            for b in range(0, data_i.shape[2] - 1):
                data_b = data_i[0:3, 0:3, b]
                data_ref_b = data_ref[:, :, b]
                data_error = data_b - data_ref_b
                error = 0
                for j in range(0, 3):
                    for k in range(0, 3):
                        error = error + data_error[j, k] ** 2
                Err[b] = error
                ##print(error)
            t = np.linspace(0, 5, data_i.shape[2] - 1)

            plt.plot(t, Err, label=legend[i - 1])
            plt.xlabel('time(s)')
            plt.ylabel('error')
            plt.legend()
        if data2 != None:
            data_ref2 = data2[0]
            for i in range(1, len(data2)):
                data_i = data2[i]
                Err = np.empty(data_i.shape[2] - 1)
                for b in range(0, data_i.shape[2] - 1):
                    data_b = data_i[0:3, 0:3, b]
                    # print('calculé',b)
                    # print(data_b)
                    data_ref_b = data_ref2[:, :, b]
                    # print('analogue',b)
                    # print(data_ref_b)
                    data_error = data_b - data_ref_b
                    error = 0
                    for j in range(0, 3):
                        for k in range(0, 3):
                            error = error + data_error[j, k] ** 2
                    Err[b] = error
                    ##print(error)
                t = np.linspace(0, 5, data_i.shape[2] - 1)

                plt.plot(t, Err, label=legend2[i - 1])
                plt.xlabel('time(s)')
                plt.ylabel('error')
                plt.legend()
        plt.grid()
        plt.show()

    def draw_error_config_head2(self, data=[], legend=[], title=''):
        data_ref = data[0]
        plt.title(title)
        for i in range(1, len(data)):
            data_i = data[i]
            Err = np.empty(data_i.shape[2] - 1)
            for b in range(0, data_i.shape[2] - 1):
                data_b = data_i[0:3, 0:3, b]
                data_ref_b = data_ref[:, :, b]
                data_error = data_b - data_ref_b
                error = 0
                for j in range(0, 3):
                    for k in range(0, 3):
                        error = error + data_error[j, k] ** 2
                Err[b] = error
                ##print(error)
            t = np.linspace(0, 5, data_i.shape[2] - 1)

            plt.plot(t, Err, label=legend[i - 1])
            plt.xlabel('time(s)')
            plt.ylabel('error')
            plt.legend()
        plt.grid()
        plt.show()

    def draw_frame_head(self, list_head):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for i in range(len(list_head)):
            x = list_head[i][0:3, 0]
            y = list_head[i][0:3, 1]
            z = list_head[i][0:3, 2]
            P = list_head[i][0:3, 3]
            ax.quiver(P[0], P[1], P[2], x[0], x[1], x[2], length=0.1, color='blue')
            ax.quiver(P[0], P[1], P[2], y[0], y[1], y[2], length=0.1, color='blue')
            ax.quiver(P[0], P[1], P[2], z[0], z[1], z[2], length=0.1, color='blue')
            Y = np.linspace(-0.5 / 3, 0.5 / 3, 100)
            XX = np.array([]);
            YY = np.array([]);
            ZZ = np.array([])
            for j in range(Y.shape[0]):
                temp = np.array([0, Y[j], sqrt(1 - (Y[j] / (0.5 / 3)) ** 2) * pow(1 / 3, 2), 1])
                temp = np.dot(list_head[i], temp)
                XX = np.append(XX, temp[0]); YY = np.append(YY, temp[1]); ZZ = np.append(ZZ, temp[2])
            for k in range(Y.shape[0]):
                temp = np.array([0, Y[-k - 1], -sqrt(1 - (Y[-k - 1] / (0.5 / 3)) ** 2) * pow(1 / 3, 2), 1])
                temp = np.dot(list_head[i], temp)
                XX = np.append(XX, temp[0]);
                YY = np.append(YY, temp[1]);
                ZZ = np.append(ZZ, temp[2])
            ax.plot(XX, YY, ZZ)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.show()

    def draw_complet_model_varied_color(self,j,r,n,h):
        import matplotlib.colors as mcol
        import matplotlib.cm as cm
        '''
        start_time = 100
        end_time = 120

        # Generate some dummy data.
        tim = range(start_time,end_time+1)#np.linspace(start_time, end_time + 1,j.shape[0])


        # Make a user-defined colormap.
        cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["r", "b"])

        # Make a normalizer that will map the time values from
        # [start_time,end_time+1] -> [0,1].
        cnorm = mcol.Normalize(vmin=min(tim), vmax=max(tim))

        # Turn these into an object that can be used to map time values to colors and
        # can be passed to plt.colorbar().
        cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
        cpick.set_array([])
        '''
        fig = plt.figure()
        for i in range(j.shape[0]):
            self.list_of_DH_matrices_head = h[i*4:i*4+4,:]
            self.joint_config = j[i,:]
            self.antirouli_config = r[i,:]
            self.neck_config = n[i,:]
            self.get_neck_frame();self.get_list_joint_frame();self.get_list_roulie_frame();self.get_list_frame_R02()
            #print(cpick.to_rgba)
            self.draw_complet_model(fig, color=None)
        #plt.colorbar(cpick, label="Time (seconds)")
    def draw_complet_model(self,fig,color = None):
        #fig = plt.figure()
        if color == None:
            color = 'blue'
        ax = fig.gca(projection='3d')
        list_head = self.list_of_DH_matrices_head
        x = [list_head[0, 3]]
        y = [list_head[1,3]]
        z = [list_head[2, 3] ]
        ###################################################
        ######## fleche of frame
        ##################################################
        xx = list_head[0:3, 0]
        yy = list_head[0:3, 1]
        zz = list_head[0:3, 2]
        P = list_head[0:3, 3]
        ax.quiver(P[0], P[1], P[2], xx[0], xx[1], xx[2], length=0.2, color='blue')
        ax.quiver(P[0], P[1], P[2], yy[0], yy[1], yy[2], length=0.2, color='blue')
        ax.quiver(P[0], P[1], P[2], zz[0], zz[1], zz[2], length=0.2, color='blue')
        ##################################################
        y_z = self.surf.contour_surface()
        y_z = np.transpose(y_z)
        r = self.r_head
        y_cercle = np.linspace(-r,r,35)
        yz_cercle = np.array([])
        for o in range(y_cercle.shape[0]):
            yz_cercle = np.append(yz_cercle,np.array([y_cercle[o],sqrt(r**2-y_cercle[o]**2)]))
            if y_cercle[o] == 0:
                yz_cercle = np.append(yz_cercle, np.array([0, 0]))
        for o in range(y_cercle.shape[0]):
            yz_cercle = np.append(yz_cercle, np.array([y_cercle[y_cercle.shape[0]-o-1], -sqrt(r ** 2 - y_cercle[y_cercle.shape[0]-o-1] ** 2)]))

        yz_cercle = np.reshape(yz_cercle,(-1,2))
        X_cercle = []; Y_cercle = []; Z_cercle = []
        for i in range(yz_cercle.shape[0]):
            Coord = np.dot(list_head,np.array([0,yz_cercle[i,0],yz_cercle[i,1],1]))
            X_cercle.append(Coord[0]);Y_cercle.append(Coord[1]); Z_cercle.append(Coord[2])
        list_ims = []
        p1, = ax.plot(X_cercle, Y_cercle, Z_cercle, color=color)
        list_ims.append(p1)
        X_cercle = [];
        Y_cercle = [];
        Z_cercle = []
        list_neck = self.DH_matrix_neck_in_R0
        x.append(list_neck[0,3]);y.append(list_neck[1,3]);z.append(list_neck[2,3])
        for i in range(yz_cercle.shape[0]):
            Coord = np.dot(list_neck,np.array([0,yz_cercle[i,0],yz_cercle[i,1],1]))
            X_cercle.append(Coord[0]);Y_cercle.append(Coord[1]); Z_cercle.append(Coord[2])
        p2, = ax.plot(X_cercle, Y_cercle, Z_cercle, color=color)
        list_ims.append(p2)
        for i in range(self.nb):
            body_i = self.list_of_DH_matrices_joint_in_R0[:,4*i:4*i+4]
            #temp = np.eye(4); temp[0,3] = -self.P/2
            body_shape_i = self.list_of_DH_matrices_roulie_in_R0[:,4*i:4*i+4]
            #body_shape_i = np.dot(body_shape_i, temp)
            x.append(body_i[0,3]);y.append(body_i[1,3]); z.append(body_i[2,3])
            if i == self.nb-1:
                temp = np.eye(4); temp[0,3] = self.P
                body_i = np.dot(body_i,temp)
                x.append(body_i[0, 3]); y.append(body_i[1, 3]); z.append(body_i[2, 3])
            X_cercle = [];
            Y_cercle = [];
            Z_cercle = []
            for j in range(y_z.shape[0]):
                Coord = np.dot(body_shape_i, np.array([0, y_z[j, 0], y_z[j, 1],1]))
                X_cercle.append(Coord[0]); Y_cercle.append(Coord[1]); Z_cercle.append(Coord[2])
            p3, = ax.plot(X_cercle, Y_cercle, Z_cercle, color)
            list_ims.append(p3)
        p4, = ax.plot(x, y, z,color)
        list_ims.append(p4)
        #ax.scatter(x,y,z)
        xx = np.arange(0, 2.0, .1)
        yy = np.arange(-1.0, 1.0, .1)
        X, Y = np.meshgrid(xx, yy)
        #ax.plot_surface(X, Y, np.zeros((20, 20)), color='blue', alpha=0.1)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)

        return list_ims
    def draw_complet_model_2(self,fig,g0,g1):
        #fig = plt.figure()
        ax = fig.gca(projection='3d')
        list_head = g0

        r = self.r_head
        y_cercle = np.linspace(-r,r,35)
        yz_cercle = np.array([])
        for o in range(y_cercle.shape[0]):
            yz_cercle = np.append(yz_cercle,np.array([y_cercle[o],sqrt(r**2-y_cercle[o]**2)]))
        for o in range(y_cercle.shape[0]):
            yz_cercle = np.append(yz_cercle, np.array([y_cercle[y_cercle.shape[0]-o-1], -sqrt(r ** 2 - y_cercle[y_cercle.shape[0]-o-1] ** 2)]))
        yz_cercle = np.reshape(yz_cercle,(-1,2))
        X_cercle = []; Y_cercle = []; Z_cercle = []
        for i in range(yz_cercle.shape[0]):
            Coord = np.dot(list_head,np.array([0,yz_cercle[i,0],yz_cercle[i,1],1]))
            X_cercle.append(Coord[0]);Y_cercle.append(Coord[1]); Z_cercle.append(Coord[2])
        ax.plot(X_cercle, Y_cercle, Z_cercle, color='blue')
        xx = list_head[0:3, 0]
        yy = list_head[0:3, 1]
        zz = list_head[0:3, 2]
        P = list_head[0:3, 3]
        ax.quiver(P[0], P[1], P[2], xx[0], xx[1], xx[2], length=0.2, color='blue')
        ax.quiver(P[0], P[1], P[2], yy[0], yy[1], yy[2], length=0.2, color='blue')
        ax.quiver(P[0], P[1], P[2], zz[0], zz[1], zz[2], length=0.2, color='blue')

        list_head = g1

        r = self.r_head
        y_cercle = np.linspace(-r, r, 35)
        yz_cercle = np.array([])
        for o in range(y_cercle.shape[0]):
            yz_cercle = np.append(yz_cercle, np.array([y_cercle[o], sqrt(r ** 2 - y_cercle[o] ** 2)]))
        for o in range(y_cercle.shape[0]):
            yz_cercle = np.append(yz_cercle, np.array(
                [y_cercle[y_cercle.shape[0] - o - 1], -sqrt(r ** 2 - y_cercle[y_cercle.shape[0] - o - 1] ** 2)]))
        yz_cercle = np.reshape(yz_cercle, (-1, 2))
        X_cercle = [];
        Y_cercle = [];
        Z_cercle = []
        for i in range(yz_cercle.shape[0]):
            Coord = np.dot(list_head, np.array([0, yz_cercle[i, 0], yz_cercle[i, 1], 1]))
            X_cercle.append(Coord[0]);
            Y_cercle.append(Coord[1]);
            Z_cercle.append(Coord[2])
        ax.plot(X_cercle, Y_cercle, Z_cercle, color='red')
        xx = list_head[0:3, 0]
        yy = list_head[0:3, 1]
        zz = list_head[0:3, 2]
        P = list_head[0:3, 3]
        ax.quiver(P[0], P[1], P[2], xx[0], xx[1], xx[2], length=0.2, color='red')
        ax.quiver(P[0], P[1], P[2], yy[0], yy[1], yy[2], length=0.2, color='red')
        ax.quiver(P[0], P[1], P[2], zz[0], zz[1], zz[2], length=0.2, color='red')
        '''
        xx = np.arange(0, 2.0, .1)
        yy = np.arange(-1.0, 1.0, .1)
        X, Y = np.meshgrid(xx, yy)
        ax.plot_surface(X, Y, np.zeros((20, 20)), color='blue', alpha=0.1)
        '''
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_xlim(0, 0.2)
        ax.set_ylim(-0.1, 0.1)
        ax.set_zlim(-0.1, 0.1)
    def draw_the_whole_body_section_trig(self,fig):
        #fig = plt.figure()
        ax = fig.gca(projection='3d')
        list_head = self.list_of_DH_matrices_head
        x = list_head[0:3, 0]
        y = list_head[0:3, 1]
        z = list_head[0:3, 2]
        P = list_head[0:3, 3]
        ax.quiver(P[0], P[1], P[2], x[0], x[1], x[2], length=2, color='blue')
        ax.quiver(P[0], P[1], P[2], y[0], y[1], y[2], length=2, color='blue')
        ax.quiver(P[0], P[1], P[2], z[0], z[1], z[2], length=2, color='blue')
        suf = surface(0.3, 1)
        y_z = suf.contour_surface()
        r = sqrt(3)/2*0.4
        y_cercle = np.linspace(-r,r,35);z_cercle = np.array([])
        yz_cercle = np.array([])
        for o in range(y_cercle.shape[0]):
            yz_cercle = np.append(yz_cercle,np.array([y_cercle[o],sqrt(r**2-y_cercle[o]**2)]))
        for o in range(y_cercle.shape[0]):
            yz_cercle = np.append(yz_cercle, np.array([y_cercle[y_cercle.shape[0]-o-1], -sqrt(r ** 2 - y_cercle[y_cercle.shape[0]-o-1] ** 2)]))
        yz_cercle = np.reshape(yz_cercle,(-1,2))
        yz_cercle = np.transpose(yz_cercle)
        print(yz_cercle.shape)
        x = np.linspace(0,self.P, 50)
        for i in range(x.shape[0]):
            gi = np.eye(4); gi[0,3] = x[i]
            g = np.dot(list_head,gi)
            X = np.array([]);Y = np.array([]); Z = np.array([])
            for j in range(yz_cercle.shape[1]):
                coord = np.dot(g,np.array([0,yz_cercle[0,j],yz_cercle[1,j],1]))
                X=np.append(X,coord[0]);Y=np.append(Y,coord[1]);Z=np.append(Z,coord[2])
            ax.plot(X, Y, Z, color='blue')
        color = ['green','yellow','red']
        for i in range(self.nb):
            body_shape_i = self.list_of_DH_matrices_roulie_in_R0[:,4*i:4*i+4]
            gi = np.eye(4); gi[0, 3] = -self.P/2
            body_shape_i = np.dot(body_shape_i, gi)
            x = body_shape_i[0:3, 0]
            y = body_shape_i[0:3, 1]
            z = body_shape_i[0:3, 2]
            P = body_shape_i[0:3, 3]
            ax.quiver(P[0], P[1], P[2], x[0], x[1], x[2], length=2, color=color[i%3])
            ax.quiver(P[0], P[1], P[2], y[0], y[1], y[2], length=2, color=color[i%3])
            ax.quiver(P[0], P[1], P[2], z[0], z[1], z[2], length=2, color=color[i%3])
            x = np.linspace(0, self.P, 50)
            for j in range(x.shape[0]):
                gi = np.eye(4); gi[0, 3] = x[j]
                g = np.dot(body_shape_i, gi)
                X = np.array([]);  Y = np.array([]);  Z = np.array([])
                for k in range(y_z.shape[1]):
                    coord = np.dot(g, np.array([0, y_z[0, k], y_z[1, k], 1]))
                    X = np.append(X, coord[0]);
                    Y = np.append(Y, coord[1]);
                    Z = np.append(Z, coord[2])
                ax.plot(X, Y, Z, color[i%3])
        x = y = np.arange(-10.0, 10.0, .1)
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, np.zeros((200, 200)), color='blue', alpha=0.1)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_xlim(0, 10)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-3, 3)

    def head_config_generation2(self, frequence, periode):
        dt = 1 / frequence
        nb = int(periode / dt)
        V = -pi / 20
        head_conf = np.empty((nb - 1, 4))
        for i in range(0, nb - 1):
            head_conf[i, :] = np.array([0, 1, 0, V])
            ##head_conf[i,0:3]=(1/np.linalg.norm(head_conf[i,0:3]))*head_conf[i,0:3]
        ref_data = np.empty((3, 3, nb))
        for i in range(0, nb):
            ref_data[:, :, i] = np.array([[cos(V * (i * dt)), 0, sin(V * (i * dt))],
                                          [0, 1, 0],
                                          [-sin(V * (i * dt)), 0, cos(V * (i * dt))]])
        self.head_config = head_conf
        return ref_data
    def R0_inertial_moment(self):
        self.I_head = np.array([[182.598,0.,0.],[0.,181.333,0.],[0.,0.,174.73]])*1E-6
        self.I_neck = np.array([[182.598,0.,0.],[0.,181.333,0.],[0.,0.,174.73]])*1E-6
        self.I_seg = np.array([[259.577,0.,0.],[0.,1390,0.],[0.,0.,1360]])*1E-6*0.3941/0.488876
        self.I_roll = np.array([[316.142, 0., 0.], [0., 243.911, 0.], [0., 0., 249.183]]) * 1E-6
        temp = np.eye(4)
        temp[:4,3]=self.centre_head
        T_g0 = np.dot(self.list_of_DH_matrices_head,temp)
        S_hat = self.skew_symetric_matrix(T_g0[:3,3])
        'head to R0'
        I=self.I_head-self.m_head*np.dot(S_hat,S_hat)
        I = np.dot(T_g0[:3,:3],np.dot(I,T_g0[:3,:3].T))
        'neck to R0'
        temp = np.eye(4)
        temp[:4, 3] = self.centre_neck
        T_g0 = np.dot(self.DH_matrix_neck_in_R0, temp)
        S_hat = self.skew_symetric_matrix(T_g0[:3, 3])
        I_neck_R0 = self.I_neck-self.m_neck*np.dot(S_hat,S_hat)
        I += np.dot(T_g0[:3,:3],np.dot(I_neck_R0,T_g0[:3,:3].T))
        for i in range(self.nb):
            T_seg = self.list_of_DH_matrices_joint_in_R0[:,i*4:i*4+4]
            temp = np.eye(4)
            if i != self.nb-1:
              temp[:4, 3] = self.centre_segment
              m = self.m_seg
            else:
                temp[:4, 3] = self.centre_last_seg
                m=self.m_last_seg
            T_seg = np.dot(T_seg, temp)
            S_hat = self.skew_symetric_matrix(T_seg[:3, 3])
            I_seg_R0 = self.I_seg - m * np.dot(S_hat, S_hat)
            I += np.dot(T_seg[:3, :3], np.dot(I_seg_R0, T_seg[:3, :3].T))

            T_roll = self.list_of_DH_matrices_roulie_in_R0[:,i * 4:i * 4 + 4]
            temp = np.eye(4)
            temp[:4, 3] = self.centre_roll
            T_roll = np.dot(T_roll, temp)
            S_hat = self.skew_symetric_matrix(T_roll[:3, 3])
            I_roll_R0 = self.I_roll - self.m_roll * np.dot(S_hat, S_hat)
            I += np.dot(T_roll[:3, :3], np.dot(I_roll_R0, T_roll[:3, :3].T))
        return I

    def dyn_simplified(self,t,eta,g0):
        eta_p = np.zeros((6),dtype=float)
        eta_p[:3] = eta[3:]
        if t < 3.:
            f_ext = np.array([0., 0.02*t, 0.])
        else:
            f_ext = np.array([0., 0., 0.])
        ###deplace the head
        X = np.array([0.,0.,eta[0],eta[1],eta[2],0.])
        X_hat = np.zeros((4, 4),dtype=float)
        X_hat[:3, :3] = self.skew_symetric_matrix(X[3:6]);
        X_hat[:3, 3] = X[:3]
        n_g0_np1 = expm(X_hat)
        for i in range(3):
            n_g0_np1[:3, i] = n_g0_np1[:3, i] / np.linalg.norm(n_g0_np1[:3, i])
        self.list_of_DH_matrices_head=np.dot(n_g0_np1,g0)
        self.get_list_frame_R02()
        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        w_R0 = wb + wg.reshape((6, 1))
        w_R0 = w_R0[2:5,0]
        J = self.R0_inertial_moment()
        I = np.zeros((3,3))
        I[0,0]= self.M; I[1:,1:]=J[:2,:2]
        eta_p[3:] = np.dot(np.linalg.inv(I), f_ext+w_R0)
        return eta_p