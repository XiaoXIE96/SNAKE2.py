from SNAKE_2 import snake
import numpy as np
from math import *
import math
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy import integrate,optimize
from mpl_toolkits.mplot3d import Axes3D
from Surface_2 import surface
import copy
class Dynamic_snake(snake):
    def __init__(self,nb=6, P=0.171,P_head = 0.107 , ro_seg=312,r_head=0.04,qp_neck=np.zeros((2)),qp_roll = np.zeros((6)),
               qp_joint = np.zeros((6)),vel_head = np.zeros((6)),acc_head = np.zeros((6)),qpp_neck=np.zeros((2)),qpp_roll = np.zeros((6)),
               qpp_joint = np.zeros((6))):
        snake.__init__(self,nb=nb, P=P,P_head = P_head, ro_seg=ro_seg,r_head=r_head)
        self.vel_head = vel_head #head's velocity
        self.qp_neck = qp_neck
        self.qp_roll = qp_roll
        self.qp_joint = qp_joint
        self.acc_head = acc_head
        self.qpp_neck = qpp_neck
        self.qpp_roll = qpp_roll
        self.qpp_joint = qpp_joint
        self.J_head = np.array([[182.598, 0., 0.], [0., 181.333, 0.], [0., 0., 174.73]]) * 1E-6
        self.J_neck = np.array([[182.598, 0., 0.], [0., 181.333, 0.], [0., 0., 174.73]]) * 1E-6
        self.J_seg = np.array([[259.577, 0., 0.], [0., 1390, 0.], [0., 0., 1360]]) * 1E-6 * 0.3941 / 0.488876
        self.J_roll = np.array([[316.142, 0., 0.], [0., 243.911, 0.], [0., 0., 249.183]]) * 1E-6
        S_hat = self.skew_symetric_matrix(self.centre_head[:3])
        self.I_head = np.zeros((6,6))
        self.I_head[:3,:3] = self.m_head*np.eye(3)
        self.I_head[:3,3:] = -self.m_head*S_hat
        self.I_head[3:, :3] = self.m_head * S_hat
        self.I_head[3:, 3:] = self.J_head-self.m_head*np.dot(S_hat,S_hat)
        S_hat = self.skew_symetric_matrix(self.centre_neck[:3])
        self.I_neck = np.zeros((6, 6))
        self.I_neck[:3, :3] = self.m_neck * np.eye(3)
        self.I_neck[:3, 3:] = -self.m_neck * S_hat
        self.I_neck[3:, :3] = self.m_neck * S_hat
        self.I_neck[3:, 3:] = self.J_neck-self.m_neck*np.dot(S_hat,S_hat)

        S_hat = self.skew_symetric_matrix(self.centre_segment[:3])
        #self.m_seg = 0
        #self.m_last_seg = 0
        #self.J_seg = np.zeros((3,3))
        self.I_seg = np.zeros((6, 6))
        self.I_seg[:3,:3] = self.m_seg*np.eye(3)
        self.I_seg[:3,3:] = -self.m_seg*S_hat
        self.I_seg[3:, :3] = self.m_seg * S_hat
        self.I_seg[3:, 3:] = self.J_seg-self.m_seg*np.dot(S_hat,S_hat)

        S_hat = self.skew_symetric_matrix(self.centre_roll[:3])
        self.I_roll = np.zeros((6, 6))
        self.I_roll[:3, :3] = self.m_roll * np.eye(3)
        self.I_roll[:3, 3:] = -self.m_roll * S_hat
        self.I_roll[3:, :3] = self.m_roll * S_hat
        self.I_roll[3:, 3:] = self.J_roll-self.m_roll*np.dot(S_hat,S_hat)

        self.list_of_velocity=None; self.beta=None; self.gamma=None
        self.tau = np.zeros((self.nb*2+2))

    def geometrical_model(self,q=None,qn=None,qf=None,qr = None):
        if q is not None:
            self.neck_config = q[:2]
            for i in range(self.nb):
                self.joint_config[i] = q[2*i+2]
                self.antirouli_config[i] = q[2 * i + 3]
        elif qn is not None:
            self.neck_config = qn
            self.joint_config = qf
            self.antirouli_config = qr
        self.get_neck_frame()
        self.get_list_joint_frame()
        self.get_list_roulie_frame()
        self.get_list_frame_R02()

    def quaterion_product(self,a,b):
        A = np.array([[a[0],-a[1],-a[2],-a[3]],
                      [a[1],a[0],-a[3],a[2]],
                      [a[2],a[3],a[0],-a[1]],
                      [a[3],-a[2],a[1],a[0]]])
        return np.dot(A,b)
    def get_zeta_j(self,i_g_im1,qd_i,eta_im1):
        '''
                :param i_g_im1: pose igi-1
                :param qd_i: joint velo
                :param eta_im1: velocity i-1
                :return: centrifuge acceleration (zeta)
        '''
        zeta_j = np.zeros((6))
        _0V1 = np.dot(i_g_im1[:3, :3], eta_im1[:3]);
        _0Omega1 = np.dot(i_g_im1[:3, :3], eta_im1[3:])
        temp = _0V1 + np.cross(i_g_im1[:3, 3], _0Omega1)
        zeta_j[:3] = np.cross(temp, qd_i)
        zeta_j[3:] = np.cross(_0Omega1, qd_i)
        return zeta_j
    def get_gamma_j(self,i_g_im1,qd_i,qdd_i,eta_im1):
        '''
        :param i_g_im1: pose igi-1
        :param qd_i: joint velo
        :param qdd_i: joint accel
        :param eta_im1: velocity i-1
        :return: centrifuge acceleration + joint acceleration (zeta + qdd)
        '''
        gamma_j = np.zeros((6))
        _0V1 = np.dot(i_g_im1[:3, :3], eta_im1[:3]); _0Omega1 = np.dot(i_g_im1[:3, :3], eta_im1[3:])
        temp = _0V1 + np.cross(i_g_im1[:3, 3], _0Omega1)
        gamma_j[:3] = np.cross(temp, qd_i)
        gamma_j[3:] = np.cross(_0Omega1, qd_i) + qdd_i
        return gamma_j

    def get_beta_j(self,eta_i,Si,Mi,I,gi,Fb):
        '''
        :param eta_i: velocity i
        :param Si: center of mass in Fi
        :param Mi: mass
        :param I: moment inertie
        :param gi:pose of body i % Fg
        :param Fb: buoyancy wrench in Fg
        :return: Centrifuge/Corioli force and external force(buoyancy and gravity)
        '''

        ###### Buoyancy Force

        Ad_gi = self.adjoint_matrix_2(gi)
        Fbuo = np.dot(Ad_gi.T,Fb)
        ###### Gravity Force
        Fg = np.zeros((6))
        fg = np.dot(gi[:3,:3].T,np.array([0,0,-Mi*self.g]))
        cg = np.cross(Si,fg)
        Fg[:3]=fg;Fg[3:] = cg
        ###### Centrifuge/Corioli
        Vi = eta_i[:3]; Oi = eta_i[3:]
        beta = np.zeros((6))
        S_hat = self.skew_symetric_matrix(Si)
        '''
        temp1 = Mi*np.dot(S_hat,Oi)
        beta[:3] = -np.cross(Oi,temp1)+Mi*np.cross(Oi,Vi)
        temp2 = Mi*np.cross(Oi,Vi)
        beta[3:] = np.cross(Oi,np.dot(I,Oi))+np.dot(S_hat,temp2)
        
        temp = -np.cross(Si,Oi)+Vi
        beta[:3] = Mi*np.cross(Oi,temp)
        temp2 = np.dot(I,Oi)+Mi*np.cross(Si,Vi)
        beta[3:] = np.cross(Oi,temp2)
        '''
        beta[:3] = Mi * np.cross(Oi, np.cross(Oi, Si))
        beta[3:] = np.cross(Oi, np.dot(I, Oi))
        return beta#-Fbuo-Fg

    def get_fb_i(self, g, head = False, neck = False, last_seg = False, roll = False):
        '''
        :param g: the configuration of segment
        :param nb: number of gauss nodes
        :param head: if the segment is the head?
        :return: fb_i and fb_r,i if has shell
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
                  F, e = integrate.fixed_quad(buoyancy_force, interval[0], interval[1], n=30, args=(gi, circ))
                  Mi = np.array([F[1], F[2], 0.])
                  Mi = np.cross(Mi, e3)
                  buoyancy_wrench = 9800 * np.array([0, 0, F[0], Mi[0], Mi[1], 0])
              else:
                  buoyancy_wrench = np.zeros((6))
              return buoyancy_wrench
        if head:
          Fb = integrate_for_one_seg(g,0,self.P_head,True)
        elif neck:
          Fb = integrate_for_one_seg(g,0,self.P_neck,True)
        elif roll:
          l = self.P - 0.036-0.045
          Fb = integrate_for_one_seg(g,-l/2,l/2,False)
        elif last_seg:
          Fb = integrate_for_one_seg(g,0.,0.045,True)+integrate_for_one_seg(g,self.P - 0.036, self.P - 0.023,True)
        else:
          Fb = integrate_for_one_seg(g,0.,0.045,True)+integrate_for_one_seg(g,self.P - 0.036, self.P,True)

        return Fb
################ Inverse recursive NE
    def get_velocity_gamma_beta(self,inverse_NE = False):
        '''
        first forward recursion from head to tail for inverse NE or direct NE

        :return:
        velocity: 6*nb_body velocities of each body
        gamma: 6*(nb_body-1) centrifuge acceleration and joint acceleration, don't depend on the head acceleration
        zeta:
        beta: 6*(nb_body) centrifuge force + exterior fore
        '''
        self.list_of_velocity = np.empty((6,self.nb*2+2))
        self.gamma = np.zeros((6, self.nb * 2 + 1))
        self.zeta = np.zeros((6, self.nb * 2 + 1))
        self.beta = np.zeros((6, self.nb * 2 + 2))
        self.Adj_gip1i = np.zeros((6,6,self.nb))
        self.Adj_grolli = np.zeros((6,6,self.nb))
        self.Adj_gnh = np.zeros((6,6))#neck to head
        if self.qp_neck.shape[0] != 2:
            AssertionError('neck qp must be 2')
        if self.qp_roll.shape[0] !=self.nb or self.qp_joint.shape[0] != self.nb:
            AssertionError('joint or roll velocity dim is wrong')
        self.list_of_velocity[:, 0] = self.vel_head
        fb = self.get_fb_i(self.list_of_DH_matrices_head, head=True)
        self.beta[:,0] = self.get_beta_j(self.vel_head, self.centre_head[:3], self.m_head, self.J_head, self.list_of_DH_matrices_head, fb)

        g01 = self.DH_matrix_neck
        g10 = self.inverse_configuration(g01)
        Ad_g10 = self.adjoint_matrix_2(g10)
        self.Adj_gnh=Ad_g10
        ########## velocity
        eta_i = np.dot(Ad_g10,self.vel_head)+np.array([0,0,0,0,self.qp_neck[0],self.qp_neck[1]])
        self.list_of_velocity[:,1] = eta_i
        ######### gamma
        if inverse_NE:
            self.gamma[:,0] = self.get_gamma_j(g10,np.array([0,self.qp_neck[0],self.qp_neck[1]]),np.array([0,self.qpp_neck[0],self.qpp_neck[1]]),self.vel_head)
        else:
            self.zeta[:, 0] = self.get_zeta_j(g10, np.array([0, self.qp_neck[0], self.qp_neck[1]]),
                                                 self.vel_head)
        ######### beta
        fb = self.get_fb_i(self.DH_matrix_neck_in_R0, neck=True)
        self.beta[:,1] = self.get_beta_j(eta_i, self.centre_neck[:3], self.m_neck, self.J_neck,
                                  self.DH_matrix_neck_in_R0, fb)
        for i in range(self.nb):
            giip1 = self.list_of_DH_matrices_joint[:,i*4:i*4+4]
            gip1i = self.inverse_configuration(giip1)
            Ad_gip1i = self.adjoint_matrix_2(gip1i)
            self.Adj_gip1i[:, :, i] = Ad_gip1i
            gi = self.list_of_DH_matrices_joint_in_R0[:,i*4:i*4+4]
            if i == 0:
               eta_i = np.dot(Ad_gip1i, eta_i) + np.array([0, 0, 0, 0, self.qp_joint[i],0])
               if inverse_NE:
                   self.gamma[:, i*2+1] = self.get_gamma_j(gip1i, np.array([0, self.qp_joint[i],0]),
                                                  np.array([0, self.qpp_joint[i],0]), self.list_of_velocity[:, i * 2 + 1])
               else:
                   self.zeta[:, i * 2 + 1] = self.get_zeta_j(gip1i, np.array([0, self.qp_joint[i], 0]),
                                                               self.list_of_velocity[:, i * 2 + 1])
            else:

               eta_i = np.dot(Ad_gip1i,eta_i)+np.array([0,0,0,0,0,self.qp_joint[i]])
               if inverse_NE:
                   self.gamma[:, i*2+1] = self.get_gamma_j(gip1i, np.array([0, 0, self.qp_joint[i]]),
                                                  np.array([0, 0, self.qpp_joint[i]]),
                                                  self.list_of_velocity[:, i * 2 ])   #there was +1
               else:
                   self.zeta[:, i * 2 + 1] = self.get_zeta_j(gip1i, np.array([0, 0, self.qp_joint[i]]),
                                                               self.list_of_velocity[:, i * 2 ])  #there was +1
            self.list_of_velocity[:, i * 2 + 2] = eta_i
            ######### beta
            if i == self.nb-1:
               fb = self.get_fb_i(gi, last_seg=True)
               self.beta[:,2*i+2] = self.get_beta_j(eta_i, self.centre_last_seg[:3], self.m_last_seg, self.J_seg,
                                          gi, fb)
            else:
               fb = self.get_fb_i(gi, False, False, False, False)
               self.beta[:,2*i+2] = self.get_beta_j(eta_i, self.centre_segment[:3], self.m_seg, self.J_seg,
                                      gi, fb)

            ############ for roll
            gi_roll=self.list_of_DH_matrices_roulie[:,i*4:i*4+4]
            groll_i = self.inverse_configuration(gi_roll)
            Ad_groll_i = self.adjoint_matrix_2(groll_i)
            self.Adj_grolli[:, :, i] = Ad_groll_i
            g_ri = self.list_of_DH_matrices_roulie_in_R0[:,i*4:i*4+4]
            self.list_of_velocity[:, i * 2 + 3] = np.dot(Ad_groll_i,eta_i)+np.array([0,0,0,self.qp_roll[i],0,0])
            if inverse_NE:
                self.gamma[:, i * 2 + 2] = self.get_gamma_j(groll_i, np.array([self.qp_roll[i], 0, 0]),
                                              np.array([self.qpp_roll[i], 0, 0]), eta_i)
            else:
                self.zeta[:, i * 2 + 2] = self.get_zeta_j(groll_i, np.array([self.qp_roll[i], 0, 0]),eta_i)
            fb = self.get_fb_i(g_ri, roll=True)
            self.beta[:,2*i+3] = self.get_beta_j(self.list_of_velocity[:, i * 2 + 3], self.centre_roll[:3], self.m_roll, self.J_roll,
                                      g_ri, fb)
        return self.list_of_velocity, self.beta, self.gamma, self.zeta

    def inverse_backward_NE(self):
#       if self.beta == None or self.gamma == None or self.list_of_velocity == None:
 #           self.get_velocity_gamma_beta(inverse_NE=False)
        self.list_I_c = np.zeros((6,6,self.nb*2+2))
        self.list_beta_c = np.zeros((6,self.nb*2+2))
        for i in reversed(range(self.nb)):
            if i == self.nb-1:
                self.list_I_c[:,:,2*i+3] = self.I_roll
                self.list_beta_c[:,2*i+3] = self.beta[:,2*i+3]
                Ad_grolli=self.Adj_grolli[:,:,i]
                self.list_I_c[:,:,2 * i + 2] = self.I_seg+np.dot(Ad_grolli.T,np.dot(self.I_roll,Ad_grolli))
                gamma_rolli = self.gamma[:, i * 2 + 2]
                self.list_beta_c[:,2 * i + 2] = self.beta[:,2 * i + 2]+np.dot(Ad_grolli.T,self.list_beta_c[:,2 * i + 3])\
                                              +np.dot(Ad_grolli.T,np.dot(self.list_I_c[:,:,2*i+3],gamma_rolli))
            else:
                self.list_I_c[:,:,2*i+3] = self.I_roll
                self.list_beta_c[:,2*i+3] = self.beta[:,2*i+3]
                Ad_gip1i = self.Adj_gip1i[:, :, i + 1]
                Ad_grolli = self.Adj_grolli[:,:,i]
                self.list_I_c[:,:,2*i+2] = self.I_seg+np.dot(Ad_grolli.T,np.dot(self.I_roll,Ad_grolli))\
                                       +np.dot(Ad_gip1i.T,np.dot(self.list_I_c[:,:,2*i+4],Ad_gip1i))
                gamma_rolli = self.gamma[:,2*i+2]
                gamma_ip1i = self.gamma[:,2*i+3]
                self.list_beta_c[:,2*i+2] = self.beta[:,2 * i + 2]+np.dot(Ad_grolli.T,self.list_beta_c[:,2 * i + 3])\
                                              +np.dot(Ad_grolli.T,np.dot(self.list_I_c[:,:,2*i+3],gamma_rolli))+np.dot(Ad_gip1i.T,self.list_beta_c[:,2 * i + 4])\
                                              +np.dot(Ad_gip1i.T,np.dot(self.list_I_c[:,:,2*i+4],gamma_ip1i))

        Ad_gip1i = self.Adj_gip1i[:, :, 0]
        self.list_I_c[:,:,1] = self.I_neck + np.dot(Ad_gip1i.T, np.dot(self.list_I_c[:,:,2], Ad_gip1i))

        self.list_beta_c[:,1] = self.beta[:,1] + np.dot(Ad_gip1i.T, self.list_beta_c[:,2]) \
                              +np.dot(Ad_gip1i.T, np.dot(self.list_I_c[:,:,2], self.gamma[:,1]))
        Ad_gnh = self.Adj_gnh
        self.list_I_c[:,:,0] = self.I_head + np.dot(Ad_gnh.T, np.dot(self.list_I_c[:,:,1], Ad_gnh))
        self.list_beta_c[:,0] = self.beta[:,0] + np.dot(Ad_gnh.T, self.list_beta_c[:,1]) \
                              + np.dot(Ad_gnh.T, np.dot(self.list_I_c[:,:,1], self.gamma[:,0]))
        return self.list_I_c,self.list_beta_c

    def inverse_forward_NE(self, qd, qdd):
        #####reload state
        self.qp_neck = qd[:2];
        self.qpp_neck = qdd[:2];
        for i in range(self.nb):
            self.qp_joint[i] = qd[2 + 2 * i];
            self.qp_roll[i] = qd[3 + 2 * i]
            self.qpp_joint[i] = qdd[2+ 2 * i];
            self.qpp_roll[i] = qdd[3 + 2 * i]

        ##################################
        self.get_velocity_gamma_beta(True)
        self.inverse_backward_NE()
        self.acc_head = -np.dot(np.linalg.inv(self.list_I_c[:,:,0]),self.list_beta_c[:,0])

        self.torque = np.zeros((2 * self.nb + 2))
        eta_p = np.dot(self.Adj_gnh,self.acc_head)+self.gamma[:,0]
        f_neck = np.dot(self.list_I_c[:,:,1],eta_p)+self.list_beta_c[:,1]
        self.torque[0] = f_neck[4];self.torque[1] = f_neck[5]
        for i in range(self.nb):
            Ad_gip1i = self.Adj_gip1i[:,:,i]
            eta_p = np.dot(Ad_gip1i, eta_p) + self.gamma[:, 2*i+1]
            f_neck = np.dot(self.list_I_c[:, :, 2*i+2], eta_p) + self.list_beta_c[:, 2*i+2]
            if i == 0:
                self.torque[2 * i + 2] = f_neck[4]
            else:
                self.torque[2 * i + 2] = f_neck[5]
            Ad_grolli = self.Adj_grolli[:, :, i]
            eta_roll_p = np.dot(Ad_grolli, eta_p) + self.gamma[:, 2 * i + 2]
            f_neck = np.dot(self.list_I_c[:, :, 2 * i + 3], eta_roll_p) + self.list_beta_c[:, 2 * i + 3]

            self.torque[2 * i + 3] = f_neck[3]
        return self.acc_head, self.torque,self.list_I_c[:,:,0],self.list_beta_c[:,0]

    def orin_algo_direct_dyn(self,q=None,qd=None,tau=None):
        if q is not None:
            self.geometrical_model(q=q)
        A = np.zeros((2*self.nb+8,2*self.nb+8))
        C = np.zeros((2*self.nb+8))
        temp_vel = copy.copy(self.vel_head)
        self.vel_head = np.zeros((6)) #eta_0 = 0
        ###### step 1
        acc_idm_1,tau_idm_1,A11,Cext_1= self.inverse_forward_NE(qd=np.zeros((2*self.nb+2)),qdd=np.zeros((2*self.nb+2)))
        A[:6,:6] = A11
        ###### step 2
        list_acc = np.zeros((6,2*self.nb+2))
        list_tau = np.zeros((2 * self.nb + 2, 2 * self.nb + 2))
        A12 = np.zeros((6,2 * self.nb + 2))
        A22 = np.zeros((2 * self.nb + 2,2 * self.nb + 2))
        for i in range(2*self.nb+2):
            temp = np.zeros((2 * self.nb + 2)); temp[i] = 1
            acc_idm_2, tau_idm_2, _, _ = self.inverse_forward_NE( qd=np.zeros((2 * self.nb + 2)),qdd=temp)
            list_acc[:,i] = acc_idm_2
            list_tau[:,i] = tau_idm_2
            A12[:,i] = -Cext_1-np.dot(A11,acc_idm_2)
        Cext_2 = tau_idm_1-np.dot(A12.T,acc_idm_1)
        for i in range(2*self.nb+2):
            A22[:,i] = list_tau[:,i]-np.dot(A12.T,list_acc[:,i])-Cext_2
        A[:6,6:]=A12
        A[6:,:6]=A12.T
        A[6:,6:] = A22
        '''
        for i in range(A22.shape[0]):
            for j in range(A22.shape[1]):
                print(i,j,A22[i,j],A22[j,i])
        '''
        ######## step 3
        self.vel_head = temp_vel
        acc_idm_3, tau_idm_3, _, _ = self.inverse_forward_NE(qd=qd,
                                                                    qdd=np.zeros((2*self.nb+2)))
        Cc_1 = -np.dot(A11,acc_idm_3)-Cext_1
        Cc_2 = tau_idm_3-np.dot(A12.T,acc_idm_3)-Cext_2
        C[:6] = Cc_1+Cext_1
        C[6:] = Cc_2+Cext_2
        torque = np.zeros((self.nb*2+8)); torque[6:] = tau
        etat = np.dot(np.linalg.inv(A),torque-C)
        self.acc_head = etat[:6]
        return etat[:6], etat[6:],A11,Cc_1

    def orin_algo_inverse_dyn(self, q=None, qd=None, qdd = None):
        if q is not None:
            self.geometrical_model(q = q)
        temp_vel = copy.copy(self.vel_head)
        self.vel_head = np.zeros((6))  # eta_0 = 0
        ###### step 1
        acc_idm_1, tau_idm_1, A11, Cext_1 = self.inverse_forward_NE(qd=np.zeros((2 * self.nb + 2)),
                                                                    qdd=np.zeros((2 * self.nb + 2)))
        ###### step 2
        A12 = np.zeros((6, 2 * self.nb + 2))

        for i in range(2 * self.nb + 2):
            temp = np.zeros((2 * self.nb + 2));
            temp[i] = 1
            acc_idm_2, tau_idm_2, _, _ = self.inverse_forward_NE(qd=np.zeros((2 * self.nb + 2)), qdd=temp)

            A12[:, i] = -Cext_1 - np.dot(A11, acc_idm_2)
        '''
        ######## step 3
        self.vel_head = temp_vel
        acc_idm_3, tau_idm_3, _, _ = self.inverse_forward_NE(qd=qd,
                                                             qdd=np.zeros((2 * self.nb + 2)))
        C = -np.dot(A11, acc_idm_3)


        temp = -np.dot(A12,qdd)-C
        self.acc_head = np.dot(np.linalg.inv(A11),temp) 
        return self.acc_head
        '''
        return A11,A12

    def motion_equation_head(self,qd,qdd):
        self.M = self.m_roll * (self.nb) + self.m_seg * (self.nb-1)+self.m_head+self.m_neck+self.m_last_seg
        self.qp_neck = qd[:2];
        self.qpp_neck = qdd[:2];
        for i in range(self.nb):
            self.qp_joint[i] = qd[2 + 2 * i];
            self.qp_roll[i] = qd[3 + 2 * i]
            self.qpp_joint[i] = qdd[2 + 2 * i];
            self.qpp_roll[i] = qdd[3 + 2 * i]

        def get_zeta_j_(eta_i,qd_i):
            Vi = eta_i[:3];
            Oi = eta_i[3:]
            adj_eta = np.zeros((6, 6))
            adj_eta[:3, :3] = self.skew_symetric_matrix(Oi)
            adj_eta[3:, 3:] = self.skew_symetric_matrix(Oi)
            adj_eta[:3, 3:] = self.skew_symetric_matrix(Vi)
            return np.dot(adj_eta,qd_i)

        def get_beta_j_(eta_i, I):
            ###### Centrifuge/Corioli
            Vi = eta_i[:3];
            Oi = eta_i[3:]
            adj_eta = np.zeros((6,6))
            adj_eta[:3,:3] = self.skew_symetric_matrix(Oi)
            adj_eta[3:, 3:] = self.skew_symetric_matrix(Oi)
            adj_eta[:3, 3:] = self.skew_symetric_matrix(Vi)
            beta = -np.dot(adj_eta.T,np.dot(I,eta_i))
            return beta


        def get_zeta_j(i_g_im1, qd_i, eta_im1):
            zeta_j = np.zeros((6))
            _0V1 = np.dot(i_g_im1[:3, :3], eta_im1[:3]);
            _0Omega1 = np.dot(i_g_im1[:3, :3], eta_im1[3:])
            temp = _0V1 + np.cross(i_g_im1[:3, 3], _0Omega1)
            zeta_j[:3] = np.cross(temp, qd_i)
            zeta_j[3:] = np.cross(_0Omega1, qd_i)
            return zeta_j

        def get_beta_j(eta_i, Si, Mi, I):
            ###### Centrifuge/Corioli
            Vi = eta_i[:3];
            Oi = eta_i[3:]
            beta = np.zeros((6))
            S_hat = self.skew_symetric_matrix(Si)
            #beta[:3] = Mi*np.cross(Oi,np.cross(Oi,Si))
            #beta[3:] = np.cross(Oi,np.dot(I,Oi))

            temp = -np.cross(Si,Oi)+Vi
            beta[:3] = Mi*np.cross(Oi,temp)
            beta[3:] = np.cross(Oi,np.dot(I,Oi))+Mi*np.cross(Si,np.cross(Oi,Vi))
          
            return beta
        I_virtuel = np.zeros((6,6))

        g01 = np.array([[cos(self.neck_config[0]), -sin(self.neck_config[0]), 0,self.P_head], [sin(self.neck_config[0]), cos(self.neck_config[0]), 0,0 ], [0, 0, 1,0],[0.,0.,0.,1.]])
        g10 = self.inverse_configuration(g01)
        Adj_gvirh = self.adjoint_matrix_2(g10)
        g01_ = np.array([[cos(self.neck_config[1]), 0, sin(self.neck_config[1]),0.], [0., 1., 0.,0.], [-sin(self.neck_config[1]),0, cos(self.neck_config[1]),0.],[0., 0., 0.,1.]])
        g1_0 = self.inverse_configuration(g01_)
        Adj_gnvir= self.adjoint_matrix_2(g1_0)

        Adj_gnh = np.dot(Adj_gnvir,Adj_gvirh)

        self.Adj_gip1i = np.zeros((6,6,self.nb))
        self.Adj_grolli = np.zeros((6,6,self.nb))
        for i in range(self.nb):
            giip1 = self.list_of_DH_matrices_joint[:, i * 4:i * 4 + 4]
            gip1i = self.inverse_configuration(giip1)  # ipi_T_i
            self.Adj_gip1i[:,:,i] = self.adjoint_matrix_2(gip1i)
            gi_roll = self.list_of_DH_matrices_roulie[:, i * 4:i * 4 + 4]
            groll_i = self.inverse_configuration(gi_roll)
            self.Adj_grolli[:,:,i] = self.adjoint_matrix_2(groll_i)
        #beta = get_beta_j(self.vel_head,self.centre_head[:3],self.m_head,self.I_head[3:,3:])#
        beta = get_beta_j_(self.vel_head,self.I_head)
        #print('beta_head',beta-beta_)
        self.list_I_c = np.zeros((6,6,self.nb+3))
        for i in reversed(range(self.nb)):
            if i == self.nb-1:
                Ad_grolli=self.Adj_grolli[:,:,i]
                self.list_I_c[:,:, i + 3] = self.I_seg+np.dot(Ad_grolli.T,np.dot(self.I_roll,Ad_grolli))

            else:
                Ad_gip1i = self.Adj_gip1i[:, :, i + 1]
                Ad_grolli = self.Adj_grolli[:,:,i]
                self.list_I_c[:,:,i+3] = self.I_seg+np.dot(Ad_grolli.T,np.dot(self.I_roll,Ad_grolli))\
                                       +np.dot(Ad_gip1i.T,np.dot(self.list_I_c[:,:,i+4],Ad_gip1i))
        self.list_I_c[:, :, 2] = self.I_neck + np.dot(self.Adj_gip1i[:, :, 0].T, np.dot(self.list_I_c[:, :, 3], self.Adj_gip1i[:, :, 0]))
        self.list_I_c[:,:,1] = I_virtuel + np.dot(Adj_gnvir.T, np.dot(self.list_I_c[:,:,2], Adj_gnvir))
        self.list_I_c[:,:,0] = self.I_head + np.dot(Adj_gvirh.T, np.dot(self.list_I_c[:,:,1],Adj_gvirh))

        eta_i = np.dot(Adj_gvirh, self.vel_head) + np.array([0, 0, 0, 0, 0, self.qp_neck[0]])
        #beta_i = get_beta_j(eta_i,self.centre_neck[:3],self.m_neck,self.I_neck[3:,3:])#
        beta_i=get_beta_j_(eta_i, I_virtuel)
        #print('beta_neck',beta_i-beta_i_)
        #zeta_i = get_zeta_j(g10, np.array([0, 0, self.qp_neck[0]]),self.vel_head)#
        zeta_i = get_zeta_j_(eta_i,np.array([0,0,0,0, 0, self.qp_neck[0]]))
        #print('zeta_neck',zeta_i-zeta_i_)
        zeta_i = np.dot(self.list_I_c[:,:,1], zeta_i)
        beta = beta + np.dot(Adj_gvirh.T,(beta_i+zeta_i))
        M_q = np.zeros((6,self.nb*2+2))
        Ai = np.array([0.,0.,0.,0.,0.,1.])
        M_q[:,0] = np.dot(Adj_gvirh.T,np.dot(self.list_I_c[:,:,1],Ai))
        ######### beta

        eta_i =  np.dot(Adj_gnvir, eta_i) + np.array([0, 0, 0, 0, self.qp_neck[1], 0])
        beta_i = get_beta_j_(eta_i, self.I_neck)
        zeta_i = get_zeta_j_(eta_i, np.array([0, 0, 0, 0, self.qp_neck[1],0]))
        zeta_i = np.dot(self.list_I_c[:, :, 2], zeta_i)
        beta = beta + np.dot(Adj_gnh.T, (beta_i + zeta_i))
        Ai =  np.array([0., 0., 0., 0., 1., 0.])
        M_q[:, 1] = np.dot(Adj_gnh.T, np.dot(self.list_I_c[:, :, 2], Ai))

        eta_im1 = eta_i

        for i in range(self.nb):
            i_g_G = self.inverse_configuration(self.list_of_DH_matrices_joint_in_R0[:, i * 4:i * 4 + 4])
            i_g_h = np.dot(i_g_G,self.list_of_DH_matrices_head)
            Ad_igh = self.adjoint_matrix_2(i_g_h) # for transfer the force
            if i == 0:
                eta_i = np.dot(self.Adj_gip1i[:,:,i], eta_im1) + np.array([0., 0., 0., 0., self.qp_joint[i], 0.])
                #zeta_i = get_zeta_j(gip1i, np.array([0., self.qp_joint[i],0.]), eta_im1)#
                zeta_i=get_zeta_j_(eta_i,np.array([0.,0.,0.,0., self.qp_joint[i],0.]))
                #print('zeta_joint',i,zeta_i-zeta_i_)
                zeta_i = np.dot(self.list_I_c[:,:,3+i], zeta_i)
                M_q[:, 2 * i + 2] = np.dot(Ad_igh.T, np.dot(self.list_I_c[:,:,i+3], np.array([0., 0., 0., 0., 1., 0.])))
            else:
                eta_i = np.dot(self.Adj_gip1i[:,:,i], eta_im1) + np.array([0., 0., 0., 0., 0, self.qp_joint[i]])
                #zeta_i = get_zeta_j(gip1i, np.array([0., 0., self.qp_joint[i]]), eta_im1)#
                zeta_i=get_zeta_j_(eta_i,np.array([0.,0.,0.,0., 0., self.qp_joint[i]]))
                #print('zeta_joint',i,zeta_i-zeta_i_)
                zeta_i = np.dot(self.list_I_c[:,:,3+i], zeta_i)
                M_q[:, 2 * i + 2] = np.dot(Ad_igh.T, np.dot(self.list_I_c[:,:,i+3], np.array([0., 0., 0., 0., 0., 1.])))
            if i == self.nb - 1:
                #beta_i = get_beta_j(eta_i, self.centre_last_seg[:3], self.m_last_seg, self.I_seg[3:,3:])#
                beta_i = get_beta_j_(eta_i,self.I_seg)
            else:
                #beta_i = get_beta_j(eta_i, self.centre_segment[:3], self.m_seg, self.I_seg[3:,3:])#
                beta_i = get_beta_j_(eta_i,self.I_seg)
            #print('beta_joint', i, beta_i - beta_i_)
            beta = beta + np.dot(Ad_igh.T,(beta_i + zeta_i))
            eta_im1 = eta_i

            ri_g_G = self.inverse_configuration(self.list_of_DH_matrices_roulie_in_R0[:, i * 4:i * 4 + 4])
            ri_g_h = np.dot(ri_g_G, self.list_of_DH_matrices_head)
            Ad_righ = self.adjoint_matrix_2(ri_g_h)  # for transfer the force

            eta_ir = np.dot(self.Adj_grolli[:,:,i], eta_im1) + np.array([0.,0.,0.,self.qp_roll[i],0.,0.])
            #zeta_i = get_zeta_j(groll_i, np.array([self.qp_roll[i],0.,0.]), eta_im1)#
            zeta_i = get_zeta_j_(eta_ir,np.array([0.,0.,0.,self.qp_roll[i],0.,0.]))
            #print('zeta_roll',i,zeta_i-zeta_i_)

            zeta_i = np.dot(self.I_roll, zeta_i)

            #beta_i = get_beta_j(eta_ir, self.centre_roll[:3], self.m_roll, self.I_roll[3:,3:])#
            beta_i = get_beta_j_(eta_ir, self.I_roll)
            #print('beta_roll',i,beta_i-beta_i_)
            M_q[:, 2 * i + 3] = np.dot(Ad_righ.T, np.dot(self.I_roll, np.array([0., 0., 0., 1., 0., 0.])))
            beta = beta + np.dot(Ad_righ.T, (beta_i + zeta_i))

        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        W = wb.reshape((6)) + wg
        Ad_g0 = self.adjoint_matrix_2(self.list_of_DH_matrices_head)
        W0 = np.dot(Ad_g0.T, W)
        dot_eta = np.dot(np.linalg.inv(self.list_I_c[:,:,0]),(-beta-np.dot(M_q,qdd)+W0))#-beta-np.dot(M_q,qdd)+W0
        return dot_eta,self.list_I_c[:,:,0]

    def motion_equation_head_(self, qd, qdd):
        self.M = self.m_roll * (self.nb) + self.m_seg * (self.nb - 1) + self.m_head + self.m_neck + self.m_last_seg
        self.qp_neck = qd[:2];
        self.qpp_neck = qdd[:2];
        for i in range(self.nb):
            self.qp_joint[i] = qd[2 + 2 * i];
            self.qp_roll[i] = qd[3 + 2 * i]
            self.qpp_joint[i] = qdd[2 + 2 * i];
            self.qpp_roll[i] = qdd[3 + 2 * i]

        def get_zeta_j_(eta_i, qd_i):
            Vi = eta_i[:3];
            Oi = eta_i[3:]
            adj_eta = np.zeros((6, 6))
            adj_eta[:3, :3] = self.skew_symetric_matrix(Oi)
            adj_eta[3:, 3:] = self.skew_symetric_matrix(Oi)
            adj_eta[:3, 3:] = self.skew_symetric_matrix(Vi)
            return np.dot(adj_eta, qd_i)

        def get_beta_j_(eta_i, I):
            ###### Centrifuge/Corioli
            Vi = eta_i[:3];
            Oi = eta_i[3:]
            adj_eta = np.zeros((6, 6))
            adj_eta[:3, :3] = self.skew_symetric_matrix(Oi)
            adj_eta[3:, 3:] = self.skew_symetric_matrix(Oi)
            adj_eta[:3, 3:] = self.skew_symetric_matrix(Vi)
            beta = -np.dot(adj_eta.T, np.dot(I, eta_i))
            return beta

        def get_zeta_j(i_g_im1, qd_i, eta_im1):
            zeta_j = np.zeros((6))
            _0V1 = np.dot(i_g_im1[:3, :3], eta_im1[:3]);
            _0Omega1 = np.dot(i_g_im1[:3, :3], eta_im1[3:])
            temp = _0V1 + np.cross(i_g_im1[:3, 3], _0Omega1)
            zeta_j[:3] = np.cross(temp, qd_i)
            zeta_j[3:] = np.cross(_0Omega1, qd_i)
            return zeta_j

        def get_beta_j(eta_i, Si, Mi, I):
            ###### Centrifuge/Corioli
            Vi = eta_i[:3];
            Oi = eta_i[3:]
            beta = np.zeros((6))
            S_hat = self.skew_symetric_matrix(Si)
            # beta[:3] = Mi*np.cross(Oi,np.cross(Oi,Si))
            # beta[3:] = np.cross(Oi,np.dot(I,Oi))

            temp = -np.cross(Si, Oi) + Vi
            beta[:3] = Mi * np.cross(Oi, temp)
            beta[3:] = np.cross(Oi, np.dot(I, Oi)) + Mi * np.cross(Si, np.cross(Oi, Vi))

            return beta

        I_virtuel = np.zeros((6, 6))

        g01 = np.array([[cos(self.neck_config[0]), -sin(self.neck_config[0]), 0, self.P_head],
                        [sin(self.neck_config[0]), cos(self.neck_config[0]), 0, 0], [0, 0, 1, 0], [0., 0., 0., 1.]])
        g10 = self.inverse_configuration(g01)
        Adj_gvirh = self.adjoint_matrix_2(g10)
        g01_ = np.array([[cos(self.neck_config[1]), 0, sin(self.neck_config[1]), 0.], [0., 1., 0., 0.],
                         [-sin(self.neck_config[1]), 0, cos(self.neck_config[1]), 0.], [0., 0., 0., 1.]])
        g1_0 = self.inverse_configuration(g01_)
        Adj_gnvir = self.adjoint_matrix_2(g1_0)

        Adj_gnh = np.dot(Adj_gnvir, Adj_gvirh)

        self.Adj_gip1i = np.zeros((6, 6, self.nb))
        self.Adj_grolli = np.zeros((6, 6, self.nb))
        for i in range(self.nb):
            giip1 = self.list_of_DH_matrices_joint[:, i * 4:i * 4 + 4]
            gip1i = self.inverse_configuration(giip1)  # ipi_T_i
            self.Adj_gip1i[:, :, i] = self.adjoint_matrix_2(gip1i)
            gi_roll = self.list_of_DH_matrices_roulie[:, i * 4:i * 4 + 4]
            groll_i = self.inverse_configuration(gi_roll)
            self.Adj_grolli[:, :, i] = self.adjoint_matrix_2(groll_i)
        # beta = get_beta_j(self.vel_head,self.centre_head[:3],self.m_head,self.I_head[3:,3:])#
        beta = get_beta_j_(self.vel_head, self.I_head)
        # print('beta_head',beta-beta_)
        self.list_I_c = np.zeros((6, 6, self.nb + 3))
        for i in reversed(range(self.nb)):
            if i == self.nb - 1:
                Ad_grolli = self.Adj_grolli[:, :, i]
                self.list_I_c[:, :, i + 3] = self.I_seg + np.dot(Ad_grolli.T, np.dot(self.I_roll, Ad_grolli))

            else:
                Ad_gip1i = self.Adj_gip1i[:, :, i + 1]
                Ad_grolli = self.Adj_grolli[:, :, i]
                self.list_I_c[:, :, i + 3] = self.I_seg + np.dot(Ad_grolli.T, np.dot(self.I_roll, Ad_grolli)) \
                                             + np.dot(Ad_gip1i.T, np.dot(self.list_I_c[:, :, i + 4], Ad_gip1i))
        self.list_I_c[:, :, 2] = self.I_neck + np.dot(self.Adj_gip1i[:, :, 0].T,
                                                      np.dot(self.list_I_c[:, :, 3], self.Adj_gip1i[:, :, 0]))
        self.list_I_c[:, :, 1] = I_virtuel + np.dot(Adj_gnvir.T, np.dot(self.list_I_c[:, :, 2], Adj_gnvir))
        self.list_I_c[:, :, 0] = self.I_head + np.dot(Adj_gvirh.T, np.dot(self.list_I_c[:, :, 1], Adj_gvirh))

        eta_i = np.dot(Adj_gvirh, self.vel_head) + np.array([0, 0, 0, 0, 0, self.qp_neck[0]])
        # beta_i = get_beta_j(eta_i,self.centre_neck[:3],self.m_neck,self.I_neck[3:,3:])#
        beta_i = get_beta_j_(eta_i, I_virtuel)
        # print('beta_neck',beta_i-beta_i_)
        # zeta_i = get_zeta_j(g10, np.array([0, 0, self.qp_neck[0]]),self.vel_head)#
        zeta_i = get_zeta_j_(eta_i, np.array([0, 0, 0, 0, 0, self.qp_neck[0]]))
        # print('zeta_neck',zeta_i-zeta_i_)
        zeta_i = np.dot(self.list_I_c[:, :, 1], zeta_i)
        beta = beta + np.dot(Adj_gvirh.T, (beta_i + zeta_i))
        M_q = np.zeros((6, self.nb * 2 + 2))
        Ai = np.array([0., 0., 0., 0., 0., 1.])
        M_q[:, 0] = np.dot(Adj_gvirh.T, np.dot(self.list_I_c[:, :, 1], Ai))
        ######### beta

        eta_i = np.dot(Adj_gnvir, eta_i) + np.array([0, 0, 0, 0, self.qp_neck[1], 0])
        beta_i = get_beta_j_(eta_i, self.I_neck)
        zeta_i = get_zeta_j_(eta_i, np.array([0, 0, 0, 0, self.qp_neck[1], 0]))
        zeta_i = np.dot(self.list_I_c[:, :, 2], zeta_i)
        beta = beta + np.dot(Adj_gnh.T, (beta_i + zeta_i))
        Ai = np.array([0., 0., 0., 0., 1., 0.])
        M_q[:, 1] = np.dot(Adj_gnh.T, np.dot(self.list_I_c[:, :, 2], Ai))

        eta_im1 = eta_i

        for i in range(self.nb):
            i_g_G = self.inverse_configuration(self.list_of_DH_matrices_joint_in_R0[:, i * 4:i * 4 + 4])
            i_g_h = np.dot(i_g_G, self.list_of_DH_matrices_head)
            Ad_igh = self.adjoint_matrix_2(i_g_h)  # for transfer the force
            if i == 0:
                eta_i = np.dot(self.Adj_gip1i[:, :, i], eta_im1) + np.array([0., 0., 0., 0., self.qp_joint[i], 0.])
                # zeta_i = get_zeta_j(gip1i, np.array([0., self.qp_joint[i],0.]), eta_im1)#
                zeta_i = get_zeta_j_(eta_i, np.array([0., 0., 0., 0., self.qp_joint[i], 0.]))
                # print('zeta_joint',i,zeta_i-zeta_i_)
                zeta_i = np.dot(self.list_I_c[:, :, 3 + i], zeta_i)
                M_q[:, 2 * i + 2] = np.dot(Ad_igh.T,
                                           np.dot(self.list_I_c[:, :, i + 3], np.array([0., 0., 0., 0., 1., 0.])))
            else:
                eta_i = np.dot(self.Adj_gip1i[:, :, i], eta_im1) + np.array([0., 0., 0., 0., 0, self.qp_joint[i]])
                # zeta_i = get_zeta_j(gip1i, np.array([0., 0., self.qp_joint[i]]), eta_im1)#
                zeta_i = get_zeta_j_(eta_i, np.array([0., 0., 0., 0., 0., self.qp_joint[i]]))
                # print('zeta_joint',i,zeta_i-zeta_i_)
                zeta_i = np.dot(self.list_I_c[:, :, 3 + i], zeta_i)
                M_q[:, 2 * i + 2] = np.dot(Ad_igh.T,
                                           np.dot(self.list_I_c[:, :, i + 3], np.array([0., 0., 0., 0., 0., 1.])))
            if i == self.nb - 1:
                # beta_i = get_beta_j(eta_i, self.centre_last_seg[:3], self.m_last_seg, self.I_seg[3:,3:])#
                beta_i = get_beta_j_(eta_i, self.I_seg)
            else:
                # beta_i = get_beta_j(eta_i, self.centre_segment[:3], self.m_seg, self.I_seg[3:,3:])#
                beta_i = get_beta_j_(eta_i, self.I_seg)
            # print('beta_joint', i, beta_i - beta_i_)
            beta = beta + np.dot(Ad_igh.T, (beta_i + zeta_i))
            eta_im1 = eta_i

            ri_g_G = self.inverse_configuration(self.list_of_DH_matrices_roulie_in_R0[:, i * 4:i * 4 + 4])
            ri_g_h = np.dot(ri_g_G, self.list_of_DH_matrices_head)
            Ad_righ = self.adjoint_matrix_2(ri_g_h)  # for transfer the force

            eta_ir = np.dot(self.Adj_grolli[:, :, i], eta_im1) + np.array([0., 0., 0., self.qp_roll[i], 0., 0.])
            # zeta_i = get_zeta_j(groll_i, np.array([self.qp_roll[i],0.,0.]), eta_im1)#
            zeta_i = get_zeta_j_(eta_ir, np.array([0., 0., 0., self.qp_roll[i], 0., 0.]))
            # print('zeta_roll',i,zeta_i-zeta_i_)

            zeta_i = np.dot(self.I_roll, zeta_i)

            # beta_i = get_beta_j(eta_ir, self.centre_roll[:3], self.m_roll, self.I_roll[3:,3:])#
            beta_i = get_beta_j_(eta_ir, self.I_roll)
            # print('beta_roll',i,beta_i-beta_i_)
            M_q[:, 2 * i + 3] = np.dot(Ad_righ.T, np.dot(self.I_roll, np.array([0., 0., 0., 1., 0., 0.])))
            beta = beta + np.dot(Ad_righ.T, (beta_i + zeta_i))

        wb = self.get_list_buoyancy_wrench()
        wg = self.gravity_wrench()
        W = wb.reshape((6)) + wg
        Ad_g0 = self.adjoint_matrix_2(self.list_of_DH_matrices_head)

        W0 = np.dot(Ad_g0.T, W)
        #dot_eta = np.dot(np.linalg.inv(self.list_I_c[:, :, 0]),
                         #(-beta - np.dot(M_q, qdd) + W0))  # -beta-np.dot(M_q,qdd)+W0
        return self.list_I_c[:, :, 0], beta,np.dot(M_q, qdd),-W0

    def IDM_one_module(self,qd,qdd):
        def beta(gi,Sj,mj,eta_j,Ij,roll,last_seg,neck,head):
            Fb_i = self.get_fb_i(g=gi, head=head, neck=neck, last_seg=last_seg, roll=roll)
            S0 = np.dot(gi[:3,:3],Sj)+gi[:3,3]

            Fg = np.array([0.,0.,-mj*self.g])
            Fg_i = np.zeros((6))
            Fg_i[:3] = Fg;
            Fg_i[3:] = np.cross(S0,Fg)
            F_i = Fg_i+Fb_i
            Adgi = self.adjoint_matrix_2(gi)
            F_i_inFi = np.dot(Adgi.T,F_i)
            Vi = eta_j[:3];
            Oi = eta_j[3:]
            beta = np.zeros((6))
            S_hat = self.skew_symetric_matrix(Sj)
            #beta[:3] = mj * np.cross(Oi, np.cross(Oi, Sj))
            #beta[3:] = np.cross(Oi, np.dot(Ij, Oi))

            temp = -np.cross(Sj,Oi)+Vi
            beta[:3] = mj*np.cross(Oi,temp)
            beta[3:] = np.cross(Oi,np.dot(Ij,Oi))+mj*np.cross(Sj,np.cross(Oi,Vi))
            '''
            #amphibot
            temp1 = Mi * np.dot(S_hat, Oi)
            beta[:3] = -np.cross(Oi, temp1) + Mi * np.cross(Oi, Vi)
            temp2 = Mi * np.cross(Oi, Vi)
            beta[3:] = np.cross(Oi, np.dot(I, Oi)) + np.dot(S_hat, temp2)

            temp = -np.cross(Si, Oi) + Vi
            beta[:3] = Mi * np.cross(Oi, temp)
            temp2 = np.dot(I, Oi) + Mi * np.cross(Si, Vi)
            beta[3:] = np.cross(Oi, temp2)
            '''
            return beta-F_i_inFi,Fg_i,Fb_i

        def get_zeta_j(i_g_im1, qd_i, eta_im1):
            zeta_j = np.zeros((6))
            _0V1 = np.dot(i_g_im1[:3, :3], eta_im1[:3]);
            _0Omega1 = np.dot(i_g_im1[:3, :3], eta_im1[3:])
            temp = _0V1 + np.cross(i_g_im1[:3, 3], _0Omega1)
            zeta_j[:3] = np.cross(temp, qd_i)
            zeta_j[3:] = np.cross(_0Omega1, qd_i)
            return zeta_j
        # recursion explicite
        self.qp_neck = qd[:2];
        self.qpp_neck = qdd[:2];
        for i in range(self.nb):
            self.qp_joint[i] = qd[2 + 2 * i];
            self.qp_roll[i] = qd[3 + 2 * i]
            self.qpp_joint[i] = qdd[2 + 2 * i];
            self.qpp_roll[i] = qdd[3 + 2 * i]
        eta_0 = self.vel_head
        beta_0,Fg0,Fb0 = beta(self.list_of_DH_matrices_head,self.centre_head[:3],self.m_head,eta_0,self.I_head[3:,3:],False,False,False,True)
        Adj_gnh = self.adjoint_matrix_2(self.inverse_configuration(self.DH_matrix_neck))
        eta_1 = np.dot(Adj_gnh, eta_0)+np.array([0.,0.,0.,0., self.qp_neck[0], self.qp_neck[1]])
        gamma1 = get_zeta_j(self.inverse_configuration(self.DH_matrix_neck),np.array([0, self.qp_neck[0], self.qp_neck[1]]), eta_0)+ np.array([0., 0., 0., 0., self.qpp_neck[0], self.qpp_neck[1]])
        beta1,Fg1,Fb1 = beta(self.DH_matrix_neck_in_R0, self.centre_neck[:3],self.m_neck, eta_1,self.I_neck[3:,3:],False,False,True,False)
        Adj_gsn = self.adjoint_matrix_2(self.inverse_configuration(self.list_of_DH_matrices_joint))
        eta_2 = np.dot(Adj_gsn, eta_1) + np.array([0., 0., 0., 0., self.qp_joint[0], 0.])
        beta2,Fg2,Fb2  = beta(self.list_of_DH_matrices_joint_in_R0, self.centre_last_seg[:3], self.m_last_seg, eta_2, self.I_seg[3:, 3:], False,
                     True, False, False)
        gamma2 = get_zeta_j(self.inverse_configuration(self.list_of_DH_matrices_joint),np.array([0.,qd[2],0.]),eta_1)+np.array([0.,0.,0.,0.,qdd[2],0.])
        Adj_grs = self.adjoint_matrix_2(self.inverse_configuration(self.list_of_DH_matrices_roulie))
        eta_3 = np.dot(Adj_grs, eta_2) + np.array([0., 0., 0., self.qp_roll[0], 0., 0.])
        beta3,Fg3,Fb3  = beta(self.list_of_DH_matrices_roulie_in_R0, self.centre_roll[:3], self.m_roll, eta_3,
                     self.I_roll[3:, 3:], True, False, False, False)
        gamma3 = get_zeta_j(self.inverse_configuration(self.list_of_DH_matrices_roulie),np.array([qd[3],0.,0.]),eta_2)+np.array([0.,0.,0.,qdd[3],0.,0.])
        I_head = self.I_head+np.dot(Adj_gnh.T,np.dot(self.I_neck,Adj_gnh))
        ghs = np.dot(self.DH_matrix_neck,self.list_of_DH_matrices_joint)
        Adj_gsh = self.adjoint_matrix_2(self.inverse_configuration(ghs))
        ghr = np.dot(ghs, self.list_of_DH_matrices_roulie)
        Adj_grh = self.adjoint_matrix_2(self.inverse_configuration(ghr))
        I_head+=np.dot(Adj_gsh.T,np.dot(self.I_seg,Adj_gsh))+np.dot(Adj_grh.T,np.dot(self.I_roll,Adj_grh))

        I2c = self.I_seg + np.dot(Adj_grs.T,np.dot(self.I_roll,Adj_grs))
        I1c = self.I_neck + np.dot(Adj_gsn.T,np.dot(I2c,Adj_gsn))

        beta2star = beta2+np.dot(Adj_grs.T,beta3)+np.dot(Adj_grs.T,np.dot(self.I_roll,gamma3))
        beta1star = beta1+ np.dot(Adj_gsn.T,beta2star)+np.dot(Adj_gsn.T,np.dot(I2c,gamma2))
        beta0star = beta_0+np.dot(Adj_gnh.T,beta1star)+np.dot(Adj_gnh.T,np.dot(I1c,gamma1))

        return  -np.dot(np.linalg.inv(I_head),beta0star),I_head

    def center_of_masse(self,X,t,a,T,X2):
        Px = np.zeros((t.shape[0]))
        Py = np.zeros((t.shape[0]))
        Pz = np.zeros((t.shape[0]))
        Euler_anglex = np.zeros((t.shape[0]))
        Euler_angley = np.zeros((t.shape[0]))
        Euler_anglez = np.zeros((t.shape[0]))
        quantity_mvt = np.zeros((t.shape[0],6))
        list_beta = np.zeros((t.shape[0],6))
        Px2 = np.zeros((t.shape[0]))
        Py2 = np.zeros((t.shape[0]))
        Pz2 = np.zeros((t.shape[0]))
        Euler_anglex2 = np.zeros((t.shape[0]))
        Euler_angley2 = np.zeros((t.shape[0]))
        Euler_anglez2 = np.zeros((t.shape[0]))
        quantity_mvt2 = np.zeros((t.shape[0], 6))
        #list_beta = np.zeros((t.shape[0], 6))
        def f(t, a,T):
            t = t % T
            return (a[0] * t ** 7 + a[1] * t ** 6 + a[2] * t ** 5 + a[3] * t ** 4 + a[4] * t ** 3 + a[5] * t ** 2 + a[
                6] * t + a[
                        7])  # *st

        def df(t, a,T):
            t = t % T
            return 7 * a[0] * t ** 6 + 6 * a[1] * t ** 5 + 5 * a[2] * t ** 4 + 4 * a[3] * t ** 3 + 3 * a[
                4] * t ** 2 + 2 * a[5] * t + a[
                       6]  # (5*a[0]*t**4+4*a[1]*t**3+3*a[2]*t**2+a[3])*st+2*pi/1*(a[0]*t**5+a[1]*t**4+a[2]*t**3+a[3]*t)*ct

        def ddf(t, a,T):
            t = t % T
            return 42 * a[0] * t ** 5 + 30 * a[1] * t ** 4 + 20 * a[2] * t ** 3 + 12 * a[3] * t ** 2 + 6 * a[
                4] * t + 2 * a[5]  # temp1+temp2+temp3+temp4
        for i in range(t.shape[0]):
            x=X[:,i]
            P0 = x[:3]
            # x[3:7] = x[3:7]/np.linalg.norm(x[3:7]);
            quat = x[3:7]
            quat = quat / np.linalg.norm(quat)
            Q1 = quat[0];
            Q2 = quat[1];
            Q3 = quat[2];
            Q4 = quat[3];
            self.vel_head = x[7:]
            # O1,O2,O3 = eta0[3],eta0[4],eta0[5]
            R = np.array([[2 * (Q1 ** 2 + Q2 ** 2) - 1, 2 * (Q2 * Q3 - Q1 * Q4), 2 * (Q2 * Q4 + Q1 * Q3)],
                          [2 * (Q2 * Q3 + Q1 * Q4), 2 * (Q1 ** 2 + Q3 ** 2) - 1, 2 * (Q3 * Q4 - Q1 * Q2)],
                          [2 * (Q2 * Q4 - Q1 * Q3), 2 * (Q3 * Q4 + Q1 * Q2), 2 * (Q1 ** 2 + Q4 ** 2) - 1]])
            self.list_of_DH_matrices_head[:3, :3] = R;
            self.list_of_DH_matrices_head[:3, 3] = P0

            G = Rotation.from_matrix(R)
            angle_euler = G.as_euler('xyz', degrees=True)
            Euler_anglex[i] = angle_euler[0]
            Euler_angley[i] = angle_euler[1]
            Euler_anglez[i] = angle_euler[2]

            self.neck_config = np.ones((2)) * f(t[i], a,T)/2
            #self.neck_config[0] = 0.
            self.joint_config = np.ones((self.nb)) * f(t[i], a,T)
            self.joint_config[0] = -f(t[i], a,T)/2
            self.antirouli_config = np.ones((self.nb)) * f(t[i], a,T)

            qnd = np.ones((2)) * df(t[i], a, T)/2
            #qnd[0] = 0.
            qjd = np.ones((self.nb)) * df(t[i], a, T)
            qjd[0] = -df(t[i], a, T)/2
            qrd = np.ones((self.nb)) * df(t[i], a, T)

            qndd = np.ones((2)) * ddf(t[i], a, T)/2
            #qndd[0] = 0.
            qjdd = np.ones((self.nb)) * ddf(t[i], a, T)
            qjdd[0] = -ddf(t[i], a, T)/2
            qrdd = np.ones((self.nb)) * ddf(t[i], a, T)
            qd = np.zeros((2 * self.nb + 2))
            qdd = np.zeros((2 * self.nb + 2))
            qd[:2] = qnd
            qdd[:2] = qndd
            for j in range(self.nb):
                qd[2 * j + 2] = qjd[j]
                qd[2 * j + 3] = qrd[j]
                qdd[2 * j + 2] = qjdd[j]
                qdd[2 * j + 3] = qrdd[j]
            self.qp_neck = qd[:2];
            for j in range(self.nb):
                self.qp_joint[j] = qd[2 + 2 * j];
                self.qp_roll[j] = qd[3 + 2 * j]

            self.geometrical_model()
            '''
            etad,I,beta = self.motion_equation_head_(qd,qdd)
            list_beta[i,:] = beta
            '''
            MP = self.m_head*(np.dot(self.list_of_DH_matrices_head[:3,:3],self.centre_head[:3])+self.list_of_DH_matrices_head[:3,3])
            MP += self.m_neck * (np.dot(self.DH_matrix_neck_in_R0[:3, :3],
                                       self.centre_neck[:3]) + self.DH_matrix_neck_in_R0[:3, 3])
            g01 = self.DH_matrix_neck
            g10 = self.inverse_configuration(g01)
            self.Adj_gnh = self.adjoint_matrix_2(g10)
            quantity = np.dot(self.I_head,self.vel_head) #quantité de mouvement
            eta_i = np.dot(self.Adj_gnh, self.vel_head) + np.array([0, 0, 0, 0, self.qp_neck[0], self.qp_neck[1]])
            quantity+=np.dot(self.Adj_gnh.T,np.dot(self.I_neck,eta_i))
            eta_im1 = eta_i
            for k in range(self.nb):
                giip1 = self.list_of_DH_matrices_joint[:, k * 4:k * 4 + 4]
                gip1i = self.inverse_configuration(giip1)
                Ad_gip1i = self.adjoint_matrix_2(gip1i)  # for transfer the velocity

                i_g_G = self.inverse_configuration(self.list_of_DH_matrices_joint_in_R0[:, k * 4:k * 4 + 4])
                i_g_h = np.dot(i_g_G, self.list_of_DH_matrices_head)
                Ad_igh = self.adjoint_matrix_2(i_g_h)  # for transfer the force
                if k == 0:
                    eta_i = np.dot(Ad_gip1i, eta_im1) + np.array([0., 0., 0., 0., self.qp_joint[k], 0.])
                else:
                    eta_i = np.dot(Ad_gip1i, eta_im1) + np.array([0, 0, 0, 0, 0, self.qp_joint[k]])
                quantity += np.dot(Ad_igh.T, np.dot(self.I_seg, eta_i))

                eta_im1 = eta_i

                gi_roll = self.list_of_DH_matrices_roulie[:, k * 4:k * 4 + 4]
                groll_i = self.inverse_configuration(gi_roll)
                Ad_groll_i = self.adjoint_matrix_2(groll_i)

                ri_g_G = self.inverse_configuration(self.list_of_DH_matrices_roulie_in_R0[:, k * 4:k * 4 + 4])
                ri_g_h = np.dot(ri_g_G, self.list_of_DH_matrices_head)
                Ad_righ = self.adjoint_matrix_2(ri_g_h)  # for transfer the force

                eta_ir = np.dot(Ad_groll_i, eta_im1) + np.array([0., 0., 0., self.qp_roll[k], 0., 0.])
                quantity+=np.dot(Ad_righ.T,np.dot(self.I_roll,eta_ir))


                gs = self.list_of_DH_matrices_joint_in_R0[:,4*k:4*k+4]
                gr = self.list_of_DH_matrices_roulie_in_R0[:,4*k:4*k+4]
                if k != self.nb-1:
                    MP += self.m_seg * (np.dot(gs[:3, :3],self.centre_segment[:3]) + gs[:3, 3])
                else:
                    MP += self.m_last_seg * (np.dot(gs[:3, :3],self.centre_last_seg[:3]) + gs[:3, 3])
                MP +=  self.m_roll * (np.dot(gr[:3, :3],self.centre_last_seg[:3]) + gr[:3, 3])
            MP = MP/self.M
            Px[i] = MP[0] * 1000
            Py[i] = MP[1] * 1000
            Pz[i] = MP[2] * 1000
            quantity_mvt[i,:] = quantity
        for i in range(t.shape[0]):
            x=X2[:,i]
            P0 = x[:3]
            # x[3:7] = x[3:7]/np.linalg.norm(x[3:7]);
            quat = x[3:7]
            quat = quat / np.linalg.norm(quat)
            Q1 = quat[0];
            Q2 = quat[1];
            Q3 = quat[2];
            Q4 = quat[3];
            self.vel_head = x[7:]
            # O1,O2,O3 = eta0[3],eta0[4],eta0[5]
            R = np.array([[2 * (Q1 ** 2 + Q2 ** 2) - 1, 2 * (Q2 * Q3 - Q1 * Q4), 2 * (Q2 * Q4 + Q1 * Q3)],
                          [2 * (Q2 * Q3 + Q1 * Q4), 2 * (Q1 ** 2 + Q3 ** 2) - 1, 2 * (Q3 * Q4 - Q1 * Q2)],
                          [2 * (Q2 * Q4 - Q1 * Q3), 2 * (Q3 * Q4 + Q1 * Q2), 2 * (Q1 ** 2 + Q4 ** 2) - 1]])
            self.list_of_DH_matrices_head[:3, :3] = R;
            self.list_of_DH_matrices_head[:3, 3] = P0

            G = Rotation.from_matrix(R)
            angle_euler = G.as_euler('xyz', degrees=True)
            Euler_anglex2[i] = angle_euler[0]
            Euler_angley2[i] = angle_euler[1]
            Euler_anglez2[i] = angle_euler[2]

            self.neck_config = -np.ones((2)) * f(t[i], a,T)/2
            #self.neck_config[0] = 0.
            self.joint_config = -np.ones((self.nb)) * f(t[i], a,T)
            self.joint_config[0] = f(t[i], a,T)/2
            self.antirouli_config = -np.ones((self.nb)) * f(t[i], a,T)

            qnd = -np.ones((2)) * df(t[i], a, T)/2
            #qnd[0] = 0.
            qjd = -np.ones((self.nb)) * df(t[i], a, T)
            qjd[0] = df(t[i], a, T)/2
            qrd = -np.ones((self.nb)) * df(t[i], a, T)

            qndd = -np.ones((2)) * ddf(t[i], a, T)/2
            #qndd[0] = 0.
            qjdd = -np.ones((self.nb)) * ddf(t[i], a, T)
            qjdd[0] = ddf(t[i], a, T)/2
            qrdd = -np.ones((self.nb)) * ddf(t[i], a, T)
            qd = np.zeros((2 * self.nb + 2))
            qdd = np.zeros((2 * self.nb + 2))
            qd[:2] = qnd
            qdd[:2] = qndd
            for j in range(self.nb):
                qd[2 * j + 2] = qjd[j]
                qd[2 * j + 3] = qrd[j]
                qdd[2 * j + 2] = qjdd[j]
                qdd[2 * j + 3] = qrdd[j]
            self.qp_neck = qd[:2];
            for j in range(self.nb):
                self.qp_joint[j] = qd[2 + 2 * j];
                self.qp_roll[j] = qd[3 + 2 * j]

            self.geometrical_model()

            #etad,I,beta = self.motion_equation_head_(qd,qdd)
            #list_beta[i,:] = beta

            MP = self.m_head*(np.dot(self.list_of_DH_matrices_head[:3,:3],self.centre_head[:3])+self.list_of_DH_matrices_head[:3,3])
            MP += self.m_neck * (np.dot(self.DH_matrix_neck_in_R0[:3, :3],
                                       self.centre_neck[:3]) + self.DH_matrix_neck_in_R0[:3, 3])
            g01 = self.DH_matrix_neck
            g10 = self.inverse_configuration(g01)
            self.Adj_gnh = self.adjoint_matrix_2(g10)
            quantity = np.dot(self.I_head,self.vel_head) #quantité de mouvement
            eta_i = np.dot(self.Adj_gnh, self.vel_head) + np.array([0, 0, 0, 0, self.qp_neck[0], self.qp_neck[1]])
            quantity+=np.dot(self.Adj_gnh.T,np.dot(self.I_neck,eta_i))
            eta_im1 = eta_i
            for k in range(self.nb):
                giip1 = self.list_of_DH_matrices_joint[:, k * 4:k * 4 + 4]
                gip1i = self.inverse_configuration(giip1)
                Ad_gip1i = self.adjoint_matrix_2(gip1i)  # for transfer the velocity

                i_g_G = self.inverse_configuration(self.list_of_DH_matrices_joint_in_R0[:, k * 4:k * 4 + 4])
                i_g_h = np.dot(i_g_G, self.list_of_DH_matrices_head)
                Ad_igh = self.adjoint_matrix_2(i_g_h)  # for transfer the force
                if k == 0:
                    eta_i = np.dot(Ad_gip1i, eta_im1) + np.array([0., 0., 0., 0., self.qp_joint[k], 0.])
                else:
                    eta_i = np.dot(Ad_gip1i, eta_im1) + np.array([0, 0, 0, 0, 0, self.qp_joint[k]])
                quantity += np.dot(Ad_igh.T, np.dot(self.I_seg, eta_i))

                eta_im1 = eta_i

                gi_roll = self.list_of_DH_matrices_roulie[:, k * 4:k * 4 + 4]
                groll_i = self.inverse_configuration(gi_roll)
                Ad_groll_i = self.adjoint_matrix_2(groll_i)

                ri_g_G = self.inverse_configuration(self.list_of_DH_matrices_roulie_in_R0[:, k * 4:k * 4 + 4])
                ri_g_h = np.dot(ri_g_G, self.list_of_DH_matrices_head)
                Ad_righ = self.adjoint_matrix_2(ri_g_h)  # for transfer the force

                eta_ir = np.dot(Ad_groll_i, eta_im1) + np.array([0., 0., 0., self.qp_roll[k], 0., 0.])
                quantity+=np.dot(Ad_righ.T,np.dot(self.I_roll,eta_ir))


                gs = self.list_of_DH_matrices_joint_in_R0[:,4*k:4*k+4]
                gr = self.list_of_DH_matrices_roulie_in_R0[:,4*k:4*k+4]
                if k != self.nb-1:
                    MP += self.m_seg * (np.dot(gs[:3, :3],self.centre_segment[:3]) + gs[:3, 3])
                else:
                    MP += self.m_last_seg * (np.dot(gs[:3, :3],self.centre_last_seg[:3]) + gs[:3, 3])
                MP +=  self.m_roll * (np.dot(gr[:3, :3],self.centre_last_seg[:3]) + gr[:3, 3])
            MP = MP/self.M
            Px2[i] = MP[0] * 1000
            Py2[i] = MP[1] * 1000
            Pz2[i] = MP[2] * 1000
            quantity_mvt2[i,:] = quantity
        Px = Px-Px[0]*np.ones((Px.shape[0]))
        Py = Py - Py[0] * np.ones((Py.shape[0]))
        Pz = Pz - Pz[0] * np.ones((Pz.shape[0]))
        Px2 = Px2 - Px2[0] * np.ones((Px.shape[0]))
        Py2 = Py2 - Py2[0] * np.ones((Py.shape[0]))
        Pz2 = Pz2 - Pz2[0] * np.ones((Pz.shape[0]))
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(t,Px)
        plt.plot(t, Px2)
        plt.xlabel('t/s')
        plt.ylabel('CM x/mm ')
        plt.grid()
        plt.subplot(3, 1, 2)
        plt.plot(t, Py)
        plt.plot(t, Py2)
        plt.xlabel('t/s')
        plt.ylabel('CM y/mm')
        plt.grid()
        plt.subplot(3, 1, 3)
        plt.plot(t, Pz)
        plt.plot(t, Pz2)
        plt.xlabel('t/s')
        plt.ylabel('CM z/mm')
        plt.grid()
        plt.figure()
        plt.subplot(6, 1, 1)
        plt.plot(t, X[0,:])
        plt.plot(t, X2[0, :])
        plt.xlabel('t/s')
        plt.ylabel('head x coordinate')
        plt.grid()
        plt.subplot(6, 1, 2)
        plt.plot(t, X[1,:])
        plt.plot(t, X2[1, :])
        plt.xlabel('t/s')
        plt.ylabel('head y coordinate')
        plt.grid()
        plt.subplot(6, 1, 3)
        plt.plot(t, X[2,:])
        plt.plot(t, X2[2, :])
        plt.xlabel('t/s')
        plt.ylabel('head z coordinate')
        plt.grid()
        plt.subplot(6, 1, 4)
        plt.plot(t, Euler_anglex)
        plt.plot(t, Euler_anglex2)
        plt.xlabel('t/s')
        plt.ylabel('euler angle x/deg')
        plt.grid()
        plt.subplot(6, 1, 5)
        plt.plot(t, Euler_angley)
        plt.plot(t, Euler_angley2)
        plt.xlabel('t/s')
        plt.ylabel('euler angle y/deg')
        plt.grid()
        plt.subplot(6, 1, 6)
        plt.plot(t, Euler_anglez)
        plt.plot(t, Euler_anglez2)
        plt.xlabel('t/s')
        plt.ylabel('euler angle z/deg')
        plt.grid()

        plt.figure()
        plt.subplot(6, 1, 1)
        plt.plot(t, quantity_mvt[:,0])
        plt.plot(t, quantity_mvt2[:, 0])
        plt.xlabel('t/s')
        plt.ylabel('réultant x')
        plt.grid()
        plt.subplot(6, 1, 2)
        plt.plot(t, quantity_mvt[:,1])
        plt.plot(t, quantity_mvt2[:, 1])
        plt.xlabel('t/s')
        plt.ylabel('réultant y')
        plt.grid()
        plt.subplot(6, 1, 3)
        plt.plot(t, quantity_mvt[:,2])
        plt.plot(t, quantity_mvt2[:, 2])
        plt.xlabel('t/s')
        plt.ylabel('réultant z')
        plt.grid()
        plt.subplot(6, 1, 4)
        plt.plot(t, quantity_mvt[:,3])
        plt.plot(t, quantity_mvt2[:, 3])
        plt.xlabel('t/s')
        plt.ylabel('moment cinétique x')
        plt.grid()
        plt.subplot(6, 1, 5)
        plt.plot(t, quantity_mvt[:,4])
        plt.plot(t, quantity_mvt2[:, 4])
        plt.xlabel('t/s')
        plt.ylabel('moment cinétique y')
        plt.grid()
        plt.subplot(6, 1, 6)
        plt.plot(t, quantity_mvt[:,5])
        plt.plot(t, quantity_mvt2[:, 5])
        plt.xlabel('t/s')
        plt.ylabel('moment cinétique z')
        plt.grid()
        '''
        plt.figure()
        plt.subplot(6, 1, 1)
        plt.plot(t, list_beta[:, 0])
        plt.xlabel('t/s')
        plt.ylabel('beta_t x')
        plt.grid()
        plt.subplot(6, 1, 2)
        plt.plot(t, list_beta[:, 1])
        plt.xlabel('t/s')
        plt.ylabel('beta_t y')
        plt.grid()
        plt.subplot(6, 1, 3)
        plt.plot(t, list_beta[:, 2])
        plt.xlabel('t/s')
        plt.ylabel('beta_t z')
        plt.grid()
        plt.subplot(6, 1, 4)
        plt.plot(t, list_beta[:, 3])
        plt.xlabel('t/s')
        plt.ylabel('beta_r x')
        plt.grid()
        plt.subplot(6, 1, 5)
        plt.plot(t, list_beta[:, 4])
        plt.xlabel('t/s')
        plt.ylabel('beta_r y')
        plt.grid()
        plt.subplot(6, 1, 6)
        plt.plot(t, list_beta[:, 5])
        plt.xlabel('t/s')
        plt.ylabel('beta_r z')
        plt.grid()
        '''
        plt.figure()
        plt.subplot(6, 1, 1)
        plt.plot(t, X[7,:])
        plt.plot(t, X2[7, :])
        plt.xlabel('t/s')
        plt.ylabel('V x')
        plt.grid()
        plt.subplot(6, 1, 2)
        plt.plot(t, X[8,:])
        plt.plot(t, X2[8, :])
        plt.xlabel('t/s')
        plt.ylabel('V y')
        plt.grid()
        plt.subplot(6, 1, 3)
        plt.plot(t, X[9,:])
        plt.plot(t, X2[9, :])
        plt.xlabel('t/s')
        plt.ylabel('V z')
        plt.grid()
        plt.subplot(6, 1, 4)
        plt.plot(t, X[10,:])
        plt.plot(t, X2[10, :])
        plt.xlabel('t/s')
        plt.ylabel('w x')
        plt.grid()
        plt.subplot(6, 1, 5)
        plt.plot(t, X[11,:])
        plt.plot(t, X2[11, :])
        plt.xlabel('t/s')
        plt.ylabel('w y')
        plt.grid()
        plt.subplot(6, 1, 6)
        plt.plot(t, X[12,:])
        plt.plot(t, X2[12, :])
        plt.xlabel('t/s')
        plt.ylabel('w z')
        plt.grid()
        plt.show()
#################################Direct recursive NE
    def direct_backward_NE(self):
        self.I_star = np.zeros((6,6,2*self.nb+2))
        self.beta_star = np.zeros((6,2*self.nb+2))
        self.H_star = []
        self.K_star = np.zeros((6,6,2*self.nb+1))
        self.alpha_star = np.zeros((6,2*self.nb+1))
        for i in reversed(range(self.nb)):
            ####### roll
            self.I_star[:,:,i*2+3] = self.I_roll
            self.beta_star[:,i*2+3] = self.beta[:,i*2+3]
            H_r = np.dot(self.I_roll, np.array([0,0,0,1,0,0]))
            H_r = np.dot(np.array([0,0,0,1,0,0]),H_r)
            self.H_star.insert(0, H_r)
            K_r = self.calcul_K(J = self.I_roll, aj=np.array([0,0,0,1,0,0]),H=H_r,double_Dof=False)
            self.K_star[:,:,i*2+2] = K_r
            self.alpha_star[:,i*2+2] = self.calcul_alpha(self.I_roll, K_r, np.array([0,0,0,1,0,0]), H_r, self.beta_star[:,i*2+3]
                                                           ,self.zeta[:,i*2+2],self.tau[i*2+3], False)

            ########## body segment
            if i == self.nb-1:
                self.I_star[:,:,i*2+2]= self.I_seg+np.dot(self.Adj_grolli[:,:,i].T,
                                                               np.dot(K_r,self.Adj_grolli[:,:,i]))
                self.beta_star[:,i*2+2] = self.beta[:,i*2+2]+np.dot(self.Adj_grolli[:,:,i].T,self.alpha_star[:,i*2+2])
            else:
                self.I_star[:, :, i * 2 + 2] = self.I_seg + np.dot(self.Adj_grolli[:, :, i].T,
                                                                        np.dot(K_r, self.Adj_grolli[:, :, i]))\
                                                    + np.dot(self.Adj_gip1i[:, :, i+1].T,np.dot(K_s, self.Adj_gip1i[:, :, i+1]))
                self.beta_star[:, i * 2 + 2] = self.beta[:, i * 2 + 2] + np.dot(self.Adj_grolli[:, :, i].T,
                                                                                      self.alpha_star[:, i * 2 + 2])+\
                                               np.dot(self.Adj_gip1i[:, :, i+1].T, self.alpha_star[:,i*2+3])
            if i == 0:
                aj = np.array([0, 0, 0, 0, 1, 0])
            else:
                aj = np.array([0, 0, 0, 0, 0, 1])
            H_s = np.dot(self.I_star[:, :, i * 2 + 2] , aj)
            H_s = np.dot(aj, H_s)
            self.H_star.insert(0, H_s)
            K_s = self.calcul_K(J=self.I_star[:, :, i * 2 + 2] , aj=aj, H=H_s, double_Dof=False)
            self.K_star[:, :, i * 2 + 1] = K_s
            self.alpha_star[:, i * 2 + 1] = self.calcul_alpha(self.I_star[:, :, i * 2 + 2], K_s, aj, H_s,
                                                                 self.beta_star[:, i * 2 + 2]
                                                                 , self.zeta[:, i * 2 + 1], self.tau[i * 2 + 2],
                                                                 False)
        self.I_star[:, :, 1] = self.I_neck + np.dot(self.Adj_gip1i[:, :, 0].T,
                                                           np.dot(K_s, self.Adj_gip1i[:, :, 0]))
        self.beta_star[:,1] = self.beta[:,1]+np.dot(self.Adj_gip1i[:,:,0].T,self.alpha_star[:,1])
        aj_T = np.array([[0,0,0,0,1,0],[0,0,0,0,0,1]]) ;aj = aj_T.T
        H_n = np.dot(self.I_star[:, :, 1], aj)
        H_n = np.dot(aj_T, H_n)
        self.H_star.insert(0, H_n)
        K_n = self.calcul_K(J=self.I_star[:, :, 1], aj=aj, H=H_n, double_Dof=True)
        self.K_star[:, :, 0] = K_n
        self.alpha_star[:, 0] = self.calcul_alpha(self.I_star[:, :, 1], K_n, aj, H_n,
                                                             self.beta_star[:, 1]
                                                             , self.zeta[:, 0], self.tau[:2],
                                                             True)
        self.I_star[:, :, 0] = self.I_head + np.dot(self.Adj_gnh[:, :].T,
                                                    np.dot(K_n, self.Adj_gnh[:, :]))
        self.beta_star[:, 0] = self.beta[:, 0] + np.dot(self.Adj_gnh[:, :].T, self.alpha_star[:, 0])

    def direct_forward_NE(self,qd,tau):
        self.qp_neck = qd[:2];
        for i in range(self.nb):
            self.qp_joint[i] = qd[2+2*i];
            self.qp_roll[i] = qd[3+2*i]
        #self.qp_joint = qd[2:2 + self.nb];
        #self.qp_roll = qd[2 + self.nb:]
        self.tau = tau
        self.get_velocity_gamma_beta(False)
        self.direct_backward_NE()
        self.acc_head = -np.dot(np.linalg.inv(self.I_star[:,:,0]),self.beta_star[:,0])

        def calcul_qdd(a_T,J,beta,zeta,tau,Vd,H,double_Dof):
            temp = np.dot(J,(Vd+zeta))
            temp = -np.dot(a_T,temp)+tau-np.dot(a_T,beta)
            if double_Dof:
                qdd = np.dot(np.linalg.inv(H),temp)
            else:
                qdd = temp/H
            return qdd
        aj_T = np.array([[0,0,0,0,1,0],[0,0,0,0,0,1]])
        n_Vd_h = np.dot(self.Adj_gnh,self.acc_head)
        self.qpp_neck = calcul_qdd(aj_T,self.I_star[:,:,1],self.beta_star[:,1],self.zeta[:,0],self.tau[:2],n_Vd_h,self.H_star[0],True)
        q = np.zeros((self.nb * 2 + 2))
        q[:2] = self.qpp_neck
        Vd_n = n_Vd_h + self.zeta[:,0] + np.dot(aj_T.T,self.qpp_neck)
        i_Vd_im1 = np.dot(self.Adj_gip1i[:,:,0],Vd_n)
        for i in range(self.nb):
            if i == 0:
                aj_T = np.array([0,0,0,0,1,0])
            else:
                aj_T = np.array([0,0,0,0,0,1])
            self.qpp_joint[i] = calcul_qdd(aj_T, self.I_star[:,:,i*2+2],self.beta_star[:,2*i+2],self.zeta[:,2*i+1],self.tau[2*i+2],i_Vd_im1,self.H_star[2*i+1],False)
            q[2*i+2] = self.qpp_joint[i]
            Vd_i = i_Vd_im1 + self.zeta[:, 2*i+1] + self.qpp_joint[i]*aj_T

            ri_Vd_i = np.dot(self.Adj_grolli[:,:,i],Vd_i)

            aj_T = np.array([0,0,0,1,0,0])
            self.qpp_roll[i] = calcul_qdd(aj_T, self.I_star[:,:,i * 2 + 3], self.beta_star[:,2 * i + 3], self.zeta[:,2 * i + 2],
                                           self.tau[2 * i + 3], ri_Vd_i, self.H_star[2 * i + 2], False)
            q[2*i+3] = self.qpp_roll[i]
            if i < self.nb-1:
                i_Vd_im1 = np.dot(self.Adj_gip1i[:,:,i+1],Vd_i)

        '''
        for i in range(6):
            q[2+i]=self.qpp_joint[i];
        for i in range(6):
            q[8+i]=self.qpp_roll[i]
        '''
        return self.acc_head,q

    def calcul_K(self, J, aj, H, double_Dof):
        '''
        :param J: J*i
        :param aj: rotation axis
        :param H: Hi
        :param double_Dof: bool, True: two Dof (neck) aj = np.array([[0,0,0,1,0,0],[0,0,0,0,0,1]).T
                                 False: one Dof. aj = np.array([0,0,0,1,0,0])
        :return: coefficient K
        '''
        if double_Dof:
            inv_H = np.linalg.inv(H)
            aj_T = aj.T
            K = np.dot(aj_T, J)  # aj_T*J
            K = np.dot(inv_H, K)  # inv_H*aj_T*J
            K = np.dot(J, np.dot(aj, K))
        else:
            K = np.dot(aj,J) # aj_T*J
            K = K/H # inv_H*aj_T*J
            aj_T = np.reshape(aj,(6,1))
            K = aj_T*K
            K = np.dot(J, K)
        K = J-K
        return K

    def calcul_alpha(self, J,K,aj,H,beta,zeta,tau,double_Dof):
        temp = np.dot(K, zeta) + beta
        if double_Dof:
            alpha = -np.dot(aj.T,beta)+tau
            alpha = np.dot(np.linalg.inv(H),alpha)
            alpha = np.dot(aj,alpha)
            alpha = np.dot(J,alpha)
            alpha += temp
        else:
            aj_T = np.reshape(aj,(6,1))
            alpha = (-np.dot(aj,beta)+tau)/H
            alpha = np.dot(J,aj_T*alpha)
            alpha = np.reshape(alpha,(6))
            alpha += temp
        return alpha
    def simu_dynamic_system(self,t,x,a=None,a_qd=None,a_qdd = None, stiff_body=True):
        '''

        :param x: (quaternion, head position, head velocity)
        :param t: time
        :param a: Dof x n+1  (n: number of degree of polynomial)
        :param a_qd: Dof x n
        :param a_qdd: Dof x n-1
        :return:
        '''
        #state: [quaternion,P0,eta0], derivate[go_d, eta0_d]\
        ####geometric model
        '''
        index = np.where(time<= t)
        print('here',t,index[0][-1])
        qn = qn[index[0][-1],:]
        qj = qj[index[0][-1], :]
        qr = qr[index[0][-1], :]
        qd = qd[index[0][-1], :]
        qdd = qdd[index[0][-1], :]
        '''
        print('tt',t)
        if stiff_body == True:
            qd = np.zeros((2 * self.nb + 2))
            qdd = np.zeros((2 * self.nb + 2))
            qn = self.neck_config
            qj = self.joint_config;
            qr = self.antirouli_config

        else:
            time = np.zeros((a.shape[2]))
            for i in range(a.shape[2]):
                time[i] = pow(t,i)
            q = np.dot(a[0,:,:],time)
            qd = np.dot(a[1,:,:],time)
            qdd =  np.dot(a[2,:,:],time)
            '''
            for i in range(2 * self.nb + 2):
                temp = 0.
                temp2 = 0.
                temp3 = 0.
                for j in range(a.shape[1]):
                    temp += a[i, j] * pow(t, j)
                    if j <= a.shape[1] - 2:
                        temp2 += a_qd[i, j] * pow(t, j)
                    if j <= a.shape[1] - 3:
                        temp3 += a_qdd[i, j] * pow(t, j)
                q[i] = temp;
                qd[i] = temp2;
                qdd[i] = temp3
            '''
            qn = q[:2]
            qj = np.zeros((self.nb));
            qr = np.zeros((self.nb))
            for i in range(self.nb):
                qj[i] = q[2 * i + 2]
                qr[i] = q[2 * i + 3]

        P0 = x[:3]
        #x[3:7] = x[3:7]/np.linalg.norm(x[3:7]);
        quat = x[3:7]
        quat = quat/np.linalg.norm(quat)
        Q1 = quat[0];Q2 = quat[1];Q3 = quat[2];Q4 = quat[3];
        eta0 = x[7:]
        #O1,O2,O3 = eta0[3],eta0[4],eta0[5]
        R = np.array([[2*(Q1**2+Q2**2)-1,2*(Q2*Q3-Q1*Q4),2*(Q2*Q4+Q1*Q3)],
                      [2*(Q2*Q3+Q1*Q4),2*(Q1**2+Q3**2)-1,2*(Q3*Q4-Q1*Q2)],
                      [2*(Q2*Q4-Q1*Q3),2*(Q3*Q4+Q1*Q2),2*(Q1**2+Q4**2)-1]])
        self.vel_head = eta0
        self.list_of_DH_matrices_head[:3,:3] = R; self.list_of_DH_matrices_head[:3,3] = P0

        self.neck_config = qn
        self.joint_config = qj
        self.antirouli_config = qr
        self.get_neck_frame();
        self.get_list_joint_frame();
        self.get_list_roulie_frame()
        self.get_list_frame_R02()

        xd = np.zeros((13))
        ##################################################
        '''
        Ad_g0 = self.adjoint_matrix_2(self.list_of_DH_matrices_head)
        G_eta_0 = np.dot(Ad_g0,eta0)
        O1, O2, O3 = G_eta_0[3], G_eta_0[4], G_eta_0[5]
        #################################################
        Omega_p = np.array([[0.,-O1,-O2,-O3],
                            [O1,0.,-O3,O2],
                            [O2,O3,0.,-O1],
                            [O3,-O2,O1,0]])
        xd[:3] = G_eta_0[:3]+np.cross(G_eta_0[3:],P0)
        xd[3:7] = np.dot(Omega_p,quat)/(2*np.linalg.norm(quat))
        '''
        w = np.zeros((4))
        w[1:] = eta0[3:]
        G = np.array([[-Q2,Q1,Q4,-Q3],[-Q3,-Q4,Q1,Q2],[-Q4,Q3,-Q2,Q1]])
        xd[3:7] =0.5*np.dot(G.T, eta0[3:]) #0.5 * self.quaterion_product(quat,w)
        xd[:3] = np.dot(R,eta0[:3])

        #eta0_d,_,_,_ = self.inverse_forward_NE(qd=qd,qdd=qdd)
        eta0_d,M = self.motion_equation_head(qd,qdd)
        #eta0_d2,M2 = self.IDM_one_module(qd,qdd)

        #eta0_d = self.orin_algo_inverse_dyn(qd=qd,qdd=qdd)
        xd[7:] = eta0_d
        '''
        print('pos',self.list_of_DH_matrices_head[2,3])
        print('vel',self.vel_head)
        print('acc',eta0_d[2])
        
        print('x',x[:7],x[7:])
        print('xd',xd[:7],xd[7:])
        '''
        return xd

    def simu_dynamic_system_test_centrifuge(self, t, x, a,T,neg):
        '''

        :param x: (quaternion, head position, head velocity)
        :param t: time
        :param a: Dof x n+1  (n: number of degree of polynomial)
        :param a_qd: Dof x n
        :param a_qdd: Dof x n-1
        :return:
        '''
        # state: [quaternion,P0,eta0], derivate[go_d, eta0_d]\
        ####geometric model
        '''
        index = np.where(time<= t)
        print('here',t,index[0][-1])
        qn = qn[index[0][-1],:]
        qj = qj[index[0][-1], :]
        qr = qr[index[0][-1], :]
        qd = qd[index[0][-1], :]
        qdd = qdd[index[0][-1], :]
        '''
        print('tt', t)
        if neg :
            fac = -1
        else:
            fac = 1
        # a = res.x
        def f(t, a, T):
            t = t % T
            return (a[0] * t ** 7 + a[1] * t ** 6 + a[2] * t ** 5 + a[3] * t ** 4 + a[4] * t ** 3 + a[5] * t ** 2 + a[
                6] * t + a[
                        7])  # *st

        def df(t, a, T):
            t = t % T
            return 7 * a[0] * t ** 6 + 6 * a[1] * t ** 5 + 5 * a[2] * t ** 4 + 4 * a[3] * t ** 3 + 3 * a[
                4] * t ** 2 + 2 * a[5] * t + a[
                       6]  # (5*a[0]*t**4+4*a[1]*t**3+3*a[2]*t**2+a[3])*st+2*pi/1*(a[0]*t**5+a[1]*t**4+a[2]*t**3+a[3]*t)*ct

        def ddf(t, a, T):
            t = t % T
            return 42 * a[0] * t ** 5 + 30 * a[1] * t ** 4 + 20 * a[2] * t ** 3 + 12 * a[3] * t ** 2 + 6 * a[
                4] * t + 2 * a[5]
        qn = np.ones((2))*f(t,a, T)*fac/2
        #qn[0] =0.
        qj = np.ones((self.nb))*f(t,a, T)*fac
        qj[0] = -f(t,a, T)*fac/2
        qr = np.ones((self.nb))*f(t,a, T)*fac

        qnd = np.ones((2)) *df(t,a, T)*fac/2
        #qnd[0] =0.
        qjd = np.ones((self.nb))*df(t,a, T)*fac
        qjd[0] =-df(t,a, T)*fac/2
        qrd = np.ones((self.nb))*df(t,a, T)*fac

        qndd = np.ones((2)) *ddf(t,a, T)*fac/2
        #qndd[0] = 0.
        qjdd = np.ones((self.nb)) *ddf(t,a, T)*fac
        qjdd[0] = -ddf(t,a, T)*fac/2
        qrdd = np.ones((self.nb))*ddf(t,a, T)*fac
        qd = np.zeros((2*self.nb+2))
        qdd = np.zeros((2*self.nb+2))
        qd[:2] = qnd
        qdd[:2] = qndd
        for i in range(self.nb):
            qd[2*i+2] = qjd[i]
            qd[2*i+3] = qrd[i]
            qdd[2 * i + 2] = qjdd[i]
            qdd[2 * i + 3] = qrdd[i]


        P0 = x[:3]
        # x[3:7] = x[3:7]/np.linalg.norm(x[3:7]);
        quat = x[3:7]
        quat = quat / np.linalg.norm(quat)
        Q1 = quat[0];
        Q2 = quat[1];
        Q3 = quat[2];
        Q4 = quat[3];
        eta0 = x[7:]
        # O1,O2,O3 = eta0[3],eta0[4],eta0[5]

        R = np.array([[2 * (Q1 ** 2 + Q2 ** 2) - 1, 2 * (Q2 * Q3 - Q1 * Q4), 2 * (Q2 * Q4 + Q1 * Q3)],
                      [2 * (Q2 * Q3 + Q1 * Q4), 2 * (Q1 ** 2 + Q3 ** 2) - 1, 2 * (Q3 * Q4 - Q1 * Q2)],
                      [2 * (Q2 * Q4 - Q1 * Q3), 2 * (Q3 * Q4 + Q1 * Q2), 2 * (Q1 ** 2 + Q4 ** 2) - 1]])
        self.vel_head = eta0
        self.list_of_DH_matrices_head[:3, :3] = R;
        self.list_of_DH_matrices_head[:3, 3] = P0

        self.neck_config = qn
        self.joint_config = qj
        self.antirouli_config = qr
        self.get_neck_frame();
        self.get_list_joint_frame();
        self.get_list_roulie_frame()
        self.get_list_frame_R02()

        xd = np.zeros((13))
        ##################################################
        '''
        Ad_g0 = self.adjoint_matrix_2(self.list_of_DH_matrices_head)
        G_eta_0 = np.dot(Ad_g0, eta0)
        O1, O2, O3 = G_eta_0[3], G_eta_0[4], G_eta_0[5]
        #################################################
        Omega_p = np.array([[0., -O1, -O2, -O3],
                            [O1, 0., -O3, O2],
                            [O2, O3, 0., -O1],
                            [O3, -O2, O1, 0]])
        xd[:3] = G_eta_0[:3] + np.cross(G_eta_0[3:], P0)
        xd[3:7] = np.dot(Omega_p, quat) / (2 * np.linalg.norm(quat))
        
        '''
        w = np.zeros((4))
        w[1:] = eta0[3:]
        G = np.array([[-Q2, Q1, Q4, -Q3], [-Q3, -Q4, Q1, Q2], [-Q4, Q3, -Q2, Q1]])
        xd[3:7] = 0.5 * np.dot(G.T, eta0[3:])  # 0.5 * self.quaterion_product(quat,w)
        xd[:3] = np.dot(R, eta0[:3])
        eta0_d, _ = self.motion_equation_head(qd, qdd)
        # eta0_d = self.orin_algo_inverse_dyn(qd=qd,qdd=qdd)
        xd[7:] = eta0_d
        return xd
    def simu_dynamic_system_temp(self, t, x, alpha, T, qcos,qsin):
        '''

        :param x: (quaternion, head position, head velocity)
        :param t: time
        :param a: Dof x n+1  (n: number of degree of polynomial)
        :param a_qd: Dof x n
        :param a_qdd: Dof x n-1
        :return:
        '''
        # state: [quaternion,P0,eta0], derivate[go_d, eta0_d]\
        ####geometric model
        '''
        index = np.where(time<= t)
        print('here',t,index[0][-1])
        qn = qn[index[0][-1],:]
        qj = qj[index[0][-1], :]
        qr = qr[index[0][-1], :]
        qd = qd[index[0][-1], :]
        qdd = qdd[index[0][-1], :]
        '''
        print('tt', t)
        w=2 * pi / T
        qd = np.zeros((2 * self.nb + 2))
        qdd = np.zeros((2 * self.nb + 2))
        '''
        tf = 10
        time_matrix = np.array([[tf ** 5, tf ** 4, tf ** 3], [5 * tf ** 4, 4 * tf ** 3, 3 * tf ** 2],
                                [20 * tf ** 3, 12 * tf ** 2, 6 * tf]])
        x_0 = np.array([1, 0, 0])
        a_i = np.dot(np.linalg.inv(time_matrix), x_0)
        if t<tf:
            self.joint_config[1:] = (a_i[0]*t**5 + a_i[1]*t**4 + a_i[2]*t**3)*alpha*sin(w*t)*qcos
            qd_f = (5*a_i[0]*t**4 + 4*a_i[1]*t**3 + 3*a_i[2]*t**2)*alpha*sin(w*t)*qcos+(a_i[0]*t**5 + a_i[1]*t**4 + a_i[2]*t**3)*alpha*w*cos(w*t)*qcos
            qdd_f = (20*a_i[0]*t**3 + 12*a_i[1]*t**2 + 6*a_i[2]*t)*alpha*sin(w*t)*qcos+(a_i[0]*t**5 + a_i[1]*t**4 + a_i[2]*t**3)*alpha*(-w**2 * sin(w * t) * qcos)+2*(5*a_i[0]*t**4 + 4*a_i[1]*t**3 + 3*a_i[2]*t**2)*alpha*w*cos(w*t)*qcos
        else:
            self.joint_config[1:] = alpha * (sin(w * t) * qcos + cos(w * t) * qsin * 0)
            qd_f = alpha * (w * cos(w * t) * qcos - w * sin(w * t) * qsin * 0)
            qdd_f = alpha * (-w ** 2 * sin(w * t) * qcos - w ** 2 * cos(w * t) * qsin * 0)
        '''
        self.joint_config[1:] = alpha * (sin(w * t) * qcos + cos(w * t) * qsin)
        qd_f = alpha * (w * cos(w * t) * qcos - w * sin(w * t) * qsin)
        qdd_f = alpha * (-w ** 2 * sin(w * t) * qcos - w ** 2 * cos(w * t) * qsin )
        for i in range(5):
            qd[4+2*i]=qd_f[i]
            qdd[4+2*i]=qdd_f[i]
        P0 = x[:3]
        # x[3:7] = x[3:7]/np.linalg.norm(x[3:7]);
        quat = x[3:7]
        quat = quat / np.linalg.norm(quat)
        Q1 = quat[0];
        Q2 = quat[1];
        Q3 = quat[2];
        Q4 = quat[3];
        eta0 = x[7:]
        # O1,O2,O3 = eta0[3],eta0[4],eta0[5]

        R = np.array([[2 * (Q1 ** 2 + Q2 ** 2) - 1, 2 * (Q2 * Q3 - Q1 * Q4), 2 * (Q2 * Q4 + Q1 * Q3)],
                      [2 * (Q2 * Q3 + Q1 * Q4), 2 * (Q1 ** 2 + Q3 ** 2) - 1, 2 * (Q3 * Q4 - Q1 * Q2)],
                      [2 * (Q2 * Q4 - Q1 * Q3), 2 * (Q3 * Q4 + Q1 * Q2), 2 * (Q1 ** 2 + Q4 ** 2) - 1]])
        self.vel_head = eta0
        self.list_of_DH_matrices_head[:3, :3] = R;
        self.list_of_DH_matrices_head[:3, 3] = P0

        self.get_neck_frame();
        self.get_list_joint_frame();
        self.get_list_roulie_frame()
        self.get_list_frame_R02()

        xd = np.zeros((13))
        ##################################################
        '''
        Ad_g0 = self.adjoint_matrix_2(self.list_of_DH_matrices_head)
        G_eta_0 = np.dot(Ad_g0, eta0)
        O1, O2, O3 = G_eta_0[3], G_eta_0[4], G_eta_0[5]
        #################################################
        Omega_p = np.array([[0., -O1, -O2, -O3],
                            [O1, 0., -O3, O2],
                            [O2, O3, 0., -O1],
                            [O3, -O2, O1, 0]])
        xd[:3] = G_eta_0[:3] + np.cross(G_eta_0[3:], P0)
        xd[3:7] = np.dot(Omega_p, quat) / (2 * np.linalg.norm(quat))

        '''
        w = np.zeros((4))
        w[1:] = eta0[3:]
        G = np.array([[-Q2, Q1, Q4, -Q3], [-Q3, -Q4, Q1, Q2], [-Q4, Q3, -Q2, Q1]])
        xd[3:7] = 0.5 * np.dot(G.T, eta0[3:])  # 0.5 * self.quaterion_product(quat,w)
        xd[:3] = np.dot(R, eta0[:3])
        eta0_d, _ = self.motion_equation_head(qd, qdd)
        # eta0_d = self.orin_algo_inverse_dyn(qd=qd,qdd=qdd)
        xd[7:] = eta0_d
        return xd
    def simu_dynamic_system_non_propag_onde(self, t, x, alpha, T, a_i, tf, qsin, qcos):
        w = 2*pi/T
        def f(t):
            if t <= tf:
                Vt = np.array([t ** 5, t ** 4, t ** 3, t ** 2, t])
                f = np.dot(a_i, Vt) + np.ones((self.nb - 1))
                # f = a_i[0] * t ** 5 + a_i[1] * t ** 4 + a_i[2] * t ** 4 + a_i[3] * t ** 3 + 1
                f = f * (qcos * sin(w * t) + qsin * cos(w * t))
            else:
                f = qcos * sin(w * t) + qsin * cos(w * t)
            return f * alpha

        def df(t):
            if t <= tf:
                Vt = np.array([t ** 5, t ** 4, t ** 3, t ** 2, t])
                f = np.dot(a_i, Vt) + np.ones((self.nb - 1))
                # f = a_i[0] * t ** 6 + a_i[1] * t ** 5 + a_i[2] * t ** 4 + a_i[3] * t ** 3 + 1
                Vdt = np.array([5 * t ** 4, 4 * t ** 3, 3 * t ** 2, 2 * t, 1])
                df = np.dot(a_i, Vdt)

                # df = 6 * a_i[0] * t ** 5 + 5 * a_i[1] * t ** 4 + 4 * a_i[2] * t ** 3 + 3 * a_i[3] * t ** 2
                df = df * (qcos * sin(w * t) + qsin * cos(w * t)) + f * (qcos * cos(w * t) - qsin * sin(w * t)) * w

            else:
                df = (qcos * cos(w * t) - qsin * sin(w * t)) * w

            return df * alpha

        def ddf(t):
            if t <= tf:
                Vt = np.array([t ** 5, t ** 4, t ** 3, t ** 2, t])
                f = np.dot(a_i, Vt) + np.ones((self.nb - 1))
                # f = a_i[0] * t ** 6 + a_i[1] * t ** 5 + a_i[2] * t ** 4 + a_i[3] * t ** 3 + 1
                Vdt = np.array([5 * t ** 4, 4 * t ** 3, 3 * t ** 2, 2 * t, 1])
                df = np.dot(a_i, Vdt)
                Vddt = np.array([20 * t ** 3, 12 * t ** 2, 6 * t, 2, 0])
                ddf = np.dot(a_i, Vddt)
                # ddf = 30 * a_i[0] * t ** 4 + 20 * a_i[1] * t ** 3 + 12 * a_i[2] * t ** 2 + 6 * a_i[3] * t
                ddf = 2 * df * (qcos * cos(w * t) - qsin * sin(w * t)) * w + f * (
                            -qcos * sin(w * t) - qsin * cos(w * t)) * w ** 2 + ddf * (
                                  qcos * sin(w * t) + qsin * cos(w * t))
            else:
                ddf = (-qcos * sin(w * t) - qsin * cos(w * t)) * w ** 2
            return ddf * alpha

        qd = np.zeros((self.nb * 2 + 2))
        qdd = np.zeros((self.nb * 2 + 2))

        self.joint_config[1:] = f(t)
        qd_j = df(t)
        qdd_j = ddf(t)
        for i in range(self.nb-1):
            qd[4+2*i] = qd_j[i]
            qdd[4+2*i] = qdd_j[i]
        P0 = x[:3]
        # x[3:7] = x[3:7]/np.linalg.norm(x[3:7]);
        quat = x[3:7]
        quat = quat / np.linalg.norm(quat)
        Q1 = quat[0];
        Q2 = quat[1];
        Q3 = quat[2];
        Q4 = quat[3];
        eta0 = x[7:]
        # O1,O2,O3 = eta0[3],eta0[4],eta0[5]

        R = np.array([[2 * (Q1 ** 2 + Q2 ** 2) - 1, 2 * (Q2 * Q3 - Q1 * Q4), 2 * (Q2 * Q4 + Q1 * Q3)],
                      [2 * (Q2 * Q3 + Q1 * Q4), 2 * (Q1 ** 2 + Q3 ** 2) - 1, 2 * (Q3 * Q4 - Q1 * Q2)],
                      [2 * (Q2 * Q4 - Q1 * Q3), 2 * (Q3 * Q4 + Q1 * Q2), 2 * (Q1 ** 2 + Q4 ** 2) - 1]])
        self.vel_head = eta0
        self.list_of_DH_matrices_head[:3, :3] = R;
        self.list_of_DH_matrices_head[:3, 3] = P0

        self.get_neck_frame();
        self.get_list_joint_frame();
        self.get_list_roulie_frame()
        self.get_list_frame_R02()

        xd = np.zeros((13))
        ##################################################
        w = np.zeros((4))
        w[1:] = eta0[3:]
        G = np.array([[-Q2, Q1, Q4, -Q3], [-Q3, -Q4, Q1, Q2], [-Q4, Q3, -Q2, Q1]])
        xd[3:7] = 0.5 * np.dot(G.T, eta0[3:])  # 0.5 * self.quaterion_product(quat,w)
        xd[:3] = np.dot(R, eta0[:3])
        eta0_d, _ = self.motion_equation_head(qd, qdd)
        # eta0_d = self.orin_algo_inverse_dyn(qd=qd,qdd=qdd)
        xd[7:] = eta0_d
        return xd
    def simu_dynamic_system_onde(self, t, x, alpha, T, a_i,L,tf):
        '''
        :param x: (head position, quaternion, head velocity)
        :param t: time
        :param alpha:  propulsion amplitute
        :param T: periode
        :param a_i: polynome a5*t^5+a4*t^4+a3*t^3
        :param L: length of robot
        :param tf: periode of acceleration
        :return:
        '''
        ####geometric model
        print('tt',t)
        def f(t,l):
            if t<=tf:
                #f = a_i[0]*t**5 + a_i[1]*t**4+a_i[2]*t**3
                #f = f*exp(l/(L))*np.sin(2*pi*(l/L+t/T))
                f = exp(l / (L)) * np.sin(2 * pi * (l / L + t / T))
            else:
                f = exp(l / (L)) * np.sin(2 * pi * (l / L + t / T))
            return f*alpha
        def df(t,l):
            if t<=tf:
                #f = a_i[0]*t**5 + a_i[1]*t**4+a_i[2]*t**3
                #df = 5*a_i[0]*t**4 + 4*a_i[1]*t**3+3*a_i[2]*t**2
                #df = df*exp(l/(L))*np.sin(2*pi*(l/L+t/T))+f*exp(l/(L))*np.cos(2*pi*(l/L+t/T))*2*pi/T
                df = exp(l / (L)) * np.cos(2 * pi * (l / L + t / T)) * 2 * pi / T
            else:
                df = exp(l/(L))*np.cos(2*pi*(l/L+t/T))*2*pi/T
            return df*alpha
        def ddf(t,l):
            if t<=tf:
                '''
                f = a_i[0]*t**5 + a_i[1]*t**4+a_i[2]*t**3
                df = 5*a_i[0]*t**4 + 4*a_i[1]*t**3+3*a_i[2]*t**2
                ddf = 20 * a_i[0] * t ** 3 + 12 * a_i[1] * t ** 2 + 6 * a_i[2] * t
                ddf = 2*df*exp(l/(L))*np.cos(2*pi*(l/L+t/T))*2*pi/T-f*exp(l/(L))*np.sin(2*pi*(l/L+t/T))*(2*pi/T)**2+ddf*exp(l / (L)) * np.sin(2 * pi * (l / L + t / T))
                '''
                ddf = -exp(l / (L)) * np.sin(2 * pi * (l / L + t / T)) * (2 * pi / T) ** 2
            else:
                ddf = -exp(l/(L))*np.sin(2*pi*(l/L+t/T))*(2*pi/T)**2
            return ddf*alpha
        qd = np.zeros((self.nb*2+2))
        qdd = np.zeros((self.nb * 2 + 2))
        '''
        self.neck_config[0] = f(t,self.P_head)
        qd[0] = df(t,self.P_head)
        qdd[0] = ddf(t, self.P_head)
        '''
        for i in range(1,self.nb):
            self.joint_config[i] = f(t,self.P_head+self.P_neck+self.P*i)
            qd[2+2*i] = df(t, self.P_head+self.P_neck+self.P*i)
            qdd[2+2*i] = ddf(t, self.P_head+self.P_neck+self.P*i)
        P0 = x[:3]
        # x[3:7] = x[3:7]/np.linalg.norm(x[3:7]);
        quat = x[3:7]
        quat = quat / np.linalg.norm(quat)
        Q1 = quat[0];
        Q2 = quat[1];
        Q3 = quat[2];
        Q4 = quat[3];
        eta0 = x[7:]
        # O1,O2,O3 = eta0[3],eta0[4],eta0[5]

        R = np.array([[2 * (Q1 ** 2 + Q2 ** 2) - 1, 2 * (Q2 * Q3 - Q1 * Q4), 2 * (Q2 * Q4 + Q1 * Q3)],
                      [2 * (Q2 * Q3 + Q1 * Q4), 2 * (Q1 ** 2 + Q3 ** 2) - 1, 2 * (Q3 * Q4 - Q1 * Q2)],
                      [2 * (Q2 * Q4 - Q1 * Q3), 2 * (Q3 * Q4 + Q1 * Q2), 2 * (Q1 ** 2 + Q4 ** 2) - 1]])
        self.vel_head = eta0
        self.list_of_DH_matrices_head[:3, :3] = R;
        self.list_of_DH_matrices_head[:3, 3] = P0

        self.get_neck_frame();
        self.get_list_joint_frame();
        self.get_list_roulie_frame()
        self.get_list_frame_R02()

        xd = np.zeros((13))
        ##################################################
        '''
        Ad_g0 = self.adjoint_matrix_2(self.list_of_DH_matrices_head)
        G_eta_0 = np.dot(Ad_g0, eta0)
        O1, O2, O3 = G_eta_0[3], G_eta_0[4], G_eta_0[5]
        #################################################
        Omega_p = np.array([[0., -O1, -O2, -O3],
                            [O1, 0., -O3, O2],
                            [O2, O3, 0., -O1],
                            [O3, -O2, O1, 0]])
        xd[:3] = G_eta_0[:3] + np.cross(G_eta_0[3:], P0)
        xd[3:7] = np.dot(Omega_p, quat) / (2 * np.linalg.norm(quat))

        '''
        w = np.zeros((4))
        w[1:] = eta0[3:]
        G = np.array([[-Q2, Q1, Q4, -Q3], [-Q3, -Q4, Q1, Q2], [-Q4, Q3, -Q2, Q1]])
        xd[3:7] = 0.5 * np.dot(G.T, eta0[3:])  # 0.5 * self.quaterion_product(quat,w)
        xd[:3] = np.dot(R, eta0[:3])
        eta0_d, _ = self.motion_equation_head(qd, qdd)
        # eta0_d = self.orin_algo_inverse_dyn(qd=qd,qdd=qdd)
        xd[7:] = eta0_d
        return xd
    def qd_qdd_generator(self,pos=None,t=None,n=None):
        '''

        :param pos: number of step x number of Dof
        :param t: number of step
        :param n: degree of polynomial
        :return: list_a: Dof x n+1
                    list_a_qd: Dof x n
                    list_a_qdd: Dof x n-1
        '''
        m = pos.shape[0] # m: number of step
        if n == None:
            n = m # n: number of degree of polynomial
        list_a = np.zeros((pos.shape[1],n+1))
        list_a_qd = np.zeros((pos.shape[1],n))
        list_a_qdd = np.zeros((pos.shape[1], n-1))
        for i in range(pos.shape[1]):
            q = pos[:,i]

            A = np.zeros((n+1,n+1))
            sum_t = np.zeros((2*n+1))
            sum_tq = np.zeros((n+1))
            sum_tq[0] = np.sum(q)
            q_multi_t = q
            for j in range(1,2*n+1):
                temp = np.power(t,j) #t0+t1+...+tm
                sum_t[j] = np.sum(temp)
                if j <= n:
                    q_multi_t = q_multi_t*t
                    sum_tq[j] = np.sum(q_multi_t)
            sum_t = sum_t[1:]
            for j in range(n+1):
                if j == 0:
                    A[0,0] = m; A[0,1:] = sum_t[:n]
                else:
                    A[j,:] = sum_t[j-1:j+n]
            a = np.dot(np.linalg.inv(A),sum_tq)
            list_a[i, :] = a
            a_qd = np.zeros((n))
            for j in range(1,n+1):
                a_qd[j-1] = j*a[j]
            a_qdd = np.zeros((n-1))
            for j in range(1,n):
                a_qdd[j-1] = j*a_qd[j]
            list_a_qd[i,:] = a_qd
            list_a_qdd[i, :] = a_qdd
            '''
            qd = np.zeros(q.shape);qdd = np.zeros(q.shape)
            for j in range(qd.shape[0]):
                temp = 0
                temp2 = 0
                for k in range(n):
                    temp += a_qd[k]*pow(t[j],k)
                    if k<=n-2:
                        temp2 += a_qdd[k]*pow(t[j],k)
                qd[j] = temp;qdd[j] = temp2
            vel[:,i] = qd;acc[:,i] = qdd
            '''
        return list_a,list_a_qd,list_a_qdd
    def simple_plot_test_centrifugue(self,h,t,a,T):
        """
        simple plot
        """

        # 生成画布
        fig = plt.figure(figsize=(8, 6), dpi=80)

        # 打开交互模式
        plt.ion()

        # 循环
        h = h.T

        def f(t, a,T):
            t = t % T
            return (a[0] * t ** 7 + a[1] * t ** 6 + a[2] * t ** 5 + a[3] * t ** 4 + a[4] * t ** 3 + a[5] * t ** 2 + a[
                6] * t + a[
                        7])  # *st

        def df(t, a,T):
            t = t % T
            return 7 * a[0] * t ** 6 + 6 * a[1] * t ** 5 + 5 * a[2] * t ** 4 + 4 * a[3] * t ** 3 + 3 * a[
                4] * t ** 2 + 2 * a[5] * t + a[
                       6]  # (5*a[0]*t**4+4*a[1]*t**3+3*a[2]*t**2+a[3])*st+2*pi/1*(a[0]*t**5+a[1]*t**4+a[2]*t**3+a[3]*t)*ct

        def ddf(t, a,T):
            t = t % T
            return 42 * a[0] * t ** 5 + 30 * a[1] * t ** 4 + 20 * a[2] * t ** 3 + 12 * a[3] * t ** 2 + 6 * a[
                4] * t + 2 * a[
                       5]  # temp1+temp2+temp3+temp4

        for index in range(int(h.shape[0])):
            # 清除原有图像
            quat = h[index, 3:7]
            Q1 = quat[0];
            Q2 = quat[1];
            Q3 = quat[2];
            Q4 = quat[3];

            R = np.array([[2 * (Q1 ** 2 + Q2 ** 2) - 1, 2 * (Q2 * Q3 - Q1 * Q4), 2 * (Q2 * Q4 + Q1 * Q3)],
                          [2 * (Q2 * Q3 + Q1 * Q4), 2 * (Q1 ** 2 + Q3 ** 2) - 1, 2 * (Q3 * Q4 - Q1 * Q2)],
                          [2 * (Q2 * Q4 - Q1 * Q3), 2 * (Q3 * Q4 + Q1 * Q2), 2 * (Q1 ** 2 + Q4 ** 2) - 1]])
            g = np.eye(4)
            g[:3, :3] = R
            P = h[index, :3]
            g[:3, 3] = P
            self.list_of_DH_matrices_head = g


            T = 10.

            self.neck_config = np.ones((2)) * f(t[index], a,T)/2
            #self.neck_config[0] = 0.
            self.joint_config = np.ones((self.nb)) * f(t[index], a,T)
            self.joint_config[0] = -f(t[index], a,T)/2
            self.antirouli_config = np.ones((self.nb)) * f(t[index], a,T)
            self.geometrical_model()

            plt.cla()

            # 生成测试数据
            self.draw_complet_model(fig)
            '''
            x,y = get_list_point(joint_shape[index,:],0)
            print(joint_shape[index,:])
            plt.plot(x, y)
            plt.scatter(x, y)
            plt.axis('equal')
            '''

            # 暂停
            plt.pause(0.01)

        # 关闭交互模式
        plt.ioff()

        # 图形显示
        plt.show()
        return
    def double_pendule(self,q,qd,qdd,eta0,t):
        def get_zeta_j_(qd_i, eta_i):
            Vi = eta_i[:3];
            Oi = eta_i[3:]
            adj_eta = np.zeros((6, 6))
            adj_eta[:3, :3] = self.skew_symetric_matrix(Oi)
            adj_eta[3:, 3:] = self.skew_symetric_matrix(Oi)
            adj_eta[:3, 3:] = self.skew_symetric_matrix(Vi)
            return np.dot(adj_eta,qd_i)

        def get_zeta_j(i_g_im1, qd_i, eta_im1):
            zeta_j = np.zeros((6))
            _0V1 = np.dot(i_g_im1[:3, :3], eta_im1[:3]);
            _0Omega1 = np.dot(i_g_im1[:3, :3], eta_im1[3:])
            temp = _0V1 + np.cross(i_g_im1[:3, 3], _0Omega1)
            zeta_j[:3] = np.cross(temp, qd_i)
            zeta_j[3:] = np.cross(_0Omega1, qd_i)
            '''
            im1_g_i = self.inverse_configuration(i_g_im1)
            temp = np.cross(eta_im1[3:],np.cross(eta_im1[3:],im1_g_i[:3, 3]))
            zeta_j[:3] = np.dot(i_g_im1[:3, :3],temp)
            zeta_j[3:] = np.cross(_0Omega1,qd_i)
            '''
            print(zeta_j)
            return zeta_j

        def get_beta_j(eta_i, Si, Mi, I):
            ###### Centrifuge/Corioli
            Vi = eta_i[:3];
            Oi = eta_i[3:]
            beta = np.zeros((6))
            S_hat = self.skew_symetric_matrix(Si)
            #beta[:3] = Mi*np.cross(Oi,np.cross(Oi,Si))
            #beta[3:] = np.cross(Oi,np.dot(I,Oi))
            '''
            temp = -np.cross(Si, Oi) + Vi
            beta[:3] = Mi * np.cross(Oi, temp)
            beta[3:] = np.cross(Oi, np.dot(I, Oi)) + Mi * np.cross(Si, np.cross(Oi, Vi))
            print('beta',beta)
            '''
            adj_eta = np.zeros((6,6))
            adj_eta[:3,:3] = self.skew_symetric_matrix(Oi)
            adj_eta[3:, 3:] = self.skew_symetric_matrix(Oi)
            adj_eta[:3, 3:] = self.skew_symetric_matrix(Vi)
            beta = -np.dot(adj_eta.T,np.dot(I,eta_i))
            print('beta', beta)
            #print('?',np.cross(Oi, Vi),Oi,Vi)
            '''
            temp = -np.cross(Si, Oi)
            beta[:3] = Mi * np.cross(Oi, temp)
            beta[3:] = np.cross(Oi, np.dot(I, Oi))
            '''
            return beta
        J = np.eye(3)
        m = 1.
        S = np.array([0.5,0.,0.])
        S_hat = self.skew_symetric_matrix(S)
        I = np.zeros((6, 6))
        I[:3, :3] = m * np.eye(3)
        I[:3, 3:] = -m * S_hat
        I[3:, :3] = m * S_hat
        I[3:, 3:] = J - m * np.dot(S_hat, S_hat)

        g = np.eye(4)
        g[:3,:3] = np.array([[cos(q),-sin(q),0],
                             [sin(q),cos(q),0],
                             [0.,0.,1.]])
        g[:3,3] = np.array([1.,0.,0.])
        Adg = self.adjoint_matrix_2(self.inverse_configuration(g))
        eta1 = np.dot(Adg,eta0)+np.array([0,0,0,0,0,qd])
        Ig = np.zeros((6,6))
        Ig[:3, :3] = m * np.eye(3)
        Ig[3:, 3:] = J
        eta1g = np.zeros((6))
        eta1g[:3] = eta1[:3] + np.cross(-S,eta1[3:])
        eta1g[3:] = eta1[3:]

        beta0 = get_beta_j(eta0,S,m,I)#[:3,:3])

        beta1 = get_beta_j(eta1,S,m,I)#[:3,:3])

        zeta1 = get_zeta_j_(np.array([0,0,0,0,0,qd]),eta1) + np.array([0,0,0,0,0,qdd])#
        zeta1_ = get_zeta_j(self.inverse_configuration(g),np.array([0,0,qd]),eta0) + np.array([0,0,0,0,0,qdd])
        print('?',zeta1-zeta1_)
        beta_c = beta0+np.dot(Adg.T,beta1)+np.dot(Adg.T,np.dot(I,zeta1))

        Ic = I+np.dot(Adg.T,np.dot(I,Adg))
        invg = self.inverse_configuration(g)
        etad = -np.dot(np.linalg.inv(Ic),beta_c)
        
        L0  = 1.
        L1 = 1.
        vx,vy,vz,wx,wy,wz = eta0[0],eta0[1],eta0[2],eta0[3],eta0[4],eta0[5]
        Ix,Iy,Iz = 1,1,1
        beta = np.zeros((6))
        Vxd, Vyd, Vzd, Wxd, Wyd, Wzd = 0,0,0,0,0,0
        Qd = qd
        Qdd = qdd
        beta[0]=      -m*(L0*wz*sin(q) + vx*cos(q) + vy*sin(q))*sin(q)*Qd + m*(L0*wz*cos(q)*Qd + L0*sin(q)*Wzd - vx*sin(q)*Qd + vy*cos(q)*Qd + sin(q)*Vyd + cos(q)*Vxd)*cos(q) + m*Vxd+ (-L1*m*(wz + Qd)/2 - m*(L0*wz*cos(q) - vx*sin(q) + vy*cos(q)))*cos(q)*Qd + (-L1*m*(Qdd + Wzd)/2 - m*(-L0*wz*sin(q)*Qd + L0*cos(q)*Wzd - vx*cos(q)*Qd - vy*sin(q)*Qd - sin(q)*Vxd+ cos(q)*Vyd))*sin(q)

        beta[1]=L1*m*Wzd/2 + m*(L0*wz*sin(q) + vx*cos(q) + vy*sin(q))*cos(q)*Qd + m*(L0*wz*cos(q)*Qd + L0*sin(q)*Wzd - vx*sin(q)*Qd + vy*cos(q)*Qd + sin(q)*Vyd + cos(q)*Vxd)*sin(q) + m*Vyd - (L1*m*(wz + Qd)/2 + m*(L0*wz*cos(q) - vx*sin(q) + vy*cos(q)))*sin(q)*Qd + (L1*m*(Qdd + Wzd)/2 + m*(-L0*wz*sin(q)*Qd + L0*cos(q)*Wzd - vx*cos(q)*Qd - vy*sin(q)*Qd - sin(q)*Vxd+ cos(q)*Vyd))*cos(q)
        beta[2]=-L1*m*(-wx*cos(q)*Qd - wy*sin(q)*Qd - sin(q)*Wxd + cos(q)*Wyd)/2 - L1*m*Wyd/2 + m*((-L0*sin(q)**2 - L0*cos(q)**2)*Wyd + Vzd) + m*Vzd
        beta[3]=-Ix*(wx*cos(q) + wy*sin(q))*sin(q)*Qd + Ix*(-wx*sin(q)*Qd + wy*cos(q)*Qd + sin(q)*Wyd + cos(q)*Wxd)*cos(q) + Ix*Wxd + (L1*m*(vz + wy*(-L0*sin(q)**2 - L0*cos(q)**2))/2 - (Iy + L1**2*m/4)*(-wx*sin(q) + wy*cos(q)))*cos(q)*Qd + (L1*m*((-L0*sin(q)**2 - L0*cos(q)**2)*Wyd + Vzd)/2 - (Iy + L1**2*m/4)*(-wx*cos(q)*Qd - wy*sin(q)*Qd - sin(q)*Wxd + cos(q)*Wyd))*sin(q)
        beta[4]=Ix*(wx*cos(q) + wy*sin(q))*cos(q)*Qd + Ix*(-wx*sin(q)*Qd + wy*cos(q)*Qd + sin(q)*Wyd + cos(q)*Wxd)*sin(q) - L1*m*Vzd/2 + (Iy + L1**2*m/4)*Wyd + (-L0*sin(q)**2 - L0*cos(q)**2)*(-L1*m*(-wx*cos(q)*Qd - wy*sin(q)*Qd - sin(q)*Wxd + cos(q)*Wyd)/2 + m*((-L0*sin(q)**2 - L0*cos(q)**2)*Wyd + Vzd)) - (-L1*m*(vz + wy*(-L0*sin(q)**2 - L0*cos(q)**2))/2 + (Iy + L1**2*m/4)*(-wx*sin(q) + wy*cos(q)))*sin(q)*Qd + (-L1*m*((-L0*sin(q)**2 - L0*cos(q)**2)*Wyd + Vzd)/2 + (Iy + L1**2*m/4)*(-wx*cos(q)*Qd - wy*sin(q)*Qd - sin(q)*Wxd + cos(q)*Wyd))*cos(q)
        beta[5]=L0*m*(L0*wz*sin(q) + vx*cos(q) + vy*sin(q))*cos(q)*Qd + L0*m*(L0*wz*cos(q)*Qd + L0*sin(q)*Wzd - vx*sin(q)*Qd + vy*cos(q)*Qd + sin(q)*Vyd + cos(q)*Vxd)*sin(q) - L0*(L1*m*(wz + Qd)/2 + m*(L0*wz*cos(q) - vx*sin(q) + vy*cos(q)))*sin(q)*Qd + L0*(L1*m*(Qdd + Wzd)/2 + m*(-L0*wz*sin(q)*Qd + L0*cos(q)*Wzd - vx*cos(q)*Qd - vy*sin(q)*Qd - sin(q)*Vxd+ cos(q)*Vyd))*cos(q) + L1*m*(-L0*wz*sin(q)*Qd + L0*cos(q)*Wzd - vx*cos(q)*Qd - vy*sin(q)*Qd - sin(q)*Vxd+ cos(q)*Vyd)/2 + L1*m*Vyd/2 + (Iz + L1**2*m/4)*(Qdd + Wzd) + (Iz + L1**2*m/4)*Wzd
        
        M = np.array([[m*sin(q)**2 + m*cos(q)**2 + m, 0, 0, 0, 0, L0*m*sin(q)*cos(q) + (-L0*m*cos(q) - L1*m/2)*sin(q)], [0, m*sin(q)**2 + m*cos(q)**2 + m, 0, 0, 0, L0*m*sin(q)**2 + L1*m/2 + (L0*m*cos(q) + L1*m/2)*cos(q)], [0, 0, 2*m, L1*m*sin(q)/2, -L1*m*cos(q)/2 - L1*m/2 + m*(-L0*sin(q)**2 - L0*cos(q)**2), 0], [0, 0, L1*m*sin(q)/2, Ix*cos(q)**2 + Ix - (-Iy - L1**2*m/4)*sin(q)**2, Ix*sin(q)*cos(q) + (L1*m*(-L0*sin(q)**2 - L0*cos(q)**2)/2 + (-Iy - L1**2*m/4)*cos(q))*sin(q), 0], [0, 0, -L1*m*cos(q)/2 - L1*m/2 + m*(-L0*sin(q)**2 - L0*cos(q)**2), Ix*sin(q)*cos(q) + L1*m*(-L0*sin(q)**2 - L0*cos(q)**2)*sin(q)/2 - (Iy + L1**2*m/4)*sin(q)*cos(q), Ix*sin(q)**2 + Iy + L1**2*m/4 + (-L0*sin(q)**2 - L0*cos(q)**2)*(-L1*m*cos(q)/2 + m*(-L0*sin(q)**2 - L0*cos(q)**2)) + (-L1*m*(-L0*sin(q)**2 - L0*cos(q)**2)/2 + (Iy + L1**2*m/4)*cos(q))*cos(q), 0], [-L1*m*sin(q)/2, L0*m*sin(q)**2 + L0*m*cos(q)**2 + L1*m*cos(q)/2 + L1*m/2, 0, 0, 0, 2*Iz + L0**2*m*sin(q)**2 + L0*L1*m*cos(q)/2 + L0*(L0*m*cos(q) + L1*m/2)*cos(q) + L1**2*m/2]])
        if t==80.95438291976366:
            print(q,qd,qdd)
        print(q, qd, qdd)
        print('diff', np.linalg.norm(beta-beta_c)/np.linalg.norm(beta)*100)
        print('diff2', beta - beta_c,beta,beta_c)
        print('diff3',M-Ic)

        etad_ = np.dot(np.linalg.inv(M),-beta)
        print('diff4', etad, etad_)
        return etad

    def simu_dynamic_system_duble_pendule(self, t, x, a, T):

        print('tt', t)

        def f(t, a,T):
            t = t % T
            return (a[0] * t ** 7 + a[1] * t ** 6 + a[2] * t ** 5 + a[3] * t ** 4 + a[4] * t ** 3 + a[5] * t ** 2 + a[
                6] * t + a[
                        7])  # *st

        def df(t, a, T):
            t = t % T
            return 7 * a[0] * t ** 6 + 6 * a[1] * t ** 5 + 5 * a[2] * t ** 4 + 4 * a[3] * t ** 3 + 3 * a[
                4] * t ** 2 + 2 * a[5] * t + a[6]  # (5*a[0]*t**4+4*a[1]*t**3+3*a[2]*t**2+a[3])*st+2*pi/1*(a[0]*t**5+a[1]*t**4+a[2]*t**3+a[3]*t)*ct

        def ddf(t, a, T):
            t = t % T
            return 42 * a[0] * t ** 5 + 30 * a[1] * t ** 4 + 20 * a[2] * t ** 3 + 12 * a[3] * t ** 2 + 6 * a[
                4] * t + 2 * a[
                       5]

        q = f(t,a,T)
        qd = df(t,a,T)
        qdd = ddf(t,a,T)

        P0 = x[:3]
        # x[3:7] = x[3:7]/np.linalg.norm(x[3:7]);
        quat = x[3:7]
        quat = quat / np.linalg.norm(quat)
        Q1 = quat[0];
        Q2 = quat[1];
        Q3 = quat[2];
        Q4 = quat[3];
        eta0 = x[7:]
        # O1,O2,O3 = eta0[3],eta0[4],eta0[5]
        R = np.array([[2 * (Q1 ** 2 + Q2 ** 2) - 1, 2 * (Q2 * Q3 - Q1 * Q4), 2 * (Q2 * Q4 + Q1 * Q3)],
                      [2 * (Q2 * Q3 + Q1 * Q4), 2 * (Q1 ** 2 + Q3 ** 2) - 1, 2 * (Q3 * Q4 - Q1 * Q2)],
                      [2 * (Q2 * Q4 - Q1 * Q3), 2 * (Q3 * Q4 + Q1 * Q2), 2 * (Q1 ** 2 + Q4 ** 2) - 1]])
        R = R/np.linalg.det(R)
        print('detR',np.linalg.det(R))
        ##################################################
        xd = np.zeros((13))
        w = np.zeros((4))
        w[1:] = eta0[3:]
        G = np.array([[-Q2, Q1, Q4, -Q3], [-Q3, -Q4, Q1, Q2], [-Q4, Q3, -Q2, Q1]])
        xd[3:7] = 0.5 * np.dot(G.T, eta0[3:])  # 0.5 * self.quaterion_product(quat,w)
        xd[:3] = np.dot(R, eta0[:3])

        eta0_d= self.double_pendule(q,qd, qdd,eta0,t)
        # eta0_d = self.orin_algo_inverse_dyn(qd=qd,qdd=qdd)
        print(eta0_d)
        xd[7:] = eta0_d
        return xd
    def center_of_masse_double_pendule(self,X,t,a,T):
        Px = np.zeros((t.shape[0]))
        Py = np.zeros((t.shape[0]))
        Pz = np.zeros((t.shape[0]))
        quantity_mvt = np.zeros((t.shape[0],6))
        Euler_anglex = np.zeros((t.shape[0]))
        Euler_angley = np.zeros((t.shape[0]))
        Euler_anglez = np.zeros((t.shape[0]))
        def f(t, a, T):
            t = t % T
            return (a[0] * t ** 7 + a[1] * t ** 6 + a[2] * t ** 5 + a[3] * t ** 4 + a[4] * t ** 3 + a[5] * t ** 2 + a[
                6] * t + a[
                        7])  # *st

        def df(t, a, T):
            t = t % T
            return 7 * a[0] * t ** 6 + 6 * a[1] * t ** 5 + 5 * a[2] * t ** 4 + 4 * a[3] * t ** 3 + 3 * a[
                4] * t ** 2 + 2 * a[5] * t + a[6]  # (5*a[0]*t**4+4*a[1]*t**3+3*a[2]*t**2+a[3])*st+2*pi/1*(a[0]*t**5+a[1]*t**4+a[2]*t**3+a[3]*t)*ct

        def ddf(t, a, T):
            t = t % T
            return 42 * a[0] * t ** 5 + 30 * a[1] * t ** 4 + 20 * a[2] * t ** 3 + 12 * a[3] * t ** 2 + 6 * a[
                4] * t + 2 * a[
                       5]

        def get_zeta_j(i_g_im1, qd_i, eta_im1):
            zeta_j = np.zeros((6))
            _0V1 = np.dot(i_g_im1[:3, :3], eta_im1[:3]);
            _0Omega1 = np.dot(i_g_im1[:3, :3], eta_im1[3:])
            temp = _0V1 + np.cross(i_g_im1[:3, 3], _0Omega1)
            zeta_j[:3] = np.cross(temp, qd_i)
            zeta_j[3:] = np.cross(_0Omega1, qd_i)
            '''
            im1_g_i = self.inverse_configuration(i_g_im1)
            temp = np.cross(eta_im1[3:],np.cross(eta_im1[3:],im1_g_i[:3, 3]))
            zeta_j[:3] = np.dot(i_g_im1[:3, :3],temp)
            zeta_j[3:] = np.cross(_0Omega1,qd_i)
            '''
            return zeta_j

        def get_beta_j(eta_i, Si, Mi, I):
            ###### Centrifuge/Corioli
            Vi = eta_i[:3];
            Oi = eta_i[3:]
            beta = np.zeros((6))
            S_hat = self.skew_symetric_matrix(Si)
            #beta[:3] = Mi*np.cross(Oi,np.cross(Oi,Si))
            #beta[3:] = np.cross(Oi,np.dot(I,Oi))
            temp = -np.cross(Si, Oi) + Vi
            beta[:3] = Mi * np.cross(Oi, temp)
            beta[3:] = np.cross(Oi, np.dot(I, Oi)) + Mi * np.cross(Si, np.cross(Oi, Vi))
            #print('?',np.cross(Oi, Vi),Oi,Vi)
            '''
            temp = -np.cross(Si, Oi)
            beta[:3] = Mi * np.cross(Oi, temp)
            beta[3:] = np.cross(Oi, np.dot(I, Oi))
            '''
            return beta
        list_beta = np.zeros((t.shape[0]))
        for i in range(t.shape[0]):
            print(i)
            x=X[:,i]
            P0 = x[:3]
            # x[3:7] = x[3:7]/np.linalg.norm(x[3:7]);
            quat = x[3:7]
            quat = quat / np.linalg.norm(quat)
            Q1 = quat[0];
            Q2 = quat[1];
            Q3 = quat[2];
            Q4 = quat[3];
            eta0 = x[7:]
            # O1,O2,O3 = eta0[3],eta0[4],eta0[5]
            R = np.array([[2 * (Q1 ** 2 + Q2 ** 2) - 1, 2 * (Q2 * Q3 - Q1 * Q4), 2 * (Q2 * Q4 + Q1 * Q3)],
                          [2 * (Q2 * Q3 + Q1 * Q4), 2 * (Q1 ** 2 + Q3 ** 2) - 1, 2 * (Q3 * Q4 - Q1 * Q2)],
                          [2 * (Q2 * Q4 - Q1 * Q3), 2 * (Q3 * Q4 + Q1 * Q2), 2 * (Q1 ** 2 + Q4 ** 2) - 1]])
            G = Rotation.from_matrix(R)
            angle_euler = G.as_euler('xyz', degrees=True)
            Euler_anglex[i] = angle_euler[0]
            Euler_angley[i] = angle_euler[1]
            Euler_anglez[i] = angle_euler[2]
            g0 = np.eye(4)
            g0[:3, :3] = R;
            g0[:3, 3] = P0
            q = f(t[i], a,T)
            qd = df(t[i], a,T)
            qdd = ddf(t[i], a, T)
            g1 = np.eye(4)
            g1[:3, :3] = np.array([[cos(q), -sin(q), 0],
                                  [sin(q), cos(q), 0],
                                  [0., 0., 1.]])

            g1[:3, 3] = np.array([1., 0., 0.])
            m=1.
            center = np.array([0.5,0.,0.])
            g_1 = np.dot(g0,g1)
            MP = m*(np.dot(g0[:3,:3],center)+g0[:3,3])+m*(np.dot(g_1[:3,:3],center)+g_1[:3,3])
            MP = MP/2
            Px[i] = MP[0]
            Py[i] = MP[1]
            Pz[i] = MP[2]

            Adg = self.adjoint_matrix_2(self.inverse_configuration(g1))
            eta1 = np.dot(Adg, eta0) + np.array([0, 0, 0, 0, 0, qd])
            J = np.eye(3)
            m = 1.
            S = np.array([0.5, 0., 0.])
            S_hat = self.skew_symetric_matrix(S)
            I = np.zeros((6, 6))
            I[:3, :3] = m * np.eye(3)
            I[:3, 3:] = -m * S_hat
            I[3:, :3] = m * S_hat
            I[3:, 3:] = J - m * np.dot(S_hat, S_hat)
            quantity = np.dot(I,eta0)+np.dot(Adg.T,np.dot(I,eta1))
            quantity_mvt[i,:] = quantity

            g = np.eye(4)
            g[:3, :3] = np.array([[cos(q), -sin(q), 0],
                                  [sin(q), cos(q), 0],
                                  [0., 0., 1.]])
            g[:3, 3] = np.array([1., 0., 0.])
            Adg = self.adjoint_matrix_2(self.inverse_configuration(g))
            eta1 = np.dot(Adg, eta0) + np.array([0, 0, 0, 0, 0, qd])
            Ig = np.zeros((6, 6))
            Ig[:3, :3] = m * np.eye(3)
            Ig[3:, 3:] = J
            eta1g = np.zeros((6))
            eta1g[:3] = eta1[:3] + np.cross(-S, eta1[3:])
            eta1g[3:] = eta1[3:]

            beta0 = get_beta_j(eta0, S, m, I[:3, :3])

            beta1 = get_beta_j(eta1, S, m, I[:3, :3])

            zeta1 = get_zeta_j(self.inverse_configuration(g), np.array([0, 0, qd]), eta0) + np.array(
                [0, 0, 0, 0, 0, qdd])
            beta_c = beta0 + np.dot(Adg.T, beta1) + np.dot(Adg.T, np.dot(I, zeta1))

            Ic = I + np.dot(Adg.T, np.dot(I, Adg))
            invg = self.inverse_configuration(g)
            etad = -np.dot(np.linalg.inv(Ic), beta_c)
            print('beta',qdd)
            L0 = 1.
            L1 = 1.
            vx, vy, vz, wx, wy, wz = eta0[0], eta0[1], eta0[2], eta0[3], eta0[4], eta0[5]
            Ix, Iy, Iz = 1, 1, 1
            beta = np.zeros((6))
            Vxd, Vyd, Vzd, Wxd, Wyd, Wzd = 0, 0, 0, 0, 0, 0
            Qdd = qdd ; Qd = qd
            beta[0] = -m * (L0 * wz * sin(q) + vx * cos(q) + vy * sin(q)) * sin(q) * Qd + m * (
                        L0 * wz * cos(q) * Qd + L0 * sin(q) * Wzd - vx * sin(q) * Qd + vy * cos(q) * Qd + sin(
                    q) * Vyd + cos(q) * Vxd) * cos(q) + m * Vxd + (
                                  -L1 * m * (wz + Qd) / 2 - m * (L0 * wz * cos(q) - vx * sin(q) + vy * cos(q))) * cos(
                q) * Qd + (-L1 * m * (Qdd + Wzd) / 2 - m * (
                        -L0 * wz * sin(q) * Qd + L0 * cos(q) * Wzd - vx * cos(q) * Qd - vy * sin(q) * Qd - sin(
                    q) * Vxd + cos(q) * Vyd)) * sin(q)

            beta[1] = L1 * m * Wzd / 2 + m * (L0 * wz * sin(q) + vx * cos(q) + vy * sin(q)) * cos(q) * Qd + m * (
                        L0 * wz * cos(q) * Qd + L0 * sin(q) * Wzd - vx * sin(q) * Qd + vy * cos(q) * Qd + sin(
                    q) * Vyd + cos(q) * Vxd) * sin(q) + m * Vyd - (
                                  L1 * m * (wz + Qd) / 2 + m * (L0 * wz * cos(q) - vx * sin(q) + vy * cos(q))) * sin(
                q) * Qd + (L1 * m * (Qdd + Wzd) / 2 + m * (
                        -L0 * wz * sin(q) * Qd + L0 * cos(q) * Wzd - vx * cos(q) * Qd - vy * sin(q) * Qd - sin(
                    q) * Vxd + cos(q) * Vyd)) * cos(q)
            beta[2] = -L1 * m * (-wx * cos(q) * Qd - wy * sin(q) * Qd - sin(q) * Wxd + cos(
                q) * Wyd) / 2 - L1 * m * Wyd / 2 + m * ((-L0 * sin(q) ** 2 - L0 * cos(q) ** 2) * Wyd + Vzd) + m * Vzd
            beta[3] = -Ix * (wx * cos(q) + wy * sin(q)) * sin(q) * Qd + Ix * (
                        -wx * sin(q) * Qd + wy * cos(q) * Qd + sin(q) * Wyd + cos(q) * Wxd) * cos(q) + Ix * Wxd + (
                                  L1 * m * (vz + wy * (-L0 * sin(q) ** 2 - L0 * cos(q) ** 2)) / 2 - (
                                      Iy + L1 ** 2 * m / 4) * (-wx * sin(q) + wy * cos(q))) * cos(q) * Qd + (
                                  L1 * m * ((-L0 * sin(q) ** 2 - L0 * cos(q) ** 2) * Wyd + Vzd) / 2 - (
                                      Iy + L1 ** 2 * m / 4) * (
                                              -wx * cos(q) * Qd - wy * sin(q) * Qd - sin(q) * Wxd + cos(
                                          q) * Wyd)) * sin(q)
            beta[4] = Ix * (wx * cos(q) + wy * sin(q)) * cos(q) * Qd + Ix * (
                        -wx * sin(q) * Qd + wy * cos(q) * Qd + sin(q) * Wyd + cos(q) * Wxd) * sin(
                q) - L1 * m * Vzd / 2 + (Iy + L1 ** 2 * m / 4) * Wyd + (-L0 * sin(q) ** 2 - L0 * cos(q) ** 2) * (
                                  -L1 * m * (
                                      -wx * cos(q) * Qd - wy * sin(q) * Qd - sin(q) * Wxd + cos(q) * Wyd) / 2 + m * (
                                              (-L0 * sin(q) ** 2 - L0 * cos(q) ** 2) * Wyd + Vzd)) - (
                                  -L1 * m * (vz + wy * (-L0 * sin(q) ** 2 - L0 * cos(q) ** 2)) / 2 + (
                                      Iy + L1 ** 2 * m / 4) * (-wx * sin(q) + wy * cos(q))) * sin(q) * Qd + (
                                  -L1 * m * ((-L0 * sin(q) ** 2 - L0 * cos(q) ** 2) * Wyd + Vzd) / 2 + (
                                      Iy + L1 ** 2 * m / 4) * (
                                              -wx * cos(q) * Qd - wy * sin(q) * Qd - sin(q) * Wxd + cos(
                                          q) * Wyd)) * cos(q)
            beta[5] = L0 * m * (L0 * wz * sin(q) + vx * cos(q) + vy * sin(q)) * cos(q) * Qd + L0 * m * (
                        L0 * wz * cos(q) * Qd + L0 * sin(q) * Wzd - vx * sin(q) * Qd + vy * cos(q) * Qd + sin(
                    q) * Vyd + cos(q) * Vxd) * sin(q) - L0 * (
                                  L1 * m * (wz + Qd) / 2 + m * (L0 * wz * cos(q) - vx * sin(q) + vy * cos(q))) * sin(
                q) * Qd + L0 * (L1 * m * (Qdd + Wzd) / 2 + m * (
                        -L0 * wz * sin(q) * Qd + L0 * cos(q) * Wzd - vx * cos(q) * Qd - vy * sin(q) * Qd - sin(
                    q) * Vxd + cos(q) * Vyd)) * cos(q) + L1 * m * (
                                  -L0 * wz * sin(q) * Qd + L0 * cos(q) * Wzd - vx * cos(q) * Qd - vy * sin(
                              q) * Qd - sin(q) * Vxd + cos(q) * Vyd) / 2 + L1 * m * Vyd / 2 + (Iz + L1 ** 2 * m / 4) * (
                                  Qdd + Wzd) + (Iz + L1 ** 2 * m / 4) * Wzd

            list_beta[i] = np.linalg.norm(beta-beta_c)/np.linalg.norm(beta)*100
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(t,Px)
        plt.xlabel('t/s')
        plt.ylabel('x coordinate')
        plt.grid()
        plt.subplot(3, 1, 2)
        plt.plot(t, Py)
        plt.xlabel('t/s')
        plt.ylabel('y coordinate')
        plt.grid()
        plt.subplot(3, 1, 3)
        plt.plot(t, Pz)
        plt.xlabel('t/s')
        plt.ylabel('z coordinate')
        plt.grid()

        plt.figure()
        plt.subplot(6, 1, 1)
        plt.plot(t, X[0,:])
        plt.xlabel('t/s')
        plt.ylabel('head x coordinate')
        plt.grid()
        plt.subplot(6, 1, 2)
        plt.plot(t, X[1,:])
        plt.xlabel('t/s')
        plt.ylabel('head y coordinate')
        plt.grid()
        plt.subplot(6, 1, 3)
        plt.plot(t, X[2,:])
        plt.xlabel('t/s')
        plt.ylabel('head z coordinate')
        plt.grid()
        plt.subplot(6, 1, 4)
        plt.plot(t, Euler_anglex)
        plt.xlabel('t/s')
        plt.ylabel('x angle')
        plt.grid()
        plt.subplot(6, 1, 5)
        plt.plot(t, Euler_angley)
        plt.xlabel('t/s')
        plt.ylabel('y angle')
        plt.grid()
        plt.subplot(6, 1, 6)
        plt.plot(t, Euler_anglez)
        plt.xlabel('t/s')
        plt.ylabel('z angle')
        plt.grid()
        plt.figure()
        plt.subplot(6, 1, 1)
        plt.plot(t, quantity_mvt[:, 0])
        plt.xlabel('t/s')
        plt.ylabel('réultant x')
        plt.grid()
        plt.subplot(6, 1, 2)
        plt.plot(t, quantity_mvt[:, 1])
        plt.xlabel('t/s')
        plt.ylabel('réultant y')
        plt.grid()
        plt.subplot(6, 1, 3)
        plt.plot(t, quantity_mvt[:, 2])
        plt.xlabel('t/s')
        plt.ylabel('réultant z')
        plt.grid()
        plt.subplot(6, 1, 4)
        plt.plot(t, quantity_mvt[:, 3])
        plt.xlabel('t/s')
        plt.ylabel('moment cinétique x')
        plt.grid()
        plt.subplot(6, 1, 5)
        plt.plot(t, quantity_mvt[:, 4])
        plt.xlabel('t/s')
        plt.ylabel('moment cinétique y')
        plt.grid()
        plt.subplot(6, 1, 6)
        plt.plot(t, quantity_mvt[:, 5])
        plt.xlabel('t/s')
        plt.ylabel('moment cinétique z')
        plt.grid()
        plt.figure()
        plt.plot(t,list_beta)
        plt.xlabel('t/s')
        plt.ylabel('error beta %')
        plt.show()
if __name__ == "__main__":
    from scipy.integrate import solve_ivp
    snake = Dynamic_snake(nb=6)
    # modèle géométrique
    snake.list_of_DH_matrices_head = np.eye(4)
    snake.antirouli_config = 0. * np.ones((snake.nb))
    snake.neck_config = 0. * np.ones((2))
    snake.joint_config = 0.* np.ones((snake.nb))

    snake.get_list_frame_R02()
    snake.vel_head = 0.*np.ones((6))
    R = Rotation.from_matrix(snake.list_of_DH_matrices_head[:3, :3])
    quat = R.as_quat()
    Q1 = quat[3];
    Q2 = quat[0];
    Q3 = quat[1];
    Q4 = quat[2];
    quat = np.array([Q1, Q2, Q3, Q4])
    T = 10

    x0 = np.zeros((13))
    x0[3:7] = quat;
    x0[:3] = snake.list_of_DH_matrices_head[:3, 3];
    x0[7:] = snake.vel_head
    n = int(3*T*100+1)
    t = np.linspace(0, 3*T, n)


    A = np.array([[T**7, T ** 6, T ** 5, T ** 4, T ** 3, T ** 2, T, 1], [0, 0, 0, 0, 0, 0, 0, 1],
                  [(T / 4) ** 7,(T /4) ** 6, (T / 4) ** 5, (T / 4) ** 4, (T / 4) ** 3, (T / 4) ** 2, (T / 4), 1],
                  [(3*T / 4) ** 7,(3*T / 4) ** 6, (3*T / 4) ** 5, (3*T / 4) ** 4, (3*T / 4) ** 3, (3*T / 4) ** 2, (3*T / 4), 1],
                  [7 * T ** 6,6 * T ** 5, 5 * T ** 4, 4 * T ** 3, 3 * T ** 2, 2 * T, 1, 0], [0,0, 0, 0, 0, 0, 1, 0],
                  [42 * T ** 5,30 * T ** 4, 20 * T ** 3, 12 * T ** 2, 6 * T, 2, 0, 0], [0,0, 0, 0, 0, 2, 0, 0]])
    y = np.array([0, 0, 1 / 2,-1/2, 0, 0, 0, 0])
    a = np.dot(np.linalg.inv(A), y)

    q=np.array([])
    qd = np.array([])
    qdd = np.array([])

    def f(t, a,T):
        t = t%T
        return (a[0] * t ** 7+a[1] * t ** 6 + a[2] * t ** 5 + a[3] * t ** 4 + a[4] * t ** 3 + a[5] * t ** 2 + a[6] * t + a[
            7])  # *st


    def df(t, a,T):
        t = t %T
        return 7*a[0]*t**6+6 * a[1] * t ** 5 + 5 * a[2] * t ** 4 + 4 * a[3] * t ** 3 + 3 * a[4] * t ** 2 + 2 * a[5] * t + a[
            6]  # (5*a[0]*t**4+4*a[1]*t**3+3*a[2]*t**2+a[3])*st+2*pi/1*(a[0]*t**5+a[1]*t**4+a[2]*t**3+a[3]*t)*ct


    def ddf(t, a,T):
        t = t % T
        return 42*a[0]*t**5+30 * a[1] * t ** 4 + 20 * a[2] * t ** 3 + 12 * a[3] * t ** 2 + 6 * a[4] * t + 2 * a[
            5]  # temp1+temp2+temp3+temp4


    snake.double_pendule(f(5, a, T)*0, df(5, a, T),ddf(5, a, T)*0,np.array([1,0,0,0,0,1]), 5)
    for i in range(t.shape[0]):
        q = np.append(q, f(t[i],a,T))
        qd = np.append(qd, df(t[i], a,T))
        qdd = np.append(qdd, ddf(t[i], a,T))
    plt.subplot(3,1,1)
    plt.plot(t,q,label='q')
    plt.xlabel('t')
    plt.ylabel('position/ rad')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(t,qd,label='dq')
    plt.xlabel('t')
    plt.ylabel('vitesse/ rad/s')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(t, qdd,label='ddq')
    plt.xlabel('t')
    plt.ylabel('accélération/ rad/(s^2)')
    plt.legend()
    plt.show()
    b=0

    sol = solve_ivp(snake.simu_dynamic_system_test_centrifuge, [t[0], t[-1]], x0, methode='RK45', t_eval=t,args=(a,T,False))  # ,mxstep=50)
    X = sol.y
    snake.list_of_DH_matrices_head = np.eye(4)
    sol = solve_ivp(snake.simu_dynamic_system_test_centrifuge, [t[0], t[-1]], x0, methode='RK45', t_eval=t,
                    args=(a, T, True))
    X2 = sol.y
    snake.center_of_masse(X,t,a,T,X2)
    snake.simple_plot_test_centrifugue(X, t,a,T)

    #plt.show()
    sol2 = solve_ivp(snake.simu_dynamic_system_duble_pendule, [t[0], t[-1]], x0, methode='RK45', t_eval=t,args=(a,T))
    X2 = sol2.y
    snake.center_of_masse_double_pendule(X, t, a,T)



    '''
    plt.figure()
    plt.imshow(np.abs(A-M))  # ,cmap = plt.cm.gray)
    for i in range(6):
        for j in range(6):
            text = plt.text(j, i, round((A-M)[i, j], 2),
                            ha="center", va="center", color="w")
    plt.show()
    '''
    '''
    acc,q1 = snake.direct_forward_NE(qd,tau)
    print('acc',acc)
    print('qdd1', q1)
    
    acc2,q2,A,C=snake.orin_algo_direct_dyn(qd=qd,tau=tau)
    print('acc2', acc2)
    print('qdd2', q2)
    print('M',A)
    
    x,tau,_,_ = snake.inverse_forward_NE(qd = qd, qdd=q1)
    print('acc2',x)
    print('tau',tau)
    eta_d,M_ = snake.motion_equation_head(qd, q1)
    print('new',A-M_)
    A11,A12 = snake.orin_algo_inverse_dyn(conf,qd,q1)
    print('new2', A - A11)
    print('new3', A11 - M_)

    g1 = snake.list_of_DH_matrices_joint[:,:4]
    g2 = snake.DH_matrix_neck
    g12 = np.dot(g1,g2)
    invg1 = snake.inverse_configuration(g1)
    invg2 = snake.inverse_configuration(g2)
    invg12 = snake.inverse_configuration(g12)
    Ad1 = snake.adjoint_matrix_2(invg1)
    Ad2 = snake.adjoint_matrix_2(invg2)
    Ad12 = snake.adjoint_matrix_2(invg12)
    temp = np.dot(Ad2,Ad1)
    print(temp)
    print(Ad12)
    #print(np.dot(A,x)+C)
    #print(q1-q2)
    '''