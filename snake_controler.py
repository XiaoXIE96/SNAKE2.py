from SNAKE_2 import snake
from Dynamic_Snake import Dynamic_snake
import numpy as np
from math import cos, sin, pi, sqrt, atan2, acos,exp
import math
import matplotlib.pyplot as plt
from scipy.linalg import expm, null_space
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy import integrate,optimize
from mpl_toolkits.mplot3d import Axes3D
from Surface_2 import surface
import copy
from scipy.integrate import solve_ivp
import control as control
class snake_controler(Dynamic_snake):
    def __init__(self):
        Dynamic_snake.__init__(self,nb=6)
    def simu_dynamic_system_2(self, t, x, u, A_q, B_q,q0,g0):
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

        print('tt', t)
        n = self.nb*2+2
        delta_q = x[12:12+n]
        dot_q = x[12+n:]
        ddot_q = np.dot(A_q,delta_q)+np.dot(B_q,u)

        delta_psi = x[:6] ; dot_psi = x[6:12]
        psi_hat = np.zeros((4, 4))
        psi_hat[:3, :3] = self.skew_symetric_matrix(delta_psi[3:])
        psi_hat[:3, 3] = delta_psi[:3]
        self.list_of_DH_matrices_head = np.dot(g0, expm(psi_hat))
        self.geometrical_model(delta_q+q0)
        self.vel_head = dot_psi
        xd = np.zeros((12+2*n))
        ##################################################
        xd[12:12+n] = dot_q
        xd[12+n:] = ddot_q
        eta0_d, _, _, _ = self.inverse_forward_NE(qd=dot_q, qdd=ddot_q)
        # eta0_d = self.orin_algo_inverse_dyn(qd=qd,qdd=qdd)
        xd[:6] = dot_psi
        xd[6:12] = eta0_d

        return xd

    def linearised_dynamic_system(self,q0):
        self.geometrical_model(q=q0)
        self.find_equilibrum_head_config_test(tau=1E-6)
        g0 = self.list_of_DH_matrices_head
        n = 2*self.nb+2

        '''
        R = Rotation.from_matrix(self.list_of_DH_matrices_head[:3,:3])
        angle_euler = R.as_euler('XYZ') # euler augles expressed in F_head
        P_0 = np.dot(self.list_of_DH_matrices_head[:3,:3].T,self.list_of_DH_matrices_head[:3, 3])
        X_0[:3] = P_0; X_0[3:6] = angle_euler*
        '''

        M_g0, M_q0 = self.orin_algo_inverse_dyn()
        DpsiW = self.R_h_wrench_jacobian_matrix_psi(1E-5)
        DqW = self.R_h_wrench_jacobian_matrix_q(1E-5)
        inv_Mg0 = np.linalg.inv(M_g0)
        A = np.zeros((12+2*n,12+2*n))
        A[6:12,:6] = np.dot(inv_Mg0,DpsiW); A[:6,6:12] = np.eye(6)
        A[12:12+n,12+n:] = np.eye(n)
        A_q= np.dot(np.linalg.pinv(M_q0), DqW)
        A[12+n:,12:12+n] = A_q

        print('eigen',np.linalg.eigvals(A))

        B = np.zeros((12+2*n, 6))
        B[6:12,:] = np.eye(6)
        B_q =  -np.dot(np.linalg.pinv(M_q0),M_g0)
        B[12 + n:, :] = B_q
        C = np.zeros((6,12))
        for i in range(6):
            C[i,i] = 1
        controlability_mat = np.zeros((12+2*n,(12+2*n)*6))
        temp = B
        for i in range(12+2*n):
            controlability_mat[:,i*6:(i+1)*6] = temp
            temp = np.dot(A,temp)
        print('controlability',np.linalg.matrix_rank(controlability_mat),12+2*n)
        return A,B,g0,C,A_q,B_q,M_q0,DqW

    def linearised_dynamic_system_2(self,q0):
        self.geometrical_model(q=q0)
        self.find_equilibrum_head_config_test(tau=1E-6)
        g0 = self.list_of_DH_matrices_head
        n = 2*self.nb+2

        '''
        R = Rotation.from_matrix(self.list_of_DH_matrices_head[:3,:3])
        angle_euler = R.as_euler('XYZ') # euler augles expressed in F_head
        P_0 = np.dot(self.list_of_DH_matrices_head[:3,:3].T,self.list_of_DH_matrices_head[:3, 3])
        X_0[:3] = P_0; X_0[3:6] = angle_euler*
        '''

        M_g0, M_q0 = self.orin_algo_inverse_dyn()
        DpsiW = self.R_h_wrench_jacobian_matrix_psi(1E-5)
        print('D_g0_W dans Fh',np.linalg.eigvals(DpsiW))
        DpsiW0 = self.R0_wrench_jacobian_matrix(1E-5)
        print('D_g0_W dans Fg',np.linalg.eigvals(DpsiW0))
        DqW = self.R_h_wrench_jacobian_matrix_q(1E-5)
        inv_Mg0 = np.linalg.inv(M_g0)
        A1 = np.zeros((12,12))
        A2 = np.zeros((2 * n, 2 * n))
        A1[6:,:6] = np.dot(inv_Mg0,DpsiW); A1[:6,6:] = np.eye(6)
        A2[:n,n:] = np.eye(n)
        A_q= np.dot(np.linalg.pinv(M_q0), DqW)
        A2[n:,:n] = A_q


        #B = np.zeros((12,2*(2*self.nb+2)))
        #B[6:12,:(2*self.nb+2)] = np.dot(inv_Mg0,DqW); B[6:12,(2*self.nb+2):(2*self.nb+2)*2] = np.dot(-inv_Mg0,M_q0)
        B1 = np.zeros((12, 6))
        B2= np.zeros((2 * n, 6))
        B1[6:,:] = np.eye(6)
        B_q =  -np.dot(np.linalg.pinv(M_q0),M_g0)
        B2[n:, :] = B_q
        C = np.zeros((6,12))
        C[:6,:6] = np.eye(6)
        controlability_mat = np.zeros((12,(12)*6))
        temp = B1
        for i in range(12):
            controlability_mat[:,i*6:(i+1)*6] = temp
            temp = np.dot(A1,temp)
        print('controlability 1',np.linalg.matrix_rank(controlability_mat),12)
        controlability_mat2 = np.zeros((2 * n, (2 * n) * 6))
        temp = B2
        for i in range(2*n):
            controlability_mat2[:, i * 6:(i + 1) * 6] = temp
            temp = np.dot(A2, temp)
        print('controlability 2', np.linalg.matrix_rank(controlability_mat2), 2 * n)

        print('eigen', np.linalg.eigvals(A1))
        D = 1 * np.eye(6)
        K = np.zeros((6, 12));
        K[:, 6:] = D
        AA = A1 - np.dot(B1, K)
        print('eigen', np.linalg.eigvals(AA))
        T = np.zeros((12,n))
        T[6:,:] = np.dot(inv_Mg0, DqW)
        return A1,B1,g0,C,A2,B2,T,M_q0,DqW


    def simul_Full_state_feedback(self,q0):
        def calculate_step(t,qn_moins_qnm1,dq_nm1,ddq_n):
            left = t**2/2*ddq_n+t*dq_nm1
            right = qn_moins_qnm1
            return np.linalg.norm(left-right)
        from scipy import signal
        from scipy import optimize
        n = self.nb*2+2
        A,B,g0,C,A_q,B_q = self.linearised_dynamic_system(q0)
        self.vel_head = np.zeros((6))
        self.acc_head = np.zeros((6))
        psi_hat = np.zeros((4, 4))
        d_psi = np.array([0.,0.,0.,pi/9,0.,0.])
        psi_hat[:3, :3] = self.skew_symetric_matrix(d_psi[3:])
        psi_hat[:3, 3] = d_psi[:3]
        self.list_of_DH_matrices_head = np.dot(g0,expm(psi_hat))
        print('g0',g0)
        self.geometrical_model(q=q0)
        ####get Xn
        X = np.zeros((12+2*n))
        '''
        R = Rotation.from_matrix(self.list_of_DH_matrices_head[:3, :3])
        angle_euler = R.as_euler('XYZ')  # euler augles expressed in F_head
        P = np.dot(self.list_of_DH_matrices_head[:3, :3].T, self.list_of_DH_matrices_head[:3, 3])
        X[:3] = P;
        X[3:6] = angle_euler
        '''
        X[:6] = d_psi;
        ########################### u=-kx
        K =0.1*np.eye(6)
        t = np.linspace(0,10,1001)
        dt = 0.01
        list_X=np.zeros((40,1001))
        list_X[:, 0] = X
        for i in range(t.shape[0]-1):
            u = -np.dot(K,X[6:12])
            sol = solve_ivp(snake.simu_dynamic_system_2, [t[i], t[i+1]], X, methode='RK45', t_eval=[t[i], t[i+1]],args=(u, A_q, B_q,q0,g0))
            X = sol.y
            X = X[:,-1]
            list_X[:,i+1]=X

        return list_X

    def simul_Full_state_feedback_linearized(self, q0,D1,D2):
        def sys_dyn(t,x,A,B,u):

            return np.dot(A,x)+np.dot(B,u)

        from scipy import signal
        from scipy import optimize
        n = self.nb * 2 + 2
        A1,B1,g0,C,A2,B2,T,M_q0,DqW = self.linearised_dynamic_system_2(q0)

        self.vel_head = np.zeros((6))
        self.acc_head = np.zeros((6))
        psi_hat = np.zeros((4, 4))
        d_psi = np.array([0., 0., 0., pi / 18, 0., 0.])
        psi_hat[:3, :3] = self.skew_symetric_matrix(d_psi[3:])
        psi_hat[:3, 3] = d_psi[:3]
        self.list_of_DH_matrices_head = np.dot(g0, expm(psi_hat))
        print('g0', g0)
        self.geometrical_model(q=q0)
        ####get Xn
        x = np.zeros((12))
        '''
        R = Rotation.from_matrix(self.list_of_DH_matrices_head[:3, :3])
        angle_euler = R.as_euler('XYZ')  # euler augles expressed in F_head
        P = np.dot(self.list_of_DH_matrices_head[:3, :3].T, self.list_of_DH_matrices_head[:3, 3])
        X[:3] = P;
        X[3:6] = angle_euler
        '''
        x[:6] = d_psi

        ########################### u=-kx
        K = np.zeros((6, 12))
        K[:, :6] = D1 * np.eye(6)
        K[:, 6:] = D2 * np.eye(6)
        t = np.linspace(0, 10, 1001)
        dt = 0.01
        list_X = np.zeros((12, 1001))
        list_X[:, 0] = x
        list_Q = np.zeros((2*n, 1001))
        Q = np.zeros((2*n))
        #####
        dq = np.ones((2*self.nb+2))*(5/180*pi)
        for i in range(t.shape[0] - 1):
            u = -np.dot(K, x)
            print('input',u)
            #sol = solve_ivp(sys_dyn, [t[i], t[i + 1]], x, t_eval=[t[i + 1]], args=(A1,B1,u))
            sol = solve_ivp(sys_dyn, [t[i], t[i + 1]], x, t_eval=[t[i + 1]], args=(A1, B1, u))
            x = sol.y
            list_X[:, i + 1] = x[:, 0]
            x = x[:, 0]

            sol2 = solve_ivp(sys_dyn, [t[i], t[i + 1]], Q,
                            t_eval=[t[i + 1]], args=(A2, B2, u))
            Q = sol2.y
            Q = Q[:, 0]
            #Q = Q[:, -1]
            list_Q[:, i + 1] = Q
        return list_X, list_Q
    def test_linear_sys(self,q0):
        def sys_dyn(t,x,A,B,u):
            print('here',t)
            return np.dot(A,x)+np.dot(B,u)
        n = self.nb * 2 + 2
        A1, B1, g0, C, A2, B2, T, M_q0, DqW = self.linearised_dynamic_system_2(q0)
        psi_hat = np.zeros((4, 4))
        #d_psi = np.array([0., 0., 0., pi / 9, 0., 0.])
        d_psi = np.array([0., 0., 0., pi / 36, 0., 0.])
        psi_hat[:3, :3] = self.skew_symetric_matrix(d_psi[3:])
        psi_hat[:3, 3] = d_psi[:3]
        self.list_of_DH_matrices_head = np.dot(g0, expm(psi_hat))
        self.geometrical_model(q=q0)
        x = np.zeros((12))
        x[:6] = d_psi
        list_X = np.zeros((12, 2001))
        list_X[:, 0] = x
        t = np.linspace(0, 20, 2001)
        for i in range(t.shape[0] - 1):
            u=np.zeros((6))
            #sol = solve_ivp(sys_dyn, [t[i], t[i + 1]], x, t_eval=[t[i + 1]], args=(A1,B1,u))
            sol = solve_ivp(sys_dyn, [t[i], t[i + 1]], x, t_eval=[t[i + 1]], args=(A1,B1,u))

            x = sol.y
            list_X[:, i + 1] = x[:, 0]
            x = x[:, 0]

        return list_X

    def simul_Full_state_feedback_linearized_nullspace(self, q0):
        def sys_dyn(t,x,A,B,u):
            print('here',t)
            return np.dot(A,x)+np.dot(B,u)
        from scipy import signal
        from scipy import optimize
        n = self.nb * 2 + 2
        A1,B1,g0,C,A2,B2,T,M_q0,DqW=self.linearised_dynamic_system_2(q0)
        nS = null_space(M_q0)

        self.vel_head = np.zeros((6))
        self.acc_head = np.zeros((6))
        psi_hat = np.zeros((4, 4))
        d_psi = np.array([0., 0., 0., pi / 9, 0., 0.])
        psi_hat[:3, :3] = self.skew_symetric_matrix(d_psi[3:])
        psi_hat[:3, 3] = d_psi[:3]
        self.list_of_DH_matrices_head = np.dot(g0, expm(psi_hat))
        self.geometrical_model(q=q0)
        ####get Xn
        x = np.zeros((12))
        x[:6] = d_psi
        ########################### u=-kx

        t = np.linspace(0, 30, 3001)

        list_X = np.zeros((12, 3001))
        list_X[:, 0] = x
        list_Q = np.zeros((n, 3001))
        list_u = np.zeros((6, 3001))
        list_ddq = np.zeros((n, 3001))
        Q = np.zeros((2*n))
        #####
        dq = np.ones((2*self.nb+2))*(5/180*pi)
        for i in range(t.shape[0] - 1):
            #u = -np.dot(K, x[6:12])
            K = np.zeros((6, 12))
            if i<1000:
                K[:, :6] = 0.2/(exp(1)-1)*(exp(i/1000)-1) * np.eye(6)
                K[:, 6:] = 0.2/(exp(1)-1)*(exp(i/1000)-1) * np.eye(6)
                print('check',0.2*(exp(i/1000)-1))
            else:
                K[:, :6] = 0.2 * np.eye(6)
                K[:, 6:] = 0.2  * np.eye(6)
            #K[:, :6] =  np.eye(6)
            #K[:, 6:] =  np.eye(6)
            u = -np.dot(K,x)
            #null_space of qdd
            qdd = np.ones((14))
            qdd = qdd / np.linalg.norm(qdd)
            temp = np.dot(nS.T, qdd)
            qdd = np.dot(nS, temp)
            #calculate the q and recalculate the u
            if np.linalg.norm(u)==0.:
                q=np.zeros((n))
            else:
                q = np.dot(np.linalg.pinv(DqW), u)
                #q = 0.3*q/np.linalg.norm(q)
                #u = np.dot(DqW,q)
            list_u[:,i] = u
            print('input',u)
            #sol = solve_ivp(sys_dyn, [t[i], t[i + 1]], x, t_eval=[t[i + 1]], args=(A1,B1,u))
            sol = solve_ivp(sys_dyn, [t[i], t[i + 1]], x, t_eval=[t[i + 1]], args=(A1, B1, u))
            if i == 999:
                print('look',np.dot(A1,x)+np.dot(B1,u))
                print(np.dot(A1,x))
                print(np.dot(B1,u))
            x = sol.y
            list_X[:, i + 1] = x[:, 0]
            x = x[:, 0]
            dQ = np.dot(A2, Q) + np.dot(B2, u)
            '''
            sol2 = solve_ivp(sys_dyn, [t[i], t[i + 1]], Q,
                            t_eval=[t[i + 1]], args=(A2, B2, u))
            Q = sol2.y
            Q = Q[:, 0]
            #Q = Q[:, -1]
            list_Q[:, i + 1] = Q
            '''
            list_Q[:, i + 1] = np.dot(np.linalg.pinv(DqW),u)
            list_ddq[:,i] = dQ[n:]
        return list_X, list_Q, list_u,list_ddq

    def simul_Full_state_feedback_linearized_lqr(self,q0):
        def sys_dyn(t,x,A,B,u):
            print('here',t)
            return np.dot(A,x)+np.dot(B,u)
        from scipy import signal
        from scipy import optimize
        self.geometrical_model(q=q0)
        self.find_equilibrum_head_config_test(tau=1E-6)
        n = self.nb * 2 + 2
        M_g0, M_q0 = self.orin_algo_inverse_dyn()
        DpsiW = self.R_h_wrench_jacobian_matrix_psi(1E-5)
        DqW = self.R_h_wrench_jacobian_matrix_q(1E-5)
        inv_Mg0 = np.linalg.inv(M_g0)
        A = np.zeros((12, 12))
        A[6:, :6] = np.dot(inv_Mg0, DpsiW);
        A[:6, 6:] = np.eye(6)
        B = np.zeros((12,n))
        B[6:,:] = np.dot(inv_Mg0,DqW)
        nS = null_space(M_q0)
        psi_hat = np.zeros((4, 4))
        d_psi = np.array([0., 0., 0., pi / 9, 0., 0.])

        ####get Xn
        x = np.zeros((12))
        x[:6] = d_psi
        ########################### u=-kx

        t = np.linspace(0, 30, 3001)

        list_X = np.zeros((12, 3001))
        list_X[:, 0] = x
        list_Q = np.zeros((n, 3001))
        def solve_DARE(A,B,Q,R):
            X=Q
            maxiter=500
            eps = 0.01
            for i in range(maxiter):
                Xn=A.T@X@A-A.T@X@B@np.linalg.pinv(R+B.T*X*B)*B.T*X*A+Q

        #K,_,__ = control.lqr(A=A,B=B,Q=np.ones((12,12)),R = np.ones((n,n))*10)
        ''' 
        for i in range(t.shape[0] - 1):
            #u = -np.dot(K, x[6:12])
            K = np.zeros((n, 12))

            q = -np.dot(K,x)
            #null_space of qdd
            #sol = solve_ivp(sys_dyn, [t[i], t[i + 1]], x, t_eval=[t[i + 1]], args=(A1,B1,u))
            sol = solve_ivp(sys_dyn, [t[i], t[i + 1]], x, t_eval=[t[i + 1]], args=(A, B, q))

            x = sol.y
            list_X[:, i + 1] = x[:, 0]
            x = x[:, 0]


            list_Q[:, i + 1] = q
        
        return list_X, list_Q
        '''
    def global_frame_linearized_sys(self):
        inv_g0 = self.inverse_configuration(self.list_of_DH_matrices_head)
        Ad_inv_g0 = self.adjoint_matrix_2(inv_g0)
        M_g0, M_q0 = self.orin_algo_inverse_dyn()
        M_G = np.dot(Ad_inv_g0.T,np.dot(M_g0,Ad_inv_g0))
        M_q0_G = np.dot(Ad_inv_g0.T,M_q0)
        print(M_q0)
        plt.figure()
        plt.imshow(np.abs(M_G))#,cmap = plt.cm.gray)
        for i in range(6):
            for j in range(6):
                text = plt.text(j, i, round(M_G[i, j],2),
                               ha="center", va="center", color="w")
        plt.figure()
        plt.imshow(np.abs(M_q0))
        for i in range(6):
            for j in range(14):
                text = plt.text(j, i, round(M_q0_G[i, j],2),
                               ha="center", va="center", color="w")
        plt.show()

    def simple_plot(self,h, q, n, r,stiff_body):
        """
        simple plot
        """

        # 生成画布
        fig = plt.figure(figsize=(8, 6), dpi=80)

        # 打开交互模式
        plt.ion()

        # 循环
        if stiff_body:
            self.neck_config = n
            self.joint_config = q
            self.antirouli_config = r
            self.get_neck_frame();
            self.get_list_joint_frame();
            self.get_list_roulie_frame();
        k=10
        for index in range(int(h.shape[0]/10)):
            # 清除原有图像
            quat = h[index*k, 3:7]
            Q1 = quat[0];
            Q2 = quat[1];
            Q3 = quat[2];
            Q4 = quat[3];

            R = np.array([[2 * (Q1 ** 2 + Q2 ** 2) - 1, 2 * (Q2 * Q3 - Q1 * Q4), 2 * (Q2 * Q4 + Q1 * Q3)],
                          [2 * (Q2 * Q3 + Q1 * Q4), 2 * (Q1 ** 2 + Q3 ** 2) - 1, 2 * (Q3 * Q4 - Q1 * Q2)],
                          [2 * (Q2 * Q4 - Q1 * Q3), 2 * (Q3 * Q4 + Q1 * Q2), 2 * (Q1 ** 2 + Q4 ** 2) - 1]])
            T = np.eye(4)
            T[:3, :3] = R
            P = h[index*k, :3]
            T[:3, 3] = P
            self.list_of_DH_matrices_head = T
            if not stiff_body:
                self.neck_config = n[index*k, :]
                self.joint_config = q[index*k, :]
                self.antirouli_config = r[index*k, :]
                self.get_neck_frame();
                self.get_list_joint_frame();
                self.get_list_roulie_frame();
            self.get_list_frame_R02()

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

if __name__ == "__main__":
    snake = snake_controler()
    snake.list_of_DH_matrices_head = np.eye(4)
    snake.neck_config = np.array([0.0, 0.4749838594336039])
    snake.joint_config = np.array(
        [-0.5252277318821159, 0.2617993877991494, 0.3490658503988659, -0.6108652381980153, -0.6108652381980153,
         0.3490658503988659])
    snake.antirouli_config = np.zeros((6))
    snake.geometrical_model()
    snake.find_equilibrum_head_config_test(tau=1E-6)

    snake.global_frame_linearized_sys()
    '''
    snake.list_of_DH_matrices_head = np.eye(4)
    snake.neck_config = np.array([0.0, 0.4749838594336039])
    snake.joint_config = np.array(
        [-0.5252277318821159, 0.2617993877991494, 0.3490658503988659, -0.6108652381980153, -0.6108652381980153,
         0.3490658503988659])
    snake.antirouli_config = np.zeros((6))
    q0 = np.zeros((2 * snake.nb + 2))
    q0[:2] = snake.neck_config
    for i in range(snake.nb):
        q0[2 * i + 2] = snake.joint_config[i]
        q0[2 * i + 3] = snake.antirouli_config[i]
    X,Q,u,dqq=snake.simul_Full_state_feedback_linearized_nullspace(q0=q0)
    t = np.linspace(0, 10, 1001)
    plt.figure()
    plt.suptitle('input u', fontsize=14)
    for i in range(6):
        plt.subplot(6, 1, i + 1)
        plt.plot(t[:1000], u[i , :1000])
        plt.xlabel('time/s')
        #plt.ylabel(name2[i])

    plt.figure()
    plt.suptitle('delta_q', fontsize=14)
    for i in range(14):
        plt.subplot(7, 2, i + 1)
        plt.plot(t, Q[i,:])
        plt.xlabel('time/s')
        # plt.ylabel(name2[i])

    plt.figure()
    plt.suptitle('ddot_q', fontsize=14)
    for i in range(14):
        plt.subplot(7, 2, i + 1)
        plt.plot(t[:10], dqq[i, :10])
        plt.xlabel('time/s')
        # plt.ylabel(name2[i])
    plt.show()
    
    import csv

    with open("controller/linear_system_null_space_X.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(X)

    with open("controller/linear_system_null_space_q.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(Q)

    
    #complet_simu with delta_psi
    snake.list_of_DH_matrices_head = np.eye(4)
    snake.neck_config = np.array([0.0, 0.4749838594336039])
    snake.joint_config = np.array(
        [-0.5252277318821159, 0.2617993877991494, 0.3490658503988659, -0.6108652381980153, -0.6108652381980153,
         0.3490658503988659])
    snake.antirouli_config = np.zeros((6))
    q0 = np.zeros((2 * snake.nb + 2))
    q0[:2] = snake.neck_config
    for i in range(snake.nb):
        q0[2 * i + 2] = snake.joint_config[i]
        q0[2 * i + 3] = snake.antirouli_config[i]
    X=snake.simul_Full_state_feedback(q0=q0)

    import csv

    with open("controller/simul_Full_state_feedback.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(X)
    
    #no input non linear simu
    g0 = np.array([[0.99904884, 0.00222477, 0.04354833, 0.],
                   [0., 0.99869758, -0.05102095, 0.],
                   [-0.04360512, 0.05097242, 0.99774767, 0.06286409],
                   [0., 0., 0., 1.]])
    psi_hat = np.zeros((4,4))
    psi = np.array([0.,0.,0.,pi/9,0.,0.])
    psi_hat[:3, :3] = snake.skew_symetric_matrix(psi[3:])
    psi_hat[:3, 3] = psi[:3]
    snake.list_of_DH_matrices_head = np.dot(g0, expm(psi_hat))
    R = Rotation.from_matrix(snake.list_of_DH_matrices_head[:3, :3])
    quat = R.as_quat()
    Q1 = quat[3];
    Q2 = quat[0];
    Q3 = quat[1];
    Q4 = quat[2];
    quat = np.array([Q1, Q2, Q3, Q4])
    x0 = np.zeros((13))
    x0[3:7] = quat;
    x0[:3] = snake.list_of_DH_matrices_head[:3, 3];
    x0[7:] = snake.vel_head
    t = np.linspace(0, 10, 1001)
    sol = solve_ivp(snake.simu_dynamic_system, [t[0], t[-1]], x0, methode='RK45', t_eval=t,
                    args=(0, 0, 0, True))  # ,mxstep=50)
    print(sol.y.shape)
    import csv

    with open("controller/non_linear_system_d_psi.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(sol.y)
    
    #change delta_psi no input linear
    snake.list_of_DH_matrices_head = np.eye(4)
    snake.neck_config = np.array([0.0, 0.4749838594336039])
    snake.joint_config = np.array(
        [-0.5252277318821159, 0.2617993877991494, 0.3490658503988659, -0.6108652381980153, -0.6108652381980153,
         0.3490658503988659])
    snake.antirouli_config = np.zeros((6))
    q0 = np.zeros((2 * snake.nb + 2))
    q0[:2] = snake.neck_config
    for i in range(snake.nb):
        q0[2 * i + 2] = snake.joint_config[i]
        q0[2 * i + 3] = snake.antirouli_config[i]
    X,Q = snake.simul_Full_state_feedback_linearized(q0=q0)

    import csv

    with open("controller/linear_delta_psi.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(X)

    with open("controller/simul_Full_state_feedback_linearized_Q_K=0.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(Q)
    
    #change delta_q no input linear
    snake.list_of_DH_matrices_head = np.eye(4)
    snake.neck_config = np.array([0.0, 0.4749838594336039])
    snake.joint_config = np.array(
        [-0.5252277318821159, 0.2617993877991494, 0.3490658503988659, -0.6108652381980153, -0.6108652381980153,
         0.3490658503988659])
    snake.antirouli_config = np.zeros((6))
    q0 = np.zeros((2 * snake.nb + 2))
    q0[:2] = snake.neck_config
    for i in range(snake.nb):
        q0[2 * i + 2] = snake.joint_config[i]
        q0[2 * i + 3] = snake.antirouli_config[i]
    X, Q = snake.simul_Full_state_feedback_linearized(q0=q0)

    import csv

    with open("controller/linear_system_dq.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(X)
    
    #  controller/non_linear_system_dq  
    g0 = np.array([[0.99904884, 0.00222477, 0.04354833, 0.],
                   [0., 0.99869758, -0.05102095, 0.],
                   [-0.04360512, 0.05097242, 0.99774767, 0.06286409],
                   [0., 0., 0., 1.]])
    snake.list_of_DH_matrices_head = g0
    snake.neck_config = np.array([0.0, 0.4749838594336039])+np.ones(2)*(5/180*pi)
    snake.joint_config = np.array(
        [-0.5252277318821159, 0.2617993877991494, 0.3490658503988659, -0.6108652381980153, -0.6108652381980153,
         0.3490658503988659])+np.zeros((6))+np.ones(6)*(5/180*pi)
    snake.antirouli_config = np.zeros((6))+np.ones(6)*(5/180*pi)
    snake.geometrical_model()
    R = Rotation.from_matrix(snake.list_of_DH_matrices_head[:3, :3])
    quat = R.as_quat()
    Q1 = quat[3];
    Q2 = quat[0];
    Q3 = quat[1];
    Q4 = quat[2];
    quat = np.array([Q1, Q2, Q3, Q4])
    x0 = np.zeros((13))
    x0[3:7] = quat;
    x0[:3] = snake.list_of_DH_matrices_head[:3, 3];
    x0[7:] = snake.vel_head
    t = np.linspace(0, 10, 1001)
    sol = solve_ivp(snake.simu_dynamic_system, [t[0], t[-1]], x0, methode='RK45', t_eval=t,
                    args=(0, 0, 0, True))  # ,mxstep=50)
    print(sol.y.shape)
    import csv

    with open("controller/non_linear_system_dq.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(sol.y)
    '''


