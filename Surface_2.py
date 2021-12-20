import numpy as np
from math import sqrt, acos, pi,asin,sin,cos
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import time
class surface(object):
    def __init__(self,tau):
        # omega1 the angle of the first cercle
        self.R2 = 0.045 * tau
        self.R1 = 0.058 *tau
        self.a = (0.032 - 0.045 * sin(8.14 / 180 * pi) - 0.01168)*tau
        omega_ = asin(self.a / (0.058*tau))
        self.omega1 = pi - 2 * omega_

        #b = 0.045 * sin((86.99 - 90 + 8.14) / 180 * pi)*tau
        c = 0.058*cos(omega_)*tau
        f = 0.032*tau
        e = 0.006908*tau

        self.A = np.array([-c,self.a])
        self.B = np.array([-c-e,-f+self.a])
        self.C = np.array([c, self.a])
        self.D = np.array([c + e, -f + self.a])
        self.G = np.array([self.B[0]+self.R2*cos(8.14 / 180 * pi),self.B[1]+self.R2*sin(8.14 / 180 * pi)])
        self.H = np.array([-self.G[0],self.G[1]])
        b = -self.G[0]-0.012*tau
        omega__ = asin(b/self.R2)
        self.omega2 = pi/2-8.14 / 180 * pi+omega__
        self.E = np.array([-0.012*tau, self.G[1]-self.R2*cos(omega__)])
        self.F = np.array([-self.E[0],self.E[1]])
        self.g1 = np.array([[1,0,0],
                            [0,1,self.a],
                            [0,0,1]])
        self.g2 = np.array([[-sin(8.14/180*pi+self.omega2/2),-cos(8.14/180*pi+self.omega2/2),(self.B+self.E)[0]/2],
                            [cos(8.14/180*pi+self.omega2/2),-sin(8.14/180*pi+self.omega2/2),(self.B+self.E)[1]/2],
                            [0,0,1]])
        self.g3 = np.array([[-sin(8.14 / 180 * pi + self.omega2 / 2), cos(8.14 / 180 * pi + self.omega2 / 2), (self.D+self.F)[0]/2],
                            [-cos(8.14 / 180 * pi + self.omega2 / 2), -sin(8.14 / 180 * pi + self.omega2 / 2), (self.D+self.F)[1]/2],
                            [0, 0, 1]])
        self.S_sector_1 = pow(self.R1,2)*(self.omega1-sin(self.omega1))/2
        self.SQ_sector_1 = 4 * self.R1 * pow(sin(self.omega1/2), 3) / (3*(self.omega1 - sin(self.omega1)))-self.R1*cos(self.omega1/2)
        self.SQ_sector_1 = self.SQ_sector_1*self.S_sector_1
        self.S_sector_2 = pow(self.R2, 2) * (self.omega2 - sin(self.omega2)) / 2
        self.SQ_sector_2 = 4 * self.R2 * pow(sin(self.omega2 / 2), 3) / (3 * (self.omega2 - sin(self.omega2))) - self.R2 * cos(self.omega2 / 2)
        self.SQ_sector_2 = self.SQ_sector_2 * self.S_sector_2
        list_point = np.array([self.A, self.C, self.D, self.F, self.E, self.B])
        S,Sq = self.get_centerpoint(np.reshape(list_point,(-1,2)))
        self.section = S + self.S_sector_1 + self.S_sector_2 * 2
        self.center = Sq+np.dot(self.g1,np.array([0,self.SQ_sector_1,self.S_sector_1]))[:1]+np.dot(self.g2,np.array([0,self.SQ_sector_2,self.S_sector_2]))[:1]+np.dot(self.g3,np.array([0,self.SQ_sector_2,self.S_sector_2]))[:1]
        self.center = self.center/self.section
        '''
        self.r = r
        self.a = a
        #self.g = g

        ################!!!!!!!!!!!!!!!!!#####################
        sq3 = sqrt(3)
        self.R = r/sq3
        self.g1 = np.array([[1,0,0],
                       [0,1,a/sqrt(3)-sqrt(3)*self.r/2],
                       [0,0,1]])
        self.g2 = np.array([[-1/2, -sqrt(3)/2, -self.a/2+0.75*self.r],
                          [sqrt(3)/2, -1/2, sqrt(3)*(-self.a/6+self.r/4)],
                          [0, 0, 1]])
        self.g3 = np.array([[-1/2, sqrt(3)/2, self.a/2-0.75*self.r],
                          [-sqrt(3)/2, -1/2, sqrt(3)*(-self.a/6+self.r/4)],
                          [0, 0, 1]])

        S_secteur = pi * self.R ** 2 / 3#pi / 6 * self.r ** 2  ## secteur circulaire
        S_losange =self.r*self.R# (sq3/4+1/(sq3*4))*pow(r,2)
        self.S_cutted = S_losange - S_secteur #(sqrt(3) / 2 - pi / 6) * self.r ** 2
        #P_secteur = S_secteur * (-sqrt(3) / 2 + 2 / pi) * self.r
        #SQ_losange = pow(r,3)/9
        #SQ_secteur = S_secteur*(-self.R/2+sqrt(3)*self.R/pi)
        #SQ_cutted = SQ_losange - SQ_secteur

        SQ_triang = np.array([0, self.r ** 3 / 8])
        S_sec, SQ_sec = self.calculat_S_SP(x=-self.r / 2, x2=self.r / 2)

        #print(self.R**3*pi/6/self.S_cutted)
        self.P_cutted = self.R**3*pi/6/self.S_cutted#(SQ_triang - SQ_sec)[1]/self.S_cutted
        #print(self.P_cutted)
        #self.P_cutted = self.R/3#SQ_cutted/self.S_cutted

        self.S_triang = sqrt(3) / 4 * pow(self.a, 2)
        ##P_cutted = np.dot(g, np.array([0, P_cutted, 1]))
        self.section = self.S_triang - 3 * self.S_cutted
        '''
    def approximation_surface(self,z):
        y = np.linspace(-sqrt(3) * self.a / 6, z-0.01, int((z+sqrt(3) * self.a / 6 ) / 0.02))
        #y = np.linspace(-sqrt(3)*self.a/6, self.a/sqrt(3),int(sqrt(3)*self.a/2/0.01))
        list_point = np.array([])
        for i in range(y.shape[0]):
            '''
            if -sqrt(3)*self.a/6 <= y[i] <= self.a/sqrt(3):
                    x = (-y[i]+self.a/sqrt(3))/sqrt(3)
                    list_x = np.linspace(-x, x, int((2 * x) / 0.05))
                    for j in range(list_x.shape[0]):
                        list_point = np.append(list_point, np.array([[list_x[j], y[i]]]))
            '''
            if self.a/sqrt(3)-3*self.R/2<y[i]<=self.a/sqrt(3)-self.R:
                x = sqrt(self.R**2-(y[i]-self.a/sqrt(3)+2*self.R)**2)
                list_x = np.linspace(-x,x,int((2*x)/0.02))
                for j in range(list_x.shape[0]):
                    list_point = np.append(list_point,np.array([[list_x[j],y[i]]]))

            elif -sqrt(3)*self.a/6+sqrt(3)*self.r/2 <= y[i] <= self.a/sqrt(3)-3*self.R/2:
                    x = (-y[i]+self.a/sqrt(3))/sqrt(3)
                    list_x = np.linspace(-x, x, int((2 * x) / 0.02))
                    for j in range(list_x.shape[0]):
                        list_point = np.append(list_point, np.array([[list_x[j], y[i]]]))
            elif -sqrt(3)*self.a/6<=y[i]<-sqrt(3)*self.a/6+sqrt(3)*self.r/2:
                xr = sqrt(self.R**2-(y[i]+sqrt(3)*self.a/6-self.R)**2)+self.a/2-self.r
                xl = -sqrt(self.R ** 2 - (y[i] + sqrt(3) * self.a / 6 - self.R) ** 2) - self.a / 2 + self.r
                list_x = np.linspace(xl, xr, int((xr-xl) / 0.02))
                for j in range(list_x.shape[0]):
                    list_point = np.append(list_point, np.array([[list_x[j], y[i]]]))

        s=list_point.shape[0]/2*0.01*0.01
        return s,list_point
    def contour_surface(self):
        x = np.linspace(self.A[0],self.C[0],100)
        y = np.array([])
        for i in range(x.shape[0]):
            y = np.append(y,sqrt(self.R1**2-x[i]**2))

        x1 = np.linspace(-self.R2*sin(self.omega2/2),self.R2*sin(self.omega2/2),25)
        y1 = np.array([])
        for i in range(x1.shape[0]):

            y1 = np.append(y1, sqrt(self.R2 ** 2 - x1[i] ** 2)-self.R2*cos(self.omega2/2))
        for i in range(x1.shape[0]):

            temp = np.dot(self.g3, np.array([x1[i],y1[i],1]))
            x = np.append(x,temp[0]);y = np.append(y,temp[1])
        for i in range(x1.shape[0]):
            temp = np.dot(self.g2, np.array([x1[i],y1[i],1]))
            x = np.append(x,temp[0]);y = np.append(y,temp[1])
        x = np.append(x,self.A[0]);y = np.append(y,self.A[1])
        x_y = np.empty((2,x.shape[0]))

        x_y[0,:] = x
        x_y[1,:] = y
        return x_y


    def search_equilibrum_line(self,rho_w, rho_r):
        '''

        :param rho_w: density of water
        :param rho_r: density of robot
        :return:
        '''
        F = lambda S: S*rho_w-rho_r*self.section
        a=-sqrt(3)*self.a / 6

        b = self.a/sqrt(3)

        while (b - a) / 2 > 1E-15:
            g1 = np.eye(4);
            g1[2, 3] = a
            S1, SQ1 = self.Immersed_surface_theoric(g1)
            F1 = F(S1)

            g2 = np.eye(4);
            g2[2, 3] = b
            S2, SQ2 = self.Immersed_surface_theoric(g2)
            F2 = F(S2)
            m = (a + b) / 2
            g = np.eye(4); g[2, 3] = m
            S,SQ = self.Immersed_surface_theoric(g)
            if F(S) * F1 < 0:
                b = m
            else:
                a = m
        return a

    def find_intersection_between_2_line(self,A,B, R33, R32, P3):
        x1 = A[0];y1 = A[1]; x2 = B[0]; y2 = B[1]
        k = (y1-y2)/(x1-x2)
        x = ( R33*(k*x1-y1)-P3 )/ (R32+R33*k)
        y = (-R32*x-P3)/R33
        return np.array([x,y])
    def Immersed_surface_theoric(self,g):
        R33 = g[2, 2]
        R32 = g[2, 1]
        P3 = g[2, 3]
        Sim = 0
        SimQ = np.zeros((2))
        list_point = [self.A, self.C, self.D, self.F, self.E, self.B, self.A]
        list_point_immersed = np.array([])
        for i in range(len(list_point)-1):
            temp = R32*list_point[i][0]+R33*list_point[i][1]+P3
            temp2 = R32*list_point[i+1][0]+R33*list_point[i+1][1]+P3
            if temp<=0:
                list_point_immersed=np.r_[list_point_immersed,list_point[i]]
            if temp*temp2<0:
                intersection = self.find_intersection_between_2_line(list_point[i],list_point[i+1],R33,R32,P3)
                list_point_immersed=np.r_[list_point_immersed,intersection]
        if list_point_immersed.shape[0] != 0:
            Sim_, SimQ_ = self.get_centerpoint(np.reshape(list_point_immersed,(-1,2)))
            Sim += Sim_; SimQ += SimQ_
        for i in range(3):
            Sim_,SimQ_ = self.find_intersection(i, R32, R33, P3)
            Sim += Sim_; SimQ += SimQ_
        return Sim,SimQ


    def calculat_S_SP(self, x , x2 ,number):
        if number == 0:
            r = self.R1; c = self.R1*cos(self.omega1/2)
        elif number == 1:
            r = self.R2;c = self.R2*cos(self.omega2/2)
        elif number == 2:
            r = self.R2;c = self.R2*cos(self.omega2/2)
        temp1 = pow(r, 2) - pow(x2, 2);
        temp2= pow(r, 2) - pow(x, 2)
        temp3 = pow(r, 2) * asin(x2 / r) + x2 * sqrt(temp1) #r2 * asin(x/r) + x*sqrt(r2-x2)
        temp4 = pow(r, 2) * asin(x / r) + x * sqrt(temp2)
        temp5 = pow(temp1, 1.5); temp6 = pow(temp2, 1.5)
        S_sector1 = temp3 /2 -c*x2
        S_sector2 = temp4 /2 -c*x
        P_sector_x_1 = -temp5/ 3 - c * pow(x2, 2)/2
        P_sector_x_2 = -temp6/ 3 - c * pow(x, 2)/2
        P_sector_y_1 = (-c*temp3-pow(x2,3)/3+x2*(r**2+c**2))/2
        P_sector_y_2 = (-c * temp4 - pow(x, 3) / 3 + x * (r ** 2 + c ** 2))/2
        #P_sector_y_1 = (-self.R * temp3/2 - pow(x2, 3) / 3+5*pow(self.R,2)*x2/4)/2
        #P_sector_y_2 = (-self.R * temp4/2 - pow(x, 3) / 3+5*pow(self.R,2)*x/4)/2
        if x2 > x:
            S_sector = S_sector1 - S_sector2
            P_sector_x = P_sector_x_1 - P_sector_x_2
            P_sector_y = P_sector_y_1 - P_sector_y_2
        else:
            S_sector = S_sector2 - S_sector1
            P_sector_x = P_sector_x_2 - P_sector_x_1
            P_sector_y = P_sector_y_2 - P_sector_y_1

        return S_sector, np.array([P_sector_x, P_sector_y])

    def find_intersection(self, number, R32, R33, P):
        if number == 0:
            g = self.g1
            x_left = self.A[0];x_right=self.C[0]
            y_up = self.R1*(1-cos(self.omega1/2))
            r = self.R1; c = self.R1*cos(self.omega1/2)
            S_section = self.S_sector_1; SQ_section = np.array([0, self.SQ_sector_1])
        elif number == 1:
            g = self.g2
            x_left = -self.R2*sin(self.omega2/2);x_right = -x_left
            y_up = self.R2 * (1 - cos(self.omega2 / 2))
            r = self.R2;c = self.R2*cos(self.omega2/2)
            S_section = self.S_sector_2; SQ_section = np.array([0, self.SQ_sector_2])
        elif number == 2:
            g = self.g3
            x_left = -self.R2 * sin(self.omega2 / 2); x_right = -x_left
            y_up = self.R2 * (1 - cos(self.omega2 / 2))
            r = self.R2;c = self.R2*cos(self.omega2/2)
            S_section = self.S_sector_2; SQ_section = np.array([0, self.SQ_sector_2])
        Sim = 0.; SimQ = np.zeros((2))
        r11 = g[0, 0]; r12 = g[0, 1]; r21 = g[1, 0]; r22 = g[1, 1]; Px = g[0, 2]; Py = g[1, 2]
        R32_ = R32*r11 + R33*r21; R33_ = R32*r12 + R33*r22; P_ = P+R32*Px + R33*Py
        if abs(R32_)>1E-10:
           a = pow(R33_,2)/pow(R32_, 2)+1; b = 2*R33_*P_/pow(R32_, 2) + 2*c
           cc = pow(P_, 2)/pow(R32_, 2) + pow(c, 2)-pow(r, 2)
           delta = pow(b, 2)-4*a*cc
           if delta > 0.:
                list_xy = []
                for i in range(2):
                    y = (-b + pow(-1, i)*sqrt(delta))/(2*a)
                    x = -R33_ * y / R32_ - P_ / R32_
                    if y >= 0. and x_left < x < x_right:
                        list_xy.append(np.array([x, y]))
                #### only 1 intersection
                if len(list_xy) == 1:
                    y0 = 0.
                    x0 = - P_ / R32_
                    x = list_xy[0][0]
                    y = list_xy[0][1]
                    if R33_ != 0:
                        k = -R32_ / R33_
                        if k > 0:
                            Sim1,SimQ1 = self.get_centerpoint(np.array([[x,y],[x0,y0],[x,0]]))
                            Sim2,SimQ2 = self.calculat_S_SP(x,x_right,number)
                            Sim = Sim1+Sim2; SimQ = SimQ1+SimQ2
                            if R32_*x_right+P_>0:
                                Sim = S_section-Sim; SimQ = SQ_section -SimQ
                        else:
                            Sim1, SimQ1 = self.get_centerpoint(np.array([[x, y], [x0, y0], [x, 0]]))
                            Sim2, SimQ2 = self.calculat_S_SP(x, x_left,number)
                            Sim = Sim1 + Sim2;
                            SimQ = SimQ1 + SimQ2
                            if R32_ * x_left + P_ > 0:
                                Sim = S_section - Sim; SimQ = SQ_section - SimQ
                    else:
                        if R32_ * x_right + P_ < 0:
                             Sim, SimQ = self.calculat_S_SP(x, x_right,number)
                        else:
                            Sim, SimQ = self.calculat_S_SP(x, x_left,number)

                ###### 2 intersection
                elif len(list_xy) == 2:
                    Sim, SimQ = self.calculat_S_SP(x=list_xy[1][0], x2=list_xy[0][0],number = number)
                    list_polygon = np.array([[list_xy[0][0], list_xy[0][1]],[list_xy[1][0], list_xy[1][1]], [list_xy[1][0], 0], [list_xy[0][0], 0]])
                    Sim_, SimQ_ = self.get_centerpoint(list_polygon)
                    Sim -= Sim_; SimQ -= SimQ_
                    x_ = (list_xy[1][0]+list_xy[0][0])/2; y_ = sqrt(r**2-x_**2)-c
                    if R32_*x_+R33_*y_+P_>0:
                        Sim = S_section-Sim; SimQ = SQ_section-SimQ
                else:
                    if P_ < 0:  # origine immersed
                        Sim = S_section;
                        SimQ = SQ_section
           else:
               if P_ < 0: #origine immersed
                   Sim = S_section;SimQ = SQ_section
        else:
            ###### That mean there are 2 intersections
            y = -P_/R33_
            if 0 < y < y_up:
                list_xy = []
                temp = pow(r, 2)-pow((y+c),2)
                for i in range(2):
                    x = pow(-1, i) * sqrt(temp)
                    list_xy.append(np.array([x, y]))
                Sim, SimQ = self.calculat_S_SP(x=list_xy[1][0], x2=list_xy[0][0],number = number)
                Sim_ = 2 * list_xy[0][0] * y
                SimQ_ = np.array([0, y / 2]) * Sim_
                Sim =Sim- Sim_; SimQ = SimQ - SimQ_
                if R33_*y_up+P_ > 0:
                    Sim = S_section-Sim; SimQ = SQ_section-SimQ

            else:
                    if P_<=0 and R33_*y_up+P_<=0:
                        Sim = S_section ; SimQ = SQ_section

        SimQ = np.dot(g, np.array([SimQ[0], SimQ[1], Sim]))
        SimQ = SimQ[:2]
        return Sim,SimQ

    def get_centerpoint(self, lis):
        area = 0.0
        x, y = 0.0, 0.0
        ##a = len(lis)
        a = lis.shape[0]
        for i in range(a):
            lat = lis[i, 0]  # weidu
            lng = lis[i, 1]  # jingdu
            if i == 0:
                lat1 = lis[-1, 0]
                lng1 = lis[-1, 1]
            else:
                lat1 = lis[i - 1][0]
                lng1 = lis[i - 1][1]
            fg = (lat * lng1 - lng * lat1) / 2.0
            area += fg
            x += fg * (lat + lat1) / 3.0
            y += fg * (lng + lng1) / 3.0
        x = x/area
        y = y/area
        area = abs(area)
        return area, np.array([x, y])*area

if __name__ == '__main__':
    surf = surface(1)
    print(surf.section)
    t = np.linspace(-pi, pi,500)
    Sim = []; SimQ1 = [];SimQ2 = []
    for i in range(500):
        print('it i',i)
        g = np.eye(4)
        g[2,3] = 0.0
        g[1,1] = cos(t[i])
        g[1, 2] = -sin(t[i])
        g[2,1] = sin(t[i])
        g[2, 2] = cos(t[i])
        sim,simq = surf.Immersed_surface_theoric(g)
        Sim.append(sim)
        SimQ1.append(simq[0])
        SimQ2.append(simq[1])
    plt.figure()
    plt.plot(Sim,label = 'Sim')
    #plt.plot(SimQ2,label = 'SimQ2')
    plt.legend()
    plt.figure()
    plt.plot(SimQ2, label='SimQ2')
    plt.legend()
    plt.figure()
    plt.plot(SimQ1, label='SimQ1')
    plt.legend()
    plt.figure()
    xy = surf.contour_surface()
    plt.plot(xy[0,:],xy[1,:])
    R33 = cos(t[226])
    R32 = sin(t[226])
    P3 =0.04
    xx = np.linspace(-0.07,0.07,1000);yy=np.array([])
    for i in range(xx.shape[0]):
        yy = np.append(yy,(-R32*xx[i]-P3)/R33)
    #plt.plot(xx,yy)
    plt.scatter(surf.A[0],surf.A[1])
    print(surf.A)
    print(surf.B)
    print(surf.C)
    print(surf.D)
    print(surf.E)
    print(surf.F)
    print(surf.G)
    print(surf.H)
    plt.scatter(surf.B[0],surf.B[1])
    plt.scatter(surf.C[0],surf.C[1])
    plt.scatter(surf.D[0],surf.D[1])
    plt.scatter(surf.E[0],surf.E[1])
    plt.scatter(surf.F[0],surf.F[1])
    plt.scatter(surf.G[0],surf.G[1])
    plt.scatter(surf.H[0],surf.H[1])
    plt.Circle([0,0], radius=0.04)
    plt.axis('equal')
    plt.show()