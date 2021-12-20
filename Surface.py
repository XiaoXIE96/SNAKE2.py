import numpy as np
from math import sqrt, acos, pi,asin
import matplotlib.pyplot as plt
import time
class surface(object):
    def __init__(self, r, a):
        self.r = r
        self.a = a
        #self.g = g

        ################!!!!!!!!!!!!!!!!!#####################
        sq3 = sqrt(3)
        self.R = r/sq3
        '''
        self.g1= np.array([[1,0,0],
                       [0,1,a/sqrt(3)-sq3*self.r/2],
                       [0,0,1]])
        self.g2 = np.array([[-1 / 2, -sqrt(3) / 2, -self.a / 2 + 3*self.r/4],
                            [sqrt(3) / 2, -1 / 2, -self.a * sq3 / 6 +sq3*self.r/4],
                            [0, 0, 1]])
        self.g3 = np.array([[-1 / 2, sqrt(3) / 2, self.a / 2 - 3*self.r/4],
                            [-sqrt(3) / 2, -1 / 2,-self.a * sq3 / 6 + sq3*self.r/4],
                            [0, 0, 1]])
        ################!!!!!!!!!!!!!!!!!#####################
        '''
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
        x_y = np.array([-self.a/2+self.r/2,-sqrt(3)*self.a/6+sqrt(3)*self.r/2])
        for i in range(3):
            x_cercle1 = np.linspace(-self.r/2,self.r/2,20)
            y_cercle1 = np.empty((x_cercle1.shape))
            if i == 0:
                g = self.g1
            if i == 1:
                g = self.g3
            if i == 2:
                g = self.g2
            for i in range(x_cercle1.shape[0]):
                y_cercle1[i] = sqrt(self.R**2-x_cercle1[i]**2)-self.R/2
                x_y_new = np.dot(g, np.array([x_cercle1[i], y_cercle1[i],1]))
                x_y = np.c_[x_y,x_y_new[:2]]
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

    def barycenter_irregular_shape(self, nb):
        if nb == 0:
            g = self.g1
        if nb == 1:
            g = self.g2
        if nb == 2:
            g = self.g3
        '''
        S_secteur = pi / 6 * self.r ** 2 ## secteur circulaire
        S_cutted = (sqrt(3)/2-pi/6)*self.r**2
        P_secteur = S_secteur * (-sqrt(3)/2 + 2/pi)*self.r
        P_cutted = -P_secteur/S_cutted
        '''
        P_cutted = np.dot(g,np.array([0, self.P_cutted*self.S_cutted, self.S_cutted]))
        return self.S_cutted, P_cutted[:2]

    def Immersed_surface_theoric(self,g):
        R33 = g[2, 2]
        R32 = g[2, 1]
        P3 = g[2, 3]
        '''
        list_sec = np.array([[-r/2, a/sqrt(3)-sqrt(3)/2*r],
                             [r/2, a/sqrt(3)-sqrt(3)/2*r],
                             [(-a+r)/2, -sqrt(3)*a/6+sqrt(3)*r/2],
                             [(-a+2*r)/2, -sqrt(3)*a/6],
                             [(a-r)/2, -sqrt(3)*a/6+sqrt(3)*r/2],
                             [(a-2*r)/2, -sqrt(3)*a/6]])
        '''
        ############################################################
        ### ici c'est pour calculer les intersecteurs entre la section et la droite
        ###########################################################
        list_extreme = np.array([[0, self.a/sqrt(3)],
                             [-self.a/2, -sqrt(3)*self.a/6],
                             [self.a/2, -sqrt(3)*self.a/6]])
        nb = []
        nb_complet = [0, 1, 2]
        for i in range(3):
            x = list_extreme[i, 0]
            y = list_extreme[i, 1]
            if R32 * x + R33 * y + P3 <= 0:
                nb.append(i)
        if len(nb) == 3:
            area = self.section
            center = np.array([0, 0])
        elif len(nb) == 0:
            area = 0.
            center = np.array([0, 0])
        else:
            list_point = np.array([])
            o1 = (P3 + self.a * R33/sqrt(3))
            o2 = (-R32 - sqrt(3) * R33)
            o3 = (-R32 + sqrt(3) * R33)
            if o2 != 0:
                x1 = o1 / o2
                #print((self.r-self.a)/2,'x1',x1,-self.r/2)
                if (self.r-self.a)/2 <= x1 <= -self.r/2:
                    y1 = sqrt(3)*x1+self.a/sqrt(3)
                    list_point = np.append(list_point, np.array([x1, y1]))
            if o3 != 0:
                x2 = o1 / o3
                #print(self.r/2,'x2', x2,(self.a-self.r)/2)
                if self.r/2 <= x2 <= (self.a-self.r)/2:
                    y2 = -sqrt(3)*x2+self.a/sqrt(3)
                    list_point = np.append(list_point, np.array([x2, y2]))
            if R32 != 0:
                x3 = (sqrt(3)/6*R33*self.a - P3) / R32
                #print((-self.a/2+self.r),'x3', x3,self.a/2-self.r)
                if (-self.a/2+self.r) <= x3 <= (self.a/2-self.r):
                    y3 = -sqrt(3) / 6 * self.a
                    list_point = np.append(list_point, np.array([x3, y3]))
            ################################################################################

            #######################################################
            ####### 2 intersecteurs
            ########################################################
            if list_point.shape[0] == 4:
                # the polygon is immersed
                if len(nb) == 2:
                    for i in nb:
                        nb_complet.remove(i)
                    point_extreme = list_extreme[nb_complet[0], :]
                    list_point = np.append(list_point,point_extreme)
                    area1, center1 = self.get_centerpoint(np.reshape(list_point, (-1, 2)))
                    area2, center2 = self.barycenter_irregular_shape(nb_complet[0])
                    area3 = area1-area2
                    center3 = center1 - center2
                    area = self.section - area3
                    center = -center3
                # the triangular is immersed
                elif len(nb) == 1:
                    point_extreme = list_extreme[nb[0], :]
                    list_point = np.append(list_point, point_extreme)
                    area1, center1 = self.get_centerpoint(np.reshape(list_point, (-1, 2)))
                    area2, center2 = self.barycenter_irregular_shape(nb[0])
                    area = area1 - area2
                    center = center1 - center2
                else:
                    AssertionError('There is something wrong...')
            elif list_point.shape[0] == 2:

                # the polygon is immersed
                if len(nb) == 2:
                    for i in nb:
                        nb_complet.remove(i)

                    point_extreme = list_extreme[nb_complet[0], :]
                    i = -1
                    Intersection = []
                    while True:
                        i += 1
                        Intersection_i = self.find_intersection(i, R32, R33, P3)
                        if len(Intersection_i) != 0:
                            Intersection.append(Intersection_i)
                        if len(Intersection) != 0 or i >= 2:
                            break
                    if i == nb_complet[0]: #the angle non immersed is the angle whose shoulder passed by the ligne
                        area3 = 0; center3 = np.zeros((2))
                        for j in nb:
                            temp_1,temp_2 = self.barycenter_irregular_shape(j)
                            area3 += temp_1; center3 += temp_2
                    else:
                        nb.remove(i)

                        area3, center3 = self.barycenter_irregular_shape(nb[0])
                    area2 = Intersection[0][0][0]
                    center2 = Intersection[0][0][1]
                    point2 = Intersection[0][0][2:]
                    list_point = np.append(list_point, point2)
                    list_point = np.append(list_point, point_extreme)
                    area1, center1 = self.get_centerpoint(np.reshape(list_point, (-1, 2)))
                    area = self.S_triang-area1 - area2 - area3
                    center = - center1 - center2 - center3
                ##### the triangular part is immersed
                elif len(nb) == 1:
                    point_extreme = list_extreme[nb[0], :]
                    i = -1
                    Intersection = []
                    while True:
                        i += 1
                        Intersection_i = self.find_intersection(i, R32, R33, P3)
                        if len(Intersection_i) != 0:
                            Intersection.append(Intersection_i)
                        if len(Intersection) != 0 or i >= 2:
                            break
                    area2 = Intersection[0][0][0]
                    center2 = Intersection[0][0][1]

                    list_point = np.append(list_point, Intersection[0][0][2:])
                    list_point = np.append(list_point, point_extreme)

                    area1, center1 = self.get_centerpoint(np.reshape(list_point, (-1, 2)))
                    if i!=nb[0]:
                        area3, center3 = self.barycenter_irregular_shape(nb[0])
                        area = area1 - area2 - area3
                        center =  center1 - center2 - center3
                    else:
                        area = area1 - area2
                        center = center1 - center2
                else:
                    AssertionError('There is something wrong...')
            else:
                i = -1
                Intersection = []
                nb_3_point = [0, 1, 2]
                while True:
                    i += 1
                    Intersection_i = self.find_intersection(i, R32, R33, P3)
                    if len(Intersection_i) != 0:
                        nb_3_point.remove(i)
                        Intersection.append(Intersection_i)
                    if len(Intersection) >= 2 or i >= 2:
                        break
                # 2 intersections on 2 different shoulders
                if len(Intersection) == 2:
                    if len(nb) == 2:
                        for j in nb:
                            nb_complet.remove(j)
                        point_extreme = list_extreme[nb_complet[0], :]
                        area2 = Intersection[0][0][0]
                        center2 = Intersection[0][0][1]
                        point2 = Intersection[0][0][2:]
                        area3 = Intersection[1][0][0]
                        center3 = Intersection[1][0][1]
                        point3 = Intersection[1][0][2:]
                        list_point = np.append(list_point, point2)
                        list_point = np.append(list_point, point3)
                        list_point = np.append(list_point, point_extreme)

                        area1, center1 = self.get_centerpoint(np.reshape(list_point, (-1, 2)))
                        '''
                        nb_3_point is the angle whose shoulder is not passed
                        nb_complet is the angle which is on water
                        '''
                        if nb_3_point[0] == nb_complet[0]:
                            area = self.S_triang - area1 - area2 - area3
                            center = -center1 -center2 - center3
                        else:
                            area4, center4 = self.barycenter_irregular_shape(nb_3_point[0])
                            area = self.S_triang-area1-area2-area3-area4
                            center = -center1 -center2 - center3 - center4
                    elif len(nb) == 1:
                        point_extreme = list_extreme[nb[0], :]
                        area2 = Intersection[0][0][0]
                        center2 = Intersection[0][0][1]
                        point2 = Intersection[0][0][2:]
                        area3 = Intersection[1][0][0]
                        center3 = Intersection[1][0][1]
                        point3 = Intersection[1][0][2:]
                        list_point = np.append(list_point, point2)
                        list_point = np.append(list_point, point3)
                        list_point = np.append(list_point, point_extreme)
                        area1, center1 = self.get_centerpoint(np.reshape(list_point, (-1, 2)))
                        '''
                        nb is the only angle under water
                        nb_3_point is the angle whose shoulder is not passed
                        '''
                        if nb[0] == nb_3_point[0]:
                           area4, center4 = self.barycenter_irregular_shape(nb[0])
                           area = area1 - area2 - area3 - area4
                           center = center1 - center2 - center3 -center4
                        else:
                            area =  area1 - area2 - area3
                            center = center1 - center2 - center3

                    else:
                        AssertionError('There is something wrong...')
                # 2 intersections on a same shoulder
                elif len(Intersection) == 1:
                        area = Intersection[0][0][0]
                        center = Intersection[0][0][1]
                else:
                    if P3 < 0:
                        area = self.section
                        center = np.array([0,0])
                    else:
                        area = 0.
                        center = np.array([0, 0])
        return area, center

    def calculat_S_SP(self, x0 = None, x = float, x2 = float, R32 = float, R33 = float, P = float):
        temp1 = pow(self.R, 2) - pow(x2, 2);
        temp2= pow(self.R, 2) - pow(x, 2)
        temp3 = pow(self.R, 2) * asin(x2 / self.R) + x2 * sqrt(temp1) #r2 * asin(x/r) + x*sqrt(r2-x2)
        temp4 = pow(self.R, 2) * asin(x / self.R) + x * sqrt(temp2)
        temp5 = pow(temp1, 1.5); temp6 = pow(temp2, 1.5)
        S_sector1 = (temp3 - self.R * x2)/2
        S_sector2 = (temp4 - self.R * x)/2
        P_sector_x_1 = -temp5/ 3 - self.R * pow(x2, 2)/4
        P_sector_x_2 = -temp6/ 3 - self.R * pow(x, 2)/4

        P_sector_y_1 = (-self.R * temp3/2 - pow(x2, 3) / 3+5*pow(self.R,2)*x2/4)/2
        P_sector_y_2 = (-self.R * temp4/2 - pow(x, 3) / 3+5*pow(self.R,2)*x/4)/2
        if x2 > x:
            S_sector = S_sector1 - S_sector2
            P_sector_x = P_sector_x_1 - P_sector_x_2
            P_sector_y = P_sector_y_1 - P_sector_y_2
        else:
            S_sector = S_sector2 - S_sector1
            P_sector_x = P_sector_x_2 - P_sector_x_1
            P_sector_y = P_sector_y_2 - P_sector_y_1
        if x0 != None and x != x0:

            S_triang1 = (R32 * x + 2 * P) * x
            S_triang2 = (R32 * x0 + 2 * P) * x0
            P_triang_x_1 = pow(x, 2) * (2 * R32 * x + 3 * P)
            P_triang_x_2 = pow(x0, 2) * (2 * R32 * x0 + 3 * P)
            P_triang_y_1 = pow(R32 * x + P, 3)
            P_triang_y_2 = pow(R32 * x0 + P, 3)
            if x > x0:
                S_triang = (S_triang1 - S_triang2) / (-2*R33)
                P_triang_x = (P_triang_x_1 - P_triang_x_2) / (-6*R33)
                P_triang_y = (P_triang_y_1 - P_triang_y_2) / (6 * pow(R33, 2) * R32)
            else:
                S_triang = (S_triang2 - S_triang1) / (-2*R33)
                P_triang_x = (P_triang_x_2 - P_triang_x_1) / (-6*R33)
                P_triang_y = (P_triang_y_2 - P_triang_y_1) / (6 * pow(R33, 2) * R32)
            '''
            list_point = np.array([x0,0])
            list_point = np.append(list_point,np.array([x,sqrt(self.R**2-x**2)-self.R/2]))
            list_point = np.append(list_point, np.array([x, 0]))
            area, center = self.get_centerpoint(np.reshape(list_point, (-1, 2)))
            '''
            S_sector += S_triang#area
            P_sector_x += P_triang_x#center[0]
            P_sector_y += P_triang_y#center[1]
        return S_sector, np.array([P_sector_x, P_sector_y])

    def find_intersection(self, number, R32, R33, P):
        if number == 0:
            g = self.g1
        elif number == 1:
            g = self.g2
        elif number == 2:
            g = self.g3
        Solution = []
        r11 = g[0, 0]; r12 = g[0, 1]; r21 = g[1, 0]; r22 = g[1, 1]; Px = g[0, 2]; Py = g[1, 2]
        R32_ = R32*r11 + R33*r21; R33_ = R32*r12 + R33*r22; P_ = P+R32*Px + R33*Py
        if R32_!=0.:
           a = pow(R33_,2)/pow(R32_, 2)+1; b = 2*R33_*P_/pow(R32_, 2) + self.R
           c = pow(P_, 2)/pow(R32_, 2)-3*pow(self.R, 2)/4;
           delta = pow(b, 2)-4*a*c
           if delta > 0. and abs(R32_) > 1E-8:
                list_xy = []
                for i in range(2):
                    y = (-b + pow(-1, i)*sqrt(delta))/(2*a)
                    x = -R33_ * y / R32_ - P_ / R32_
                    if y >= 0. and -self.r/2 < x < self.r/2:
                        list_xy.append(np.array([x, y]))
                #### only 1 intersection
                if len(list_xy) == 1:
                    y0 = 0.
                    x0 = - P_ / R32_
                    x = list_xy[0][0]
                    y = list_xy[0][1]
                    # the condition that one intersection on right part of triangular
                    if R32_ - R33_ * sqrt(3) != 0:
                        x1 = (-P_-R33_*sqrt(3)/2*self.r)/(R32_ - R33_*sqrt(3))
                        if 0 < x1 < self.r/2:
                            y1 = sqrt(3)*(self.r/2-x1)
                            x2 = self.r/2
                            y2 = 0.
                            Sim1,SimQ1 = self.get_centerpoint(np.array([[x0, y0], [x1, y1], [x2, y2]]))
                            Sim2, SimQ2 = self.calculat_S_SP(x0, x, x2, R32_, R33_, P_)
                            '''
                            lis = np.linspace(x, self.r/2, 8)
                            list_irregular = np.array([x0, y0])
                            list_irregular = np.append(list_irregular, np.array([x, y]))
                            for i in range(1, lis.shape[0]-2):
                                yi = sqrt(pow(self.r, 2)-pow(lis[i], 2))-sqrt(3)/2*self.r
                                list_irregular = np.append(list_irregular, np.array([lis[i], yi]))
                            list_irregular = np.append(list_irregular, np.array([x2, y2]))
                            Sim2, SimQ2 = self.get_centerpoint(np.reshape(list_irregular, (-1, 2)))
                            '''
                            x1y1 = np.dot(g, np.array([x1, y1, 1]))
                            if R32_ * x2 + P_ < 0:
                                Sim = Sim1-Sim2
                                SimQ = SimQ1 - SimQ2
                                SimQ = np.dot(g, np.array([SimQ[0], SimQ[1], Sim]))
                                SimQ = SimQ[:2]
                                Solution.append(np.array([Sim, SimQ, x1y1[0], x1y1[1]]))
                            else:
                                Sim = self.S_cutted - (Sim1-Sim2)
                                SimQ = np.array([0, self.P_cutted*self.S_cutted]) - (SimQ1 - SimQ2)
                                SimQ = np.dot(g, np.array([SimQ[0], SimQ[1], Sim]))
                                SimQ = SimQ[:2]
                                Solution.append(np.array([Sim, SimQ, x1y1[0], x1y1[1]]))
                    # the condition that one intersection on left part of triangular
                    if R32_ + R33_ * sqrt(3) != 0 and len(Solution) == 0:
                        x1 = (-P_ - R33_ * sqrt(3) / 2 * self.r) / (R32_ + R33_ * sqrt(3))
                        if -self.r / 2 < x1 <= 0:
                            y1 = sqrt(3) * (self.r / 2 + x1)
                            x2 = -self.r / 2
                            y2 = 0.
                            Sim1, SimQ1 = self.get_centerpoint(np.array([[x0, y0], [x1, y1], [x2, y2]]))
                            '''
                            lis = np.linspace(-self.r / 2, x, 8)
                            list_irregular = np.array([x2, y2])
                            for i in range(1, lis.shape[0] - 1):
                                yi = sqrt(pow(self.r, 2) - pow(lis[i], 2)) - sqrt(3) / 2*self.r
                                list_irregular = np.append(list_irregular, np.array([lis[i], yi]))
                            list_irregular = np.append(list_irregular, np.array([x, y]))
                            list_irregular = np.append(list_irregular, np.array([x0, y0]))
                            Sim2, SimQ2 = self.get_centerpoint(np.reshape(list_irregular,(-1,2)))
                            '''
                            Sim2, SimQ2 = self.calculat_S_SP(x0, x, x2, R32_, R33_, P_)
                            x1y1 = np.dot(g, np.array([x1, y1, 1]))
                            if R32_*x2+P_ < 0:
                                Sim = Sim1 - Sim2
                                SimQ = SimQ1 - SimQ2
                                SimQ = np.dot(g, np.array([SimQ[0], SimQ[1], Sim]))
                                SimQ = SimQ[:2]
                                Solution.append(np.array([Sim, SimQ, x1y1[0], x1y1[1]]))
                            else:
                                Sim = self.S_cutted - (Sim1 - Sim2)
                                SimQ = np.array([0, self.P_cutted*self.S_cutted]) - (SimQ1 - SimQ2)
                                SimQ = np.dot(g, np.array([SimQ[0], SimQ[1], Sim]))
                                SimQ = SimQ[:2]
                                Solution.append(np.array([Sim, SimQ, x1y1[0], x1y1[1]]))
                ###### 2 intersection
                elif len(list_xy) == 2:
                    '''
                    lis = np.linspace(list_xy[1][0], list_xy[0][0], 8)
                    list_irregular = list_xy[1]
                    for i in range(1, lis.shape[0] - 2):
                        yi = sqrt(pow(self.r, 2) - pow(lis[i], 2)) - sqrt(3) / 2
                        list_irregular = np.append(list_irregular, np.array([lis[i], yi]))
                    list_irregular = np.append(list_irregular, list_xy[0])
                    Sim, SimQ = self.get_centerpoint(list_irregular)
                    '''
                    Sim, SimQ = self.calculat_S_SP(x=list_xy[1][0], x2=list_xy[0][0], R32=R32_, R33=R33_, P=P_)
                    list_polygon = np.array([[list_xy[0][0], list_xy[0][1]],[list_xy[1][0], list_xy[1][1]], [list_xy[1][0], 0], [list_xy[0][0], 0]])
                    Sim_, SimQ_ = self.get_centerpoint(list_polygon)
                    Sim -= Sim_; SimQ -= SimQ_
                    SimQ = np.dot(g, np.array([SimQ[0], SimQ[1], Sim]))
                    SimQ = SimQ[:2]
                    if R33_*sqrt(3)/2*self.r + P_ >= 0:
                        Sim = self.section - Sim
                        SimQ = - SimQ
                    Solution.append(np.array([Sim, SimQ]))
        else:
            ###### That mean there are 2 intersections
            y = -P_/R33_
            if 0 < y < self.R/2:
                list_xy = []
                temp = y + self.R / 2; temp2 = pow(self.R, 2) - pow((temp), 2)

                for i in range(2):
                    x = pow(-1, i) * sqrt(temp2)
                    list_xy.append(np.array([x, y]))
                if len(list_xy) == 2:
                    '''
                    lis = np.linspace(list_xy[1][0], list_xy[0][0], 8)                    
                    list_irregular = list_xy[1]
                    for i in range(1, lis.shape[0] - 2):
                        yi = sqrt(pow(self.r, 2) - pow(lis[i], 2)) - sqrt(3) / 2
                        list_irregular = np.append(list_irregular, np.array([lis[i], yi]))
                    list_irregular = np.append(list_irregular, list_xy[0])
                    Sim, SimQ = self.get_centerpoint(list_irregular)
                    '''
                    Sim, SimQ = self.calculat_S_SP(x=list_xy[1][0], x2=list_xy[0][0], R32=R32_, R33=R33_, P=P_)
                    '''
                    list_polygon = np.array(
                        [[list_xy[0][0], list_xy[0][1]], [list_xy[1][0], list_xy[1][1]], [list_xy[1][0], 0],
                         [list_xy[0][0], 0]])
                    Sim_, SimQ_ = self.get_centerpoint(list_polygon)
                    '''
                    Sim_ = 2 * list_xy[0][0] * y
                    SimQ_ = np.array([0, y / 2]) * Sim
                    Sim -= Sim_; SimQ -= SimQ_
                    SimQ = np.dot(g, np.array([SimQ[0], SimQ[1], Sim]))
                    SimQ = SimQ[:2]
                    if R33_ * self.R/2 + P_ >= 0:
                        Sim = self.section - Sim
                        SimQ = - SimQ
                    Solution.append(np.array([Sim, SimQ]))
                elif len(list_xy) == 1:
                    AssertionError('bro, wtf')
        return Solution

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
    g = np.identity(4)
    g[2, 3] = sqrt(3)/6 - 0.05*0 - sqrt(3)/4*0.2
    #g[2, 3] = 5
    surface = surface(0.2, 1, g)
    t1 = time.time()
    for i in range(1000):
        area,center = surface.Immersed_surface_theoric()
    t2 = time.time()

 #   print(-sqrt(3)/6 + sqrt(3)/4*0.2)
  #  print(surface.section)