import numpy as np
import math
import matplotlib.pyplot as plt
import time
# import nprofile


class Ginger_RNN_inverse():
    def __init__(self):
        #定义基本参量
        self.unit=10**-3#位置数量级
        self.bodyLen = [i * self.unit for i in [160, 1370, 120]]#参数描述：肩宽,肩到腰的高度，脖子高度
        self.arm = [270, 245, 100]
        #算法基本参数
        self.dt=12*10**-4 #迭代步长，相当于1ms，1000HZ
        self.epsilon=self.dt
        self.apha = 50
        self.beta = 10
        self.gama = 5

        self.lamda1 = np.zeros((6,1))
        self.dlamda1 = np.zeros((6,1))

        self.th_R = np.zeros((7, 1))
        self.th_L = np.zeros((7, 1))
        self.th = np.vstack((self.th_R,self.th_L))

        self.vth_R = np.zeros((7, 1))
        self.vth_L = np.zeros((7, 1))
        self.vth = np.vstack((self.vth_R,self.vth_L))

        #关节限位
        self.th_min_R = np.array([-90,-135,-90,-95,-90,-30,-30])*np.pi/180
        self.th_max_R = np.array([ 90,   8, 90,  0, 90, 30, 30])*np.pi/180
        self.th_min_L = np.array([-90, - 8,-90,-95,-90,-30,-30])*np.pi/180
        self.th_max_L = np.array([ 90, 135, 90,  0, 90, 30, 30])*np.pi/180
        self.th_min = np.concatenate([self.th_min_R,self.th_min_L])
        self.th_max = np.concatenate([self.th_max_R,self.th_max_L])
        #速度限位
        self.dth_min = np.array([-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2])
        self.dth_max = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2])

        self.num = 1
        self.i=1
        self.freq=200

        #仿人参数
        self.th_comfort=np.array([0,-50,0,-90,0,0,0,0,50,0,-90,0,0,0])*np.pi/180
        # self.th_comfort=np.array([0,0,0,-90,-90,0,0,0,0,0,-90,90,0,0])*np.pi/180
        self.th_comfort = self.th_comfort.reshape(14, 1)
        self.delta_theta = np.array(self.th_max) - np.array(self.th_min)
        #调节系数
        self.k = np.eye(14)
        self.scale_factors = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        self.k *= self.scale_factors[:, np.newaxis]

        # 创建 14x14 零矩阵
        self.A = np.zeros((14, 14))

        # 假设 delta_th 是已定义的数组（长度为 14），k 是之前定义的矩阵
        # 计算对角线元素：A(e,e) = k(e,e) / (delta_th(e) * delta_th(e))
        np.fill_diagonal(self.A, np.diag(self.k) / (self.delta_theta ** 2))
        # np.size(np.array(self.A))


        #雅可比矩阵参数
        #R右臂参数
        self.th0_R =np.array([0, 90, 90, 90, 0, 90, 0, 0])*np.pi/180
        self.th0_R= self.th0_R.reshape(8,1)     

        self.rot0_R = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
        self.pos0_R = np.array([[0],[-self.bodyLen[0]],[self.bodyLen[1]]])
        self.TB0_3x4_R = np.hstack((self.rot0_R, self.pos0_R))
        self.TB0_R = np.vstack((self.TB0_3x4_R, np.array([[0, 0, 0, 1]])))

        self.a_R = np.array([0,0,0,0,0,0,0,-self.arm[2]])*self.unit
        self.alpha_R =np.array([0, -90, 90, -90, 90, -90, 90, 0])*np.pi/180
        self.d_R = np.array([0, 0, self.arm[0], 0, self.arm[1], 0, 0, 0])*self.unit


        #L左臂参数
        self.th0_L =np.array([0, 90, 90, 90, 0, 90, 0, 0])*np.pi/180
        self.th0_L= self.th0_L.reshape(8,1)

        self.rot0_L = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
        self.pos0_L = np.array([[0],[self.bodyLen[0]],[self.bodyLen[1]]])
        self.TB0_3x4_L = np.hstack((self.rot0_L, self.pos0_L))
        self.TB0_L = np.vstack((self.TB0_3x4_L, np.array([[0, 0, 0, 1]])))

        self.a_L = np.array([0,0,0,0,0,0,0,-self.arm[2]])*self.unit
        self.alpha_L = np.array([0, -90, 90, -90, 90, -90, 90, 0])*np.pi/180
        self.d_L = np.array([0, 0, self.arm[0], 0, self.arm[1], 0, 0, 0])*self.unit

     
    def Transform_Mdh(self,a,alpha,d,th):
        #alpha = alpha*np.pi/180
        T = np.vstack(( 
            np.hstack((np.cos(th), -np.sin(th), 0, a.item())),  
            np.hstack((np.sin(th) * np.cos(alpha), np.cos(th) * np.cos(alpha), -np.sin(alpha), -d.item() * np.sin(alpha))),  
            np.hstack((np.sin(th) * np.sin(alpha), np.cos(th) * np.sin(alpha), np.cos(alpha), d.item() * np.cos(alpha))),  
            np.hstack((0, 0, 0, 1))  
            ))

        return T
    def Jacobian_R(self,th):
        th = np.vstack((th, np.zeros((1, 1)))) + self.th0_R
        T01=self.Transform_Mdh(self.a_R[0], self.alpha_R[0], self.d_R[0], th[0])
        T12=self.Transform_Mdh(self.a_R[1], self.alpha_R[1], self.d_R[1], th[1])
        T23=self.Transform_Mdh(self.a_R[2], self.alpha_R[2], self.d_R[2], th[2])
        T34=self.Transform_Mdh(self.a_R[3], self.alpha_R[3], self.d_R[3], th[3])
        T45=self.Transform_Mdh(self.a_R[4], self.alpha_R[4], self.d_R[4], th[4])
        T56=self.Transform_Mdh(self.a_R[5], self.alpha_R[5], self.d_R[5], th[5])
        T67=self.Transform_Mdh(self.a_R[6], self.alpha_R[6], self.d_R[6], th[6])
        T78=self.Transform_Mdh(self.a_R[7], self.alpha_R[7], self.d_R[7], th[7])

        TB1 = np.matmul(self.TB0_R , T01)
        TB8_R,TB5_R,TB3_R,TB1_R,Jac_R,Jjac_R=self.Jacobian(TB1,T12,T23,T34,T45,T56,T67,T78)
        return TB8_R,TB5_R,TB3_R,TB1_R,Jac_R,Jjac_R
    
    def Jacobian_L(self,th):
        th = np.vstack((th, np.zeros((1, 1)))) + self.th0_L
        T01=self.Transform_Mdh(self.a_L[0], self.alpha_L[0], self.d_L[0], th[0])
        T12=self.Transform_Mdh(self.a_L[1], self.alpha_L[1], self.d_L[1], th[1])
        T23=self.Transform_Mdh(self.a_L[2], self.alpha_L[2], self.d_L[2], th[2])
        T34=self.Transform_Mdh(self.a_L[3], self.alpha_L[3], self.d_L[3], th[3])
        T45=self.Transform_Mdh(self.a_L[4], self.alpha_L[4], self.d_L[4], th[4])
        T56=self.Transform_Mdh(self.a_L[5], self.alpha_L[5], self.d_L[5], th[5])
        T67=self.Transform_Mdh(self.a_L[6], self.alpha_L[6], self.d_L[6], th[6])
        T78=self.Transform_Mdh(self.a_L[7], self.alpha_L[7], self.d_L[7], th[7])

        TB1 = np.matmul(self.TB0_L,T01)
        TB8_L,TB5_L,TB3_L,TB1_L,Jac_L,Jjac_L=self.Jacobian(TB1,T12,T23,T34,T45,T56,T67,T78)
        return TB8_L,TB5_L,TB3_L,TB1_L,Jac_L,Jjac_L
          
    def Jacobian(self,TB1,T12,T23,T34,T45,T56,T67,T78):
        TB2 = np.matmul(TB1 , T12)
        TB3 = np.matmul(TB2 , T23)
        TB4 = np.matmul(TB3 , T34)
        TB5 = np.matmul(TB4 , T45)
        TB6 = np.matmul(TB5 , T56)
        TB7 = np.matmul(TB6 , T67)
        TB8 = np.matmul(TB7 , T78)

        ZB1 = TB1[0 : 3, 2]  # 转换为列向量
        ZB2 = TB2[0 : 3, 2]
        ZB3 = TB3[0 : 3, 2]
        ZB4 = TB4[0 : 3, 2]
        ZB5 = TB5[0 : 3, 2]
        ZB6 = TB6[0 : 3, 2]
        ZB7 = TB7[0 : 3, 2]

        P18 = TB8[0 : 3, 3] - TB1[0 : 3, 3]
        P28 = TB8[0 : 3, 3] - TB2[0 : 3, 3]
        P38 = TB8[0 : 3, 3] - TB3[0 : 3, 3]
        P48 = TB8[0 : 3, 3] - TB4[0 : 3, 3]
        P58 = TB8[0 : 3, 3] - TB5[0 : 3, 3]
        P68 = TB8[0 : 3, 3] - TB6[0 : 3, 3]
        P78 = TB8[0 : 3, 3] - TB7[0 : 3, 3]

    # 带位姿雅可比 grasp
        Jj1 = np.hstack((np.cross(ZB1, P18), ZB1))
        Jj2 = np.hstack((np.cross(ZB2, P28), ZB2))
        Jj3 = np.hstack((np.cross(ZB3, P38), ZB3))
        Jj4 = np.hstack((np.cross(ZB4, P48), ZB4))
        Jj5 = np.hstack((np.cross(ZB5, P58), ZB5))
        Jj6 = np.hstack((np.cross(ZB6, P68), ZB6))
        Jj7 = np.hstack((np.cross(ZB7, P78), ZB7))
        Jjac = np.vstack((Jj1, Jj2, Jj3, Jj4, Jj5, Jj6, Jj7))
        Jjac = np.transpose(Jjac)
        
        
    # 不带位姿雅可比 reach
        J1 = np.array(np.cross(ZB1, P18))
        J2 = np.array(np.cross(ZB2, P28))
        J3 = np.array(np.cross(ZB3, P38))
        J4 = np.array(np.cross(ZB4, P48))
        J5 = np.array(np.cross(ZB5, P58))
        J6 = np.array(np.cross(ZB6, P68))
        J7 = np.array(np.cross(ZB7, P78))
        Jac = np.vstack((J1, J2, J3, J4, J5, J6, J7))
        Jac=np.transpose(Jac)

        return TB8,TB5,TB3,TB1,Jac,Jjac


    def Inverse(self,t_end,pos_R,pos_L,dxp_R,dxp_L,th_R,th_L):
        n=1
        self.pos_R = np.zeros((3,1))
        self.pos_L = np.zeros((3,1))
        self.dxp_R = np.zeros((3,1))
        self.dxp_L = np.zeros((3,1))
        self.th_R = np.asarray(th_R)
        self.th_L = np.asarray(th_L)
        self.th = np.vstack((self.th_R,self.th_L))
        self.vth = np.vstack((self.vth_R,self.vth_L))
        t_values = np.arange(0, t_end, self.dt)

        for tm in t_values:
            self.pos_R = pos_R*self.unit
            self.pos_L = pos_L*self.unit
            self.dxp_R = dxp_R*self.unit
            self.dxp_L = dxp_L*self.unit
            self.xp = np.vstack((self.pos_R,self.pos_L))#期望位置
            self.dxp = np.vstack((self.dxp_R,self.dxp_L))#期望速度
            [TB8_R,TB5_R,TB3_R,TB1_R,Jac_R,Jjac_R] = self.Jacobian_R(th_R)
            [TB8_L,TB5_L,TB3_L,TB1_L,Jac_L,Jjac_L] = self.Jacobian_L(th_L)

            self.xt =np.hstack((TB8_R[0:3,3],TB8_L[0:3,3]))#实际位置
            self.xt = self.xt.reshape(6,1)    
            self.jac = np.vstack((np.hstack((Jac_R,np.zeros((3,7)))),np.hstack((np.zeros((3,7)),Jac_L))))
            self.dxt = np.matmul(self.jac,self.vth)#实际速度
            self.dvth1 = np.matmul(np.transpose(self.jac),self.lamda1)
            self.dvth = self.dvth1

            for j in range(0,len(self.dvth)):
                upper_bound = self.dth_max[j]
                lower_bound = self.dth_min[j]
                self.dvth[j] = np.clip(self.dvth[j], lower_bound, upper_bound)

            self.ath = (self.dvth-self.vth)/self.epsilon
            self.dlamda1 = ((self.dxp-self.dxt)+self.gama*(self.xp-self.xt))/self.epsilon
            #更新变量
            self.th = self.th + self.vth*self.dt
            self.vth = self.vth + self.ath*self.dt
            self.lamda1 = self.lamda1 + self.dlamda1*self.dt
            th_R = self.th[0:7]
            th_L = self.th[7:14]
            n += 1

        return self.xt,self.xp ,self.jac
    def Inverse_with_time(self, pos_R, pos_L, dxp_R, dxp_L, th_R, th_L, iterations=1000):
        self.pos_R = pos_R * self.unit
        self.pos_L = pos_L * self.unit
        self.dxp_R = dxp_R * self.unit
        self.dxp_L = dxp_L * self.unit
        self.xp = np.vstack((self.pos_R, self.pos_L))
        self.dxp = np.vstack((self.dxp_R, self.dxp_L))
        
        for _ in range(iterations):
            [TB8_R, TB5_R, TB3_R, TB1_R, Jac_R, Jjac_R] = self.Jacobian_R(th_R)
            [TB8_L, TB5_L, TB3_L, TB1_L, Jac_L, Jjac_L] = self.Jacobian_L(th_L)

            self.xt = np.hstack((TB8_R[0:3, 3], TB8_L[0:3, 3]))
            self.xt = self.xt.reshape(6, 1)
            self.jac = np.vstack((np.hstack((Jac_R, np.zeros((3, 7)))), np.hstack((np.zeros((3, 7)), Jac_L))))
            self.dxt = np.matmul(self.jac, self.vth)
            self.dvth1 = np.matmul(np.transpose(self.jac), self.lamda1)
            dH = self.A @ (self.th - self.th_comfort)
            # print("dH: ----------",dH.shape)
            self.dvth = self.dvth1 - dH
            # 关节速度限位
            for j in range(0, len(self.dvth)):
                upper_bound = min(self.dth_max[j],self.apha * (self.th_max[j] - self.th[j])) 
                lower_bound = max(self.apha * (self.th_min[j] - self.th[j]),self.dth_min[j])
                self.dvth[j] = np.clip(self.dvth[j], lower_bound, upper_bound)
            # 速度限位
            # for j in range(0, len(self.dvth)):
            #     upper_bound = self.dth_max[j]
            #     lower_bound = self.dth_min[j]
            #     self.dvth[j] = np.clip(self.dvth[j], lower_bound, upper_bound)


            self.ath = (self.dvth - self.vth) / self.epsilon
            self.dlamda1 = ((self.dxp - self.dxt) + self.gama * (self.xp - self.xt)) / self.epsilon
            self.th = self.th + self.vth * self.dt
            self.vth = self.vth + self.ath * self.dt
            self.lamda1 = self.lamda1 + self.dlamda1 * self.dt
            th_R = self.th[0:7]
            th_L = self.th[7:14]

            yield self.th, self.xt  # 使用 yield 返回每次迭代的结果







if __name__ == '__main__':

    RNN=Ginger_RNN_inverse()
    t_end = 2
    #t_end = 0.1
    pos_R = np.array([[345],[0],[1100]])
    pos_L = np.array([[345],[160],[1100]])
    dxp_R = np.zeros((3,1))
    dxp_L = np.zeros((3,1))
    th_R  = np.zeros((7,1))
    th_L  = np.zeros((7,1))
    start_time = time.time()
    xt,xp,jac = RNN.Inverse(t_end,pos_R,pos_L,dxp_R,dxp_L,th_R,th_L)
    end_time = time.time()
    print(f"xt\n{xt}")
    print(f"程序时间\n{round(end_time-start_time,4)}")
