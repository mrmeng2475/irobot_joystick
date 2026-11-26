import numpy as np
import math
import matplotlib.pyplot as plt
import time
# import nprofile


_MULTIPLIER_DAMPING_SQUARED = 0.02**2
_MULTIPLIER_RIDGE = _MULTIPLIER_DAMPING_SQUARED * np.eye(6)
_MULTIPLIER_LEAK = 0.006


class Ginger_RNN_inverse():
    def __init__(self, dt=1.2 * 10**-4):
        #定义基本参量
        self.unit=10**-3#位置数量级
        self.bodyLen = [i * self.unit for i in [160, 1370, 120]]#参数描述：肩宽,肩到腰的高度，脖子高度
        self.arm = [270, 245, 100]
        #算法基本参数
        self.dt = dt  # 数值积分步长，单位为秒
        self.epsilon = self.dt
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

     
    @staticmethod
    def _mdh_transforms(a, alpha, d, theta):
        """Build the modified DH matrices for all links in one arm."""
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)

        matrices = np.zeros((theta.size, 4, 4))
        matrices[:, 0, 0] = cos_theta
        matrices[:, 0, 1] = -sin_theta
        matrices[:, 0, 3] = a
        matrices[:, 1, 0] = sin_theta * cos_alpha
        matrices[:, 1, 1] = cos_theta * cos_alpha
        matrices[:, 1, 2] = -sin_alpha
        matrices[:, 1, 3] = -d * sin_alpha
        matrices[:, 2, 0] = sin_theta * sin_alpha
        matrices[:, 2, 1] = cos_theta * sin_alpha
        matrices[:, 2, 2] = cos_alpha
        matrices[:, 2, 3] = d * cos_alpha
        matrices[:, 3, 3] = 1
        return matrices

    def Transform_Mdh(self, a, alpha, d, th):
        theta = np.asarray(th).reshape(1)
        return self._mdh_transforms(np.asarray(a), np.asarray(alpha), np.asarray(d), theta)[0]

    def _arm_jacobian(self, th, offsets, base, a, alpha, d):
        angles = offsets[:, 0].copy()
        angles[:7] += np.asarray(th).reshape(7)
        transforms = self._mdh_transforms(a, alpha, d, angles)
        return self.Jacobian(base @ transforms[0], *transforms[1:])

    def Jacobian_R(self, th):
        return self._arm_jacobian(th, self.th0_R, self.TB0_R, self.a_R, self.alpha_R, self.d_R)

    def Jacobian_L(self, th):
        return self._arm_jacobian(th, self.th0_L, self.TB0_L, self.a_L, self.alpha_L, self.d_L)

    def Jacobian(self, TB1, T12, T23, T34, T45, T56, T67, T78):
        frames = [TB1]
        for transform in (T12, T23, T34, T45, T56, T67, T78):
            frames.append(frames[-1] @ transform)

        joint_frames = np.asarray(frames[:-1])
        axes = joint_frames[:, :3, 2]
        lever_arms = frames[-1][:3, 3] - joint_frames[:, :3, 3]
        axis_x, axis_y, axis_z = axes.T
        lever_x, lever_y, lever_z = lever_arms.T
        position_jacobian = np.array((
            axis_y * lever_z - axis_z * lever_y,
            axis_z * lever_x - axis_x * lever_z,
            axis_x * lever_y - axis_y * lever_x,
        ))
        pose_jacobian = np.concatenate((position_jacobian, axes.T))
        return frames[-1], frames[4], frames[2], frames[0], position_jacobian, pose_jacobian

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

    def _limit_multiplier_increment(self, position_error, unconstrained_velocity, increment):
        # Keep the original update around the goal and blend in anti-windup from 2 to 10 mm.
        error_by_arm = np.linalg.norm(position_error.reshape(2, 3), axis=1)
        blend = np.clip((error_by_arm - 0.002) / 0.008, 0, 1)
        if not blend.any():
            return increment

        # Remove Jacobian columns where the increment would push further into a limit.
        velocity_loss = unconstrained_velocity - self.dvth
        blocked = velocity_loss * (self.jac.T @ increment) > 1e-12
        available_jacobian = self.jac * (~blocked).T
        gram = available_jacobian @ available_jacobian.T
        correction = np.linalg.solve(gram + _MULTIPLIER_RIDGE, increment + _MULTIPLIER_LEAK * self.lamda1)
        return increment - _MULTIPLIER_DAMPING_SQUARED * np.repeat(blend, 3)[:, None] * correction

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

            self.xt = np.concatenate((TB8_R[:3, 3], TB8_L[:3, 3])).reshape(6, 1)
            self.jac = np.zeros((6, 14))
            self.jac[:3, :7] = Jac_R
            self.jac[3:, 7:] = Jac_L
            self.dxt = np.matmul(self.jac, self.vth)
            self.dvth1 = np.matmul(np.transpose(self.jac), self.lamda1)
            dH = self.A @ (self.th - self.th_comfort)
            unconstrained_velocity = self.dvth1 - dH
            # 根据当前角度收紧每个关节的速度范围。
            angle = self.th[:, 0]
            upper_bound = np.minimum(self.dth_max, self.apha * (self.th_max - angle))
            lower_bound = np.maximum(self.dth_min, self.apha * (self.th_min - angle))
            self.dvth = np.clip(unconstrained_velocity, lower_bound[:, None], upper_bound[:, None])

            position_error = self.xp - self.xt
            self.ath = (self.dvth - self.vth) / self.epsilon
            self.dlamda1 = ((self.dxp - self.dxt) + self.gama * position_error) / self.epsilon
            multiplier_increment = self._limit_multiplier_increment(
                position_error, unconstrained_velocity, self.dlamda1 * self.dt,
            )
            self.th = self.th + self.vth * self.dt
            self.vth = self.vth + self.ath * self.dt
            self.lamda1 = self.lamda1 + multiplier_increment
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
