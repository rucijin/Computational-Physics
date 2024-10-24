import numpy as np
import matplotlib.pyplot as plt

dt = 0.001             # 定义时间步长
tf = 2.0                # 模拟时间
ntime = round(tf/dt)    # 时间步数

xl = -2.0*np.pi         # 左边界
xr = 2.0*np.pi          # 右边界
nx = 1000               # 空间步长数
dx = (xr-xl)/nx         # 空间步长

# 创建双精度浮点数
x = np.arange(xl, xr + dx, dx, dtype=np.float64)
u = np.sin(x)           # 初始条件，假设是三角函数
u_2 = np.copy(u)        # 初始条件，用于RK2求解
u0 = np.copy(u)         # 记录u的初始值
c = 1.0                 # 对流速度场V
mu = 0.1               # 扩散系数

CFL = c*dt/dx       # CFL条件数
print(f"CFL number is {CFL}")
CFL2 = mu*dt/(dx**2)
print(f"mu*dt/(dx^2)={CFL2}")

def spatial_der(u,c,mu):
    ul = ur = np.zeros(np.size(u), dtype=np.float64)
    ul = np.roll(u,1)               # 数组右移
    ur = np.roll(u,-1)              # 数组左移
    dudx = (ur-ul)/(2*dx)           # 空间偏导 中心差分
    du2dx = (ur -2*u + ul)/(dx**2)  # 二阶偏导 欧拉向前差分
    return -c*dudx + mu*du2dx     # 时间偏导

# RK2
tc = 0  # 起始时间
for t in range(ntime):
    tc += dt
    if tc <= 0.5:   u_21 = np.copy(u_2)
    elif tc <= 1.0: u_22 = np.copy(u_2)
    elif tc <= 1.5: u_23 = np.copy(u_2)
    else:           u_24 = np.copy(u_2)
    k1 = spatial_der(u_2,         c, mu)
    k2 = spatial_der(u_2+dt*k1, c, mu)
    u_2 += dt*(k1+k2)/2

# RK4
tc = 0  # 起始时间
for t in range(ntime):
    tc += dt
    if tc <= 0.5:   u1 = np.copy(u)
    elif tc <= 1.0: u2 = np.copy(u)
    elif tc <= 1.5: u3 = np.copy(u)
    else:           u4 = np.copy(u)
    k1 = spatial_der(u,             c, mu)
    k2 = spatial_der(u+dt*1/2*k1,   c, mu)
    k3 = spatial_der(u+dt*1/2*k2,   c, mu)
    k4 = spatial_der(u+dt*1.0*k3,   c, mu)
    u += dt*(k1+2*k2+2*k3+k4)/6

# for i in range(0,2):
#     i = i+1
#     if i == 1:
#         tm = 'RK2'
#         for j in range(4):
#             j = j+1
#             if j == 1:
#                 utemp = u_21
#                 t = '0.5s'
                
#             elif j == 2:
#                 utemp = u_22
#                 t = '1.0s'

#             elif j == 3:
#                 utemp = u_23
#                 t = '1.5s'

#             elif j == 4:
#                 utemp = u_24
#                 t = '2.0s'
#             plt.close()
#             plt.plot(x,u0,label='t=0s')
#             plt.plot(x,utemp,label='t='+t)
#             plt.xlabel("x")
#             plt.ylabel("u(x)")
#             plt.legend()
#             plt.title(f'{tm} $\mu$={mu} Total time $t={tc}$')
#             plt.tight_layout()
#             plt.savefig(f'{tm}-{str(mu)}-{t}.svg')

#     elif i == 2:
#         tm = 'RK4'
#         for j in range(4):
#             j = j+1
#             if j == 1:
#                 utemp = u1
#                 t = '0.5s'
                
#             elif j == 2:
#                 utemp = u2
#                 t = '1.0s'

#             elif j == 3:
#                 utemp = u3
#                 t = '1.5s'

#             elif j == 4:
#                 utemp = u4
#                 t = '2.0s'
#             plt.close()
#             plt.plot(x,u0,label='t=0s')
#             plt.plot(x,utemp,label='t='+t)
#             plt.xlabel("x")
#             plt.ylabel("u(x)")
#             plt.legend()
#             plt.title(f'{tm} $\mu$={mu} Total time $t={tc}$')
#             plt.tight_layout()
#             plt.savefig(f'{tm}-{str(mu)}-{t}.svg')



plt.close()
fig, axs = plt.subplots(2, 1)
axs[0].plot(x, u0, label='tc=0s')
axs[0].plot(x, u_21, label='tc=0.5s')
axs[0].plot(x, u_22, label='tc=1.0s')
axs[0].plot(x, u_23, label='tc=1.5s')
axs[0].plot(x, u_24, label='tc=2.0s')
# axs[0].legend()
axs[0].set_ylabel("u(x)")
axs[0].set_title('Runge-Kutta-2')

axs[1].plot(x, u0, label='tc=0s')
axs[1].plot(x, u1, label='tc=0.5s')
axs[1].plot(x, u2, label='tc=1.0s')
axs[1].plot(x, u3, label='tc=1.5s')
axs[1].plot(x, u4, label='tc=2.0s')
axs[1].legend()
axs[1].set_xlabel("x")
axs[1].set_ylabel("u(x)")
axs[1].set_title('Runge-Kutta-4')
plt.suptitle(f'$\mu={mu}$ Total time $t={tc}s$')
plt.tight_layout()
plt.savefig('totle.svg')
plt.show()

