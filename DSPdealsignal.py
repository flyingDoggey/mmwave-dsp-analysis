import numpy as np
import mmwave as mw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
## 解析ADC原始数据
#定义天线
RadarTx_num = 3
RadarRx_num = 4

#距离点数：每个脉冲点数
Range_num = 256
#多普勒点数：重复脉冲数
Doppler_num = 32

class Params:
    NChirps = 32
    NChannel = 4
    NSaMP = 256
    Fs = 5.5e6
    c = 3.0e9 #光速
    StartFreq = 60e9
    FreqSlope = 68e12 #斜率
    bandwidth = 3.165e9 #带宽
    Lambda= c/StartFreq #波长
    Tc = 195e-6 #脉冲宽度
global FFT2_mag
X, Y = np.meshgrid(Params.c * np.arange(Params.NSaMP) * Params.Fs / 2 / Params.FreqSlope / Params.NSaMP,
                   np.arange(-Params.NChirps/2, Params.NChirps/2) * Params.Lambda / Params.Tc / Params.NChirps / 2)

# 读取数据

with open('D:/Mmwave_radar/codes/orignaldatasample/usb1.txt', 'r') as f:
    Data_hex = f.read().replace('\n', '')

Data_hex_list = Data_hex.split(' ') #将字符串转换为列表
for hex_data in Data_hex_list:
    if hex_data == '':
        Data_hex_list.remove(hex_data) #去除空格
#print(Data_hex_list)
Data_dec = [int(hex_data,16) for hex_data in Data_hex_list] #将16进制转换为10进制
#print(Data_dec)
#组合数据
Data_dec = np.array(Data_dec)#将列表转换为数组
Data_combin = np.zeros((RadarRx_num*RadarTx_num*Range_num*Doppler_num*2))
for i in range(1,RadarTx_num*RadarRx_num*Range_num*Doppler_num*2+1):
    Data_combin[i-1] = Data_dec[(i-1)*2] + Data_dec[(i-1)*2+1]*256
    if Data_combin[i-1] > 32767:
        Data_combin[i-1] = Data_combin[i-1] - 65536

#数据分离
Data_ADC = np.zeros((RadarTx_num,Doppler_num,RadarRx_num,Range_num*2))
for t in range(1,RadarTx_num+1):
    for i in range(1,Doppler_num+1):
        for j in range(1,RadarRx_num+1):
            for k in range(1,Range_num*2+1):
                Data_ADC[t-1,i-1,j-1,k-1] = Data_combin[(((t-1)*Doppler_num+(i-1))*RadarRx_num+(j-1))*Range_num*2+k-1]

#打印虚实数据
Re_Data_all= np.zeros((RadarTx_num*Doppler_num*RadarRx_num*Range_num))#实部
Im_Data_all= np.zeros((RadarTx_num*Doppler_num*RadarRx_num*Range_num))#虚部
for i in range(1,RadarTx_num*Doppler_num*RadarRx_num*Range_num+1):
    Im_Data_all[i-1] = Data_combin[(i-1)*2]
    Re_Data_all[i-1] = Data_combin[(i-1)*2+1]
#print(Re_Data_all)
#print(Im_Data_all)

## 绘制虚部实部图像 即未处理过的频幅图
#plt.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(Re_Data_all)
plt.title('实部波形')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(Im_Data_all)
plt.title('虚部波形')
plt.grid(True)
plt.show()

# 打印32个chirp的数据
Re_Data_chirp = np.zeros((Doppler_num,Range_num))#实部
Im_Data_chirp = np.zeros((Doppler_num,Range_num))#虚部

for chirps in range(1,Doppler_num+1):
    #plt.figure()
    for j in range(1,RadarTx_num+1):
        for k in range(1,RadarRx_num+1):
            for i in range(1,Range_num+1):
                Re_Data_chirp[chirps-1,i-1] = Data_ADC[j-1,chirps-1,k-1,2*(i-1)+1]
                Im_Data_chirp[chirps-1,i-1] = Data_ADC[j-1,chirps-1,k-1,2*(i-1)]
            #plt.subplot(RadarTx_num,RadarRx_num,(j-1)*RadarRx_num+k)
            #plt.plot(Re_Data_chirp[chirps-1,:])
            #plt.title('chirp=%d'%chirps)
# 把chirp数据组合
REandIM_Data = Re_Data_chirp + Im_Data_chirp*1j #转换为复数信号
Im_Data1 = Im_Data_chirp.reshape(1,-1)#转换为一维数组
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.figure(2)
plt.plot(Im_Data1[0,:])
plt.title('chirps')
plt.grid(True)
plt.show()

# 1DFFT
fft1d = np.zeros((Doppler_num,Range_num))
for i in range(1,Doppler_num+1):
    fft1d[i-1,:] = np.fft.fft(REandIM_Data[i-1,:])
# 打印1DFFT图像
FFT1D = np.abs(fft1d)
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
fig = plt.figure(3) #定义图片
ax = Axes3D(fig) #定义3D坐标轴
ax = fig.add_subplot(111, projection='3d')
X,Y = np.meshgrid(range(Range_num),range(Doppler_num))
#ax.plot3D(X,Y,FFT1D,camp = 'rainbow')
ax.plot_surface(X,Y,FFT1D,cmap='viridis')
ax.set_xlabel('距离')
ax.set_ylabel('速度')
ax.set_zlabel('幅度')
ax.set_title('距离维FFT结果')
plt.show()

# 2DFFT
fft2d = np.zeros((Doppler_num,Range_num))
for i in range(1,Range_num+1):
    fft2d[:,i-1] = np.fft.fftshift(np.fft.fft(fft1d[:,i-1]))#通过将零频分量移动到数组中心来移动频率分量
FFT2D = np.abs(fft2d)
# 打印2DFFT图像
fig2 = plt.figure(4) #定义图片
ax2 = Axes3D(fig2) #定义3D坐标轴
ax2 = fig2.add_subplot(111, projection='3d')
X,Y = np.meshgrid(range(Range_num),range(Doppler_num))
ax2.plot_surface(X,Y,FFT2D,cmap='viridis')
ax2.set_xlabel('距离')
ax2.set_ylabel('速度')
ax2.set_zlabel('幅度')
ax2.set_title('速度维FFT结果')
plt.show()

# 对原始数据进行 FFT 处理
Data_fft = np.fft.fftshift(np.fft.fft(Data_ADC, axis=3), axes=3)

# 提取距离和速度维度
Range_num = Data_fft.shape[3] // 2
Doppler_num = Data_fft.shape[1]
Range_res = 0.0375
Doppler_res = 0.078125
Range_grid = np.arange(Range_num) * Range_res
Doppler_grid = np.arange(Doppler_num) * Doppler_res

# 将 FFT 结果按照距离和速度维度排列
Data_range = np.abs(Data_fft[:, :, :, :Range_num])
Data_doppler = np.abs(Data_fft[:, :, :, Range_num:])
Data_doppler_new = Data_doppler.sum(axis=0).reshape(128, 256)
#a11111 = Data_doppler.sum(axis=0)
#a11111 = np.transpose(a11111,(0,2,1))
# 绘制 range-doppler 热图
plt.figure()
plt.imshow(Data_doppler_new, extent=[Doppler_grid[0], Doppler_grid[-1], Range_grid[-1], Range_grid[0]], aspect='auto')
plt.xlabel('Doppler Velocity (m/s)')
plt.ylabel('Range (m)')
plt.title('Range-Doppler Heatmap')
plt.colorbar()
plt.show()
## 加窗降低速度维的旁瓣
# 在信号处理中，加窗是一种常见的技术，用于减少信号的频谱泄漏和旁瓣。
# 在这里，加窗被用于降低速度维的旁瓣。旁瓣是指频谱中除了主要峰值之外的其他峰值或波纹。
# 在进行 FFT 分析时，由于信号是有限的，因此会出现频谱泄漏和旁瓣。
# 加窗技术通过在信号的时间域上乘以一个窗函数，来减少频谱泄漏和旁瓣的影响。
# 在这里，加窗被用于减少速度维的旁瓣，以提高 FFT 分析的精度和准确性。

## 加窗降低距离维的旁瓣
# 在这里，我们可以使用汉宁窗（Hanning window）来对 chirp 数据进行加窗处理，以降低距离维的旁瓣。
# 汉宁窗是一种常用的窗函数，其形式为 w(n) = 0.5 - 0.5 * cos(2 * pi * n / (N - 1))，其中 N 是窗口长度，n 是窗口中的样本点索引。
# 在这里，我们将窗口长度设置为 256，即与 chirp 数据的长度相同。

data_new = np.zeros((1,256))
data_new1 = np.zeros((1,256))

count = 0
chirpcount = 96

for i in range(0,len(Re_Data_all),1024):
    for i in range(0,32768,1024):
        count = count + 1
        data_new[count-1,:] = Re_Data_all[i:i+256]
        data_new1[count-1,:] = Im_Data_all[i:i+256]
        #print(data_new)
        #print(data_new1)
data_new = data_new.reshape(1,-1)
data_new1 = data_new1.reshape(1,-1)
# 带窗的1dfft
ReIm_Data1 = data_new + data_new1*1j
ReIm_Data1 = ReIm_Data1.reshape(chirpcount,256)
fft1d1 = np.zeros((chirpcount,Range_num))
plt.figure(5)
plt.plot(data_new1[0])
    

