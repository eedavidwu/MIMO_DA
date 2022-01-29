import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

x=["1,4","9,12","16,19"]


from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


fig=plt.figure()
ax=plt.axes(projection='3d')
ax.invert_xaxis()
SNR_1=[1,4,9,12,16,19]
SNR_2=[1,4,9,12,16,19]
XX,YY=np.meshgrid(SNR_1,SNR_2)
PSNR_1_4=[[30.4755, 31.1118, 31.59732, 31.72337, 31.79335, 31.8196], [31.11123, 31.83558, 32.41785, 32.55857, 32.64618, 32.67641], [31.59718, 32.41422, 33.07126, 33.23895, 33.33987, 33.37738], [31.72308, 32.56221, 33.24109, 33.41134, 33.5174, 33.55275], [31.79113, 32.6453, 33.33848, 33.51972, 33.62817, 33.66148], [31.82386, 32.67798, 33.37583, 33.5548, 33.66218, 33.70026]]
PSNR_9_12=[[29.37653, 30.22705, 30.90928, 31.0848, 31.19688, 31.232], [30.22104, 31.26058, 32.14815, 32.37952, 32.53253, 32.57654], [30.90977, 32.15389, 33.27251, 33.57502, 33.77347, 33.84134], [31.08862, 32.3867, 33.5767, 33.9057, 34.12137, 34.19174], [31.18942, 32.53956, 33.77497, 34.12027, 34.34263, 34.41991], [31.22725, 32.58273, 33.83743, 34.18753, 34.42082, 34.49796]]
PSNR_16_19=[[28.02022, 28.97241, 29.78234, 29.98573, 30.1134, 30.16428], [28.97469, 30.20748, 31.30626, 31.59562, 31.78711, 31.85091], [29.78058, 31.29878, 32.7846, 33.21215, 33.49391, 33.5961], [29.98821, 31.59578, 33.21229, 33.69212, 34.01494, 34.12896], [30.12646, 31.78976, 33.49375, 34.01361, 34.36354, 34.48497], [30.16405, 31.85684, 33.59322, 34.12693, 34.48538, 34.61236]]

PSNR_1_4=np.array(PSNR_1_4)
PSNR_9_12=np.array(PSNR_9_12)

surf_1 = ax.plot_surface(XX, YY, PSNR_1_4, color='r',
                       linewidth=0, antialiased=True)
surf_2 = ax.plot_surface(XX, YY, PSNR_9_12,color='g', 
                       linewidth=0, antialiased=True)

#surf_2 = ax.plot_surface(XX, YY, PSNR_9_12,color='r', cmap=cm.coolwarm, linewidth=0, antialiased=True)
#fig.colorbar(surf, shrink=0.5, aspect=5)
ax.view_init(20,30)
plt.savefig('surface.jpg')

for c, m ,out,data_label in [('r', 'o',PSNR_1_4,'1-4'), ('b', '^',30,'9-12'),('g', '*',0,'16-19')]:
    for i in range(6):
        for j in range(6):
            xs = SNR_1[i]
            ys = SNR_1[j]
            zs = PSNR_1_4[i][j]
            if ((i==0)&(j==0)):
                ax.scatter(xs, ys, zs, c=c, marker=m,label=data_label)
            else:
                ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('SNR_1')
ax.set_ylabel('SNR_2')
ax.set_zlabel('PSNR')
ax.view_init(elev=20., azim=-35)
ax.legend()
#ax.grid(True)
plt.show()
fig.savefig('./3_d_test.jpg')

SNR_1_4=[31.11444, 33.23727, 33.66461]
SNR_9_12=[30.22054, 33.57658, 34.41987]
SNR_16_19=[28.9796, 33.20848, 34.48501]
SNR_attention=[32.52459, 32.53044, 32.52574] 
plt.title('Performance of different models')
plt.xlabel('SNR (dB)', size=10)
plt.ylabel('Ave_PSNR (dB)', size=10)
plt.plot(x, SNR_1_4, color='r', linestyle='-', marker='o', label='1-4')
plt.plot(x, SNR_9_12, color='g', linestyle='-', marker='*',label='9-12')
plt.plot(x, SNR_16_19, color='b', linestyle='-', marker='*',label='16-19')
plt.plot(x, SNR_attention, color='grey', linestyle='-', marker='*',label='random')


plt.legend()
plt.show()
#plt.savefig('./PSNR_MIMO_example.jpg')

