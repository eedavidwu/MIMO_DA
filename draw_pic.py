import matplotlib.pyplot as plt
import numpy as np

x=["1,4","9,12","16,19"]

SNR_no_decom=[28.13384, 30.40684, 30.59468]
SNR_decom=[30.45692, 34.80637, 35.11107]
SNR_1_4_decompose=[31.25963, 33.72071, 34.24459]
SNR_9_12_decompose=[30.20294, 34.25294, 35.43972]
SNR_16_19_decompose=[29.3582, 34.04726, 35.60438]

#SNR_attention=[33.11577, 33.11031, 33.10403]

plt.title('Performance of different models all with attention')
plt.xlabel('SNR (dB)', size=10)
plt.ylabel('Ave_PSNR (dB)', size=10)
plt.plot(x, SNR_no_decom, color='g', linestyle='-', marker='o', label='attention without decom')
plt.plot(x, SNR_decom, color='r', linestyle='-', marker='*',label='attentionwith decompose')
plt.plot(x, SNR_1_4_decompose, color='b', linestyle='-', marker='*',label='traditional_JSCC_1_4_decom')
plt.plot(x, SNR_9_12_decompose, color='grey', linestyle='-', marker='*',label='traditional_JSCC_9_12_decom')
plt.plot(x, SNR_16_19_decompose, color='black', linestyle='-', marker='*',label='traditional_JSCC_16_19_decom')


plt.legend()
plt.show()
plt.savefig('./PSNR_MIMO_decom.jpg')

