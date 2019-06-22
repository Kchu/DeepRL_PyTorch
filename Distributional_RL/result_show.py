import os
import pickle
import matplotlib.pyplot as plt

# paths for result log
RESULT_PATH = ['./data/plots/quota_result.pkl','./data/plots/C51_result.pkl',\
				'./data/plots/qr-dqn_result.pkl','./data/plots/iqn_result.pkl'\
				'./data/plots/aqn_result.pkl']

# model load with check
result = []
for i in range(len(RESULT_PATH)):
	if os.path.isfile(RESULT_PATH[i]):
	    pkl_file = open(RESULT_PATH[i],'rb')
	    result.append(pickle.load(pkl_file))
	    pkl_file.close()
	else:
	    print('Can not find:', RESULT_PATH[i])

# plot the figure
print('Load complete!')
print('Plotting the curves!')
if len(result) > 0:
	plt.plot(range(len(result[0])), result[0], label="QUOTA")
if len(result) > 1:
	plt.plot(range(len(result[1])), result[1], label="C51")
if len(result) > 2:
	plt.plot(range(len(result[2])), result[2], label="QR-DQN")
if len(result) > 3:
	plt.plot(range(len(result[3])), result[3], label="IQN")
if len(result) > 4:
	plt.plot(range(len(result[3])), result[3], label="AQN")
plt.legend()
plt.xlabel('iteration times')
plt.ylabel('score')
plt.tight_layout()
plt.show()