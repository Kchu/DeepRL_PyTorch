# Author for codes: sungyubkim, Chu Kun(kun_chu@outlook.com)
import os
import pickle
import matplotlib.pyplot as plt

# paths for result log
RESULT_PATH = ['./data/plots/quota_result.pkl','./data/plots/iqn_result.pkl',]

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

plt.plot(range(len(result[0])), result[0], label="QUOTA")
plt.plot(range(len(result[1])), result[1], label="IQN")

plt.legend()
plt.xlabel('Iteration times(Thousands)')
plt.ylabel('Score')
plt.tight_layout()
plt.grid()
plt.show()