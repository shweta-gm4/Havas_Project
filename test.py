import plotly 
print(plotly.__version__)

'''
import matplotlib.pyplot as plt

load = [0, 1650, 3300, 5200, 6800, 7750, 8650, 9300, 10150, 10400]

length = [2, 2.002, 2.004, 2.006, 2.008, 2.01, 2.02, 2.04, 2.08, 2.120]

fig1 = plt.figure()
plt.plot(length, load, 'x') #plot each point with a cross
#plt plot(length, load) # plot data in x and y axis
plt.xlabel('load [lbf]')
plt.ylabel('length [in]')
plt.title('Load vs Length')
plt.grid()
plt.show()
fig1.savefig('load_vs_length.png')
fig1.savefig('load_vs_length.pdf')
plt.close(fig1)
from unittest import result

import math 

dataset0 = [1, -2, 3, -4, 5]
dataset1 = [10, -20, 30, -40, 50]
dataset2 = [1.03, -2.04, 3.40, -40.05, 5.3]
dataset3 = [1.5, 1.3, 1.7, 2.5, 2.3]


def mean_abs (data_list):
    total = 0
    for datum in data_list:
        total += abs(datum)
        means_abs_value = total / len(data_list)
        return means_abs_value
    
#dataset0 
mean_abs0 = mean_abs(dataset0)
print ('dataset 0 average =', mean_abs0)

#dataset1
mean_abs1 = mean_abs(dataset1)
print ('dataset 1 average =', mean_abs1)

datasets = [dataset0, dataset1, dataset2, dataset3]

#loop through datasets
#the function mean_abs computes the mean of the absolute value of the elements in a list
for k in range(len(datasets)):
    mean_abs_k = mean_abs(datasets[k])
    print ('dataset', k, 'average =', mean_abs_k)


from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Flask is working!"

if __name__ == "__main__":
    app.run(debug=True)

'''


        
'''
KEY_FILE = r"C:\Users\shweta.ann-george\OneDrive - Havas\Bureau\Projet_Havas\DraftDYA\config\google-credentials.json"

with open (KEY_FILE, "r") as f:
    creds_info = json.load(f)

credentials = Credentials.from_authorized_user_info(creds_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]) 
'''