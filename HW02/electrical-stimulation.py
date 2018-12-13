import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import spline

def read_simu_file(filename): 
	data = None 
	with open(filename,'r') as f: 
		first = True
		keysval = []
		for l in f: 
			if first:
				for t in l.split(' '): 
					if t != '' and t.split('(')[0] != '': 

						keysval.append([t.split('(')[0],[]])
				first = False
			else: 
				tok = 0
				for t in l.split(' '): 
					if t != '': 
						keysval[tok][1].append(float(t))
						tok += 1
		for k in keysval: 
			k[1] = np.asarray(k[1])
		data=dict(keysval)
		print(data)
		return data

if __name__ == "__main__": 
	F = "HW2-q2a-test.txt"
	F2 = "HW2-q2b-test.txt"

	simu_data = read_simu_file(F2)

	simu_data['L'] /= 1e6
	simu_data['U'] /= simu_data['U'].max() 
	simu_data['U'] *= 0.025

	Ia = 0.018
	h2 = 0.01**2
	sig0 = 0.15552
	Vmono = lambda x : Ia / (4 * np.pi * np.sqrt(h2 + (x - np.mean(x))**2))
	Vcat = lambda x : Ia / (4 * np.pi * np.sqrt(h2 + (x+0.005- np.mean(x))**2))

	Van = lambda x : -Ia / (4 * np.pi * np.sqrt(h2 + (x-0.005- np.mean(x))**2))

	X = np.linspace(-0,0.32,300)

	l = np.linspace(simu_data['L'].min(), simu_data['L'].max(), 300)

	print(np.sum((Vmono(X)-spline(simu_data['L'],simu_data['U']* 1e3,l))**2))

	l1, = plt.plot(X,(Vcat(X)+Van(X))*1e3/2.,label='Formula')
	# l1, = plt.plot(X, Vmono(X) * 1e3, label='Formula')
	# l2, = plt.plot(simu_data['L'],simu_data['U']*1e3,label='Simulation')
	l2, = plt.plot(l, spline(simu_data['L'],simu_data['U']* 1e3,l),label='Simulation')
	plt.ylabel('Axon potential (mV)')
	plt.xlabel('Distance along the axon (m)')
	plt.legend(handles=[l1, l2])
	plt.show()

