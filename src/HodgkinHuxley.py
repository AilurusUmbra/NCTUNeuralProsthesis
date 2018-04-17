import numpy as np
import scipy.integrate as scpi

import matplotlib.pyplot as plt

class Stimuli(object): 
	def __init__(self, time_function, time_start, time_stop, resolution):
		self.time_start = time_start
		self.time_stop = time_stop
		self.resolution = resolution
		self.timepoints = np.linspace(self.time_start, self.time_stop, self.resolution)

		assert(callable(time_function))
		self.time_function = time_function

	def __call__(self, t): 
		return self.time_function(t)

class HodgkinHuxley(object): 

	def __init__(self, gNa=120., gK=36., gL=0.3, Cm=1.0, ENa=115.0, EK=-12.0, El=10.613): 
		# Mean sodium channel conductivity
		self.gNa = gNa
		# Sodium potential
		self.ENa = ENa

		# Mean potassium channel conductivity
		self.gK = gK
		# Potassium potential
		self.EK = EK

		# Mean leek conductivity
		self.gL = gL
		# Leak potential
		self.El = El

		# Membrane capacitance
		self.Cm = Cm

		# Current stimuli
		self.Iinj = None

	# Potassium channel

	def alpha_n(self, Vm): 
		return (10 - Vm) / (100 * np.exp((10-Vm)/10) - 1)

	def beta_n(self, Vm): 
		return 0.125 * np.exp(-Vm / 80)

	def n_inf(self, Vm): 
		return self.alpha_n(Vm) / (self.alpha_n(Vm) + self.beta_n(Vm))

	# Potassium current
	def IK(self, Vm, n): 
		return self.gK * n **4 * (Vm - self.EK)


	# Sodium channels

	# Fast channel 
	def alpha_m(self, Vm): 
		return (25 - Vm) / (10 * np.exp((25-Vm) / 10) - 1) 

	def beta_m(self, Vm): 
		return 4 * np.exp(-Vm/18)

	def m_inf(self, Vm): 
		return self.alpha_m(Vm) / (self.alpha_m(Vm) + self.beta_m(Vm))

	# Slow channel
	def alpha_h(self, Vm): 
		return 0.07*np.exp(-Vm/20)

	def beta_h(self, Vm): 
		return 1 / (np.exp((30-Vm) / 10) +1)

	def h_inf(self, Vm): 
		return self.alpha_h(Vm) / (self.alpha_h(Vm) + self.beta_h(Vm))

	# Sodium current
	def INa(self, Vm, m, h): 
		return self.gNa * m**3 * h * (Vm - self.ENa) 


	# Leak current 
	def Il(self, Vm): 
		return self.gL * (Vm - self.El)



	def compute_dydt(self, y, t): 
		Vm, n, m, h = y

		# Membrane potential dynamics
		dVmdt = (self.Iinj(t) - self.IK(Vm, n) - self.INa(Vm, m, h) - self.Il(Vm)) / self.Cm

		# Potassium channels dynamics
		dndt = self.alpha_n(Vm) * (1. - n) - self.beta_n(Vm) * n

		# Sodium channels dynamics
		# Fast channels
		dmdt = self.alpha_m(Vm) * (1. - m) - self.beta_m(Vm) * m
		# Slow channel
		dhdt = self.alpha_h(Vm) * (1. - h) - self.beta_h(Vm) * h

		dydt = [dVmdt, dndt, dmdt, dhdt]

		return dydt




	def stimulate(self, stimuli, Vm_0=0.):
		# The input stimuli is the injected current
		assert(isinstance(stimuli, Stimuli))
		self.Iinj = stimuli

		# Initial y point
		y_0 = [Vm_0, self.n_inf(Vm_0), self.m_inf(Vm_0), self.h_inf(Vm_0)]	

		# Integrate to get a time serie of y
		y = scpi.odeint(self.compute_dydt, y_0, stimuli.timepoints)

		return y

if __name__ == "__main__": 
	hh = HodgkinHuxley()

	y = hh.stimulate( Stimuli(lambda t : 0 if t < 5 or t > 20 else 50,0,100,10000) )
	Vm = [y[i][0] for i in range(10000)]

	plt.plot(np.linspace(0,100,10000),Vm)
	plt.show()
	print(Vm)