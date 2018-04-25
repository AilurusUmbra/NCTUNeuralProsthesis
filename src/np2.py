from neuron import h, gui
import numpy as np

import matplotlib.pyplot as plt
def make_myelinated_axon(diameter, nseg=3, innode_len=500, ranvier_len=1, n_nodes=4, n_innode=3):
    tag = np.random.randint(0,9)
    axon = []
    
    for i in range(n_innode + n_nodes): 
        if i % 2 == 0: 
            r = h.Section(name='ranvier%d%d'%(tag,i))
            r(0.5).diam = diameter
            r.nseg = nseg
            r.Ra = 123.
            r.L = ranvier_len
            r.insert('hh')
            if i > 0: 
                r.connect(axon[-1](1))
            axon.append(r)
            
        if i % 2 == 1: 
            m = h.Section(name='myelin%d%d'%(tag,i))
            m(0.5).diam = diameter
            m.nseg = nseg
            m.Ra = 123.
            m.L = innode_len
            m.insert('pas')
            m.g_pas = 0.000
            m.e_pas = -65

            m.connect(axon[-1](1))
            axon.append(m)  

    return axon    

def get_spike_time(v_vec, t_vec, thresh=0.2):
    dt_vec = [(t_vec[i] + t_vec[i+1])/2 for i in range(len(t_vec) - 1)]
    dv_vec = [(v_vec[i+1]-v_vec[i])/(t_vec[i+1]-t_vec[i]) for i in range(len(t_vec) - 1)]
    spikes = [dt_vec[i] if dv_vec[i]>thresh and dv_vec[i+1]<thresh else 1e9 for i in range(len(dv_vec) - 1)]
    return np.min(spikes)

nseg = 4 
diam = 1

a1 = make_myelinated_axon(diam, nseg)
print(a1)
for i in a1:
    h.psection(sec=i)
v_vec = [h.Vector() for i in range(7*nseg)]
t = h.Vector() 
h.topology()
shape_window = h.PlotShape()
shape_window.exec_menu('Show Diam')

loc = np.linspace(0,1,nseg)
for i in range(len(v_vec)): 
    v_vec[i].record(a1[i//nseg](loc[i%nseg])._ref_v)
t.record(h._ref_t)

stim = h.IClamp(a1[0](0))
stim.delay = 50
stim.dur = 100
stim.amp = 0.05


h.tstop = 1500
h.run() 

times = []
for v in v_vec:
	print(times)
	times.append(get_spike_time(v, t))

times = np.array(times)
times -= (times[0])

vitesse = []
pos = lambda i : loc[i%nseg] * a1[i//nseg].L
for i in range(7*nseg - 1): 
	if((times[i+1] - times[i]) != 0):
		vitesse.append((pos(i+1) - pos(i)) / (times[i+1] - times[i]))  
print(times) 
print(np.mean(vitesse))

for i in range(7 * nseg): 
    # plt.subplot(nseg,1,i+1)
    plt.plot(t, v_vec[i])


plt.show()