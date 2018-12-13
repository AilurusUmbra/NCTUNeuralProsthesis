from neuron import h, gui
import numpy as np

import matplotlib.pyplot as plt

# Function make_myelinated_axon creates an axon with myelin
def make_myelinated_axon(diameter, nseg=3, innode_len=500, ranvier_len=1, n_nodes=4, n_innode=3, tag=None):
    if tag is None: 
        tag = np.random.randint(0,9)
    axon = []
    
    for i in range(n_innode + n_nodes): 

        if i % 2 == 0: 
            # Create a Ranvier Node 
            r = h.Section(name='ranvier%d%d'%(tag,i))
            # Configure the section 
            r(0.5).diam = diameter
            r.nseg = nseg
            r.Ra = 123.
            r.L = ranvier_len

            # A Ranvier node is a scection with a HH model
            r.insert('hh')

            if i > 0: 
                # Connect it to the previous element
                r.connect(axon[-1](1))
            axon.append(r)
            
        if i % 2 == 1: 
            # Create a section surrounded by myelin 
            m = h.Section(name='myelin%d%d'%(tag,i))
            # Configure the section 
            m(0.5).diam = diameter
            m.nseg = nseg
            m.Ra = 123.
            m.L = innode_len

            # Insert a passive channel 
            m.insert('pas')
            # Our assumption is the myelin is totally insulator
            m.g_pas = 0.000
            m.e_pas = -65

            # Connect it to the previous element
            m.connect(axon[-1](1))
            axon.append(m)  

    return axon

# Function make_non_myelinated_axon creates an axon without myelin 
def make_non_myelinated_axon(diameter,nseg=3,  axon_len=1504, n_nodes=7, tag=None):
    if tag is None: 
       tag = np.random.randint(0,9)
    # This axon constist of only one homogeneous section
    axon = h.Section(name='axon')
    # Configure the section
    axon(0.5).diam = diameter
    
    axon.Ra = 123. 
    axon.L = axon_len 
    
    # The section is caracterized by a HH model 
    axon.insert('hh')

    # axon(0.5).cm = 0.0714

    axon.nseg = nseg

    return axon

# Function get_spike_time gets the time at which the derivative  
def get_spike_time(v_vec, t_vec):
    dt_vec = [(t_vec[i] ) for i in range(len(t_vec) - 1)]
    dv_vec = [(v_vec[i+1]-v_vec[i])/(t_vec[i+1]-t_vec[i]) for i in range(len(t_vec) - 1)]

    dv2_vec = [(dv_vec[i+1] - dv_vec[i])/(dt_vec[i+1]-dt_vec[i])for i in range(len(dt_vec) - 1)]
    dt2_vec = [(dt_vec[i]) for i in range(len(dt_vec) - 1)]

    # plt.plot(t_vec, v_vec)
    # # plt.plot(dt_vec, dv_vec)
    # plt.plot(dt2_vec, (dv2_vec/np.max(dv2_vec) * 100.))
    # plt.show()

    # spikes = [dt_vec[i] if dv_vec[i]>thresh and dv_vec[i+1]<thresh else 1e9 for i in range(len(dv_vec) - 1)]
    return dt2_vec[np.argmax(dv2_vec)]

# Set up the simulation parameters
nseg = 15

# Range of diameter to test, including 1, 4, 9
diams = np.linspace(1,9,500)
# diams = [1,4,9]
speeds_1 = []
speeds_2 = []

delay = 1

duration = 2
amplitude = 6
h.tstop = 15

for diam in diams:
    # Create two different axons 
    a1 = make_myelinated_axon(diam, nseg, tag=1)
    a2 = make_non_myelinated_axon(diam, nseg*7, tag=2)

    # Create two stimulis with the same parameters
    stim = h.IClamp(a1[0](0))
    stim.delay = delay
    stim.dur = duration
    stim.amp = amplitude

    stim2 = h.IClamp(a2(0))
    stim2.delay = delay
    stim2.dur = duration
    stim2.amp = amplitude

    # Create a vector to store the time 
    t = h.Vector() 
    t.record(h._ref_t)

    loc = np.linspace(0,1,nseg)
    v_vec = [h.Vector() for i in range(7*nseg)]
    for i in range(len(v_vec)): 
        v_vec[i].record(a1[i//nseg](loc[i%nseg])._ref_v)

    v_vec_2 = [h.Vector() for i in range(nseg)] 
    for i in range(len(v_vec_2)): 
        v_vec_2[i].record(a2(loc[i])._ref_v)

    # Run the simulation 
    h.run() 

    # Total length of the axon
    total_len = 1504.0

    # Get the spike times for the first axon and compute the speed
    times = []
    for v in v_vec:
        times.append(get_spike_time(v, t))
    travel_time = times[-1] - times[0]
    speed = total_len / travel_time
    speeds_1.append(speed)

    # Get the spike time for the second axon and compute the speed
    times = []
    for v in v_vec_2:
        times.append(get_spike_time(v, t))
    travel_time = times[-1] - times[0]
    speed = total_len / travel_time
    speeds_2.append(speed)

    # print(times)

# Function linreg_2params performs linear regression with a 2-columns design matrix parametered by kernel
def linreg_2params(x, y, kernel, lambda_l2=1e-3): 
    assert(callable(kernel))
    phi = np.array([[1, kernel(v)] for v in x])
    w = np.dot(np.dot(np.linalg.pinv(np.dot(phi.transpose(), phi) + lambda_l2 * np.eye(2)), phi.transpose()), y)
    return w

# Model parameters - Optimum of LSE with L2 
w_myelinated = linreg_2params(diams, speeds_1, lambda x:x)
w_unmyelinated = linreg_2params(diams, speeds_2, lambda x: np.sqrt(x-1))

# Inference models
speed_myelinated_reg = lambda x : w_myelinated[0] + x * w_myelinated[1]
speed_unmyelinated_reg = lambda x : w_unmyelinated[0] + np.sqrt(x-1) * w_unmyelinated[1]

# Plot the results
l1, = plt.plot(diams, speeds_1, label='Myelinated axon')
l1bis, = plt.plot(diams, [speed_myelinated_reg(d) for d in diams], label='y = %d * x + %d'%(w_myelinated[1], w_myelinated[0]))
l2, = plt.plot(diams, speeds_2, label='Unmyelinated axon')
l2bis, = plt.plot(diams, [speed_unmyelinated_reg(d) for d in diams], label='y = %d * sqrt(x-1) + %d'%(w_unmyelinated[1], w_unmyelinated[0]))
plt.plot(diams, [speed_unmyelinated_reg(d) for d in diams])
plt.title("Conduction speed against diameter curve for myelinated and unmyelinated axons")
plt.xlabel("Axon diameter (µm)")
plt.ylabel("Conduction speed (µm/ms)")
plt.legend(handles=[l1, l1bis, l2, l2bis])

# Log results for further inspections 
with open('log.csv', 'w') as f: 
    for i in range(len(diams)): 
        f.write('%f,%f,%f\n'%(diams[i], speeds_1[i], speeds_2[i]))
    f.close()

plt.show()

