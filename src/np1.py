# Import the NEURON module
from neuron import h, gui

# Import useful modules
import numpy as np
from matplotlib import pyplot as plt
import scipy.fftpack as scfft

# Create 3 soma, one for each kind of simulation (1 clean clamp, 2 noisy)
soma = [h.Section(name='soma%d'%(i)) for i in range(3)]

v_vec = []
for s in soma: 
    # Tune the soma, inserting HH model
    s.insert('hh')
    # Change diameter, length and axial resistance
    s.L = 18.8
    s(0.5).diam = 18.8
    s.Ra = 123.
    # Some output to check the good configuration
    h.psection(sec = s)

    # Create a vector recording the membrane reversal potential during the simulation 
    v = h.Vector() 
    v.record(s(0.5)._ref_v)
    v_vec.append(v)

# Get a time vector
t_vec = h.Vector()
t_vec.record(h._ref_t)

# Configure the simulation 
res = 1000                  # Nb. of current points
min_curr = 0                # Starting stimuli current
max_curr = .4               # Max stim. current
duration = 1000             # Duration of the clamp
delay = 0                   # Start time of the clamp 
h.tstop = duration + delay  # Simulation time

# Pre-configure the clean stimuli 
stim_clean = h.IClamp(soma[0](0))
stim_clean.delay = delay
stim_clean.dur = duration 

# Pre-configure noisy stimulis
stim_noisy = [h.IClamp(soma[i + 1](0)) for i in range(2)]
clamp_vec = []
N = int((delay + duration) / h.dt)      # Number of timesteps
for s in stim_noisy:
    s.delay = 0                         # This values are setted because we will
    s.dur = 1e9                         # manually play the clamp in the point process
    stim_v = h.Vector(N)            
    stim_v.play(s._ref_amp, h.dt)       # This vector will be set with a noisy clamp vector
    clamp_vec.append(stim_v)            # at each loop

# Function make_clamp_noisy creates a clamp with a defined amplitude and noise standard deviation (noise_amp)
def make_clamp_noisy(amp, noise_amp, delay, duration): 
    N = int((delay + duration) / h.dt)
    noise0 = np.random.normal(0., noise_amp, int( 2000*(delay + duration)*1e-3))   # Noise bandwith is limited to 2000Hz
    noise = np.real(scfft.ifft(scfft.rfft(noise0), N))                             # Use fft and ifft to scale the number of points
    noise = (noise - np.mean(noise)) * noise_amp/np.std(noise)                     # Rescale the noise

    # Create a clean clamp
    clean_clamp = np.array([0 if t < delay else amp for t in np.linspace(0, delay + duration, N)])

    #print("mean %f; std %f"%(np.mean(noise), np.std(noise)))

    return noise + clean_clamp

# Function get_spikes_count counts the number of spikes 
def get_spikes_count(v_vec, t_vec, thresh=85): 
    """ 
    We are looking for the top of the spike, so we can apply the first order criterion, 
    dV/dt = 0. 
    We don't use the second order, but we know the spike is sharp in time, so the derivative 
    is widely variating from positive to negative. We set a threshold and record all the 
    spikes over this values. 
    """
    dv_vec = [(v_vec[i+1]-v_vec[i])/(t_vec[i+1]-t_vec[i]) for i in range(len(t_vec) - 1)]
    #dt_vec = [(t_vec[i] + t_vec[i+1])/2 for i in range(len(t_vec) - 1)]
    # plt.plot(dt_vec, dv_vec)
    # plt.show()
    spikes_count = np.sum([dv_vec[i]>thresh and dv_vec[i+1]<thresh for i in range(len(dv_vec) - 1)])
    
    # The spikes count can be converted in Hertz, and is Hertz for duration = 1000
    return spikes_count


voltages =[[],[],[]]
# Loop other all the current steps
for current in np.linspace(min_curr, max_curr, res): 
    # Make a clamp with 2 different noises for this particular current step 
    v_05 = make_clamp_noisy(current,0.05, delay, duration)
    v_1 = make_clamp_noisy(current,.1, delay, duration) 

    # Assign it to the vector played in amp
    for i in range(len(v_05)):
        clamp_vec[0].x[i] = v_05[i]
        clamp_vec[1].x[i] = v_1[i]

    # Configure the clean clamp
    stim_clean.amp = current

    # Run the simulation 
    h.run() 

    # Store the voltages records
    for i in range(len(voltages)): 
        c = h.Vector() 
        c.copy(v_vec[i])
        voltages[i].append(c)


spikes_count = [[], [], []]
# Count the spikes for each voltage recorded
for i in range(len(spikes_count)): 
    for c in range(res): 
        spikes_count[i].append(get_spikes_count(voltages[i][c], t_vec)) 

# Plot and show the desired results

# for i in range(len(voltages)): 
#   plt.subplot(3,2,i*2+1)
#   plt.plot(t_vec, voltages[i][-1])


for i in range(len(spikes_count)):
    # plt.subplot(3, 2, 2)
    # plt.subplot(3, 2, (1 +i) * 2)
    plt.plot(np.linspace(min_curr, max_curr, res), spikes_count[i])

plt.show()



