import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import scipy
import scipy.integrate
import time

import testingfuncs
import orbitfuncs
import initfuncs
import plotfuncs
#Program currently setup to plot snapshots for a parabolic orbit

#Setting initial positions and velocities of masses for a parabolic orbit
f=1 #f=M2/M1
r_min=10
x_separation=7.6
#initial_separation=12.4
x0=-x_separation/(1+1/f) #must be less than r_min *(f/(1+f)) in magnitude, ensures a constant initial separation of the masses when unequal masses are used but r_min is specified
#x0=(initial_separation-2*r_min)/2 #ensures a constant initial separation of the masses when different distances of closest approach are used
masses=np.array([1,f])
n_m=len(masses)
mass1_position0,mass1_velocity0,mass2_position0,mass2_velocity0=initfuncs.calc_parabolic(f,r_min,x0)


#Start and end times at which to calculate solution
t_min=0
t_max=100
n_intervals=10000 #same or factor of 10 more than t_max for smooth plots, must be high for trajectory plot, *10 for animations


#Setting particles
radii=np.linspace(2,6,5)
number_radii=np.linspace(12,36,5,dtype=int)
n_p=np.sum(number_radii)
print("Number of particles = {}".format(n_p))


#Setting initial values of particles and masses for input into the integrator
#initial_values=initfuncs.init_equations_one(radii,number_radii,mass1_position0,mass1_velocity0) #if one mass only
initial_values=initfuncs.init_equations_two(radii,number_radii,mass1_position0,mass2_position0,mass1_velocity0,mass2_velocity0) #for two masses
#initial_values=initfuncs.init_equations_masses(mass1_position0,mass2_position0,mass1_velocity0,mass2_velocity0)   #for two masses and no particles


#Calculating solution & printing runtime
start=time.time()
solution=orbitfuncs.orbit_solution(t_min,t_max,initial_values,n_m,n_p,masses,n_intervals,'DOP853')
end=time.time()
print("Time taken to calculate solution: {}".format(end-start))


#Plotting solution at different, equally spaced, times
n_snapshots=4
t_snapshots=np.zeros(n_snapshots)
for i in range (n_snapshots):
    t_snapshots[i]=i*(t_max-t_min)/n_snapshots    #not strictly necessary, only for purpose of plotting continuously and snapshots in same run of program


plotfuncs.snapshot_plot_samefig(t_snapshots,solution.t,solution.y,n_m,n_p,35,35,2,2,'parabolic')  # for plots on the same figure
#plotfuncs.snapshot_plot(t_snapshots,solution.t,solution.y,n_m,n_p,60,60) #plots of the solution at snapshots in time
#plotfuncs.trajectory_plot(0,n_m,solution.y,1000,1000) #for plots of the trajectories of all particles and masses


