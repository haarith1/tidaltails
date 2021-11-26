import numpy as np
import time
from orbitfuncs import orbit_solution

def calc_energy(masses,n_p,time_index,solution):
    
    """Takes positions and velocities of particles and heavy masses (contained in solution) and returns the energy of the heavy masses and the energy of the particles at a certain time.
    As the heavy masses are not attracted to the light ones in this simulation, the total energy of the masses will be conserved but the total energy
    of the particles will only be conserved when there is a single heavy mass at rest."""

    n_m=len(masses) #number of masses
    position_masses=(solution[0:2*n_m]).T[time_index].reshape(n_m,2) #easier to work with column vectors for positions
    position_particles=(solution[2*n_m:2*n_m+2*n_p]).T[time_index].reshape(n_p,2)

    velocity_masses=(solution[2*(n_m+n_p):4*n_m+2*n_p]).T[time_index].reshape(n_m,2)
    velocity_particles=(solution[4*n_m+2*n_p:4*(n_m+n_p)]).T[time_index].reshape(n_p,2)

    kenergy_particles=0 #initailising the different energies, this is kinetic energy of particles
    venergy_particles=0 #potential energy of particles
    kenergy_masses=0
    venergy_masses=0

    kenergy_particles=np.sum(0.5*np.linalg.norm(velocity_particles,axis=1)**2) #vectorized way of calculating the kinetic energy
    
    for i in range(n_p):

        for j in range(n_m):

            venergy_particles+=-masses[j]/(np.linalg.norm(position_particles[i]-position_masses[j])) #calculating the potential energy of the particles due to their attraction to the heavy masses
    
    energy_particles=kenergy_particles+venergy_particles #total energy of the particles
    
    kenergy_masses=np.sum(0.5*np.linalg.norm(velocity_masses,axis=1)**2)

    for i in range(n_m):

        for j in range(n_m):

            if j>i: #ensures that interactions are only singly counted

                venergy_masses+=-masses[i]*masses[j]/(np.linalg.norm(position_masses[i]-position_masses[j])) #calculating the potential energy of the masses
            
            else:

                continue

    energy_masses=kenergy_masses+venergy_masses #total energy of the heavy masses

    return energy_masses,energy_particles


def captured_count(solution_pos,solution_time,time,n_m,n_p):

    """Counts how many particles have y>0 at an inputted time as this is a measure of the number of particles captured by the perturbing galaxy"""

    time_index=np.abs(solution_time-time).argmin()  #the time index of the closest time in the solution to the inputted time
    y_particles=solution_pos[2*n_m+1:2*(n_m+n_p):2].T[time_index] #y values of particles

    signs=np.sign(y_particles) #returns the signs of the y coordinates of the particles
    n_captured=np.count_nonzero(signs==1) #returns the number of particles with y>0 by counting how many elements of signs are equal to 1

    return n_captured


def tail_count(radii,n_p,time,solution_pos,solution_time):

    """Estimates how many particles are in the tidal tail by counting the number of particles with y<0 that are beyond the max radius of the heavy mass at a given time."""

    max_radius=np.amax(radii) 
    n_m=2 #number of masses
   
    time_index=np.abs(solution_time-time).argmin()  #the time index of the closest time in the solution to the inputted time
    
    position_mass=solution_pos[0:2].T[time_index]
    position_particles=np.copy((solution_pos[2*n_m:2*n_m+2*n_p]).T[time_index].reshape(n_p,2)) #don't want to change the solution hence make a copy

    for i in range(len(position_particles)):  #don't want to count particles with y>0

        if position_particles[i][1]>0:

            position_particles[i]=position_mass
    
    separations=np.linalg.norm(position_particles-position_mass,axis=1) #distance of particles from heavy mass
    separations=separations-max_radius  #distance of particles from heavy mass - max initial radius, if positive then particle is (likely) in the tidal tail, the above code has ensured that
    #particles with y<0 have a negative value and therefore aren't counted

    signs=np.sign(separations)
    n_tail=np.count_nonzero(signs==1) #number of particles beyond the max radius with y<0

    return n_tail


def disrupted(solution_pos,radii,number_radii,threshold,solution_time,time,n_m,n_p):

    """Calculates the fraction of particles at each initial radius which have moved away from their initial radius by a certain inputted threshold at an inputted time."""
    
    initial_radii=np.repeat(radii,number_radii) #create an array containing the initial radius of each particle
    
    time_index=np.abs(solution_time-time).argmin()  #the time of calculation

    position_particles=(solution_pos[2*n_m:2*n_m+2*n_p]).T[time_index].reshape(n_p,2)
    position_mass=solution_pos[0:2].T[time_index].reshape(1,2)

    current_radii=np.linalg.norm(position_particles-position_mass,axis=1) #the current distance particles are away from the mass which they initially orbited
 
    displacement_particles=current_radii-initial_radii #the radial displacement of the particles

    threshold_count=np.zeros(n_p) #+1 if a particle has been displaced by more than the threshold and 0 otherwise

    for i in range (n_p):

        if np.abs(displacement_particles[i])>=threshold:

            threshold_count[i]=1

        else:

            threshold_count[i]=0

    fraction_disrupted=np.zeros(len(radii)) #will contain the fraction of particles disrupted for each initial radius

    for i in range(len(radii)):
        
        fraction_disrupted[i]=np.sum(threshold_count[np.sum(number_radii[0:i]):np.sum(number_radii[0:i])+number_radii[i]])/number_radii[i] #sums how many particles are displaced by more than the threshold for each radius

    return number_disrupted





