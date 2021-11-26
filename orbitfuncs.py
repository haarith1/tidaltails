import numpy as np
import scipy
import scipy.integrate


def acc_particle_calc(position_masses,position_particle,masses): 

    """Used to calculate the acceleration of the particles in a vectorised fashion, sums over the forces from each mass,
    position_particle may either be a 1D or 2D array, the latter case allowing for the accelerations to be calculated for each particle in a vectorized manner"""

    n_p=len(position_particle)   #number of particles and masses
    n_m=len(masses)

    acceleration_particle=-masses[0]*(position_particle-position_masses[0])/(np.linalg.norm(position_particle-position_masses[0],axis=1).reshape(n_p,1)**3) #initalize the array of accelerations
    #newton's law of gravitation, axis=1 means that the modulus of each "vector" in array is calculated i.e along the rows, have to reshape for division 

    for j in range(1,n_m):  #sum over the force from each mass, only have at most 2 heavy masses so vectorization not necessary here

         acceleration_particle+=-masses[j]*(position_particle-position_masses[j])/(np.linalg.norm(position_particle-position_masses[j],axis=1).reshape(n_p,1)**3)

    return acceleration_particle 


def acc_particles_calc(position_masses,position_particles,masses):

  """Used to calculate the acceleration of the particles in a vectorised fashion, sums over the forces from each mass. More efficient for a large number of heavy masses(galaxies)."""

  n_m=len(masses)
  n_p=len(position_particles)

  posp_temp=np.full((n_m,n_p,2),position_particles) #repeats position_particles n_m times 
  posm_temp=np.repeat(position_masses,n_p,axis=0).reshape(n_m,n_p,2) #sets up masses for subtraction from the position of each particle

  calc=-masses.reshape(n_m,1,1)*(posp_temp-posm_temp)/((np.linalg.norm(np.ndarray.flatten(posp_temp-posm_temp).reshape(n_p*n_m,2),axis=1).reshape(n_m,n_p))**3).reshape(n_m,n_p,1)
  # calculates the contribution to the acceleration of each particle due to each mass, of form ([[[acc of particle 1 due to mass 1],[acc of particle 2 due to mass1 ]],[[acc of particle 1 due to mass 2],[acc of particle 2 due to mass2 ]]] etc.)
  acceleration_particles=np.sum(calc,axis=0) #sums over all masses

  return acceleration_particles


def orbit_equations(t,q,n_m,n_p,masses):

    """ Returns the equations of the system, n_m heavy masses and n_p particles, masses inputted as an array,
    q (to avoid confusion with y coordinate) is of the form (position of masses, position of particles, velocities of masses, velocities of particles)."""

    position_masses=np.zeros((n_m,2))    #stores the positions of the heavy masses 
    position_particles=np.zeros((n_p,2)) #positions of the test particles
             
    position_masses=q[0:2*n_m].reshape(n_m,2) #easier to work with higher dimensional arrays than q which is "flattened", vectorized
    position_particles=q[2*n_m:2*n_m+2*n_p].reshape(n_p,2)

    acceleration_masses=np.zeros((n_m,2))
    acceleration_particles=np.zeros((n_p,2))  #will hold the accelerations of the particles as vectors

    for i in range(0,n_m):   #calculating the acceleration of a mass due to all other masses, as only 2 masses used, only have to sum 1 force for each mass so vectorization not needed.

        for j in range(0,n_m):

            if i==j:  #don't add force from mass itself

                continue #move to next index in loop without executing below code

            else:

                acceleration_masses[i]+=-masses[j]*(position_masses[i]-position_masses[j])/(np.linalg.norm(position_masses[i]-position_masses[j])**3)  #newton's law of gravitation
    

    acceleration_particles=acc_particle_calc(position_masses,position_particles,masses)  #function acts on all particles i.e. vectorized

    accelerations=np.concatenate((acceleration_masses,acceleration_particles),axis=None) #putting all the accelerations together, axis=None flattens the arrays
    
    velocities=q[2*(n_m+n_p):4*(n_m+n_p)] #must return d(positions)/dt = velocities, vectorized

    return (np.concatenate((velocities,accelerations)))   #since equations must be of form dq/dt =f(t,q) for integrator to work must put the velocity and acceleration arrays together
 

def orbit_solution(t_min,t_max,initial_values,n_m,n_p,masses,n_intervals,integration_method):

    """Takes all the parameters of the problem and returns the solution to the ODES, t_min and t_max
    are the evaluation times, n_m and n_p are the number of massses and particles respectively, n_intervals is the number
    of evaluation points, integration_method determines the algorithm used to solve the DEs."""

    solution=scipy.integrate.solve_ivp(
    fun=orbit_equations,
    t_span=(0,t_max),
    y0=initial_values,
    args=(n_m,n_p,masses),
    t_eval=np.linspace(t_min,t_max,n_intervals,endpoint=False),method=integration_method,atol=1e-6,rtol=1e-3) 

    return solution



