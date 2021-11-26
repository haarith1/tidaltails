import numpy as np

def set_particles(radii,number_radii,origin,velocity,direction):

    """Sets the initial positions and velocities of particles, radii is an array holding the radii on which to put particles and number_radii holds the number of particles on each radius,
    direction=+1 for anticlockwise and -1 for clockwise. Origin and velocity are the position and velocity respectively of the mass which the particles orbit."""
    
    theta=2*np.pi/number_radii  #used to calculate equally spaced particles on each radius

    coords=np.zeros((np.sum(number_radii), 2))  #hold the initial coordinates and velocities of each particle 
    velocities=np.zeros((np.sum(number_radii), 2))

    for i in range(len(radii)):
        
        for j in range(number_radii[i]):

            coords[np.sum(number_radii[0:i])+j]=origin+np.array([radii[i]*np.cos(j*theta[i]),radii[i]*np.sin(j*theta[i])]) #setting coordinates of equally spaced particles at each radius
            velocities[np.sum(number_radii[0:i])+j]=velocity+direction*np.array([-radii[i]**-0.5*np.sin(j*theta[i]),radii[i]**-0.5*np.cos(j*theta[i])]) #setting velocities of particles,
            #must add on velocity of mass which the particles are orbitting

    return (np.ndarray.flatten(coords),np.ndarray.flatten(velocities))  #return a flattened array so that these values are compatible with the ode integrator
    

def calc_parabolic(f,r_min,x0):
    
    """Takes the distance of closest approach of 2 masses, the mass fraction f=M2/M1 and the starting x coordinate
   of one of them and returns the initial position and velocity vectors of the masses."""
    
    c=2*r_min #a constant used in the calcualtion of the parabolic orbit shape
    y0=(f/(1+f))*(c**2+2*c*x0*(1+f)/f)**0.5 #the initial y coordinate of the mass
    vx0=-((2*f**3)/((1+f)**2*(1+c**2*f**2/((1+f)**2*y0**2))*(x0**2+y0**2)**0.5))**0.5 #initial velocity in x direction of the mass
    vy0=c*f*vx0/((1+f)*y0) #initial velocity in y direction of the mass

    mass1_position0=np.array([x0,y0]) #position of mass1 etc.
    mass1_velocity0=np.array([vx0,vy0]) 
    mass2_position0=(-1/f)*mass1_position0 
    mass2_velocity0=-(1/f)*mass1_velocity0

    return mass1_position0,mass1_velocity0,mass2_position0,mass2_velocity0


def init_equations_two(radii,number_radii,mass1_position0,mass2_position0,mass1_velocity0,mass2_velocity0):

    """Initalises the positions and velocities of 2 heavy masses and the particles."""

    particle_positions1, particle_velocities1 = set_particles(radii,number_radii,mass1_position0,mass1_velocity0,1) #set initial coords and velocities of all particles around mass 1
    initial_values=np.concatenate((mass1_position0,mass2_position0,particle_positions1,mass1_velocity0,mass2_velocity0,particle_velocities1)) #puts all values together for input into ODE integrator
    
    return initial_values


def init_equations_one(radii,number_radii,mass1_position0,mass1_velocity0):

    """Initalises the positions and velocities of 1 heavy mass and the particles."""

    particle_positions1, particle_velocities1 = set_particles(radii,number_radii,mass1_position0,mass1_velocity0,1) #set initial coords and velocities of all particles, mass 1
    initial_values=np.concatenate((mass1_position0,particle_positions1,mass1_velocity0,particle_velocities1))

    return initial_values


def init_equations_masses(mass1_position0,mass2_position0,mass1_velocity0,mass2_velocity0):

    """Initalises the positions and velocities of 2 heavy masses and no particles."""
    
    initial_values=np.concatenate((mass1_position0,mass2_position0,mass1_velocity0,mass2_velocity0))

    return initial_values