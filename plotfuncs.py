import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from initfuncs import init_equations_two
from initfuncs import calc_parabolic
from testingfuncs import captured_count
from testingfuncs import tail_count
from orbitfuncs import orbit_solution
from scipy import stats

def snapshot_plot(t_snapshots,solution_time,solution_pos,n_m,n_p,x_lim,y_lim):

    """Used to plot the positions of heavy masses and test particles at the times inputted"""

    for t in t_snapshots:
    
        snapshot_index=np.abs(solution_time-t).argmin()  #finds the closest times to the inputted ones where the solution was actually calculated 
        print("Time of snaphot= {}".format(solution_time[snapshot_index])) #prints the time that the positions are plotted
    
        plt.scatter((solution_pos[2*n_m:2*(n_m+n_p):2]).T[snapshot_index],(solution_pos[2*n_m+1:2*(n_m+n_p):2]).T[snapshot_index],s=1,c='black') #plot positions of particles, vectorized
        # 2*n_m:2*(n_m+n_p):2 means the indexes from 2*n_m up to 2*(n_m+n_p) in step sizes of 2 i.e this gives all the x coordinates of the particles

        plt.scatter((solution_pos[0:2*n_m:2]).T[snapshot_index],(solution_pos[1:2*n_m:2]).T[snapshot_index],s=50,c='black')  #plot positions of masses, vectorized
        
        plt.xlim([-x_lim,x_lim])
        plt.ylim([-y_lim,y_lim])
        plt.show()


def snapshot_plot_samefig(t_snapshots,solution_time,solution_pos,n_m,n_p,x_lim,y_lim,n_rows,n_cols,orbit_type):

    """Used to plot the positions of heavy masses and test particles at the times inputted on the same figure. Orbit_type = parabolic, bound or hyperbolic"""

    fig,ax = plt.subplots(n_rows,n_cols,figsize=(9,9))
    fig.set_tight_layout(True)
    col_count=0
    row_count=0

    for t in t_snapshots:

        snapshot_index=np.abs(solution_time-t).argmin()  #finds the closest times to the inputted ones where the solution was actually calculated 
        print("Time of snapshot ={}".format(solution_time[snapshot_index])) #prints the time that the positions are plotted
    
        ax[row_count][col_count].scatter((solution_pos[2*n_m:2*(n_m+n_p):2]).T[snapshot_index],(solution_pos[2*n_m+1:2*(n_m+n_p):2]).T[snapshot_index],s=1,c='black')
        ax[row_count][col_count].scatter((solution_pos[0:2*n_m:2]).T[snapshot_index],(solution_pos[1:2*n_m:2]).T[snapshot_index],s=50,c='black')  #plot positions of masses, vectorized
        ax[row_count][col_count].set_xlim([-x_lim,x_lim])
        ax[row_count][col_count].set_ylim([-x_lim,x_lim])
        ax[row_count][col_count].set_xlabel("x")
        ax[row_count][col_count].set_ylabel("y")
        ax[row_count][col_count].set(aspect='equal')

        if orbit_type=='bound':
            ax[row_count][col_count].set_title("Equal masses in a bound orbit with \n {} particles after {} time units".format(n_p,t))
        if orbit_type=='hyperbolic':
            ax[row_count][col_count].set_title("Equal masses in a hyperbolic orbit with \n {} particles after {} time units".format(n_p,t))
        if orbit_type=='parabolic':
            ax[row_count][col_count].set_title("Equal masses in a parabolic orbit with \n {} particles after {} time units".format(n_p,t))
        
        col_count+=1
        if(col_count==n_cols):
            row_count+=1
            col_count=0

    fig.tight_layout()    
    plt.show()


def trajectory_plot(n_p,n_m,solution_pos,x_lim,y_lim):

    """ Plot the trajectories of the heavy masses and test particles"""

    for i in range(0,n_p):

        plt.plot(solution_pos[2*n_m+2*i],solution_pos[2*n_m+1+2*i],c='black') #Plotting position of particles
    
    for i in range(0,n_m):

        plt.plot(solution_pos[2*i],solution_pos[2*i+1],c='black')
        #plt.plot(solution_pos[2*i],solution_pos[2*n_m+2*i+1]) #x vx
    
    plt.xlim([-x_lim,x_lim])
    plt.ylim([-y_lim,y_lim])
    plt.show()


def animate_plot(solution_pos,x_lim,y_lim,n_m,n_p,n_frames):

    """Produces an animation of the movement of the heavy masses and test particles"""
    
    fig=plt.figure() #setting up the figure to plot on
    ax=fig.add_subplot(111)
    ax.set_xlim(-x_lim,x_lim)
    ax.set_ylim(-y_lim,y_lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("120 particles over 75 time units")

    
    lines=np.zeros(n_p+n_m,dtype=object) #plot different line for each test particle and heavy mass

    for i in range(n_m):

        lines[i],=ax.plot([],[],'o',c='black',markersize=8) #lines for the masses, really points

    for i in range(n_p):

        lines[n_m+i],=ax.plot([],[],'o',c='black',markersize=2) #points for the particles

    def animate(i):
        """returns the lines for each particle and heavy mass so it can be animated"""
        
        for j in range(n_m+n_p):

            lines[j].set_xdata((solution_pos[2*j])[i])
            lines[j].set_ydata((solution_pos[2*j+1])[i])

            #lines[:n_p+n_m].set_xdata((solution.y[0:2*(n_p+n_m-1):2])[i])
            #lines[:n_p+n_m].set_ydata((solution.y[1:2*(n_p+n_m):2])[i])

        return lines

    ani=animation.FuncAnimation(fig,animate,frames=int(n_frames),interval=1)

    return ani
     

def snapshot_plot_masses(f,r_min,radii,number_radii,t_max,n_rows,n_cols,time,x_lim,y_lim,x_separation):

    """Calculates and plots the positions of unequal masses in a parabolic orbit of a given distance of closest approach and initial x separation of the masses.
    Same x separation results in the same initial separation in this case. The mass fractions (=mass for unit mass of other galaxy) of the perturbing galaxy are passed as an array""" 

    n_m=2
    n_p=np.sum(number_radii)

    fig,ax = plt.subplots(n_rows,n_cols,figsize=(9,9))
    fig.set_tight_layout(True)

    col_count=0 #used to plot the figures from left to right across columns and then down the rows
    row_count=0

    for i in range(len(f)):
    
        x0=-x_separation/(1+1/f[i]) #ensures that the initial separation of the heavy masses is the same 
        
        masses=np.array([1,f[i]])
        mass1_position0,mass1_velocity0,mass2_position0,mass2_velocity0=calc_parabolic(f[i],r_min,x0) #calculating the initial positions and velocities of the masses for a parabolic orbit

        initial_values=init_equations_two(radii,number_radii,mass1_position0,mass2_position0,mass1_velocity0,mass2_velocity0)
        
        solution=orbit_solution(0,t_max,initial_values,n_m,n_p,masses,10000,'DOP853')
        solution_time=solution.t
        solution_pos=solution.y

        snapshot_index=np.abs(solution_time-time).argmin() #calculating the closest time to the one inputted at which the solution was actually calculated
        
        ax[row_count][col_count].scatter((solution_pos[2*n_m:2*(n_m+n_p):2]).T[snapshot_index],(solution_pos[2*n_m+1:2*(n_m+n_p):2]).T[snapshot_index],s=1,c='black') #plotting positions of particles
        ax[row_count][col_count].scatter((solution_pos[0:2*n_m:2]).T[snapshot_index],(solution_pos[1:2*n_m:2]).T[snapshot_index],s=50,c='black')  #plot positions of masses, vectorized
        ax[row_count][col_count].set_xlim([-x_lim,x_lim])
        ax[row_count][col_count].set_ylim([-x_lim,x_lim])
        ax[row_count][col_count].set_xlabel("x")
        ax[row_count][col_count].set_ylabel("y")
        ax[row_count][col_count].set(aspect='equal') #square plots
        ax[row_count][col_count].set_title("Perturbing galaxy of mass {} units \n orbiting unit mass with {} particles \n after {} time units".format(f[i],n_p,time))
     
        col_count+=1
        if(col_count==n_cols):
            row_count+=1
            col_count=0
            
    plt.show()


def snapshot_plot_distances(r_mins,masses,radii,number_radii,t_max,n_rows,n_cols,time,x_lim,y_lim,initial_separation):
    
    """Calculates and plots the positions of equal masses in a parabolic orbit for different closest approach distances, initial separation of masses is an input argument.
     The distances of closest approach are passed as an array.""" 

    n_m=2
    n_p=np.sum(number_radii) #number of particles

    fig,ax = plt.subplots(n_rows,n_cols,figsize=(9,9)) #setting up the figure
    fig.set_tight_layout(True)

    col_count=0 #plot left to right and then downwards
    row_count=0

    for i in range(len(r_mins)):

        x0=(initial_separation-2*r_mins[i])/2 #ensures the initial separation is constant
        
        mass1_position0,mass1_velocity0,mass2_position0,mass2_velocity0=calc_parabolic(1,r_mins[i],x0)
        
        initial_values=init_equations_two(radii,number_radii,mass1_position0,mass2_position0,mass1_velocity0,mass2_velocity0)
        
        solution=orbit_solution(0,t_max,initial_values,n_m,n_p,masses,10000,'DOP853')
        solution_time=solution.t
        solution_pos=solution.y
        snapshot_index=np.abs(solution_time-time).argmin()
        
        ax[row_count][col_count].scatter((solution_pos[2*n_m:2*(n_m+n_p):2]).T[snapshot_index],(solution_pos[2*n_m+1:2*(n_m+n_p):2]).T[snapshot_index],s=1,c='black')
        ax[row_count][col_count].scatter((solution_pos[0:2*n_m:2]).T[snapshot_index],(solution_pos[1:2*n_m:2]).T[snapshot_index],s=50,c='black')  #plot positions of masses, vectorized
        ax[row_count][col_count].set_xlim([-x_lim,x_lim])
        ax[row_count][col_count].set_ylim([-x_lim,x_lim])
        ax[row_count][col_count].set_xlabel("x")
        ax[row_count][col_count].set_ylabel("y")
        ax[row_count][col_count].set(aspect='equal')
        ax[row_count][col_count].set_title("Equal masses in a parabolic orbit \n with distance of closest approach = {} units \n with {} particles after {} time units".format(r_mins[i],n_p,time))
     
        col_count+=1
        if(col_count==n_cols):
            row_count+=1
            col_count=0
            
    plt.show()


def captured_plot_masses(f,r_min,x_separation,radii,number_radii,t_max,time):

    """Plots the fraction of particles captured for different masses of the perturbing galaxy at a given time."""

    n_p=np.sum(number_radii)
    n_m=2

    captured=np.zeros(len(f)) #will hold the number of particles captured

    for i in range(len(f)):

        x0=-x_separation/(1+1/f[i]) #ensures same initial separation of masses

        masses=np.array([1,f[i]])
        mass1_position0,mass1_velocity0,mass2_position0,mass2_velocity0=calc_parabolic(f[i],r_min,x0)

        initial_values=init_equations_two(radii,number_radii,mass1_position0,mass2_position0,mass1_velocity0,mass2_velocity0) #setting the initial values for input into the integrator

        solution=orbit_solution(0,t_max,initial_values,n_m,n_p,masses,10000,'DOP853')
        
        captured[i]=captured_count(solution.y,solution.t,time,n_m,n_p) #calculates the number of particles with y>0

    regression=stats.linregress(np.log10(f),np.log10(captured/n_p)) #calculates line of best fit and y intercept
    print("Gradient= {}".format(regression.slope))
    plt.xlabel("Mass of perturbing galaxy")
    plt.ylabel("Fraction of particles captured")
    plt.title("Fraction of particles captured by the perturbing galaxy \n as a function of its mass")
    plt.plot(f,captured/n_p,c='black')
    #plt.loglog(f,10**(regression.intercept)*(f)**regression.slope) #superposes the best fit line on a log log plot
    plt.show()

    
def tails_plot_masses(f,r_min,x_separation,radii,number_radii,t_max,time):

    """Plots the fraction of particles in the tidal tail for different masses of the perturbing galaxy at a given time."""

    n_p=np.sum(number_radii)
    n_m=2

    tails=np.zeros(len(f)) #holds the number of particles in the tail

    for i in range(len(f)):

        x0=-x_separation/(1+1/f[i]) #ensures same initial separation of the masses

        masses=np.array([1,f[i]])
        mass1_position0,mass1_velocity0,mass2_position0,mass2_velocity0=calc_parabolic(f[i],r_min,x0) 

        initial_values=init_equations_two(radii,number_radii,mass1_position0,mass2_position0,mass1_velocity0,mass2_velocity0)
        solution=orbit_solution(0,t_max,initial_values,n_m,n_p,masses,10000,'DOP853')

        tails[i]=tail_count(radii,n_p,time,solution.y,solution.t) #calculates the number of particles with y<0 that are beyond the initial max radius of particles from the mass

    plt.xlabel("Mass of perturbing galaxy")
    plt.ylabel("Fraction of particles in tail")
    plt.title("Fraction of particles in the tidal tail \n as a function of the mass of the perturbing galaxy")
    plt.plot(f,tails/n_p,c='black')
    plt.show()


def captured_plot_distances(r_mins,initial_separation,radii,number_radii,t_max,time):

    """Plots the fraction of particles captured for different distances of closest approach of the masses at a given time."""

    masses=np.array([1,1])
    n_p=np.sum(number_radii)
    n_m=2

    captured=np.zeros(len(r_mins)) #holds the number of particles captured for different distances of closest approach

    for i in range(len(r_mins)):

        x0=(initial_separation-2*r_mins[i])/2 #ensures that the initial separation of the masses is constant for all distances of closest approach
        
        mass1_position0,mass1_velocity0,mass2_position0,mass2_velocity0=calc_parabolic(1,r_mins[i],x0)

        initial_values=init_equations_two(radii,number_radii,mass1_position0,mass2_position0,mass1_velocity0,mass2_velocity0)
        solution=orbit_solution(0,t_max,initial_values,n_m,n_p,masses,10000,'DOP853')

        captured[i]=captured_count(solution.y,solution.t,time,n_m,n_p) #number of particles captured 
    
    #regression=stats.linregress(np.log10(r_mins),np.log10(captured/n_p))
    regression=stats.linregress(r_mins,captured/n_p)
    print("Gradient= {}".format(regression.slope))
    print("R^2 ={}".format(regression.rvalue**2))

    plt.xlabel("Distance of closest approach")
    plt.ylabel("Fraction of particles captured")
    plt.title("Fraction of particles captured by the perturbing galaxy \n as a function of the distance of closest approach")
    plt.plot(r_mins,captured/n_p,c='black')
    plt.plot(r_mins,regression.intercept+regression.slope*r_mins)
    #plt.loglog(r_mins,captured/n_p,c='black')
    #plt.loglog(r_mins,10**(regression.intercept)*(r_mins)**regression.slope)
    plt.show()

    
def tails_plot_distances(r_mins,initial_separation,radii,number_radii,t_max,time):

    """Plots the fraction of particles in the tidal tail for different distances of closest approach of the masses at a given time."""

    masses=np.array([1,1])
    n_p=np.sum(number_radii)
    n_m=2

    tails=np.zeros(len(r_mins)) #holds the number of particles in the tidal tail for different distances of closest approach

    for i in range(len(r_mins)):

        x0=(initial_separation-2*r_mins[i])/2
        
        mass1_position0,mass1_velocity0,mass2_position0,mass2_velocity0=calc_parabolic(1,r_mins[i],x0)

        initial_values=init_equations_two(radii,number_radii,mass1_position0,mass2_position0,mass1_velocity0,mass2_velocity0)
        solution=orbit_solution(0,t_max,initial_values,n_m,n_p,masses,10000,'DOP853')

        tails[i]=tail_count(radii,n_p,time,solution.y,solution.t) #number of particles in the tail

    regression=stats.linregress(r_mins,tails/n_p)
    #regression=stats.linregress(np.log10(r_mins),np.log10(tails/n_p))
    print("Gradient= {}".format(regression.slope))
    print("R^2 ={}".format(regression.rvalue**2))
    
    plt.xlabel("Distance of closest approach")
    plt.ylabel("Fraction of particles in tail")
    plt.title("Fraction of particles in the tidal tail \n as a function of the distance of closest approach")
    plt.plot(r_mins,tails/n_p,c='black')
    plt.plot(r_mins,regression.intercept+regression.slope*r_mins)
    plt.show()


def runtime_average(t_min,t_max,initial_values,n_m,n_p,masses,n_intervals,method,n_runs):

    """Returns the average runtime of n_runs runs of the solution function given the inputted parameters"""

    start=time.time() #starts timing

    for i in range(n_runs): #calculate solution n_runs times

        orbit_solution(t_min,t_max,initial_values,n_m,n_p,masses,n_intervals,method)
    
    end=time.time()

    print("Average of {} runs = ".format(n_runs),(end-start)/n_runs)