import SimplePend2D as SP2D
import SimplePend3D as SP3D
import numpy as np
#import matplotlib.pyplot as plt
from time import time

if __name__ == '__main__':
    start_time = time()
    parameters = {"Debugmode":True,
                  "Saveplots":True,
                  "Showplots":True,
                  "Nframes":400,
                  "tmax":40,
                  "framerate":20,
                  "theta_0":np.pi/8,
                  "dtheta_0":0.0,
                  "phi_0":np.pi/4,
                  "dphi_0":0.0,
                  "l":1.5,
                  "m":1.5,
                  "k":0.1,
                  "option": 'Euler',
                  }

    def test_2D():
        SP2D.set_constants(**parameters)
        SP2D.generate_states(minimal = False)
        SP2D.animate_positions()
        SP2D.plot_energies()
        print(str(SP2D.get_period()))
        SP2D.animate_phasespace(trail=True)
        #SP2D.small_angle_errors(N = 100, thetazero = np.pi/10,thetamax = np.pi/4)

    def test_3D():
        SP3D.set_constants(**parameters)
        SP3D.generate_states(minimal = False)
        #SP3D.animate_positions()
        #SP3D.plot_energies()
        print(str(SP3D.get_periods()))

    #test_2D()
    test_3D()
    
    
    print(f"main() done in {time()-start_time} s")