import numpy as np
from typing import Tuple
from .skeleton import Environment


class Cartpole(Environment):
    """
    The cart-pole environment as described in the 687 course material. This
    domain is modeled as a pole balancing on a cart. The agent must learn to
    move the cart forwards and backwards to keep the pole from falling.

    Actions: left (0) and right (1)
    Reward: 1 always

    Environment Dynamics: See the work of Florian 2007
    (Correct equations for the dynamics of the cart-pole system) for the
    observation of the correct dynamics.
    """

    def __init__(self):
        self._name = "Cartpole"
        
        # TODO: properly define the variables below
        self._action = None
        self._reward = 1.0
        self._isEnd = False
        self._gamma = 1.0

        # define the state # NOTE: you must use these variable names
        self._x = 0.0  # horizontal position of cart
        self._v = 0.0  # horizontal velocity of the cart
        self._theta = 0.0  # angle of the pole
        self._dtheta = 0.0  # angular velocity of the pole

        # dynamics
        self._g = 9.8  # gravitational acceleration (m/s^2)
        self._mp = 0.1  # pole mass
        self._mc = 1.0  # cart mass
        self._l = 0.5  # (1/2) * pole length
        self._dt = 0.02  # timestep
        self._t = 0.0  # total time elapsed  NOTE: you must use this variable

    @property
    def name(self)->str:
        return self._name

    @property
    def reward(self) -> float:
        # TODO
        return self._reward
#        pass

    @property
    def gamma(self) -> float:
        # TODO
        return self._gamma
#        pass

    @property
    def action(self) -> int:
        # TODO
        return self._action
        pass

    @property
    def isEnd(self) -> bool:
        # TODO
        return self._isEnd
        pass

    @property
    def state(self) -> np.ndarray:
        # TODO
        return np.array([self._x,self._v,self._theta,self._dtheta])
    
#        pass

    def nextState(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Compute the next state of the pendulum using the euler approximation to the dynamics
        """
        # TODO
                        
        self._action = action
        
        F=0.0
        if(self._action==0):
            F = -10
        elif(self._action==1):
            F = 10
        
        x_dot = self._v
        
        omega_dot_num_a = self._g*np.sin(self._theta)
        omega_dot_num_b = np.cos(self._theta)
        omega_dot_num_c = (-F - self._mp*self._l*(self._dtheta**2)*np.sin(self._theta))/(self._mc + self._mp)
        omega_dot_den = self._l*(4/3 - (self._mp*(np.cos(self._theta)**2))/(self._mc + self._mp))
        
        omega_dot = (omega_dot_num_a + omega_dot_num_b*omega_dot_num_c)/omega_dot_den
        
        v_dot_num = (F + self._mp*self._l*((self._dtheta**2)*np.sin(self._theta) - omega_dot*np.cos(self._theta)))
        v_dot_den = (self._mc + self._mp)
        
        v_dot = v_dot_num/v_dot_den
        
        theta_dot = self._dtheta

        self._x = self._x + self._dt*x_dot
        self._v = self._v + self._dt*v_dot
        self._theta = self._theta + self._dt*theta_dot
        self._dtheta = self._dtheta + self._dt*omega_dot
        
        return np.array([self._x,self._v,self._theta,self._dtheta])
        
#        pass

    def R(self, state: np.ndarray, action: int, nextState: np.ndarray) -> float:
        # TODO
        return self._reward
#        pass

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        takes one step in the environment and returns the next state, reward, and if it is in the terminal state
        """
        # TODO
        
#        print("action: ",action)
#        print(self._t)
        
        state = np.array([self._x,self._v,self._theta,self._dtheta])
        next_state = self.nextState(np.array([self._x,self._v,self._theta,self._dtheta]), action)
        self._reward = self.R(state,action,next_state)
        self._t+=self._dt
        self._isEnd = self.terminal()
        
        return(next_state,self._reward,self._isEnd)
        
#        pass

    def reset(self) -> None:
        """
        resets the state of the environment to the initial configuration
        """
        # TODO
        
        self._action = None
        self._reward = 1.0
        self._isEnd = False
        self._gamma = 1.0

        # define the state # NOTE: you must use these variable names
        self._x = 0.0  # horizontal position of cart
        self._v = 0.0  # horizontal velocity of the cart
        self._theta = 0.0  # angle of the pole
        self._dtheta = 0.0  # angular velocity of the pole

        # dynamics
        self._g = 9.8  # gravitational acceleration (m/s^2)
        self._mp = 0.1  # pole mass
        self._mc = 1.0  # cart mass
        self._l = 0.5  # (1/2) * pole length
        self._dt = 0.02  # timestep
        self._t = 0.0  # total time elapsed  NOTE: you must use this variable
        
#        pass

    def terminal(self) -> bool:
        """
        The episode is at an end if:
            time is greater that 20 seconds
            pole falls |theta| > (pi/12.0)
            cart hits the sides |x| >= 3
        """
        # TODO
        if(self._t > 20-1e-6 or self._x<=-3 or self._x>=3 or self._theta > np.pi/12 or self._theta < -np.pi/12 ):
            self._isEnd = True
        else:
            self._isEnd = False
            
        return self._isEnd
#        pass
