import pybullet as p
import os


class deathTrap:
    def __init__(self, client, base):
        f_name = os.path.join(os.path.dirname(__file__), 'deathTrapOfDeath.urdf')
        self.deathTrap = client.loadURDF(fileName=f_name,
                   basePosition=[base[0], base[1], 0])


