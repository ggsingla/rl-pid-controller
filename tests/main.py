from controller import PID

from tclab import clock, setup, Historian, Plotter
import pandas as pd

TCLab = setup(connected=False, speedup=10)

controller = PID(0.4, 0.66, 0.01)        # create pid control
controller.send(None)              # initialize

tfinal = 800

with TCLab() as lab:
    h = Historian([('SP', lambda: SP), ('IV', lambda: lab.T1), ('MV', lambda: MV)])
    p = Plotter(h, tfinal)
    T1 = lab.T1
    for t in clock(tfinal, 2):
        SP = T1 if t < 50 else 50           # get setpoint
        PV = lab.T1                         # get measurement
        MV = controller.send([t, PV, SP])   # compute manipulated variable
        lab.U1 = MV                         # apply 
        p.update(t)                         # update information display

    h.to_csv('pid.csv')
    data = pd.read_csv('pid.csv')
    data[['SP','IV', 'MV']].plot(grid=True)
