

import numpy as np

data = open('log', 'r').read().split('\n')

inertia = []
viscosity = []
buoyancy = []
coriolis = []
pressure = []
ageoStrophic = []
 


for lines in data:

    if 'Forces' in lines:
        for idx, values in enumerate(lines.split('->')[1].split(',')):

            if idx == 0:
                viscosity.append(float(values.split(':')[1]))

            if idx == 1:
                coriolis.append(float(values.split(':')[1]))

            if idx == 2:
                inertia.append(float(values.split(':')[1]))

            if idx == 3:
                buoyancy.append(float(values.split(':')[1]))

            if idx == 4:
                pressure.append(float(values.split(':')[1]))
       
            if idx == 5:
                ageoStrophic.append(float(values.split(':')[1]))


    elif 'Vorticty' in lines:
        print(lines)


print('inertia = np.array({})'.format(inertia))
print('viscosity = np.array({})'.format(viscosity))
print('coriolis = np.array({})'.format(coriolis))
print('buoyancy = np.array({})'.format(buoyancy))
print('pressure = np.array({})'.format(pressure))
print('ageoStrophic = np.array({})'.format(ageoStrophic))
