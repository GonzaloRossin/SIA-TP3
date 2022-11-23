import random
import numpy as np

def binary_noise(prob, data):
    new_data = []
    line = []
    for row in data:
        #print(len(row))
        for x in row:
            r = random.randint(0,10)
            if(r < prob * 10):
                if x == 1:
                    x = 0
                else: 
                    x = 1        
            line.append(x)
        new_data.append(np.array(line))
        line = []
    
    return np.array(new_data)
            
            
            
def distribution_noise(prob, data):
    new_data = []
    line = []
    for row in data:
        for x in row:
            r = random.randint(0,10)
            if(r < prob * 10):
                r1 = random.randint(0,8)
                if x == 1:
                    x = 1 - r1 * 0.1
                else: 
                    x = r1 * 0.1
            x = round(x,1)        
            line.append(x)
        new_data.append(np.array(line))
        line = []

    return np.array(new_data)