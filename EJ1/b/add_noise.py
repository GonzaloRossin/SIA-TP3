import random
import math


def binary_noise(prob, data):
    new_data = []
    font = 0
    for row in data:
        new_data.append([])
        for x in row:
            r = random.randint(0,10)
            if(r < prob * 10):
                if x == 1:
                    x = 0
                else: 
                    x = 1
            new_data[font].append(x)
        font = font + 1 
        
    return new_data
            
            
            
def distribution_noise(prob, data):
    new_data = []
    font = 0
    for row in data:
        new_data.append([])
        for x in row:
            r = random.randint(0,10)
            if(r < prob * 10):
                r1 = random.randint(0,8)
                if x == 1:
                    x = 1 - r1 * 0.1
                else: 
                    x = r1 * 0.1
            x = round(x,1)        
            new_data[font].append(x)
        font = font + 1 

    return new_data