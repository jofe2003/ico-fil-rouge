from numpy import exp

def temperature_standard(x):
    return 100*((-exp(-x/8))+1)+0.01