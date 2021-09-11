# Functions to find percentiles (25, 50, 75)

def first_qr(data):
    return data.quantile(0.25)

def median(data):
    return data.quantile(0.5)

def third_qr(data):
    return data.quantile(0.75)
