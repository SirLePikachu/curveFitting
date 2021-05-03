import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg

#program for simple ml-fitting(polynomial) using gradient descent



#step-size for differentiation
delta = 0.0001
beta = 0.00000000000001
max = 20000
#gamma = 0.001

#training data and initial parameters
x = np.array([0,1,2,3.5,4,5])
t = np.array([-1,1,2.5,4,-3,-4])
initw = np.array([0,1,1,-0.3,0.3,-0.1])






#errorfunction of an polynomial of grade len(w) with input data (x,t) and parameters w
def errorFkt(x,t,w):
    F = 0
    for i in range(len(x)):
        poly = nPolynomial(x[i],w)
        F = F + (poly - t[i])**2
        #print(F)
        #print(F)
    return F





#calculates polynomial of grade len(w) for one value of x
def nPolynomial(x,w):
    poly = 0
    for j in range(len(w)):
        poly = poly + w[j] * x ** j
    return poly


#minimization of an errorfunction with training data (x,t) and parameters w
#using simple gradient descent using the  rule obtained from wikipedia
#and the step width by the line search method described also there
#see: https://bit.ly/3tfQhY6
def gradDescent(x,t,w):
    h = np.zeros(np.shape(w))
    n = 1
    R = errorFkt(x,t,w)
    R_alt = 0
    w_alt = w-0.001
    grad_alt = 0
    #calculating gradient of the error function
    while n < max and  abs(R - R_alt) > beta:
        g = np.zeros(np.shape(w))
        for l in range(len(w)):
            h[l] = delta
            g[l] = errorFkt(x,t,w+h) - errorFkt(x,t,w)
            h[l] = 0

        grad_F = g/delta

        #estimating step size gamma
        q = np.abs(np.dot((w-w_alt),(grad_F-grad_alt)))
        gamma = q/np.linalg.norm(grad_F-grad_alt)**2
        grad_alt = grad_F
        w_alt = w
        w = w - gamma*grad_F
        n = n+1
        R_alt = R
        R = errorFkt(x,t,w)
    if n < max:
        print("the fit converged after {} iterations".format(n))
    else:
        print("the fit did not converge")
    print("The residual sum of squares is {}".format(R))
    print("The final parameters are:\n{}".format(w))
    return w, R , n


#array for plotting of fitted function
lel = np.linspace(0,1,2000)

#calculating parameters
w , R, n = gradDescent(x/5,t/4,initw)


#printing best fit parameters and plotting data and fit
plt.plot(lel,nPolynomial(lel,w))
plt.scatter(x/5,t/4)
plt.show()
