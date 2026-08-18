#Matrix Multiplication
def multiplication(A,B):                     
  	row_A = len(A)
 	  row_B = len(B)
  	col_A = len(A[0])
   	col_B = len(B[0])
  	if len(A[0]) != len(B):
    		return "Multiplication is Not Possible."
  	result = []
  	for i in range(row_A):
    		row = []
    		for j in range(col_B):
      			total = 0
      			for k in range (col_A):
        				total += A[i][k] * B[k][j]
      			row.append(total)
    		result.append(row)
  	return result

#Dot Product
def dotproduct(v1,v2):	
  	if len(v1) != len(v2):
    		return "Dot Product is Not possible."
  	total = 0
  	for i in range(len(v1)):
    		total += v1[i] * v2[i]
  	return total

#Transpose
def transpose(A):		
  	num_rows = len(A)
  	num_cols = len(A[0])
  	result = []
  	for i in range(num_cols):
    		new_row = []
    		for j in range(num_rows):
      			new_row.append(A[j][i])
    		result.append(new_row)
  	return result

#Symmetric
def symmetric(A):
    	row = len(A)
    	column = len(A[0])
    	if row != column:
        		return "It is Not Symmetric"
    	for i in range(row):
        		for j in range(column):
            			if A[i][j] != A[j][i]:
                			return "It is Not Symmetric"
    	return "It is Symmetric"

#Triangular
def triangular(A):
  	n = len(A)
  	if n != len(A[0]):
    		return "Not Square Matrix"
  	upper = True
  	lower = True
  	for i in range(n):
    		for j in range(n):
      			if i>j and A[i][j] != 0:
        				upper = False
      			if i<j and A[i][j] !=0:
        				lower = False
  			if upper and lower:
   				 return"Diagonal Matrix"
  			elif upper:
    				return"Upper Triangular Matrix"
  			elif lower:
    				return"Lower Triangular Matrix"
  			else:
    				return"Neither"

#Gaussian random variables (Box–Muller)
import random
import math
import matplotlib.pyplot as plt
def generate_uniform(a, b, n):
    	numbers = []
    	for i in range(n):
        		numbers.append(random.uniform(a, b))
    	return numbers

def generate_gaussian(mean, variance, n):
    	gaussian = []
    	sigma = math.sqrt(variance)
    	while len(gaussian) < n:
        		u1 = random.uniform(-1, 1)
        		u2 = random.uniform(-1, 1)
        		s = u1 * u1 + u2 * u2
        		if s > 0 and s < 1:
            			k = math.sqrt((-2 * math.log(s)) / s)
            			x = u1 * k
            			y = u2 * k
            			gaussian.append(mean + sigma * x)
            			if len(gaussian) < n:
                			gaussian.append(mean + sigma * y)
    	return gaussian
a = float(input("Enter lower limit (a): "))        
b = float(input("Enter upper limit (b): "))            
n = int(input("Enter number of samples: "))           
mean = float(input("Enter Mean (μ): "))                
variance = float(input("Enter Variance (σ²): "))      
uniform_numbers = generate_uniform(a, b, n)
gaussian_numbers = generate_gaussian(mean, variance, n)

# Histogram for Uniform Distribution
plt.figure(figsize=(9,5))
plt.hist(uniform_numbers, bins=20, edgecolor='black')
plt.title("Uniform Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Histogram for Gaussian Distribution
plt.figure(figsize=(9,5))
plt.hist(gaussian_numbers, bins=20, edgecolor='black')
plt.title("Gaussian Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
