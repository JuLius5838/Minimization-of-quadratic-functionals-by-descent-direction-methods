import numpy as np
import matplotlib.pyplot as plt 
import time

x = np.loadtxt("dataP.dat")
y = np.loadtxt("dataQ.dat")

def display_data(x,y, xtry1, xtry2, xtry3):
	plt.figure("2.1.2 Ajustement linéaire")
	
	c1 = xtry1[0,:]
	c2 = xtry1[1,:]
	c1_2 = xtry2[0,:]
	c2_2 = xtry2[1,:]
	c1_3 = xtry3[0,:]
	c2_3 = xtry3[1,:]
	Q1 = list()
	Q2 = list()
	Q3 = list()
	for i in range (len(x)):
		q1 = c1 + c2*x[i]
		q2 = c1_2 + c2_2*x[i]
		q3 = c1_3 + c2_3*x[i]
		Q1.append(q1)
		Q2.append(q2)
		Q3.append(q3)

	plt.subplot(2,2,1)
	plt.plot(x,Q1, color = 'green', label='approximation linéaire gradient à pas fixe')
	plt.plot(x,y, color ='black', marker ='x', linestyle='none')
	plt.xlabel("Age des enfants")
	plt.ylabel("Taille des enfants")
	plt.title("Taille en fonction de l'age")
	plt.legend()


	plt.subplot(2,2,2)
	plt.plot(x,Q2, color = 'orange', label='approximation linéaire gradient à pas optimal')
	plt.plot(x,y, color ='black', marker ='x', linestyle='none')
	plt.xlabel("Age des enfants")
	plt.ylabel("Taille des enfants")
	plt.title("Taille en fonction de l'age")
	plt.legend()


	plt.subplot(2,2,3)
	plt.plot(x,Q3, color = 'red', label='approximation linéaire gradient conjugué')
	plt.plot(x,y, color ='black', marker ='x', linestyle='none')
	plt.xlabel("Age des enfants")
	plt.ylabel("Taille des enfants")
	plt.title("Taille en fonction de l'age")
	plt.legend()


	plt.subplot(2,2,4)
	plt.plot(x,Q1, color = 'green', label='approximation linéaire gradient à pas fixe')
	plt.plot(x,Q2, color = 'orange', label='approximation linéaire gradient à pas optimal')
	plt.plot(x,Q3, color = 'red', label='approximation linéaire gradient conjugué')
	plt.plot(x,y, color ='black', marker ='x', linestyle='none')
	plt.xlabel("Age des enfants")
	plt.ylabel("Taille des enfants")
	plt.title("Taille en fonction de l'age")
	plt.legend()
	plt.show()


def X_build(x,y):
	n = len(y)
	q = np.zeros((n,1))
	X = np.ones((n,2))

	for i in range(0,n):
		q[i,:] = y[i]
		X[i,1] = x[i]

	XTX = (np.transpose(X))@X
	XTq = (np.transpose(X))@q

	return X,q,XTX,XTq

def fixed_step_gradient(x,y,x0,rho,epsilon,itmax):
	X,q,A,b = X_build(x,y)
	x = x0
	i = 0
	xit = [x]
	r = A@x - b
	while(np.linalg.norm(r) >epsilon and i < itmax):
		d = -r
		alpha = rho
		x = x + alpha*d
		xit.append(x)
		r = A@x - b
		i += 1

	return(x,i,xit)

def optimal_step_gradient(x,y,x0,epsilon,itmax):
	X,q,A,b = X_build(x,y)
	x = x0
	i = 0
	xit = [x]
	r = A@x - b
	while(np.linalg.norm(r)>epsilon and i<itmax):
		
		d = -r
		rho = (np.transpose(r)@r)/(np.transpose(r)@A@r)
		x = x+ rho*d
		xit.append(x)
		r = A@x -b
		i += 1
	return(x,i,xit)

def conjugued_gradient(x,y,x0, epsilon, itmax):
	X,q,A,b = X_build(x,y)
	x = x0
	i = 1
	xit = [x]
	r = A@x - b
	while(np.linalg.norm(r)>epsilon and i<itmax):
		if (i ==1):
			d = -r
			d_1 = d
		else:
			beta = ((np.transpose(r)@r))/((np.transpose(r_1)@r_1)) 
			d = -r + beta*d_1
			d_1 = d
		rho = (np.transpose(r)@r)/(np.transpose(d)@A@d)
		x = x+ rho*d
		xit.append(x)
		r_1 = r
		r = A@x - b
		i+=1

	return(x,i,xit)

def level_map(xit_, xit2_, xit3_):

	X,q,Z,w = X_build(x,y)
	s = np.transpose(q)@q


	x_axis = np.arange(-10,10.5,0.5)
	y_axis = np.arange(-10,10.5,0.5)
	xx_axis, yy_axis = np.meshgrid(x_axis,y_axis)
	
	F = list()
	f = (1/2)*(Z[0,0]*xx_axis**2 + 2*Z[0,1]*xx_axis*yy_axis+Z[1,1]*yy_axis**2 - 2*(w[0]*xx_axis+w[1]*yy_axis)+s)
	plt.figure("Level map")
	plt.plot(xit_[:,0,0], xit_[:,1,0], color='gold', label='direction de descente pas fixe')
	plt.plot(xit2_[:,0,0], xit2_[:,1,0], color='tomato', marker = "o", label='direction de descente pas optimal')
	plt.plot(xit3_[:,0,0], xit3_[:,1,0], color='maroon', marker = "v", label='direction de descente gradient conjugué')
	LM = plt.contour(x_axis,y_axis,f,30, cmap = "gnuplot")
	plt.title("ligne de niveaux de F (Repère canonique)")
	plt.colorbar(LM)
	plt.legend()

	fig = plt.figure("gradient")
	quiv = plt.quiver(xx_axis, yy_axis, f, cmap = "gnuplot")
	plt.colorbar(quiv)
	plt.title("Gradient de F($c_1, c_2$)")


	fig = plt.figure("Level map 2")
	ax3d = fig.gca(projection = "3d")
	LM_2 = ax3d.plot_surface(xx_axis, yy_axis,f, cmap = "gnuplot", edgecolor ='none')
	plt.title("Surface représentative de f (Repère canonique)")
	plt.colorbar(LM_2)
	plt.show()

c = np.array([[-9,-7]])
x0 = np.transpose(c)

startFSG = time.time()
xtry,i,xit = fixed_step_gradient(x,y,x0,(10**(-3)),(10**(-6)),(5*10**(4)))
endFSG = time.time()
paceFSG = endFSG - startFSG
print("\n\n\t\t----------With fixed step gradient----------")
xit_ = np.array(xit)
print("\tfixed step gradient speed (s) =", paceFSG)
print("\tfixed step gradient result =\n", xtry)
print("\tfixed step gradient iteration = ",i)

Steps = np.linspace(10**(-1), 10**(-5), 50)
STEPS_I = list()
TIME = list()
for i in range(len(Steps)) :
	timeC_start = time.time()
	xtryS,iS,xitS = fixed_step_gradient(x,y,x0,Steps[i],(10**(-6)),(5*10**(4)))
	timeC_end = time.time()
	timeC_final = timeC_end - timeC_start
	STEPS_I.append(iS)
	TIME.append(timeC_final)

plt.figure("Nombre d'itérations en fonction du pas")
plt.title("Nombre d'itérations en fonction du pas")
plt.loglog(Steps, STEPS_I)
plt.ylabel("Itérations")
plt.xlabel("Step")

plt.figure("Temps en fonction du pas")
plt.title("Temps en fonction du pas")
plt.loglog(Steps, TIME, color='orange')
plt.ylabel("Time")
plt.xlabel("Steps")
plt.show()

startOSG = time.time()
xtry2,i2,xit2 = optimal_step_gradient(x,y,x0,(10**(-6)),(5*10**(4)))
endOSG = time.time()
paceOSG = endOSG - startOSG
print("\n\n\t\t----------With optimal step gradient----------")
xit2_ = np.array(xit2)
print("\toptimal step gradient speed (s) =", paceOSG)
print("\toptimal step gradient result =\n",xtry2)
print("\toptimal step gradient iterations =",i2)


startCG = time.time()
xtry3,i3,xit3 = conjugued_gradient(x,y,x0,(10**(-6)),(5*10**(4)))
endCG = time.time()
paceCG = endCG - startCG
print("\n\n\t\t----------With conjugued gradient----------")
xit3_ = np.array(xit3)
print("\tconjugued gradient speed (s) =", paceCG)
print("\t conjugued gradient result =\n", xtry3)
print("\tconjugued gradient iteration =",i3)




level_map(xit_, xit2_, xit3_)
display_data(x,y, xtry, xtry2,xtry3)

X,q,XTX,XTq = X_build(x,y)
A = XTX
b = XTq

print("\n")
print("XTX = ", XTX)
print("Xtq = ", XTq)

lbd, v = np.linalg.eig(XTX)
print("Les couples propres sont les suivants", lbd[0], ",",v[0],"), (",lbd[1],",",v[1],")")

cond = np.linalg.cond(XTX)
print("le condtionnement est cond(XTX) = ",cond)

cstar = np.linalg.solve(A,b)
print(cstar)

FCST = (1/2)*(np.transpose(cstar)@A@cstar)-np.transpose(b)@cstar
print(FCST)

d1= np.array([[1],[0]])
d2 = np.array([[0],[1]])


def partial_function (d, cstar, A, b):
	t = np.arange(-10,10.5, 0.05)
	F_cstar = list()
	for i in t: 
		f_cstar = (1/2)* (np.transpose(d)@A@d*(i**2))-np.transpose((b - A@cstar))@d*i + (1/2)*(np.transpose(cstar)@A@cstar)-np.transpose(b)@cstar
		F_cstar.append(f_cstar[0])

	return(F_cstar ,t)


F_cstar1, t= partial_function(d1, cstar,A,b)
F_cstar2, t2= partial_function(d2, cstar,A,b)
F_cstar3, t3= partial_function(v[0], cstar,A,b)
F_cstar4, t4= partial_function(v[1], cstar,A,b)
plt.figure("Partials functions")
plt.subplot(2,2,1)
plt.plot(t,F_cstar1, color ='deepskyblue')
plt.title("Courbe de la fonction partielle $F_{c^*,d}$ pour $d=e_1$")
plt.subplot(2,2,2)
plt.plot(t2,F_cstar2, color ='springgreen')
plt.title("Courbe de la fonction partielle $F_{c^*,d}$ pour $d=e_2$")
plt.subplot(2,2,3)
plt.plot(t3,F_cstar3, color ='mediumorchid')
plt.title("Courbe de la fonction partielle $F_{c^*,d}$ pour $d=v_1$")
plt.subplot(2,2,4)
plt.plot(t4,F_cstar4, color ='darkorange')
plt.title("Courbe de la fonction partielle $F_{c^*,d}$ pour $d=v_2$")
plt.show()






















