#Various routines to generate HDF5 grids to read into Gamera

# Standard modules
import sys
import os

# Third-party modules
import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import importlib.resources as pkg_resources

# Kaipy modules
import kaipy.kaiViz as kv
import kaipy.kdefs as kdefs
from scipy.interpolate import RectBivariateSpline
from kaipy import gamera

#Use routine to generate xx,yy = corners of active upper half plane
#Use Aug2D to add ghosts in 2D: xx,yy -> xxG,yyG
#Use Aug3D to rotate in 3D w/ ghosts: xxG,yyG -> X3,Y3,Z3

Ng = 4
Ni0 = 32
Nj0 = 64
Nk0 = 32

#Gen elliptical grid
def genEllip(Ni=Ni0,Nj=Nj0,Rin=3.0,Rout=30,TINY=1.0e-8):
	"""
	Generate an elliptical grid.

	Args:
		Ni (int): Number of grid points in the x-direction. Default is Ni0.
		Nj (int): Number of grid points in the y-direction. Default is Nj0.
		Rin (float): Inner radius of the ellipse. Default is 3.0.
		Rout (float): Outer radius of the ellipse. Default is 30.0.
		TINY (float): A small value to avoid division by zero. Default is 1.0e-8.

	Returns:
		tuple: A tuple containing the x-coordinates and y-coordinates of the grid points.

	"""
	XX = np.zeros((Ni+1,Nj+1))
	YY = np.zeros((Ni+1,Nj+1))

	P = np.linspace(0+TINY,np.pi-TINY,Nj+1)
	Tau = np.linspace(Rin,Rout,Ni+1)
	A = 6
	A = 1
	#eMax = 0.75
	#eI = np.linspace(0,eMax,Ni+1)
	e = 0.75
	#B = 0.1
	#x0 = 0.0
	for i in range(Ni+1):
		for j in range(Nj+1):
			phi = P[j]
			#e = eI[i]
			r = Tau[i]*A*(1-e*e)/(1+e*np.cos(phi))

			#XX[i,j] = x0 + A*Tau[i]*np.cos(P[j])
			#YY[i,j] = B*Tau[i]*np.sin(P[j])
			XX[i,j] = r*np.cos(P[j])
			YY[i,j] =r*np.sin(P[j])

	return XX,YY

#Gen spherical grid
def genSph(Ni=Ni0, Nj=Nj0, Rin=3.0, Rout=30, TINY=1.0e-8):
	"""
	Generate a spherical grid.

	Args:
		Ni (int): Number of grid points in the radial direction.
		Nj (int): Number of grid points in the angular direction.
		Rin (float): Inner radial bound of the grid.
		Rout (float): Outer radial bound of the grid.
		TINY (float): A small value used for numerical stability.

	Returns:
		tuple: A tuple containing two numpy arrays (XX, YY) representing the coordinates of the grid points.
			XX (numpy.ndarray): Array of shape (Ni+1, Nj+1) containing the x-coordinates of the grid points.
			YY (numpy.ndarray): Array of shape (Ni+1, Nj+1) containing the y-coordinates of the grid points.
	"""
	print("Generating spherical grid w/ radial bounds %3.2f->%3.2f" % (Rin, Rout))
	XX = np.zeros((Ni + 1, Nj + 1))
	YY = np.zeros((Ni + 1, Nj + 1))
	R = np.linspace(Rin, Rout, Ni + 1)
	P = np.linspace(0 + TINY, np.pi - TINY, Nj + 1)
	for i in range(Ni + 1):
		for j in range(Nj + 1):
			XX[i, j] = R[i] * np.cos(P[j])
			YY[i, j] = R[i] * np.sin(P[j])
	return XX, YY


#Gen Gamera egg
#A is theta-warp parameter (<0 to concentrate at poles)
def genEgg(Ni=Ni0, Nj=Nj0, Rin=3.0, Rout=30.0, xtail=250, NumSph=5, TINY=1.0e-8, A=0.0):
	"""
	Generate an egg-shaped grid.

	Args:
		Ni (int): Number of grid points in the x-direction.
		Nj (int): Number of grid points in the y-direction.
		Rin (float): Inner radius of the egg.
		Rout (float): Outer radius of the egg.
		xtail (int): Tail length of the egg.
		NumSph (int): Number of spheres in the egg.
		TINY (float): A small value used for numerical stability.
		A (float): Warping parameter for the egg shape.

	Returns:
		tuple: A tuple containing the x-coordinates and y-coordinates of the grid points.

	"""
	# Get ellipse parameters
	x0, a, b = Egglipses(Ni, Nj, Rin, Rout, xtail, NumSph)

	# Eta is index (0,1) for theta mapping
	eta = np.linspace(0 + TINY, 1.0 - TINY, Nj + 1)
	L = np.pi
	xC = 0.5 * np.pi  # Any warping is symmetric
	theta = L * eta + A * (xC - L * eta) * (1 - eta) * eta

	xx = np.zeros((Ni + 1, Nj + 1))
	yy = np.zeros((Ni + 1, Nj + 1))

	for j in range(Nj + 1):
		for i in range(Ni + 1):
			th = theta[j]
			A = (np.cos(th) / a[i]) ** 2.0 + (np.sin(th) / b[i]) ** 2.0
			B = -2 * np.cos(th) * x0[i] / a[i] ** 2.0
			C = x0[i] ** 2.0 / a[i] ** 2.0 - 1.0
			r = (-B + np.sqrt(B * B - 4 * A * C)) / (2 * A)
			xx[i, j] = r * np.cos(th)
			yy[i, j] = r * np.sin(th)
	
	return xx, yy


#Gen teardrop egg
#A is theta-warp parameter (j stretching)
#Q is egg-warp parameter (i fattening)
def genFatEgg(Ni=Ni0,Nj=Nj0,Rin=3.0,Rout=30.0,xtail=250,NumSph=5,TINY=1.0e-8,A=0.0):
	"""
	Generate a fat egg-shaped grid.

	Args:
		Ni (int): Number of grid points in the x-direction. Default is Ni0.
		Nj (int): Number of grid points in the y-direction. Default is Nj0.
		Rin (float): Inner radius of the egg. Default is 3.0.
		Rout (float): Outer radius of the egg. Default is 30.0.
		xtail (int): Tail length of the egg. Default is 250.
		NumSph (int): Number of spheres in the egg. Default is 5.
		TINY (float): A small value used for numerical stability. Default is 1.0e-8.
		A (float): Warping parameter. Default is 0.0.

	Returns:
		tuple: A tuple containing the x-coordinates and y-coordinates of the grid points.

	"""
	#Get ellipse parameters
	x0,a,b = Egglipses(Ni,Nj,Rin,Rout,xtail,NumSph)

	xSun = np.max(x0+a) #Forward boundary
	xBack = np.abs(np.min(x0-a)) #Back boundary

	#Eta is index (0,1) for theta mapping
	eta = np.linspace(0+TINY,1.0-TINY,Nj+1)
	L = np.pi
	xC = 0.5*np.pi #Any warping is symmetric

	#theta = np.linspace(0+TINY,np.pi-TINY,Nj+1)
	xx = np.zeros((Ni+1,Nj+1))
	yy = np.zeros((Ni+1,Nj+1))
	tau0 = 0.35
	
	di = Ni/8
	i1 = Ni-di
	
	AScl = 3*A

	for i in range(Ni+1):
		#Calculate theta profile for this shell
		xTi = np.abs(x0[i]-a[i])
		
		theta = L*eta + A*(xC-L*eta)*(1-eta)*eta #Lines of constant phi
		if (xTi>xSun):
			#Outer region, keep tailward near-axis region packed tightly
			Ai = A - (A-AScl)*RampUp(xTi,xSun,xBack)
			thTail = L*eta + Ai*(xC-L*eta)*(1-eta)*eta
			#Only use back half (past half-way)
			theta[Nj//2:] = thTail[Nj//2:]

		for j in range(Nj+1):
			th = theta[j]
			Ae = (np.cos(th)/a[i])**2.0 + (np.sin(th)/b[i])**2.0
			Be = -2*np.cos(th)*x0[i]/a[i]**2.0
			Ce = x0[i]**2.0/a[i]**2.0 - 1.0
			r = (-Be+np.sqrt(Be*Be-4*Ae*Ce))/(2*Ae)

			xx[i,j] = r*np.cos(th)
			
			taui = min(tau0*RampUp(1.0*i,i1,di),1.0)
			tauscl = taui*(xx[i,j])/a[i]
			tau = min(1.0+tauscl,1)
			yy[i,j] = r*np.sin(th)/np.sqrt(tau)				


	return xx,yy

def RampUp(r, rC, lC):
	"""
	Calculate the ramp-up value based on the given parameters.

	Args:
		r (float): The input value.
		rC (float): The center value for the ramp-up.
		lC (float): The length of the ramp-up.

	Returns:
		float: The ramp-up value.

	"""
	rScl = (r - rC) / lC
	if rScl <= 0:
		M = 0
	elif rScl >= 1.0:
		M = 1.0
	else:
		eArg = -1.0 / (rScl * rScl)
		M = np.exp(1.0) * np.exp(eArg)
	return M


#Radial ellipse parameters for egg grid
def Egglipses(Ni=Ni0,Nj=Nj0,Rin=3.0,Rout=30.0,rtail=250.0,NumSph=5):
	"""
	Compute the parameters of ellipses.

	Args:
		Ni (int): Number of points in the x-direction.
		Nj (int): Number of points in the y-direction.
		Rin (float): Inner radius of the ellipses.
		Rout (float): Outer radius of the ellipses.
		rtail (float): Tail radius of the ellipses.
		NumSph (int): Number of spheres.

	Returns:
		tuple: A tuple containing the following parameters:
			- x0 (ndarray): x-coordinate of the center of the ellipses.
			- a (ndarray): Semi-major axis of the ellipses.
			- b (ndarray): Semi-minor axis of the ellipses.
	"""
	#Shells
	NI = Ni+1
	NJ = Nj+1

	#Compute Xs, sunward x values
	dScl = 1.22*(Rout/27.4675)
	#Magic numbers for min/max dx
	d1 = dScl*0.74/(NI/32.0)
	d2 = dScl*0.985925/(NI/32.0)

	dx = np.zeros(Ni+1)
	xs = np.zeros(Ni+1)

	dx[0:NumSph-1] = d1
	dx[NumSph-1:] = np.linspace(d1,d2,Ni-NumSph+2)
	xs[0] = Rin

	for i in range(Ni):
		xs[i+1] = xs[i]+dx[i]

	#Now compute tailward x values
	#d2 = 40.0/(NI/32.0) #Max dx
	d2 = 50.0/(NI/32.0) #Max dx
	d2 = (rtail/237.710955)*d2
	xp = np.linspace(-4,0,Ni-NumSph+2)
	yp = np.tanh(xp)+1
	zp = (d2-d1)*yp+d1
	xe = np.zeros(Ni+1)
	xe[0] = -Rin

	dx[0:NumSph-1] = d1
	dx[NumSph-1:] = zp

	for i in range(Ni):
		xe[i+1] = xe[i]-dx[i]

	#Now compute Y
	d2 = 9.0/(NI/32.0)
	xp = np.linspace(-4,0,Ni-NumSph+2)
	yp = np.tanh(xp)+1
	zp = (d2-d1)*yp+d1
	dy = np.zeros(Ni+1)
	yc = np.zeros(Ni+1)
	dy[0:NumSph-1] = d1
	dy[NumSph-1:] = zp

	yc[0] = Rin
	for i in range(Ni):
		yc[i+1] = yc[i]+dy[i]

	#Ellipse calculations
	a = 0.5*(xs-xe)
	xb = (xs+xe)/(xs-xe)

	b = yc/np.sqrt(1-xb**2.0)
	x0 = 0.5*(xs+xe)
	return x0,a,b

#Generate 3D spherical grid with Z-axis north pole
#Spherical :: r,theta,phi = ijk
#theta = [0,1] -> [0,pi]
#phi = [0,1] -> [0,2pi]
def GenKSph(Ni=Ni0,Nj=Nj0,Nk=Nk0,Rin=5,Rout=40,tMin=0.2,tMax=0.8):
	"""
	Generate a spherical grid in 3D.

	Args:
		Ni: Number of grid points in the radial direction.
		Nj: Number of grid points in the polar angle direction.
		Nk: Number of grid points in the azimuthal angle direction.
		Rin: Inner radius of the grid.
		Rout: Outer radius of the grid.
		tMin: Minimum polar angle of the grid.
		tMax: Maximum polar angle of the grid.

	Returns:
		X3: X-coordinate values of the grid points.
		Y3: Y-coordinate values of the grid points.
		Z3: Z-coordinate values of the grid points.
	"""
	#Number of nodes with ghost corners
	Ngi = Ni+1+2*Ng
	Ngj = Nj+1+2*Ng
	Ngk = Nk+1+2*Ng

	dx1 = (Rout-Rin)/Ni
	dx2 = (tMax-tMin)/Nj
	dx3 = (1.0 - 0.0)/Nk

	# check that ghosts don't take us across the axis
	# need to generalize later to include the axis (do full 4pi)
	if ((tMin-Ng*dx2)<=0) or ((tMax+Ng*dx2)>=1.):
		sys.exit("Ghost cell region includes the spherical axis. This is not implemented yet.")

	r = np.linspace(Rin-Ng*dx1,Rout+Ng*dx1,Ngi)
	t = np.linspace(tMin-Ng*dx2,tMax+Ng*dx2,Ngj)*np.pi
	p = np.linspace(-Ng*dx3,1.+Ng*dx3,Ngk)*2*np.pi

	# note the indexing flag for proper ordering for writeGrid later
	R,T,P = np.meshgrid(r,t,p,indexing='ij')

	X3 = R*np.sin(T)*np.cos(P)
	Y3 = R*np.sin(T)*np.sin(P)
	Z3 = R*np.cos(T)

	return X3,Y3,Z3

#Generate 3D spherical grid with non-uniform grid in r and with Z-axis north pole
#Spherical :: r,theta,phi = ijk
#theta = [0,1] -> [0,pi]
#phi = [0,1] -> [0,2pi]
def GenKSphNonU(Ni=Ni0,Nj=Nj0,Nk=Nk0,Rin=5,Rout=40,tMin=0.2,tMax=0.8):
	"""
	Generate a non-uniform spherical grid.

	Args:
		Ni (int): Number of nodes in the radial direction.
		Nj (int): Number of nodes in the polar angle direction.
		Nk (int): Number of nodes in the azimuthal angle direction.
		Rin (float): Inner radius of the grid.
		Rout (float): Outer radius of the grid.
		tMin (float): Minimum value of the polar angle.
		tMax (float): Maximum value of the polar angle.

	Returns:
		X3 (ndarray): x-coordinates of the grid points.
		Y3 (ndarray): y-coordinates of the grid points.
		Z3 (ndarray): z-coordinates of the grid points.
	"""
	
	#Number of nodes with ghost corners
	Ngi = Ni+1+2*Ng
	Ngj = Nj+1+2*Ng
	Ngk = Nk+1+2*Ng

	dx2 = (tMax-tMin)/Nj
	dx3 = (1.0 - 0.0)/Nk

	# check that ghosts don't take us across the axis
	# need to generalize later to include the axis (do full 4pi)
	if ((tMin-Ng*dx2)<=0) or ((tMax+Ng*dx2)>=1.):
			sys.exit("Ghost cell region includes the spherical axis. This is not implemented yet.")

	nu = np.linspace(0,1,Ni+1)
	r0 = []
	r1 = [((Rout - Rin)*(x*x+x)/2. + Rin) for x in nu]
	dxN = Rout - ((Rout-Rin)*(nu[Ni-1]*nu[Ni-1]+nu[Ni-1])/2. + Rin)
	dx0 = (Rout-Rin)*(nu[1]*nu[1]+nu[1])/2.
	for i in range(Ng):
			r1.append(Rout + (i+1)*dxN)
			r0.append(Rin - (Ng-i)*dx0)
	r = r0 + r1

	t = np.linspace(tMin-Ng*dx2,tMax+Ng*dx2,Ngj)*np.pi
	p = np.linspace(-Ng*dx3,1.+Ng*dx3,Ngk)*2*np.pi

	# note the indexing flag for proper ordering for writeGrid later
	R,T,P = np.meshgrid(r,t,p,indexing='ij')

	X3 = R*np.sin(T)*np.cos(P)
	Y3 = R*np.sin(T)*np.sin(P)
	Z3 = R*np.cos(T)

	return X3,Y3,Z3

#Generate 3D spherical grid with non-uniform grid in r and with Z-axis north pole
#Spherical :: r,theta,phi = ijk
#theta = [0,1] -> [0,pi]
#phi = [0,1] -> [0,2pi]
def GenKSphNonUGL(Ni=Ni0,Nj=Nj0,Nk=Nk0,Rin=5,Rout=40,tMin=0.2,tMax=0.8):
	"""
	Generate a non-uniform grid in a spherical coordinate system.

	Args:
		Ni: Number of grid points in the radial direction.
		Nj: Number of grid points in the polar angle direction.
		Nk: Number of grid points in the azimuthal angle direction.
		Rin: Inner radius of the grid.
		Rout: Outer radius of the grid.
		tMin: Minimum value of the polar angle.
		tMax: Maximum value of the polar angle.

	Returns:
		X3: Array of x-coordinates of the grid points.
		Y3: Array of y-coordinates of the grid points.
		Z3: Array of z-coordinates of the grid points.
	"""
	#Number of nodes with ghost corners
	Ngi = Ni+1+2*Ng
	Ngj = Nj+1+2*Ng
	Ngk = Nk+1+2*Ng

	dx2 = (tMax-tMin)/Nj
	dx3 = (1.0 - 0.0)/Nk

	# check that ghosts don't take us across the axis
	# need to generalize later to include the axis (do full 4pi)
	if ((tMin-Ng*dx2)<=0) or ((tMax+Ng*dx2)>=1.):
			sys.exit("Ghost cell region includes the spherical axis. This is not implemented yet.")

	#grid in r
	Nwl = 194
	Rmid = 64.5
	dtau = np.arctan((Rmid-Rin)/Rin)/Nwl  #dtau in radians
	r1 = [Rin + Rin*(np.tan(i*dtau)) for i in range(Nwl+1)] #194 cells

	Nout = Ni - Nwl
	coeff = (Rout-Rmid)/Nout*2.-0.9-0.9
	dr = [0.9 + coeff*i/(Nout-1) for i in range(Nout)]

	r = Rmid
	for i in range(Nout):
			r = r + dr[i]
			r1.append(r)

	dx0 = r1[1] - r1[0]
	dxN = r1[Ni] - r1[Ni-1]
	r0 = [ ]

	for i in range(Ng):
			r1.append(Rout + (i+1)*dxN)
			r0.append(Rin - (Ng-i)*dx0)
	r = r0 + r1

	t = np.linspace(tMin-Ng*dx2,tMax+Ng*dx2,Ngj)*np.pi
	p = np.linspace(-Ng*dx3,1.+Ng*dx3,Ngk)*2*np.pi

	# note the indexing flag for proper ordering for writeGrid later
	R,T,P = np.meshgrid(r,t,p,indexing='ij')

	X3 = R*np.sin(T)*np.cos(P)
	Y3 = R*np.sin(T)*np.sin(P)
	Z3 = R*np.cos(T)

	return X3,Y3,Z3


#Now have 2D grid, augment with ghosts
#KeepOut => Keep inner boundary outside of R=1 (i.e. planet)
def Aug2D(XX,YY,doEps=False,TINY=1.0e-8,KeepOut=True,Rpx=1.15):
	"""
	Perform 2D augmentation on the given XX and YY arrays.

	Args:
		XX (ndarray): 2D array representing the x-coordinates.
		YY (ndarray): 2D array representing the y-coordinates.
		doEps (bool, optional): Flag indicating whether to perform epsilon adjustment. Defaults to False.
		TINY (float, optional): Small value used for epsilon adjustment. Defaults to 1.0e-8.
		KeepOut (bool, optional): Flag indicating whether to keep the inner boundary outside. Defaults to True.
		Rpx (float, optional): Critical radius for the inner boundary. Defaults to 1.15.

	Returns:
		ndarray: Augmented xxG array.
		ndarray: Augmented yyG array.
	"""
	Ni = XX.shape[0]-1
	Nj = XX.shape[1]-1

	if (doEps):
		YY[:,0]  = TINY
		YY[:,-1] = TINY

	RR = np.sqrt(XX**2.0 + YY**2.0)
	PP = np.arctan2(YY,XX)

	R0 = RR.min()
	if (R0<=Rpx):
		print("Inner boundary below critical (%f), bailing ..."%(Rpx))
		quit()
	#print("Radial domain = [%3.2f,%3.2f]"%(R0,RR.max()))
	xs = RR[:,0]
	xe = RR[:,-1]
	#print("\tSunward  = [%3.2f,%3.2f]"%(xs.min(),xs.max()))
	#print("\tTailward = [%3.2f,%3.2f]"%(xe.min(),xe.max()))

	xxG = np.zeros((Ni+1+2*Ng,Nj+1+2*Ng))
	yyG = np.zeros((Ni+1+2*Ng,Nj+1+2*Ng))
	#0:Ng (non-inclusive end), Ng:Ng+nOut
	iS = Ng
	iE = Ng+Ni+1
	jS = Ng
	jE = Ng+Nj+1
	xxG[iS:iE,jS:jE] = XX
	yyG[iS:iE,jS:jE] = YY

	#Get spacings, ensure inner boundary is outside of 1
	drIn = RR[1,:]-RR[0,:] #Function of j
	drAvg = drIn.mean()

	RMin = R0-Ng*drAvg
	
	if (RMin<Rpx):
		#Adjust spacing to keep inner boundary outside
		drAvg = (R0-Rpx-TINY)/Ng
	pIn = PP[0,:]

	#Do inner I, active J
	for i in range(1,Ng+1):
		rIn = R0-i*drAvg
		xxG[iS-i,jS:jE] = rIn*np.cos(pIn)
		yyG[iS-i,jS:jE] = rIn*np.sin(pIn)
	

	#Do outer I, active J
	Dx = XX[-1,:]-XX[-2,:]
	Dy = YY[-1,:]-YY[-2,:]
	Dp = PP[-1,:]-PP[-2,:]
	Dr = RR[-1,:]-RR[-2,:]

	nD = np.sqrt(Dx**2.0+Dy**2.0)
	dBar = nD.mean()
	dO = np.minimum(nD,dBar)
	#dO = nD

	J4 = Nj//4
	xOut = XX[-1,:]
	yOut = YY[-1,:]
	sig = 1.5
	jSigS = J4
	jSigE = 3*J4

	Dx[jSigS:jSigE] = gaussian_filter(Dx[jSigS:jSigE],sigma=sig,mode='nearest')
	Dy[jSigS:jSigE] = gaussian_filter(Dy[jSigS:jSigE],sigma=sig,mode='nearest')
	xS = xOut[0]+Ng*Dr[0]
	xT = xOut[-1]-Ng*Dr[-1]
	
	#Dx[0:J4] = (xS-xOut[0:J4])/Ng

	for i in range(0,Ng):
		xxG[iE+i,jS:jE] = xOut+(i+1)*dO*Dx/nD
		yyG[iE+i,jS:jE] = yOut+(i+1)*dO*Dy/nD



	#Now finish by doing all J boundaries
	#Just reflect about X-axis
	for i in range(0,Ng):
		ip = i+1
		xxG[:,jS-ip]    =  xxG[:,jS+ip]
		yyG[:,jS-ip]    = -yyG[:,jS+ip]
		xxG[:,jE-1+ip]  =  xxG[:,jE-1-ip]
		yyG[:,jE-1+ip]  = -yyG[:,jE-1-ip]

	return xxG,yyG

# vgm: add a version of the above function to extend the grid in the i-direction
# this never worked; keeping for completeness.
# what's worked, though, is the scale option in regrid at the bottom of this file
def Aug2Dext(XX,YY,Nadd):
	"""
	Extend a 2D grid by adding points in the i-direction.

	Args:
		XX (ndarray): Input grid in the x-direction.
		YY (ndarray): Input grid in the y-direction.
		Nadd (int): Number of points to add in the i-direction.

	Returns:
		ndarray: Extended grid in the x-direction.
		ndarray: Extended grid in the y-direction.
	"""
	Ni = XX.shape[0]
	Nj = XX.shape[1]

	xxG = np.zeros((Ni + Nadd, Nj))
	yyG = np.zeros((Ni + Nadd, Nj))

	xxG[:-Nadd, :] = XX
	yyG[:-Nadd, :] = YY

	# Do outer I, active J
	Dx = XX[-1, :] - XX[-2, :]
	Dy = YY[-1, :] - YY[-2, :]
	nD = np.sqrt(Dx**2.0 + Dy**2.0)
	dBar = nD.mean()
	dO = nD

	for i in np.arange(Nadd):
		xxG[-Nadd+i:, :] = 2 * xxG[-Nadd+i-1, :] - xxG[-Nadd+i-2, :]
		yyG[-Nadd+i:, :] = 2 * yyG[-Nadd+i-1, :] - yyG[-Nadd+i-2, :]

	return xxG, yyG


#Do ring recommendations
def genRing(XX, YY, Nk=64, Tol=1.0, doVerb=False):
	"""
	Generate a ring grid based on the input coordinates.

	Args:
		XX (ndarray): Array of x-coordinates.
		YY (ndarray): Array of y-coordinates.
		Nk (int): Number of divisions in the ring grid. Default is 64.
		Tol (float): Tolerance value for determining the maximum scale of each ring. Default is 1.0.
		doVerb (bool): Flag to enable verbose output. Default is False.

	Returns:
		None

	"""
	Ni = XX.shape[0] - 1
	Nj = XX.shape[1] - 1
	dTh = 2 * np.pi / Nk
	# For each ring (j), calculate max ring value
	rr = np.sqrt(XX ** 2.0 + YY ** 2.0)
	pp = np.arctan2(YY, XX)
	dI = np.zeros((Ni, Nj))
	dJ = np.zeros((Ni, Nj))
	dK = np.zeros((Ni, Nj))
	for i in range(Ni):
		for j in range(Nj):
			dR = 0.5 * (rr[i + 1, j] + rr[i + 1, j + 1]) - 0.5 * (rr[i, j] + rr[i, j + 1])
			dP = 0.5 * (pp[i, j + 1] + pp[i + 1, j + 1]) - 0.5 * (pp[i, j] + pp[i + 1, j])
			Rc = 0.25 * (rr[i, j] + rr[i, j + 1] + rr[i + 1, j] + rr[i + 1, j + 1])
			Yc = 0.25 * (YY[i, j] + YY[i, j + 1] + YY[i + 1, j] + YY[i + 1, j + 1])

			dI[i, j] = dR
			dJ[i, j] = Rc * dP
			dK[i, j] = Yc * dTh
	Nrng = Nj // 2

	NChs = np.zeros(Nrng, dtype=int)
	NCha = np.zeros(Nrng, dtype=int)
	rS = []
	for n in range(Nrng):
		kiP = np.max(dK[:, n] / dI[:, n])
		kjP = np.max(dK[:, n] / dJ[:, n])
		kiM = np.max(dK[:, -n] / dI[:, -n])
		kjM = np.max(dK[:, -n] / dJ[:, -n])
		dkoMax = np.max([kiP, kjP, kiM, kjM])
		dRng = dkoMax / Tol
		# Do safe and aggressive
		djRs = int(2 ** np.floor(np.log2(Tol / dkoMax)))
		djRa = int(2 ** np.ceil(np.log2(Tol / dkoMax)))

		if dRng <= 1.0:
			if doVerb:
				print("Ring %d, MaxScl = %5.3f" % (n + 1, 1 / dRng))
				# print("Ring %d, MaxChunk = %5.3f"%(n+1,Nk*dRng))
			# print("\tChunking: %d (Safe) / %d (Aggressive)"%(djRs,djRa))
			rS.append("%4.2f" % (Nk * dkoMax))
			if djRs >= 2:
				NChs[n] = Nk / djRs
			if djRa >= 2:
				NCha[n] = Nk / djRa

	# print(rS)
	PrintRing(NChs, "Safe")
	if doVerb:
		PrintRing(NCha, "Aggressive", doWarn=True)


def PrintRing(NCh, rID="safe", doWarn=False):
	"""
	Print the ring configuration.

	Args:
		NCh (numpy array): Array containing the number of channels for each ring.
		rID (str): Identifier for the ring configuration. Default is "safe".
		doWarn (bool): Flag indicating whether to display a warning message. Default is False.
	"""
	print("%s ring configuration ..." % (rID))

	if doWarn:
		print("Don't say I didn't warn you")

	Nr = (NCh > 0).sum()
	rStr = '<ring gid="lfm" doRing="T" ' + 'Nr="%d"' % (Nr)
	for i in range(Nr):
		Nc = NCh[i]
		if Nc < 8:
			Nc = 8
		rStr = rStr + ' Nc%d="%d"' % (i + 1, Nc)
	rStr = rStr + '/>'
	print(rStr)
	print("")

#Now do full 3D by rotating about X-axis
def Aug3D(xxG, yyG, Nk=32, TINY=1.0e-8):
	"""
	Generate a 3D grid with augmented dimensions.

	Args:
		xxG (ndarray): 2D array representing the x-coordinates of the grid.
		yyG (ndarray): 2D array representing the y-coordinates of the grid.
		Nk (int): Number of grid points in the z-direction (default: 32).
		TINY (float): Small value used for spacing error check (default: 1.0e-8).

	Returns:
		X3 (ndarray): 3D array representing the augmented x-coordinates of the grid.
		Y3 (ndarray): 3D array representing the augmented y-coordinates of the grid.
		Z3 (ndarray): 3D array representing the augmented z-coordinates of the grid.
	"""
	Ni = xxG.shape[0] - 1 - 2 * Ng
	Nj = xxG.shape[1] - 1 - 2 * Ng

	nFi = Ni + 1 + 2 * Ng
	nFj = Nj + 1 + 2 * Ng
	nFk = Nk + 1 + 2 * Ng

	iS = Ng
	iE = Ng + Ni + 1
	jS = Ng
	jE = Ng + Nj + 1
	kS = Ng
	kE = Ng + Nk + 1

	X3 = np.zeros((nFi, nFj, nFk))
	Y3 = np.zeros((nFi, nFj, nFk))
	Z3 = np.zeros((nFi, nFj, nFk))

	# Angle about axis including ghosts
	A = np.linspace(0, 2 * np.pi, Nk + 1)

	for n in range(Nk + 1):
		X3[:, :, kS + n] = xxG
		Y3[:, :, kS + n] = yyG * np.cos(A[n])
		Z3[:, :, kS + n] = yyG * np.sin(A[n])

	# K is periodic so ensure the last entry equals the first
	X3[:, :, kE - 1] = X3[:, :, kS]
	Y3[:, :, kE - 1] = Y3[:, :, kS]
	Z3[:, :, kE - 1] = Z3[:, :, kS]

	# Correct K periodic ghosts
	for i in range(0, Ng):
		ip = i + 1
		X3[:, :, kS - ip] = X3[:, :, kE - 1 - ip]
		Y3[:, :, kS - ip] = Y3[:, :, kE - 1 - ip]
		Z3[:, :, kS - ip] = Z3[:, :, kE - 1 - ip]
		X3[:, :, kE - 1 + ip] = X3[:, :, kS + ip]
		Y3[:, :, kE - 1 + ip] = Y3[:, :, kS + ip]
		Z3[:, :, kE - 1 + ip] = Z3[:, :, kS + ip]

	# Correct J reflective ghosts
	for k in range(0, nFk):
		kp = k + Nk // 2
		if kp >= kE - 1:
			kp = k - Nk // 2
		for i in range(0, Ng):
			ip = i + 1
			X3[:, jS - ip, k] = X3[:, jS + ip, kp]
			Y3[:, jS - ip, k] = Y3[:, jS + ip, kp]
			Z3[:, jS - ip, k] = Z3[:, jS + ip, kp]
			X3[:, jE - 1 + ip, k] = X3[:, jE - 1 - ip, kp]
			Y3[:, jE - 1 + ip, k] = Y3[:, jE - 1 - ip, kp]
			Z3[:, jE - 1 + ip, k] = Z3[:, jE - 1 - ip, kp]

	# Force points to plane
	Z3[:, :, Ng] = 0.0
	Z3[:, :, -Ng - 1] = 0.0
	Z3[:, :, Ng + Nk // 2] = 0.0

	Y3[:, :, Ng + Nk // 4] = 0.0
	Y3[:, :, -Ng - Nk // 4 - 1] = 0.0

	x0 = X3[:, Ng, Ng]
	if x0.min() <= TINY:
		print("Spacing error on inner I")
		print(x0)

	return X3, Y3, Z3


def WriteGrid(X3, Y3, Z3, fOut="gGrid.h5"):
	"""
	Write the grid data to an HDF5 file.

	Args:
		X3 (numpy.ndarray): The X coordinates of the grid.
		Y3 (numpy.ndarray): The Y coordinates of the grid.
		Z3 (numpy.ndarray): The Z coordinates of the grid.
		fOut (str): The output file name. Default is "gGrid.h5".
	"""
	import h5py
	#print("Writing out grid to %s"%fOut)
	with h5py.File(fOut, 'w') as hf:
		hf.create_dataset("X", data=X3.T)
		hf.create_dataset("Y", data=Y3.T)
		hf.create_dataset("Z", data=Z3.T)

#Write grid as CHIMP grid, only active nodes/single precision
def WriteChimp(X3, Y3, Z3, fOut="cGrid.h5"):
	"""
	Write the CHIMP X, Y, and Z arrays to a grid file in single precision.

	Args:
		X3 (ndarray): The X array.
		Y3 (ndarray): The Y array.
		Z3 (ndarray): The Z array.
		fOut (str, optional): The output file name. Defaults to "cGrid.h5".
	"""
	Xc = np.single(X3[Ng:-Ng, Ng:-Ng, Ng:-Ng])
	Yc = np.single(Y3[Ng:-Ng, Ng:-Ng, Ng:-Ng])
	Zc = np.single(Z3[Ng:-Ng, Ng:-Ng, Ng:-Ng])
	WriteGrid(Xc, Yc, Zc, fOut=fOut)
	
	#print("Orig shape = %s"%(str(X3.shape)))
	#print(" New shape = %s"%(str(Xc.shape)))

def VizGrid(XX,YY,xxG=None,yyG=None,doGhost=False,doShow=True,xyBds=None,fOut="grid.png"):
	"""
	Visualizes a grid using matplotlib.

	Args:
		XX (numpy.ndarray): X-coordinates of the grid.
		YY (numpy.ndarray): Y-coordinates of the grid.
		xxG (numpy.ndarray, optional): X-coordinates of the ghost points.
		yyG (numpy.ndarray, optional): Y-coordinates of the ghost points.
		doGhost (bool, optional): Whether to plot the ghost points.
		doShow (bool, optional): Whether to display the plot.
		xyBds (list, optional): Bounds of the plot in the form [x_max, x_min, y_min, y_max].
		fOut (str, optional): Output file name for the plot.

	Returns:
		None
	"""

	fSz = (10,4)
	Alph = 0.35
	fig = plt.figure(figsize=fSz)
	Ax = plt.gca()
	C = 'dodgerblue'
	Ni = XX.shape[0]-1
	Nj = XX.shape[1]-1
	G = np.zeros((Ni,Nj))

	G[-Ng:,:] = 1.0
	Ax.pcolormesh(XX,YY,G,cmap="RdGy")
	if (doGhost):
		Ax.plot(xxG,yyG,'r-',xxG.T,yyG.T,'r-',linewidth=0.5)
	Ax.plot(XX,YY    ,color=C,linewidth=1.5,alpha=Alph)
	Ax.plot(XX.T,YY.T,color=C,linewidth=1.5,alpha=Alph)

	Nr4 = int(Ni/4)
	Np4 = int(Nj/4)
	C4 = "blue"
	LW4 = 2.0
	for dx in range(1,4):
		j0 = dx*Np4
		i0 = dx*Nr4
		Ax.plot(XX[:,j0],YY[:,j0],color=C4,linewidth=LW4,alpha=Alph)
		Ax.plot(XX[i0,:],YY[i0,:],color=C4,linewidth=LW4,alpha=Alph)

	if (xyBds is None):
		xyBds = [XX[-1,-1],XX[-1,0],0,np.max(YY)]

	kv.SetAx(xyBds,Ax)
	kv.addEarth2D(ax=Ax)
	kv.SetAxLabs(Ax,'SM-X [Rx]','SM-Y [Rx]')

	Ni,Nj = XX.shape
	tStr = "Grid Size = %d,%d"%(Ni-1,Nj-1)
	Ax.set_title(tStr)
	fname,fext = os.path.splitext(fOut)
	fPic = fname+".png"
	print("Saving image to %s"%(fPic))
	kv.savePic(fPic)

	if (doShow):
		plt.show()

def LoadTabG(fIn="lfmG", Nc=0):
	"""
	LoadTabG function loads tabular data from files LFM X and Y grid files.

	Args:
		fIn (str): The base filename of the input files (without extension). Default is "lfmG".
		Nc (int): The number of rows to exclude from the loaded data. Default is 0.

	Returns:
		tuple: A tuple containing two numpy arrays, xxi and yyi, representing the loaded data.

	"""
	import os
	#__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

	fInX = pkg_resources.files(gamera).joinpath(f"{fIn}.X.txt") #os.path.join(__location__, fIn + ".X.txt")
	fInY = pkg_resources.files(gamera).joinpath(f"{fIn}.Y.txt")#os.path.join(__location__, fIn + ".Y.txt")
	xxi = np.loadtxt(fInX)
	yyi = np.loadtxt(fInY)

	if Nc > 0:
		xxi = xxi[0:-Nc, :]
		yyi = yyi[0:-Nc, :]
	return xxi, yyi


#Regrid xx/yy (corners) to new size


def regrid(xxi, yyi, Ni, Nj, Rin=0.0, Rout=0.0, TINY=1.0e-8, scale=False):
	"""
	Regrids a grid defined by `xxi` and `yyi` onto a new grid with dimensions `Ni` and `Nj`.

	Parameters:
		xxi (ndarray): The x-coordinate values of the original grid.
		yyi (ndarray): The y-coordinate values of the original grid.
		Ni (int): The number of points in the x-direction of the new grid.
		Nj (int): The number of points in the y-direction of the new grid.
		Rin (float, optional): The minimum radial value for rescaling. Defaults to 0.0.
		Rout (float, optional): The maximum radial value for rescaling. Defaults to 0.0.
		TINY (float, optional): A small value used for numerical stability. Defaults to 1.0e-8.
		scale (bool, optional): Whether to extend the grid for low Mach numbers. Defaults to False.

	Returns:
		ndarray: The x-coordinate values of the regridded grid.
		ndarray: The y-coordinate values of the regridded grid.
	"""
	Ni0 = xxi.shape[0] - 1
	Nj0 = xxi.shape[1] - 1
	rr0 = np.sqrt(xxi ** 2.0 + yyi ** 2.0)
	pp0 = np.arctan2(yyi, xxi)

	# index space -> (r,phi)
	iLFM = np.linspace(0, 1, Ni0 + 1)
	jLFM = np.linspace(0, 1, Nj0 + 1)

	if (Rin > TINY) or (Rout > TINY):
		# Rescale radial range
		xMin = rr0.min()
		xMax = xxi.max()

		if Rin > TINY:
			xSclIn = Rin / xMin
		else:
			xSclIn = 1.0

		if Rout > TINY:
			xSclOut = Rout / xMax
		else:
			xSclOut = 1.0
		dxScl = (xSclOut - xSclIn)

		# Now rescale each i shell
		for i in range(Ni0 + 1):
			rScl = xSclIn + iLFM[i] * dxScl
			rr0[i, :] = rScl * rr0[i, :]

	# Create interpolants using RectBivariateSpline
	fR = RectBivariateSpline(iLFM, jLFM, rr0, kx=3, ky=3)
	fP = RectBivariateSpline(iLFM, jLFM, pp0, kx=3, ky=3)

	# Regrid onto new
	XXi = np.zeros((Ni + 1, Nj + 1))
	YYi = np.zeros((Ni + 1, Nj + 1))

	si = np.linspace(0, 1, Ni + 1)
	sj = np.linspace(0, 1, Nj + 1)
	for i in range(Ni + 1):
		for j in range(Nj + 1):
			r = fR(si[i], sj[j])[0, 0]
			phi = fP(si[i], sj[j])[0, 0]
			phi = np.maximum(TINY, phi)
			phi = np.minimum(np.pi - TINY, phi)

			XXi[i, j] = r * np.cos(phi)
			YYi[i, j] = r * np.sin(phi)

	# vgm: added scaling option to extend the grid for low Mach numbers
	# needs playing around with the numbers below
	if scale:
		dx = (XXi[1:, :] - XXi[:-1, :])
		dy = (YYi[1:, :] - YYi[:-1, :])
		nscl = dx.shape[0]
		scale = np.ones(nscl)
		scale[nscl // 2:] = 1.25  # 1.5
		scale[3 * nscl // 4:] = 1.5  # 2.
		scale[-4:] = 2.  # 4.
		for i in np.arange(1, Ni + 1):
			XXi[i, :] = XXi[i - 1, :] + scale[i - 1] * dx[i - 1, :]
			YYi[i, :] = YYi[i - 1, :] + scale[i - 1] * dy[i - 1, :]

	return XXi, YYi

