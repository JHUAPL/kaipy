#Various routines to help with restart upscaling

# Standard modules

# Third-party modules
import h5py
import numpy as np
import scipy
from scipy.spatial import ConvexHull

# Kaipy modules
import kaipy.kaiH5 as kh5
import kaipy.gamera.gamGrids as gg
TINY = 1.0e-8
NumG = 4
IDIR = 0
JDIR = 1
KDIR = 2

#Push restart data to an MPI tiling
#fInA is a restart file to pull attributes from
# add oG and oM for one step older for reproducible restart.
# These variables are used by mhd predictor/corrector.
def PushRestartMPI(outid, nRes, Ri, Rj, Rk, X, Y, Z, G, M, oG, oM, fInA, G0=None):
	"""
	Generate and write restart files for MPI simulation.

	Args:
		outid (int): Output ID.
		nRes (int): Number of restarts.
		Ri (int): Number of MPI processes in the i-direction.
		Rj (int): Number of MPI processes in the j-direction.
		Rk (int): Number of MPI processes in the k-direction.
		X (ndarray): Grid coordinates in the X-direction.
		Y (ndarray): Grid coordinates in the Y-direction.
		Z (ndarray): Grid coordinates in the Z-direction.
		G (ndarray): Gas data.
		M (ndarray): Magnetic flux data.
		oG (ndarray): Offset gas data.
		oM (ndarray): Offset magnetic flux data.
		fInA (str): Input file name.
		G0 (ndarray, optional): Additional gas data. Defaults to None.
	"""
	doGas0 = (G0 is not None)  # Make sure doGas0 is either True or False

	print("Reading attributes from %s" % (fInA))
	iH5 = h5py.File(fInA, 'r')

	Ns, Nv, Nk, Nj, Ni = G.shape
	# Create output files
	Nkp = Nk // Rk
	Njp = Nj // Rj
	Nip = Ni // Ri

	print("Splitting (%d,%d,%d) cells into (%d,%d,%d) x (%d,%d,%d) [Cells,MPI]" % (Ni, Nj, Nk, Nip, Njp, Nkp, Ri, Rj, Rk))
	# Loop over output slices and create restarts
	for i in range(Ri):
		for j in range(Rj):
			for k in range(Rk):
				fOut = kh5.genName(outid, i, j, k, Ri, Rj, Rk, nRes)

				# Open output file
				oH5 = h5py.File(fOut, 'w')

				iS = i * Nip
				iE = iS + Nip
				jS = j * Njp
				jE = jS + Njp
				kS = k * Nkp
				kE = kS + Nkp

				# Indices for offset ghost grid
				iSg = iS - NumG + NumG  # Last numG to offset to 0-based index
				iEg = iS + Nip + NumG + 1 + NumG  # Last numG to offset to 0-based index
				jSg = jS - NumG + NumG  # Last numG to offset to 0-based index
				jEg = jS + Njp + NumG + 1 + NumG  # Last numG to offset to 0-based index
				kSg = kS - NumG + NumG  # Last numG to offset to 0-based index
				kEg = kS + Nkp + NumG + 1 + NumG  # Last numG to offset to 0-based index

				print("Writing %s" % (fOut))
				print("\tMPI (%d,%d,%d) = [%d,%d]x[%d,%d]x[%d,%d]" % (i, j, k, iS, iE, jS, jE, kS, kE))
				print("\tGrid indices = (%d,%d)x(%d,%d)x(%d,%d)" % (iSg, iEg, jSg, jEg, kSg, kEg))

				# Slice subgrids
				ijkX = X[kSg:kEg, jSg:jEg, iSg:iEg]
				ijkY = Y[kSg:kEg, jSg:jEg, iSg:iEg]
				ijkZ = Z[kSg:kEg, jSg:jEg, iSg:iEg]

				# Slice pieces out of gas and magflux
				ijkG = G[:, :, kS:kE, jS:jE, iS:iE]
				ijkM = M[:, kS:kE + 1, jS:jE + 1, iS:iE + 1]
				ijkoG = oG[:, :, kS:kE, jS:jE, iS:iE]
				ijkoM = oM[:, kS:kE + 1, jS:jE + 1, iS:iE + 1]
				if (doGas0):
					ijkG0 = G0[:, :, kS:kE, jS:jE, iS:iE]
				# Write heavy variables
				oH5.create_dataset("Gas", data=ijkG)
				oH5.create_dataset("magFlux", data=ijkM)
				oH5.create_dataset("oGas", data=ijkoG)
				oH5.create_dataset("omagFlux", data=ijkoM)
				if (doGas0):
					oH5.create_dataset("Gas0", data=ijkG0)

				# Write subgrid
				oH5.create_dataset("X", data=ijkX)
				oH5.create_dataset("Y", data=ijkY)
				oH5.create_dataset("Z", data=ijkZ)

				# Transfer attributes to output
				for ak in iH5.attrs.keys():
					aStr = str(ak)
					oH5.attrs.create(ak, iH5.attrs[aStr])

				# Close this output file
				oH5.close()
	# Close input file
	iH5.close()


#Get full data from a tiled restart file
def PullRestartMPI(bStr, nRes, Ri, Rj, Rk, dIn=None, oH5=None):
	"""
	Reads and pulls data from multiple files and returns the combined data.

	Args:
		bStr (str): Base string used to generate file names.
		nRes (int): Number of resolutions.
		Ri (int): Number of iterations in the i-direction.
		Rj (int): Number of iterations in the j-direction.
		Rk (int): Number of iterations in the k-direction.
		dIn (str, optional): Directory path for input data files. Defaults to None.
		oH5 (str, optional): Output H5 file path. Defaults to None.

	Returns:
		tuple: A tuple containing the following arrays:
			- G: Gas array with shape (Ns, Nv, Nk, Nj, Ni).
			- M: MagFlux array with shape (3, Nk+1, Nj+1, Ni+1).
			- G0: Gas0 array with shape (Ns, Nv, Nk, Nj, Ni) or None if not available.
			- oG: oGas array with shape (Ns, Nv, Nk, Nj, Ni).
			- oM: omagFlux array with shape (3, Nk+1, Nj+1, Ni+1).
	"""
	doInit = True
	for i in range(Ri):
		for j in range(Rj):
			for k in range(Rk):
				fID = kh5.genName(bStr, i, j, k, Ri, Rj, Rk, nRes)
				print("Reading from %s" % (fID))

				# Start with input data
				if (dIn is not None):
					fIn = dIn + "/" + fID
				else:
					fIn = fID
				iH5 = h5py.File(fIn, 'r')

				if (doInit):
					Ns, Nv, Nkp, Njp, Nip = iH5['Gas'].shape
					doGas0 = ('Gas0' in iH5.keys())

					Nk = Rk * Nkp
					Nj = Rj * Njp
					Ni = Ri * Nip
					G = np.zeros((Ns, Nv, Nk, Nj, Ni))
					oG = np.zeros((Ns, Nv, Nk, Nj, Ni))
					if (doGas0):
						G0 = np.zeros((Ns, Nv, Nk, Nj, Ni))
					else:
						G0 = None
					M = np.zeros((3, Nk + 1, Nj + 1, Ni + 1))
					oM = np.zeros((3, Nk + 1, Nj + 1, Ni + 1))
					if (oH5 is not None):
						for ka in iH5.attrs.keys():
							aStr = str(ka)
							# print(aStr)
							oH5.attrs.create(ka, iH5.attrs[aStr])
					doInit = False
				iS = i * Nip
				iE = iS + Nip
				jS = j * Njp
				jE = jS + Njp
				kS = k * Nkp
				kE = kS + Nkp

				# print("MPI (%d,%d,%d) = [%d,%d]x[%d,%d]x[%d,%d]"%(i,j,k,iS,iE,jS,jE,kS,kE))

				G[:, :, kS:kE, jS:jE, iS:iE] = iH5['Gas'][:]
				oG[:, :, kS:kE, jS:jE, iS:iE] = iH5['oGas'][:]
				if (doGas0):
					G0[:, :, kS:kE, jS:jE, iS:iE] = iH5['Gas0'][:]

				M[:, kS:kE + 1, jS:jE + 1, iS:iE + 1] = iH5['magFlux'][:]
				oM[:, kS:kE + 1, jS:jE + 1, iS:iE + 1] = iH5['omagFlux'][:]

				# Close up
				iH5.close()

	if (not doGas0):
		G0 = None

	return G, M, G0, oG, oM


#Downscale a grid (with ghosts, k-j-i order)
def downGrid(X,Y,Z):
	"""
	Rescales a grid by reducing its size by half in each dimension.

	Args:
		X (ndarray): The X-coordinate grid.
		Y (ndarray): The Y-coordinate grid.
		Z (ndarray): The Z-coordinate grid.

	Returns:
		ndarray: The rescaled X-coordinate grid.
		ndarray: The rescaled Y-coordinate grid.
		ndarray: The rescaled Z-coordinate grid.
	"""
	Ngk,Ngj,Ngi = X.shape
	
	Nk = Ngk-2*NumG-1
	Nj = Ngj-2*NumG-1
	Ni = Ngi-2*NumG-1

	# Assuming LFM-style grid, get upper half plane
	
	xx = X[NumG,NumG:-NumG,NumG:-NumG].T
	yy = Y[NumG,NumG:-NumG,NumG:-NumG].T

	# Now half cells in r-phi polar
	rr = np.sqrt(xx**2.0+yy**2.0)
	pp = np.arctan2(yy,xx)

	NiD = Ni//2
	NjD = Nj//2
	NkD = Nk//2

	rrD = np.zeros((NiD+1,NjD+1))
	ppD = np.zeros((NiD+1,NjD+1))

	for i in range(NiD+1):
		for j in range(NjD+1):
			rrD[i,j] = rr[2*i,2*j]
			ppD[i,j] = pp[2*i,2*j]

	# Convert back to 2D Cartesian
	xxD = rrD*np.cos(ppD)
	yyD = rrD*np.sin(ppD)

	# Augment w/ 2D ghosts
	xxG,yyG = gg.Aug2D(xxD,yyD,doEps=True,TINY=TINY)

	# Augment w/ 3D ghosts
	X,Y,Z = gg.Aug3D(xxG,yyG,Nk=NkD,TINY=TINY)

	return X,Y,Z


#Upscale a grid (with ghosts, k-j-i order)
def upGrid(X,Y,Z):
	"""
	Rescales the grid by doubling the number of grid points in each dimension.

	Args:
		X (ndarray): The X-coordinate grid.
		Y (ndarray): The Y-coordinate grid.
		Z (ndarray): The Z-coordinate grid.

	Returns:
		ndarray: The rescaled X-coordinate grid.
		ndarray: The rescaled Y-coordinate grid.
		ndarray: The rescaled Z-coordinate grid.
	"""
	Ngk,Ngj,Ngi = X.shape
	
	Nk = Ngk-2*NumG-1
	Nj = Ngj-2*NumG-1
	Ni = Ngi-2*NumG-1
	

	#Assuming LFM-style grid, get upper half plane
	
	xx = X[NumG,NumG:-NumG,NumG:-NumG].T
	yy = Y[NumG,NumG:-NumG,NumG:-NumG].T

	#Now half cells in r-phi polar
	rr = np.sqrt(xx**2.0+yy**2.0)
	pp = np.arctan2(yy,xx)

	NiH = 2*Ni
	NjH = 2*Nj
	NkH = 2*Nk

	rrH = np.zeros((NiH+1,NjH+1))
	ppH = np.zeros((NiH+1,NjH+1))

	#Embed old points into new grid
	for i in range(Ni+1):
		for j in range(Nj+1):
			rrH[2*i,2*j] = rr[i,j]
			ppH[2*i,2*j] = pp[i,j]

	#Create I midpoints
	for i in range(Ni):
		rrH[2*i+1,:] = 0.5*( rrH[2*i,:] + rrH[2*i+2,:] )
		ppH[2*i+1,:] = 0.5*( ppH[2*i,:] + ppH[2*i+2,:] )

	#Create J midpoints
	for j in range(Nj):
		rrH[:,2*j+1] = 0.5*( rrH[:,2*j] + rrH[:,2*j+2])
		ppH[:,2*j+1] = 0.5*( ppH[:,2*j] + ppH[:,2*j+2])

	#Create I-J midpoints
	for i in range(Ni):
		for j in range(Nj):
			rrH[2*i+1,2*j+1] = 0.25*( rrH[2*i,2*j] + rrH[2*i,2*j+2] + rrH[2*i+2,2*j] + rrH[2*i+2,2*j+2] )
			ppH[2*i+1,2*j+1] = 0.25*( ppH[2*i,2*j] + ppH[2*i,2*j+2] + ppH[2*i+2,2*j] + ppH[2*i+2,2*j+2] )


	#Convert back to 2D Cartesian
	xxH = rrH*np.cos(ppH)
	yyH = rrH*np.sin(ppH)

	#Augment w/ 2D ghosts
	xxG,yyG = gg.Aug2D(xxH,yyH,doEps=True,TINY=TINY)

	#Augment w/ 3D ghosts
	X,Y,Z = gg.Aug3D(xxG,yyG,Nk=NkH,TINY=TINY)


	return X,Y,Z
	
#Downscale gBAvg variable (G) on grid X,Y,Z to halved grid.
def downVolt(G):
	"""
	Downscale the volt variables in the input array G.

	Args:
		G (ndarray): Input array of shape (Nd, Nk, Nj, Ni) representing the volt variables.

	Returns:
		ndarray: Downscaled array of shape (Nd, Nk//2, Nj//2, Ni) representing the downscaled volt variables.
	"""
	Nd, Nk, Nj, Ni = G.shape
	Gu = np.zeros((Nd, Nk//2, Nj//2, Ni))
	print("Downscaling volt variables gBAvg etc...")
	# Loop over fine grid
	for d in range(Nd):
		for i in range(Ni):
			for k in range(Nk//2):
				for j in range(Nj//2):
					j0 = 2*j
					k0 = 2*k
					Gjk = G[d, k0:k0+2, j0:j0+2, i].mean()
					Gu[d, k, j, i] = Gjk
	return Gu


#Upscale gBAvg variable (G) on grid X,Y,Z to doubled grid with simple linear interpolation.
#Typical size at quad for example: gBAvg                    Dataset {3, 128, 96, 1}
def upVolt(G):
	def upVolt(G):
		"""
		Upscales the volt variables in the input array G.

		Args:
			G (ndarray): Input array representing the volt variables.

		Returns:
			ndarray: Upscaled array of volt variables.

		Raises:
			None

		"""
		try:
			Nd, Nk, Nj, Ni = G.shape
			Gu = np.zeros((Nd, 2*Nk, 2*Nj, Ni))
			print("Upscaling volt variables gBAvg etc...")
			# Loop over coarse grid
			for d in range(Nd):
				for i in range(Ni):
					for k in range(Nk):
						for j in range(Nj):
							Gjk = G[d, k, j, i]
							j0 = 2*j
							k0 = 2*k
							Gu[d, k0:k0+2, j0:j0+2, i] = Gjk
		except:
			Nd, Nk, Nj, Ni, Nh = G.shape
			Gu = np.zeros((Nd, 2*Nk, 2*Nj, Ni, Nh))
			print("Upscaling volt variables gBHist etc...")
			print(G.shape)
			# Loop over coarse grid
			for h in range(Nh):
				for d in range(Nd):
					for i in range(Ni):
						for k in range(Nk):
							for j in range(Nj):
								Gjk = G[d, k, j, i, h]
								j0 = 2*j
								k0 = 2*k
								Gu[d, k0:k0+2, j0:j0+2, i, h] = Gjk
		return Gu
	try:
		Nd, Nk, Nj, Ni = G.shape
		Gu = np.zeros((Nd, 2*Nk, 2*Nj, Ni))
		print("Upscaling volt variables gBAvg etc...")
		# Loop over coarse grid
		for d in range(Nd):
			for i in range(Ni):
				for k in range(Nk):
					for j in range(Nj):
						Gjk = G[d, k, j, i]
						j0 = 2*j
						k0 = 2*k
						Gu[d, k0:k0+2, j0:j0+2, i] = Gjk
	except:
		Nd, Nk, Nj, Ni, Nh = G.shape
		Gu = np.zeros((Nd, 2*Nk, 2*Nj, Ni, Nh))
		print("Upscaling volt variables gBHist etc...")
		print(G.shape)
		# Loop over coarse grid
		for h in range(Nh):
			for d in range(Nd):
				for i in range(Ni):
					for k in range(Nk):
						for j in range(Nj):
							Gjk = G[d, k, j, i, h]
							j0 = 2*j
							k0 = 2*k
							Gu[d, k0:k0+2, j0:j0+2, i, h] = Gjk
	return Gu


#Downscale gas variable (G) on grid X,Y,Z (w/ ghosts) to halved grid
def downGas(X,Y,Z,G,Xd,Yd,Zd):
	"""
	Downscale gas variables from a coarse grid to a fine grid.

	Args:
		X (ndarray): X-coordinates of the coarse grid.
		Y (ndarray): Y-coordinates of the coarse grid.
		Z (ndarray): Z-coordinates of the coarse grid.
		G (ndarray): Gas variables on the coarse grid.
		Xd (ndarray): X-coordinates of the fine grid.
		Yd (ndarray): Y-coordinates of the fine grid.
		Zd (ndarray): Z-coordinates of the fine grid.

	Returns:
		ndarray: Gas variables downscaled to the fine grid.
	"""
	Ns,Nv,Nk,Nj,Ni = G.shape
	dV  = Volume(X ,Y ,Z )
	dVd = Volume(Xd,Yd,Zd)
	print("Volume ratio (Coarse/Fine) = %f"%(dVd.sum()/dV.sum()))
	Gd = np.zeros((Ns,Nv,Nk//2,Nj//2,Ni//2))
	print("Downscaling gas variables ...")

	#Loop over coarse grid
	for s in range(Ns):
		for v in range(Nv):
			print("\tDownscaling Species %d, Variable %d"%(s,v))
			for k in range(Nk//2):
				for j in range(Nj//2):
					for i in range(Ni//2):
						dVijk = dV[2*k:2*k+2,2*j:2*j+2,2*i:2*i+2] #Volumes of the finer subgrid
						dQijk = G[s,v,2*k:2*k+2,2*j:2*j+2,2*i:2*i+2] #stuff density in finer subgrid
						#Stuff in the subchunk
						QdV = (dVijk*dQijk).sum()
						Gd[s,v,k,j,i] = QdV/dVd[k,j,i] #Scale back to density
			#Test conservation
			print("\t\tCoarse (Total) = %e"%(Gd[s,v,:,:,:]*dVd).sum())
			print("\t\tFine   (Total) = %e"%(G [s,v,:,:,:]*dV ).sum())

	return Gd
	
#Upscale gas variable (G) on grid X,Y,Z (w/ ghosts) to doubled grid
def upGas(X, Y, Z, G, Xu, Yu, Zu):
	"""
	Upscales gas variables from a coarse grid to a finer grid.

	Args:
		X (ndarray): Coarse grid X coordinates.
		Y (ndarray): Coarse grid Y coordinates.
		Z (ndarray): Coarse grid Z coordinates.
		G (ndarray): Coarse grid gas variables.
		Xu (ndarray): Fine grid X coordinates.
		Yu (ndarray): Fine grid Y coordinates.
		Zu (ndarray): Fine grid Z coordinates.

	Returns:
		ndarray: Upscaled gas variables on the fine grid.

	Raises:
		None

	"""
	Ns, Nv, Nk, Nj, Ni = G.shape

	dV = Volume(X, Y, Z)
	dVu = Volume(Xu, Yu, Zu)

	print("Volume ratio (Coarse/Fine) = %f" % (dV.sum() / dVu.sum()))
	Gu = np.zeros((Ns, Nv, 2 * Nk, 2 * Nj, 2 * Ni))
	print("Upscaling gas variables ...")
	# Loop over coarse grid
	for s in range(Ns):
		for v in range(Nv):
			print("\tUpscaling Species %d, Variable %d" % (s, v))
			for k in range(Nk):
				for j in range(Nj):
					for i in range(Ni):
						QdV = G[s, v, k, j, i] * dV[k, j, i]
						dVijk = dVu[2 * k:2 * k + 2, 2 * j:2 * j + 2, 2 * i:2 * i + 2]  # Volumes of the finer subgrid
						vScl = dVijk.sum()  # Total volume of the finer subchunks
						QdVu = QdV * (dVijk / vScl)  # Give weighted contribution to each subcell
						Gu[s, v, 2 * k:2 * k + 2, 2 * j:2 * j + 2, 2 * i:2 * i + 2] = QdVu / dVijk  # Scale back to density
			# Test conservation
			print("\t\tCoarse (Total) = %e" % (G[s, v, :, :, :] * dV).sum())
			print("\t\tFine   (Total) = %e" % (Gu[s, v, :, :, :] * dVu).sum())
	return Gu


#Upscale RCMCPL variable
def upRCMCpl(Q, N=1):
	"""
	Upscales the input array `Q` for the Rice Convection Model Plasma Variable (RCM).

	Args:
		Q (numpy.ndarray): The input array to be upscaled.
		N (int, optional): The number of times to upscale the array. Default is 1.

	Returns:
		numpy.ndarray: The upscaled array.

	Examples:
		>>> Q = np.array([[1, 2], [3, 4]])
		>>> upRCMCpl(Q)
		array([[1., 1., 2.],
			   [1., 1., 2.],
			   [3., 3., 4.]])
	"""
	Nd = len(Q.shape)
	if (Nd >= 3):
		Nk, Ni, Nj = Q.shape
		Qr = np.zeros((Nk, Ni * 2, Nj))
		for n in range(Nk):
			Qr[n, :, :] = upRCM1D(Q[n, :, :])
	elif (Nd == 1):
		# Just leave it alone
		if (len(Q) == N):
			# Upscale the one dimension
			Q2D = np.zeros((N, 1))
			Q2D[:, 0] = Q
			Qr2D = upRCM1D(Q2D)
			Qr = Qr2D[:, 0]
		else:
			Qr = Q
	else:
		# 2D
		Qr = upRCM1D(Q)
	return Qr


#Upscale RCM variable
def upRCM(Q, Ni=1, Nj=1, Nk=1):
	"""
	Upscales the input array Q for Rice Convection Model (RCM).

	Args:
		Q (numpy.ndarray): The input array to be upscaled.
		Ni (int, optional): Number of columns in the output array. Default is 1.
		Nj (int, optional): Number of rows in the output array. Default is 1.
		Nk (int, optional): Number of slices in the output array. Default is 1.

	Returns:
		numpy.ndarray: The upscaled array.

	Notes:
		- If the input array Q has 3 dimensions or more, the upscaled array will have the shape (Nk, (Nj-2)*2+2, Ni).
		- If the input array Q has 1 dimension, the upscaled array will have the shape (Nj, 1).
		- If the input array Q has any other number of dimensions, the upscaled array will have the same shape as Q.
	"""
	Nd = len(Q.shape)
	if (Nd >= 3):
		Nk = Q.shape[0]
		Qr = np.zeros((Nk, (Nj-2)*2+2, Ni))
		for n in range(Nk):
			Qr[n, :, :] = upRCM1Dw(Q[n, :, :])
	elif (Nd == 1):
		if (len(Q) == Nj):
			Q2D = np.zeros((Nj, 1))
			Q2D[:, 0] = Q
			Qr2D = upRCM1Dw(Q2D)
			Qr = Qr2D[:, 0]
		else:
			Qr = Q
	else:
		Qr = upRCM1Dw(Q)
	return Qr


#Upscale first dimension of 2D array w/ strange rcm-wrap
def upRCM1Dw(Q):
	"""
	Upscale the input array Q for RCM using its wrapping method.

	Parameters:
	Q (ndarray): Input array of shape (Ni, Nj).

	Returns:
	Qr (ndarray): Upscaled array of shape (Nri, Nj).
	"""
	Ni, Nj = Q.shape
	Nip = Ni - 2
	Nri = Nip * 2 + 2
	Qr = np.zeros((Nri, Nj))
	Qr[0:Nri-2, :] = upRCM1D(Q[0:-2, :])

	Qr[Nri-2, :] = Qr[0, :]
	Qr[Nri-1, :] = Qr[1, :]
	return Qr


#Upscale just first dimension of 2D array
def upRCM1D(Q):
	"""
	Rescales a 1D array by duplicating each element.

	Args:
		Q (ndarray): Input array of shape (Ni, Nj).

	Returns:
		ndarray: Rescaled array of shape (2*Ni, Nj).
	"""
	Ni, Nj = Q.shape
	Qr = np.zeros((Ni*2, Nj))
	for j in range(Nj):
		for i in range(Ni):
			Qij = Q[i, j]
			i0 = 2*i
			j0 = j
			Qr[i0:i0+2, j0] = Qij
	return Qr


#Upscale mix variable
def upMIX(Q):
	"""
	Upscales a MIX Q by a factor of 2 in both dimensions.

	Args:
		Q (ndarray): Input matrix of shape (Ni, Nj).

	Returns:
		ndarray: Upscaled matrix of shape (2*Ni, 2*Nj).
	"""
	Ni, Nj = Q.shape

	Qr = np.zeros((2*Ni, 2*Nj))
	for j in range(Nj):
		for i in range(Ni):
			Qij = Q[i, j]
			i0 = 2*i
			j0 = 2*j
			Qr[i0:i0+2, j0:j0+2] = Qij

	return Qr


#Downscale mix variable
def downMIX(Q):
	"""
	Downmixes a given matrix by averaging 2x2 blocks of values.

	Args:
		Q (ndarray): Input matrix of shape (Ni, Nj).

	Returns:
		ndarray: Downmixed matrix of shape (Ni//2, Nj//2).
	"""
	Ni, Nj = Q.shape
	Qr = np.zeros((Ni//2, Nj//2))

	# Loop over coarse grid
	for j in range(Nj//2):
		for i in range(Ni//2):
			i0 = 2*i
			j0 = 2*j
			Qij = Q[i0:i0+2, j0:j0+2].mean()
			Qr[i, j] = Qij
	return Qr

	
#Return cell centered volume (active only) from grid X,Y,Z (w/ ghosts)
def Volume(Xg, Yg, Zg):
	"""
	Calculate the volume of a grid.

	Args:
		Xg (ndarray): X-coordinate grid.
		Yg (ndarray): Y-coordinate grid.
		Zg (ndarray): Z-coordinate grid.

	Returns:
		ndarray: Volume of the grid.
	"""

	Ngk, Ngj, Ngi = Xg.shape

	Nk = Ngk - 2 * NumG - 1
	Nj = Ngj - 2 * NumG - 1
	Ni = Ngi - 2 * NumG - 1

	X = Xg[NumG:-NumG, NumG:-NumG, NumG:-NumG]
	Y = Yg[NumG:-NumG, NumG:-NumG, NumG:-NumG]
	Z = Zg[NumG:-NumG, NumG:-NumG, NumG:-NumG]

	print("Calculating volume of grid of size (%d,%d,%d)" % (Ni, Nj, Nk))

	dV = np.zeros((Nk, Nj, Ni))
	ijkPts = np.zeros((8, 3))

	# Assuming LFM-like symmetry
	k = 0
	for j in range(Nj):
		for i in range(Ni):
			ijkPts[:, 0] = X[k:k + 2, j:j + 2, i:i + 2].flatten()
			ijkPts[:, 1] = Y[k:k + 2, j:j + 2, i:i + 2].flatten()
			ijkPts[:, 2] = Z[k:k + 2, j:j + 2, i:i + 2].flatten()

			dV[:, j, i] = ConvexHull(ijkPts, incremental=False).volume

	return dV


#Downscale magnetic fluxes (M) on grid X,Y,Z (w/ ghosts) to halved grid
def downFlux(X,Y,Z,M,Xu,Yu,Zu):
	"""
	Downscale the face fluxes of a given 4D array.

	Args:
		X (ndarray): Array of X coordinates.
		Y (ndarray): Array of Y coordinates.
		Z (ndarray): Array of Z coordinates.
		M (ndarray): 4D array of face fluxes.
		Xu (ndarray): Array of X coordinates for the upscaled grid.
		Yu (ndarray): Array of Y coordinates for the upscaled grid.
		Zu (ndarray): Array of Z coordinates for the upscaled grid.

	Returns:
		ndarray: 4D array of downscaled face fluxes.
	"""

	Nd,Nkc,Njc,Nic = M.shape
	Nk = Nkc-1
	Nj = Njc-1
	Ni = Nic-1

	Md = np.zeros((Nd,Nk//2+1,Nj//2+1,Ni//2+1))

	#Loop over coarse grid cells
	print("Downscaling face fluxes ...")
	for k in range(Nk//2):
		for j in range(Nj//2):
			for i in range(Ni//2):
				ip = 2*i
				jp = 2*j
				kp = 2*k

				#West/east i faces
				Md[IDIR,k  ,j  ,i  ] = M[IDIR,kp:kp+2,jp:jp+2,ip  ].sum()
				Md[IDIR,k  ,j  ,i+1] = M[IDIR,kp:kp+2,jp:jp+2,ip+2].sum()

				#South/north j faces
				Md[JDIR,k  ,j  ,i  ] = M[JDIR,kp:kp+2,jp  ,ip:ip+2].sum()
				Md[JDIR,k  ,j+1,i  ] = M[JDIR,kp:kp+2,jp+2,ip:ip+2].sum()
				
				#Bottom/top k faces
				Md[KDIR,k  ,j  ,i  ] = M[KDIR,kp  ,jp:jp+2,ip:ip+2].sum()
				Md[KDIR,k+1,j  ,i  ] = M[KDIR,kp+2,jp:jp+2,ip:ip+2].sum()

	return Md

#Upscale magnetic fluxes (M) on grid X,Y,Z (w/ ghosts) to doubled grid
def upFlux(X,Y,Z,M,Xu,Yu,Zu):
	"""
	Upscale magnetic fluxes (M) on grid X,Y,Z (w/ ghosts) to doubled grid.

	Args:
		X (array): X coordinates.
		Y (array): Y coordinates.
		Z (array): Z coordinates.
		M (array): Input fluxes.
		Xu (array): Upscaled X coordinates.
		Yu (array): Upscaled Y coordinates.
		Zu (array): Upscaled Z coordinates.

	Returns:
		array: Upscaled fluxes.
	"""
	Nd,Nkc,Njc,Nic = M.shape
	Nk = Nkc-1
	Nj = Njc-1
	Ni = Nic-1

	Mu = np.zeros((Nd,2*Nk+1,2*Nj+1,2*Ni+1))

	#Loop over coarse grid
	print("Upscaling face fluxes ...")
	for k in range(Nk):
		for j in range(Nj):
			for i in range(Ni):
				ip = 2*i
				jp = 2*j
				kp = 2*k

				#West i face (4)
				Mu[IDIR,kp:kp+2,jp:jp+2,ip] = 0.25*M[IDIR,k,j,i]
				#East i face (4)
				Mu[IDIR,kp:kp+2,jp:jp+2,ip+2] = 0.25*M[IDIR,k,j,i+1]

				#South j face (4)
				Mu[JDIR,kp:kp+2,jp,ip:ip+2] = 0.25*M[JDIR,k,j,i]
				#North j face (4)
				Mu[JDIR,kp:kp+2,jp+2,ip:ip+2] = 0.25*M[JDIR,k,j+1,i]

				#Bottom k face (4)
				Mu[KDIR,kp,jp:jp+2,ip:ip+2] = 0.25*M[KDIR,k,j,i]
				#Top k face (4)
				Mu[KDIR,kp+2,jp:jp+2,ip:ip+2] = 0.25*M[KDIR,k+1,j,i]

				#Now all exterior faces are done
				#12 remaining interior faces

				#Interior i faces (4)
				Mu[IDIR,kp:kp+2,jp:jp+2,ip+1] = 0.5*( Mu[IDIR,kp:kp+2,jp:jp+2,ip] + Mu[IDIR,kp:kp+2,jp:jp+2,ip+2] )

				#Interior j faces (4)
				Mu[JDIR,kp:kp+2,jp+1,ip:ip+2] = 0.5*( Mu[JDIR,kp:kp+2,jp,ip:ip+2] + Mu[JDIR,kp:kp+2,jp+2,ip:ip+2] )

				#Interior k faces (4)
				Mu[KDIR,kp+1,jp:jp+2,ip:ip+2] = 0.5*( Mu[KDIR,kp,jp:jp+2,ip:ip+2] + Mu[KDIR,kp+2,jp:jp+2,ip:ip+2] )
	#MaxDiv(M)

	return Mu

#Calculates maximum divergence of mag flux data
def MaxDiv(M):
	"""
	Calculate the maximum and mean divergence of a given 3D array.

	Args:
		M (ndarray): The input 3D array of shape (Nd, Nkc, Njc, Nic), where
						Nd, Nkc, Njc, Nic are the dimensions of the array.

	Returns:
		ndarray: The divergence array of shape (Nk, Nj, Ni), where
					Nk = Nkc - 1, Nj = Njc - 1, and Ni = Nic - 1.

	"""
	Nd,Nkc,Njc,Nic = M.shape
	Nk = Nkc-1
	Nj = Njc-1
	Ni = Nic-1

	Div = np.zeros((Nk,Nj,Ni))
	for k in range(Nk):
		for j in range(Nj):
			for i in range(Ni):
				Div[k,j,i] = M[IDIR,k,j,i+1]-M[IDIR,k,j,i]+M[JDIR,k,j+1,i]-M[JDIR,k,j,i]+M[KDIR,k+1,j,i]-M[KDIR,k,j,i]

	mDiv = np.abs(Div).max()
	bDiv = np.abs(Div).mean()
	print("Max/Mean divergence = %e,%e"%(mDiv,bDiv))
	print("Sum divergence = %e"%(Div.sum()))

	return Div
