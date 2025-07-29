#Gamera post-processing routines
#Get data from serial/MPI gamera output

# Standard modules
import glob

# Third-party modules
import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from alive_progress import alive_bar

# Kaipy modules
from kaipy.kaiTools import MJD2UT
import kaipy.kdefs as kdefs
import kaipy.kaiH5 as kh5
#Object to use to pull data from HDF5 structure (serial or mpi)

#Initialize,
#gIn = gampp.GameraPipe(fdir,ftag)
#Set grid from data
#gIn.GetGrid()
#V = gIn.GetVar("D",stepnum)
#doFast=True skips various data scraping
idStr = "_0000_0000_0000.gam.h5"

class GameraPipe(object):
	"""
	GameraPipe class represents a pipe object for Gamera.

	Args:
		fdir (str): Directory to h5 files.
		ftag (str): Stub of h5 files.
		doFast (bool): Flag indicating whether to use fast mode. Default is False.
		doVerbose (bool): Flag indicating whether to print verbose output. Default is True.
		doParallel (bool): Flag indicating whether to use parallel processing. Default is False.
		nWorkers (int): Number of workers for parallel processing. Default is 4.
	"""
	#Initialize GP object
	#fdir = directory to h5 files
	#ftag = stub of h5 files
	def __init__(self,fdir,ftag,doFast=False,doVerbose=True,doParallel=False,nWorkers=4):
		"""
		GameraPipe class represents a pipe object for Gamera.

		Args:
			fdir (str): Directory to h5 files.
			ftag (str): Stub of h5 files.
			doFast (bool): Flag indicating whether to use fast mode. Default is False.
			doVerbose (bool): Flag indicating whether to print verbose output. Default is True.
			doParallel (bool): Flag indicating whether to use parallel processing. Default is False.
			nWorkers (int): Number of workers for parallel processing. Default is 4.
		"""
		self.fdir = fdir
		self.ftag = ftag

		#Here just create variables
		#-----------
		#Ranks
		self.isMPI = False
		self.Nr = 0
		self.Ri = 1 ; self.Rj = 1 ; self.Rk = 1
		#Global/local cells
		self.is2D = False
		self.Ni = 0 ; self.Nj = 0 ; self.Nk = 0
		self.dNi= 0 ; self.dNj= 0 ; self.dNk= 0

		#Variables/slices
		self.Nv = 0 ; self.Nt = 0 ; self.Nv0 = 0
		self.T = [] ; self.vIDs = [] ; self.v0IDs = []
		self.s0 = 0 ; self.sFin = 0
		self.sids = np.array([])

		#Grids
		self.X = [] ; self.Y = [] ; self.Z = []

		self.gridLoaded = False
		self.doFast = doFast
		self.doParallel = doParallel
		self.UnitsID = "NONE"
		self.nWorkers = nWorkers
		#Example file data
		self.f0 = []

		#Stubs for MJD stuff
		self.hasMJD = False
		self.MJDs = []

		#Stub for UT date time object
		self.hasUT = False
		self.UT = []

		#Scrape data from directory
		self.OpenPipe(doVerbose)


	def OpenPipe(self,doVerbose=True):
		"""
		Opens a pipe for reading data from a database file.

		Parameters:
			doVerbose (bool): Flag indicating whether to print verbose output. Default is True.

		Returns:
			None

		Raises:
			None

		"""
		if (doVerbose):
			print("Opening pipe: %s : %s"%(self.fdir,self.ftag))

		#Test for serial (either old or new naming convention)
		fOld    = "%s/%s.h5"%(self.fdir,self.ftag)
		fNew    = "%s/%s.gam.h5"%(self.fdir,self.ftag)

		if ( len(glob.glob(fOld)) == 1):
			#Found old-style serial
			self.isMPI = False
			if (doVerbose):
				print("Found serial database")
			f0 = fOld
		elif ( len(glob.glob(fNew)) == 1):
			#Found new-style serial
			self.isMPI = False
			self.ftag = self.ftag + ".gam" #Add .gam to tag
			if (doVerbose):
				print("Found serial database")
			f0 = fNew
		else:
			print("%s not found, looking for MPI database"%(fOld))
			self.isMPI = True
			sStr = "%s/%s_*%s"%(self.fdir,self.ftag,idStr)

			fIns = glob.glob(sStr)
			if (len(fIns)>1):
				print("This shouldn't happen, bailing ...")
			if (len(fIns) == 0):
				print("No MPI database found, all out of options, bailing ...")
				quit()
			f0 = fIns[0]
			Ns = [int(s) for s in f0.split('_') if s.isdigit()]

			self.Ri = Ns[-5]
			self.Rj = Ns[-4]
			self.Rk = Ns[-3]
			self.Nr = self.Ri*self.Rj*self.Rk
			if (doVerbose):
				print("\tFound %d = (%d,%d,%d) ranks"%(self.Nr,self.Ri,self.Rj,self.Rk))

		#In either case, f0 is defined.  use it to get per file stuff
		self.Nt,sids = kh5.cntSteps(f0)
		if (self.doFast):
			self.T = np.zeros(self.Nt)
		else:
			self.T = kh5.getTs(f0,sids)

		self.sids = sids
		self.s0 = sids.min()
		self.sFin = sids.max()
		if (doVerbose):
			print("Found %d timesteps\n\tTime = [%f,%f]"%(self.Nt,self.T.min(),self.T.max()))
			print("\tSteps = [%d,%d]"%(sids.min(),sids.max()))

		#Get MJD if present
		MJDs = kh5.getTs(f0,sids,"MJD",-np.inf)
		if (MJDs.max()>0):
			self.hasMJD = True
			self.MJDs = MJDs
			self.hasUT = True
			self.UT = MJD2UT(MJDs)


		#Get grid stuff
		Dims = kh5.getDims(f0)
		Nd = len(Dims)
		self.dNi = Dims[0]-1
		self.dNj = Dims[1]-1
		if (Nd>2):
			self.dNk = Dims[2]-1

		else:
			self.dNk = 0
			self.Rk = 1
		self.Ni = self.dNi*self.Ri
		self.Nj = self.dNj*self.Rj
		self.Nk = self.dNk*self.Rk


		#Variables
		self.v0IDs = kh5.getRootVars(f0)
		self.vIDs  = kh5.getVars(f0,sids.min())
		self.Nv0 = len(self.v0IDs)
		self.Nv  = len(self.vIDs)
		if (Nd>2):
			self.is2D = False
		else:
			self.is2D = True
		if (doVerbose):
			if (Nd>2):
				nCells = self.Ni*self.Nj*self.Nk
			else:
				nCells = self.Ni*self.Nj
			print("Grid size = (%d,%d,%d)"%(self.Ni,self.Nj,self.Nk))
			print("\tCells = %e"%(nCells))
			print("Variables (Root/Step) = (%d,%d)"%(self.Nv0,self.Nv))
			print("\tRoot: %s"%(self.v0IDs))
			print("\tStep: %s"%(self.vIDs))

		self.SetUnits(f0)
		if (doVerbose):
			print("Units Type = %s"%(self.UnitsID))
			#print("Pulling grid ...")
		self.GetGrid(doVerbose)
		self.f0 = f0

	def SetUnits(self, f0):
		"""
		Sets the units for the given file.

		Args:
			f0 (str): The file path.

		Returns:
			None
		"""

		uID = kh5.PullAtt(f0, "UnitsID",a0="CODE") #Setting default
		#with h5py.File(f0, 'r') as hf:
		#	uID = hf.attrs.get("UnitsID", "CODE")
		print(f'GameraPipe: {uID}')
		if not isinstance(uID, str):
			print('setting UnitsID')
			self.UnitsID = uID.decode('utf-8')
		else:
			self.UnitsID = uID

	def GetGridParallel(self, doVerbose):
			"""Parallel read of grid datasets

			This method performs a parallel read of grid datasets using multiple processes. It populates the `X`, `Y`, and `Z` arrays with the corresponding dataset values.

			Args:
				doVerbose (bool): Flag indicating whether to display verbose output.

			Returns:
				None

			"""


			if (self.is2D):
				self.X = np.zeros((self.Ni+1,self.Nj+1))
				self.Y = np.zeros((self.Ni+1,self.Nj+1))
				coords = [self.X,self.Y]
			else:
				self.X = np.zeros((self.Ni+1,self.Nj+1,self.Nk+1))
				self.Y = np.zeros((self.Ni+1,self.Nj+1,self.Nk+1))
				self.Z = np.zeros((self.Ni+1,self.Nj+1,self.Nk+1))
				coords = [(self.X,'X'),(self.Y,'Y'),(self.Z,'Z')]

			if (doVerbose):
				#print("Del = (%d,%d,%d)"%(self.dNi,self.dNj,self.dNk))
				titStr = "%s/Grid"%(self.ftag)
			else:
				titStr = None

			files = []
			#print("Bounds = (%d,%d,%d,%d,%d,%d)"%(iS,iE,jS,jE,kS,kE))
			for (i,j,k) in itertools.product(range(self.Ri),range(self.Rj),range(self.Rk)):
				files.append(((self.fdir + "/" + kh5.genName(self.ftag,i,j,k,self.Ri,self.Rj,self.Rk),[i,j,k])))
			
			NrX = max(self.Nr,1)

			for data,vID in coords:
				datasets = []
				with alive_bar(NrX,title=f"{titStr}/{vID}".ljust(kdefs.barLab),length=kdefs.barLen,bar=kdefs.barDef2) as bar, \
					ProcessPoolExecutor(max_workers=self.nWorkers) as executor:
					futures = [executor.submit(kh5.PullVarLoc, fIn, vID, loc=loc) for fIn, loc in files]
					for future in as_completed(futures):
						datasets.append(future.result())
						bar()
				
				for dataset,loc in datasets:
					i = loc[0]
					j = loc[1]
					k = loc[2]

					iS = i*self.dNi
					jS = j*self.dNj
					kS = k*self.dNk
					iE = iS+self.dNi
					jE = jS+self.dNj
					kE = kS+self.dNk

					if (self.is2D):
						data[iS:iE+1,jS:jE+1] = dataset
					else:
						data[iS:iE+1,jS:jE+1,kS:kE+1] = dataset

	def GetGrid(self, doVerbose):
			"""Load Grid from Gamera HDF5 file

			Args:
				doVerbose (bool): Flag indicating whether to display verbose output.

			Returns:
				None

			"""

			if(not self.gridLoaded):
				if (self.isMPI and self.doParallel):
					self.GetGridParallel(doVerbose)
				else:
					if (self.is2D):
						self.X = np.zeros((self.Ni+1,self.Nj+1))
						self.Y = np.zeros((self.Ni+1,self.Nj+1))
					else:
						self.X  = np.zeros((self.Ni+1,self.Nj+1,self.Nk+1))
						self.Y  = np.zeros((self.Ni+1,self.Nj+1,self.Nk+1))
						self.Z  = np.zeros((self.Ni+1,self.Nj+1,self.Nk+1))
						self.dV = np.zeros((self.Ni  ,self.Nj  ,self.Nk  ))
					if (doVerbose):
						#print("Del = (%d,%d,%d)"%(self.dNi,self.dNj,self.dNk))
						titStr = "%s/Grid"%(self.ftag)
					else:
						titStr = None
					NrX = max(self.Nr,1)
					with alive_bar(NrX,title=titStr,length=kdefs.barLen,bar=kdefs.barDef2) as bar:
						for (i,j,k) in itertools.product(range(self.Ri),range(self.Rj),range(self.Rk)):
							iS = i *self.dNi
							jS = j *self.dNj
							kS = k *self.dNk
							iE = iS+self.dNi
							jE = jS+self.dNj
							kE = kS+self.dNk
							#print("Bounds = (%d,%d,%d,%d,%d,%d)"%(iS,iE,jS,jE,kS,kE))
							if (self.isMPI):
								fIn = self.fdir + "/" + kh5.genName(self.ftag,i,j,k,self.Ri,self.Rj,self.Rk)
							else:
								fIn = self.fdir + "/" + self.ftag + ".h5"
							if (self.is2D):
								self.X[iS:iE+1,jS:jE+1] = kh5.PullVar(fIn,"X")
								self.Y[iS:iE+1,jS:jE+1] = kh5.PullVar(fIn,"Y")
							else:
								self.X[iS:iE+1,jS:jE+1,kS:kE+1] = kh5.PullVar(fIn,"X")
								self.Y[iS:iE+1,jS:jE+1,kS:kE+1] = kh5.PullVar(fIn,"Y")
								self.Z[iS:iE+1,jS:jE+1,kS:kE+1] = kh5.PullVar(fIn,"Z")
								self.dV[iS:iE , jS:jE , kS:kE ] = kh5.PullVar(fIn, "dV")
							bar()
			else:
				print("Grid Previously Loaded")
			self.gridLoaded = True

	def GetVarParallel(self,vID,sID=None,vScl=None,doVerb=True):
			''' Parallel read of Var

			This method performs a parallel read of a variable from the dataset.
			

			Args:
				vID (str): The ID of the variable to read.
				sID (int, optional): The step ID. Defaults to None.
				vScl (float, optional): The scaling factor for the variable. Defaults to None.
				doVerb (bool, optional): Whether to display progress bars and verbose output. Defaults to True.

			Returns:
				ndarray: The variable data as a NumPy array.

			'''
	
			if (doVerb):
				if (sID is None):
					titStr = "%s/%s"%(self.ftag,vID)
					
				else:
					titStr = "%s/Step#%d/%s"%(self.ftag,sID,vID)
			else:
				titStr = ''

			if (self.is2D):
				V = np.zeros((self.Ni,self.Nj))
			else:
				V = np.zeros((self.Ni,self.Nj,self.Nk))

			files = []
			#print("Bounds = (%d,%d,%d,%d,%d,%d)"%(iS,iE,jS,jE,kS,kE))
			for (i,j,k) in itertools.product(range(self.Ri),range(self.Rj),range(self.Rk)):
				files.append(((self.fdir + "/" + kh5.genName(self.ftag,i,j,k,self.Ri,self.Rj,self.Rk),[i,j,k])))
			
			NrX = max(self.Nr,1)

			datasets = []
			with alive_bar(NrX,title=titStr.ljust(kdefs.barLab),length=kdefs.barLen,bar=kdefs.barDef,disable=not doVerb) as bar, \
					ProcessPoolExecutor(max_workers=self.nWorkers) as executor:
				futures = [executor.submit(kh5.PullVarLoc, fIn, vID, sID, loc=loc) for fIn, loc in files]
				for future in as_completed(futures):
					datasets.append(future.result())
					bar()
			
			for dataset,loc in datasets:
				i = loc[0]
				j = loc[1]
				k = loc[2]

				iS = i*self.dNi
				jS = j*self.dNj
				kS = k*self.dNk
				iE = iS+self.dNi
				jE = jS+self.dNj
				kE = kS+self.dNk
				
				if (self.is2D):
					V[iS:iE,jS:jE] = dataset
				else:
					V[iS:iE,jS:jE,kS:kE] = dataset
			if (vScl is not None):
				V = vScl*V
			return V

	#Get 3D variable "vID" from Step# sID

	def GetVar(self, vID, sID=None, vScl=None, doVerb=True):
		"""Reads a variable with the given name.

		Args:
			vID (str): The name of the variable to be read.
			sID (int, optional): The step ID. Default is None.
			vScl (float, optional): The scaling factor for the variable. Default is None.
			doVerb (bool, optional): A flag indicating whether to display progress bar and verbose output. Default is True.

		Returns:
			np.ndarray: The variable data read from the file.

		"""


		if (self.is2D):
			V = np.zeros((self.Ni,self.Nj))
		else:
			V = np.zeros((self.Ni,self.Nj,self.Nk))

		if (self.isMPI and self.doParallel):
			V = self.GetVarParallel(vID,sID,vScl,doVerb)
		else:
			if (doVerb):
				if (sID is None):
					titStr = "%s/%s"%(self.ftag,vID)
					
				else:
					titStr = "%s/Step#%d/%s"%(self.ftag,sID,vID)
			else:
				titStr = ''
			NrX = max(self.Nr,1)
			with alive_bar(NrX,title=titStr.ljust(kdefs.barLab),length=kdefs.barLen,bar=kdefs.barDef,disable=not doVerb) as bar:
				for (i,j,k) in itertools.product(range(self.Ri),range(self.Rj),range(self.Rk)):

					iS = i*self.dNi
					jS = j*self.dNj
					kS = k*self.dNk
					iE = iS+self.dNi
					jE = jS+self.dNj
					kE = kS+self.dNk
					#print("Bounds = (%d,%d,%d,%d,%d,%d)"%(iS,iE,jS,jE,kS,kE))
					if (self.isMPI):
						fIn = self.fdir + "/" + kh5.genName(self.ftag,i,j,k,self.Ri,self.Rj,self.Rk)
					else:
						fIn = self.fdir + "/" + self.ftag + ".h5"

					if (self.is2D):
						V[iS:iE,jS:jE] = kh5.PullVar(fIn,vID,sID)

					else:
						V[iS:iE,jS:jE,kS:kE] = kh5.PullVar(fIn,vID,sID)
					bar()

		return V


	#FIXME: Currently pulling whole 3D array and then slicing, lazy
	def GetSlice(self, vID, sID, ijkdir='idir', n=1, vScl=None, doVerb=True):
		"""Get variable slice of constant i, j, k
		Directions = idir/jdir/kdir strings
		Indexing = (1, Nijk)

		Args:
			vID (str): The variable ID.
			sID (int): The step ID.
			ijkdir (str, optional): The slice direction. Defaults to 'idir'.
			n (int, optional): The slice index. Defaults to 1.
			vScl (float, optional): The variable scale. Defaults to None.
			doVerb (bool, optional): Whether to print verbose information. Defaults to True.

		Returns:
			Vs (ndarray): The sliced variable.

		"""
		sDirs = ['IDIR', 'JDIR', 'KDIR']
		ijkdir = ijkdir.upper()

		if (not (ijkdir in sDirs)):
			print("Invalid slice direction, defaulting to I")
			ijkdir = 'IDIR'
		if (sID is None):
			cStr = "Reading %s/%s" % (self.ftag, vID)
		else:
			cStr = "Reading %s/Step#%d/%s" % (self.ftag, sID, vID)
		V = self.GetVar(vID, sID, vScl, doVerb=False)
		# Now slice
		np = n - 1  # Convert from Fortran to Python indexing
		if (ijkdir == "IDIR"):
			Vs = V[np, :, :]
			sStr = "(%d,:,:)" % (n)
		elif (ijkdir == "JDIR"):
			Vs = V[:, np, :]
			sStr = "(:,%d,:)" % (n)
		elif (ijkdir == "KDIR"):
			Vs = V[:, :, np]
			sStr = "(:,:,%d)" % (n)
		if (doVerb):
			print(cStr + sStr)
		return Vs

	#Wrappers for root variables (or just set sID to "None")
	def GetRootVar(self, vID, vScl=None, doVerb=True):
		"""
		Get the root variable with the given ID.

		Args:
			vID (str): The ID of the variable.
			vScl (float, optional): The scale of the variable.
			doVerb (bool, optional): Whether to perform verbose logging.

		Returns:
			np.ndarray: The root variable with the given ID.
		"""
		V = self.GetVar(vID, sID=None, vScl=vScl, doVerb=doVerb)
		return V

	
	def GetRootSlice(self, vID, ijkdir='idir', n=1, vScl=None, doVerb=True):
		"""
		Retrieves the root slice of a given variable.

		Args:
			vID (int): The ID of the variable.
			ijkdir (str): The direction of the slice. Defaults to 'idir'.
			n (int): The number of slices to retrieve. Defaults to 1.
			vScl (float): The scaling factor for the variable. Defaults to None.
			doVerb (bool): Whether to print verbose output. Defaults to True.

		Returns:
			Vs: The root slice of the variable.
		"""
		Vs = self.GetSlice(vID, sID=None, ijkdir=ijkdir, n=n, vScl=vScl, doVerb=doVerb)
		return Vs
