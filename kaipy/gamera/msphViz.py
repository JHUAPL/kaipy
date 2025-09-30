#Various helper routines for magnetosphere quick look plots

# Standard modules
import argparse
from argparse import RawTextHelpFormatter
import os

# Third-party modules
import matplotlib as mpl
import numpy as np

# Kaipy modules
import kaipy.kaiViz as kv
import kaipy.kaiTools as kt
import kaipy.gamera.magsphere as msph
import kaipy.remix.remix as remix


dbMax = 25.0
dbCM = "RdGy_r"
bzCM = "bwr"
cLW = 0.25
bz0Col = "magenta"
mpiCol = "deepskyblue"

jMax = 10.0 #Max current for contours

eMax = 5.0  #Max current for contours

#Default pressure colorbar
vP = kv.genNorm(vMin=1.0e-2,vMax=10.0,doLog=True)
szStrs = ['small','std','big','bigger','fullD','dm']
szBds = {}
szBds["std"]      = [-40.0 ,20.0,2.0]
szBds["big"]      = [-100.0,20.0,2.0]
szBds["bigger"]   = [-200.0,25.0,2.0]
szBds["fullD"]    = [-300.0,30.0,3.0] # full domain for double res
szBds["small"]    = [-10.0 , 5.0,2.0]
szBds["dm"]       = [-30.0 ,10.0,40.0/15.0]

#Add different size options to argument
def AddSizeArgs(parser):
	"""
	Add size arguments to the parser.

	Args:
		parser (argparse.ArgumentParser): The argument parser object.

	Returns:
		None
	"""
	parser.add_argument('-size', type=str, default="std", choices=szStrs, help="Domain bounds options (default: %(default)s)")


#Return domain size from parsed arguments
def GetSizeBds(args):
	"""
	Calculate the bounding box size based on the given size argument.

	Args:
		args (dict): A dictionary containing the size argument.

	Returns:
		list: A list containing the bounding box coordinates [xTail, xSun, -yMax, yMax].
	"""

	szStr = args.size
	szBd = szBds[szStr]

	xTail = szBd[0]
	xSun  = szBd[1]
	yMax = (xSun-xTail)/szBd[2]
	xyBds = [xTail,xSun,-yMax,yMax]

	return xyBds

#Plot absolute error in the requested, or given, equatorial field
def PlotEqErrAbs(gsphP, gsphO, nStp, xyBds, Ax, fieldNames, AxCB=None, doClear=True, doDeco=True, vMin=1e-9, vMax=1e-4, doLog=True, doVerb=True):
	"""
	PlotEqErrAbs function plots the absolute error between two gsph objects.

	Args:
		gsphP (gsph): The gsph object representing the predicted values.
		gsphO (gsph): The gsph object representing the observed values.
		nStp (int): Step number to use 
		xyBds (list): The bounds of the x and y axes.
		Ax (matplotlib.axes.Axes): The matplotlib Axes object to plot on.
		fieldNames (list): A list of field names.
		AxCB (matplotlib.axes.Axes, optional): The matplotlib Axes object to add the colorbar to. (default: None)
		doClear (bool, optional): Whether to clear the Axes object before plotting. (default: True)
		doDeco (bool, optional): Whether to add additional decorations to the plot. (default: True)
		vMin (float, optional): The minimum value for the colorbar. (default: 1e-9)
		vMax (float, optional): The maximum value for the colorbar. (default: 1e-4)
		doLog (bool, optional): Whether to use logarithmic scale for the colorbar. (default: True)
		doVerb (bool, optional): Whether to print verbose output. (default: True)

	Returns:
		dataAbs (ndarray): The absolute error data.
	"""
	# specify two gsph objects as gsphPredicted (gsphP) and gsphObserved (gsphO)
	#
	normAbs = kv.genNorm(vMin, vMax, doLog=doLog)
	cmapAbs = "PRGn"
	if doLog:
		cmapAbs = "magma"

	if AxCB is not None:
		# Add the colorbar to AxCB
		AxCB.clear()

		cbString = ""
		for fn in fieldNames:
			cbString = cbString + "'" + fn + "', "
		cbString = cbString + " Absolute Error"
		kv.genCB(AxCB, normAbs, cbString, cM=cmapAbs)
	# Now do main plotting
	if doClear:
		Ax.clear()
	dataAbs = None
	for fn in fieldNames:
		dataP = gsphP.EggSlice(fn, nStp, doEq=True, doVerb=doVerb)
		dataO = gsphO.EggSlice(fn, nStp, doEq=True, doVerb=doVerb)
		if dataAbs is None:
			dataAbs = np.square(dataO - dataP)
		else:
			dataAbs = dataAbs + np.square(dataO - dataP)
	dataAbs = np.sqrt(dataAbs)
	Ax.pcolormesh(gsphP.xxi, gsphP.yyi, dataAbs, cmap=cmapAbs, norm=normAbs)

	kv.SetAx(xyBds, Ax)

	if doDeco:
		kv.addEarth2D(ax=Ax)
		Ax.set_xlabel('SM-X [Re]')
		Ax.set_ylabel('SM-Y [Re]')
	return dataAbs

#Plot relative error in the requested, or given, equatorial field
def PlotEqErrRel(gsphP, gsphO, nStp, xyBds, Ax, fieldNames, AxCB=None, doClear=True, doDeco=True, vMin=1e-16, vMax=1, doLog=True, doVerb=True):
	"""
	PlotEqErrRel function plots the relative error between two gsph objects.

	Args:
		gsphP (gsph): The gsph object representing the predicted values.
		gsphO (gsph): The gsph object representing the observed values.
		nStp (int): Step number to use 
		xyBds (list): The bounds of the x and y axes.
		Ax (matplotlib.axes.Axes): The matplotlib axis object to plot on.
		fieldNames (list): A list of field names to plot.
		AxCB (matplotlib.axes.Axes, optional): The matplotlib axis object to add the colorbar to. (default: None)
		doClear (bool, optional): Whether to clear the axis before plotting. (default: True)
		doDeco (bool, optional): Whether to add additional decorations to the plot. (default: True)
		vMin (float, optional): The minimum value for the colorbar. (default: 1e-16)
		vMax (float, optional): The maximum value for the colorbar. (default: 1)
		doLog (bool, optional): Whether to use logarithmic scale for the colorbar. (default: True)
		doVerb (bool, optional): Whether to print verbose output. (default: True)

	Returns:
		dataRel (ndarray): The relative error data.

	"""
	# specify two gsph objects as gsphPredicted (gsphP) and gsphObserved (gsphO)
	#
	normRel = kv.genNorm(vMin, vMax, doLog=doLog)
	cmapRel = "PRGn"
	if doLog:
		cmapRel = "magma"

	if AxCB is not None:
		# Add the colorbar to AxCB
		AxCB.clear()
		cbString = ""
		for fn in fieldNames:
			cbString = cbString + "'" + fn + "', "
		cbString = cbString + " Relative Error"
		kv.genCB(AxCB, normRel, cbString, cM=cmapRel)
	# Now do main plotting
	if doClear:
		Ax.clear()
	dataAbs = None
	dataBase = None
	for fn in fieldNames:
		dataP = gsphP.EggSlice(fn, nStp, doEq=True, doVerb=doVerb)
		dataO = gsphO.EggSlice(fn, nStp, doEq=True, doVerb=doVerb)
		if dataAbs is None:
			dataBase = np.square(dataP)
			dataAbs = np.square(dataO - dataP)
		else:
			dataBase = dataBase + np.square(dataP)
			dataAbs = dataAbs + np.square(dataO - dataP)
	dataBase = np.sqrt(dataBase)
	dataAbs = np.sqrt(dataAbs)
	# math breaks when Base field is exactly 0
	dataBase[dataBase == 0] = np.finfo(np.float32).tiny
	dataRel = np.absolute(dataAbs / dataBase)
	Ax.pcolormesh(gsphP.xxi, gsphP.yyi, dataRel, cmap=cmapRel, norm=normRel)

	kv.SetAx(xyBds, Ax)

	if doDeco:
		kv.addEarth2D(ax=Ax)
		Ax.set_xlabel('SM-X [Re]')
		Ax.set_ylabel('SM-Y [Re]')
	return dataRel

#Plot absolute error along the requested logical axis
def PlotLogicalErrAbs(gsphP, gsphO, nStp, Ax, fieldNames, meanAxis, AxCB=None, doClear=True, doDeco=True, vMin=1e-16, vMax=1, doLog=True, doVerb=True):
	"""
	Plot the absolute error between two gsph objects.

	Args:
		gsphP (gsph): The gsph object representing the predicted values.
		gsphO (gsph): The gsph object representing the observed values.
		nStp (int): Step number to use 
		Ax (matplotlib.axes.Axes): The matplotlib Axes object to plot on.
		fieldNames (list): A list of field names.
		meanAxis (int): The axis along which to calculate the mean.
		AxCB (matplotlib.axes.Axes, optional): The matplotlib Axes object to add the colorbar to.
		doClear (bool, optional): Whether to clear the Axes before plotting. (default: True)
		doDeco (bool, optional): Whether to add decorations to the plot. (default: True)
		vMin (float, optional): The minimum value for the colorbar. (default: 1e-16)
		vMax (float, optional): The maximum value for the colorbar. (default: 1)
		doLog (bool, optional): Whether to use logarithmic scale for the colorbar. (default: True)
		doVerb (bool, optional): Whether to print verbose output. (default: True)

	Returns:
		ndarray: The calculated absolute error data.

	"""
	#specify two gsph objects as gsphPredicted (gsphP) and gsphObserved (gsphO)
	#
	normAbs = kv.genNorm(vMin,vMax,doLog=doLog)
	cmapAbs = "PRGn"
	if doLog:
		cmapAbs = "magma"

	if (AxCB is not None):
		#Add the colorbar to AxCB
		AxCB.clear()
		cbString = ""
		for fn in fieldNames:
			cbString = cbString + "'" + fn + "', "
		cbString = cbString + " Absolute Error along "
		if meanAxis == 0:
			cbString = cbString + "I axis"
		elif meanAxis == 1:
			cbString = cbString + "J axis"
		elif meanAxis == 2:
			cbString = cbString + "K axis"
		kv.genCB(AxCB,normAbs,cbString,cM=cmapAbs)
	#Now do main plotting
	if (doClear):
		Ax.clear()
	dataAbs = CalcTotalErrAbs(gsphP,gsphO,nStp,fieldNames,doVerb=doVerb,meanAxis=meanAxis)
	dataAbs = np.transpose(dataAbs) # transpose to put I/J on the horizontal axis
	Ax.pcolormesh(dataAbs,cmap=cmapAbs,norm=normAbs)

	if (doDeco):
		if meanAxis == 0:
			Ax.set_xlabel('J Indices')
			Ax.set_ylabel('K Indices')
		elif meanAxis == 1:
			Ax.set_xlabel('I Indices')
			Ax.set_ylabel('K Indices')
		elif meanAxis == 2:
			Ax.set_xlabel('I Indices')
			Ax.set_ylabel('J Indices')
	return dataAbs

#Plot relative error along the requested logical axis
def PlotLogicalErrRel(gsphP, gsphO, nStp, Ax, fieldNames, meanAxis, AxCB=None, doClear=True, doDeco=True, vMin=1e-16, vMax=1, doLog=True, doVerb=True):
	"""
	Plot the logical error relative to the observed values.

	Args:
		gsphP (gsph): The predicted gsph object.
		gsphO (gsph): The observed gsph object.
		nStp (int): Step number to use 
		Ax (matplotlib.axes.Axes): The matplotlib Axes object to plot on.
		fieldNames (list): A list of field names.
		meanAxis (int): The axis along which to calculate the mean.
		AxCB (matplotlib.axes.Axes, optional): The matplotlib Axes object to add the colorbar to. (default: None)
		doClear (bool, optional): Whether to clear the Axes object before plotting. (default: True)
		doDeco (bool, optional): Whether to add decorations to the plot. (default: True)
		vMin (float, optional): The minimum value for the colorbar. (default: 1e-16)
		vMax (float, optional): The maximum value for the colorbar. (default: 1)
		doLog (bool, optional): Whether to use logarithmic scale for the colorbar. (default: True)
		doVerb (bool, optional): Whether to print verbose output. (default: True)

	Returns:
		dataRel (ndarray): The calculated relative error data.

	"""
	# specify two gsph objects as gsphPredicted (gsphP) and gsphObserved (gsphO)
	#
	normRel = kv.genNorm(vMin, vMax, doLog=doLog)
	cmapRel = "PRGn"
	if doLog:
		cmapRel = "magma"

	if AxCB is not None:
		# Add the colorbar to AxCB
		AxCB.clear()
		cbString = ""
		for fn in fieldNames:
			cbString = cbString + "'" + fn + "', "
		cbString = cbString + " Relative Error along "
		if meanAxis == 0:
			cbString = cbString + "I axis"
		elif meanAxis == 1:
			cbString = cbString + "J axis"
		elif meanAxis == 2:
			cbString = cbString + "K axis"
		kv.genCB(AxCB, normRel, cbString, cM=cmapRel)
	
	# Now do main plotting
	if doClear:
		Ax.clear()
	dataRel = CalcTotalErrRel(gsphP, gsphO, nStp, fieldNames, doVerb=doVerb, meanAxis=meanAxis)
	dataRel = np.transpose(dataRel)  # transpose to put I/J on the horizontal axis
	Ax.pcolormesh(dataRel, cmap=cmapRel, norm=normRel)

	if doDeco:
		if meanAxis == 0:
			Ax.set_xlabel('J Indices')
			Ax.set_ylabel('K Indices')
		elif meanAxis == 1:
			Ax.set_xlabel('I Indices')
			Ax.set_ylabel('K Indices')
		elif meanAxis == 2:
			Ax.set_xlabel('I Indices')
			Ax.set_ylabel('J Indices')
	
	return dataRel

#Calculate total cumulative absolute error between two cases
def CalcTotalErrAbs(gsphP, gsphO, nStp, fieldNames, doVerb=True, meanAxis=None):
	"""
	Calculate the total absolute error between two sets of data.

	Args:
		gsphP (gsph): The first set of data.
		gsphO (gsph): The second set of data.
		nStp (int): Step number to use 
		fieldNames (list): A list of field names.
		doVerb (bool, optional): Whether to print verbose output (default is True).
		meanAxis (int, optional): The axis along which to calculate the mean (default is None).

	Returns:
		float: The mean of the absolute error.

	"""
	dataAbs = None
	for fn in fieldNames:
		dataP = gsphP.GetVar(fn, nStp, doVerb=doVerb)
		dataO = gsphO.GetVar(fn, nStp, doVerb=doVerb)
		if dataAbs is None:
			dataAbs = np.square(dataO - dataP)
		else:
			dataAbs = dataAbs + np.square(dataO - dataP)
	dataAbs = np.sqrt(dataAbs)
	return np.mean(dataAbs, axis=meanAxis)

#Calculate total cumulative relative error between two cases
def CalcTotalErrRel(gsphP, gsphO, nStp, fieldNames, doVerb=True, meanAxis=None):
	"""
	Calculate the total relative error for given field names.

	Args:
		gsphP (object): The gsphP object.
		gsphO (object): The gsphO object.
		nStp (int): Step number to use 
		fieldNames (list): A list of field names.
		doVerb (bool): Whether to print verbose output. Default is True.
		meanAxis (int): The axis along which to compute the mean. Default is None.

	Returns:
		float: The mean of the total relative error.

	"""
	dataAbs = None
	dataBase = None
	for fn in fieldNames:
		dataP = gsphP.GetVar(fn, nStp, doVerb=doVerb)
		dataO = gsphO.GetVar(fn, nStp, doVerb=doVerb)
		if dataAbs is None:
			dataBase = np.square(dataP)
			dataAbs = np.square(dataO - dataP)
		else:
			dataBase = dataBase + np.square(dataP)
			dataAbs = dataAbs + np.square(dataO - dataP)
	dataBase = np.sqrt(dataBase)
	dataAbs = np.sqrt(dataAbs)
	# math breaks when Base field is exactly 0
	dataBase[dataBase == 0] = np.finfo(np.float32).tiny
	dataRel = np.absolute(dataAbs / dataBase)
	return np.mean(dataRel, axis=meanAxis)

#Plot equatorial field
def PlotEqB(gsph, nStp, xyBds, Ax, AxCB=None, doClear=True, doDeco=True, doBz=False):
	"""
	Plot the equatorial magnetic field.

	Args:
		gsph (object): The gsph object containing the data.
		nStp (int): Step number to use 
		xyBds (tuple): The x and y boundaries.
		Ax (object): The axis object for the main plot.
		AxCB (object, optional): The axis object for the colorbar. Defaults to None.
		doClear (bool, optional): Whether to clear the axis before plotting. Defaults to True.
		doDeco (bool, optional): Whether to add decorations to the plot. Defaults to True.
		doBz (bool, optional): Whether to plot the vertical field (Bz) or the residual field (dbz). Defaults to False.

	Returns:
		Bz (array): The plotted data.
	"""
	vBZ = kv.genNorm(dbMax)
	vDB = kv.genNorm(dbMax)

	if (AxCB is not None):
		# Add the colorbar to AxCB
		AxCB.clear()
		if (doBz):
			kv.genCB(AxCB, vBZ, "Vertical Field [nT]", cM=bzCM, Ntk=7)
		else:
			kv.genCB(AxCB, vDB, "Residual Field [nT]", cM=dbCM, Ntk=7)

	# Now do main plotting
	if (doClear):
		Ax.clear()

	Bz = gsph.EggSlice("Bz", nStp, doEq=True)
	if (doBz):
		Ax.pcolormesh(gsph.xxi, gsph.yyi, Bz, cmap=bzCM, norm=vBZ)
	else:
		dbz = gsph.DelBz(nStp)
		Ax.pcolormesh(gsph.xxi, gsph.yyi, dbz, cmap=dbCM, norm=vDB)

	Ax.contour(kv.reWrap(gsph.xxc), kv.reWrap(gsph.yyc), kv.reWrap(Bz), [0.0], colors=bz0Col, linewidths=cLW)

	kv.SetAx(xyBds, Ax)

	if (doDeco):
		kv.addEarth2D(ax=Ax)
		Ax.set_xlabel('SM-X [Re]')
		Ax.set_ylabel('SM-Y [Re]')

	return Bz

def PlotMerid(gsph, nStp, xyBds, Ax, doDen=False, doRCM=False, AxCB=None, doClear=True, doDeco=True, doSrc=False):
	"""
	Plot the meridional slice of a spherical grid.

	Args:
		gsph (object): The spherical grid object.
		nStp (int): Step number to use 
		xyBds (tuple): The bounds of the plot.
		Ax (object): The matplotlib axis object to plot on.
		doDen (bool, optional): Whether to plot density (default: False).
		doRCM (bool, optional): Whether to use RCM normalization (default: False).
		AxCB (object, optional): The matplotlib axis object to add the colorbar to (default: None).
		doClear (bool, optional): Whether to clear the axis before plotting (default: True).
		doDeco (bool, optional): Whether to add Earth decoration to the plot (default: True).
		doSrc (bool, optional): Whether to plot source density or pressure (default: False).
	"""
	CMx = "viridis"
	if doDen:
		if doRCM:
			vN = kv.genNorm(vMin=1.0, vMax=1.0e+3, doLog=True)
		else:
			vN = kv.genNorm(0, 25)
		if doSrc:
			vID = "SrcD_RING"
			cbStr = "Source Density [#/cc]"
		else:
			vID = "D"
			cbStr = "Density [#/cc]"
		Q = gsph.EggSlice(vID, nStp, doEq=False)
	else:
		vN = vP
		if doSrc:
			vID = "SrcP_RING"
			cbStr = "Source Pressure [nPa]"
		else:
			vID = "P"
			cbStr = "Pressure [nPa]"
		Q = gsph.EggSlice(vID, nStp, doEq=False)
	if AxCB is not None:
		# Add the colorbar to AxCB
		AxCB.clear()
		kv.genCB(AxCB, vN, cbStr, cM=CMx)
	Ax.pcolormesh(gsph.xxi, gsph.yyi, Q, cmap=CMx, norm=vN)

	kv.SetAx(xyBds, Ax)
	if doDeco:
		kv.addEarth2D(ax=Ax)
		Ax.set_xlabel('SM-X [Re]')
		Ax.set_ylabel('SM-Z [Re]')
		Ax.yaxis.tick_right()
		Ax.yaxis.set_label_position('right')

def PlotJyXZ(gsph, nStp, xyBds, Ax, AxCB=None, jScl=None, doDeco=True):
	"""
	Plot the Jy component of a spherical grid on the XZ plane.

	Args:
		gsph (object): The spherical grid object.
		nStp (int): Step number to use 
		xyBds (tuple): The bounds of the X and Y axes.
		Ax (object): The matplotlib axis object to plot on.
		AxCB (object): The matplotlib colorbar axis object.
		jScl (float): The scaling factor for the Jy component.
		doDeco (bool): Whether to add additional decorations to the plot.

	Returns:
		None
	"""
	if jScl is None:
		# Just assuming current scaling is nA/m2
		jScl = 1.0
	vJ = kv.genNorm(jMax)
	jCMap = "PRGn"
	Nc = 15
	cVals = np.linspace(-jMax, jMax, Nc)

	if AxCB is not None:
		AxCB.clear()
		kv.genCB(AxCB, vJ, "Jy [nA/m2]", cM=jCMap)
	Q = jScl * gsph.EggSlice("Jy", nStp, doEq=False)
	# Zero out first shell b/c bad derivative
	print(Q.shape)
	Q[0:2, :] = 0.0
	# Ax.contour(kv.reWrap(gsph.xxc), kv.reWrap(gsph.yyc), kv.reWrap(Q), cVals, norm=vJ, cmap=jCMap, linewidths=cLW)
	Ax.pcolormesh(gsph.xxi, gsph.yyi, Q, norm=vJ, cmap=jCMap)
	kv.SetAx(xyBds, Ax)
	if doDeco:
		kv.addEarth2D(ax=Ax)
		Ax.set_xlabel('SM-X [Re]')
		Ax.set_ylabel('SM-Z [Re]')
		Ax.yaxis.tick_right()
		Ax.yaxis.set_label_position('right')

#Plot equatorial azimuthal electric field
def PlotEqEphi(gsph, nStp, xyBds, Ax, AxCB=None, doClear=True, doDeco=True):
	"""
	PlotEqEphi function plots the electric field in the phi direction (E_phi) on a given axis.

	Args:
		gsph (object): The gsph object representing the spherical grid.
		nStp (int): Step number to use 
		xyBds (tuple): The bounds of the x and y coordinates.
		Ax (object): The axis object on which to plot the electric field.
		AxCB (object, optional): The axis object for the colorbar. (default: None)
		doClear (bool, optional): Whether to clear the axis before plotting. (default: True)
		doDeco (bool, optional): Whether to add decorations to the plot. (default: True)

	Returns:
		None
	"""

	vE = kv.genNorm(eMax)
	vEMap = "PRGn"
	if (AxCB is not None):
		# Add the colorbar to AxCB
		AxCB.clear()
		kv.genCB(AxCB, vE, r"E$_{phi}$ [mV/m]", cM=vEMap)

	# Now do main plotting
	if (doClear):
		Ax.clear()
	Bx = gsph.EggSlice("Bx", nStp, doEq=True)
	By = gsph.EggSlice("By", nStp, doEq=True)
	Bz = gsph.EggSlice("Bz", nStp, doEq=True)
	Vx = gsph.EggSlice("Vx", nStp, doEq=True)
	Vy = gsph.EggSlice("Vy", nStp, doEq=True)
	Vz = gsph.EggSlice("Vz", nStp, doEq=True)

	# calculating some variables to plot
	# E = -VxB
	Ex = -(Vy * Bz - Vz * By) * 0.001  # [mV/m]
	Ey = (Vx * Bz - Vz * Bx) * 0.001
	Ez = -(Vx * By - Vy * Bx) * 0.001

	# coordinate transform
	ppc = np.arctan2(gsph.yyc, gsph.xxc)
	theta = np.pi  # eq plane
	Er, Et, Ep = kt.xyz2rtp(ppc, theta, Ex, Ey, Ez)

	Ax.pcolormesh(gsph.xxi, gsph.yyi, Ep, cmap=vEMap, norm=vE)

	kv.SetAx(xyBds, Ax)

	if (doDeco):
		kv.addEarth2D(ax=Ax)
		Ax.set_xlabel('SM-X [Re]')
		Ax.set_ylabel('SM-Y [Re]')
		Ax.yaxis.tick_right()
		Ax.yaxis.set_label_position('right')

#Add MPI contours
def PlotMPI(gsph, Ax, ashd=0.5):
	"""
	Add the MPI (Message Passing Interface) boundaries to the visualization.

	Args:
		gsph (object): The gsph object containing the data for plotting.
		Ax (object): The matplotlib Axes object on which to plot.
		ashd (float, optional): The alpha shading value for the plot (default: 0.5).

	Returns:
		None
	"""
	gCol = mpiCol
	for i in range(gsph.Ri):
		i0 = i * gsph.dNi
		Ax.plot(gsph.xxi[i0, :], gsph.yyi[i0, :], mpiCol, linewidth=cLW, alpha=ashd)

	if gsph.Rj > 1:
		for j in range(1, gsph.Rj):
			j0 = j * gsph.dNj
			Ax.plot(gsph.xxi[:, j0], gsph.yyi[:, j0], gCol, linewidth=cLW, alpha=ashd)
			Ax.plot(gsph.xxi[:, j0], -gsph.yyi[:, j0], gCol, linewidth=cLW, alpha=ashd)
		# X-axis (+)
		Ax.plot(gsph.xxi[:, 0], gsph.yyi[:, 0], gCol, linewidth=cLW, alpha=ashd)
		# X-axis (-)
		j0 = gsph.Rj * gsph.dNj
		Ax.plot(gsph.xxi[:, j0], gsph.yyi[:, j0], gCol, linewidth=cLW, alpha=ashd)


def AddIonBoxes(gs, ion):
	"""
	Adds ion boxes to the given gs (GridSpec) object.

	Args:
		gs (GridSpec): The GridSpec object to which the ion boxes will be added.
		ion: The ion object.

	Returns:
		None
	"""
	gsRM = gs.subgridspec(20, 20)

	wXY = 6
	dX = 1
	dY = 1

	# Northern
	ion.init_vars('NORTH')
	ax = ion.plot('current', gs=gsRM[dY:dY+wXY, dX:dX+wXY], doInset=True)

	# Southern
	ion.init_vars('SOUTH')
	ax = ion.plot('current', gs=gsRM[-dY-wXY:-dY, dX:dX+wXY], doInset=True)

def plotPlane(gsph, data, xyBds, Ax, AxCB, var='D', vMin=None, vMax=None, doDeco=True, cmap='viridis', doLog=False, midp=None, doVert=False):
	"""
	Base plotting routine for gsph plane

	Args:
		gsph (object): The gsph object representing the plane.
		data (array): The data to be plotted.
		xyBds (tuple): The bounds of the x and y axes.
		Ax (object): The axis on which to plot the plane.
		AxCB (object): The colorbar axis.
		var (str, optional): The variable to be plotted (default is 'D').
		vMin (float, optional): The minimum value for the colorbar (default is None).
		vMax (float, optional): The maximum value for the colorbar (default is None).
		doDeco (bool, optional): Whether to apply decorations to the plot (default is True).
		cmap (str, optional): The colormap to be used (default is 'viridis').
		doLog (bool, optional): Whether to use logarithmic scaling for the colorbar (default is False).
		midp (float, optional): The midpoint value for the colorbar (default is None).
		doVert (bool, optional): Whether to display the colorbar vertically (default is False).

	Returns:
		None
	"""
	if (AxCB is not None):
		AxCB.clear()
	if (not midp):
		if (vMin is None):
			vMin = np.min(data)
		if (vMax is None):
			vMax = np.max(data)
	else:
		if ((vMin is None) and (vMax is None)):
			vMax = np.max(np.abs([np.min(data), np.max(data)]))
			vMin = -1.0 * vMax
	vNorm = kv.genNorm(vMin, vMax=vMax, doLog=doLog, midP=midp)
	kv.genCB(AxCB, vNorm, cbT=var, cM=cmap, Ntk=7, doVert=doVert)
	Ax.pcolormesh(gsph.xxi, gsph.yyi, data, cmap=cmap, norm=vNorm)
	kv.SetAx(xyBds, Ax)
	return

def plotXY(gsph, nStp, xyBds, Ax, AxCB, var='D', vMin=None, vMax=None, doDeco=True, cmap='viridis', doLog=False, midp=None, doVert=False):
	"""
	Plot a 2D slice of data on the XY plane of the gsph object.

	Args:
		gsph (object): The gsph object containing the data.
		nStp (int): Step number to use for the slice.
		xyBds (tuple): The bounds of the XY plane.
		Ax (object): The matplotlib axis to plot on.
		AxCB (object): The matplotlib colorbar axis.
		var (str, optional): The variable to plot (default is 'D').
		vMin (float, optional): The minimum value for the colorbar (default is None).
		vMax (float, optional): The maximum value for the colorbar (default is None).
		doDeco (bool, optional): Whether to add additional decorations to the plot (default is True).
		cmap (str, optional): The colormap to use for the plot (default is 'viridis').
		doLog (bool, optional): Whether to use a logarithmic scale for the colorbar (default is False).
		midp (float, optional): The midpoint value for the colorbar (default is None).
		doVert (bool, optional): Whether to display the colorbar vertically (default is False).

	Returns:
		data: The 2D slice of data.
	"""
	data = gsph.EggSlice(var, nStp, doEq=True)
	plotPlane(gsph, data, xyBds, Ax, AxCB, var, vMin=vMin, vMax=vMax, doDeco=doDeco, cmap=cmap, doLog=doLog, midp=midp, doVert=doVert)
	if doDeco:
		kv.addEarth2D(ax=Ax)
		Ax.set_xlabel('SM_X [Re]')
		Ax.set_ylabel('SM-Y [Re]')
	return data

def plotXZ(gsph, nStp, xzBds, Ax, AxCB, var='D', vMin=None, vMax=None, doDeco=True, cmap='viridis', doLog=False, midp=None, doVert=False):
	"""
	Plot a 2D slice of data in the XZ plane.

	Args:
		gsph (GameraSphere): The GameraSphere object containing the data.
		nStp (int): Step number to use for the slice.
		xzBds (tuple): The bounds of the XZ plane in the form (xmin, xmax, zmin, zmax).
		Ax (Axes): The matplotlib Axes object to plot on.
		AxCB (Axes): The matplotlib Axes object to use for the colorbar.
		var (str): The variable to plot. Default is 'D'.
		vMin (float): The minimum value for the colorbar. Default is None.
		vMax (float): The maximum value for the colorbar. Default is None.
		doDeco (bool): Whether to add Earth decoration to the plot. Default is True.
		cmap (str): The colormap to use. Default is 'viridis'.
		doLog (bool): Whether to use a logarithmic scale for the colorbar. Default is False.
		midp (float): The midpoint value for the colorbar. Default is None.
		doVert (bool, optional): Whether to display the colorbar vertically (default is False).

	Returns:
		data (numpy.ndarray): The 2D slice of data.
	"""
	data = gsph.EggSlice(var,nStp,doEq=False)
	plotPlane(gsph,data,xzBds,Ax,AxCB,var,vMin=vMin,vMax=vMax,doDeco=doDeco,cmap=cmap,doLog=doLog,midp=midp,doVert=doVert)
	if (doDeco):
		kv.addEarth2D(ax=Ax)
	Ax.set_xlabel('SM_X [Re]')
	Ax.set_ylabel('SM-Z [Re]')
	return data
