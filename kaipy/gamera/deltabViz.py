#Various helper routines for making ground DB pics

# Standard modules
import sys

# Third-party modules
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker

# Kaipy modules
import kaipy.kaiH5 as kh5
import kaipy.kaiTools as ktools

rad2deg = 180.0/np.pi
dbMag = 1000.0
dbLin = 10.0
jMag = 0.5

#Do various decorations on a plot to make it look pretty
def DecorateDBAxis(Ax, crs, utDT):
	"""
	Decorates the given axis with gridlines, coastlines, and nightshade feature.

	Args:
		Ax (matplotlib.axes.Axes): The axis to decorate.
		crs (cartopy.crs.Projection): The projection of the axis.
		utDT (datetime.datetime): The datetime object representing the Universal Time.

	Returns:
		None
	"""
	Ax.add_feature(Nightshade(utDT, alpha=0.10))
	Ax.coastlines(resolution='110m', color='black', linewidth=0.25)

	gl = Ax.gridlines(crs, draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
	gl.top_labels = False
	gl.left_labels = False
	gl.xlines = False
	gl.ylines = False
	gl.xlocator = mticker.FixedLocator([-120, 0, 120])
	gl.ylocator = mticker.FixedLocator([-45, 0, 45])
	# Ax.contour(LonC,LatC,smlon,[0.5,90,180,270],linewidths=1.25, linestyles='--',alpha=0.5,colors='grey')


#Get coordinate information
def GetCoords(fname):
	"""
	Retrieves the coordinates and radius of the Earth from the given file.

	Args:
		fname (str): The file name or path.

	Returns:
		tuple: A tuple containing the coordinates ID and the radius of the Earth.

	"""
	cStr = kh5.PullAtt(fname,"CoordinatesID",s0=None)
	Re   = kh5.PullAtt(fname,"Re",s0=None) #km
	try:
		CoordID = cStr.decode('utf-8')
	except (UnicodeDecodeError, AttributeError):
		CoordID = cStr
	return CoordID, Re


#Generate uniform lat/lon grid
def GenUniformLL(dbdata, k0=0):
	"""
	Generate uniform latitude and longitude grids based on the given data.

	Args:
		dbdata (numpy.ndarray): The input data.
		k0 (int, optional): The index of the data to use (default is 0).

	Returns:
		tuple: A tuple containing the latitude grid, longitude grid, latitude grid at cell centers, and longitude grid at cell centers.
	"""
	X0 = dbdata.X[:, :, k0]
	Y0 = dbdata.Y[:, :, k0]
	Z0 = dbdata.Z[:, :, k0]

	NLat = Z0.shape[0] - 1
	NLon = Z0.shape[1] - 1

	R0 = np.sqrt(X0**2.0 + Y0**2.0 + Z0**2.0)
	LonI = np.arctan2(Y0, X0) * rad2deg
	LatI = np.arcsin(Z0 / R0) * rad2deg

	LonI = np.linspace(0, 360, NLon + 1)
	LatI = LatI[:, 0]

	LonC = 0.5 * (LonI[0:-1] + LonI[1:])
	LatC = 0.5 * (LatI[0:-1] + LatI[1:])

	return LatI, LonI, LatC, LonC


def CheckLevel(dbdata, k0, Re):
	"""
	Check if the given vertical level is valid and return the corresponding height.

	Args:
		dbdata (numpy.ndarray): The database data object.
		k0 (int): The vertical level to check.
		Re (float): The Earth's radius.

	Returns:
		float: The height corresponding to the given vertical level.

	Raises:
		SystemExit: If the given vertical level is invalid.

	"""
	NLat = dbdata.Z.shape[0] - 1
	NLon = dbdata.Z.shape[1] - 1
	Nz = dbdata.Z.shape[2] - 1

	Rcc = dbdata.GetRootVar("Radcc")

	if k0 >= Nz:
		print("Invalid vertical level, only %d levels found!" % (Nz))
		sys.exit()
	else:
		h0 = (Rcc.mean() * Re) - Re
		# print("Height = %6.2f [km]"%(h0))
	return h0


def GenTStr(Ax, fname, nStp, doVerb=True):
	"""
	Generate a title string for a plot based on the given parameters.

	Args:
		Ax (matplotlib.axes.Axes): The axes object on which the title will be displayed.
		fname (str): The file name.
		nStp (int): The time step.
		doVerb (bool, optional): Whether to enable verbose mode. Defaults to True.

	Returns:
		str: The generated title string.

	"""
	# Get MJD to UT
	MJD = kh5.tStep(fname, nStp, aID="MJD")
	utS = ktools.MJD2UT([MJD])
	utDT = utS[0]

	# Get SMRs/SMLs
	SMR = kh5.PullAtt(fname, "SMR", nStp)
	SMR06 = kh5.PullAtt(fname, "SMR_06", nStp)
	SMR12 = kh5.PullAtt(fname, "SMR_12", nStp)
	SMR18 = kh5.PullAtt(fname, "SMR_18", nStp)
	SMR00 = kh5.PullAtt(fname, "SMR_00", nStp)

	SML = kh5.PullAtt(fname, "SML", nStp)
	SMU = kh5.PullAtt(fname, "SMU", nStp)
	SME = kh5.PullAtt(fname, "SME", nStp)

	tStr = utDT.strftime("%m/%d/%Y\n%H:%M:%S")
	aStr = "Auroral Indices [nT]\nSME = %8.2f\nSML = %8.2f\nSMU = %8.2f" % (SME, SML, SMU)
	rStr = "Ring Current Indices [nT]\nSMR       = %8.2f\nDawn-Dusk = %8.2f\nDay-Night = %8.2f" % (
		SMR,
		SMR06 - SMR18,
		SMR12 - SMR00,
	)

	iSize = "medium"
	Ax.set_title(tStr, fontsize="x-large", loc="center")
	Ax.set_title(aStr, fontsize=iSize, loc="right", fontname="monospace")
	Ax.set_title(rStr, fontsize=iSize, loc="left", fontname="monospace")

	return aStr + "\n" + rStr