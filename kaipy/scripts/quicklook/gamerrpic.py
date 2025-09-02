#!/usr/bin/env python
#Make video of error between two Gamera cases

# Standard modules
import argparse
from argparse import RawTextHelpFormatter
import os
import errno

# Third-party modules
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import kaipy.kaiViz as kv
import matplotlib.gridspec as gridspec
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Kaipy modules
import kaipy.gamera.msphViz as mviz
import kaipy.gamera.magsphere as msph
import kaipy.gamera.rcmpp as rcmpp

cLW = 0.25
def create_command_line_parser():
	"""Create the command-line argument parser.

	Create the parser for command-line arguments.

	Returns:
		argparse.ArgumentParser: Command-line argument parser for this script.
	"""
	#Defaults
	fdir1 = os.getcwd()
	ftag1 = "msphere"
	fdir2 = os.getcwd()
	ftag2 = "msphere"
	nStp=1
	fieldNames = "Bx, By, Bz"
	noMPI = False

	MainS = """Creates simple multi-panel figure for Gamera magnetosphere run
	Left Panel - Residual vertical magnetic field
	Right Panel - Pressure (or density) and hemispherical insets
	"""

	parser = argparse.ArgumentParser(description=MainS, formatter_class=RawTextHelpFormatter)
	parser.add_argument('-d1',type=str,metavar="directory1",default=fdir1,help="Directory to read first dataset from (default: %(default)s)")
	parser.add_argument('-id1',type=str,metavar="runid1",default=ftag1,help="RunID of first dataset (default: %(default)s)")
	parser.add_argument('-d2',type=str,metavar="directory2",default=fdir2,help="Directory to read second dataset from (default: %(default)s)")
	parser.add_argument('-id2',type=str,metavar="runid2",default=ftag2,help="RunID of second dataset (default: %(default)s)")
	parser.add_argument('-n',type=int,metavar="nStp",default=nStp,help="Step number to plot (default: %(default)s)")
	parser.add_argument('-f',type=str,metavar="fieldnames",default=fieldNames,help="Comma-separated fields to plot (default: %(default)s)")
	#parser.add_argument('-nompi', action='store_true', default=noMPI,help="Don't show MPI boundaries (default: %(default)s)")

	mviz.AddSizeArgs(parser)

	return parser

def main():
	#Defaults
	doMPI = False #[Add MPI tiling]
	# Set up the command-line parser.
	parser = create_command_line_parser()
	#Finalize parsing
	args = parser.parse_args()
	fdir1 = args.d1
	ftag1 = args.id1
	fdir2 = args.d2
	ftag2 = args.id2
	nStp = args.n
	fieldNames = args.f
	oName = "gamErrPic.png"
	
	#Get domain size
	xyBds = mviz.GetSizeBds(args)
	
	#---------
	#Figure parameters
	figSz = (12,7.5)


	#======
	#Init data
	gsph1 = msph.GamsphPipe(fdir1,ftag1)
	gsph2 = msph.GamsphPipe(fdir2,ftag2)
		
	#======
	#Setup figure
	fig = plt.figure(figsize=figSz)
	gs = gridspec.GridSpec(2,2,height_ratios=[20,1],hspace=0.025)
	

	AxL = fig.add_subplot(gs[0,0])
	AxR = fig.add_subplot(gs[0,1])

	AxCL = fig.add_subplot(gs[-1,0])
	AxCR = fig.add_subplot(gs[-1,1])

	fnList = [item.strip() for item in fieldNames.split(',')]

	AxL.clear()
	AxR.clear()

	mviz.PlotErrRel(gsph1,gsph2,nStp,xyBds,AxL,fnList,AxCB=AxCL)
	mviz.PlotErrAbs(gsph1,gsph2,nStp,xyBds,AxR,fnList,AxCB=AxCR)

	gsph1.AddTime(nStp,AxL,xy=[0.025,0.89],fs="x-large")

	#Add MPI decomp
	if (doMPI):
		mviz.PlotMPI(gsph2,AxL)
		mviz.PlotMPI(gsph2,AxR)

	kv.savePic(oName,bLenX=45)


if __name__ == "__main__":
	main()
