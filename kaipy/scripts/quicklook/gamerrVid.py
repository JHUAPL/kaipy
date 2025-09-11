#!/usr/bin/env python
#Make video of error between two Gamera cases

# Standard modules
import argparse
from argparse import RawTextHelpFormatter
import os
import errno
import subprocess
import shutil
import concurrent.futures
import multiprocessing
import traceback


# Third-party modules
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from alive_progress import alive_bar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Kaipy modules
import kaipy.kaiViz as kv
import kaipy.gamera.msphViz as mviz
import kaipy.gamera.magsphere as msph
import kaipy.kdefs as kdefs
import kaipy.gamera.rcmpp as rcmpp
cLW = 0.25
relColor = "tab:blue"
absColor = "tab:orange"

def makeMovie(frame_dir,movie_name):
	frame_pattern = frame_dir + "/vid.%04d.png"
	movie_file = os.getcwd() + "/" + movie_name + ".mp4"
	ffmpegExe = "ffmpeg"
	if shutil.which(ffmpegExe) is None:
		ffmpegExe = "ffmpeg4"
		if shutil.which(ffmpegExe) is None:
			print("Could not find any ffmpeg executable. Video will not be generated.")
			return

	cmd = [
		ffmpegExe, "-nostdin", "-i", frame_pattern,
		"-vcodec", "libx264", "-crf", "14", "-profile:v", "high", "-pix_fmt", "yuv420p",
		movie_file,"-y"
	]
	subprocess.run(cmd, check=True)

# python allows changes by reference to the errTimes,errListRel, errListAbs lists
def makeImage(i,gsph1,gsph2,tOut,doVerb,doEq,logAx,xyBds,fnList,oDir,errTimes,errListRel,errListAbs,cv,dataCounter, vO, figSz, noMPI, noLog, fieldNames):
	if doVerb:
		print("Making image %d"%(i))
	#Convert time (in seconds) to Step #
	nStp = np.abs(gsph1.T-tOut[i]).argmin()+gsph1.s0
	if doVerb:
		print("Minute = %5.2f / Step = %d"%(tOut[i]/60.0,nStp))
	npl = vO[i]
	
	#======
	#Setup figure
	fig = plt.figure(figsize=figSz)
	gs = gridspec.GridSpec(5,2,height_ratios=[20,5,1,5,9],hspace=0.025)
	
	AxTL = fig.add_subplot(gs[0,0])
	AxTR = fig.add_subplot(gs[0,1])
	AxB = fig.add_subplot(gs[-1,0:2])
	AxB2 = AxB.twinx() # second plot on bottom axis
	
	AxCT = fig.add_subplot(gs[2,0:2])
	
	AxTL.clear()
	AxTR.clear()
	AxB.clear()
	AxB2.clear()
	
	#plot upper left msph error
	mviz.PlotErrRel(gsph1,gsph2,nStp,xyBds,AxTL,fnList,AxCB=AxCT,doVerb=doVerb,doEq=doEq)
	if doEq:
		AxTL.set_title("Equatorial Slice of Relative Error")
	else:
		AxTL.set_title("Meridional Slice of Relative Error")
	
	#plot upper right cumulative logical error
	mviz.PlotLogicalErrRel(gsph1,gsph2,nStp,AxTR,fnList,logAx,doVerb=doVerb)
	if logAx == 0:
		AxTR.set_title("Per-Cell Relative Error along I-Axis")
	elif logAx == 1:
		AxTR.set_title("Per-Cell Relative Error along J-Axis")
	else:
		AxTR.set_title("Per-Cell Relative Error along K-Axis")
	if (not noMPI):
		#plot I-MPI decomp on logical plot
		if(gsph2.Ri > 1 and logAx != 0):
			for im in range(gsph2.Ri):
				i0 = im*gsph2.dNi
				if logAx == 1:
					AxTR.plot([i0, i0],[0, gsph2.Nk],"deepskyblue",linewidth=0.25,alpha=0.5)
				else:
					AxTR.plot([i0, i0],[0, gsph2.Nj],"deepskyblue",linewidth=0.25,alpha=0.5)
		#plot J-MPI decomp on logical plot
		if (gsph2.Rj>1 and logAx != 1):
			for jm in range(1,gsph2.Rj):
				j0 = jm*gsph2.dNj
				if logAx == 0:
					AxTR.plot([j0, j0],[0, gsph2.Nk],"deepskyblue",linewidth=0.25,alpha=0.5)
				else:
					AxTR.plot([0, gsph2.Ni],[j0, j0],"deepskyblue",linewidth=0.25,alpha=0.5)
		#plot K-MPI decomp on logical plot
		if (gsph2.Rk>1 and logAx != 2):
			for km in range(1,gsph2.Rk):
				k0 = km*gsph2.dNk
				if logAx == 0:
					AxTR.plot([0, gsph2.Nj],[k0, k0],"deepskyblue",linewidth=0.25,alpha=0.5)
				else:
					AxTR.plot([0, gsph2.Ni],[k0, k0],"deepskyblue",linewidth=0.25,alpha=0.5)
	
	#plot bottom line plot
	etval = tOut[i]/60.0
	erval = mviz.CalcTotalErrRel(gsph1,gsph2,nStp,fnList,doVerb=doVerb)
	eaval = mviz.CalcTotalErrAbs(gsph1,gsph2,nStp,fnList,doVerb=doVerb)
	
	# this section is the code must be performed sequentially to add data to the line plots one-by-one
	with cv:
		while not dataCounter.value == i:
			cv.wait()
		errTimes.append(etval)
		errListRel.append(erval)
		errListAbs.append(eaval)
		if noLog:
			AxB.plot(errTimes, errListRel,color=relColor)
			AxB2.plot(errTimes, errListAbs,color=absColor)
		else:
			AxB.semilogy(errTimes, errListRel,color=relColor)
			AxB2.semilogy(errTimes, errListAbs,color=absColor)
		dataCounter.value = i+1
		cv.notify_all()
	# end of sequential region
	AxB.set_xlabel('Time (min)')
	AxB.set_ylabel('Per-Cell Mean Relative Error',color=relColor)
	# values and thresholds for background coloring
	bgAlpha=0.2
	greenYellow=1.0e-6
	yellowRed=1.0e-2
	# coloring background messes with y-axis autoscaling
	if AxB.get_ylim()[1] < greenYellow:
		AxB.axhspan(AxB.get_ylim()[0],AxB.get_ylim()[1],color='green',alpha=bgAlpha)
	elif AxB.get_ylim()[0] < greenYellow:
		AxB.axhspan(AxB.get_ylim()[0],greenYellow,color='green',alpha=bgAlpha)
	
	if AxB.get_ylim()[0] > greenYellow and AxB.get_ylim()[1] < yellowRed:
		AxB.axhspan(AxB.get_ylim()[0],AxB.get_ylim()[1],color='yellow',alpha=bgAlpha)
	elif AxB.get_ylim()[0] < greenYellow and AxB.get_ylim()[1] > yellowRed:
		AxB.axhspan(greenYellow,yellowRed,color='yellow',alpha=bgAlpha)
	elif AxB.get_ylim()[0] < greenYellow:
		AxB.axhspan(greenYellow,AxB.get_ylim()[1],color='yellow',alpha=bgAlpha)
	elif AxB.get_ylim()[1] > yellowRed:
		AxB.axhspan(AxB.get_ylim()[0],yellowRed,color='yellow',alpha=bgAlpha)
	
	if AxB.get_ylim()[0] > yellowRed:
		AxB.axhspan(AxB.get_ylim()[0],AxB.get_ylim()[1],color='red',alpha=bgAlpha)
	elif AxB.get_ylim()[1] > yellowRed:
		AxB.axhspan(yellowRed,AxB.get_ylim()[1],color='red',alpha=bgAlpha)

	AxB.tick_params(axis='y',which='both',colors=relColor,left=True,right=True,labelleft=True,labelright=False)
	AxB2.set_ylabel('Per-Cell Mean Absolute Error',color=absColor)
	#AxB2.yaxis.tick_right()
	AxB2.yaxis.set_label_position("right")
	AxB2.tick_params(axis='y',which='both',colors=absColor,left=True,right=True,labelleft=False,labelright=True)
	AxB.set_title("'" + fieldNames + "' Per-Cell Error Over Time")
	
	gsph1.AddTime(nStp,AxTL,xy=[0.025,0.84],fs="x-large")
	
	#Add MPI decomp
	if (not noMPI):
		mviz.PlotMPI(gsph2,AxTL)
	
	fOut = oDir+"/vid.%04d.png"%(npl)
	kv.savePic(fOut,bLenX=45,saveFigure=fig,doClose=True)


def create_command_line_parser():
	"""Set up the command-line parser.
	"""
	#Defaults
	fdir1 = os.getcwd()
	ftag1 = "msphere"
	fdir2 = os.getcwd()
	ftag2 = "msphere"
	oDir = "vid2D"
	ts = 0.0	  #[min]
	te = 200.0	#[min]
	dt = 0.0 #[sec] 0 default means every timestep
	logAx = 2 # axis to accumulate logical error along
	Nth = 1 #Number of threads
	noMPI = False # Don't add MPI tiling
	noLog = False
	fieldNames = "Bx, By, Bz"
	doVerb = False
	skipMovie = False
	merid = False

	MainS = """Creates simple multi-panel figure for Gamera magnetosphere run
	Left Panel - Residual vertical magnetic field
	Right Panel - Pressure (or density) and hemispherical insets
	"""

	parser = argparse.ArgumentParser(description=MainS, formatter_class=RawTextHelpFormatter)
	parser.add_argument('-d1',type=str,metavar="directory",default=fdir1,help="Directory to read first dataset from (default: %(default)s)")
	parser.add_argument('-id1',type=str,metavar="runid",default=ftag1,help="RunID of first dataset (default: %(default)s)")
	parser.add_argument('-d2',type=str,metavar="directory",default=fdir2,help="Directory to read second dataset from (default: %(default)s)")
	parser.add_argument('-id2',type=str,metavar="runid",default=ftag2,help="RunID of second dataset (default: %(default)s)")
	parser.add_argument('-o',type=str,metavar="directory",default=oDir,help="Subdirectory to write to (default: %(default)s)")
	parser.add_argument('-ts' ,type=float,metavar="tStart",default=ts,help="Starting time [min] (default: %(default)s)")
	parser.add_argument('-te' ,type=float,metavar="tEnd"	,default=te,help="Ending time	[min] (default: %(default)s)")
	parser.add_argument('-dt' ,type=float,metavar="dt"	,default=dt,help="Cadence		[sec] (default: %(default)s)")
	parser.add_argument('-logAx',type=int,metavar="logAx",default=logAx,help="Index of the axis to accumulate along in the upper-right plot (default: %(default)s)")
	parser.add_argument('-Nth' ,type=int,metavar="Nth",default=Nth,help="Number of threads to use (default: %(default)s)")
	parser.add_argument('-f',type=str,metavar="fieldnames",default=fieldNames,help="Comma-separated fields to plot (default: %(default)s)")
	parser.add_argument('-linear',action='store_true', default=noLog,help="Plot linear line plot instead of logarithmic (default: %(default)s)")
	parser.add_argument('-v',action='store_true', default=doVerb,help="Do verbose output (default: %(default)s)")
	parser.add_argument('-skipMovie',action='store_true', default=skipMovie,help="Skip automatic movie generation afterwards (default: %(default)s)")
	parser.add_argument('-merid',action='store_true', default=merid,help="Plot meridional instead of equatorial slice (default: %(default)s)")
	#parser.add_argument('-nompi', action='store_true', default=noMPI,help="Don't show MPI boundaries (default: %(default)s)")


	return parser

def main():
	#Defaults
	fdir1 = os.getcwd()
	ftag1 = "msphere"
	fdir2 = os.getcwd()
	ftag2 = "msphere"
	oDir = "vid2D"
	ts = 0.0	  #[min]
	te = 200.0	#[min]
	dt = 0.0 #[sec] 0 default means every timestep
	logAx = 2
	Nth = 1 #Number of threads
	noMPI = False # Don't add MPI tiling
	noLog = False
	fieldNames = "Bx, By, Bz"
	doVerb = False
	skipMovie = False
	doEq = True

	parser = create_command_line_parser()
	mviz.AddSizeArgs(parser)

	#Finalize parsing
	args = parser.parse_args()
	fdir1 = args.d1
	ftag1 = args.id1
	fdir2 = args.d2
	ftag2 = args.id2
	ts	= args.ts
	te	= args.te
	dt	= args.dt
	logAx = args.logAx
	oSub = args.o
	Nth = args.Nth
	fieldNames = args.f
	noLog = args.linear
	doVerb = args.v
	doEq = not args.merid
	#noMPI = args.noMPI
	
	fnList = [item.strip() for item in fieldNames.split(',')]

	#Setup output directory
	oDir = os.getcwd() + "/" + oSub
	print("Writing output to %s"%(oDir))

	#Check/create directory if necessary
	if (not os.path.exists(oDir)):
		try:
			print("Creating directory %s"%(oDir))
			os.makedirs(oDir)
		except OSError as exc:
			if exc.errno == errno.EEXIST and os.path.isdir(oDir):
				pass
			else:
				raise

	#Get domain size
	xyBds = mviz.GetSizeBds(args)
	
	#---------
	#Figure parameters
	figSz = (12,7.5)

	#======
	#Init data
	gsph1 = msph.GamsphPipe(fdir1,ftag1)
	gsph2 = msph.GamsphPipe(fdir2,ftag2)
	
	#Setup timing info
	if(dt > 0):
		tOut = np.arange(ts*60.0,te*60.0,dt)
	else:
		tOut = [t for t in gsph1.T if t > ts*60.0 and t < te*60.0]
	Nt = len(tOut)
	vO = np.arange(0,Nt)

	print(f"Writing {Nt} outputs between minutes {ts} and {te}")
	print(f"Using {Nth} threads")
	
	errTimes = []
	errListRel = []
	errListAbs = []

	#Loop over sub-range
	titstr = "Comparing '%s' to '%s'"%(fdir1,fdir2)
	with alive_bar(Nt,title=titstr.ljust(kdefs.barLab),length=kdefs.barLen,bar=kdefs.barDef,disable=doVerb) as bar:
		#with concurrent.futures.ThreadPoolExecutor(max_workers=Nth) as executor:
		with concurrent.futures.ProcessPoolExecutor(max_workers=Nth) as executor:
			m = multiprocessing.Manager()
			cv = m.Condition()
			dataCounter = m.Value('i',0)
			met = m.list(errTimes)
			melr = m.list(errListRel)
			mela = m.list(errListAbs)
			#imageFutures = {executor.submit(makeImage,i,gsph1,gsph2,tOut,doVerb,doEq,logAx,xyBds,fnList,oDir,errTimes,errListRel,errListAbs,cv): i for i in range(0,Nt)}
			imageFutures = {executor.submit(makeImage,i,gsph1,gsph2,tOut,doVerb,doEq,logAx,xyBds,fnList,oDir,met,melr,mela,cv,dataCounter,vO,figSz,noMPI,noLog,fieldNames): i for i in range(0,Nt)}
			for future in concurrent.futures.as_completed(imageFutures):
				try:
					retVal = future.result()
				except Exception as e:
					print("Exception")
					print(e)
					traceback.print_exc()
					exit()
				bar()
		
	makeMovie(oDir,oSub)


if __name__ == "__main__":
	main()
