#Various scripts to support visualization of Kaiju data

# Standard modules
import os
from operator import sub

# Third-party modules
import h5py
import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Wedge
from matplotlib import ticker
import shutil

# Kaipy modules
from kaipy.kdefs import *


#Create 2D equatorial grid (Ni,Nj*2+1) from lfm/egg-style
# def genEQGrid(fIn):
#     with h5py.File(fIn,'r') as hf:
#         Nk,Nj,Ni = hf.get("X")[()].shape
#         X3 = hf.get("X")[()].T
#         Y3 = hf.get("Y")[()].T
#         Z3 = hf.get("Z")[()].T
#     print("Ni,Nj,Nk = %d,%d,%d"%(Ni,Nj,Nk))
#     #Create grid
#     Nr = Ni-1
#     Np = 2*(Nj-1)
#     xx = np.zeros((Nr+1,Np+1))
#     yy = np.zeros((Nr+1,Np+1))

#     for i in range(Nr+1):
#         for j in range(Nj):
#             xx[i,j] = X3[i,j,0]
#             yy[i,j] = Y3[i,j,0]
#         for j in range(Nj,Np+1):
#             jp = Np-j
#             xx[i,j] =  X3[i,jp,0]
#             yy[i,j] = -Y3[i,jp,0]
#     return xx,yy
# def getEQVar(fIn,StpN=0,vID="D"):
#     gID = "Step#%d"%(StpN)
#     with h5py.File(fIn,'r') as hf:
#         V = hf.get(gID).get(vID)[()].T
#         Ni,Nj,Nk = V.shape #Cell-centered (not node)
#     Nr = Ni
#     Np = 2*(Nj)
#     kp = (Nk)/2
#     vv = np.zeros((Nr,Np))
#     for i in range(Nr):
#         for j in range(Np):
#             if (j>=Nj):
#                 jp = Np-j-1
#                 vv[i,j] = V[i,jp,kp]
#             else:
#                 jp = j
#                 vv[i,j] = V[i,jp,0]
#     return vv

#Calculate equatorial Bz-D
# def getEQBzD(xx,yy,MagM=-EarthM0g):
#     Nip,Njp = xx.shape
#     Ni = Nip-1
#     Nj = Njp-1

#     BzD = np.zeros((Ni,Nj))
#     xxc = np.zeros((Ni,Nj))
#     yyc = np.zeros((Ni,Nj))

#     for i in range(Ni):
#         for j in range(Nj):
#             xc = 0.25*(xx[i,j]+xx[i+1,j]+xx[i,j+1]+xx[i+1,j+1])
#             yc = 0.25*(yy[i,j]+yy[i+1,j]+yy[i,j+1]+yy[i+1,j+1])
#             xxc[i,j] = xc
#             yyc[i,j] = yc
#             r = np.sqrt(xc**2.0+yc**2.0)
#             rm5 = r**(-5.0)
#             BzD[i,j] = -r*r*MagM*rm5
#     return xxc,yyc,BzD

#---------------------------------
#Matplotlib helpers


def SetAx(xyBds=[-1, 1, -1, 1], ax=None, Adj='box'):
    """Set axis plot bounds and aspect ratio.

    Ensure that the plot has an equal aspect ratio and that the axis limits
    are set to the specified values.

    Parameters:
        xyBds : list of 4 float, optional, default [-1, 1, -1, 1]
            Axis limits for x and y as (xmin, xmax, ymin, ymax).
        ax: matplotlib.Axes, optional, default None
            Axes object to modify.
        Adj: str, optional, default 'box
            Specify what aspect of the plot to adjust to achieve equal aspect
            ratio.

    Returns:
        None
    """
    # If no Axes object is specified, fetch the current Axes object.
    if ax is None:
        ax = plt.gca()

    # Set the axis limits for x and y.
    ax.set_xlim(xyBds[0], xyBds[1])
    ax.set_ylim(xyBds[2], xyBds[3])

    # Make the plot have an equal aspect ratio, adjusting positions as
    # allowed by Adj.
    ax.set_aspect('equal', adjustable=Adj)


#Set axis labels and locations
def SetAxLabs(Ax, xLab, yLab, doBot=True, doLeft=True, fs="medium"):
    """Set axis plot bounds and aspect ratio.

    Ensure that the plot has an equal aspect ratio and that the axis limits
    are set to the specified values.

    Parameters:
        xyBds : list of 4 float, optional, default [-1, 1, -1, 1]
            Axis limits for x and y as (xmin, xmax, ymin, ymax).
        ax: matplotlib.Axes, optional, default None
            Axes object to modify.
        Adj: str, optional, default 'box
            Specify what aspect of the plot to adjust to achieve equal aspect
            ratio.

    Returns:
        None
    """
    # """Set the x and y axis labels for a given matplotlib Axes object.

    # Parameters
    # ----------
    # Ax (matplotlib.axes.Axes): The Axes object to set the labels for.
    # xLab (str): The label for the x-axis.
    # yLab (str): The label for the y-axis.
    # doBot (bool, optional): Whether to display the x-axis labels and ticks at the bottom. Default is True.
    # doLeft (bool, optional): Whether to display the y-axis labels and ticks on the left. Default is True.
    # fs (str, optional): The font size for the labels. Default is "medium".

    # Returns
    # -------
    # None
    # """
    Ax.set_xlabel(xLab, fontsize=fs)
    Ax.set_ylabel(yLab, fontsize=fs)
    if not doBot:
        Ax.xaxis.tick_top()
        Ax.xaxis.set_label_position('top')
    if not doLeft:
        Ax.yaxis.tick_right()
        Ax.yaxis.set_label_position('right')

    # Kill labels if not string is None
    if xLab is None:
        Ax.xaxis.label.set_visible(False)
        Ax.xaxis.set_visible(False)
        plt.setp(Ax.get_xticklabels(), visible=False)

    if yLab is None:
        Ax.yaxis.label.set_visible(False)
        Ax.yaxis.set_visible(False)
        plt.setp(Ax.get_yticklabels(), visible=False)

#Set X axis to labels to well formatted date time
def SetAxDate(Ax, fmt='%H:%M \n%Y-%m-%d'):
    """Set the x-axis of the given Axes object to display dates.

    Parameters:
        Ax: The Axes object to set the x-axis date format for.
        fmt: The date format string to use. Defaults to '%H:%M%Y-%m-%d'.

    Returns:
        None
    """
    Ax.xaxis_date()
    Ax.xaxis.set_major_formatter(mpl.dates.DateFormatter(fmt))

#Adds 2D earth w/ dawn/dusk
def addEarth2D(Re=1, angle=-90, ax=None):
    """
    Adds a 2D representation of Earth to the given matplotlib axes.

    Parameters:
        Re (float): Radius of the Earth. Default is 1.
        angle (float): Angle at which the Earth is rotated. Default is -90.
        ax (matplotlib.axes.Axes): The axes to which the Earth will be added. If None, the current axes will be used.

    Returns:
        list: A list containing the two Wedge objects representing the Earth.
    """
    

    if ax is None:
        ax = plt.gca()
    colors=('w','k')
    center = 0
    theta1, theta2 = angle, angle + 180

    w1 = Wedge(center, Re,-90,90,   fc=colors[0],ec='k')
    w2 = Wedge(center, Re, 90, -90, fc=colors[1],ec='k')
    for wedge in [w1, w2]:
        ax.add_artist(wedge)
    return [w1, w2]

#Add inner cutout
def DrawCut(Rin=2.5, ax=None):
    """
    Draw a cut using matplotlib.patches.Wedge.

    Parameters:
        Rin (float): The inner radius of the cut. Default is 2.5.
        ax (matplotlib.axes.Axes, optional): The axes on which to draw the cut. If not provided, the current axes will be used.

    Returns:
        matplotlib.patches.Wedge: The created Wedge object representing the cut.
    """
    
    if ax is None:
        ax = plt.gca()
    w1 = Wedge(0, Rin, 0, 360, fc=None, ec='k')
    ax.add_artist(w1)
    return w1

#Take cell-centered polar values and add extra phi layer
#Useful for contour plots through +X
def reWrap(V):
    """
    Re-wraps a 2D array by adding an extra column at the end, where the values in the new column are the same as the values in the first column.

    Parameters:
        V (numpy.ndarray): The input 2D array.

    Returns:
        numpy.ndarray: The re-wrapped 2D array with an extra column.
    """
    Ni, Nj = V.shape
    Vp = np.zeros((Ni, Nj+1))
    Vp[:, 0:Nj] = V
    Vp[:, -1] = V[:, 0]
    return Vp

#Image files
#Wrapper to save (and trim) figure
def savePic(fOut, dpiQ=300, doTrim=True, bLenX=20, bLenY=None, doClose=False, doEps=False, saveFigure=None):
    """
    Save a matplotlib figure to a file.

    Parameters:
        fOut (str): The output file path.
        dpiQ (int): The resolution of the saved figure in dots per inch (default: 300).
        doTrim (bool): Whether to trim the saved figure (default: True).
        bLenX (int): The length of the x-axis trim boundary (default: 20).
        bLenY (int): The length of the y-axis trim boundary (default: None).
        doClose (bool): Whether to close all figures after saving (default: False).
        doEps (bool): Whether to save the figure in EPS format (default: False).
        saveFigure (matplotlib figure): A predefined figure to plot into (default: None).

    Returns:
        Image File saved to disk.
    """
    if doEps:
        if saveFigure is None:
            plt.savefig(fOut, dpi=dpiQ, format='eps')
        else:
            saveFigure.savefig(fOut, dpi=dpiQ, format='eps')
    else:
        if saveFigure is None:
            plt.savefig(fOut, dpi=dpiQ)
        else:
            saveFigure.savefig(fOut, dpi=dpiQ)
        if doTrim:
            trimFig(fOut, bLenX, bLenY)

    if doClose:
        if saveFigure is None:
            plt.close('all')
        else:
            plt.close(saveFigure)

#Checks to see if image magick is in the current path
def isMagick():
    path = shutil.which('magick')
    if (path is None):
        return False
    else:
        return True

#Use imagemagick to trim whitespace off figure
#doEven: Guarantee even number of pixels in X/Y
def trimFig(fName, bLenX=20, bLenY=None, doEven=True):
    """
    Trims the figure image file by removing the borders and resizing it to have even dimensions.

    Parameters:
        fName (str): The file name of the figure image.
        bLenX (int): The length of the border to be added on the X-axis. Default is 20.
        bLenY (int): The length of the border to be added on the Y-axis. If not provided, it will be set to bLenX.
        doEven (bool): Flag indicating whether to resize the image to have even dimensions. Default is True.

    Returns:
        Image file saved to disk.
    """


    if bLenY is None:
        bLenY = bLenX

    if (not isMagick()):
        return
    ComS = 'magick ' + fName + ' -trim -border %dx%d -bordercolor "#FFFFFF" '%(bLenX,bLenY) + ' ' + fName
    os.system(ComS)

    if doEven:
        Nx, Ny = picSz(fName)

        while (Nx % 2) != 0:
            ShaveX(fName)
            Nx, Ny = picSz(fName)

        while (Ny % 2) != 0:
            ShaveY(fName)
            Nx, Ny = picSz(fName)

        Nx, Ny = picSz(fName)
        if (Nx % 2) != 0 or (Ny % 2) != 0:
            print("Parity failure on pic sizing")
            print(Nx, Ny)


def picSz(fName):
    """
    Get the size of an image.

    Parameters:
        fName (str): The file path of the image.

    Returns:
        tuple: A tuple containing the width and height of the image.
    """

    
    with Image.open(fName) as img:
        Nx, Ny = img.size
    return Nx, Ny

def ShaveX(fName):
    """
    ShaveX function crops an image file by removing one pixel from the right side.

    Parameters:
        fName (str): The file name of the image to be cropped.

    Returns:
        fName file is cropped by one pixel on the right side.
    """

    ComS = 'magick %s -crop -1+0 +repage %s'%(fName,fName)
    os.system(ComS)

def ShaveY(fName):
    """
    Shave one pixel from the top of the image.

    Parameters:
        fName (str): The file name of the image.

    Returns:
        fName file is cropped by one pixel on the top.
    """

    ComS = 'magick %s -crop +0-1 +repage %s'%(fName,fName)
    os.system(ComS)

#---------------------------------
#Create colorbar with specified midpoint (grabbed from stack overflow)
class MidpointNormalize(Normalize):
    """
    Custom normalization class that maps values to the range [0, 1] with a specified midpoint.

    Parameters:
        vmin (float, optional): The minimum value of the input range. If not provided, the minimum value of the input data will be used.
        vmax: (float, optional): The maximum value of the input range. If not provided, the maximum value of the input data will be used.
        midpoint (float, optional): The midpoint value that will be mapped to 0.5 in the output range. If not provided, the average of vmin and vmax will be used.
        clip (bool, default False): If True, values outside the range [vmin, vmax] will be clipped to vmin or vmax.

    Returns:
        normalized_value (numpy.ma.masked_array): The normalized value(s) within the range [0, 1].

    Examples:
    >>> norm = MidpointNormalize(vmin=0, vmax=10, midpoint=5)
    >>> normalized_value = norm(7)
    >>> print(normalized_value)  # Output: 0.7

    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

#Create norm object for MPL
def genNorm(vMin, vMax=None, doLog=False, doSymLog=False, midP=None, linP=1.0):
    """
    Generate a matplotlib color normalization object based on the given parameters.

    Parameters:
        vMin (float): the minimum value of the data range
        vMax (float): optional, the maximum value of the data range (default: None)
        doLog (bool, optional): whether to use logarithmic normalization (default: False)
        doSymLog (bool, optional): whether to use symmetric logarithmic normalization (default: False)
        midP (float, optional): the midpoint value for midpoint normalization (default: None)
        linP (float, optional): the linear threshold for symmetric logarithmic normalization (default: 1.0)

    Returns:
        vN (matplotlib.colors.Normalize): the color normalization object

    Note:
        If vMax is not provided, the absolute value of vMin is used as both vMin and vMax.
        If midP is not provided, midpoint normalization is not used.
        If doLog is True, logarithmic normalization is used.
        If doSymLog is True, symmetric logarithmic normalization is used.
        If none of the above conditions are met, linear normalization is used.
    """


    if vMax is None:
        vMin = -np.abs(vMin)
        vMax = np.abs(vMin)

    if midP is None:
        doMid = False
    else:
        doMid = True

    if doMid:
        vN = MidpointNormalize(vmin=vMin, vmax=vMax, midpoint=midP)
    elif doLog:
        vN = LogNorm(vmin=vMin, vmax=vMax)
    elif doSymLog:
        vN = SymLogNorm(linthresh=linP, vmin=vMin, vmax=vMax, base=10)
    else:
        vN = Normalize(vmin=vMin, vmax=vMax)

    return vN


#Create colorbar object into specified axis
def genCB(AxCB, vN, cbT="Title", cM="viridis", doVert=False, cbSz="medium", Ntk=None):
    """
    Generate a colorbar using matplotlib.

    Parameters:
        AxCB (Axes object): The axes object where the colorbar will be drawn.
        vN (Normalize object): The normalization object used to map data values to colors.
        cbT (str, optional): The title of the colorbar. Default is "Title".
        cM (str, optional): The name of the colormap to use. Default is "viridis".
        doVert (bool, optional): Whether to display the colorbar vertically. Default is False.
        cbSz (str, optional): The size of the colorbar labels and title. Default is "medium".
        Ntk (int, optional): The number of ticks to display on the colorbar. Default is None.

    Returns:
        cb (Colorbar object): The generated colorbar object.

    """
    

    if doVert:
        cbOr = "vertical"
    else:
        cbOr = "horizontal"
    cmData = plt.get_cmap(cM)
    cb = mpl.colorbar.ColorbarBase(AxCB, cmap=cmData, norm=vN, orientation=cbOr)
    if Ntk is not None:
        cb.locator = ticker.MaxNLocator(nbins=Ntk)
        cb.update_ticks()

    cb.set_label(cbT, fontsize=cbSz)
    cb.ax.tick_params(labelsize=cbSz)
    return cb


def labelStr(data, key, vecComp):
    """
    Generate a label string based on the given data, key, and vector component.

    Parameters:
        data (dict): A dictionary containing the data.
        key (str): The key used to access the data.
        vecComp (int): The vector component.

    Returns:
        str: The generated label string.
    """
    vecLabel = ['x', 'y', 'z']
    if key == "Velocity":
        key = "Speed"
    if vecComp < 3 and vecComp > -1:
        label = (
            data['GAMERA_' + key].attrs['AXISLABEL'] +
            vecLabel[vecComp] + ' [' +
            data['GAMERA_' + key].attrs['UNITS'].decode() + ']'
        )
    else:
        key_name = "GAMERA_" + key
        label = data[key_name].attrs['AXISLABEL'] + ' [' + data[key_name].attrs['UNITS'].decode() + ']'
    return label


def itemPlot(Ax, data, key, plotNum, numPlots, vecComp=-1):
    """
    Plot the data for a specific item.

    Parameters:
        Ax (matplotlib.axes.Axes): The axes object to plot on.
        data (dict): The data dictionary containing the data to plot.
        key (str): The key representing the item to plot.
        plotNum (int): The plot number.
        numPlots (int): The total number of plots.
        vecComp (int, optional): The vector component to plot. Defaults to -1.

    Returns:
        None
    """
    if -1 == vecComp:
        # if key == "Br":
        #     # Plot a horizontal line at Br=0. This indicates passage of the
        #     # heliospheric current sheet.
        #     Ax.axhline([0], linestyle="--", color="black")
        # <HACK>
        # if key == "Density":
        #     # Reduce the density vertical scale.
        #     Ax.set_ylim(0, 50)  # cm**-3
        # </HACK>
        if key == "Velocity":
            maskedData = np.ma.masked_where(
                data["GAMERA_inDom"][:] == 0.0,
                data[key].flatten()[0]["VR"][:]
            )
            maskedGamera = np.ma.masked_where(
                data["GAMERA_inDom"][:] == 0.0, data['GAMERA_Speed'][:]
            )
        else:
            maskedData = np.ma.masked_where(
                data["GAMERA_inDom"][:] == 0.0, data[key][:]
            )
            maskedGamera = np.ma.masked_where(
                data["GAMERA_inDom"][:] == 0.0, data['GAMERA_' + key][:]
            )
        Ax.plot(data['Epoch_bin'], maskedData)
        Ax.plot(data['Epoch_bin'], maskedGamera)
    else:
        maskedData = np.ma.masked_where(data["GAMERA_inDom"][:] == 0.0,
                                        data[key][:, vecComp])
        Ax.plot(data['Epoch_bin'], maskedData)
        maskedGamera = np.ma.masked_where(data["GAMERA_inDom"][:] == 0.0,
                                          data['GAMERA_' + key][:, vecComp])
        Ax.plot(data['Epoch_bin'], maskedGamera)
    if (plotNum % 2) == 0:
        left = True
    else:
        left = False
    label = labelStr(data, key, vecComp)
    if plotNum == (numPlots - 1):
        SetAxLabs(Ax, 'UT', label, doLeft=left)
        SetAxDate(Ax)
    else:
        SetAxLabs(Ax, None, label, doLeft=left)
    return


def compPlot(plotname, scId, data):
    """
    Generate a composite plot with multiple subplots based on the given data.

    Parameters:
        plotname (str): The title of the composite plot.
        scId (str): The identifier for the subplot legend.
        data (dict): A dictionary containing the data to be plotted.

    Returns:
        Image file saved to disk.

    """
    numPlots = 0
    variables_to_plot = []
    keys = data.keys()

    if 'Density' in keys:
        numPlots += 1
        variables_to_plot.append('Density')
    if 'Pressue' in keys:
        numPlots += 1
        variables_to_plot.append('Pressue')
    if 'Temperature' in keys:
        numPlots += 1
        variables_to_plot.append('Temperature')
    if 'MagneticField' in keys:
        numPlots += 3
        variables_to_plot.append('MagneticField')
    if 'Velocity' in keys:
        numPlots += 3
        variables_to_plot.append('Velocity')

    figsize = (10, 10)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(numPlots, 1)
    plotNum = 0

    for key in variables_to_plot:
        if 'MagneticField' == key or 'Velocity' == key:
            doVecPlot = True
        else:
            doVecPlot = False

        if 0 == plotNum:
            Ax1 = fig.add_subplot(gs[plotNum, 0])
            if doVecPlot:
                itemPlot(Ax1, data, key, plotNum, numPlots, vecComp=0)
                plotNum += 1
                Ax = fig.add_subplot(gs[plotNum, 0], sharex=Ax1)
                itemPlot(Ax, data, key, plotNum, numPlots, vecComp=1)
                plotNum += 1
                Ax = fig.add_subplot(gs[plotNum, 0], sharex=Ax1)
                itemPlot(Ax, data, key, plotNum, numPlots, vecComp=2)
                plotNum += 1
            else:
                itemPlot(Ax1, data, key, plotNum, numPlots)
                plotNum += 1
        else:
            Ax = fig.add_subplot(gs[plotNum, 0], sharex=Ax1)
            if doVecPlot:
                itemPlot(Ax, data, key, plotNum, numPlots, vecComp=0)
                plotNum += 1
                Ax = fig.add_subplot(gs[plotNum, 0], sharex=Ax1)
                itemPlot(Ax, data, key, plotNum, numPlots, vecComp=1)
                plotNum += 1
                Ax = fig.add_subplot(gs[plotNum, 0], sharex=Ax1)
                itemPlot(Ax, data, key, plotNum, numPlots, vecComp=2)
                plotNum += 1
            else:
                itemPlot(Ax, data, key, plotNum, numPlots)
                plotNum += 1

    Ax1.legend([scId, 'GAMERA'], loc='best')
    Ax1.set_title(plotname)
    plt.subplots_adjust(hspace=0)

    savePic(plotname, doClose=True)


    
def trajPlot(plotname, scId, data, toRe):
    """
    Plot the trajectory of a spacecraft.

    Parameters:
        lotname (str): The name of the plot.
        scId (str): The spacecraft ID.
        data (dict): A dictionary containing the spacecraft's ephemeris data.
        toRe (float): The conversion factor from meters to Re (Earth radii).

    Returns:
        Image file saved to disk.
    """
    figsize = (15, 5)
    # Create the figure in-memory.
    mpl.use("AGG")
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3)
    Ax1 = fig.add_subplot(gs[0, 0])
    Ax2 = fig.add_subplot(gs[0, 1])
    Ax3 = fig.add_subplot(gs[0, 2])
    Ax1.plot(data['Ephemeris'][:, 0] * toRe, data['Ephemeris'][:, 1] * toRe)
    Ax2.plot(data['Ephemeris'][:, 0] * toRe, data['Ephemeris'][:, 2] * toRe)
    Ax3.plot(data['Ephemeris'][:, 1] * toRe, data['Ephemeris'][:, 2] * toRe)
    Ax1.set_title('XY SM')
    Ax2.set_title('XZ SM')
    Ax3.set_title('YZ SM')
    titlestr = (scId + ' - ' + data['Epoch_bin'][0].strftime('%m/%d/%Y - %H:%M:%S') + ' to ' +
                data['Epoch_bin'][-1].strftime('%m/%d/%Y - %H:%M:%S'))
    fig.suptitle(titlestr)
    savePic(plotname, doClose=True)


def get_aspect(ax):
    """
    Get the current aspect ratio of the given ax.

    Parameters:
        ax: The matplotlib Axes object.

    Returns:
        The aspect ratio of the Axes object.

    """
    
    # Total figure size
    figW, figH = ax.get_figure().get_size_inches()
    # Axis size on figure
    _, _, w, h = ax.get_position().bounds
    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)
    # Ratio of data units
    # Negative over negative because of the order of subtraction
    data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())

    return disp_ratio #/ data_ratio

def setBndsByAspect(ax, bnds, axis='x', aspect=1):
    """ 
    Given bounds of one axis, set the other bounds of Axes object such that aspect ratio and Axes box size is perserved
    
    Parameters:
        ax: Axes object to set bounds for
        bnds (List[float]): [axMin, axMax] min/max bounds for fixed axis
        axis (str): ('x' or 'y') axis that fixed bounds are applied to
        aspect: x:y aspect ratio to preserve
    """
    boxRatio = get_aspect(ax)
    da = bnds[1] - bnds[0]
    
    if axis=='x':
        db = aspect*boxRatio*da
        bMin = -db/2
        bMax = db/2
        ax.set_xlim(bnds)
        ax.set_ylim([bMin, bMax])
    else:
        db = 1/aspect*boxRatio*da
        bMin = -db/2
        bMax = db/2
        ax.set_xlim([bMin, bMax])
        ax.set_ylim(bnds)


# Methods specific for comparing gamhelio output to data from heliospheric
# spacecraft.


def helio_labelStr(data, key, vecComp):
    """
    Returns the label for a given data key in the heliocentric coordinate system.

    Parameters:
        data (dict): A dictionary containing the data.
        key (str): The key for which the label is needed.
        vecComp (str): The vector component for which the label is needed.

    Returns:
        str: The label for the specified data key and vector component.
    """
    label = (
        data['GAMHELIO_' + key].attrs['AXISLABEL'] +
        ' [' + data['GAMHELIO_' + key].attrs['UNITS'] + ']'
    )
    return label


def helioItemPlot_new(Ax, sc_id, data, key, plotNum, numPlots, show_zero=False):
    """Plot a single variable for the comparison plot.

    Plot a single variable for the comparison plot.

    Parameters: 
        Ax (matplotlib.axes._subplots.AxesSubplot): Matplotlib axis for plot.
        data (spacepy.datamodel.SpaceData): Object containing the measured spacecraft data.
        key (str): Name of variable to plot.
        plotNum (int): Position of plot in sequence of plots.
        numPlots (int): Number of plots in sequence.
        show_zero (bool, default False): Set to True to show a black dashed line at the 0 level for the variable.

    Returns:
        None
    """
    # Extract the times and values which fall within the gamhelio simulation
    # domain.
    t = np.ma.masked_where(
        data["GAMHELIO_inDom"][:] == 0.0, data["Ephemeris_time"][:]
    )
    observed = None
    if key in data:
        observed = np.ma.masked_where(
            data["GAMHELIO_inDom"][:] == 0.0, data[key][:]
        )
    # gamhelio results should always be available.
    predicted = np.ma.masked_where(
        data["GAMHELIO_inDom"][:] == 0.0, data["GAMHELIO_" + key][:]
    )

    # Show a black dotted line at y = 0 if requested.
    if show_zero:
        Ax.axhline(y=0, linestyle="--", color="black")

    # Plot the observed and predicted data for the current variable.
    # if key in data:
    if observed is not None:
        Ax.plot(t, observed)
    else:
        fontsize = 18
        Ax.plot(t, [None]*len(t))
        Ax.text(
            0.5, 0.5, "No spacecraft data available.",
            transform=Ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color="darkgrey"
        )
    Ax.plot(t, predicted)

    # Even-numbered plots (1-based) show the y-axis label on the left.
    left = False
    if plotNum % 2 == 1:
        left = True

    # Compute the y-axis label string.
    label = helio_labelStr(data, key, vecComp=-1)

    # Show the x-axis on the last plot.
    if plotNum == numPlots:
        SetAxLabs(Ax, "UT", label, doLeft=left)
        SetAxDate(Ax)
    else:
        SetAxLabs(Ax, None, label, doLeft=left)

    # Add the shared legend to the first plot.
    if plotNum == 1:
        Ax.legend([sc_id, "GAMHELIO"], loc="best")


def helioCompPlot_new(plot_file_path, sc_id, sc_data):
    """Create comparison plots for heliospheric results.

    Create comparison plots for heliospheric results and save the plots to a
    file. The plots will compare simulation results to measured spacecraft
    data. The compared variables are radial velocity, radial magnetic field,
    number density, and temperature.

    Parameters
    ----------
    plotname : str
        Path to file to hold plot image.
    scId : str
        ID string for spacecraft.
    data : spacepy.datamodel.SpaceData
        The current spacecraft and model data

    Returns
    -------
    None
    """
    # Determine which data are available to plot.
    variables_to_plot = ["Speed", "Br", "Density", "Temperature"]

    # Compute the number of variables to plot.
    n_plots = len(variables_to_plot)

    # Create the figure in memory.
    mpl.use("AGG")

    # Create the figure.
    HELIO_COMP_PLOT_FIGSIZE = (10, 10)
    fig = plt.figure(figsize=HELIO_COMP_PLOT_FIGSIZE)

    # Plot each variable in its own subplot.
    for i in range(n_plots):
        plotNum = i + 1
        variable_name = variables_to_plot[i]
        show_zero = False
        if variable_name == "Br":
            show_zero = True
        ax = plt.subplot(n_plots, 1, plotNum)
        helioItemPlot_new(
            ax, sc_id, sc_data, variable_name,
            plotNum, n_plots, show_zero=show_zero
        )

    # Use the plot file path as the figure title.
    fig.suptitle(plot_file_path)

    # Stack the plots directly on top of each other.
    plt.subplots_adjust(hspace=0)

    # Save the figure.
    savePic(plot_file_path, doClose=True)


def helioTrajPlot(plot_file_path, sc_id, sc_data):
    """Create a set of trajectory plots for a heliospheric spacecraft.

    Create a set of trajectory plots for a heliospheric spacecraft.
    These plots are created in the original spacecraft frame.

    Parameters
    ----------
    plot_name: str
        Path to plot file.
    scId: str
        Name of spacecraft.
    data: spacepy.datamodel.SpaceData
        Object containing spacecraft data.

    Returns
    -------
    None
    """
    # Set the figure size (inches)
    figsize = (15, 5)

    # Create the figure in-memory.
    mpl.use("AGG")
    fig = plt.figure(figsize=figsize)

    # Create a grid for the sin-plots.
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Plot the components against each other.
    if sc_id == "ACE":
        X = sc_data["Ephemeris"][:, 0]
        Y = sc_data["Ephemeris"][:, 1]
        Z = sc_data["Ephemeris"][:, 2]
        ax1.plot(X, Y)
        ax1.set_xlabel("GSE X (km)")
        ax1.set_ylabel("GSE Y (km)")
        ax2.plot(X, Z)
        ax2.set_xlabel("GSE X (km)")
        ax2.set_ylabel("GSE Z (km)")
        ax3.plot(Y, Z)
        ax3.set_xlabel("GSE Y (km)")
        ax3.set_ylabel("GSE Z (km)")
        fig.subplots_adjust(wspace=0.25)
    # elif sc_id == "Parker_Solar_Probe":
    #     ax1.plot(sc_data["heliographicLongitude"][:], sc_data["radialDistance"][:])
    #     ax1.set_xlabel("HGI longitude (deg)")
    #     ax1.set_ylabel("HGI radius ($R_{sun})$")
    #     ax2.plot(sc_data["heliographicLatitude"][:], sc_data["radialDistance"][:])
    #     ax2.set_xlabel("HGI latitude (deg)")
    #     ax2.set_ylabel("HGI radius ($R_{sun})$")
    #     ax3.plot(sc_data["heliographicLongitude"][:], sc_data["heliographicLatitude"][:])
    #     ax3.set_xlabel("HGI longitude (deg)")
    #     ax3.set_ylabel("HGI latitude (deg)$")
    # elif sc_id == "STEREO_A":
    #     ax1.plot(sc_data["HGI_LON"][:], sc_data["RAD_AU"][:])
    #     ax1.set_xlabel("HGI longitude (deg)")
    #     ax1.set_ylabel("HGI radius ($R_{sun})$")
    #     ax2.plot(sc_data["HGI_LAT"][:], sc_data["RAD_AU"][:])
    #     ax2.set_xlabel("HGI latitude (deg)")
    #     ax2.set_ylabel("HGI radius ($R_{sun})$")
    #     ax3.plot(sc_data["HGI_LON"][:], sc_data["HGI_LAT"][:])
    #     ax3.set_xlabel("HGI longitude (deg)")
    #     ax3.set_ylabel("HGI latitude (deg)$")
    else:
        raise TypeError

    # Apply the overall title.
    title = (
        sc_id + " - " +
        sc_data["Ephemeris_Epoch"][0].strftime("%m/%d/%Y - %H:%M:%S") +
        " to " +
        sc_data["Ephemeris_Epoch"][-1].strftime("%m/%d/%Y - %H:%M:%S")
    )
    fig.suptitle(title)

    # Save the figure to a file.
    savePic(plot_file_path, doClose=True)
