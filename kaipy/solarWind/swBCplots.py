"""
Generate plots of 1d time series data stored as kaipy.solarWind.TimeSeries
objects.  See scritps/preproc/cda2wind.py for example usage.
"""
# Standard modules
import datetime

# Third party modules
from matplotlib import dates
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Kaipy modules
from kaipy.solarWind.TimeSeries import TimeSeries
import kaipy.kaiViz as kv


def BasicPlot(VarDict, Xname, Yname, Xlabel=True, color='b'):
    """
    Plot a basic line graph using the given variables.

    Args:
        VarDict (dict): A dictionary containing the variables.
        Xname (str): The name of the variable in the dictionary to be plotted on the x-axis.
        Yname (str): The name of the variable in the dictionary to be plotted on the y-axis.
        Xlabel (bool, optional): Whether to display the x-axis label. Defaults to True.
        color (str or list, optional): The color(s) of the line(s) in the plot. Defaults to 'b'.

    Raises:
        Exception: If datetime time-axis elements are mixed with other types.

    Returns:
        None

    Notes:
        - The function uses plt.plot(...) to create the line graph.
        - The y variable can be a time series of tuples/lists of variables to plot simultaneously,
          using different colors.
        - If Xname points to a list of datetime.datetime objects, the plot will be formatted as datetimes.
        - If y['name'] is not a scalar, the y-axis label will only display the units.

    Example:
        >>> VarDict = {'X': {'name': 'Time', 'data': [1, 2, 3]}, 'Y': {'name': 'Value', 'data': [4, 5, 6], 'units': 'm'}}
        >>> BasicPlot(VarDict, 'X', 'Y')
    """

    x = VarDict[Xname]
    y = VarDict[Yname]

    if (np.array(y['data'][0]).size > 1 and
        np.array(y['data'][0]).size == len(color) and
        all([ylen == np.array(y['data'][0]).size for ylen in map(len, y['data'])])):
        for i in range(np.array(y['data'][0]).size):
            plt.plot(x['data'], [yts[i] for yts in y['data']], color=color[i])
    else:
        plt.plot(x['data'], y['data'], color=color)

    if all([type(ts) == datetime.datetime for ts in x['data']]):
        dfmt = dates.DateFormatter('%m/%d/%y-%H:%M')
        plt.gca().xaxis.set_major_formatter(dfmt)
    elif any([type(ts) == datetime.datetime for ts in x['data']]):
        raise Exception('Cannot mix datetime time-axis elements with other types')

    if Xlabel:
        xStr = x['name']
        if len(x['units']) > 0:
            xStr += ' [' + x['units'] + ']'
        plt.xlabel(xStr)
    else:
        plt.xlabel(' ')

    if (np.array(y['name']).size) > 1:
        plt.ylabel('[' + y['units'] + ']', fontsize='small')
    else:
        plt.ylabel(y['name'] + ' [' + y['units'] + ']', fontsize='small')
  
def SummaryPlot(VarDict, Xname):
    """
    Plots every variable in VarDict.

    This is a simple wrapper around the more generic MultiPlotN.  This
    code may have problems with varDicts that store non-time series
    data.  You've been warned.

    Args:
        VarDict (dict): A dictionary containing variables to be plotted. Each variable should be a dictionary with a 'data' key.
        Xname (str): The name of the x-axis variable.

    Returns:
        None
    """
    # Loop over elements in Dict to make sure they all have data 
    plotVariables = []
    for var in VarDict.keys():
        if isinstance(VarDict[var], dict):
            if 'data' in VarDict[var]:
                plotVariables.append(var)

    MultiPlotN([VarDict], Xname, plotVariables)


def MultiPlot(VarDict, Xname, Items, color='b'):
    """
    Plot variables stored in TimeSeries object 'varDict'.

    This function is a simple wrapper around the more generic MultiPlotN function.

    Args:
        VarDict (TimeSeries): The TimeSeries object containing the variables to be plotted.
        Xname (str): The name of the x-axis variable.
        Items (list): A list of variable names to be plotted.
        color (str, optional): The color of the plot. Defaults to 'b'.

    Returns:
        None
    """
    MultiPlotN([VarDict], Xname, Items, [color], [])

    
def MultiPlot2(VarDict, VarDict2, Xname, Items, color1='b', color2='r'):
    """
    Plot items (stored in TimeSeries objects VarDict and VarDict2)
    against one another. One sub plot is created for each item.

    Args:
        VarDict (TimeSeries): A TimeSeries object containing the first set of data.
        VarDict2 (TimeSeries): A TimeSeries object containing the second set of data.
        Xname (str): The name of the x-axis variable.
        Items (list): A list of items to plot against each other.
        color1 (str, optional): The color of the plot for VarDict. Default is 'b' (blue).
        color2 (str, optional): The color of the plot for VarDict2. Default is 'r' (red).

    Returns:
        None
    """
    MultiPlotN([VarDict, VarDict2],
                Xname,
                Items,
                [color1, color2],
                [])


def MultiPlotN(varDicts, Xname, variables, colors = [], legendLabels=[]):
    """
    Creates one subplot for each variable. Each subplot renders a plot
    of that variable for each varDict passed.     
    
    For example:
        one varDict and 5 variables will give 5 subplots with one line each
        5 varDicts and one variable will give 1 subplot with 5 lines

    Args:
        varDicts (list): List of TimeSeries data dictionaries.
        Xname (str): Key of horizontal axes label (must be defined in all varDicts).
        variables (list): List of keys to plot for each varDict.
        colors (list, optional): Color of each line to be drawn. Defaults to [].
        legendLabels (list, optional): Display a legend on the plot. Defaults to [].

    Returns:
        None
    """
    nSubplots = len(variables)

    # Set default colors (all blue)
    if not colors:
        for d in varDicts:
            colors.append('b')    
  
    # Need to declare these here for proper scope:
    axes=None
    ax=None
  
    for plotIndex, variable in enumerate(variables):
        # Sharing the x-axes applies the same zoom to all subplots when
        # user interacts with a single subplot.
        if plotIndex == 0:
            axes = plt.subplot(nSubplots, 1, plotIndex+1)
            ax=axes
        else:
            ax =  plt.subplot(nSubplots, 1, plotIndex+1, sharex=axes)
    
        # Turn off x-axis to prevent pointillism problem with upper subplots.
        #ax.xaxis.set_visible(False)
        
        # Alternate vertical axes
        if plotIndex%2 == 0:
            #ax.yaxis.tick_left()
            ax.yaxis.set_ticks_position('left')
            ax.yaxis.set_label_position('left')
        else:
            #ax.yaxis.tick_right()
            ax.yaxis.set_ticks_position('right')
            ax.yaxis.set_label_position('right')
    
        # Fill in subplot data
        for (idx, data) in enumerate(varDicts):
            if plotIndex < nSubplots-1:
                BasicPlot(data, Xname, variable, Xlabel=False, color=colors[idx])
                
                # remove xticklabels from all but bottom subplot...it seems like
                # someone was attempting something similar with ax.xaxis.set_visible(False)
                # in the past, then commented this out; might be worth asking why -EJR 2/2014
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                BasicPlot(data, Xname, variable, Xlabel=True, color=colors[idx])
                
    plt.subplots_adjust(hspace=0)
    #ax.xaxis.set_visible(True)
    
    plt.subplot(nSubplots, 1, 1)
    if legendLabels:
        plt.legend(legendLabels, loc='best')

def swQuickPlot(UT,D,Temp,Vx,Vy,Vz,Bx,By,Bz,SYMH,interped,fname,
                xBS=None,yBS=None,zBS=None,
                t0fmt='%Y-%m-%d %H:%M:%S',
                utfmt='%H:%M\n%Y-%m%d',
                title="Solar Wind",
                doTrim=True,doEps=False,returnAxs=False):
    """
    Plot solar wind parameters over a specified time period.

    Args:
        UT (array-like): Array of time values.
        D (array-like): Array of solar wind density values.
        Temp (array-like): Array of solar wind temperature values.
        Vx (array-like): Array of solar wind velocity values in the x-direction.
        Vy (array-like): Array of solar wind velocity values in the y-direction.
        Vz (array-like): Array of solar wind velocity values in the z-direction.
        Bx (array-like): Array of solar wind magnetic field values in the x-direction.
        By (array-like): Array of solar wind magnetic field values in the y-direction.
        Bz (array-like): Array of solar wind magnetic field values in the z-direction.
        SYMH (array-like): Array of SYM/H values.
        interped (array-like): Array indicating whether the data is interpolated or not.
        fname (str): File name to save the plot.
        xBS (array-like, optional): Array of Bow Shock position values in the x-direction.
        yBS (array-like, optional): Array of Bow Shock position values in the y-direction.
        zBS (array-like, optional): Array of Bow Shock position values in the z-direction.
        t0fmt (str, optional): Format string for the time values in UT.
        utfmt (str, optional): Format string for the x-axis labels.
        title (str, optional): Title of the plot.
        doTrim (bool, optional): Whether to trim the plot or not.
        doEps (bool, optional): Whether to save the plot in EPS format or not.
        returnAxs (bool, optional): Whether to return the axes objects or not.

    Returns:
        axDict (dict): Dictionary containing the axes objects if returnAxs is True.
    """

    utall = []
    for n in range(len(UT)):
        utall.append(datetime.datetime.strptime(UT[n].decode('utf-8'),t0fmt))

    # constants
    gamma = 5/3.0
    mp = 1.67e-27 #Proton mass [kg]w
    
    # calculating the solar wind dynamic pressure 
    Vmag = np.sqrt(Vx**2+Vy**2+Vz**2)
    Pram = mp*D*Vmag**2*1.0e15 # nPa

    #Setup figure
    fSz = (10,14)
    if xBS is None:
        Nr = 6
    else:
        Nr = 7
    Nc = 1
    clrs = ['#7570b3','#1b9e77','#d95f02','black'] # from colorbrewer2.org for colorblind safe

    fig = plt.figure(figsize=fSz)

    gs = gridspec.GridSpec(Nr,Nc,hspace=0.05,wspace=0.05)

    ax11 = fig.add_subplot(gs[0,0])
    ax12 = fig.add_subplot(gs[3,0])
    ax21 = fig.add_subplot(gs[1,0])
    ax22 = fig.add_subplot(gs[4,0])
    ax31 = fig.add_subplot(gs[2,0])
    if xBS is None:
        ax32 = fig.add_subplot(gs[5,0])
    else:
        ax32 = fig.add_subplot(gs[6,0])
        ax41 = fig.add_subplot(gs[5,0])
    
    smlabel = ['SM-X','SM-Y','SM-Z']
    xvec = np.zeros((len(D),3))+1e9

    fig.suptitle(title,y=0.92,fontsize=14)
    Dlim=np.max(D)-np.min(D)
    ax11.plot(utall,D,color=clrs[3])
    for i in range(3):
        ax11.plot(utall,xvec[:,i],linewidth=4,label=smlabel[i],color=clrs[i])
    kv.SetAxLabs(ax11,"","n [cm^-3]",doBot=True,doLeft=True)
    ax11.set_ylim(np.min(D)-0.05*Dlim,np.max(D)+0.05*Dlim)
    ax11.tick_params(axis="x",direction="in")
    ax11.fill_between(utall, 0, 1, where=interped, alpha=0.4, transform=ax11.get_xaxis_transform())
    plt.setp(ax11.get_xticklabels(),visible=False)
    ax11.legend(ncol=len(smlabel), bbox_to_anchor=(0.5,1),loc='lower center', fontsize='small')
    
    TScl = 1.0e-6
    ax21.plot(utall,Temp*TScl,color=clrs[3])
    kv.SetAxLabs(ax21,"","T [MK]",doBot=True,doLeft=False)
    ax21.tick_params(axis="x",direction="in")
    ax21.fill_between(utall, 0, 1, where=interped, alpha=0.4, transform=ax21.get_xaxis_transform())
    plt.setp(ax21.get_xticklabels(),visible=False)
    
    ax31.plot(utall,Pram,color=clrs[3])
    ax31.xaxis_date()
    kv.SetAxLabs(ax31,"","Dynamic P [nPa]",doBot=True,doLeft=True)
    ax31.tick_params(axis="x",direction="in")
    ax31.fill_between(utall, 0, 1, where=interped, alpha=0.4, transform=ax31.get_xaxis_transform())
    plt.setp(ax31.get_xticklabels(),visible=False)
    
    vScl = 1.0e-3
    secax12 = ax12.twinx()
    ax12.plot(utall,Vx*vScl,color=clrs[0],linewidth=0.95)
    secax12.plot(utall,Vy*vScl,color=clrs[1],linewidth=0.95)
    secax12.plot(utall,Vz*vScl,color=clrs[2],linewidth=0.95)
    secax12.set_ylabel('Vy,z [km/s]')
    kv.SetAxLabs(ax12,"","Vx [km/s]",doBot=True,doLeft=True)
    ax12.tick_params(axis="x",direction="in")
    ax12.fill_between(utall, 0, 1, where=interped, alpha=0.4, transform=ax12.get_xaxis_transform())
    plt.setp(ax12.get_xticklabels(),visible=False)
    
    ax22.plot(utall,Bx,color=clrs[0],linewidth=0.95)
    ax22.plot(utall,By,color=clrs[1],linewidth=0.95)
    ax22.plot(utall,Bz,color=clrs[2],linewidth=0.95)
    ax22.axhline(y=0.0, color='black', linestyle='--',alpha=0.6,linewidth=0.9)
    kv.SetAxLabs(ax22,"","B [nT]",doBot=True,doLeft=False)
    ax22.tick_params(axis="x",direction="in")
    ax22.fill_between(utall, 0, 1, where=interped, alpha=0.4, transform=ax22.get_xaxis_transform())
    plt.setp(ax22.get_xticklabels(),visible=False)
    
    if xBS is not None:
        ax41.plot(utall,xBS,color=clrs[0],linewidth=0.95)
        ax41.plot(utall,yBS,color=clrs[1],linewidth=0.95)
        ax41.plot(utall,zBS,color=clrs[2],linewidth=0.95)
        ax41.axhline(y=0.0, color='black', linestyle='--',alpha=0.6,linewidth=0.9)
        kv.SetAxLabs(ax41,"","BowS [RE]",doBot=True,doLeft=False)
        ax41.tick_params(axis="x",direction="in")
        ax41.fill_between(utall, 0, 1, where=interped, alpha=0.4, transform=ax41.get_xaxis_transform())
        plt.setp(ax41.get_xticklabels(),visible=False)

    ax32.plot(utall,SYMH,color=clrs[3])
    ax32.axhline(y=0.0, color='black', linestyle='--',alpha=0.6,linewidth=0.9)
    ax32.fill_between(utall, 0, 1, where=interped, alpha=0.4, transform=ax32.get_xaxis_transform())
    ax32.xaxis_date()
    xfmt = dates.DateFormatter(utfmt)
    ax32.xaxis.set_major_formatter(xfmt)
    kv.SetAxLabs(ax32,"UT","SYM/H [nT]",doBot=True,doLeft=True)

    # Give axes back to user if they want them
    if returnAxs:
        axDict = {}
        axDict['D'] = ax11
        axDict['Temp'] = ax21
        axDict['dynP'] = ax31
        axDict['V'] = ax12
        axDict['B'] = ax22
        axDict['symh'] = ax32
        if xBS is not None:
            axDict['BowS'] = ax41

        return axDict
    
    # Otherwise, save the figure and leave
    kv.savePic(fname,doEps=doEps,doTrim=doTrim)
    plt.close('all')

