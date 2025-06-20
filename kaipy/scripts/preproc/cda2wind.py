#!/usr/bin/env python

#Converts OMNI output data to Gamera solar wind file to be used as boundary conditions

#Reads from ASCII file
#Time(min) Density (AMU/cm^-3) Vx(km/s) Vy(km/s) Vz(km/s) Cs(km/s) Bx(nT) By(nT) Bz(nT) B(nT) tilt(rad)

#Writes to HDF5 Gamera wind file
#t,D,V,P,B = [s],[#/cm3],[m/s],[nPa],[nT]

#Utilizes cdasws and geopack, make sure to install modules before running. For more info go to https://bitbucket.org/aplkaiju/kaiju/wiki/Gamerasphere

Mp = 1.67e-27 #Proton mass [kg]
gamma = 5/3.0

# Standard modules
import argparse
from argparse import RawTextHelpFormatter
import os
import datetime
import sys

# Third-party modules
import numpy as np
import h5py
import matplotlib.pyplot as plt
from astropy.time import Time
from cdasws import CdasWs

# Kaipy modules
import kaipy.solarWind
from  kaipy.solarWind import swBCplots
from  kaipy.solarWind.OMNI import OMNI
from  kaipy.solarWind.WIND import WIND
from  kaipy.solarWind.SWPC import DSCOVRNC
from gfz_client import GFZClient

cdas = CdasWs()

# ANSI color codes for color output to terminal
class Color:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BOLD = '\033[1m'
    END = '\033[0m'

def bxFit(sw, fileType, filename):
    def bxFitPlot(bxFit_array):
        kaipy.solarWind.swBCplots.BasicPlot(sw.data, 'time_doy', 'bx', color='k')
        plt.plot(sw.data.getData('time_doy'), bxFit_array, 'g')
        plt.title('Bx Fit Coefficients ('+fileType+'):\n$Bx_{fit}(0)$=%f      $By_{coef}$=%f      $Bz_{coef}$=%f' % (coef[0], coef[1], coef[2]) )
        plt.legend(('$Bx$','$Bx_{fit}$'))

    coef = sw.bxFit()

    print('Bx Fit Coefficients are ', coef)
    by = sw.data.getData('by')
    bz = sw.data.getData('bz')
    bxFit = coef[0] + coef[1] * by + coef[2] * bz

    # Save plot
    bxFitPlot(bxFit)
    bxPlotFilename = os.path.basename(filename) + '_bxFit.png'
    print('Saving "%s"' % bxPlotFilename)
    plt.savefig(bxPlotFilename)

    return coef

def ChkTimes(starttime,endtime):
    time_difference = endtime - starttime
    hours_difference = time_difference.total_seconds()/3600.0
    if (starttime > endtime) or (hours_difference < 2.0):
        tsStr = starttime.strftime("%Y-%m-%dT%H:%M:%S")
        teStr = endtime.strftime("%Y-%m-%dT%H:%M:%S")
        sys.exit("Error! Start time (%s) must be al least 2 hours before the end time (%s)"%(tsStr,teStr))

def printErrMsg(errStr):
    print(Color.BOLD+Color.YELLOW+'!!!!!!!!!! ERROR: %s'%(errStr)+ Color.END)
    print(Color.BOLD+Color.YELLOW+'!!!!!!!!!! Not writing bcWind.h5 file'+ Color.END)
    print(Color.BOLD+Color.YELLOW+'!!!!!!!!!! Contact model developers to proceed'+ Color.END)
    sys.exit()

def getPrevDayF107(t0):
    tm1  = t0-datetime.timedelta(days=1)
    tm1  = tm1.replace(hour=0, minute=0, second=0, microsecond=0)
    te1  = tm1.replace(hour=23, minute=59, second=59, microsecond=9999)
    tm1r = tm1.strftime("%Y-%m-%dT%H:%M:%SZ")
    te1r = te1.strftime("%Y-%m-%dT%H:%M:%SZ")

    status,data = cdas.get_data('OMNI2_H0_MRG1HR', ['F10_INDEX1800'], tm1r,te1r)

    #daily values so just return first value
    prevF107 = data['F10_INDEX1800'][0]

    return prevF107

def create_command_line_parser():
    """Create the command-line argument parser.

    Returns:
        argparse.ArgumentParser: Command-line argument parser for this script.
    """
    #Defaults
    fOut = "bcwind.h5"
    mod = "LFM"
    t0Str="2010-01-01T00:00:00"
    t1Str="2010-01-01T02:00:00"
    Ts = 0.0
    sigma = 3.0
    tOffset = 0.0
    obs="OMNI"
    filename=None
    doBs = True
    doEps = False

    #Usually f107   above 300 is not reliable. The daily value could be distorted by flare emissions even if the flare may only last a short time during a day.


    MainS = """ This script does several things:
                  1. Fetch OMNI data from CDAWeb between the specified times (must be at least 2 hours in length)
                  2. Generate standard plots of solar wind data
                  3. Write output in a model file format.
                     - "LFM" format will:
                         a. Generate coefficients for Bx Fit
                         b. Save a bcwind.h5 file
                     - "TIEGCM" format will:
                         a. Compute 15-minute boxcar average lagged by 5 minutes
                         b. Sub-sample at 5-minutes
                         c. Write NetCDF IMF data file
    """

    parser = argparse.ArgumentParser(description=MainS, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-t0',type=str,metavar="TStart",default=t0Str,help="Start time in 'YYYY-MM-DDThh:mm:ss' (default: %(default)s)")
    parser.add_argument('-t1',type=str,metavar="TStop",default=t1Str,help="End time in 'YYYY-MM-DDThh:mm:ss' (default: %(default)s)")
    parser.add_argument('-obs',type=str,metavar="OMNI",default=obs,help="Select spacecraft to obtain observations from (default: %(default)s)")
    parser.add_argument('-offset',type=float,metavar="tOffset",default=tOffset,help="Minutes to offset spacecraft observation and simulation t0 (default: %(default)s)")
    parser.add_argument('-o',type=str,metavar="wind.h5",default=fOut,help="Output Gamera wind file (default: %(default)s)")
    parser.add_argument('-m',type=str,metavar="LFM",default=mod,help="Format to write.  Options are LFM or TIEGCM (default: %(default)s)")
    parser.add_argument('-TsG',type=float,metavar="GAMERA_TStart",default=Ts,help="Gamera start time [min] (default: %(default)s)")
    parser.add_argument('-TsL',type=float,metavar="LFM_TStart",default=Ts,help="LFM start time [min] (default: %(default)s)")
    parser.add_argument('-bx', action='store_true',default=False,help="Include Bx through ByC and BzC fit coefficients (default: %(default)s)")
    parser.add_argument('-bs', action='store_false',default=True,help="Include Bowshock location (default: %(default)s)")
    parser.add_argument('-interp', action='store_true',default=False,help="Include shaded region on plots where data is interpolated (default: %(default)s)")
    parser.add_argument('-filter', action='store_true',default=False,help="Include additional filtering of data to remove outlier points (default: %(default)s)")
    parser.add_argument('-sig',type=float,metavar="sigma",default=sigma,help="N used in N*sigma used for filtering threshold above which will be thrown out (default: %(default)s)")
    parser.add_argument('-eps', action="store_true",default=False,help="Output eps figure. (default: %(default)s)")
    parser.add_argument('-fn', type=str,metavar="filename",default=filename,help="Name of Wind file. Only used if obs is WINDF. (default: %(default)s)")
    parser.add_argument('-f107', type=float,default=None,help="Set f10.7 value to use in bcwind file. Only used if no data available. (default: %(default)s)")
    parser.add_argument('-kp',   type=float,default=None,help="Set Kp value to use in bcwind file. Only used if no data available. (default: %(default)s)")
    parser.add_argument('-safe', action='store_true',default=False,help="Run in SAFE mode. Does not create the h5 file if certain conditions are not met (default: %(default)s)")

    return parser

def main():
    global cdas
    #Defaults
    maxf107  = 300.0
    minMfast = 1.5
    parser = create_command_line_parser()
    #Finalize parsing
    args = parser.parse_args()

    fOut = args.o
    mod = args.m
    TsG = args.TsG
    TsL = args.TsL
    includeBx = args.bx
    doBs = args.bs
    plotInterped = args.interp
    doCoarseFilter = args.filter
    sigma = args.sig
    doEps = args.eps
    obs = args.obs
    f107Def = args.f107
    kpDef = args.kp
    inSafeMode = args.safe

    if (obs == 'OMNIW' and args.fn is None): raise Exception('Error: OMNIW requires -fn to specify a WIND file')

    t0Str = args.t0
    t1Str = args.t1

    tOffset = args.offset

    fmt='%Y-%m-%dT%H:%M:%S'

    t0 = datetime.datetime.strptime(t0Str,fmt)
    t1 = datetime.datetime.strptime(t1Str,fmt)
    t0r = t0.strftime("%Y-%m-%dT%H:%M:%SZ")
    t1r = t1.strftime("%Y-%m-%dT%H:%M:%SZ")

    ChkTimes(t0,t1)

    cdas = CdasWs()

    # calculating average F10.7 over specified time period, can be converted into a timeseries
    # pulling data from CDAWeb database
    print('Retrieving f10.7 data from CDAWeb')

    try:
        statusf107,data = cdas.get_data('OMNI2_H0_MRG1HR', ['F10_INDEX1800','KP1800'], t0r,t1r)

        totalMin = (t1-t0).days*24.0*60.0+(t1-t0).seconds/60
        tmin = np.arange(totalMin)
        t107 = data['Epoch']
        t107min = np.zeros(len(t107))
        for i in range(len(t107)):
            t107min[i]=(t107[i]-t0).days*24.0*60.0+(t107[i]-t0).seconds/60

        f107=data['F10_INDEX1800']

        if (np.all(f107 > maxf107)): #bad values set to 999.9 by cdas
            if inSafeMode:
                printErrMsg('No valid f10.7 data')

            print(Color.GREEN+'!!!!!!!!!! Warning: No valid f10.7 data, Attempting to take value from previous day !!!!!!!!!!'+Color.END)
            prevF107 = getPrevDayF107(t0)

            if(prevF107<=maxf107):
                print(Color.GREEN+'\tSuccesful. Setting to f10.7 to %f'%(prevF107)+Color.END)
                f107[:] = prevF107
            else:
                if f107Def is not None:
                    print(Color.GREEN+'!!!!!!!!!! Warning: No valid f10.7 data on previous day either. Setting f10.7 to %f!!!!!!!!!!'%(f107Def)+Color.END)
                    f107[:] = f107Def
                else:
                    sys.exit(Color.YELLOW+'!!!!!!!!!! Error: No valid f10.7 data on previous day either. Set f10.7 to use with -f107 flag !!!!!!!!!!'+Color.END)
        elif (f107[0] > maxf107):
            indF = np.where(f107<maxf107)[0][0]
            F107start = f107[indF]
            f107[0] = F107start
            print(Color.GREEN+'!!!!!!!!!! Warning: f10.7 starts with a bad value (>%d), setting initial value to first good value: %f !!!!!!!!!!'%(maxf107,F107start)+Color.END)

        #Linearly interpolating and converting hourly cadence to minutes
        f107min = np.interp(tmin, t107min[f107 < maxf107], f107[f107 < maxf107] )

        kp = data['KP1800']
        client = GFZClient()
        if (np.all(kp == 99)):
            try:
                    (time,index,status) = client.get_kp_index(starttime=t0Str, endtime=t1Str, index='Kp')
                    tkp = np.zeros(len(time))
                    for i in range(len(time)):
                        tkp[i] = (datetime.datetime.strptime(time[i],fmt+"Z") - t0).days*24.0*60.0 + (datetime.datetime.strptime(time[i],fmt+"Z") - t0).seconds/60
                    kpmin = np.interp(tmin,tkp,index)
            except:
                if inSafeMode:
                    printErrMsg('No valid Kp data')

                if kpDef is not None:
                    print(Color.BLUE+"!!!!!!!!!! Warning: No valid Kp data, setting all values in array to %d!!!!!!!!!!"%(kpDef)+Color.END)
                    kp[:]   = kpDef
                    kpmin   = np.interp(tmin, t107min, kp) # if no good values, setting all to bad values
                else:
                    sys.exit(Color.YELLOW+'!!!!!!!!!! Error: No valid Kp data. Set Kp to use with -kp flag !!!!!!!!!!'+Color.END)
        else:
            if (kp[0] == 99):
                indF = np.where(kp!=99)[0][0]
                KpStart = kp[indF]
                kp[0] = KpStart
                print(Color.BLUE+'!!!!!!!!!! Warning: Kp starts with a bad value, setting to first good value: %d !!!!!!!!!!'%(KpStart)+Color.END)
            kpmin   = np.interp(tmin, t107min[kp != 99], kp[kp!=99]/10.0)
    except Exception as e:
            if isinstance(e, SystemExit):
                raise  # Re-raise SystemExit exception
            else:
                if inSafeMode:
                    printErrMsg('Issue pulling f10.7 and kp data from OMNI, need to be set manually.')

                print(Color.DARKCYAN+"+'!!!!!!!!!! Issue pulling f10.7 and kp data from OMNI, setting manually"+Color.END)
                totalMin = (t1-t0).days*24.0*60.0+(t1-t0).seconds/60
                tmin = np.arange(totalMin)
                totalMin = totalMin-1
                if f107Def is None:
                    sys.exit(Color.YELLOW+'!!!!!!!!!! Error: Default f10.7 is not set. Update using -f107 flag at execution !!!!!!!!!!'+Color.END)
                else:
                    print(Color.DARKCYAN+'\tSetting f10.7 to: %f !!!!!' %(f107Def)+Color.END)
                    f107min = np.ones(int(totalMin))*f107Def
                try:
                    (time,index,status) = getKpindex.getKpindex(t0Str+"Z",t1Str+"Z",'Kp')
                    tkp = np.zeros(len(time))
                    for i in range(len(time)):
                        tkp[i] = (datetime.datetime.strptime(time[i],fmt+"Z") - t0).days*24.0*60.0 + (datetime.datetime.strptime(time[i],fmt+"Z") - t0).seconds/60
                    kpmin = np.interp(tmin,tkp,index)
                except:
                    if kpDef is None:
                        sys.exit(Color.YELLOW+'!!!!!!!!!! Error: Default Kp is not set. Update using -kp flag at execution !!!!!!!!!!'+Color.END)
                    else:
                        print(Color.DARKCYAN+'Setting kp to: %f (can be changed with -kp flag at execution) !!!!!' %(kpDef)+Color.END)
                        kpmin = np.ones(int(totalMin))*kpDef

    if (obs == 'OMNI'):
        fileType = 'OMNI'
        filename = 'OMNI_HRO_1MIN.txt'

        #obtain 1 minute resolution observations from OMNI dataset
        print('Retrieving solar wind data from CDAWeb')
        status,fIn = cdas.get_data(
           'OMNI_HRO_1MIN',
            ['BX_GSE','BY_GSE','BZ_GSE',
            'Vx','Vy','Vz',
            'proton_density','T',
            'AE_INDEX','AL_INDEX','AU_INDEX','SYM_H',
            'BSN_x','BSN_y','BSN_z'],
            t0r,t1r)
        # Read the solar wind data into 'sw' object and interpolate over the bad data.
        if (doCoarseFilter): print(f"Using Coarse Filtering, removing values {sigma} sigma from the mean")
        sw = eval('kaipy.solarWind.'+fileType+'.'+fileType)(fIn,doFilter=doCoarseFilter,sigmaVal=sigma)

    elif (obs == 'WIND'):
        # CDAS tips.
        # use CDAweb to get the name of the spacecraft variables you want, such as "C4_CP_FGM_SPIN"
        # then use cdas.get_variables('sp_phys','C4_CP_FGM_SPIN') to get a list of variables
        # variable names do not exactly match the cdaweb outputs so check to make sure variables
        fileType = 'WIND'
        filename = 'WIND'
        tBuffer = 100 # Extra padding for propagation
        t0rb = (t0 - datetime.timedelta(minutes=tBuffer)).strftime("%Y-%m-%dT%H:%M:%SZ")
        t1rb = (t1 + datetime.timedelta(minutes=tBuffer)).strftime("%Y-%m-%dT%H:%M:%SZ")

        status,fMFI = cdas.get_data(
           'WI_K0_MFI',
           ['BGSEc'],
           t0rb,
           t1rb
        )
        if status['http']['status_code'] != 200:
            printErrMsg('No valid WIND MFI data during this period')
        status,fSWE = cdas.get_data(
           'WI_K0_SWE',
           ['SC_pos_gse','QF_V', 'QF_Np', 'V_GSE','THERMAL_SPD', 'Np'],
           t0rb,
           t1rb
        )
        if status['http']['status_code'] != 200:
            printErrMsg('No valid WIND SWE data during this period')
        sw = eval('kaipy.solarWind.'+fileType+'.'+fileType)(fSWE,fMFI,t0,t1)
    elif (obs == 'OMNIW'):
        fileType = 'OMNI'
        fileType2 = 'OMNIW'
        filename = args.fn
        doBs = True

        print("Working with OMNIW algorithm")
        # Read the solar wind data into 'sw' object and interpolate over the bad data.
        sw = eval('kaipy.solarWind.'+fileType+'.'+fileType2)(filename)
        filename = 'OMNIW_'+filename

    elif (obs == 'DSCOVRNC'):
        fileType = 'SWPC'
        fileType2 = 'DSCOVRNC'
        doBs = False

        sw = eval('kaipy.solarWind.'+fileType+'.'+fileType2)(t0,t1)
        filename = fileType2
    else:
        raise Exception('Error:  Not able to obtain dataset from spacecraft. Please select another mission.')

    # Do output format-specific tasks:
    if (mod == 'TIEGCM'):
        # Write TIEGCM IMF solar wind file
        #FIXME: need to update when want to include, example code in pyLTR.SolarWind.Writer.TIEGCM
        raise Exception('Error:  Cannot currently produce TIEGCM output.')
    elif (mod == 'LFM'):

        if (includeBx):
            print("\tUsing Bx fields")
            # Bx Fit
            bCoef=bxFit( sw, fileType, filename)
            # Setting Bx0 to zero to enforce a planar front with no Bx offset
            bCoef[0] = 0.0
        else:
            print("\tNot using Bx fields")
            bCoef = [0.0, 0.0, 0.0]

        # Interpolate to one minute:
        time_1minute = range(int(sw.data.getData('time_min').min()),
                             int(sw.data.getData('time_min').max()) )
        n    = np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('n'))
        tp   = np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('t'))
        vx   = np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('vx'))
        vy   = np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('vy'))
        vz   = np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('vz'))
        cs   = np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('cs'))
        va   = np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('va'))
        bx   = np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('bx'))
        by   = np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('by'))
        bz   = np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('bz'))
        b    = np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('b'))

        try:
            ae    = np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('ae'))
        except:
            ae    = np.zeros(len(time_1minute))

        try:
            al    = np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('al'))
        except:
            al    = np.zeros(len(time_1minute))

        try:
            au    = np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('au'))
        except:
            au    = np.zeros(len(time_1minute))

        try:
            symh  = np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('symh'))
        except:
            symh  = np.zeros(len(time_1minute))

        if doBs:
            bsx   = np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('xBS'))
            bsy   = np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('yBS'))
            bsz   = np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('zBS'))

        #grab info on where data is interpolated to include on plots if wanted
        interped = np.zeros((11,len(symh)))
        interped[0,:] =np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('isBxInterped'))
        interped[1,:] =np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('isByInterped'))
        interped[2,:] =np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('isBzInterped'))
        interped[3,:] =np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('isVxInterped'))
        interped[4,:] =np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('isVyInterped'))
        interped[5,:] =np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('isVzInterped'))
        interped[6,:] =np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('isNInterped'))
        try:
            interped[7,:] =np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('isCsInterped'))
        except:
            interped[7,:] =np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('isTInterped'))

        if doBs:
          interped[8,:]  =np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('isxBSInterped'))
          interped[9,:]  =np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('isyBSInterped'))
          interped[10,:] =np.interp(time_1minute, sw.data.getData('time_min'), sw.data.getData('iszBSInterped'))

        #finding locations where any variable is interpolated
        isInterp=np.any(interped,axis=0)
        pltInterp = np.zeros(len(isInterp),dtype=bool)
        if (plotInterped):
            pltInterp = isInterp

        # calculating fast magnetosonic mach number
        mfast = np.sqrt((vx**2+vy**2+vz**2)/(cs**2+va**2))

        #initalize matrix to hold solar wind data
        if doBs:
            lfmD = np.zeros((n.shape[0],21))
        else:
            lfmD = np.zeros((n.shape[0],18))

        date = sw.data.getData('meta')['Start date']

        nSub = 0
        vxSub = []
        for i,time in enumerate(time_1minute):
            # Convert relevant quantities to SM Coordinates
            v_sm = sw._gsm2sm(date+datetime.timedelta(minutes=time), vx[i],vy[i],vz[i])
            b_sm = sw._gsm2sm(date+datetime.timedelta(minutes=time), bx[i],by[i],bz[i])
            if doBs:
                bs_sm = sw._gsm2sm(date+datetime.timedelta(minutes=time), bsx[i],bsy[i],bsz[i])
            tilt = sw._getTiltAngle(date+datetime.timedelta(minutes=time))

            if doBs:
                #lfmD[i] = [time,n[i],v_sm[0],v_sm[1],v_sm[2],cs[i],b_sm[0],b_sm[1],b_sm[2],b[i],tilt,ae[i],al[i],au[i],symh[i],tp[i],va[i],mfast[i],bs_sm[0],bs_sm[1],bs_sm[2]]
                lfmD[i,:] = [time,n[i],v_sm[0][0],v_sm[1][0],v_sm[2][0],cs[i],b_sm[0][0],b_sm[1][0],b_sm[2][0],b[i],tilt[0],ae[i],al[i],au[i],symh[i],tp[i],va[i],mfast[i],bs_sm[0][0],bs_sm[1][0],bs_sm[2][0]]
            else:
                #lfmD[i] = [time,n[i],v_sm[0],v_sm[1],v_sm[2],cs[i],b_sm[0],b_sm[1],b_sm[2],b[i],tilt,ae[i],al[i],au[i],symh[i],tp[i],va[i],mfast[i]]
                lfmD[i,:] = [time,n[i],v_sm[0],v_sm[1],v_sm[2],cs[i],b_sm[0],b_sm[1],b_sm[2],b[i],tilt,ae[i],al[i],au[i],symh[i],tp[i],va[i],mfast[i]]
            
            if mfast[i] < minMfast:
                nSub += 1
                vxSub.append(v_sm[0])

        if nSub > 0:
            import kaipy.gamera.gamGrids as gg
            #Pull defaul LFM grid
            gIn = "./lfmG"
            Nc0 = 8 #Number of outer i cells to cut out from LFM grid (OCT)
            xx0,yy0 = gg.LoadTabG(gIn,Nc0)
            #Calculate Rout in sunward direction from grid
            Rout  = np.sqrt(xx0[-1,0]**2.0  + yy0[-1,0]**2.0) #[Re]
            Re_km    = 6378.1
            maxVsub  = abs(max(vxSub))
            nSubCrit = (Rout*Re_km)/maxVsub/60.0 # mins
            if inSafeMode and (nSub > nSubCrit):
                printErrMsg("Low Mach number solar wind persists for too long (%d minutes)"%(nSub))

            print()
            print(Color.CYAN+"!!!!!!!!!! WARNING LOW MACH NUMBER:  Mfast < %.3f for %d minutes, may want to extend grid !!!!!!!!!!"%(minMfast,nSub)+Color.END)
            print()

        print("Converting to Gamera solar wind file")
        Nt,Nv = lfmD.shape
        print("\tFound %d variables and %d lines"%(Nv,Nt))

        #Convert LFM time to seconds and reset to start at 0
        print("\tOffsetting from LFM start (%5.2f min) to Gamera start (%5.2f min)"%(TsL,TsG))
        T0 = lfmD[:,0].min()
        T = (lfmD[:,0]-TsL+TsG)*60

        #Calculating time in UT
        UT = []
        [UT.append(np.bytes_(date+datetime.timedelta(seconds=i)).strip()) for i in T]

        #Calculating time in MJD
        MJD = []
        mjdRef=Time(date).mjd
        [MJD.append(mjdRef+i/86400.0) for i in T]

        #Density, temperature, magnetic field, and tilt don't require scaling
        D   = lfmD[:,1]
        ThT = lfmD[:,10]
        Bx  = lfmD[:,6] # overwritten by Gamera using the coefficients
        By  = lfmD[:,7]
        Bz  = lfmD[:,8]

        #Activity indices do not require scaling
        AE = lfmD[:,11]
        AL = lfmD[:,12]
        AU = lfmD[:,13]
        SYMH = lfmD[:,14]
        
        # scaling Temperature from kK->K
        Temp = lfmD[:,15]*1.0e+3

        #Velocity
        vScl = 1.0e+3 #km/s->m/s
        Vx  = vScl*lfmD[:,2]
        Vy  = vScl*lfmD[:,3]
        Vz  = vScl*lfmD[:,4]

        Cs = vScl*lfmD[:,5] #km/s->m/s
        Va = vScl*lfmD[:,16]

        Mfast = lfmD[:,17]

        #Bowshock position
        if doBs:
            xBS  = lfmD[:,18]
            yBS  = lfmD[:,19]
            zBS  = lfmD[:,20]

        # Save a plot of the solar wind data.
        if doEps:
            swPlotFilename = os.path.basename(filename) + '.eps'
        else:
            swPlotFilename = os.path.basename(filename) + '.png'

        print('Saving "%s"' % swPlotFilename)
        if doBs:
            kaipy.solarWind.swBCplots.swQuickPlot(UT,D,Temp,Vx,Vy,Vz,Bx,By,Bz,SYMH,pltInterp,swPlotFilename,xBS,yBS,zBS,doEps=doEps)
        else:
            kaipy.solarWind.swBCplots.swQuickPlot(UT,D,Temp,Vx,Vy,Vz,Bx,By,Bz,SYMH,pltInterp,swPlotFilename,doEps=doEps)

        print("Writing Gamera solar wind to %s"%(fOut))
        with h5py.File(fOut,'w') as hf:
            hf.create_dataset("T" ,data=T)
            hf.create_dataset("UT",data=UT)
            hf.create_dataset("MJD",data=MJD)
            hf.create_dataset("D" ,data=D)
            hf.create_dataset("Temp" ,data=Temp)
            hf.create_dataset("Vx",data=Vx)
            hf.create_dataset("Vy",data=Vy)
            hf.create_dataset("Vz",data=Vz)
            hf.create_dataset("Bx",data=Bx)
            hf.create_dataset("By",data=By)
            hf.create_dataset("Bz",data=Bz)
            hf.create_dataset("tilt",data=ThT)
            hf.create_dataset("ae",data=AE)
            hf.create_dataset("al",data=AL)
            hf.create_dataset("au",data=AU)
            hf.create_dataset("symh",data=SYMH)
            hf.create_dataset("Interped",data=1*isInterp)
            hf.create_dataset("f10.7",data=f107min)
            hf.create_dataset("Kp",data=kpmin)
            hf.create_dataset("Bx0",data=bCoef[0])
            hf.create_dataset("ByC",data=bCoef[1])
            hf.create_dataset("BzC",data=bCoef[2])
            hf.create_dataset("Va",data=Va)
            hf.create_dataset("Cs",data=Cs)
            if doBs:
                hf.create_dataset("xBS",data=xBS)
                hf.create_dataset("yBS",data=yBS)
                hf.create_dataset("zBS",data=zBS)
            hf.create_dataset("Magnetosonic Mach",data=Mfast)

    else:
        raise Exception('Error:  Misunderstood output file format.')



if __name__ == '__main__':
    main()