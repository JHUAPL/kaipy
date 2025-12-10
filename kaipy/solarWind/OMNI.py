# Standard modules
import datetime
import re

# Third-party modules
import numpy
from cdasws import CdasWs
from pytplot import get_data
try:
    # Try the newer import path first (for recent pyspedas versions)
    from pyspedas.projects import kyoto
except ImportError:
    # Fall back to the older import path
    from pyspedas import kyoto

# Kaipy modules
import kaipy.transform
from kaipy.solarWind.SolarWind import SolarWind
from kaipy.kdefs import *

class OMNI(SolarWind):
    """
    Processes OMNI Solar Wind data from CDAweb [http://cdaweb.gsfc.nasa.gov/].
    Data stored as a kaipy.solarWind.SolarWind object in GSE coordinates.

    Parameters:
        filename (str): The path to the OMNI Solar Wind data.
        doFilter (bool): Flag indicating whether to apply coarse filtering to remove outliers. Default is set to False.
        sigmaVal (float): The number of standard deviations to use for the coarse filtering. Default is to remove data points 3 standard deviation from the mean if doFilter is True.

    Attributes:
        filter (bool): Flag indicating whether filtering is enabled.
        sigma (float): The number of standard deviations used for filtering.
        bad_data (list): List of values considered as bad data.
        data (TimeSeries): TimeSeries object to store the solar wind data.

    Methods:
        __init__(self, filename=None, doFilter=False, sigmaVal=3.0): Initializes the OMNI object.
        __read(self, filename): Reads the solar wind data and stores the results in self.data as TimeSeries objects.
        __readData(self, fh): Reads the variables from the file and returns a 2D array containing the data.
        __appendMetaData(self, date, filename): Adds standard metadata to the data dictionary.
        _removeBadData(self, data, datanames=['Time','Bx','By','Bz','Vx','Vy','Vz','n','Temp','AE','AL','AU','SYMH','BowShockX','BowShockY','BowShockZ'], hasBeenInterpolated=None): Linearly interpolates over bad data in the data array.
        _coarseFilter(self, dataArray, hasBeenInterpolated): Uses coarse noise filtering to remove outliers from the data array.
    """

    def __init__(self, filename=None, doFilter=False, sigmaVal=3.0):
        """
        Initialize the OMNI class.

        Args:
            filename (str, optional): The name of the file to read data from. Defaults to None.
            doFilter (bool, optional): Flag indicating whether to apply filtering. Defaults to False.
            sigmaVal (float, optional): The value of sigma for filtering. Defaults to 3.0.
        """
        SolarWind.__init__(self)

        self.filter = doFilter
        self.sigma = sigmaVal

        self.bad_data = [-999.900,
                         99999.9,  # V
                         9999.99,  # B
                         999.990,  # density
                         1.00000E+07,  # Temperature
                         9999999.0,  # Temperature
                         99999  # Activity indices
                         ]

        self.__read(filename)

    def __read(self, filename):
        """
        Read the solar wind file and store the results in the self.data TimeSeries object.

        Args:
            filename (str): The path to the OMNI Solar Wind file.

        Returns:
            None
        """
        (startDate, dates, data, datanames) = self.__readData(filename)
        (dataArray, hasBeenInterpolated) = self._removeBadData(data,datanames)
        if self.filter:
            (dataArray, hasBeenInterpolated) = self._coarseFilter(dataArray, hasBeenInterpolated)
        self._storeDataDict(dates, dataArray, hasBeenInterpolated)
        self.__appendMetaData(startDate, filename)
        self._appendDerivedQuantities()

        
    def __readData(self, fh):
        """
        Read the solar wind file and store the results in the self.data TimeSeries object.

        Args:
            fh (file handle): The file handle of the OMNI Solar Wind file.

        Returns:
            tuple: A tuple containing the start date, dates, data, and datanames.
            - start date (str): The start date of the data.
            - dates (list): A list of dates.
            - data (list): A 2D list containing the data.
            - datanames (list): A list of variable names.
        """
        #pulling variables from file
        time=fh['Epoch']
        bx=fh['BX_GSE']
        by=fh['BY_GSE']
        bz=fh['BZ_GSE']
        vx=fh['Vx']
        vy=fh['Vy']
        vz=fh['Vz']
        n=fh['proton_density']
        T=fh['T']
        ae=fh['AE_INDEX']
        al=fh['AL_INDEX']
        au=fh['AU_INDEX']
        symh=fh['SYM_H']
        xBow = fh['BSN_x']       #RE
        yBow = fh['BSN_y']       #RE
        zBow = fh['BSN_z']       #RE

        dates = []
        rows = []
        for i in range(len(time)):
          
            startTime = time[0]
            #calculating minutes from the start time
            nMin = self._deltaMinutes(time[i],startTime)

            data = [nMin,bx[i],by[i],bz[i],vx[i],vy[i],vz[i],n[i],T[i],ae[i],al[i],au[i],symh[i],xBow[i],yBow[i],zBow[i]]

            dates.append( time[i] )
            rows.append( data )

        datanames = ['Time','Bx','By','Bz','Vx','Vy','Vz','n','Temperature','AE','AL','AU','SYMH','BowShockX','BowShockY','BowShockZ']

        return (startTime, dates, rows, datanames)

    def __appendMetaData(self, date, filename):
        """
        Add standard metadata to the data dictionary.

        Args:
            date (str): The start date of the data.
            filename (str): The path to the OMNI Solar Wind file.

        Returns:
            None
        """
        metadata = {'Model': 'OMNI',
                    'Source': filename,
                    'Date processed': datetime.datetime.now(),
                    'Start date': date
                    }
        
        self.data.append(key='meta',
                         name='Metadata for OMNI Solar Wind file',
                         units='n/a',
                         data=metadata)

    def _removeBadData(self, data, datanames=['Time','Bx','By','Bz','Vx','Vy','Vz','n','Temp','AE','AL','AU','SYMH','BowShockX','BowShockY','BowShockZ'], hasBeenInterpolated = None):
        """
        Linearly interpolate over bad data (defined by self.bad_data
        list) for each variable in dataStrs.
        
        Args:
            data (list): A 2D list containing the data.
        
        Returns:
            numpy.ndarray: Interpolated floating-point numpy array.
            numpy.ndarray: 2D array that identifies if bad values were removed/interpolated.

        Note:
            This is remarkably similar to __coarseFilter!
            Refactoring to keep it DRY wouldn't be a bad idea. . .
        """
        #assert( len(data[0]) == 13 )
        nvar = len(data[0])
        if (hasBeenInterpolated is None):
            hasBeenInterpolated = numpy.empty((len(data), nvar-1))
            hasBeenInterpolated.fill(False)

        for varIdx in range(1,nvar):

            lastValidIndex = -1
            for curIndex,row in enumerate(data):
                if row[varIdx] in numpy.float32(self.bad_data) or numpy.isnan(row[varIdx]) or numpy.ma.is_masked(row[varIdx]):
                    # This item has bad data.
                    hasBeenInterpolated[curIndex, varIdx-1] = True
                    if (lastValidIndex == -1) & (curIndex == len(data)-1):
                        # Data does not have at least one valid element!
                        # Setting all values to 0 so that file can still be made
                        print("No good elements, setting all values to 0 for variable: ", datanames[varIdx])
                        data[curIndex][varIdx] = 0.
                        #raise Exception("First & Last datapoint(s) in OMNI "+
                        #                  "solar wind file are invalid.  Not sure "+
                        #                  "how to interpolate across bad data.")
                    elif (curIndex == len(data)-1):
                        # Clamp last bad data to previous known good data.
                        data[curIndex][varIdx] = data[lastValidIndex][varIdx]
                    else:
                        # Note the bad data & skip this element for now.
                        # We will linearly interpolate between valid data
                        continue

                # At this point, curIndex has good data.
                if (lastValidIndex+1) == curIndex:
                    # Set current element containing good data.
                    data[curIndex][varIdx] = float( row[varIdx] )
                else:
                    # If first index is invalid, clamp to first good value.
                    if lastValidIndex == -1:
                        lastValidIndex = 0
                        data[lastValidIndex][varIdx] = data[curIndex][varIdx]

                    # Linearly interpolate over bad data.
                    interpolated = numpy.interp(range(lastValidIndex, curIndex), # x-coords of interpolated values
                                                [lastValidIndex, curIndex],  # x-coords of data.
                                                [float(data[lastValidIndex][varIdx]), float(data[curIndex][varIdx])]) # y-coords of data.
                    # Store the results.
                    for j,val in enumerate(interpolated):
                        data[lastValidIndex+j][varIdx] = val
                lastValidIndex = curIndex

        return (numpy.array(data, float), hasBeenInterpolated)

    def _coarseFilter(self, dataArray, hasBeenInterpolated):
        """
        Use coarse noise filtering to remove values outside the number
        deviations (set by self.sigma) from mean of all values in the plotted time
        interval.

        Args:
            dataArray (numpy.ndarray): 2D numpy array. Each row is a list containing [nMinutes, Bx, By, Bz, Vx, Vy, Vz, rho, temp, ae, al, au, symh].
            hasBeenInterpolated (numpy.ndarray): 2D boolean array. Each row is a list of boolean values denoting whether dataArray[:,1:9] was derived/interpolated from the raw data (i.e., bad points removed).

        Returns:
            numpy.ndarray: Same structure as input array with bad elements removed.
            numpy.ndarray: Same as input array with interpolated values stored.

        Note:
            This is remarkably similar to _removeBadData!
            Refactoring to keep it DRY wouldn't be a bad idea...
        """
        
        stds = []
        means = []
        nvar = len(dataArray[0])
        for varIdx in range(1,nvar):
            stds.append( numpy.nanstd(dataArray[:,varIdx]) )
            means.append( numpy.nanmean(dataArray[:,varIdx]) )
            
            # Linearly interpolate over data that exceeds # of standard
            # deviations from the mean set by self.sigma (default = 3)
            lastValidIndex = -1
            for curIndex,row in enumerate(dataArray):
                # Are we outside set # of deviations from mean?
                if abs(means[varIdx-1] - row[varIdx]) > self.sigma*stds[varIdx-1]:
                    hasBeenInterpolated[curIndex, varIdx-1] = True
                    if (curIndex == len(dataArray)-1):
                        # Clamp last bad data to previous known good data.
                        dataArray[curIndex][varIdx] = dataArray[lastValidIndex][varIdx]
                    else:
                        # Note the bad data & skip this element for now.
                        # We will linearly interpolate between valid data
                        continue

                if (lastValidIndex+1) != curIndex:
                    # If first index is invalid, clamp to first good value.
                    if lastValidIndex == -1:
                        lastValidIndex = 0
                        dataArray[lastValidIndex][varIdx] = dataArray[curIndex][varIdx]

                    # Linearly interpolate over bad data.
                    interpolated = numpy.interp(range(lastValidIndex, curIndex), # x-coords of interpolated values
                                                [lastValidIndex, curIndex],  # x-coords of data.
                                                [float(dataArray[lastValidIndex][varIdx]), float(dataArray[curIndex][varIdx])]) # y-coords of data.
                    # Store the results.
                    for j,val in enumerate(interpolated):
                        dataArray[lastValidIndex+j][varIdx] = val
                lastValidIndex = curIndex

        return (dataArray, hasBeenInterpolated)

    def _storeDataDict(self, dates, dataArray, hasBeenInterpolated):
        """
        Populate self.data TimeSeries object via the 2D dataArray read from file.

        Args:
            dates (list): A list of dates.
            dataArray (numpy.ndarray): 2D numpy array containing the solar wind data.

        Returns:
            None
        """
        self._gse2gsm(dates, dataArray)

        self.data.append('time_min', 'Time (Minutes since start)', 'min', dataArray[:,0])

        # Magnetic field
        self.data.append('bx', 'Bx (gsm)', r'$\mathrm{nT}$', dataArray[:,1])
        self.data.append('isBxInterped', 'Is index i of By interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,0])
        
        self.data.append('by', 'By (gsm)', r'$\mathrm{nT}$', dataArray[:,2])
        self.data.append('isByInterped', 'Is index i of By interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,1])

        self.data.append('bz', 'Bz (gsm)', r'$\mathrm{nT}$', dataArray[:,3])
        self.data.append('isBzInterped', 'Is index i of Bz interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,2])

        # Velocity
        self.data.append('vx', 'Vx (gsm)', r'$\mathrm{km/s}$', dataArray[:,4])
        self.data.append('isVxInterped', 'Is index i of Vx interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,3])

        self.data.append('vy', 'Vy (gsm)', r'$\mathrm{km/s}$', dataArray[:,5])
        self.data.append('isVyInterped', 'Is index i of Vy interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,4])

        self.data.append('vz', 'Vz (gsm)', r'$\mathrm{km/s}$', dataArray[:,6])
        self.data.append('isVzInterped', 'Is index i of Vz interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,5])

        # Density
        self.data.append('n', 'Density', r'$\mathrm{1/cm^3}$', dataArray[:,7])
        self.data.append('isNInterped', 'Is index i of N interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,6])

        # Temperature
        self.data.append('t', 'Temperature', r'$\mathrm{kK}$', dataArray[:,8]*1e-3)
        self.data.append('isTInterped', 'Is index i of T interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,7])

        # Activity Indices
        self.data.append('ae', 'AE-Index', r'$\mathrm{nT}$', dataArray[:,9])
        self.data.append('isAeInterped', 'Is index i of N interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,8])

        self.data.append('al', 'AL-Index', r'$\mathrm{nT}$', dataArray[:,10])
        self.data.append('isAlInterped', 'Is index i of N interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,9])

        self.data.append('au', 'AU-Index', r'$\mathrm{nT}$', dataArray[:,11])
        self.data.append('isAuInterped', 'Is index i of N interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,10])

        self.data.append('symh', 'SYM/H', r'$\mathrm{nT}$', dataArray[:,12])
        self.data.append('isSymHInterped', 'Is index i of N interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,11])
        
        # Bowshock Location
        self.data.append('xBS', 'BowShockX (gsm)', r'$\mathrm{RE}$', dataArray[:,13])
        self.data.append('isxBSInterped', 'Is index i of N interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,12])

        self.data.append('yBS', 'BowShockY (gsm)', r'$\mathrm{RE}$', dataArray[:,14])
        self.data.append('isyBSInterped', 'Is index i of N interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,13])

        self.data.append('zBS', 'BowShockZ (gsm)', r'$\mathrm{RE}$', dataArray[:,15])
        self.data.append('iszBSInterped', 'Is index i of N interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,14])

    def _deltaMinutes(self, t1, startDate):
        """
        Returns the number of minutes elapsed between t1 and startDate.

        Args:
            t1 (datetime.datetime): The end date and time.
            startDate (datetime.datetime): The start date and time.

        Returns:
            float: The number of minutes elapsed.
        """
        diff = t1 - startDate

        return (diff.days*24.0*60.0 + diff.seconds/60.0)

    def _gse2gsm(self, dates, dataArray):
        """
        Transform magnetic field B and velocity V from GSE to GSM
        coordinates. Store the results by overwriting the contents of the dataArray.

        Args:
            dates (list): A list of dates.
            dataArray (numpy.ndarray): 2D numpy array containing the solar wind data.

        Returns:
            None
        """
        for i,data in enumerate(dataArray):
            d = dates[i]

            # Update magnetic field
            b_gsm = kaipy.transform.GSEtoGSM(data[1], data[2], data[3], d)        
            data[1] = b_gsm[0]
            data[2] = b_gsm[1]
            data[3] = b_gsm[2]

            # Update Velocity
            v_gsm = kaipy.transform.GSEtoGSM(data[4], data[5], data[6], d)
            data[4] = v_gsm[0]
            data[5] = v_gsm[1]
            data[6] = v_gsm[2]

            # Update Bowshock Location
            bs_gsm = kaipy.transform.GSEtoGSM(data[13], data[14], data[15], d)
            data[13] = bs_gsm[0]
            data[14] = bs_gsm[1]
            data[15] = bs_gsm[2]

    def _readDst(self,startTime,endTime):
        """
        Reads the 'dst.dat' file and extracts the DST values within the specified time range.

        Args:
        startTime (datetime): The start time of the desired time range.
        endTime (datetime): The end time of the desired time range.

        Returns:
        tuple: A tuple containing two lists - 'dsttime' and 'dst'. 'dsttime' contains the datetime objects within the specified time range, and 'dst' contains the corresponding DST values.
        """
        dstfile = open("dst.dat",'r')
        text = dstfile.readlines()
        for i,j in enumerate(text):
            if j[0] == '2':
                iskip = i
                break
        dstfile.close()

        dat = numpy.genfromtxt("dst.dat",skip_header=iskip, autostrip=True,dtype=None,encoding='utf-8')
        dsttime = []
        dst = []
        fmt='%Y-%m-%dT%H:%M:%S.000'
        for i in dat:
            timestr = i[0]+"T"+i[1]
            currenttime = datetime.datetime.strptime(timestr,fmt)
            if currenttime >= startTime and currenttime <= endTime:
                dsttime.append(currenttime)
                dst.append(i[3])

        return (dsttime, dst)

    def _getDst(self,startTime,endTime):
        """
        Obtains DST values within the specified time range via pyspedas

        Args:
        startTime (datetime): The start time of the desired time range.
        endTime (datetime): The end time of the desired time range.

        Returns:
        tuple: A tuple containing two lists - 'dsttime' and 'dst'. 'dsttime' contains the datetime objects within the specified time range, and 'dst' contains the corresponding DST values.
        """
        # round start String down
        startStr = startTime.strftime('%Y-%m-%d')
        aware_start = startTime.replace(tzinfo=datetime.timezone.utc)
        # round end string up
        endStr   = (endTime + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        aware_end = endTime.replace(tzinfo=datetime.timezone.utc)

        dst_vars = kyoto.dst(trange=[startStr,endStr])
        dat = get_data('kyoto_dst')

        dsttime = []
        dst = []
        for i in range(len(dat[0])):
            currenttime = datetime.datetime.fromtimestamp(dat[0][i],datetime.timezone.utc)
            currenttime = currenttime.astimezone(datetime.timezone.utc)
            if currenttime >= aware_start and currenttime <= aware_end:
                dt = datetime.datetime(currenttime.year,currenttime.month,currenttime.day,currenttime.hour,currenttime.minute,currenttime.second)
                dsttime.append(dt)
                dst.append(dat[1][i])
        return (dsttime, dst)


class OMNIW(OMNI):
    """
    OMNIW Solar Wind file from CDAweb [http://cdaweb.gsfc.nasa.gov/].
    Data stored in GSE coordinates.

    Args:
        fWIND (str): Path to the WIND file.
        doFilter (bool): Flag indicating whether to apply filtering to the data.
        sigmaVal (float): Sigma value for the filtering.
        windowsize (int): Size of the window for data interpolation.

    Attributes:
        filter (bool): Flag indicating whether filtering is applied.
        sigma (float): Sigma value for the filtering.
        windowsize (int): Size of the window for data interpolation.
        bad_data (list): List of values considered as bad data.
        good_quality (list): List of values considered as good quality.
        data (TimeSeries): TimeSeries object to store the solar wind data.

    Methods:
        __init__(self, fWIND=None, doFilter=False, sigmaVal=3.0, windowsize=5): Constructor method.
        __read(self, fWIND): Read the solar wind file and store results in self.data TimeSeries object.
        __readWData(self, fWIND): Read the WIND data from file.
        __readOData(self, t0r, t1r): Read the OMNI data from CDAWeb.
        __appendMetaData(self, date, filename): Add standard metadata to the data dictionary.
        __combineData(self, Wdates, WdataArray, WhasBeenInterpolated, Odates, OdataArray, OhasBeenInterpolated): Combine the WIND and OMNI data.

    """

    def __init__(self, fWIND=None, doFilter=False, sigmaVal=3.0, windowsize=5):
        """
        Initialize the OMNI class.

        Args:
            fWIND (str): Path to the solar wind data file. Default is None.
            doFilter (bool): Flag indicating whether to apply filtering. Default is False.
            sigmaVal (float): Sigma value for filtering. Default is 3.0.
            windowsize (int): Size of the window for filtering. Should be odd and centered on index. Default is 5.
        """
        SolarWind.__init__(self)

        self.filter = doFilter
        self.sigma = sigmaVal
        self.windowsize = windowsize
        
        self.bad_data = [-999.900, 
                         99999.9, # V
                         9999.99, # B
                         999.990, # density
                         1.00000E+07, # Temperature
                         9999999.0, # Temperature
                         99999, # Activity indices 
                         9999000, # SWE del_time
                         -1e31 # SWE & MFI                      
                         ]
        self.good_quality = [4098,14338]
        
        print('Retrieving solar wind data from CDAWeb')
        self.__read(fWIND)

    def __read(self, fWIND):
        """
        Read the solar wind file and store the results in the self.data TimeSeries object.

        Args:
            fWIND (str): Path to the solar wind data file.

        Returns:
            tuple: A tuple containing the start date, end date, WIND dates, and WIND data.

        Raises:
            Exception: If the WIND and OMNI data have different variables or different times.
        """
        (startDate, endDate, Wdates, Wdata) = self.__readWData(fWIND)
        (Odates, Odata) = self.__readOData(startDate,endDate)
        (WdataArray, WhasBeenInterpolated) = self._removeBadData(Wdata)
        (OdataArray, OhasBeenInterpolated) = self._removeBadData(Odata)
        (dates,dataArray, hasBeenInterpolated, dataOrigin) = self.__combineData(Wdates,WdataArray,WhasBeenInterpolated,Odates,OdataArray,OhasBeenInterpolated)
        (dataArray, hasBeenInterpolated) = self._removeBadData(dataArray,hasBeenInterpolated)
        if self.filter:
            (dataArray, hasBeenInterpolated) = self._coarseFilter(dataArray, hasBeenInterpolated)
        self._storeDataDict(dates, dataArray, hasBeenInterpolated)
        self.__appendMetaData(startDate, fWIND)
        self._appendDerivedQuantities()

        
    def __readWData(self, fWIND):
        """
        Returns a 2D array (of strings) containing data from the file.
        
        TODO: Read the fmt and figure out which column is which. This would make things easier
        and more user-friendly. However, for now, we'll just assume the file is exactly 
        these quantities in this order.
        """
        
        cdas = CdasWs()

        print('Retrieving solar wind data from WIND file')
        filedata = numpy.genfromtxt(fWIND)
        ntimes = numpy.shape(filedata)[0]
        
        yrs = filedata[:,0]
        doy = filedata[:,1]
        hrs = filedata[:,2]
        mns = filedata[:,3]
        bx  = filedata[:,4] #nT
        by  = filedata[:,5] #nT
        bz  = filedata[:,6] #nT
        vx  = filedata[:,7] #km/s
        vy  = filedata[:,8] #km/s
        vz  = filedata[:,9] #km/s
        n   = filedata[:,10] #n/cc
        T   = filedata[:,11] #K
        xBow   = filedata[:,12] #RE
        yBow   = filedata[:,13] #RE
        zBow   = filedata[:,14] #RE
        
        t0 = datetime.datetime(int(yrs[0]),1,1,hour=int(hrs[0]),minute=int(mns[0])) + datetime.timedelta(int(doy[0])-1)
        t1 = datetime.datetime(int(yrs[-1]),1,1,hour=int(hrs[-1]),minute=int(mns[-1])) + datetime.timedelta(int(doy[-1])-1)
        t0r = t0.strftime("%Y-%m-%dT%H:%M:%SZ")
        t1r = t1.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        status,fOMNI = cdas.get_data(
           'OMNI_HRO_1MIN',
            ['BX_GSE','BY_GSE','BZ_GSE',
            'Vx','Vy','Vz',
            'proton_density','T',
            'AE_INDEX','AL_INDEX','AU_INDEX','SYM_H',
            'BSN_x','BSN_y','BSN_z'],
            t0r,t1r)
            
        tOMNI= fOMNI['Epoch']       #datetime
        ae   = fOMNI['AE_INDEX']    #nT
        al   = fOMNI['AL_INDEX']    #nT
        au   = fOMNI['AU_INDEX']    #nT
        symh = fOMNI['SYM_H']       #nT
        ovx  = fOMNI['Vx']          #kms
        ovy  = fOMNI['Vy']          #kms
        ovz  = fOMNI['Vz']          #kms
        oxBow = fOMNI['BSN_x']      #RE
        oyBow = fOMNI['BSN_y']      #RE
        ozBow = fOMNI['BSN_z']      #RE
        
        #oxBow = oxBow*1e5/Re_cgs #km -> cm -> Re
        #oyBow = oyBow*1e5/Re_cgs #km -> cm -> Re
        #ozBow = ozBow*1e5/Re_cgs #km -> cm -> Re

        dates = []
        rows  = []
        
        for i in range(ntimes):
            currentTime = datetime.datetime(int(yrs[i]),1,1,hour=int(hrs[i]),minute=int(mns[i])) + datetime.timedelta(int(doy[i])-1)
            #calculating minutes from the start time
            nMin = self._deltaMinutes(currentTime,t0)

            if (xBow[i] in self.bad_data) or (yBow[i] in self.bad_data) or (zBow[i] in self.bad_data):
                data = [nMin,bx[i],by[i],bz[i],vx[i],vy[i],vz[i],n[i],T[i],ae[i],al[i],au[i],symh[i],oxBow[i],oyBow[i],ozBow[i]]
            else:
                data = [nMin,bx[i],by[i],bz[i],vx[i],vy[i],vz[i],n[i],T[i],ae[i],al[i],au[i],symh[i],xBow[i],yBow[i],zBow[i]]

            dates.append( currentTime )
            rows.append( data )

        return (t0r, t1r, dates, rows)

    def __readOData(self, t0r,t1r):
        """
        Returns a 2D array (of strings) containing data from the file.
        
        Args:
            t0r (str): Start date in the format "%Y-%m-%dT%H:%M:%SZ".
            t1r (str): End date in the format "%Y-%m-%dT%H:%M:%SZ".
        
        Returns:
            tuple: A tuple containing the dates and rows of data.
        """
        
        cdas = CdasWs()
        
        #obtain 1 minute resolution observations from OMNI dataset
        print('Retrieving solar wind data from CDAWeb')
        status,fh = cdas.get_data(
           'OMNI_HRO_1MIN',
            ['BX_GSE','BY_GSE','BZ_GSE',
            'Vx','Vy','Vz',
            'proton_density','T',
            'AE_INDEX','AL_INDEX','AU_INDEX','SYM_H',
            'BSN_x','BSN_y','BSN_z'],
            t0r,t1r)
        
        #pulling variables from cdaweb        
        time=fh['Epoch']
        bx=fh['BX_GSE']
        by=fh['BY_GSE']
        bz=fh['BZ_GSE']
        vx=fh['Vx']
        vy=fh['Vy']
        vz=fh['Vz']
        n=fh['proton_density']
        T=fh['T']
        ae=fh['AE_INDEX']
        al=fh['AL_INDEX']
        au=fh['AU_INDEX']
        symh=fh['SYM_H']
        xBow = fh['BSN_x']       #RE
        yBow = fh['BSN_y']       #RE
        zBow = fh['BSN_z']       #RE

        #xBow = xBow*1e5/Re_cgs #km -> cm -> Re
        #yBow = yBow*1e5/Re_cgs #km -> cm -> Re
        #zBow = zBow*1e5/Re_cgs #km -> cm -> Re

        dates = []
        rows = []
        startTime = time[0]
        self.startTime = startTime
        for i in range(len(time)):
          
            #calculating minutes from the start time
            nMin = self._deltaMinutes(time[i],startTime)

            data = [nMin,bx[i],by[i],bz[i],vx[i],vy[i],vz[i],n[i],T[i],ae[i],al[i],au[i],symh[i],xBow[i],yBow[i],zBow[i]]

            dates.append( time[i] )
            rows.append( data )
        

        return (dates, rows)
        
    def __appendMetaData(self, date, filename):
        """
        Add standard metadata to the data dictionary.

        Args:
            date (str): Date of data processing.
            filename (str): Name of the source file.

        Returns:
            None

        """
        metadata = {'Model': 'OMNIW',
                    'Source': filename,
                    'Date processed': datetime.datetime.now(),
                    'Start date': self.startTime,
                    }
        
        self.data.append(key='meta',
                         name='Metadata for WIND Solar Wind file',
                         units='n/a',
                         data=metadata)
       
    def __combineData(self,Wdates,WdataArray,WhasBeenInterpolated,Odates,OdataArray,OhasBeenInterpolated):
        """
        This method combines the W data with the O data.
        Starting with the OMNI data, if there is a window of consecutive missing data
        of size windowsize, then use the W data to fill in the missing values (if the W data is good).
        
        Returns:
            tuple: A tuple containing the combined dates, dataArray, hasBeenInterpolated, and dataOrigin.
        """
        
        nvarW = len(WdataArray[0])
        nvarO = len(OdataArray[0])
        ntimesW = len(WdataArray[:,0])
        ntimesO = len(OdataArray[:,0])
        windowsize = self.windowsize
        
        if (nvarW != nvarO): raise Exception("Error: W and O have different vars")
        if (ntimesW != ntimesO): raise Exception("Error: W and O have different times")
        dates = Wdates
        hasBeenInterpolated = OhasBeenInterpolated
        dataArray = OdataArray
        dataOrigin = numpy.empty(numpy.shape(hasBeenInterpolated)) #0 is interpolated. 1 is OMNI, 2 is WIND
        
        halfwindow = int((windowsize-1)/2)
        for varIdx in range(1,nvarO):
            for curIndex,row in enumerate(WdataArray):
                if OhasBeenInterpolated[curIndex,varIdx-1]:
                    # Replace only if the whole window is missing data
                    if (OhasBeenInterpolated[curIndex-halfwindow:curIndex+halfwindow,varIdx-1].all() or OhasBeenInterpolated[curIndex:min(curIndex+windowsize,ntimesO),varIdx-1].all() or OhasBeenInterpolated[max(0,curIndex-windowsize+1):curIndex+1,varIdx-1].all()):
                        #Check if W is interpolated
                        if not WhasBeenInterpolated[curIndex,varIdx-1]:
                            dataArray[curIndex,varIdx] = row[varIdx]
                            dataArray[curIndex,-3:] = row[-3:]
                            # Use W if W was not interpolated
                            dataOrigin[curIndex,varIdx-1] = 2
                            dataOrigin[curIndex,-3:] = 2
                            hasBeenInterpolated[curIndex,varIdx-1] = False
                            hasBeenInterpolated[curIndex,-3:] = False
                        else:
                            # use the already interpolated value if W was bad
                            dataOrigin[curIndex,varIdx-1] = 0
                            # let's set it to a bad value so it reinterpolates
                            dataArray[curIndex,varIdx] = -1e31
                    else:
                        # use original OMNI if no interp
                        dataOrigin[curIndex,varIdx-1] = 0
                        # let's set it to a bad value so it reinterpolates
                        dataArray[curIndex,varIdx] = -1e31
                else:
                    # use original OMNI if no interp
                    dataOrigin[curIndex,varIdx-1] = 1
        
        return (dates,dataArray, hasBeenInterpolated, dataOrigin)
         
def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()