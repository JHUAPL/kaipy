"""
This module provides coordinate transformations relevant to geospace modeling.
Many of these transformations are wrappers to external C or Fortran libraries, 
using the NumPy vectorize() method to handle ndarray inputs when appropriate.

Note:
    This module is stripped down to only stuff that is currently used in kaipy.
    The few remaining routines only wrap spacepy.
    
    #K: Ripped out everything

Functions:
    x, y, z = SMtoGSM(x, y, z, dateTime)
        Convert from solar magnetic to geocentric solar magnetospheric coordinates.
        
    x, y, z = GSMtoSM(x, y, z, dateTime)
        Convert from geocentric solar magnetospheric to solar magnetic coordinates.
        
    x, y, z = GSEtoGSM(x, y, z, dateTime)
        Convert from geocentric solar ecliptic to geocentric magnetospheric coordinates.
"""
# Standard modules
import datetime

# Third-party modules
import numpy as np
from spacepy.coordinates import Coords
from spacepy.time import Ticktock

def SMtoGSM(x, y, z, ut):
    """
    Convert coordinates from Solar Magnetic (SM) system to Geocentric Solar Magnetospheric (GSM) system.

    Parameters:
        x (float): The x-coordinate in SM system.
        y (float): The y-coordinate in SM system.
        z (float): The z-coordinate in SM system.
        ut (datetime.datetime): The Universal Time (UT) for the conversion.

    Returns:
        tuple: A tuple containing the x, y, and z coordinates in GSM system.

    Examples:
        >>> SMtoGSM(1, 2, 3, datetime.datetime(2009, 1, 27, 0, 0, 0))
        (-0.126..., 2.0, 3.159...)
    """
    # Adapting code from scutils:convertGameraVec
    fromSys = 'SM'
    fromType = 'car'
    toSys = 'GSM'
    toType = 'car'

    invec = Coords(np.column_stack((x, y, z)), fromSys, fromType, use_irbem=False)
    invec.ticks = Ticktock(ut)
    outvec = invec.convert(toSys, toType)

    if len(outvec.x > 1):
        return outvec.x, outvec.y, outvec.z
    else:
        return outvec.x[0], outvec.y[0], outvec.z[0]


def GSMtoSM(x, y, z, ut):
    """
    Convert coordinates from GSM (Geocentric Solar Magnetospheric) system to SM (Solar Magnetic) system.

    Parameters:
        x (float): The x-coordinate in GSM system.
        y (float): The y-coordinate in GSM system.
        z (float): The z-coordinate in GSM system.
        ut (datetime.datetime): The universal time.

    Returns:
        tuple: A tuple containing the converted x, y, and z coordinates in SM system.

    Examples:
        >>> GSMtoSM(1, 2, 3, datetime.datetime(2009, 1, 27, 0, 0, 0))
        (1.997..., 2.0, 2.451...)
    """
    # Adapting code from scutils:convertGameraVec
    fromSys = 'GSM'
    fromType = 'car'
    toSys = 'SM'
    toType = 'car'

    invec = Coords(np.column_stack((x, y, z)), fromSys, fromType, use_irbem=False)
    invec.ticks = Ticktock(ut)
    outvec = invec.convert(toSys, toType)

    if len(outvec.x > 1):
        return outvec.x, outvec.y, outvec.z
    else:
        return outvec.x[0], outvec.y[0], outvec.z[0]



def GSEtoGSM(x, y, z, ut):
    """
    Convert coordinates from GSE (Geocentric Solar Ecliptic) to GSM (Geocentric Solar Magnetospheric) system.

    Args:
        x (float): X-coordinate in GSE system.
        y (float): Y-coordinate in GSE system.
        z (float): Z-coordinate in GSE system.
        ut (datetime.datetime): Universal Time (UT) for the conversion.

    Returns:
        tuple: A tuple containing the converted X, Y, and Z coordinates in GSM system.

    Examples:
        >>> GSEtoGSM(1, 2, 3, datetime.datetime(2009, 1, 27, 0, 0, 0))
        (0.9999999999999998, 0.5403023058681398, 3.564718122410546)

    Note:
        This function adapts code from scutils:convertGameraVec.
    """
    # Adapting code from scutils:convertGameraVec
    fromSys = 'GSE'
    fromType = 'car'
    toSys = 'GSM'
    toType = 'car'

    invec = Coords(np.column_stack((x, y, z)), fromSys, fromType, use_irbem=False)
    invec.ticks = Ticktock(ut)
    outvec = invec.convert(toSys, toType)

    if len(outvec.x) > 1:
        return outvec.x, outvec.y, outvec.z
    else:
        return outvec.x[0], outvec.y[0], outvec.z[0]



