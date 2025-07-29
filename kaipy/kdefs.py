"""
This module contains various constants used in the code.

Helpful conversions:
    G2nT: Conversion factor from Gauss to nanoTesla.
    kev2J: Conversion factor from keV to Joules.
    ev2J: Conversion factor from eV to Joules.
    erg2J: Conversion factor from erg to Joules.

Physical Constants:
    Mu0: Magnetic constant in Tesla meter per Ampere (Tm/A).
    Me_cgs: Electron mass in grams (g).
    Mp_cgs: Proton mass in grams (g).
    eCharge: Charge of an electron in Joules (J).
    dalton: Mass unit in kilograms (kg).

Planetary Constants:
    Re_cgs: Earth's radius in centimeters (cm).
    EarthM0g: Earth's magnetic field strength in Gauss.
    REarth: Earth's radius in meters (m).
    RionE: Earth's ionosphere radius in 1000 km.
    EarthPsi0: Corotation potential of Earth in kiloVolts (kV).
    SaturnM0g: Saturn's magnetic field strength in Gauss.
    RSaturnXE: Saturn's radius in terms of Earth's radius.
    JupiterM0g: Jupiter's magnetic field strength in Gauss.
    RJupiterXE: Jupiter's radius in terms of Earth's radius.
    MercuryM0g: Mercury's magnetic field strength in Gauss.
    RMercuryXE: Mercury's radius in terms of Earth's radius.
    NeptuneM0g: Neptune's magnetic field strength in Gauss.
    RNeptuneXE: Neptune's radius in terms of Earth's radius.

Helio:
    Rsolar: Solar radius in centimeters (cm).
    kbltz: Boltzmann constant in erg per Kelvin (erg/K).
    mp: Proton mass in grams (g).
    Tsolar: Siderial solar rotation period in days.
    Tsolar_synodic: Synodic solar rotation period in days.
    JD2MJD: Conversion factor from Julian Date (JD) to Modified Julian Date (MJD).
    Day2s: Conversion factor from days to seconds.
    vc_cgs: Speed of light in centimeters per second (cm/s).

Output defaults:
    barLen: Length of the progress bar.
    barLab: Length of the progress bar label.
    barDef: Default progress bar animation.
    grpTimeCache: Time attribute cache name for I/O.
"""
# Third-party modules
import numpy as np
import alive_progress.animations.bars

#------
#Helpful conversions
#------
G2nT  = 1E5              # Gauss->nanoTesla
kev2J = 1.602176634E-16  # keV -> J
ev2J  = 1.602176634E-19  # eV  -> J
erg2J = 1e-7             # erg -> J



#------
#Physical Constants
#------
Mu0     = 4E-7*np.pi             # [Tm/A]
Me_cgs  = 9.1093837015E-28       # [g]  Electron mass
Mp_cgs  = 1.67262192369E-24      # [g]  Proton mass
eCharge = 1.602E-19              # [J]  Charge of electron
dalton  = 1.66053906660*1.0E-27  # [kg] Mass unit



#------
#Planetary Constants
#------
Re_cgs = 6.3781E8       # [cm]  Earth's radius
EarthM0g = 0.2961737    # Gauss, Olsen++ 2000
REarth = Re_cgs*1.0e-2  # m
RionE  = 6.5            # Earth Ionosphere radius in 1000 km
EarthPsi0 = 92.4        # Corotation potential [kV]
#Saturn
SaturnM0g = 0.21        # Gauss
RSaturnXE = 9.5         # Rx = X*Re
#Jupiter
JupiterM0g = 4.8        # Gauss
RJupiterXE = 11.209       # !Rx = X*Re
#Mercury
MercuryM0g = 0.00345    # Gauss
RMercuryXE = 0.31397    # Rx = X*Re
#Neptune
NeptuneM0g = 0.142      # Gauss
RNeptuneXE = 3.860      # Rx = X*Re



#------
#Helio
#------
Rsolar = 6.957e10      # [cm] Solar radius
kbltz  = 1.38e-16      # [erg/K] Boltzmann constant
mp     = 1.67e-24      # [g] Proton mass
Tsolar = 25.38         # [days] Siderial solar rotation period
Tsolar_synodic = 27.28 # [days] Synodic solar rotation period
JD2MJD = 2400000.5     # Conversion from JD to MJD: MJD = JD 2400000.5
Day2s = 86400.         # [s] Conversion days => seconds
vc_cgs = 2.99792458e10 # [cm/s] speed of light

#------
#Output defaults
#------
barLen = 30
barLab = 30 
#barDef = 'fish'
barDef  = alive_progress.animations.bars.bar_factory(tip="><('>", chars="∙",background="")
barDef2 = alive_progress.animations.bars.bar_factory('∙○⦿●',tip=" ><>")

#------
# I/O
#------
grpTimeCache = "timeAttributeCache"
