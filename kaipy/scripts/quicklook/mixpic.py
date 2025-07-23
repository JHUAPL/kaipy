#!/usr/bin/env python


"""Create north and south REMIX plots.

Create north and south REMIX plots.

Author
------
Kareem Sorathia (kareem.sorathia@jhuapl.edu)
Eric Winter (eric.winter@jhuapl.edu)



Example PBS script to use multiprocessing to make a video on Lou Data Analysis Nodes:

#!/bin/bash
#PBS -N MixVid
#PBS -j oe
#PBS -lselect=1:ncpus=36:mem=750GB
#PBS -lwalltime=2:00:00
#PBS -q ldan
#PBS -W group_list=s2521
export RUNID=“geospace”
# load modules and set python environment
source ~/.bashrc
setPy
cd $PBS_O_WORKDIR
mixpic.py -id $RUNID --vid -ncpus 36

"""


# Import standard modules.
import argparse
import sys
import os

# Import supplemental modules.
from astropy.time import Time
# import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
from alive_progress import alive_it
from multiprocessing import Pool
from psutil import cpu_count

# Import project-specific modules.
import kaipy.cdaweb_utils as cdaweb_utils
import kaipy.kaiH5 as kaiH5
import kaipy.kaiTools as ktools
import kaipy.remix.remix as remix
import kaipy.kaiViz as kv
import kaipy.kdefs as kd


# Program constants and defaults

# Program description.
description = """Creates simple multi-panel REMIX figure for a GAMERA magnetosphere run.
Top Row - FAC (with potential contours overplotted), Pedersen and Hall Conductances
Bottom Row - Joule heating rate, particle energy and energy flux
"""

# Default identifier for results to read.
default_runid = "msphere"

# Plot the last step by default.
default_step = -1

# Color to use for magnetic footprint positions.
FOOTPRINT_COLOR = 'red'

# Coordinate system for plotting
default_coord = 'SM'


def create_command_line_parser():
    """
    Create the command-line argument parser.

    Returns:
        parser (argparse.ArgumentParser): Command-line argument parser for this script.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "-id", type=str, metavar="runid", default=default_runid,
        help="Run ID of data (default: %(default)s)"
    )
    parser.add_argument(
        "-n", type=int, metavar="step", default=default_step,
        help="Time slice to plot (default: %(default)s)"
    )
    parser.add_argument(
        '-nflux', action='store_true', default=False,
        help="Show number flux instead of energy flux (default: %(default)s)"
    )
    parser.add_argument(
        '-print', action='store_true', default=False,
        help="Print list of all steps and time labels, then exit (default: %(default)s)"
    )
    parser.add_argument(
        "--spacecraft", type=str, metavar="spacecraft", default=None,
        help="Names of spacecraft to plot magnetic footprints, separated by commas (default: %(default)s)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        '-GTYPE', action='store_true', default=False,
        help="Show RCM grid type in the eflx plot (default: %(default)s)"
    )
    parser.add_argument(
        '--nPP', type=float, metavar="nPP", default=0,
        help="Plasmasphere density contour to show (default: %(default)/cc not shown)"
    )
    parser.add_argument(
        '-MNDF', action='store_true', default=False,
        help="Show mono-diffuse eflx/nflx in linear/log green/blue colormap (default: %(default)s)"
    )
    parser.add_argument(
        '-vid', action='store_true', default=False,
        help="Make a video and store in mixVid directory (default: %(default)s)"
    )
    parser.add_argument(
        '-overwrite', action='store_true', default=False,
        help="Overwrite existing vid files (default: %(default)s)"
    )
    parser.add_argument(
        '--ncpus', type=int, metavar="ncpus", default=1,
        help="Number of threads to use with --vid (default: %(default)s)"
    )
    parser.add_argument(
        '-nohash', action='store_true', default=False,
        help="Don't display branch/hash info (default: %(default)s)"
    )
    parser.add_argument(
        '--coord', type=str, metavar="coord", default=default_coord,
        help="Coordinate system to use (default: %(default)s)"
    )
    return parser


def makePlot(i, remixFile, nStp, args, varDict):
    debug = args.debug
    runid = args.id
    do_nflux = args.nflux
    do_print = args.print
    spacecraft = args.spacecraft
    verbose = args.verbose
    do_GTYPE = args.GTYPE
    nPP = args.nPP
    do_MNDF = args.MNDF
    do_vid = args.vid
    do_overwrite = args.overwrite
    do_hash = not args.nohash
    ncpus = args.ncpus

    # Extract missing variables from varDict
    branch = varDict.get('branch', 'unknown')
    githash = varDict.get('githash', 'unknown')
    n_pad = varDict.get('n_pad', 6)
    outDir = varDict.get('outDir', 'mixVid')
    
    # Init figure
    fig = plt.figure(figsize=(12, 7.5))

    with h5.File(remixFile, 'r') as f5:
        foundT = f5['Step#'+str(nStp)].attrs['MJD']
    #foundT = T[idxStp]
    if debug:
        print("foundT = %s" % foundT)
    print('Found time:', Time(foundT, format='mjd').iso)
    utS = ktools.MJD2UT(foundT)
    if debug:
        print("utS = %s" % utS)

    # If both N and S files exist, skip it
    if do_vid:
        filenameN = "{}.{:0>{n}d}.png".format("remix_n", i, n=n_pad)
        outPathN = os.path.join(outDir, filenameN)
        filenameS = "{}.{:0>{n}d}.png".format("remix_s", i, n=n_pad)
        outPathS = os.path.join(outDir, filenameN)
        if not do_overwrite and os.path.exists(outPathN) and os.path.exists(outPathS):
            return


    # Read the data into the remix object.
    ion = remix.remix(remixFile, nStp)
    if debug:
        print("ion = %s" % ion)

    for h in ['NORTH','SOUTH']:
        if h == 'NORTH':
            hemi = 'n'
        if h == 'SOUTH':
            hemi = 's'

        if do_vid:
            filename = "{}.{:0>{n}d}.png".format("remix_"+hemi, i, n=n_pad)
            outPath = os.path.join(outDir, filename)
        else:
            outPath = "remix_"+hemi+".png"

        # Skip this file if it already exists and we're not supposed to overwrite
        if not do_overwrite and os.path.exists(outPath) and do_vid:
            continue

        plt.clf()

        # Add a label.
        plt.figtext(
            0.5, 0.94, 'MIX (' + h + ')\n' + Time(foundT, format='mjd').iso,
            fontsize=12, multialignment='center', horizontalalignment='center'
        )

        # Initialize the remix object based on the current hemisphere.
        ion.init_vars(h)

        # Create the plot layout for the current hemisphere.
        gs = gridspec.GridSpec(
            2, 3, figure=fig, left=0.03, right=0.97, top=0.9, bottom=0.03
        )

        # Create the individual plots for the current hemisphere.
        axs = [None]*6
        axs[0] = ion.plot('current', gs=gs[0, 0])
        axs[1] = ion.plot('sigmap', gs=gs[0, 1])
        axs[2] = ion.plot('sigmah', gs=gs[0, 2])
        if do_nflux:
            axs[3] = ion.plot('flux', gs=gs[1, 0],doGTYPE=do_GTYPE,nPP=nPP,doMNDF=do_MNDF)
        else:
            axs[3] = ion.plot('joule', gs=gs[1, 0])
        axs[4] = ion.plot('energy', gs=gs[1, 1],doGTYPE=do_GTYPE,nPP=nPP,doMNDF=do_MNDF)
        axs[5] = ion.plot('eflux', gs=gs[1, 2],doGTYPE=do_GTYPE,nPP=nPP,doMNDF=do_MNDF)

        # If requested, plot the magnetic footprints for the specified
        # spacecraft.
        if spacecraft:
            for sc in spacecraft:
                if verbose:
                    print("Overplotting %s magnetic footprint for %s." % (h, sc))

                # Fetch the footprint position for this hemisphere.
                if h.lower() == 'north':
                    fp_lat, fp_lon = cdaweb_utils.fetch_satellite_magnetic_northern_footprint_position(
                        sc, utS
                    )
                else:
                    fp_lat, fp_lon = cdaweb_utils.fetch_satellite_magnetic_southern_footprint_position(
                        sc, utS
                    )
                if debug:
                    print("fp_lat, fp_lon = %s, %s" % (fp_lat, fp_lon))

                # Skip if no footprint found.
                if fp_lat is None:
                    print("No %s footprint found for spacecraft %s." % (h, sc))
                    continue

                # The footprint locations are in geographic (GEO) coordinates.
                # They must be converted to Solar Magnetic (SM) coordinates
                # for plotting.

                # Convert the footprint position to the coordinate system used
                # by these plots, which show contours at the surface of the
                # ionosphere, about 122 km above the surface ofn the Earth.
                # Note that this adjustment assumes the field lines impinging
                # on the magnetic footprint descend vertically at the
                # footprint point, which is not technically accurate.
                fp_lat_rad = np.radians(fp_lat)
                fp_lon_rad = np.radians(fp_lon)
                fp_x = np.cos(fp_lat_rad)*np.cos(fp_lon_rad)
                fp_y = np.cos(fp_lat_rad)*np.sin(fp_lon_rad)
                fp_theta = np.arctan2(fp_y, fp_x)  # [-pi, pi]
                fp_r = np.sqrt(fp_x**2 + fp_y**2)

                # Plot a labelled dot at the location of each footprint.
                # Skip if no footprint position found.
                for ax in axs:
                    ax.plot(fp_theta, fp_r, 'o', c=FOOTPRINT_COLOR)
                    theta_nudge = 0.0
                    r_nudge = 0.0
                    ax.text(fp_theta + theta_nudge, fp_r + r_nudge, sc)

        # Add Branch and Hash info
        if do_hash:
            fig.text(0.1,0.95,f"branch/commit: {branch}/{githash}", fontsize=6)

        # Save to file
        kv.savePic(outPath, dpiQ=300)

        #plt.close(fig)

def main():
    """Plot remix data, either a single time step or as a movie"""

    # Set up the command-line parser.
    parser = create_command_line_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    debug = args.debug
    runid = args.id
    nStp = args.n
    do_nflux = args.nflux
    do_print = args.print
    spacecraft = args.spacecraft
    verbose = args.verbose
    do_GTYPE = args.GTYPE
    nPP = args.nPP
    do_MNDF = args.MNDF
    do_vid = args.vid
    do_overwrite = args.overwrite
    do_hash = not args.nohash
    ncpus = args.ncpus

    if debug:
        print("args = %s" % args)

    # Construct the name of the REMIX results file.
    remixFile = runid + '.mix.h5'
    if debug:
        print("remixFile = %s" % remixFile)

    # Get branch/hash info
    if do_hash:
        branch = kaiH5.GetBranch(remixFile)
        githash = kaiH5.GetHash(remixFile)
        if debug:
            print(f'branch/commit: {branch}/{githash}')

    # Split the original string into a list of spacecraft IDs.
    if spacecraft:
        spacecraft = spacecraft.split(',')
        if debug:
            print("spacecraft = %s" % spacecraft)

    # Enumerate the steps in the results file.
    nsteps, sIds = kaiH5.cntSteps(remixFile)
    sIds = sorted(sIds)
    if debug:
        print("nsteps = %s" % nsteps)
        print("sIds = %s" % sIds)

    # Check that the requested step exists.
    if nStp >= 0 and not nStp in sIds:  # ANY nStp<0 gets last step.
        raise TypeError(f"Step #{nStp} not found in {remixFile}!")

    # Get the times from the result file.
    if do_print:
        T = kaiH5.getTs(remixFile, sIds, aID='MJD')
        if debug:
            print("T = %s" % T)
        for i, tt in enumerate(T):
            print('Step#%06d: ' % sorted(sIds)[i], Time(tt, format='mjd').iso)
        sys.exit(0)

    
    # Create the plots in a memory buffer.
    mpl.use('Agg')

    # Set global plot font options.
    mpl.rc('mathtext', fontset='stixsans', default='regular')
    mpl.rc('font', size=10)


    if not do_vid:  # Then we are making a single image, keep original functionality
        # Find the time for the specified step.
        if nStp == -1:
            # Take the last step.
            nStp = sorted(sIds)[-1]
        if debug:
            print("nStp = %s" % nStp)
        varDict = {
            'branch': branch,
            'githash': githash,
            'n_pad': 0,
            'outDir': 'remix'
        }
        makePlot(nStp, remixFile, nStp, args, varDict)
    
    else:  # Then we make a video, i.e. series of images saved to mixVid
        outDir = 'mixVid'
        kaiH5.CheckDirOrMake(outDir)

        # How many 0's do we need for filenames?
        n_pad = int(np.log10((len(sIds)))) + 1

        if ncpus == 1:
            for i, nStp in enumerate(alive_it(sIds,length=kd.barLen,bar=kd.barDef)):
                varDict = {
                    'branch': branch,
                    'githash': githash,
                    'n_pad': n_pad,
                    'outDir': outDir
                }
                makePlot(i,remixFile, nStp, args, varDict)
        else:
            varDict = {
                'branch': branch,
                'githash': githash,
                'n_pad': n_pad,
                'outDir': outDir
            }
            # Make list of parallel arguments
            ag = ((i,remixFile,nStp, args, varDict) for i, nStp in enumerate(sIds) )
            # Check we're not exceeding cpu_count on computer
            ncpus = min(int(ncpus),cpu_count(logical=False))
            print('Doing multithreading on ',ncpus,' threads')
            # Do parallel job
            with Pool(processes=ncpus) as pl:
                pl.starmap(makePlot,ag)
            print("Done making all the images. Go to mixVid folder")

if __name__ == "__main__":
    main()