import pytest
import numpy as np
import h5py
import datetime
from astropy.time import Time
from kaipy.gamera.magsphere import GamsphPipe

Ni = 32
Nj = 24
Nk = 48

Nlat = 44
Nlon = 360

def write_mix_h5(file_path,mjds=None):
    lat = np.arange(Nlat+1)
    long = np.arange(Nlon+1)
    X = np.outer(np.sin(np.radians(lat)), np.cos(np.radians(long)))
    Y = np.outer(np.sin(np.radians(lat)), np.sin(np.radians(long)))
    if mjds is None:
        mjds = []
        suppliedMJDs = False
    else:
        suppliedMJDs = True
    with h5py.File(file_path, 'w') as f:
        f.create_dataset("X", data=X)
        f.create_dataset("Y", data=Y)
        for i in range(3):
            grp = f.create_group("Step#{}".format(i))
            grp.attrs['time'] = np.double(i)
            if suppliedMJDs:
                mjd = mjds[i]
            else:
                mjd = Time(datetime.datetime.now()).mjd
                mjds.append(mjd)
            grp.attrs['MJD'] = mjd
            grp.attrs['timestep'] = i
            grp.create_dataset("Potential NORTH", data=np.zeros((Nlat, Nlon)))
            grp.create_dataset("Potential SOUTH", data=np.zeros((Nlat, Nlon)))
            grp.create_dataset("Field-aligned current NORTH", data=np.zeros((Nlat, Nlon)))
            grp.create_dataset("Field-aligned current SOUTH", data=np.zeros((Nlat, Nlon)))
            grp.create_dataset("Pedersen conductance NORTH", data=np.zeros((Nlat, Nlon)))
            grp.create_dataset("Pedersen conductance SOUTH", data=np.zeros((Nlat, Nlon)))
            grp.create_dataset("Hall conductance NORTH", data=np.zeros((Nlat, Nlon)))
            grp.create_dataset("Hall conductance SOUTH", data=np.zeros((Nlat, Nlon)))
            grp.create_dataset("Average energy NORTH", data=np.zeros((Nlat, Nlon)))
            grp.create_dataset("Average energy SOUTH", data=np.zeros((Nlat, Nlon)))
            grp.create_dataset("Number flux NORTH", data=np.zeros((Nlat, Nlon)))
            grp.create_dataset("Number flux SOUTH", data=np.zeros((Nlat, Nlon)))
            grp.create_dataset("RCM beta NORTH", data=np.zeros((Nlat, Nlon)))
            grp.create_dataset("RCM beta SOUTH", data=np.zeros((Nlat, Nlon)))
            grp.create_dataset("Mono potential drop NORTH", data=np.zeros((Nlat, Nlon)))
            grp.create_dataset("Mono potential drop SOUTH", data=np.zeros((Nlat, Nlon)))
            grp.create_dataset("Auroral model type NORTH", data=np.zeros((Nlat, Nlon)))
            grp.create_dataset("Auroral model type SOUTH", data=np.zeros((Nlat, Nlon)))

    return file_path

@pytest.fixture
def gamera_pipe(tmpdir):
    # Create a temporary directory and file structure for testing
    fdir = tmpdir.mkdir("data")
    ftag = "test"
    file_path = fdir.join("test.gam.h5")
    print(f'file_path: {file_path}')
    mjds = []
    with h5py.File(file_path, 'w') as f:
        f.create_dataset("X", data=np.linspace(-100, 100, Ni+1).reshape(Ni+1, 1, 1).repeat(Nj+1, axis=1).repeat(Nk+1, axis=2).T)
        f.create_dataset("Y", data=np.linspace(-100, 100, Nj+1).reshape(1, Nj+1, 1).repeat(Ni+1, axis=0).repeat(Nk+1, axis=2).T)
        f.create_dataset("Z", data=np.linspace(-100, 100, Nk+1).reshape(1, 1, Nk+1).repeat(Ni+1, axis=0).repeat(Nj+1, axis=1).T)
        f.create_dataset("dV", data=np.zeros((Ni, Nj, Nk)).T)
        for i in range(3):
            grp = f.create_group("Step#{}".format(i))
            grp.attrs['time'] = np.double(i)
            mjd = Time(datetime.datetime.now()).mjd
            grp.attrs['MJD'] = mjd
            mjds.append(mjd)
            grp.attrs['timestep'] = i
            grp.create_dataset("D", data=np.zeros((Ni, Nj, Nk)).T)
            grp.create_dataset("P", data=np.zeros((Ni, Nj, Nk)).T)
            grp.create_dataset("Bx", data=np.zeros((Ni, Nj, Nk)).T)
            grp.create_dataset("By", data=np.zeros((Ni, Nj, Nk)).T)
            grp.create_dataset("Bz", data=np.zeros((Ni, Nj, Nk)).T)
            grp.create_dataset("Vx", data=np.zeros((Ni, Nj, Nk)).T)
            grp.create_dataset("Vy", data=np.zeros((Ni, Nj, Nk)).T)
            grp.create_dataset("Vz", data=np.zeros((Ni, Nj, Nk)).T)
            grp.create_dataset("Jx", data=np.zeros((Ni, Nj, Nk)).T)
            grp.create_dataset("Jy", data=np.zeros((Ni, Nj, Nk)).T)
            grp.create_dataset("Jz", data=np.zeros((Ni, Nj, Nk)).T)
    file_path = fdir.join("test.mix.h5")
    file_path = write_mix_h5(file_path)

    return GamsphPipe(str(fdir), ftag, doFast=True, uID="Earth")

@pytest.fixture
def mix_file(tmpdir):
    fdir = tmpdir.mkdir("data")
    ftag = "test"
    file_path = fdir.join("test.mix.h5")
    file_path = write_mix_h5(file_path)
    return file_path

def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run tests marked as slow"
    )

def pytest_runtest_setup(item):
    if "slow" in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("need --runslow option to run")