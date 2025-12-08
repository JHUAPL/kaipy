import pytest
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
from argparse import RawTextHelpFormatter

import kaipy.remix.remix as remix
from kaipy.gamera.msphViz import AddSizeArgs, GetSizeBds, PlotErrAbs, PlotErrRel, PlotLogicalErrAbs, PlotLogicalErrRel, CalcTotalErrAbs, CalcTotalErrRel, PlotEqB, PlotMerid, PlotJyXZ, PlotEqEphi, PlotMPI, AddIonBoxes, plotPlane, plotXY, plotXZ


@pytest.fixture
def parser():
    return argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

def test_AddSizeArgs(parser):
    AddSizeArgs(parser)
    args = parser.parse_args(['-size', 'big'])
    assert args.size == 'big'

def test_GetSizeBds():
    class Args:
        size = 'big'
    args = Args()
    result = GetSizeBds(args)
    assert result == [-100.0, 20.0, -60.0, 60.0]

def test_PlotErrAbs(gamera_pipe):
    fig, ax = plt.subplots()
    gsphP = gamera_pipe
    gsphO = gamera_pipe
    fieldNames = ['Bz']
    xyBds = [-100, 100, -100, 100]
    result = PlotErrAbs(gsphP, gsphO, 0, xyBds, ax, fieldNames)
    assert isinstance(result, np.ndarray)

def test_PlotErrRel(gamera_pipe):
    fig, ax = plt.subplots()
    gsphP = gamera_pipe
    gsphO = gamera_pipe
    fieldNames = ['Bz']
    xyBds = [-100, 100, -100, 100]
    result = PlotErrRel(gsphP, gsphO, 0, xyBds, ax, fieldNames)
    assert isinstance(result, np.ndarray)

def test_PlotLogicalErrAbs(gamera_pipe):
    fig, ax = plt.subplots()
    gsphP = gamera_pipe
    gsphO = gamera_pipe
    fieldNames = ['Bz']
    result = PlotLogicalErrAbs(gsphP, gsphO, 0, ax, fieldNames, 0)
    assert isinstance(result, np.ndarray)

def test_PlotLogicalErrRel(gamera_pipe):
    fig, ax = plt.subplots()
    gsphP = gamera_pipe
    gsphO = gamera_pipe
    fieldNames = ['Bz']
    result = PlotLogicalErrRel(gsphP, gsphO, 0, ax, fieldNames, 0)
    assert isinstance(result, np.ndarray)

def test_CalcTotalErrAbs(gamera_pipe):
    gsphP = gamera_pipe
    gsphO = gamera_pipe
    fieldNames = ['Bz']
    result = CalcTotalErrAbs(gsphP, gsphO, 0, fieldNames)
    assert isinstance(result, float)

def test_CalcTotalErrRel(gamera_pipe):
    gsphP = gamera_pipe
    gsphO = gamera_pipe
    fieldNames = ['Bz']
    result = CalcTotalErrRel(gsphP, gsphO, 0, fieldNames)
    assert isinstance(result, float)

def test_PlotEqB(gamera_pipe):
    fig, ax = plt.subplots()
    gsph = gamera_pipe
    xyBds = [-100, 100, -100, 100]
    result = PlotEqB(gsph, 0, xyBds, ax)
    assert isinstance(result, np.ndarray)

def test_PlotMerid(gamera_pipe):
    fig, ax = plt.subplots()
    gsph = gamera_pipe
    xyBds = [-100, 100, -100, 100]
    PlotMerid(gsph, 0, xyBds, ax)
    assert True  # No return value to check

def test_PlotJyXZ(gamera_pipe):
    fig, ax = plt.subplots()
    gsph = gamera_pipe
    xyBds = [-100, 100, -100, 100]
    PlotJyXZ(gsph, 0, xyBds, ax)
    assert True  # No return value to check

def test_PlotEqEphi(gamera_pipe):
    fig, ax = plt.subplots()
    gsph = gamera_pipe
    xyBds = [-100, 100, -100, 100]
    PlotEqEphi(gsph, 0, xyBds, ax)
    assert True  # No return value to check

def test_PlotMPI(gamera_pipe):
    fig, ax = plt.subplots()
    gsph = gamera_pipe
    PlotMPI(gsph, ax)
    assert True  # No return value to check

def test_AddIonBoxes(gamera_pipe):
    fig = plt.figure()
    gs = gridspec.GridSpec(3,6,height_ratios=[20,1,1],hspace=0.025)
    iontag = gamera_pipe.ftag.replace('.gam', '')
    ionfn = os.path.join(gamera_pipe.fdir, iontag + '.mix.h5')
    print(f'ionfn: {ionfn}')
    ion = remix.remix(ionfn, 0)
    AddIonBoxes(gs[0,3:], ion)
    assert True  # No return value to check

def test_plotPlane(gamera_pipe):
    fig = plt.figure()
    gs = gridspec.GridSpec(2,1,height_ratios=[20,1],hspace=0.025)
    AxM = fig.add_subplot(gs[0, 0])
    AxCB = fig.add_subplot(gs[-1, 0])
    gsph = gamera_pipe
    data = gsph.EggSlice('D', 0, doEq=True)
    xyBds = [-100, 100, -100, 100]
    plotPlane(gsph, data, xyBds, AxM, AxCB)
    assert True  # No return value to check

def test_plotXY(gamera_pipe):
    fig = plt.figure()
    gs = gridspec.GridSpec(2,1,height_ratios=[20,1],hspace=0.025)
    AxM = fig.add_subplot(gs[0, 0])
    AxCB = fig.add_subplot(gs[-1, 0])
    gsph = gamera_pipe
    xyBds = [-100, 100, -100, 100]
    result = plotXY(gsph, 0, xyBds, AxM, AxCB)
    assert isinstance(result, np.ndarray)

def test_plotXZ(gamera_pipe):
    fig = plt.figure()
    gs = gridspec.GridSpec(2,1,height_ratios=[20,1],hspace=0.025)
    AxM = fig.add_subplot(gs[0, 0])
    AxCB = fig.add_subplot(gs[-1, 0])
    gsph = gamera_pipe
    xzBds = [-100, 100, -100, 100]
    result = plotXZ(gsph, 0, xzBds, AxM, AxCB)
    assert isinstance(result, np.ndarray)