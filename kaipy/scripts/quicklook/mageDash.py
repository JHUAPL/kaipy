#!/usr/bin/env python


"""Display a live dashboard with information about an ongoing or completed MAGE run.

Display a live dashboard with information about an ongoing or completed MAGE run.

Author
------
Jeff Garretson
"""


# Import standard modules.
import argparse
import os
import socket
import threading
import sys
import signal
import logging
import io
import base64
import uuid
import datetime
import xml.etree.ElementTree as ET

# Import supplemental modules.
import h5py
import numpy as np
from flask_caching import Cache
from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QObject, pyqtSignal, QUrl
import matplotlib
matplotlib.use("Agg")  # safe for servers/headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import dates
import cmasher as cmr
from astropy.time import Time

# Kaipy modules
import kaipy.gamera.magsphere as msph
import kaipy.gamera.msphViz as mviz
import kaipy.raiju.raijuUtils as ru
import kaipy.raiju.raijuViz as rv
import kaipy.remix.remix as remix
import kaipy.kaiViz as kv
import kaipy.kaiTools as kt
import kaipy.kaiH5 as kaiH5

# lock because matplotlib isn't thread safe
matplotlib_lock = threading.Lock()

# parser options
caseXml = ""
firstStep = 1

# Dash GUI info below
app = Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

# Cache for large data
cache = Cache(app.server, config={"CACHE_TYPE": "simple", "CACHE_DEFAULT_TIMEOUT": 0})

stackPlot = html.Div(
    id="stackPlot",
    children=[
        dcc.Graph(id="perf-fig")
    ],
)

Dst_Plot = html.Div(
    id = "dst-panel",
    children=[
        html.Img(id="dst-image", style={"width": "95%", "height": "auto"})
    ]
)

Overview_Layout = html.Div(
    id = "overview-panel",
    children=[
        html.H5(id="overview-xml",children=""),
        html.H5(id="overview-runid",children=""),
        html.H5(id="overview-githash",children=""),
        html.H5(id="overview-gitbranch",children=""),
        html.H5(id="overview-compiler",children=""),
        html.H5(id="overview-compileropts",children="Compiler Options:"),
        dcc.Textarea(id="textarea-compileropts",disabled=True,style={'width': '100%', 'height': 75}),
        html.H5(id="overview-datetime",children=""),
        Dst_Plot
    ]
)

Performance_Layout = html.Div(
    id = "performance-panel",
    children=[
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3(children="Choose Plot Type"),
                    dcc.Dropdown(['Overall','Voltron','Gamera-Comp','Gamera-Ave','Gamera-Ind','Raiju'],
                                 'Overall', id='performance-dropdown',clearable=False,style={'width': '200px'})
                ], width=6),
                dbc.Col([
                    html.H3(children="Gamera I-Rank"),
                    dcc.Dropdown([1],value=1,id='performance-idrop',disabled=True,clearable=False,style={'width': '150px'})
                ],width=3),
                dbc.Col([
                    html.H3(children="Gamera J-Rank"),
                    dcc.Dropdown([1],value=1,id='performance-jdrop',disabled=True,clearable=False,style={'width': '150px'})
                ],width=3)
            ])
        ]),
        stackPlot
    ]
)

Msph_Plot = html.Div(
    id = "msph-panel",
    children=[
        html.Img(id="msph-image", style={"width": "95%", "height": "auto"})
    ]
)

Gamera_Layout = html.Div(
    id = "gamera-panel",
    children=[
        Msph_Plot
    ]
)

Raiju_Plot = html.Div(
    id = "raijuplot-panel",
    children=[
        html.Img(id="raiju-image", style={"width": "95%", "height": "auto"})
    ]
)

Raiju_Layout = html.Div(
    id = "raiju-panel",
    children=[
        Raiju_Plot
    ]
)

Remix_Plot = html.Div(
    id = "remixplot-panel",
    children=[
        html.Img(id="remix-image", style={"width": "95%", "height": "auto"})
    ]
)

Remix_Layout = html.Div(
    id = "remix-panel",
    children=[
        Remix_Plot
    ]
)

tab_layout = html.Div(
    id = "tab-panel",
    children=[
        dcc.Tabs([dcc.Tab(label='Overview',
                    children=[Overview_Layout]),
            dcc.Tab(label='Performance',
                    children=[Performance_Layout]),
            dcc.Tab(label='Gamera',
                    children=[Gamera_Layout]),
            dcc.Tab(label='Raiju',
                    children=[Raiju_Layout]),
            dcc.Tab(label='Remix',
                    children=[Remix_Layout]),
        ])
    ]
)

top_layout = html.Div(
    id = "top-panel",
    children=[
        dcc.Slider(0, 120, 60, value=60, id="timeSlider"),
        html.Button("Refresh Data", id="refresh-button", n_clicks=0)
    ],
)

main_panel_layout = html.Div(
    id="main-panel",
    children=[
        top_layout,
        tab_layout
    ],
)

root_layout = html.Div(
    id="root",
    children=[
        dcc.Store(id="case-store"),
        dcc.Store(id="store-data"),
        main_panel_layout
    ],
)

app.layout = root_layout


# non-dash helper routines here

def getStepFromTime(data, sliderValue):
    dataStep = np.argmin(np.abs(np.array(data['Vtime'])-sliderValue))
    return dataStep + firstStep

def fig_to_data_uri(fig, *, dpi=150, facecolor="white", tight=False):
    buf = io.BytesIO()
    canvas = FigureCanvas(fig)
    if tight:
        fig.tight_layout()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=facecolor, bbox_inches="tight" if tight else None)
    plt.close(fig)  # important to free memory when rendering in callbacks
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"

def get_free_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

class Communicator(QObject):
    dash_failed = pyqtSignal()

class DashViewer(QMainWindow):
    def __init__(self, url, shutdown_callback):
        super().__init__()
        self.setWindowTitle("MAGE Dashboard")
        self.setGeometry(100, 100, 1000, 800)

        self.shutdown_callback = shutdown_callback

        self.browser = QWebEngineView()
        self.browser.load(QUrl(url))

        layout = QVBoxLayout()
        layout.addWidget(self.browser)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def closeEvent(self, event):
        self.shutdown_callback()
        super().closeEvent(event)

def readCaseData(caseInfo):
    data = {}
    
    with h5py.File(f"{caseInfo['runid']}.volt.h5", 'r') as f:
        data['Vtime'] = f['/timeAttributeCache/time'][()][firstStep:]
        if "/timeAttributeCache/_perf_stepTime" in f:
            data['VstepTime'] = f['/timeAttributeCache/_perf_stepTime'][()][firstStep:]
            data['VdeepUpdateTime'] = f['/timeAttributeCache/_perf_deepUpdateTime'][()][firstStep:]
            data['VgamTime'] = f['/timeAttributeCache/_perf_gamTime'][()][firstStep:]
            data['VsquishTime'] = f['/timeAttributeCache/_perf_squishTime'][()][firstStep:]
            data['VimagTime'] = f['/timeAttributeCache/_perf_imagTime'][()][firstStep:]
            data['VmixTime'] = f['/timeAttributeCache/_perf_mixTime'][()][firstStep:]
            data['VtubesTime'] = f['/timeAttributeCache/_perf_tubesTime'][()][firstStep:]
            data['VioTime'] = f['/timeAttributeCache/_perf_ioTime'][()][firstStep:]

    for i in range(1,caseInfo['gamInum']+1):
        for j in range(1,caseInfo['gamJnum']+1):
            if caseInfo['gamSerial']:
                gamFilename = f'{caseInfo['runid']}.gam.h5'
            else:
                gamFilename = f'{caseInfo['runid']}_{caseInfo['gamInum']:04d}_{caseInfo['gamJnum']:04d}_0001_{i-1:04d}_{j-1:04d}_0000.gam.h5'
            with h5py.File(gamFilename, 'r') as f:
                data[f'Gtime{i}{j}'] = f['/timeAttributeCache/time'][()][firstStep:]
                data[f'Gdt{i}{j}'] = f['/timeAttributeCache/dt'][()][firstStep:]
                if "/timeAttributeCache/_perf_stepTime" in f:
                    data[f'GstepTime{i}{j}'] = f['/timeAttributeCache/_perf_stepTime'][()][firstStep:]
                    data[f'GvoltTime{i}{j}'] = f['/timeAttributeCache/_perf_voltTime'][()][firstStep:]
                    data[f'GmathTime{i}{j}'] = f['/timeAttributeCache/_perf_mathTime'][()][firstStep:]
                    data[f'GbcTime{i}{j}'] = f['/timeAttributeCache/_perf_bcTime'][()][firstStep:]
                    data[f'GhaloTime{i}{j}'] = f['/timeAttributeCache/_perf_haloTime'][()][firstStep:]
                    data[f'GioTime{i}{j}'] = f['/timeAttributeCache/_perf_ioTime'][()][firstStep:]
                    data[f'GadvanceTime{i}{j}'] = f['/timeAttributeCache/_perf_advanceTime'][()][firstStep:]
    
    data['gsph'] = str(uuid.uuid4())
    cache.set(data['gsph'], msph.GamsphPipe('.', caseInfo['runid'], doFast=True))
    
    with h5py.File(f"{caseInfo['runid']}.raiju.h5", 'r') as f:
        data['Rtime'] = f['/timeAttributeCache/time'][()][firstStep:]
        if "/timeAttributeCache/_perf_stepTime" in f:
            data['RstepTime'] = f['/timeAttributeCache/_perf_stepTime'][()][firstStep:]
            data['RpreAdvanceTime'] = f['/timeAttributeCache/_perf_preAdvance'][()][firstStep:]
            data['RadvanceTime'] = f['/timeAttributeCache/_perf_advanceState'][()][firstStep:]
            data['RmomentsTime'] = f['/timeAttributeCache/_perf_moments'][()][firstStep:]
    
    data['raiI'] = str(uuid.uuid4())
    cache.set(data['raiI'], ru.RAIJUInfo(f"{caseInfo['runid']}.raiju.h5",useTAC=True))
    
    '''
    Commented out because remix hasn't had performance data added to its output yet
    with h5py.File(f"{caseInfo['runid']}.mix.h5", 'r') as f:
        if "/timeAttributeCache/perf_stepTime" not in f:
            print("*** ERROR *** This run's output files are too old, and don't have the necessary data.")
            raise PreventUpdate
        
        data['MstepTime'] = f['/timeAttributeCache/perf_stepTime'][()][firstStep:]
        data['VgamTime'] = f['/timeAttributeCache/perf_gamTime'][()][firstStep:]
        data['VsquishTime'] = f['/timeAttributeCache/perf_squishTime'][()][firstStep:]
        data['VimagTime'] = f['/timeAttributeCache/perf_imagTime'][()][firstStep:]
        data['VmixTime'] = f['/timeAttributeCache/perf_mixTime'][()][firstStep:]
        data['VtubesTime'] = f['/timeAttributeCache/perf_tubesTime'][()][firstStep:]
        data['VioTime'] = f['/timeAttributeCache/perf_ioTime'][()][firstStep:]
        data['Vtime'] = f['/timeAttributeCache/time'][()][firstStep:]
    '''
    
    return data

def getCaseInsensitiveXmlElement(E, name):
    if E is None:
        return None
    for element in E.iter():
        if element.tag.casefold() == name.casefold():
            return element
    return None

def getCaseInsensitiveXmlAttribute(E, name):
    if E is None:
        return None
    for attr_name,attr_value in E.attrib.items():
        if attr_name.casefold() == name.casefold():
            return attr_value
    return None

def getCaseInfo():
    caseInfo = {}

    if not os.path.exists(caseXml):
        print(f"*** ERROR *** XML file {caseXml} could not be found.")
        raise PreventUpdate
    
    caseInfo['xml'] = caseXml

    tree = ET.parse(caseXml)
    root = tree.getroot()
    kaijuE = getCaseInsensitiveXmlElement(root, 'kaiju')
    gamE = getCaseInsensitiveXmlElement(kaijuE, 'gamera')
    voltE = getCaseInsensitiveXmlElement(kaijuE, 'voltron')
    
    caseInfo['gamSerial'] = getCaseInsensitiveXmlAttribute(getCaseInsensitiveXmlElement(voltE,'coupling'),'doserial')
    if caseInfo['gamSerial'] is None:
        caseInfo['gamSerial'] = False
    else:
        caseInfo['gamSerial'] = bool(caseInfo['gamSerial'])
    
    caseInfo['dtCouple'] = getCaseInsensitiveXmlAttribute(getCaseInsensitiveXmlElement(voltE,'coupling'),'dtCouple')
    if caseInfo['dtCouple'] is None:
        caseInfo['dtCouple'] = 5.0
    else:
        caseInfo['dtCouple'] = float(caseInfo['dtCouple'])

    caseInfo['runid'] = getCaseInsensitiveXmlAttribute(getCaseInsensitiveXmlElement(gamE,'sim'),'runid')
    if caseInfo['runid'] is None:
        caseInfo['runid'] = 'msphere'
    
    caseInfo['gamInum'] = getCaseInsensitiveXmlAttribute(getCaseInsensitiveXmlElement(gamE,'iPdir'),'N')
    if caseInfo['gamInum'] is None:
        caseInfo['gamInum'] = 1
    else:
        caseInfo['gamInum'] = int(caseInfo['gamInum'])
    
    caseInfo['gamJnum'] = getCaseInsensitiveXmlAttribute(getCaseInsensitiveXmlElement(gamE,'jPdir'),'N')
    if caseInfo['gamJnum'] is None:
        caseInfo['gamJnum'] = 1
    else:
        caseInfo['gamJnum'] = int(caseInfo['gamJnum'])
    
    if os.path.exists(f"{caseInfo['runid']}.gamCpl.h5"):
        caseInfo['ismpi'] = True
    else:
        caseInfo['ismpi'] = False

    if not os.path.exists(f"{caseInfo['runid']}.volt.h5"):
        print(f"*** ERROR *** Case files could not be found using runid {caseInfo['runid']}")
        raise PreventUpdate
    
    with h5py.File(f"{caseInfo['runid']}.volt.h5", 'r') as f:
        caseInfo['githash'] = np.char.decode(f.attrs['GITHASH'], encoding='utf-8')
        caseInfo['gitbranch'] = np.char.decode(f.attrs['GITBRANCH'], encoding='utf-8')
        caseInfo['compiler'] = np.char.decode(f.attrs['COMPILER'], encoding='utf-8')
        caseInfo['compileropts'] = np.char.decode(f.attrs['COMPILEROPTS'], encoding='utf-8')
        caseInfo['datetime'] = np.char.decode(f.attrs['DATETIME'], encoding='utf-8')

    return caseInfo

# dash app callbacks here
@app.callback(
    Output("case-store", "data"),
    Input("case-store", "id") # Ensures this is triggered first on startup, and then never again
)
def update_caseInfo(id):
    caseInfo = getCaseInfo()
    return caseInfo

@app.callback(
    Output("overview-xml","children"),
    Output("overview-runid","children"),
    Output("overview-githash","children"),
    Output("overview-gitbranch","children"),
    Output("overview-compiler","children"),
    Output("textarea-compileropts","value"),
    Output("overview-datetime","children"),
    Input("case-store", "data"),
    prevent_initial_call=True
)
def updateOverviewText(caseInfo):
    return [
        f"Case XML: {caseInfo['xml']}",
        f"Runid: {caseInfo['runid']}",
        f"Githash: {caseInfo['githash']}",
        f"Gitbranch: {caseInfo['gitbranch']}",
        f"Compiler: {caseInfo['compiler']}",
        caseInfo['compileropts'],
        f"Sim Started At: {caseInfo['datetime']}"
        ]

@app.callback(
    Output("performance-idrop","options"),
    Output("performance-jdrop","options"),
    Input("case-store", "data"),
    prevent_initial_call=True
)
def updateRankDropdownOptions(caseInfo):
    return [list(range(1,caseInfo['gamInum']+1)), list(range(1,caseInfo['gamJnum']+1))]

@app.callback(
    Output("store-data", "data"),
    Input("refresh-button", "n_clicks"),
    Input("case-store", "data"),
    prevent_initial_call=True
)
def update_data(nClicks, caseInfo):
    data = readCaseData(caseInfo)
    return data

@app.callback(
    Output("performance-idrop", "disabled"),
    Output("performance-jdrop", "disabled"),
    Input("performance-dropdown","value"),
    prevent_initial_call=True
)
def enableRankDropdowns(perfType):
    if perfType == 'Gamera-Ind':
        return [False,False]
    return [True,True]

@app.callback(
    Output("perf-fig", "figure"),
    Input("store-data", "data"),
    Input("case-store", "data"),
    Input("performance-dropdown","value"),
    Input("performance-idrop","value"),
    Input("performance-jdrop","value"),
    prevent_initial_call=True
)
def update_stackplot(data, caseInfo, perfType, iRank, jRank):
    fig = go.Figure()

    if 'VstepTime' not in data:
        fig.add_annotation(
            text="Performance data not in MAGE output files.<br>Please rerun simulation with newer version.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20,color="black")
        )
        return fig

    if perfType == 'Voltron':
        fig.add_trace(go.Scatter(x=data['Vtime'], y=data['VgamTime'], name='Gamera Time',
                    mode='lines', fill='tozeroy', stackgroup='one'))
        fig.add_trace(go.Scatter(x=data['Vtime'], y=data['VsquishTime'], name='Squish Time',
                    mode='lines', fill='tonexty', stackgroup='one'))
        fig.add_trace(go.Scatter(x=data['Vtime'], y=data['VimagTime'], name='Imag Time',
                    mode='lines', fill='tonexty', stackgroup='one'))
        fig.add_trace(go.Scatter(x=data['Vtime'], y=data['VmixTime'], name='Mix Time',
                    mode='lines', fill='tonexty', stackgroup='one'))
        fig.add_trace(go.Scatter(x=data['Vtime'], y=data['VtubesTime'], name='Tubes Time',
                    mode='lines', fill='tonexty', stackgroup='one'))
        fig.add_trace(go.Scatter(x=data['Vtime'], y=data['VioTime'], name='I/O Time',
                    mode='lines', fill='tonexty', stackgroup='one'))
        fig.update_layout(title="Voltron Performance")
        fig.update_layout(xaxis_title="Simulation Time", yaxis_title="Time per Update")
    elif perfType == 'Overall':
        fig.add_trace(go.Scatter(x=data['Vtime'], y=100*caseInfo['dtCouple']/np.array(data['VdeepUpdateTime']),
                    name='Voltron Performance', mode='lines'))
        if caseInfo['ismpi'] and not caseInfo['gamSerial']:
            fig.add_trace(go.Scatter(x=data['Gtime11'], y=100*np.array(data['Gdt11'])/np.array(data['GadvanceTime11']),
                        name='Gamera Performance', mode='lines'))
        fig.update_layout(title="Overall Performance")
        fig.update_layout(xaxis_title="Simulation Time", yaxis_title="% of Real-Time")
    elif perfType == 'Gamera-Comp':
        for i in range(1,caseInfo['gamInum']+1):
            for j in range(1,caseInfo['gamJnum']+1):
                fig.add_trace(go.Scatter(x=data[f'Gtime{i}{j}'],
                        y=100*np.array(data[f'Gdt{i}{j}'])/np.array(data[f'GadvanceTime{i}{j}']),
                        name=f'G-I{i}-J{j}', mode='lines'))
        fig.update_layout(title="Gamera Performance By Rank")
        fig.update_layout(xaxis_title="Simulation Time", yaxis_title="% of Real-Time")
    elif perfType == 'Gamera-Ave':
        GmathTimeAve = np.array(data['GmathTime11']).copy()
        GbcTimeAve = np.array(data['GbcTime11']).copy()
        GhaloTimeAve = np.array(data['GhaloTime11']).copy()
        GioTimeAve = np.array(data['GioTime11']).copy()
        for i in range(1,caseInfo['gamInum']+1):
            for j in range(1,caseInfo['gamJnum']+1):
                if i > 1 or j > 1:
                    GmathTimeAve = GmathTimeAve + np.array(data[f'GmathTime{i}{j}'])
                    GbcTimeAve = GbcTimeAve + np.array(data[f'GbcTime{i}{j}'])
                    GhaloTimeAve = GhaloTimeAve + np.array(data[f'GhaloTime{i}{j}'])
                    GioTimeAve = GioTimeAve + np.array(data[f'GioTime{i}{j}'])
        GmathTimeAve = GmathTimeAve / (caseInfo['gamInum']*caseInfo['gamJnum'])
        GbcTimeAve = GbcTimeAve / (caseInfo['gamInum']*caseInfo['gamJnum'])
        GhaloTimeAve = GhaloTimeAve / (caseInfo['gamInum']*caseInfo['gamJnum'])
        GioTimeAve = GioTimeAve / (caseInfo['gamInum']*caseInfo['gamJnum'])
        fig.add_trace(go.Scatter(x=data['Gtime11'], y=GmathTimeAve, name='Math Time',
                    mode='lines', fill='tozeroy', stackgroup='one'))
        fig.add_trace(go.Scatter(x=data['Gtime11'], y=GbcTimeAve, name='BC Time',
                    mode='lines', fill='tonexty', stackgroup='one'))
        fig.add_trace(go.Scatter(x=data['Gtime11'], y=GhaloTimeAve, name='Halo Time',
                    mode='lines', fill='tonexty', stackgroup='one'))
        fig.add_trace(go.Scatter(x=data['Gtime11'], y=GioTimeAve, name='IO Time',
                    mode='lines', fill='tonexty', stackgroup='one'))
        fig.update_layout(title="Gamera Performance (Averaged Over All Ranks)")
        fig.update_layout(xaxis_title="Simulation Time", yaxis_title="Time per Update")
    elif perfType == 'Gamera-Ind':
        fig.add_trace(go.Scatter(x=data[f'Gtime{iRank}{jRank}'], y=data[f'GmathTime{iRank}{jRank}'],
                    name='Math Time', mode='lines', fill='tozeroy', stackgroup='one'))
        fig.add_trace(go.Scatter(x=data[f'Gtime{iRank}{jRank}'], y=data[f'GbcTime{iRank}{jRank}'],
                    name='BC Time', mode='lines', fill='tonexty', stackgroup='one'))
        fig.add_trace(go.Scatter(x=data[f'Gtime{iRank}{jRank}'], y=data[f'GhaloTime{iRank}{jRank}'],
                    name='Halo Time', mode='lines', fill='tonexty', stackgroup='one'))
        fig.add_trace(go.Scatter(x=data[f'Gtime{iRank}{jRank}'], y=data[f'GioTime{iRank}{jRank}'],
                    name='IO Time', mode='lines', fill='tonexty', stackgroup='one'))
        fig.update_layout(title=f"Gamera Performance-I{iRank}-J{jRank}")
        fig.update_layout(xaxis_title="Simulation Time", yaxis_title="Time per Update")
        pass
    elif perfType == 'Raiju':
        fig.add_trace(go.Scatter(x=data['Rtime'], y=data['RpreAdvanceTime'], name='Pre-Advance Time',
                    mode='lines', fill='tozeroy', stackgroup='one'))
        fig.add_trace(go.Scatter(x=data['Rtime'], y=data['RadvanceTime'], name='Advance Time',
                    mode='lines', fill='tonexty', stackgroup='one'))
        fig.add_trace(go.Scatter(x=data['Rtime'], y=data['RmomentsTime'], name='Moments Time',
                    mode='lines', fill='tonexty', stackgroup='one'))
        fig.update_layout(title="Raiju Performance")
        fig.update_layout(xaxis_title="Simulation Time", yaxis_title="Time per Update")
    else:
        fig.add_annotation(
            text=f"Unrecognized plot type '{perfType}'",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20,color="black")
        )

    return fig

@app.callback(
    Output("timeSlider", "min"),
    Output("timeSlider", "max"),
    Input("store-data", "data"),
    prevent_initial_call=True
)
def update_slider(data):
    return data['Vtime'][0], data['Vtime'][-1]

@app.callback(
    Output("msph-image", "src"),
    Input("store-data", "data"),
    Input("timeSlider", "value"),
    prevent_initial_call=True
)
def updateMsphPlot(data, sliderValue):
    if data['gsph'] is None:
        return ''
    
    with matplotlib_lock:
        step = getStepFromTime(data, sliderValue)
        figSz=(12, 7.5)
        # surprisingly hard to pass the size string
        sizeArg = lambda: None
        sizeArg.size = 'std'
        xyBds=mviz.GetSizeBds(sizeArg)
        fig = plt.figure(figsize=figSz)
        gs = gridspec.GridSpec(3, 1, height_ratios=[20, 1, 1], hspace=0.025)
        Ax = fig.add_subplot(gs[0, 0])
        Clb = fig.add_subplot(gs[-1, 0])
        mviz.PlotEqB(cache.get(data['gsph']), step, xyBds, Ax, Clb, doBz=False)
        return fig_to_data_uri(fig)

@app.callback(
    Output("raiju-image", "src"),
    Input("store-data", "data"),
    Input("case-store", "data"),
    Input("timeSlider", "value"),
    prevent_initial_call=True
)
def updateRaijuPlot(data, caseInfo, sliderValue):
    if data['raiI'] is None:
        return ''
    
    with matplotlib_lock:
        raiI = cache.get(data['raiI'])

        step = getStepFromTime(data, sliderValue)
        figSz = (13,7)
        eqBnds = [-15,10,-10,10]
        fig = plt.figure(figsize=figSz)
        gs = gridspec.GridSpec(4, 1, height_ratios=[0.1,1,1,0.1],hspace=0.2,wspace=0.18)
        P_Clb = fig.add_subplot(gs[0, 0])
        P_Ax = fig.add_subplot(gs[1, 0])
        E_Ax = fig.add_subplot(gs[2, 0])
        E_Clb = fig.add_subplot(gs[3, 0])
        norm_press = kv.genNorm(05e-2,50,doLog=False)
        cmap_press = cmr.lilac
        kv.genCB(P_Clb, norm_press, "Proton Pressure [nPa]",cmap_press)
        P_Clb.xaxis.set_ticks_position("top")
        P_Clb.xaxis.set_label_position("top")
        kv.genCB(E_Clb, norm_press, "Electron Pressure [nPa]",cmap_press)

        with h5py.File(f"{caseInfo['runid']}.raiju.h5", 'r') as rFile:
            s5 = rFile[f"Step#{step}"]
            spcIdx_p = ru.spcIdx(raiI.species, ru.flavs_s['HOTP'])
            spcIdx_e = ru.spcIdx(raiI.species, ru.flavs_s['HOTE'])
            spcIdx_psph = ru.spcIdx(raiI.species, ru.flavs_s['PSPH'])
            xmin = ru.getVar(s5,'xmin')
            ymin = ru.getVar(s5,'ymin')
            topo = ru.getVar(s5,'topo')
            active = ru.getVar(s5,'active')
            # plotting only active domain for now
            #if config['domain'] == "ACTIVE":
            #    mask_cc = active != ru.domain['ACTIVE']
            #elif config['domain'] == "BUFFER":
            #    mask_cc = active != ru.domain['INACTIVE']
            mask_cc = active != ru.domain['ACTIVE']
            mask_corner = topo==ru.topo['OPEN']
            press_all = ru.getVar(s5,'Pressure',mask=mask_cc,broadcast_dims=(2,))
            press_p = press_all[:,:,spcIdx_p+1]  # First slot is bulk
            press_e = press_all[:,:,spcIdx_e+1]
            pot_corot = ru.getVar(s5, 'pot_corot', mask=mask_corner)
            pot_iono  = ru.getVar(s5, 'pot_iono' , mask=mask_corner)
            pot_total = pot_corot + pot_iono
            levels_pot = np.arange(-250, 255, 5)

            rv.plotXYMin(P_Ax, xmin, ymin, press_p,norm=norm_press,bnds=eqBnds,cmap=cmap_press)
            P_Ax.contour(xmin, ymin, pot_total, levels=levels_pot, colors='white',linewidths=0.5, alpha=0.3)
            rv.plotXYMin(E_Ax, xmin, ymin, press_e,norm=norm_press,bnds=eqBnds,cmap=cmap_press)
            E_Ax.contour(xmin, ymin, pot_total, levels=levels_pot, colors='white',linewidths=0.5, alpha=0.3)
        
        return fig_to_data_uri(fig)

@app.callback(
    Output("remix-image", "src"),
    Input("store-data", "data"),
    Input("case-store", "data"),
    Input("timeSlider", "value"),
    prevent_initial_call=True
)
def updateRemixPlot(data, caseInfo, sliderValue):
    with matplotlib_lock:
        step = getStepFromTime(data, sliderValue)
        figSz = (12,7.5)
        fig = plt.figure(figsize=figSz)
        gs = gridspec.GridSpec(1, 2, figure=fig, left=0.03, right=0.97, top=0.9, bottom=0.03)
        ion = remix.remix(f"{caseInfo['runid']}.mix.h5", step)
        ion.init_vars('NORTH')
        ion.plot('current', gs=gs[0, 0])
        ion.init_vars('SOUTH')
        ion.plot('current', gs=gs[0, 1])
        return fig_to_data_uri(fig)

@app.callback(
    Output("dst-image", "src"),
    Input("store-data", "data"),
    Input("case-store", "data"),
    prevent_initial_call=True
)
def updateDstPlot(data, caseInfo):
    with matplotlib_lock:
        LW = 0.75
        tpad = 8 #Number of hours beyond MHD to plot
        figSz = (14,7)
        fig = plt.figure(figsize=figSz)
        gs = gridspec.GridSpec(1,1,hspace=0.05,wspace=0.05)
        ax=fig.add_subplot(gs[0,0])

        #UT formats for plotting
        isotfmt = '%Y-%m-%dT%H:%M:%S.%f'
        utfmt   = '%H:%M \n%Y-%m-%d'

        ut_symh,tD,dstD = kt.GetSymH("bcwind.h5")
        fvolt = f"{caseInfo['runid']}.volt.h5"
        BSDst = kaiH5.getTs(fvolt,sIds=None,aID="BSDst")
        MJD = kaiH5.getTs(fvolt,sIds=None,aID="MJD")
        I = np.isinf(MJD)
        MJD0 = MJD[~I].min()-1
        MJD[I] = MJD0
        UT = Time(MJD,format='mjd').isot
        ut = [datetime.datetime.strptime(UT[n],isotfmt) for n in range(len(UT))]
        iMax = len(ut)-1
        # Remove Restart Step. Tends to cause weird artifacts
        deldt = []
        for it in range(iMax,1,-1):
            dt = ut[it] - ut[it-1]
            dt = dt.total_seconds()
            if dt < 2.:
                deldt.append(it)
        BSDst = np.delete( BSDst,deldt )
        ut = np.delete( ut,deldt )

        ax.plot(ut_symh,dstD,label="SYM-H",linewidth=2*LW)
        ax.plot(ut,BSDst,label="Biot-Savart Dst",linewidth=LW)
        ax.legend(loc='upper right',fontsize="small",ncol=2)
        ax.axhline(color='magenta',linewidth=0.5*LW)
        ax.xaxis_date()
        xfmt = dates.DateFormatter(utfmt)
        ax.set_ylabel("Dst [nT]")
        ax.xaxis.set_major_formatter(xfmt)
        xMinD = np.array(ut_symh).min()
        xMaxD = np.array(ut_symh).max()
        xMinS = np.array(ut).min()
        xMaxS = np.array(ut).max()
        if (xMaxD>xMaxS):
            xMax = min(xMaxS+datetime.timedelta(hours=tpad),xMaxD)
        else:
            xMax = xMaxS
        xMin = xMinD
        ax.set_xlim(xMin,xMax)
        return fig_to_data_uri(fig)


# main startup routines
def create_command_line_parser():
    """Create the command-line argument parser.

    Create the parser for command-line arguments.

    Returns:
        argparse.ArgumentParser: Command-line argument parser for this script.
    """
    parser = argparse.ArgumentParser(
        description="Shows a live dashboard for an ongoing, or completed, MAGE case.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "caseXml", help="Xml file used to run the case being shown."
    )
    parser.add_argument(
        "-firstStep", type=int, default=firstStep, help="First Step# group to read data from in the case"
    )
    
    return parser

def main():
    global caseXml
    global firstStep
    """Display a live dashboard with information about an ongoing or completed MAGE run."""

    # Set up the command-line parser.
    parser = create_command_line_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    caseXml = args.caseXml
    firstStep = args.firstStep

    port = get_free_port()
    dash_url = f"http://127.0.0.1:{port}"
    server_thread = None

    comms = Communicator()

    def shutdown_dash():
        if server_thread and server_thread.is_alive():
            print("Shutting down Dash server...")
            signal.pthread_kill(server_thread.ident, signal.SIGINT)

    # Cleanly exit Qt app if Dash crashes
    def on_dash_failed():
        print("Dash failed to start. Closing Qt app.")
        qt_app.quit()

    comms.dash_failed.connect(on_dash_failed)

    def run_dash():
        try:
            app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False)
        except Exception as e:
            print(f"Dash server encountered an error: {e}")
            comms.dash_failed.emit()  # Notify GUI thread to shut down

    # reduce dash console output
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    server_thread = threading.Thread(target=run_dash)
    server_thread.daemon = True
    server_thread.start()

    #os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"

    qt_app = QApplication(sys.argv)
    viewer = DashViewer(dash_url, shutdown_dash)
    viewer.show()
    sys.exit(qt_app.exec_())

if __name__ == "__main__":
    main()

