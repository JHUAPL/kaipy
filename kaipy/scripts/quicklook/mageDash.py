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
import kaipy.kaixml as kx

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
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3(children="Gamera Plot Size"),
                    dcc.Dropdown(['std','big','bigger','fullD','small'],
                                 'std', id='msph-sizeDropdown',clearable=False,style={'width': '400px'})
                ], width=4)
            ])
        ]),
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
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3(children="Raiju Plot Type"),
                    dcc.Dropdown(['P/E Pressure','P/E Density','bVol / FTE','Diff bVol / FTE','wIMAG / Tbounce'],
                                 'P/E Pressure', id='raiju-typeDropdown',clearable=False,style={'width': '200px'})
                ], width=4),
                dbc.Col([
                    html.H3(children="Raiju Buffer"),
                    dcc.Dropdown(['ACTIVE','BUFFER'],
                                 'ACTIVE', id='raiju-bufferDropdown',clearable=False,style={'width': '200px'})
                ], width=4)
            ])
        ]),
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
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3(children="Remix Plot Type"),
                    dcc.Dropdown(['current','sigmap','sigmah','joule','energy','flux','eflux'],
                                 'current', id='remix-typeDropdown',clearable=False,style={'width': '400px'})
                ], width=4)
            ])
        ]),
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
        html.Div([dcc.Slider(1, 2, 1, value=1, id="timeSlider")],
                style={"width": "95%", "margin-left": "auto", "margin-right": "auto", "height": "60px"}),
        html.Button("Refresh Data", id="refresh-button", n_clicks=0)
    ]
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

def getIndexFromTime(data, sliderValue):
    dataStep = np.argmin(np.abs(np.array(data['Vtime'])-sliderValue))
    return dataStep

def getStepFromTime(data, sliderValue):
    return getIndexFromTime(data, sliderValue) + firstStep

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
        data['Vmjd'] = f['/timeAttributeCache/MJD'][()][firstStep:]
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
                gamFilename = f"{caseInfo['runid']}.gam.h5"
            else:
                gamFilename = f"{caseInfo['runid']}_{caseInfo['gamInum']:04d}_{caseInfo['gamJnum']:04d}_0001_{i-1:04d}_{j-1:04d}_0000.gam.h5"
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

def getCaseInfo():
    caseInfo = {}

    if not os.path.exists(caseXml):
        print(f"*** ERROR *** XML file {caseXml} could not be found.")
        raise PreventUpdate
    
    caseInfo['xml'] = caseXml

    tree = ET.parse(caseXml)
    root = tree.getroot()
    kaijuE = kx.getXmlElement(root, 'kaiju')
    gamE = kx.getXmlElement(kaijuE, 'gamera')
    voltE = kx.getXmlElement(kaijuE, 'voltron')
    
    caseInfo['gamSerial'] = kx.getXmlAttribute(kx.getXmlElement(voltE,'coupling'),'doserial')
    if caseInfo['gamSerial'] is None:
        caseInfo['gamSerial'] = False
    else:
        caseInfo['gamSerial'] = bool(caseInfo['gamSerial'])
    
    caseInfo['dtCouple'] = kx.getXmlAttribute(kx.getXmlElement(voltE,'coupling'),'dtCouple')
    if caseInfo['dtCouple'] is None:
        caseInfo['dtCouple'] = 5.0
    else:
        caseInfo['dtCouple'] = float(caseInfo['dtCouple'])

    caseInfo['runid'] = kx.getXmlAttribute(kx.getXmlElement(gamE,'sim'),'runid')
    if caseInfo['runid'] is None:
        caseInfo['runid'] = 'msphere'
    
    caseInfo['gamInum'] = kx.getXmlAttribute(kx.getXmlElement(gamE,'iPdir'),'N')
    if caseInfo['gamInum'] is None:
        caseInfo['gamInum'] = 1
    else:
        caseInfo['gamInum'] = int(caseInfo['gamInum'])
    
    caseInfo['gamJnum'] = kx.getXmlAttribute(kx.getXmlElement(gamE,'jPdir'),'N')
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
    Output("timeSlider", "step"),
    Output("timeSlider", "marks"),
    Input("store-data", "data"),
    prevent_initial_call=True
)
def update_slider(data):
    marks = {}
    utfmt = '%Y-%m-%d %H:%M:%S.%f'
    numMarks = 5 # number of time labels to fit on the bar total
    markStep = 0.1
    duration = data['Vtime'][-1] - data['Vtime'][0]
    if duration <= (numMarks-1) * 0.1:
        markStep = 0.1
    elif duration <= (numMarks-1) * 1:
        markStep = 1
    elif duration <= (numMarks-1) * 5:
        markStep = 5
    elif duration <= (numMarks-1) * 15:
        markStep = 15
    elif duration <= (numMarks-1) * 30:
        markStep = 30
    elif duration <= (numMarks-1) * 60:
        markStep = 60
    elif duration <= (numMarks-1) * 60*5:
        markStep = 60*5
    elif duration <= (numMarks-1) * 60*15:
        markStep = 60*15
    elif duration <= (numMarks-1) * 60*30:
        markStep = 60*30
    elif duration <= (numMarks-1) * 60*60:
        markStep = 60*60
    elif duration <= (numMarks-1) * 60*60*3:
        markStep = 60*60*3
    elif duration <= (numMarks-1) * 60*60*6:
        markStep = 60*60*6
    elif duration <= (numMarks-1) * 60*60*12:
        markStep = 60*60*12
    elif duration <= (numMarks-1) * 60*60*24:
        markStep = 60*60*24
    elif duration <= (numMarks-1) * 60*60*24*2:
        markStep = 60*60*24*2
    elif duration <= (numMarks-1) * 60*60*24*4:
        markStep = 60*60*24*4
    else:
        markStep = 60*60*24*7 # one week per interval. how long is this simulation?
    
    if markStep >= 1:
        # print whole seconds only
        utfmt = '%Y-%m-%d %H:%M:%S'
    markStyle = {'min-width': '90px','max-width': '90px', 'text-align': 'center'}
    markTime = data['Vtime'][0]
    while markTime <= data['Vtime'][-1]:
        markIndex = getIndexFromTime(data, markTime)
        markData = {'label' : kt.MJD2UT(data['Vmjd'][markIndex]).strftime(utfmt), 'style' : markStyle}
        marks[int(markIndex)] = markData
        markTime = markTime + markStep
    # ensure that the last time always has a mark, and that the mark before it isn't too close
    if (data['Vtime'][-1] - data['Vtime'][max(marks.keys())]) < 0.8*markStep:
        # last mark added is less than 80% of a step from the end time
        # delete it before adding end time
        del marks[max(marks.keys())]
    markData = {'label' : kt.MJD2UT(data['Vmjd'][-1]).strftime(utfmt), 'style' : markStyle}
    marks[len(data['Vtime'])-1] = markData
    return [0,
            len(data['Vtime'])-1,
            1,
            marks]

@app.callback(
    Output("msph-image", "src"),
    Input("store-data", "data"),
    Input("timeSlider", "value"),
    Input("msph-sizeDropdown","value"),
    prevent_initial_call=True
)
def updateMsphPlot(data, sliderValue, dropSize):
    if data['gsph'] is None:
        return ''
    gsph = cache.get(data['gsph'])
    
    with matplotlib_lock:
        step =  sliderValue + firstStep
        figSz=(12, 7.5)
        # surprisingly hard to pass the size string
        sizeArg = lambda: None
        sizeArg.size = dropSize
        xyBds=mviz.GetSizeBds(sizeArg)
        fig = plt.figure(figsize=figSz)
        gs = gridspec.GridSpec(3, 1, height_ratios=[20, 3, 1], hspace=0.025)
        Ax = fig.add_subplot(gs[0, 0])
        Clb = fig.add_subplot(gs[-1, 0])
        mviz.PlotEqB(gsph, step, xyBds, Ax, Clb, doBz=False)
        gsph.AddTime(step, Ax, xy=[0.025, 0.89], fs="x-large")
        gsph.AddSW(step, Ax, xy=[0.625, 0.025], fs="small")
        mviz.PlotMPI(gsph, Ax)

        return fig_to_data_uri(fig)

# copied from raijupic
def get_bVol_dipole(f5):
    colats = ru.getVar(f5,'X')  # [rad]
    R = f5['Grid']['ShellGrid'].attrs['radius']  # [Rp]
    Leq = R/( np.sin(colats[:,0])*np.sin(colats[:,0]) )
    Ni, Nj = colats.shape
    bVol_dipole_1D = np.zeros(Ni)
    for i in range(Ni):
        bVol_dipole_1D[i] = kt.L_to_bVol(Leq[i])
    bVol_dipole_2D = np.broadcast_to(bVol_dipole_1D[:,np.newaxis], (Ni,Nj))
    return bVol_dipole_2D

@app.callback(
    Output("raiju-image", "src"),
    Input("store-data", "data"),
    Input("case-store", "data"),
    Input("timeSlider", "value"),
    Input("raiju-typeDropdown","value"),
    Input("raiju-bufferDropdown","value"),
    prevent_initial_call=True
)
def updateRaijuPlot(data, caseInfo, sliderValue, plotType, domainType):
    if data['raiI'] is None:
        return ''
    
    with matplotlib_lock:
        raiI = cache.get(data['raiI'])

        step = sliderValue + firstStep
        figSz = (13,7)
        eqBnds = [-15,10,-10,10]
        fig = plt.figure(figsize=figSz)
        gs = gridspec.GridSpec(4, 1, height_ratios=[0.1,1,1,0.1],hspace=0.2,wspace=0.18)
        P1_Clb = fig.add_subplot(gs[0, 0])
        P1_Ax = fig.add_subplot(gs[1, 0])
        P2_Ax = fig.add_subplot(gs[2, 0])
        P2_Clb = fig.add_subplot(gs[3, 0])
        with h5py.File(f"{caseInfo['runid']}.raiju.h5", 'r') as rFile:
            s5 = rFile[f"Step#{step}"]
            xmin = ru.getVar(s5,'xmin')
            ymin = ru.getVar(s5,'ymin')
            xmincc = kt.to_center2D(xmin)
            ymincc = kt.to_center2D(ymin)
            topo = ru.getVar(s5,'topo')
            active = ru.getVar(s5,'active')
            if domainType == "ACTIVE":
                mask_cc = active != ru.domain['ACTIVE']
            elif domainType == "BUFFER":
                mask_cc = active != ru.domain['INACTIVE']
            mask_corner = topo==ru.topo['OPEN']

            if plotType == 'P/E Pressure':
                norm_press = kv.genNorm(05e-2,50,doLog=False)
                cmap_press = cmr.lilac
                kv.genCB(P1_Clb, norm_press, "Proton Pressure [nPa]",cmap_press)
                P1_Clb.xaxis.set_ticks_position("top")
                P1_Clb.xaxis.set_label_position("top")
                kv.genCB(P2_Clb, norm_press, "Electron Pressure [nPa]",cmap_press)

                spcIdx_p = ru.spcIdx(raiI.species, ru.flavs_s['HOTP'])
                spcIdx_e = ru.spcIdx(raiI.species, ru.flavs_s['HOTE'])
                press_all = ru.getVar(s5,'Pressure',mask=mask_cc,broadcast_dims=(2,))
                press_p = press_all[:,:,spcIdx_p+1]  # First slot is bulk
                press_e = press_all[:,:,spcIdx_e+1]
                pot_corot = ru.getVar(s5, 'pot_corot', mask=mask_corner)
                pot_iono  = ru.getVar(s5, 'pot_iono' , mask=mask_corner)
                pot_total = pot_corot + pot_iono
                levels_pot = np.arange(-250, 255, 5)

                rv.plotXYMin(P1_Ax, xmin, ymin, press_p,norm=norm_press,bnds=eqBnds,cmap=cmap_press)
                P1_Ax.contour(xmin, ymin, pot_total, levels=levels_pot, colors='white',linewidths=0.5, alpha=0.3)
                rv.plotXYMin(P2_Ax, xmin, ymin, press_e,norm=norm_press,bnds=eqBnds,cmap=cmap_press)
                P2_Ax.contour(xmin, ymin, pot_total, levels=levels_pot, colors='white',linewidths=0.5, alpha=0.3)
            elif plotType == 'P/E Density':
                norm_den   = kv.genNorm(0,10, doLog=False)
                kv.genCB(P1_Clb, norm_den, "Proton Density [#/cc]")
                P1_Clb.xaxis.set_ticks_position("top")
                P1_Clb.xaxis.set_label_position("top")
                kv.genCB(P2_Clb, norm_den, "Electron Density [#/cc]")

                den_all   = ru.getVar(s5,'Density'  ,mask=mask_cc,broadcast_dims=(2,))
                spcIdx_p = ru.spcIdx(raiI.species, ru.flavs_s['HOTP'])
                spcIdx_e = ru.spcIdx(raiI.species, ru.flavs_s['HOTE'])
                spcIdx_psph = ru.spcIdx(raiI.species, ru.flavs_s['PSPH'])
                den_p = den_all[:,:,spcIdx_p+1]
                den_e = den_all[:,:,spcIdx_e+1]
                den_psph =  den_all[:,:,spcIdx_psph+1]

                levels_psphDen = [1,10,100,1000]
                rv.plotXYMin(P1_Ax, xmin, ymin, den_p,norm=norm_den,bnds=eqBnds)
                P1_Ax.contour(xmincc, ymincc, den_psph,levels=levels_psphDen,colors='white',linewidths=0.5,alpha=0.4)
                rv.plotXYMin(P2_Ax, xmin, ymin, den_e,norm=norm_den,bnds=eqBnds)
            elif plotType == 'bVol / FTE':
                norm_bvol = kv.genNorm(1e-4,1e0,doLog=True)
                norm_ent  = kv.genNorm(1e-4,0.5,doLog=True)
                cmap_bvol = cmr.dusk
                cmap_ent = 'gist_earth'

                kv.genCB(P1_Clb, norm_bvol, "bVol",cmap_bvol)
                P1_Clb.xaxis.set_ticks_position("top")
                P1_Clb.xaxis.set_label_position("top")
                kv.genCB(P2_Clb, norm_ent , "Flux tube entropy",cmap_ent)

                bVol    = ru.getVar(s5,'bVol'   ,mask=mask_corner)
                bVol_cc = kt.to_center2D(bVol)
                press_all = ru.getVar(s5,'Pressure',mask=mask_cc,broadcast_dims=(2,))
                entropy = press_all[:,:,0]*bVol_cc**(5./3.)  # Wolf units [nPa * (Rx/nT)^(5/3)]

                rv.plotXYMin(P1_Ax, xmin, ymin, bVol_cc,norm=norm_bvol,bnds=eqBnds,cmap=cmap_bvol)
                P1_Ax.contour(xmincc,ymincc,active,levels=[0.5],colors='orange')
                rv.plotXYMin(P2_Ax, xmin, ymin, entropy,norm=norm_ent ,bnds=eqBnds,cmap=cmap_ent)
            elif plotType == 'Diff bVol / FTE':
                norm_diff_bvol = kv.genNorm(-1e2,1e2,doSymLog=True)
                norm_ent  = kv.genNorm(1e-4,0.5,doLog=True)
                cmap_diff = 'RdBu_r'
                cmap_ent = 'gist_earth'

                kv.genCB(P1_Clb, norm_diff_bvol, "(V-V$_d$)/V$_d$",cM=cmap_diff)
                P1_Clb.xaxis.set_ticks_position("top")
                P1_Clb.xaxis.set_label_position("top")
                kv.genCB(P2_Clb, norm_ent , "Flux tube entropy",cmap_ent)

                bVol    = ru.getVar(s5,'bVol'   ,mask=mask_corner)
                bVol_cc = kt.to_center2D(bVol)
                press_all = ru.getVar(s5,'Pressure',mask=mask_cc,broadcast_dims=(2,))
                entropy = press_all[:,:,0]*bVol_cc**(5./3.)  # Wolf units [nPa * (Rx/nT)^(5/3)]
                bVol_dipole = get_bVol_dipole(rFile)
                bVol_dip_cc = kt.to_center2D(bVol_dipole)
                d_bVol_cc = (bVol_cc - bVol_dip_cc)/bVol_dip_cc

                rv.plotXYMin(P1_Ax, xmin, ymin, d_bVol_cc,norm=norm_diff_bvol,cmap=cmap_diff,bnds=eqBnds)    
                P1_Ax.contour(xmincc,ymincc,active,levels=[0.5],colors='orange')
                rv.plotXYMin(P2_Ax, xmin, ymin, entropy,norm=norm_ent ,bnds=eqBnds,cmap=cmap_ent)
            elif plotType == 'wIMAG / Tbounce':
                norm_wimag = kv.genNorm(0,1,doLog=False)
                norm_Tb = kv.genNorm(0,180,doLog=False)

                kv.genCB(P1_Clb, norm_wimag, "wIMAG")
                P1_Clb.xaxis.set_ticks_position("top")
                P1_Clb.xaxis.set_label_position("top")
                kv.genCB(P2_Clb, norm_Tb, "Tbounce [s]")

                tBounce = ru.getVar(s5,'Tbounce',mask=mask_cc)
                vaFrac  = ru.getVar(s5,'vaFrac',mask=mask_corner)

                rv.plotXYMin(P1_Ax, xmin, ymin, vaFrac,norm=norm_wimag,bnds=eqBnds)
                P1_Ax.contour(xmincc,ymincc,active,levels=[0.5],colors='orange')
                rv.plotXYMin(P2_Ax, xmin, ymin, tBounce,norm=norm_Tb,bnds=eqBnds)
        
        fig.suptitle(raiI.UTs[step])

        return fig_to_data_uri(fig)

@app.callback(
    Output("remix-image", "src"),
    Input("store-data", "data"),
    Input("case-store", "data"),
    Input("timeSlider", "value"),
    Input("remix-typeDropdown","value"),
    prevent_initial_call=True
)
def updateRemixPlot(data, caseInfo, sliderValue, plotType):
    with matplotlib_lock:
        step =  sliderValue + firstStep
        figSz = (12,7.5)
        fig = plt.figure(figsize=figSz)
        plt.figtext(
            0.5, 0.94, Time(data['Vmjd'][sliderValue], format='mjd').iso,
            fontsize=12, multialignment='center', horizontalalignment='center'
        )
        gs = gridspec.GridSpec(2, 2, figure=fig, left=0.03, right=0.97, top=0.9, bottom=0.03, height_ratios=[2, 20])
        ion = remix.remix(f"{caseInfo['runid']}.mix.h5", step)
        plt.figtext(0.25, 0.85, 'NORTH', fontsize=12, multialignment='center', horizontalalignment='center')
        plt.figtext(0.75, 0.85, 'SOUTH', fontsize=12, multialignment='center', horizontalalignment='center')
        ion.init_vars('NORTH')
        ion.plot(plotType, gs=gs[1, 0])
        ion.init_vars('SOUTH')
        ion.plot(plotType, gs=gs[1, 1])
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
    parser.add_argument(
        "--noBrowser", action='store_true', default=False, help="Use this flag to disable the Qt Web Browser"
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
    noBrowser = args.noBrowser

    if not noBrowser:
        global QWebEngineView
        from PyQt5.QtWebEngineWidgets import QWebEngineView # install as 'pip install PyQtWebEngine'

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
        if qt_app is not None:
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

    if not noBrowser:
        qt_app = QApplication(sys.argv)
        viewer = DashViewer(dash_url, shutdown_dash)
        viewer.show()
        sys.exit(qt_app.exec_())
    else:
        print("No browser is being started for you.")
        print(f"Please open your own browser and connect to the address '{dash_url}'")
        print("Use ctrl-c to end the dashboard.")
        server_thread.join()

if __name__ == "__main__":
    main()

