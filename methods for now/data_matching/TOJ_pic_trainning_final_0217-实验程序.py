#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2022.2.4),
    on 一月 13, 2023, at 15:05
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from __future__ import division
from __future__ import print_function

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding
import pylink
import platform
import random
import time
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
from math import fabs
import serial
import psychopy.iohub as io
from psychopy.hardware import keyboard
from PIL import Image  # for preparing the Host backdrop image
from string import ascii_letters, digits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ser=serial.Serial('COM4',9600)
ser.close()


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '2022.2.4'
expName = 'TOJ_pic_trainning_step_3'  # from the Builder filename that created this script
expInfo = {
    'monkey': 'M',
    'session': 'day3',
    'edf_fname': 'M0227001',
    'total_loop_times':'25',
    'fc_loop_times': '1',
    'cross_size': '110',
    'minimum_duration': '1',
    'memory_duration': '5',
    'delay_duration': '1',
    'choice_duration': '60',
    'list_index': '13',
    'list_length': '4',
    'pic_width': '576',
    'pic_height': '320',
    'roi_range_margin': '0',
    
}
# --- Show participant info dialog --
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

fc_loop_times = int(expInfo['fc_loop_times'])
cross_size = int(expInfo['cross_size'])
minimum_duration = float(expInfo['minimum_duration'])
memory_duration = int(expInfo['memory_duration'])
delay_duration = int(expInfo['delay_duration'])
choice_duration = int(expInfo['choice_duration'])
list_index =  int(expInfo['list_index'])-1
list_length =  int(expInfo['list_length'])
monkey = str(expInfo['monkey'])
pic_height = int(expInfo['pic_height'])
pic_width = int(expInfo['pic_width'])
total_loop_times = int(expInfo['total_loop_times'])
roi_range_margin = int(expInfo['roi_range_margin'])
#change you list here
df = pd.read_csv('F:\\Pic_TOJ\\list_step_3_similar_pic_formal.csv')
target_pic = 'F:\\Pic_TOJ\\pic_pool_step_3_formal\\1_1.png'
pic_chosen = target_pic

response = 0
reward_ornot = 0
jitter_iti = 0

sigma = 0.5
mu = 1.5
r = sigma * np.random.randn(fc_loop_times+50) + mu
r_0 = [ x for x in r if x>0]

data_list={'total_loop_times':[],'fc_loop_times':[],'cross_size':[],'minimum_duration':[],'memory_duration':[],
            'delay_duration':[],'choice_duration':[],'list_index':[],
           'monkey':[],'target_pic':[],'target_pic_position':[],'pic_bottom':[],'pic_left':[],
           'pic_right':[],'pic_chosen':[],'pic_chosen_position':[],'choice_rt':[],'response':[],'reward_ornot':[],'jitter_iti':[]}
           
# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s_%s_%s' % (expInfo['monkey'], expInfo['session'],expInfo['edf_fname'],expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='F:\\Pic_TOJ\\TOJ_pic_trainning_step_3.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame



# Start Code - component code to be run after the window creation

# --- Setup the Window ---
win = visual.Window(
    size=[2560, 1440], fullscr=True, screen=0, 
    winType='pyglet', allowStencil=False,
    monitor='testMonitor', color=[-1,-1,-1], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='pix')
win.mouseVisible = False
edf_fname = str(expInfo['edf_fname'])
# Set this variable to True to run the script in "Dummy Mode"
dummy_mode = False
use_retina = False
# Set up a folder to store the EDF data files and the associated resources
# e.g., files defining the interest areas used in each trial
results_folder = 'results'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# We download EDF data file from the EyeLink Host PC to the local hard
# drive at the end of each testing session, here we rename the EDF to
# include session start date/time
time_str = time.strftime("_%Y_%m_%d_%H_%M", time.localtime())
session_identifier = edf_fname + time_str

# create a folder for the current testing session in the "results" folder
session_folder = os.path.join(results_folder, session_identifier)
if not os.path.exists(session_folder):
    os.makedirs(session_folder)

# Step 1: Connect to the EyeLink Host PC
#
# The Host IP address, by default, is "100.1.1.1".
# the "el_tracker" objected created here can be accessed through the Pylink
# Set the Host PC address to "None" (without quotes) to run the script
# in "Dummy Mode"
if dummy_mode:
    el_tracker = pylink.EyeLink(None)
else:
    try:
        el_tracker = pylink.EyeLink("100.1.1.1")
    except RuntimeError as error:
        print('ERROR:', error)
        core.quit()
        sys.exit()

# Step 2: Open an EDF data file on the Host PC

edf_file = edf_fname + ".EDF"
try:
    el_tracker.openDataFile(edf_file)
except RuntimeError as err:
    print('ERROR:', err)
    # close the link if we have one open
    if el_tracker.isConnected():
        el_tracker.close()
    core.quit()
    sys.exit()

# Add a header text to the EDF file to identify the current experiment name
# This is OPTIONAL. If your text starts with "RECORDED BY " it will be
# available in DataViewer's Inspector window by clicking
# the EDF session node in the top panel and looking for the "Recorded By:"
# field in the bottom panel of the Inspector.
preamble_text = 'RECORDED BY %s' % os.path.basename(__file__)
el_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)

# Step 3: Configure the tracker
#
# Put the tracker in offline mode before we change tracking parameters
el_tracker.setOfflineMode()

# Get the software version:  1-EyeLink I, 2-EyeLink II, 3/4-EyeLink 1000,
# 5-EyeLink 1000 Plus, 6-Portable DUO
eyelink_ver = 0  # set version to 0, in case running in Dummy mode
if not dummy_mode:
    vstr = el_tracker.getTrackerVersionString()
    eyelink_ver = int(vstr.split()[-1].split('.')[0])
    # print out some version info in the shell
    print('Running experiment on %s, version %d' % (vstr, eyelink_ver))

# File and Link data control
# what eye events to save in the EDF file, include everything by default
file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
# what eye events to make available over the link, include everything by default
link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'
# what sample data to save in the EDF data file and to make available
# over the link, include the 'HTARGET' flag to save head target sticker
# data for supported eye trackers
if eyelink_ver > 3:
    file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
    link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
else:
    file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
    link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
el_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
el_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)
el_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
el_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)

# Optional tracking parameters
# Sample rate, 250, 500, 1000, or 2000, check your tracker specification
# if eyelink_ver > 2:
#     el_tracker.sendCommand("sample_rate 1000")
# Choose a calibration type, H3, HV3, HV5, HV13 (HV = horizontal/vertical),
el_tracker.sendCommand("calibration_type = HV9")
# Set a gamepad button to accept calibration/drift check target
# You need a supported gamepad/button box that is connected to the Host PC
el_tracker.sendCommand("button_function 5 'accept_target_fixation'")

# get the native screen resolution used by PsychoPy
scn_width, scn_height = win.size
# resolution fix for Mac retina displays
if 'Darwin' in platform.system():
    if use_retina:
        scn_width = int(scn_width/2.0)
        scn_height = int(scn_height/2.0)

# Pass the display pixel coordinates (left, top, right, bottom) to the tracker
# see the EyeLink Installation Guide, "Customizing Screen Settings"
el_coords = "screen_pixel_coords = 0 0 %d %d" % (scn_width - 1, scn_height - 1)
el_tracker.sendCommand(el_coords)

# Write a DISPLAY_COORDS message to the EDF file
# Data Viewer needs this piece of info for proper visualization, see Data
# Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
dv_coords = "DISPLAY_COORDS  0 0 %d %d" % (scn_width - 1, scn_height - 1)
el_tracker.sendMessage(dv_coords)

# Configure a graphics environment (genv) for tracker calibration
genv = EyeLinkCoreGraphicsPsychoPy(el_tracker, win)
print(genv)  # print out the version number of the CoreGraphics library

# Set background and foreground colors for the calibration target
# in PsychoPy, (-1, -1, -1)=black, (1, 1, 1)=white, (0, 0, 0)=mid-gray
foreground_color = (-1, -1, -1)
background_color = win.color
genv.setCalibrationColors(foreground_color, background_color)

# Set up the calibration target
#
# The target could be a "circle" (default), a "picture", a "movie" clip,
# or a rotating "spiral". To configure the type of calibration target, set
# genv.setTargetType to "circle", "picture", "movie", or "spiral", e.g.,
# genv.setTargetType('picture')
#
# Use gen.setPictureTarget() to set a "picture" target
#genv.setPictureTarget(os.path.join('images', 'fixTarget.bmp'))
#
# Use genv.setMovieTarget() to set a "movie" target
genv.setTargetType('movie')
genv.setMoiveTarget(os.path.join('videos', 'calibVid.mov'))

# Use the default calibration target ('circle')
#genv.setTargetType('spiral')

# Configure the size of the calibration target (in pixels)
# this option applies only to "circle" and "spiral" targets
#genv.setTargetSize(24)

# Beeps to play during calibration, validation and drift correction
# parameters: target, good, error
#     target -- sound to play when target moves
#     good -- sound to play on successful operation
#     error -- sound to play on failure or interruption
# Each parameter could be ''--default sound, 'off'--no sound, or a wav file
genv.setCalibrationSounds('', '', '')

# resolution fix for macOS retina display issues
if use_retina:
    genv.fixMacRetinaDisplay()

# Request Pylink to use the PsychoPy window we opened above for calibration
pylink.openGraphicsEx(genv)


# define a few helper functions for trial handling


def clear_screen(win):
    """ clear up the PsychoPy window"""

    win.fillColor = genv.getBackgroundColor()
    win.flip()


def show_msg(win, text, wait_for_keypress=True):
    """ Show task instructions on screen"""

    msg = visual.TextStim(win, text,
                          color=genv.getForegroundColor(),
                          wrapWidth=scn_width/2)
    clear_screen(win)
    msg.draw()
    win.flip()

    # wait indefinitely, terminates upon any key press
    if wait_for_keypress:
        event.waitKeys()
        clear_screen(win)


def terminate_task():
    """ Terminate the task gracefully and retrieve the EDF data file

    file_to_retrieve: The EDF on the Host that we would like to download
    win: the current window used by the experimental script
    """

    el_tracker = pylink.getEYELINK()

    if el_tracker.isConnected():
        # Terminate the current trial first if the task terminated prematurely
        error = el_tracker.isRecording()
        if error == pylink.TRIAL_OK:
            abort_trial()

        # Put tracker in Offline mode
        el_tracker.setOfflineMode()

        # Clear the Host PC screen and wait for 500 ms
        el_tracker.sendCommand('clear_screen 0')
        pylink.msecDelay(500)

        # Close the edf data file on the Host
        el_tracker.closeDataFile()

        # Show a file transfer message on the screen
        msg = 'EDF data is transferring from EyeLink Host PC...'
        show_msg(win, msg, wait_for_keypress=False)

        # Download the EDF data file from the Host PC to a local data folder
        # parameters: source_file_on_the_host, destination_file_on_local_drive
        local_edf = os.path.join(session_folder, session_identifier + '.EDF')
        try:
            el_tracker.receiveDataFile(edf_file, local_edf)
        except RuntimeError as error:
            print('ERROR:', error)

        # Close the link to the tracker.
        el_tracker.close()

    # close the PsychoPy window
    win.close()

    # quit PsychoPy
    core.quit()
    sys.exit()


def abort_trial():
    """Ends recording """

    el_tracker = pylink.getEYELINK()

    # Stop recording
    if el_tracker.isRecording():
        # add 100 ms to catch final trial events
        pylink.pumpDelay(100)
        el_tracker.stopRecording()

    # clear the screen
    clear_screen(win)
    # Send a message to clear the Data Viewer screen
    bgcolor_RGB = (116, 116, 116)
    el_tracker.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)

    # send a message to mark trial end
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_ERROR)
if dummy_mode:
    task_msg = 'Cannot run the script in Dummy mode,\n' + \
        'Press ENTER to quit the script'
else:
    task_msg = 'On each trial, look at the cross to start,\n' + \
        'then press SPACEBAR to end a trial\n' + \
        '\nPress Ctrl-C if you need to quit the task early\n' + \
        '\nNow, press ENTER twice to calibrate tracker'
show_msg(win, task_msg)

# Terminate the task if running in Dummy Mode
if dummy_mode:
    print('ERROR: This task requires real-time gaze data.\n' +
          'It cannot run in Dummy mode (with no tracker connection).')
    terminate_task()
else:
    try:
        el_tracker.doTrackerSetup()
    except RuntimeError as err:
        print('ERROR:', err)
        el_tracker.exitCalibration()
        
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess
# --- Setup input devices ---
ioConfig = {}

# Setup iohub keyboard
ioConfig['Keyboard'] = dict(use_keymap='psychopy')

ioSession = '1'
if 'session' in expInfo:
    ioSession = str(expInfo['session'])
ioServer = io.launchHubServer(window=win, **ioConfig)
eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard(backend='iohub')

pic_x, pic_y = 0, 0
cross_x, cross_y = 0, 0

# --- Initialize components for Routine "standby" ---
standby_text = visual.TextStim(win=win, name='standby_text',
    text='Welcome!\n\npress SPACE to start!',
    font='Open Sans',
    pos=(0, 0), height=100, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
standby_key_resp = keyboard.Keyboard()

# --- Initialize components for Routine "selfpaced_start_m" ---
ssm_cross = visual.TextStim(win=win, name='ssm_cross',
    text='+',
    font='Open Sans',
    pos=(0, 0), height=100, wrapWidth=None, ori=0.0, 
    color='green', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# --- Initialize components for Routine "forcedmemory" ---
momery_image = visual.ImageStim(
    win=win,
    name='momery_image', 
    image=target_pic, mask=None, anchor='center',
    ori=0.0, pos=(0, 0),size=(pic_width,pic_height),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)

# --- Initialize components for Routine "delay" ---
delay_cross = visual.TextStim(win=win, name='delay_cross',
    text='+',
    font='Open Sans',
    pos=(0, 0), height=100, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
memory_iti_cross = visual.TextStim(win=win, name='memory_iti_cross',
    text='+',
    font='Open Sans',
    pos=(0, 0), height=100, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
    
# --- Initialize components for Routine "forcedchoice" ---
top_image = visual.ImageStim(
    win=win,
    name='top_image', 
    image=target_pic, mask=None, anchor='center',
    ori=0.0, pos=(0*0.5*scn_width, -0.35*0.5*scn_height), size=(pic_width,pic_height),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
top_cross = visual.TextStim(win=win, name='top_cross',
    text='+',
    font='Open Sans',
    pos=(0*0.5*scn_width,-0.85*0.5*scn_height), height=100, wrapWidth=None, ori=0.0, 
    color='red', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
left_image = visual.ImageStim(
    win=win,
    name='left_image', 
    image=target_pic, mask=None, anchor='center',
    ori=0.0, pos=(-0.5*0.5*scn_width, 0.65*0.5*scn_height), size=(pic_width,pic_height),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
left_cross = visual.TextStim(win=win, name='left_cross',
    text='+',
    font='Open Sans',
    pos=(-0.5*0.5*scn_width, 0.15*0.5*scn_height), height=100, wrapWidth=None, ori=0.0, 
    color='red', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
right_image = visual.ImageStim(
    win=win,
    name='right_image', 
    image=target_pic, mask=None, anchor='center',
    ori=0.0, pos=(0.5*0.5*scn_width, 0.65*0.5*scn_height),size=(pic_width,pic_height),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
right_cross = visual.TextStim(win=win, name='right_cross',
    text='+',
    font='Open Sans',
    pos=(0.5*0.5*scn_width, 0.15*0.5*scn_height), height=100, wrapWidth=None, ori=0.0, 
    color='red', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# --- Initialize components for Routine "reward" ---
r_correct_image = visual.ImageStim(
    win=win,
    name='r_correct_image', 
    image=target_pic, mask=None, anchor='center',
    ori=0.0, pos=(pic_x*0.5*scn_width, pic_y*0.5*scn_height),size=(pic_width,pic_height),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
r_correct_cross = visual.TextStim(win=win, name='r_correct_cross',
    text='+',
    font='Open Sans',
    pos=(cross_x*0.5*scn_width, cross_y*0.5*scn_height), height=100, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
r_cross = visual.TextStim(win=win, name='r_cross',
    text='+',
    font='Open Sans',
    pos=(0, 0), height=100, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);

# --- Initialize components for Routine "punishment" ---
p_corret_image = visual.ImageStim(
    win=win,
    name='p_corret_image', 
    image=target_pic, mask=None, anchor='center',
    ori=0.0, pos=(pic_x*0.5*scn_width, pic_y), 
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
p_corret_cross = visual.TextStim(win=win, name='p_corret_cross',
    text='+',
    font='Open Sans',
    pos=(cross_x*0.5*scn_width, cross_y), height=100, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
p_cross = visual.TextStim(win=win, name='p_cross',
    text='+',
    font='Open Sans',
    pos=(0, 0), height=100, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);

# --- Initialize components for Routine "jitter_iti" ---
jitter_cross = visual.TextStim(win=win, name='jitter_cross',
    text='+',
    font='Open Sans',
    pos=(0, 0), height=100, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# --- Initialize components for Routine "finish" ---
finish_text = visual.TextStim(win=win, name='finish_text',
    text='finish!\n\npress SPACE to quit!',
    font='Open Sans',
    pos=(0, 0), height=100, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
finish_key_resp = keyboard.Keyboard()

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine 

# --- Prepare to start Routine "standby" ---
continueRoutine = True
routineForceEnded = False
# update component parameters for each repeat
standby_key_resp.keys = []
standby_key_resp.rt = []
_standby_key_resp_allKeys = []
# keep track of which components have finished
standbyComponents = [standby_text, standby_key_resp]
for thisComponent in standbyComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "standby" ---
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *standby_text* updates
    if standby_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        standby_text.frameNStart = frameN  # exact frame index
        standby_text.tStart = t  # local t and not account for scr refresh
        standby_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(standby_text, 'tStartRefresh')  # time at next scr refresh
        standby_text.setAutoDraw(True)
    
    # *standby_key_resp* updates
    if standby_key_resp.status == NOT_STARTED and t >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        standby_key_resp.frameNStart = frameN  # exact frame index
        standby_key_resp.tStart = t  # local t and not account for scr refresh
        standby_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(standby_key_resp, 'tStartRefresh')  # time at next scr refresh
        standby_key_resp.status = STARTED
        # keyboard checking is just starting
        standby_key_resp.clock.reset()  # now t=0
    if standby_key_resp.status == STARTED:
        theseKeys = standby_key_resp.getKeys(keyList=['space'], waitRelease=False)
        _standby_key_resp_allKeys.extend(theseKeys)
        if len(_standby_key_resp_allKeys):
            standby_key_resp.keys = _standby_key_resp_allKeys[-1].name  # just the last key pressed
            standby_key_resp.rt = _standby_key_resp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in standbyComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "standby" ---
for thisComponent in standbyComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if standby_key_resp.keys in ['', [], None]:  # No response was made
    standby_key_resp.keys = None
thisExp.addData('standby_key_resp.keys',standby_key_resp.keys)
if standby_key_resp.keys != None:  # we had a response
    thisExp.addData('standby_key_resp.rt', standby_key_resp.rt)
thisExp.nextEntry()
# the Routine "standby" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()


for loop_times_i in range(0,total_loop_times):
    #generate new list every repeat
    task_list = []
    for i in range(0,fc_loop_times+1):
        list_temp = df[['target_pic','pic_1','pic_2','pic_3']][list_index : list_index + list_length]
        list_temp = list_temp.reindex(np.random.permutation(list_temp.index))
        list_temp = list_temp.reset_index()
        task_list.append(list_temp)
    
    response = 0
    reward_ornot = 0
    jitter_iti = 0

    sigma = 0.5
    mu = 1.5
    r = sigma * np.random.randn(fc_loop_times+50) + mu
    r_0 = [ x for x in r if x>0]
    # --- Prepare to start Routine "selfpaced_start_m" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    # keep track of which components have finished
    selfpaced_start_mComponents = [ssm_cross]
    for thisComponent in selfpaced_start_mComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1

    el_tracker = pylink.getEYELINK()
    el_tracker.setOfflineMode()
    el_tracker.startRecording(1, 1, 1, 1)
    pylink.pumpDelay(100)

    event.clearEvents()  # clear cached PsychoPy events
    new_sample = None
    new_sample_top = None
    new_sample_left = None
    new_sample_right = None

    old_sample = None
    old_sample_top = None
    old_sample_left = None
    old_sample_right = None

    trigger_fired = False
    trigger_fired_top = False
    trigger_fired_left = False
    trigger_fired_right = False

    in_hit_region = False
    in_hit_region_top = False
    in_hit_region_left = False
    in_hit_region_right = False

    trigger_start_time = core.getTime()
    trigger_start_time_top = core.getTime()
    trigger_start_time_left = core.getTime()
    trigger_start_time_right = core.getTime()

    # determine which eye(s) is/are available
    # 0- left, 1-right, 2-binocular
    eye_used = el_tracker.eyeAvailable()
    if eye_used == 1:
        el_tracker.sendMessage("EYE_USED 1 RIGHT")
    elif eye_used == 0 or eye_used == 2:
        el_tracker.sendMessage("EYE_USED 0 LEFT")
        eye_used = 0
    else:
        print("Error in getting the eye information!")
    # fire the trigger following a 300-ms gaze
    gaze_start = -1
    gaze_start_top = -1
    gaze_start_left = -1
    gaze_start_right = -1

    trial_index = 0
    el_tracker.sendMessage('selfpaced_start_m_start')

    # --- Run Routine "selfpaced_start_m" ---
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # *ssm_cross* updates
        if ssm_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            ssm_cross.frameNStart = frameN  # exact frame index
            ssm_cross.tStart = t  # local t and not account for scr refresh
            ssm_cross.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(ssm_cross, 'tStartRefresh')  # time at next scr refresh
            ssm_cross.setAutoDraw(True)
        
        # Do we have a sample in the sample buffer?
        # and does it differ from the one we've seen before?
        new_sample = el_tracker.getNewestSample()

        if new_sample is not None:
            if old_sample is not None:
                if new_sample.getTime() != old_sample.getTime():
                    # check if the new sample has data for the eye
                    # currently being tracked; if so, we retrieve the current
                    # gaze position and PPD (how many pixels correspond to 1
                    # deg of visual angle, at the current gaze position)
                    if eye_used == 1 and new_sample.isRightSample():
                        g_x, g_y = new_sample.getRightEye().getGaze()
                    if eye_used == 0 and new_sample.isLeftSample():
                        g_x, g_y = new_sample.getLeftEye().getGaze()
                    # break the while loop if the current gaze position is
                    # in a 120 x 120 pixels region around the screen centered
                    fix_x, fix_y = (scn_width/2.0, scn_height/2.0)
                    if fabs(g_x - fix_x) < cross_size and fabs(g_y - fix_y) < cross_size:
                        # record gaze start time
                        if not in_hit_region:
                            if gaze_start == -1:
                                gaze_start = core.getTime()
                                in_hit_region = True
                        # check the gaze duration and fire
                        if in_hit_region:
                            gaze_dur = core.getTime() - gaze_start
                            if gaze_dur > minimum_duration:
                                trigger_fired = True
                    else:  # gaze outside the hit region, reset variables
                        in_hit_region = False
                        gaze_start = -1
            
            # update the "old_sample"
            old_sample = new_sample
    #       print(new_sample)
    #       print(old_sample)
    #       print('-----------------')
    #       print(trigger_fired)
    #       print(fix_x,fix_y)
    #       print(g_x, g_y)
    #       print('-----------------')
        if trigger_fired == True:
            trigger_fired = False
            break
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            el_tracker.stopRecording()
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in selfpaced_start_mComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    el_tracker.sendMessage('selfpaced_start_m_end')
    # --- Ending Routine "selfpaced_start_m" ---
    for thisComponent in selfpaced_start_mComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "selfpaced_start_m" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    for i in range(0, list_length):
        # --- Prepare to start Routine "forcedmemory" ---
        target_pic = task_list[0]['target_pic'][i]
        target_pic = 'F:\\Pic_TOJ\\pic_pool_step_3_formal\\' + str(target_pic)
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        momery_image.setImage(target_pic)
        # keep track of which components have finished
        forcedmemoryComponents = [momery_image]
        for thisComponent in forcedmemoryComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        memory_image_position = 'F:\\Pic_TOJ\\pic_pool_step_3_formal\\' + str(task_list[0]['target_pic'][i])
        imgload_msg_memory = '!V IMGLOAD CENTER %s %d %d %d %d' % (memory_image_position,
                                                                int(scn_width * 0.5),
                                                                int(scn_height * 0.5),
                                                                int(pic_width),
                                                                int(pic_height))
        


        memory_center_x = 1280
        memory_center_y = 720  
        memory_roi_left, memory_roi_top, memory_roi_right, memory_roi_bottom = round(memory_center_x - pic_width*0.5*(1+roi_range_margin)),round(memory_center_y - pic_height*0.5*(1+roi_range_margin)), round(memory_center_x + pic_width*0.5*(1+roi_range_margin)),round(memory_center_y + pic_height*0.5*(1+roi_range_margin))
        memory_roi_message = '!V IAREA RECTANGLE 9 ' + str(memory_roi_left) + ' '+str(memory_roi_top) + ' '+str(memory_roi_right) + ' '+str(memory_roi_bottom) + ' memory_roi'
        
        
        el_tracker.sendMessage('forcedmemory_start_pic_'+str(i))
        el_tracker.sendMessage(imgload_msg_memory)
        el_tracker.sendMessage(memory_roi_message)
        # --- Run Routine "forcedmemory" ---
        while continueRoutine:
            status_msg = 'forced MEMORY stage, Total loop %d/%d' % (loop_times_i,total_loop_times)
            el_tracker.sendCommand("record_status_message '%s'" % status_msg)
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *momery_image* updates
            if momery_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                momery_image.frameNStart = frameN  # exact frame index
                momery_image.tStart = t  # local t and not account for scr refresh
                momery_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(momery_image, 'tStartRefresh')  # time at next scr refresh
                momery_image.setAutoDraw(True)
            if momery_image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > momery_image.tStartRefresh + memory_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    momery_image.tStop = t  # not accounting for scr refresh
                    momery_image.frameNStop = frameN  # exact frame index
                    momery_image.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                el_tracker.stopRecording()
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in forcedmemoryComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        el_tracker.sendMessage('!V CLEAR 0 0 0')
        el_tracker.sendMessage('forcedmemory_end_pic_'+str(i))
        
        
        # --- Ending Routine "forcedmemory" ---
        for thisComponent in forcedmemoryComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # the Routine "forcedmemory" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    

        # --- Prepare to start Routine "memory_iti" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        # keep track of which components have finished
        memory_itiComponents = [memory_iti_cross]
        for thisComponent in memory_itiComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        el_tracker.sendMessage('memory_iti_start')
        # --- Run Routine "memory_iti" ---
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *delay_cross* updates
            if memory_iti_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                memory_iti_cross.frameNStart = frameN  # exact frame index
                memory_iti_cross.tStart = t  # local t and not account for scr refresh
                memory_iti_cross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(memory_iti_cross, 'tStartRefresh')  # time at next scr refresh
                memory_iti_cross.setAutoDraw(True)
            if memory_iti_cross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > memory_iti_cross.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    memory_iti_cross.tStop = t  # not accounting for scr refresh
                    memory_iti_cross.frameNStop = frameN  # exact frame index
                    memory_iti_cross.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                el_tracker.stopRecording()
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in memory_itiComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
                
        el_tracker.sendMessage('memory_iti_end')

        # --- Ending Routine "delay" ---
        for thisComponent in memory_itiComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # the Routine "delay" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
    el_tracker.sendMessage('!V CLEAR 0 0 0')
    # --- Prepare to start Routine "delay" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    # keep track of which components have finished
    delayComponents = [delay_cross]
    for thisComponent in delayComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    el_tracker.sendMessage('delay_start')
    # --- Run Routine "delay" ---
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *delay_cross* updates
        if delay_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            delay_cross.frameNStart = frameN  # exact frame index
            delay_cross.tStart = t  # local t and not account for scr refresh
            delay_cross.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(delay_cross, 'tStartRefresh')  # time at next scr refresh
            delay_cross.setAutoDraw(True)
        if delay_cross.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > delay_cross.tStartRefresh + delay_duration-frameTolerance:
                # keep track of stop time/frame for later
                delay_cross.tStop = t  # not accounting for scr refresh
                delay_cross.frameNStop = frameN  # exact frame index
                delay_cross.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            el_tracker.stopRecording()
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in delayComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
            
    el_tracker.sendMessage('delay_end')

    # --- Ending Routine "delay" ---
    for thisComponent in delayComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "delay" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()

    # set up handler to look after randomisation of conditions etc
    forcedchoice_loop = data.TrialHandler(nReps=fc_loop_times, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='forcedchoice_loop')
    thisExp.addLoop(forcedchoice_loop)  # add the loop to the experiment
    thisForcedchoice_loop = forcedchoice_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisForcedchoice_loop.rgb)
    if thisForcedchoice_loop != None:
        for paramName in thisForcedchoice_loop:
            exec('{} = thisForcedchoice_loop[paramName]'.format(paramName))
            
    reward_position = None
    fix_position = None
    target_pic_position = None
    start_t=0
    end_t=0
    choice_rt = 0
    fc_current_loop = 0
    for thisForcedchoice_loop in forcedchoice_loop:
        currentLoop = forcedchoice_loop
        # abbreviate parameter names if possible (e.g. rgb = thisForcedchoice_loop.rgb)
        if thisForcedchoice_loop != None:
            for paramName in thisForcedchoice_loop:
                exec('{} = thisForcedchoice_loop[paramName]'.format(paramName))
        for i in range(0,list_length):
            trial_index = trial_index + 1
            #eyelink messages 
            status_msg = 'TRIAL number in 3afc %d/%d, Total loop %d/%d ' % (trial_index,fc_loop_times*list_length,loop_times_i,total_loop_times)
            el_tracker.sendCommand("record_status_message '%s'" % status_msg)   
            pic_top, pic_left, pic_right = random.sample([task_list[fc_current_loop+1]['pic_1'][i],task_list[fc_current_loop+1]['pic_2'][i],task_list[fc_current_loop+1]['pic_3'][i]],3)
            #change your path here
            target_pic = 'F:\\Pic_TOJ\\pic_pool_step_3_formal\\' + str(task_list[fc_current_loop+1]['target_pic'][i])
            pic_top = 'F:\\Pic_TOJ\\pic_pool_step_3_formal\\'+str(pic_top)
            pic_left = 'F:\\Pic_TOJ\\pic_pool_step_3_formal\\'+str(pic_left)
            pic_right = 'F:\\Pic_TOJ\\pic_pool_step_3_formal\\'+str(pic_right)
                
            if pic_top == target_pic :
                reward_position = 'bottom'
                target_pic_position = 'bottom'
                el_tracker.sendMessage('target_position : bottom')
            if pic_left == target_pic :
                reward_position = 'left'
                target_pic_position = 'left'
                el_tracker.sendMessage('target_position : left')
            if pic_right == target_pic :
                reward_position = 'right'
                target_pic_position = 'right'
                el_tracker.sendMessage('target_position : right')
            # --- Prepare to start Routine "forcedchoice" ---
            continueRoutine = True
            routineForceEnded = False
            # update component parameters for each repeat
            top_image.setImage(pic_top)

            left_image.setImage(pic_left)

            right_image.setImage(pic_right)

            # keep track of which components have finished
            forcedchoiceComponents = [top_image, top_cross, left_image, left_cross, right_image, right_cross]
            for thisComponent in forcedchoiceComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            

            top_image_position = pic_top
            imgload_msg_top = '!V IMGLOAD CENTER %s %d %d %d %d' % (top_image_position,
                                                                int((1 + 0) * scn_width * 0.5),
                                                                int((1 + 0.35) * scn_height * 0.5),
                                                                int(pic_width),
                                                                int(pic_height))
            
            left_image_position = pic_left
            imgload_msg_left = '!V IMGLOAD CENTER %s %d %d %d %d' % (left_image_position,
                                                                int((1 - 0.5) * scn_width * 0.5),
                                                                int((1 - 0.65) * scn_height * 0.5),
                                                                int(pic_width),
                                                                int(pic_height))
            
            right_image_position = pic_right
            imgload_msg_right = '!V IMGLOAD CENTER %s %d %d %d %d' % (right_image_position,
                                                                int((1 + 0.5) * scn_width * 0.5),
                                                                int((1 - 0.65) * scn_height * 0.5),
                                                                int(pic_width),
                                                                int(pic_height))
            
            # left top right bottom 
            top_center_x = 1280
            top_center_y = 972
            top_roi_left, top_roi_top, top_roi_right, top_roi_bottom = round(top_center_x - pic_width*0.5*(1+roi_range_margin)), round(top_center_y - pic_height*0.5*(1+roi_range_margin)), round(top_center_x + pic_width*0.5*(1+roi_range_margin)),round(top_center_y + pic_height*0.5*(1+roi_range_margin))
            left_center_x = 640
            left_center_y = 252   
            left_roi_left, left_roi_top, left_roi_right, left_roi_bottom = round(left_center_x - pic_width*0.5*(1+roi_range_margin)),round(left_center_y - pic_height*0.5*(1+roi_range_margin)), round(left_center_x + pic_width*0.5*(1+roi_range_margin)),round(left_center_y + pic_height*0.5*(1+roi_range_margin))
            right_center_x = 1920
            right_center_y = 252
            right_roi_left, right_roi_top, right_roi_right, right_roi_bottom = round(right_center_x - pic_width*0.5*(1+roi_range_margin)),round(right_center_y - pic_height*0.5*(1+roi_range_margin)), round(right_center_x + pic_width*0.5*(1+roi_range_margin)),round(right_center_y + pic_height*0.5*(1+roi_range_margin))
            top_roi_message = '!V IAREA RECTANGLE 1 ' + str(top_roi_left) + ' '+str(top_roi_top) + ' '+str(top_roi_right) + ' '+str(top_roi_bottom) + ' bottom_roi'
            left_roi_message = '!V IAREA RECTANGLE 2 ' + str(left_roi_left) + ' '+str(left_roi_top) + ' '+str(left_roi_right) + ' '+str(left_roi_bottom) + ' left_roi'
            right_roi_message = '!V IAREA RECTANGLE 3 ' + str(right_roi_left) + ' '+str(right_roi_top) + ' '+str(right_roi_right) + ' '+str(right_roi_bottom) + ' right_roi'

            #<x_start> <y_start> <x_end> <y_end> 
            top_cross_x, top_cross_y = 1280, 1332
            left_cross_x, left_cross_y = 640, 612
            right_cross_x, right_cross_y = 1920, 612
            top_cross_left,top_cross_top,top_cross_right,top_cross_bottom = round(top_cross_x - cross_size), round(top_cross_y - cross_size), round(top_cross_x + cross_size),round(top_cross_y + cross_size)
            left_cross_left,left_cross_top,left_cross_right,left_cross_bottom = round(left_cross_x - cross_size), round(left_cross_y - cross_size), round(left_cross_x + cross_size),round(left_cross_y + cross_size)
            right_cross_left,right_cross_top,right_cross_right,right_cross_bottom = round(right_cross_x - cross_size), round(right_cross_y - cross_size), round(right_cross_x + cross_size),round(right_cross_y + cross_size)

            start_t = routineTimer.getTime()
            
            el_tracker.sendMessage('forcedchoice_start_'+str(i))
            el_tracker.sendMessage(imgload_msg_top)
            el_tracker.sendMessage(imgload_msg_left)
            el_tracker.sendMessage(imgload_msg_right)
            el_tracker.sendMessage(top_roi_message)
            el_tracker.sendMessage(left_roi_message)
            el_tracker.sendMessage(right_roi_message)
            el_tracker.sendMessage('!V DRAWBOX 0 255 0 '+str(top_cross_left)+' '+ str(top_cross_top)+' '+ str(top_cross_right)+' '+str(top_cross_bottom))
            el_tracker.sendMessage('!V DRAWBOX 0 255 0 '+str(left_cross_left)+' '+ str(left_cross_top)+' '+ str(left_cross_right)+' '+str(left_cross_bottom))
            el_tracker.sendMessage('!V DRAWBOX 0 255 0 '+str(right_cross_left)+' '+ str(right_cross_top)+' '+ str(right_cross_right)+' '+str(right_cross_bottom))
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *top_image* updates
                if top_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    top_image.frameNStart = frameN  # exact frame index
                    top_image.tStart = t  # local t and not account for scr refresh
                    top_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(top_image, 'tStartRefresh')  # time at next scr refresh
                    top_image.setAutoDraw(True)
                if top_image.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > top_image.tStartRefresh + choice_duration-frameTolerance:
                        # keep track of stop time/frame for later
                        top_image.tStop = t  # not accounting for scr refresh
                        top_image.frameNStop = frameN  # exact frame index
                        top_image.setAutoDraw(False)
                        fix_position = 'abort'
                # *top_cross* updates
                if top_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    top_cross.frameNStart = frameN  # exact frame index
                    top_cross.tStart = t  # local t and not account for scr refresh
                    top_cross.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(top_cross, 'tStartRefresh')  # time at next scr refresh
                    top_cross.setAutoDraw(True)
                if top_cross.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > top_cross.tStartRefresh + choice_duration-frameTolerance:
                        # keep track of stop time/frame for later
                        top_cross.tStop = t  # not accounting for scr refresh
                        top_cross.frameNStop = frameN  # exact frame index
                        top_cross.setAutoDraw(False)
                
                # *left_image* updates
                if left_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    left_image.frameNStart = frameN  # exact frame index
                    left_image.tStart = t  # local t and not account for scr refresh
                    left_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(left_image, 'tStartRefresh')  # time at next scr refresh
                    left_image.setAutoDraw(True)
                if left_image.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > left_image.tStartRefresh + choice_duration-frameTolerance:
                        # keep track of stop time/frame for later
                        left_image.tStop = t  # not accounting for scr refresh
                        left_image.frameNStop = frameN  # exact frame index
                        left_image.setAutoDraw(False)
                
                # *left_cross* updates
                if left_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    left_cross.frameNStart = frameN  # exact frame index
                    left_cross.tStart = t  # local t and not account for scr refresh
                    left_cross.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(left_cross, 'tStartRefresh')  # time at next scr refresh
                    left_cross.setAutoDraw(True)
                if left_cross.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > left_cross.tStartRefresh + choice_duration-frameTolerance:
                        # keep track of stop time/frame for later
                        left_cross.tStop = t  # not accounting for scr refresh
                        left_cross.frameNStop = frameN  # exact frame index
                        left_cross.setAutoDraw(False)
                
                # *right_image* updates
                if right_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    right_image.frameNStart = frameN  # exact frame index
                    right_image.tStart = t  # local t and not account for scr refresh
                    right_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(right_image, 'tStartRefresh')  # time at next scr refresh
                    right_image.setAutoDraw(True)
                if right_image.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > right_image.tStartRefresh + choice_duration-frameTolerance:
                        # keep track of stop time/frame for later
                        right_image.tStop = t  # not accounting for scr refresh
                        right_image.frameNStop = frameN  # exact frame index
                        right_image.setAutoDraw(False)
                
                # *right_cross* updates
                if right_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    right_cross.frameNStart = frameN  # exact frame index
                    right_cross.tStart = t  # local t and not account for scr refresh
                    right_cross.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(right_cross, 'tStartRefresh')  # time at next scr refresh
                    right_cross.setAutoDraw(True)
                if right_cross.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > right_cross.tStartRefresh + choice_duration-frameTolerance:
                        # keep track of stop time/frame for later
                        right_cross.tStop = t  # not accounting for scr refresh
                        right_cross.frameNStop = frameN  # exact frame index
                        right_cross.setAutoDraw(False)
                # Do we have a sample in the sample buffer?
                # and does it differ from the one we've seen before?
                #top

                new_sample_top = el_tracker.getNewestSample()
                if new_sample_top is not None:
                    if old_sample_top is not None:
                        if new_sample_top.getTime() != old_sample_top.getTime():
                            # check if the new sample has data for the eye
                            # currently being tracked; if so, we retrieve the current
                            # gaze position and PPD (how many pixels correspond to 1
                            # deg of visual angle, at the current gaze position)
                            if eye_used == 1 and new_sample_top.isRightSample():
                                g_x_top, g_y_top = new_sample_top.getRightEye().getGaze()
                            if eye_used == 0 and new_sample_top.isLeftSample():
                                g_x_top, g_y_top = new_sample_top.getLeftEye().getGaze()
                            # break the while loop if the current gaze position is
                            # in a 120 x 120 pixels region around the screen centered
                            
                            fix_x_top, fix_y_top = ((1 + 0) * scn_width * 0.5, (1 + 0.85) * scn_height * 0.5)
                            if fabs(g_x_top - fix_x_top) < cross_size and fabs(g_y_top - fix_y_top) < cross_size:
                                # record gaze start time
                                if not in_hit_region_top:
                                    if gaze_start_top == -1:
                                        gaze_start_top = core.getTime()
                                        in_hit_region_top = True
                                # check the gaze duration and fire
                                if in_hit_region_top:
                                    gaze_dur_top = core.getTime() - gaze_start_top
                                    if gaze_dur_top > minimum_duration:
                                        trigger_fired_top = True
                            else:  # gaze outside the hit region, reset variables
                                in_hit_region_top = False
                                gaze_start_top = -1
                #left                
                new_sample_left = el_tracker.getNewestSample()
                if new_sample_left is not None:
                    if old_sample_left is not None:
                        if new_sample_left.getTime() != old_sample_left.getTime():
                            # check if the new sample has data for the eye
                            # currently being tracked; if so, we retrieve the current
                            # gaze position and PPD (how many pixels correspond to 1
                            # deg of visual angle, at the current gaze position)
                            if eye_used == 1 and new_sample_left.isRightSample():
                                g_x_left, g_y_left = new_sample_left.getRightEye().getGaze()
                            if eye_used == 0 and new_sample_left.isLeftSample():
                                g_x_left, g_y_left = new_sample_left.getLeftEye().getGaze()
                            # break the while loop if the current gaze position is
                            # in a 120 x 120 pixels region around the screen centered  
                            fix_x_left, fix_y_left = ((1 - 0.5) * scn_width * 0.5, (1 - 0.15) * scn_height * 0.5)
                            if fabs(g_x_left - fix_x_left) < cross_size and fabs(g_y_left - fix_y_left) < cross_size:
                                # record gaze start time
                                if not in_hit_region_left:
                                    if gaze_start_left == -1:
                                        gaze_start_left = core.getTime()
                                        in_hit_region_left = True
                                # check the gaze duration and fire
                                if in_hit_region_left:
                                    gaze_dur_left = core.getTime() - gaze_start_left
                                    if gaze_dur_left > minimum_duration:
                                        trigger_fired_left = True
                            else:  # gaze outside the hit region, reset variables
                                in_hit_region_left = False
                                gaze_start_left = -1
                #right
                new_sample_right = el_tracker.getNewestSample()
                if new_sample_right is not None:
                    if old_sample_right is not None:
                        if new_sample_right.getTime() != old_sample_right.getTime():
                            # check if the new sample has data for the eye
                            # currently being tracked; if so, we retrieve the current
                            # gaze position and PPD (how many pixels correspond to 1
                            # deg of visual angle, at the current gaze position)
                            if eye_used == 1 and new_sample_right.isRightSample():
                                g_x_right, g_y_right = new_sample_right.getRightEye().getGaze()
                            if eye_used == 0 and new_sample_right.isLeftSample():
                                g_x_right, g_y_right = new_sample_right.getLeftEye().getGaze()
                            # break the while loop if the current gaze position is
                            # in a 120 x 120 pixels region around the screen centered      
                            fix_x_right, fix_y_right = ((1 + 0.5) * scn_width * 0.5, (1 - 0.15) * scn_height * 0.5)
                            if fabs(g_x_right - fix_x_right) < cross_size and fabs(g_y_right - fix_y_right) < cross_size:
                                # record gaze start time
                                if not in_hit_region_right:
                                    if gaze_start_right == -1:
                                        gaze_start_right = core.getTime()
                                        in_hit_region_right = True
                                # check the gaze duration and fire
                                if in_hit_region_right:
                                    gaze_dur_right = core.getTime() - gaze_start_right
                                    if gaze_dur_right > minimum_duration:
                                        trigger_fired_right = True
                            else:  # gaze outside the hit region, reset variables
                                in_hit_region_right = False
                                gaze_start_right = -1
                    # update the "old_sample"
                    old_sample_top = new_sample_top
                    old_sample_left = new_sample_left
                    old_sample_right = new_sample_right
            #       print(new_sample)
            #       print(old_sample)
            #       print('-----------------')
            #       print(trigger_fired)
            #       print(fix_x,fix_y)
            #       print(g_x, g_y)
            #       print('-----------------')
                if trigger_fired_top == True:
                    trigger_fired_top = False
                    fix_position = 'bottom'
                    pic_chosen = pic_top
                    routineForceEnded = True
                    break
                if trigger_fired_left == True:
                    trigger_fired_left = False
                    fix_position = 'left'
                    pic_chosen = pic_left
                    routineForceEnded = True
                    break
                if trigger_fired_right == True:
                    trigger_fired_right = False
                    fix_position = 'right'
                    pic_chosen = pic_right
                    routineForceEnded = True
                    break
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    el_tracker.stopRecording()
                    core.quit()
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in forcedchoiceComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            el_tracker.sendMessage('!V CLEAR 0 0 0')
            el_tracker.sendMessage('forcedchoice_end_'+str(i))        
            
            end_t = routineTimer.getTime()
            choice_rt = end_t - start_t - 1
            # --- Ending Routine "forcedchoice" ---
            for thisComponent in forcedchoiceComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            routineTimer.reset()
        #    print(reward_position,fix_position)
        #    print('--------------------')
            # --- Prepare to start Routine "reward" ---
            continueRoutine = True
            routineForceEnded = False
            if reward_position == fix_position:
                ser=serial.Serial('COM4',9600)
                ser.write('JON'.encode())
                ser.close()
                response = 1
                reward_ornot = 1
                # update component parameters for each repeat
                if reward_position == 'bottom':
                    cross_x = 0 * scn_width * 0.5
                    cross_y = -0.85 * scn_height * 0.5
                    pic_x = 0 * scn_width * 0.5
                    pic_y = -0.35 * scn_height * 0.5
                    dv_pic_x = (1 + 0) * scn_width * 0.5
                    dv_pic_y = (1 + 0.35) * scn_height * 0.5
                        
                if reward_position == 'left':
                    cross_x = -0.5 * scn_width * 0.5
                    cross_y = 0.15 * scn_height * 0.5
                    pic_x = -0.5 * scn_width * 0.5
                    pic_y = 0.65 * scn_height * 0.5
                    dv_pic_x = (1 - 0.5) * scn_width * 0.5
                    dv_pic_y = (1 - 0.65) * scn_height * 0.5
                        
                if reward_position == 'right':
                    cross_x = 0.5 * scn_width * 0.5
                    cross_y = 0.15 * scn_height * 0.5
                    pic_x = 0.5 * scn_width * 0.5
                    pic_y = 0.65 * scn_height * 0.5
                    dv_pic_x = (1 + 0.5) * scn_width * 0.5
                    dv_pic_y = (1 - 0.65) * scn_height * 0.5
               
               
               
               
               
                reward_image_position = 'F:\\Pic_TOJ\\pic_pool_step_3_formal\\' + str(task_list[fc_current_loop+1]['target_pic'][i])
                imgload_msg_reward = '!V IMGLOAD CENTER %s %d %d %d %d' % (reward_image_position,
                                                                int(dv_pic_x),
                                                                int(dv_pic_y),
                                                                int(pic_width),
                                                                int(pic_height))
                
                
                r_correct_cross.setPos((cross_x, cross_y))
                r_correct_image.setImage(target_pic)
                r_correct_image.setPos((pic_x, pic_y))
                # keep track of which components have finished
                rewardComponents = [r_correct_image, r_correct_cross, r_cross]
                for thisComponent in rewardComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                el_tracker.sendMessage('reward_start_'+str(i))
                el_tracker.sendMessage(imgload_msg_reward)
                
                # --- Run Routine "reward" ---
                while continueRoutine and routineTimer.getTime() < 4.0:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *r_correct_image* updates
                    if r_correct_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        r_correct_image.frameNStart = frameN  # exact frame index
                        r_correct_image.tStart = t  # local t and not account for scr refresh
                        r_correct_image.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(r_correct_image, 'tStartRefresh')  # time at next scr refresh
                        r_correct_image.setAutoDraw(True)
                    if r_correct_image.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > r_correct_image.tStartRefresh + 2-frameTolerance:
                            # keep track of stop time/frame for later
                            r_correct_image.tStop = t  # not accounting for scr refresh
                            r_correct_image.frameNStop = frameN  # exact frame index
                            r_correct_image.setAutoDraw(False)
                            ser=serial.Serial('COM4',9600)
                            ser.write('JOF'.encode())
                            ser.close()
                            el_tracker.sendMessage('!V CLEAR 0 0 0')
                            el_tracker.sendMessage('reward_end_'+str(i))
                    # *r_correct_cross* updates
                    if r_correct_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        r_correct_cross.frameNStart = frameN  # exact frame index
                        r_correct_cross.tStart = t  # local t and not account for scr refresh
                        r_correct_cross.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(r_correct_cross, 'tStartRefresh')  # time at next scr refresh
                        r_correct_cross.setAutoDraw(True)
                    if r_correct_cross.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > r_correct_cross.tStartRefresh + 2-frameTolerance:
                            # keep track of stop time/frame for later
                            r_correct_cross.tStop = t  # not accounting for scr refresh
                            r_correct_cross.frameNStop = frameN  # exact frame index
                            r_correct_cross.setAutoDraw(False)
                        
                    # *r_cross* updates
                    if r_cross.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                        # keep track of start time/frame for later
                        r_cross.frameNStart = frameN  # exact frame index
                        r_cross.tStart = t  # local t and not account for scr refresh
                        r_cross.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(r_cross, 'tStartRefresh')  # time at next scr refresh
                        r_cross.setAutoDraw(True)
                    if r_cross.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > r_cross.tStartRefresh + 2-frameTolerance:
                            # keep track of stop time/frame for later
                            r_cross.tStop = t  # not accounting for scr refresh
                            r_cross.frameNStop = frameN  # exact frame index
                            r_cross.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                        el_tracker.stopRecording()
                        core.quit()
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in rewardComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()

                # --- Ending Routine "reward" ---
                for thisComponent in rewardComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-4.000000)
            
            # --- Prepare to start Routine "punishment" ---
            continueRoutine = True
            routineForceEnded = False
            if reward_position != fix_position:
                response = 0
                reward_ornot = 0
    #            # update component parameters for each repeat
    #            if reward_position == 'top':
    #                cross_x = 0
    #                cross_y = 0.2
    #                pic_x = 0
    #                pic_y = 0.7
    #                
    #            if reward_position == 'left':
    #                cross_x = -0.5
    #                cross_y = -0.9
    #                pic_x = -0.5
    #                pic_y = -0.4
    #                
    #            if reward_position == 'right':
    #                cross_x = 0.5
    #                cross_y = -0.9
    #                pic_x = 0.5
    #                pic_y = -0.4
    #                    
    #            p_corret_cross.setPos((cross_x, cross_y))
    #            p_corret_image.setPos((pic_x, pic_y))
                # keep track of which components have finished
                punishmentComponents = [p_corret_image, p_corret_cross, p_cross]
                for thisComponent in punishmentComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                el_tracker.sendMessage('punishment_start_'+str(i))
                # --- Run Routine "punishment" ---
                while continueRoutine and routineTimer.getTime() < 8.0:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
    #                # *p_corret_image* updates
    #                if p_corret_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
    #                    # keep track of start time/frame for later
    #                    p_corret_image.frameNStart = frameN  # exact frame index
    #                    p_corret_image.tStart = t  # local t and not account for scr refresh
    #                    p_corret_image.tStartRefresh = tThisFlipGlobal  # on global time
    #                    win.timeOnFlip(p_corret_image, 'tStartRefresh')  # time at next scr refresh
    #                    p_corret_image.setAutoDraw(True)
    #                if p_corret_image.status == STARTED:
    #                    # is it time to stop? (based on global clock, using actual start)
    #                    if tThisFlipGlobal > p_corret_image.tStartRefresh + 2-frameTolerance:
    #                        # keep track of stop time/frame for later
    #                        p_corret_image.tStop = t  # not accounting for scr refresh
    #                        p_corret_image.frameNStop = frameN  # exact frame index
    #                        p_corret_image.setAutoDraw(False)
    #                
    #                # *p_corret_cross* updates
    #                if p_corret_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
    #                    # keep track of start time/frame for later
    #                    p_corret_cross.frameNStart = frameN  # exact frame index
    #                    p_corret_cross.tStart = t  # local t and not account for scr refresh
    #                    p_corret_cross.tStartRefresh = tThisFlipGlobal  # on global time
    #                    win.timeOnFlip(p_corret_cross, 'tStartRefresh')  # time at next scr refresh
    #                    p_corret_cross.setAutoDraw(True)
    #                if p_corret_cross.status == STARTED:
    #                    # is it time to stop? (based on global clock, using actual start)
    #                    if tThisFlipGlobal > p_corret_cross.tStartRefresh + 2-frameTolerance:
    #                        # keep track of stop time/frame for later
    #                        p_corret_cross.tStop = t  # not accounting for scr refresh
    #                        p_corret_cross.frameNStop = frameN  # exact frame index
    #                        p_corret_cross.setAutoDraw(False)
                    
                    # *p_cross* updates
                    if p_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        p_cross.frameNStart = frameN  # exact frame index
                        p_cross.tStart = t  # local t and not account for scr refresh
                        p_cross.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(p_cross, 'tStartRefresh')  # time at next scr refresh
                        p_cross.setAutoDraw(True)
                    if p_cross.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > p_cross.tStartRefresh + 8-frameTolerance:
                            # keep track of stop time/frame for later
                            p_cross.tStop = t  # not accounting for scr refresh
                            p_cross.frameNStop = frameN  # exact frame index
                            p_cross.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                        el_tracker.stopRecording()
                        core.quit()
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in punishmentComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                el_tracker.sendMessage('punishment_end_'+str(i))
                # --- Ending Routine "punishment" ---
                for thisComponent in punishmentComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-8.000000)
            
            # --- Prepare to start Routine "jitter_iti" ---
            continueRoutine = True
            routineForceEnded = False
            # update component parameters for each repeat
            # keep track of which components have finished
            jitter_itiComponents = [jitter_cross]
            for thisComponent in jitter_itiComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            
            jitter_duration = r_0[trial_index]
            el_tracker.sendMessage('jitter_iti_start_'+str(i))
            # --- Run Routine "jitter_iti" ---
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *jitter_cross* updates
                if jitter_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    jitter_cross.frameNStart = frameN  # exact frame index
                    jitter_cross.tStart = t  # local t and not account for scr refresh
                    jitter_cross.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(jitter_cross, 'tStartRefresh')  # time at next scr refresh
                    jitter_cross.setAutoDraw(True)
                if jitter_cross.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > jitter_cross.tStartRefresh + jitter_duration-frameTolerance:
                        # keep track of stop time/frame for later
                        jitter_cross.tStop = t  # not accounting for scr refresh
                        jitter_cross.frameNStop = frameN  # exact frame index
                        jitter_cross.setAutoDraw(False)
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    el_tracker.stopRecording()
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in jitter_itiComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            el_tracker.sendMessage('jitter_iti_end_'+str(i))
            # --- Ending Routine "jitter_iti" ---
            for thisComponent in jitter_itiComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
                    
            data_list['fc_loop_times'].append(fc_loop_times)   
            data_list['total_loop_times'].append(total_loop_times) 
            data_list['cross_size'].append(cross_size) 
            data_list['minimum_duration'].append(minimum_duration) 
            data_list['memory_duration'].append(memory_duration) 
            data_list['delay_duration'].append(delay_duration) 
            data_list['choice_duration'].append(choice_duration) 
            data_list['list_index'].append(list_index) 
            data_list['monkey'].append(monkey) 
            data_list['target_pic'].append(target_pic)
            data_list['target_pic_position'].append(target_pic_position) 
            data_list['pic_bottom'].append(pic_top) 
            data_list['pic_left'].append(pic_left) 
            data_list['pic_right'].append(pic_right)
            data_list['pic_chosen_position'].append(fix_position) 
            data_list['pic_chosen'].append(pic_chosen) 
            data_list['choice_rt'].append(choice_rt) 
            data_list['response'].append(response)   
            data_list['reward_ornot'].append(reward_ornot)   
            data_list['jitter_iti'].append(jitter_duration) 
            
            df_result = pd.DataFrame(data_list)
            df_result.to_csv(filename + '_result.csv')
        #    print(filename + 'result.csv')
            # the Routine "jitter_iti" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
        fc_current_loop = fc_current_loop + 1
# completed loop_times repeats of 'forcedchoice_loop'

el_tracker.stopRecording()
# --- Prepare to start Routine "finish" ---
continueRoutine = True
routineForceEnded = False
# update component parameters for each repeat
finish_key_resp.keys = []
finish_key_resp.rt = []
_finish_key_resp_allKeys = []
# keep track of which components have finished
finishComponents = [finish_text, finish_key_resp]
for thisComponent in finishComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "finish" ---
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *finish_text* updates
    if finish_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        finish_text.frameNStart = frameN  # exact frame index
        finish_text.tStart = t  # local t and not account for scr refresh
        finish_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(finish_text, 'tStartRefresh')  # time at next scr refresh
        finish_text.setAutoDraw(True)
    
    # *finish_key_resp* updates
    if finish_key_resp.status == NOT_STARTED and t >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        finish_key_resp.frameNStart = frameN  # exact frame index
        finish_key_resp.tStart = t  # local t and not account for scr refresh
        finish_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(finish_key_resp, 'tStartRefresh')  # time at next scr refresh
        finish_key_resp.status = STARTED
        # keyboard checking is just starting
        finish_key_resp.clock.reset()  # now t=0
    if finish_key_resp.status == STARTED:
        theseKeys = finish_key_resp.getKeys(keyList=['space'], waitRelease=False)
        _finish_key_resp_allKeys.extend(theseKeys)
        if len(_finish_key_resp_allKeys):
            finish_key_resp.keys = _finish_key_resp_allKeys[-1].name  # just the last key pressed
            finish_key_resp.rt = _finish_key_resp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        el_tracker.stopRecording()
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in finishComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "finish" ---
for thisComponent in finishComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if finish_key_resp.keys in ['', [], None]:  # No response was made
    finish_key_resp.keys = None
thisExp.addData('finish_key_resp.keys',finish_key_resp.keys)
if finish_key_resp.keys != None:  # we had a response
    thisExp.addData('finish_key_resp.rt', finish_key_resp.rt)
thisExp.nextEntry()
# the Routine "finish" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()
local_edf =_thisDir + os.sep + u'data/%s' % (expInfo['edf_fname'])  + '.EDF'
try:
    el_tracker.receiveDataFile(edf_file, local_edf)
except RuntimeError as error:
    print('ERROR:', error)
#plot results
df_1 = pd.read_csv(filename + '_result.csv')
df_1['Cumulative_Correct_Rate']=0
df_1['trial_index'] =[x for x in range(1,len(df_1)+1)]
for i in range(0,len(df_1)+1):
    df_1['Cumulative_Correct_Rate'][i] = df_1['response'][0:i+1].sum()
df_1['Cumulative_Correct_Rate'] = df_1['Cumulative_Correct_Rate']/df_1['trial_index']
plt.figure(figsize=(8,12))
ax1 = plt.subplot(311)
plt.ylabel("RT")
ax1 = plt.plot(df_1['choice_rt'], marker='o', linestyle='dashed',linewidth=2, markersize=5)
plt.legend(title='choice_rt mean = %d s'% df_1['choice_rt'].mean())
ax2 = plt.subplot(312)
plt.ylabel("RESPONSE")
ax2 = plt.plot(df_1['response'],marker='o', linestyle='dashed',linewidth=2, markersize=5)
plt.legend(title='accuracy = %.2f%%' % (df_1['response'].mean()*100))
ax2 = plt.subplot(313)
plt.ylabel("CORRECT RATE")
ax3 = plt.plot(df_1['Cumulative_Correct_Rate'],marker='o', linestyle='dashed',linewidth=2, markersize=5)
plt.legend(title='Cumulative_Correct_Rate')
#df_1.to_csv(filename + 'result.csv')
plt.savefig(filename + '_result.png')
df_1.to_csv(filename + '_result.csv')
plt.show()
# --- End experiment ---
# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
#thisExp.saveAsWideText(filename+'.csv', delim='auto')
#thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
if eyetracker:
    eyetracker.setConnectionState(False)
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
