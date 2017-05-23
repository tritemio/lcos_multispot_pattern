import matplotlib.pyplot as plt
import matplotlib as mpl

from ipywidgets import (interact, interactive, fixed, FloatText, IntText, FloatSlider, Checkbox, IntSlider,
                        Box, HBox, VBox, Label, Button, ToggleButton, Layout, Accordion)
import ipywidgets as widgets
from IPython.display import display

from patternlib import compute_pattern

def on_value_change(change):
    #print(change['name'], change['new'], change['owner'].description)
    #print([k for k in change.keys()])
    name = change['owner'].description
    value = change['new']
    if name == 'focal':
        value *= 1e-3
    d[name] = value
    a = compute_pattern(**d)
    im.set_data(a)

d = {
 'Xm': [21, 21.72, 44.87, 68.2, 91.45, 114.72, 137.87, 160.97, 184.04],
 'Ym': [21.72, 44.87, 68.2, 91.45, 114.72, 137.87, 160.97, 184.04],
 'center_x': 0,
 'center_y': 0,
 'dark_all': False,
 'focal': 0.032,
 'grid': True,
 'steer_only': False,
 'nrows': 4,
 'ncols': 12,
 'phase_factor': 82,
 'phase_max': 3.,
 'phase_wrap_neg': True,
 'phase_wrap_pos': False,
 'pitch_x': 25,
 'pitch_y': 25,
 'ref_spot': 0,
 'ref_spot_dark': False,
 'rotation': 1,
 'spotsize': 30.0,
 'steer_horiz': True,
 'steer_lw': 1,
 'steer_pad': 0,
 'steer_vmax': 95,
 'test_pattern': False,
 'wavelen': 532e-09,
 'stretch': True,
}

box_layout = Layout(display='flex', align_items='stretch', width='80%')
vbox_layout = Layout(display='flex', align_items='stretch', flex_flow = 'column', width='20%')
items_layout = Layout(flex='1 1 auto', width='auto')

kws = dict(layout=items_layout)
dw = {}
dw['nrows'] = IntText(min=1, max=25, description='nrows', **kws)
dw['ncols'] = IntText(min=1, max=25, description='ncols', **kws)
dw['pitch_x'] = FloatText(min=5, max=100, value=20, description='pitch_x', **kws)
dw['pitch_y'] = FloatText(min=5, max=100, value=30, description='pitch_y', **kws)

dw['center_x'] = FloatText(min=-400, max=400, value=0, description='center_x', **kws)
dw['center_y'] = FloatText(min=-300, max=300, value=0, description='center_y', **kws)
dw['rotation'] = FloatText(min=-90, max=90, value=0, description='rotation', **kws)
dw['focal'] = FloatText(min=5, max=100, value=32, description='focal', **kws)

dw['steer_only'] = ToggleButton(False, description='steer_only', **kws)
dw['dark_all'] = ToggleButton(False, description='dark_all', **kws)
dw['grid'] = ToggleButton(False, description='grid', **kws)
dw['test_pattern'] = ToggleButton(False, description='test_pattern', **kws)

dw['steer_horiz'] = ToggleButton(True, description='steer_horiz', **kws)
dw['steer_lw'] = IntText(min=1, max=25, value=1, description='steer_lw', **kws)
dw['steer_pad'] = IntText(min=1, max=50, value=1, description='steer_pad', **kws)
dw['steer_vmax'] = IntText(min=0, max=255, description='steer_vmax', **kws)

dw['stretch'] = ToggleButton(False, description='stretch', **kws)
dw['ref_spot'] = IntText(min=0, max=100, value=1, description='ref_spot', **kws)
dw['ref_spot_dark'] = ToggleButton(False, description='ref_spot_dark', **kws)

dw['phase_factor'] = FloatText(min=0, max=10, value=2, description='phase_factor', **kws)
dw['phase_max'] = FloatText(min=0, max=5, value=2, description='phase_max', **kws)
dw['phase_wrap_neg'] = ToggleButton(False, description='phase_wrap_neg', **kws)
dw['phase_wrap_pos'] = ToggleButton(False, description='phase_wrap_pos', **kws)


vbox1 = VBox([dw[name] for name in ('nrows', 'ncols', 'pitch_x', 'pitch_y')],
             layout=vbox_layout)

vbox2 = VBox([dw[name] for name in ('center_x', 'center_y', 'rotation', 'focal')],
             layout=vbox_layout)

vbox3 = VBox([dw[name] for name in ('steer_only', 'dark_all', 'grid', 'test_pattern')],
             layout=vbox_layout)

vbox4 = VBox([dw[name] for name in ('stretch', 'steer_horiz', 'steer_lw', 'steer_pad', 'steer_vmax')],
             layout=vbox_layout)

vbox5 = VBox([dw[name] for name in ('phase_factor', 'phase_max', 'phase_wrap_neg', 'phase_wrap_pos')],
             layout=vbox_layout)


hbox = Box([vbox1, vbox2, vbox3, vbox4, vbox5], layout=box_layout)
for name, w in dw.items():
    if type(w) == Label:
        continue
    w.value = d[name] if name != 'focal' else d[name]*1e3
    w.observe(on_value_change, names='value')
