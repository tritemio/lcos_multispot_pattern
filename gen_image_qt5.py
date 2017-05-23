"""
Show a frameless image in a QT5 window
"""

import sys
from PyQt5 import QtCore, QtGui, QtWidgets

screen_resolution = 1920, 1200

app = QtWidgets.QApplication(sys.argv)
label = QtWidgets.QLabel()
label.setWindowFlags(QtCore.Qt.FramelessWindowHint)
label.resize(800,600)
label.move(0,0)
label.setWindowTitle('LCOS Pattern')


def show_pattern(a):
    i = QtGui.QImage(a.tostring(), 800, 600, QtGui.QImage.Format_Indexed8)
    p = QtGui.QPixmap.fromImage(i)
    label.setPixmap(p)
    label.show()

def show_pattern_twin(a, im):
    show_pattern(a)
    im.set_data(a)
    plt.draw()

def clear_pattern():
    label.hide()

def move_to_2nd_screen():
    label.move(screen_resolution[0], 0)

def move_to_3rd_screen():
    label.move(0, screen_resolution[1])

def move_from_other_screen():
    label.move(0, 0)
