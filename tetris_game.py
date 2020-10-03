#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys, random
from PyQt5.QtWidgets import QMainWindow, QFrame, QDesktopWidget, QApplication, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt, QBasicTimer, pyqtSignal
from PyQt5.QtGui import QPainter, QColor

from tetris_model import BOARD_DATA, Shape
from tetris_ai import TETRIS_AI

import time

# TETRIS_AI = None

class Tetris(QMainWindow):
    def __init__(self):
        super().__init__()
        self.isStarted = False
        self.isPaused = False
        self.nextMove = None
        self.lastShape = Shape.shapeNone

        self.initUI()

    def initUI(self):
        self.gridSize = 22
        self.speed = 1000 # block drop speed

        self.timer = QBasicTimer()
        self.setFocusPolicy(Qt.StrongFocus)

        hLayout = QHBoxLayout()
        self.tboard = Board(self, self.gridSize)
        hLayout.addWidget(self.tboard)

        self.sidePanel = SidePanel(self, self.gridSize)
        hLayout.addWidget(self.sidePanel)

        self.statusbar = self.statusBar()
        self.tboard.msg2Statusbar[str].connect(self.statusbar.showMessage)

        self.start()

        self.center()
        self.setWindowTitle('Tetris')
        self.show()

        self.setFixedSize(self.tboard.width() + self.sidePanel.width(),
                          self.sidePanel.height() + self.statusbar.height())

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)

    def start(self):
        if self.isPaused:
            return

        self.isStarted = True
        self.tboard.score = 0
        BOARD_DATA.clear()

        self.tboard.msg2Statusbar.emit(str(self.tboard.score))

        BOARD_DATA.createNewPiece()
        self.timer.start(self.speed, self)

    def pause(self):
        if not self.isStarted:
            return

        self.isPaused = not self.isPaused

        if self.isPaused:
            self.timer.stop()
            self.tboard.msg2Statusbar.emit("paused")
        else:
            self.timer.start(self.speed, self)

        self.updateWindow()

    def resetfield(self):
        # self.tboard.score = 0
        self.tboard.reset_cnt += 1
        BOARD_DATA.clear()

    def updateWindow(self):
        self.tboard.updateData()
        self.sidePanel.updateData()
        self.update()

    def timerEvent(self, event):
        if event.timerId() == self.timer.timerId():
            y_operation = -1

            if TETRIS_AI and not self.nextMove:
                # get 
                # nextMove[0] : shape direction operation
                # nextMove[1] : x_operation (x move value)
                # nextMove[2] : y_operation (flag 0: move down, 1:drop down)
                self.nextMove = TETRIS_AI.nextMove()
            if self.nextMove:
                # shape direction operation
                k = 0
                while BOARD_DATA.currentDirection != self.nextMove[0] and k < 4:
                    ret = BOARD_DATA.rotateRight()
                    if ret == False:
                        print("cannot rotateRight")
                        if BOARD_DATA.currentY <= 1:
                            print("reset field.")
                            self.resetfield()
                        break
                    k += 1
                # x operatiox
                k = 0
                while BOARD_DATA.currentX != self.nextMove[1] and k < 5:
                    if BOARD_DATA.currentX > self.nextMove[1]:
                        ret = BOARD_DATA.moveLeft()
                        if ret == False:
                            print("cannot moveLeft")
                            if BOARD_DATA.currentY <= 1:
                                print("reset field.")
                                self.resetfield()
                            break
                    elif BOARD_DATA.currentX < self.nextMove[1]:
                        ret = BOARD_DATA.moveRight()
                        if ret == False:
                            print("cannot moveRight")
                            if BOARD_DATA.currentY <= 1:
                                print("reset field.")
                                self.resetfield()
                            break
                    k += 1
                # y operation
                y_operation = self.nextMove[2]

            # lines = BOARD_DATA.dropDown()
            if y_operation == 1: # dropdown
                removedlines, dropdownlines = BOARD_DATA.dropDown()
            else:
                removedlines = BOARD_DATA.moveDown()
                dropdownlines = 0

            self.UpdateScore(removedlines, dropdownlines)

            if self.lastShape != BOARD_DATA.currentShape:
                self.nextMove = None
                self.lastShape = BOARD_DATA.currentShape
            self.updateWindow()
        else:
            super(Tetris, self).timerEvent(event)

    def UpdateScore(self, removedlines, dropdownlines):
        # calculate and update current score
        # reference: gameboy tetris. http://www.din.or.jp/~koryan/tetris/d-gb1.htm
        if removedlines == 1:
            linescore = 40
        elif removedlines == 2:
            linescore = 100
        elif removedlines == 3:
            linescore = 300
        elif removedlines == 4:
            linescore = 1200
        else:
            linescore = 0
        dropdownscore = dropdownlines
        self.tboard.score += ( linescore + dropdownscore )

    def keyPressEvent(self, event):
        if not self.isStarted or BOARD_DATA.currentShape == Shape.shapeNone:
            super(Tetris, self).keyPressEvent(event)
            return

        key = event.key()
        
        if key == Qt.Key_P:
            self.pause()
            return
            
        if self.isPaused:
            return
        elif key == Qt.Key_Left:
            BOARD_DATA.moveLeft()
        elif key == Qt.Key_Right:
            BOARD_DATA.moveRight()
        elif key == Qt.Key_Up:
            BOARD_DATA.rotateLeft()
        elif key == Qt.Key_Space:
            removedlines, dropdownlines = BOARD_DATA.dropDown()
            self.UpdateScore(removedlines, dropdownlines)
        else:
            super(Tetris, self).keyPressEvent(event)

        self.updateWindow()


def drawSquare(painter, x, y, val, s):
    colorTable = [0x000000, 0xCC6666, 0x66CC66, 0x6666CC,
                  0xCCCC66, 0xCC66CC, 0x66CCCC, 0xDAAA00]

    if val == 0:
        return

    color = QColor(colorTable[val])
    painter.fillRect(x + 1, y + 1, s - 2, s - 2, color)

    painter.setPen(color.lighter())
    painter.drawLine(x, y + s - 1, x, y)
    painter.drawLine(x, y, x + s - 1, y)

    painter.setPen(color.darker())
    painter.drawLine(x + 1, y + s - 1, x + s - 1, y + s - 1)
    painter.drawLine(x + s - 1, y + s - 1, x + s - 1, y + 1)


class SidePanel(QFrame):
    def __init__(self, parent, gridSize):
        super().__init__(parent)
        self.setFixedSize(gridSize * 5, gridSize * BOARD_DATA.height)
        self.move(gridSize * BOARD_DATA.width, 0)
        self.gridSize = gridSize

    def updateData(self):
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        minX, maxX, minY, maxY = BOARD_DATA.nextShape.getBoundingOffsets(0)

        dy = 3 * self.gridSize
        dx = (self.width() - (maxX - minX) * self.gridSize) / 2

        val = BOARD_DATA.nextShape.shape
        for x, y in BOARD_DATA.nextShape.getCoords(0, 0, -minY):
            drawSquare(painter, x * self.gridSize + dx, y * self.gridSize + dy, val, self.gridSize)


class Board(QFrame):
    msg2Statusbar = pyqtSignal(str)
    speed = 1000 # block drop speed

    def __init__(self, parent, gridSize):
        super().__init__(parent)
        self.setFixedSize(gridSize * BOARD_DATA.width, gridSize * BOARD_DATA.height)
        self.gridSize = gridSize
        self.initBoard()

    def initBoard(self):
        self.score = 0
        self.reset_cnt = 0
        self.start_time = time.time() 
        BOARD_DATA.clear()

    def paintEvent(self, event):
        painter = QPainter(self)

        # Draw backboard
        for x in range(BOARD_DATA.width):
            for y in range(BOARD_DATA.height):
                val = BOARD_DATA.getValue(x, y)
                drawSquare(painter, x * self.gridSize, y * self.gridSize, val, self.gridSize)

        # Draw current shape
        for x, y in BOARD_DATA.getCurrentShapeCoord():
            val = BOARD_DATA.currentShape.shape
            drawSquare(painter, x * self.gridSize, y * self.gridSize, val, self.gridSize)

        # Draw a border
        painter.setPen(QColor(0x777777))
        painter.drawLine(self.width()-1, 0, self.width()-1, self.height())
        painter.setPen(QColor(0xCCCCCC))
        painter.drawLine(self.width(), 0, self.width(), self.height())

    def updateData(self):
        score_str = str(self.score)
        reset_cnt_str = str(self.reset_cnt)
        elapsed_time = round(time.time() - self.start_time, 3)
        elapsed_time_str = str(elapsed_time)
        self.msg2Statusbar.emit("score:" + score_str + ", gameover:" + reset_cnt_str + ", time[s]:" + elapsed_time_str ) # print string to status bar
        self.update()


if __name__ == '__main__':
    # random.seed(32)
    app = QApplication([])
    tetris = Tetris()
    sys.exit(app.exec_())
