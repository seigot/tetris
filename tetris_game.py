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

    # reference: gameboy tetris. http://www.din.or.jp/~koryan/tetris/d-gb1.htm
    LINE_SCORE_1 = 40
    LINE_SCORE_2 = 100
    LINE_SCORE_3 = 300
    LINE_SCORE_4 = 1200

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
                # get nextMove from TetrisAI
                TetrisStatus = self.getTetrisStatus()
                
                self.nextMove = TETRIS_AI.nextMove(TetrisStatus)
            if self.nextMove:
                # shape direction operation
                next_x = self.nextMove["strategy"]["x"]
                y_operation = self.nextMove["strategy"]["y_operation"]
                next_direction = self.nextMove["strategy"]["direction"]
                k = 0
                while BOARD_DATA.currentDirection != next_direction and k < 4:
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
                while BOARD_DATA.currentX != next_x and k < 5:
                    if BOARD_DATA.currentX > next_x:
                        ret = BOARD_DATA.moveLeft()
                        if ret == False:
                            print("cannot moveLeft")
                            if BOARD_DATA.currentY <= 1:
                                print("reset field.")
                                self.resetfield()
                            break
                    elif BOARD_DATA.currentX < next_x:
                        ret = BOARD_DATA.moveRight()
                        if ret == False:
                            print("cannot moveRight")
                            if BOARD_DATA.currentY <= 1:
                                print("reset field.")
                                self.resetfield()
                            break
                    k += 1

            # lines = BOARD_DATA.dropDown()
            if y_operation == 1: # dropdown
                removedlines, dropdownlines = BOARD_DATA.dropDown()
            else:
                removedlines = BOARD_DATA.moveDown()
                dropdownlines = 0

            self.UpdateScore(removedlines, dropdownlines)

            # update nextMove everytime.
            self.nextMove = None
            #if self.lastShape != BOARD_DATA.currentShape:
            #    self.nextMove = None
            #    self.lastShape = BOARD_DATA.currentShape

            self.updateWindow()
        else:
            super(Tetris, self).timerEvent(event)

    def UpdateScore(self, removedlines, dropdownlines):
        # calculate and update current score
        if removedlines == 1:
            linescore = Tetris.LINE_SCORE_1
        elif removedlines == 2:
            linescore = Tetris.LINE_SCORE_2
        elif removedlines == 3:
            linescore = Tetris.LINE_SCORE_3
        elif removedlines == 4:
            linescore = Tetris.LINE_SCORE_4
        else:
            linescore = 0
        dropdownscore = dropdownlines
        self.tboard.score += ( linescore + dropdownscore )
        self.tboard.line += removedlines
        if removedlines > 0:
            self.tboard.lineStat[removedlines - 1] += 1

    def getTetrisStatus(self):
        # return current Board status.
        # define status data.
        status = {"board":
                      {
                        "width": "none",
                        "height": "none",
                        "backboard": "none",
                      },
                  "shape":
                      {
                        "currentX":"none",
                        "currentY":"none",
                        "currentDirection":"none",
                        "currentShape":{
                           "shape":"none",
                           "direction_range":"none",
                        },
                        "nextShape":{
                           "shape":"none",
                           "direction_range":"none",
                        },
                      },
                  "judge_info":
                      {
                        "elapsed_time":"none",
                        "gameover_count":"none",
                        "score":"none",
                        "line":"none",
                      },
                  "debug_info":
                      {
                        "line_score": {
                          "1":"none",
                          "2":"none",
                          "3":"none",
                          "4":"none",
                        },
                        "shape_info": {
                          "shapeNone": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeI": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeL": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeJ": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeT": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeO": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeS": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeZ": {
                             "index" : "none",
                             "color" : "none",
                          },
                        },
                        "lineStat":"none",
                        "shapeStat":"none",
                      },
                  }
        # update status
        ## board
        status["board"]["width"] = BOARD_DATA.width
        status["board"]["height"] = BOARD_DATA.height
        status["board"]["backboard"] = BOARD_DATA.getData()
        ## shape
        status["shape"]["currentX"] = BOARD_DATA.currentX
        status["shape"]["currentY"] = BOARD_DATA.currentY
        status["shape"]["currentDirection"] = BOARD_DATA.currentDirection
        status["shape"]["currentShape"]["shape"] = BOARD_DATA.currentShape.shape
        ### current shape
        if BOARD_DATA.currentShape.shape in (Shape.shapeI, Shape.shapeZ, Shape.shapeS):
            Range = (0, 1)
        elif BOARD_DATA.currentShape.shape == Shape.shapeO:
            Range = (0,)
        else:
            Range = (0, 1, 2, 3)
        status["shape"]["currentShape"]["direction_range"] = Range
        ### next shape
        status["shape"]["nextShape"]["shape"] = BOARD_DATA.nextShape.shape
        if BOARD_DATA.nextShape.shape in (Shape.shapeI, Shape.shapeZ, Shape.shapeS):
            Range = (0, 1)
        elif BOARD_DATA.nextShape.shape == Shape.shapeO:
            Range = (0,)
        else:
            Range = (0, 1, 2, 3)
        status["shape"]["nextShape"]["direction_range"] = Range
        ## judge_info
        status["judge_info"]["elapsed_time"] = round(time.time() - self.tboard.start_time, 3)
        status["judge_info"]["gameover_count"] = self.tboard.reset_cnt
        status["judge_info"]["score"] = self.tboard.score
        status["judge_info"]["line"] = self.tboard.line
        ## debug_info
        status["debug_info"]["lineStat"] = self.tboard.lineStat
        status["debug_info"]["shapeStat"] = BOARD_DATA.shapeStat
        status["debug_info"]["line_score"]["1"] = Tetris.LINE_SCORE_1
        status["debug_info"]["line_score"]["2"] = Tetris.LINE_SCORE_2
        status["debug_info"]["line_score"]["3"] = Tetris.LINE_SCORE_3
        status["debug_info"]["line_score"]["4"] = Tetris.LINE_SCORE_4
        status["debug_info"]["shape_info"]["shapeNone"] = Shape.shapeNone
        status["debug_info"]["shape_info"]["shapeI"]["index"] = Shape.shapeI
        status["debug_info"]["shape_info"]["shapeI"]["color"] = "red"
        status["debug_info"]["shape_info"]["shapeL"]["index"] = Shape.shapeL
        status["debug_info"]["shape_info"]["shapeL"]["color"] = "green"
        status["debug_info"]["shape_info"]["shapeJ"]["index"] = Shape.shapeJ
        status["debug_info"]["shape_info"]["shapeJ"]["color"] = "purple"
        status["debug_info"]["shape_info"]["shapeT"]["index"] = Shape.shapeT
        status["debug_info"]["shape_info"]["shapeT"]["color"] = "gold"
        status["debug_info"]["shape_info"]["shapeO"]["index"] = Shape.shapeO
        status["debug_info"]["shape_info"]["shapeO"]["color"] = "pink"
        status["debug_info"]["shape_info"]["shapeS"]["index"] = Shape.shapeS
        status["debug_info"]["shape_info"]["shapeS"]["color"] = "blue"
        status["debug_info"]["shape_info"]["shapeZ"]["index"] = Shape.shapeZ
        status["debug_info"]["shape_info"]["shapeZ"]["color"] = "yellow"
        if BOARD_DATA.currentShape == Shape.shapeNone:
            print("warning: current shape is none !!!")

        return status

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
        elif key == Qt.Key_M:
            removedlines = BOARD_DATA.moveDown()
            dropdownlines = 0
            self.UpdateScore(removedlines, dropdownlines)
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
        self.line = 0
        self.lineStat = [0, 0, 0, 0]
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
        line_str = str(self.line)
        reset_cnt_str = str(self.reset_cnt)
        elapsed_time = round(time.time() - self.start_time, 3)
        elapsed_time_str = str(elapsed_time)
        self.msg2Statusbar.emit("score:" + score_str + ",line:" + line_str + ", gameover:" + reset_cnt_str + ", time[s]:" + elapsed_time_str ) # print string to status bar
        self.update()


if __name__ == '__main__':
    # random.seed(32)
    app = QApplication([])
    TETRIS_MANEGER = Tetris()
    sys.exit(app.exec_())
