from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys

import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
import numpy as np
import cv2
import time
import pickle

img_size = 160

def new_action(parent, text, slot=None, shortcut=None, icon=None, tip=None, enabled=True):
    """actionの設定と取得"""
    # アクション取得
    a = QAction(text, parent)
    # icon設定 iconsフォルダから取得
    if icon is not None:
        a.setIcon(QIcon('icons/'+icon))
    # ショートカット作成
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    # ステータスバーに表示するTip
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    # 押したときの動作
    if slot is not None:
        a.triggered.connect(slot)
    # ロックするか
    a.setEnabled(enabled)
    return a


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super(MainWindow, self).__init__()
        self.resize(QSize(600, 500))
        self.setWindowTitle('app')

        # ------------------ その2メニューの追加 --------------------
        # メニューの追加
        self.menus = {'File': self.menuBar().addMenu('File'),
                      'Edit': self.menuBar().addMenu('Edit'),
                      'Help': self.menuBar().addMenu('Help'),}

        # アクションを設定する
        quit = new_action(self,
                          text='quit',
                          slot=self.close,
                          shortcut='Ctrl+Q',
                          icon='quit',
                          tip='quitApp')
        show_info = new_action(self,
                               text='information',
                               slot=self.show_info,
                               tip='show app information')
        self.menus['File'].addAction(quit)
        self.menus['Help'].addAction(show_info)

        # ステータスバー
        self.statusBar()

        # ------------------ その3ツールバーの設定 --------------------
        # ToolBarを定義
        title = 'Tools'
        self.tools = QToolBar(title)
        self.tools.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        # self.addToolBar(Qt.LeftToolBarArea, toolbar) #左に配置
        self.addToolBar(Qt.TopToolBarArea, self.tools) #上に配置

        # ツールバーのアクションを定義
        register_user = new_action(self,
                               text='顔登録',
                               slot=self.register_user,
                               icon='registration',
                               tip='顔写真を登録します')

        match = new_action(self,
                               text='顔認証',
                               slot=self.match,
                               icon='face',
                               tip='顔認証を行います。')

        # ツールバーにアクションを追加
        self.tools.clear()
        self.tools.addAction(register_user)
        self.tools.addAction(match)

        self.load_data = None


        # ------------------ 顔認証用 --------------------
        self.member_path = 'face_data.pkl'
        if os.path.exists(self.member_path):
            with open(self.member_path, 'rb') as f:
                self.member = pickle.load(f)
        else:
            self.member = {}

    def show_info(self):
        """ Helpメニュ用メソッド """
        print("desktop app")

    def open_file(self):
        """ ツールバーメソッド """
        fp = QFileDialog.getOpenFileName(self, 'CSV 読込先を選択', "hoge.csv", "CSV (*.csv)")
        if fp[0]:
            with open(fp[0]) as f:
                self.load_data = f.read()
                QMessageBox.information(self, 'ファイル読み込み', '正常に読み込みました。')
                print(self.load_data)

    def save_file(self):
        """ ツールバーメソッド """
        if self.load_data:
            fp = QFileDialog.getSaveFileName(self, 'CSV 保存先を選択', "hoge.csv", "CSV (*.csv)")
            if fp[0]:
                with open(fp[0], mode='w') as f:
                    f.write(self.load_data)
                QMessageBox.information(self, 'ファイル保存', '正常に保存しました。')
        else:
            QMessageBox.critical(self, 'Error Message', 'データ読み込まれていません。')


    def register_user(self):

        dir = os.path.dirname(self.member_path)
        filters = "Images (*.png *.jpg *.jpeg)"

        fileObj = QFileDialog.getOpenFileName(self, " File dialog ", dir, filters)
        fn = fileObj[0]

        if fn:
            mtcnn = MTCNN(image_size=img_size, margin=10)

            # Create an inception resnet (in eval mode):
            resnet = InceptionResnetV1(pretrained='vggface2').eval()

            img_mem = Image.open(fn)
            filename = os.path.basename(fn).split(".")[0]
            img_mem = img_mem.resize((img_size, img_size))
            img_mem_cropped = mtcnn(img_mem)

            img_embedding = resnet(img_mem_cropped.unsqueeze(0))
            x1 = img_embedding.squeeze().to('cpu').detach().numpy().copy()
            self.member[filename] = x1
            with open(self.member_path, 'wb') as f:
                pickle.dump(self.member, f)

            QMessageBox.information(self, '顔認証', f'{filename}さん登録完了!')

    def match(self):

        if self.member:

            def cos_sim(v1, v2):
                return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

            # If required, create a face detection pipeline using MTCNN:
            mtcnn = MTCNN(image_size=img_size, margin=10)

            # Create an inception resnet (in eval mode):
            resnet = InceptionResnetV1(pretrained='vggface2').eval()

            cap = cv2.VideoCapture(0)

            name = ""
            score = 0.0
            check_count = 0.0
            while True:
                ret, frame = cap.read()
                cv2.imshow('Face Match', frame)

                k = cv2.waitKey(10)
                if k == 27:
                    break

                # numpy to PIL
                img_cam = Image.fromarray(frame)
                img_cam = img_cam.resize((img_size, img_size))

                img_cam_cropped = mtcnn(img_cam)
                if img_cam_cropped is not None:
                # if len(img_cam_cropped.size()) != 0:
                    img_embedding = resnet(img_cam_cropped.unsqueeze(0))

                    x2 = img_embedding.squeeze().to('cpu').detach().numpy().copy()

                    name = ""
                    score = 0.0
                    for key in self.member:
                        x1 = self.member[key]
                        if cos_sim(x1, x2) > 0.7:
                            name = key
                            score = cos_sim(x1, x2)
                            self.lock = False
                            break

                    check_count += 1.0
                else:
                    check_count += 0.2

                # 色々ループ抜ける条件
                if name:
                    break

                if check_count>20:
                    break

            cap.release()
            cv2.destroyAllWindows()

            if name:
                QMessageBox.information(self, '顔認証', f'Welcome {name}！')
            else:
                QMessageBox.information(self, '顔認証', '該当するユーザーがいません！')
        else:
            QMessageBox.information(self, '顔認証', '顔データが未登録です。')


def get_main_app(argv=[]):
    app = QApplication(argv)
    win = MainWindow()
    win.show()
    return app, win

if __name__ == '__main__':
    app, _win = get_main_app(sys.argv)
    sys.exit(app.exec_())