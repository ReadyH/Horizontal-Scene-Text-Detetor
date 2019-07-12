from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QHBoxLayout, QPushButton, QLineEdit, 
                            QVBoxLayout,QMessageBox, QToolTip, QDialog, QFileDialog, QMainWindow)

import sys
import os
import detect

class Single(QWidget):
	def __init__(self):
		super(Single,self).__init__()
		self.init()
		self.input_img_dir = ""
		self.output_img_dir = ""
		self.output_txt_dir = ""
		
	def init(self):
		self.inputLabel1 = QLabel("待检测图像文件夹：")
		self.editLine1 = QLineEdit()
		self.printButton1 = QPushButton("选择")
		self.printButton1.clicked.connect(lambda:self.openFile1(self.editLine1.text()))

		inputLayout1 = QHBoxLayout()
		inputLayout1.addWidget(self.inputLabel1)
		inputLayout1.addWidget(self.editLine1)
		inputLayout1.addWidget(self.printButton1)

		self.inputLabel2 = QLabel("图像结果输出文件夹：")
		self.editLine2 = QLineEdit()
		self.printButton2 = QPushButton("选择")
		self.printButton2.clicked.connect(lambda:self.openFile2(self.editLine2.text()))

		inputLayout2 = QHBoxLayout()
		inputLayout2.addWidget(self.inputLabel2)
		inputLayout2.addWidget(self.editLine2)
		inputLayout2.addWidget(self.printButton2)

		self.inputLabel3 = QLabel("文档结果输出文件夹：")
		self.editLine3 = QLineEdit()
		self.printButton3 = QPushButton("选择")
		self.printButton3.clicked.connect(lambda:self.openFile3(self.editLine3.text()))

		inputLayout3 = QHBoxLayout()
		inputLayout3.addWidget(self.inputLabel3)
		inputLayout3.addWidget(self.editLine3)
		inputLayout3.addWidget(self.printButton3)

		self.detection = QPushButton("开始检测")
		self.detection.clicked.connect(self.detect)

		detecLayout = QHBoxLayout()
		detecLayout.addWidget(self.detection)

		mainLayout = QVBoxLayout()
		mainLayout.addLayout(inputLayout1)
		mainLayout.addLayout(inputLayout2)
		mainLayout.addLayout(inputLayout3)
		mainLayout.addLayout(detecLayout)
		
		self.setLayout(mainLayout)
		self.setGeometry(500, 500, 500, 100)
		self.setWindowTitle('自然场景图像水平文本检测软件')
		self.show()

	def openFile1(self,filePath1):
		if os.path.exists(filePath1):
			input_path = QFileDialog.getExistingDirectory(self,
                  "选取待检测图像文件夹",
                  "./")
		else:
			input_path = QFileDialog.getExistingDirectory(self,
                  "选取待检测图像文件夹",
                  "./")
		self.input_img_dir = str(input_path)
		self.editLine1.setText(str(input_path))

	def openFile2(self,filePath2):
		if os.path.exists(filePath2):
			output_img_path = QFileDialog.getExistingDirectory(self,
                  "选取图像结果输出文件夹",
                  "./")
		else:
			output_img_path = QFileDialog.getExistingDirectory(self,
                  "选取图像结果输出文件夹",
                  "./")
		self.output_img_dir = str(output_img_path)
		self.editLine2.setText(str(output_img_path))

	def openFile3(self,filePath3):
		if os.path.exists(filePath3):
			output_txt_path = QFileDialog.getExistingDirectory(self,
                  "选取文档结果输出文件夹",
                  "./")
		else:
			output_txt_path = QFileDialog.getExistingDirectory(self,
                  "选取文档结果输出文件夹",
                  "./")
		self.output_txt_dir = str(output_txt_path)
		self.editLine3.setText(str(output_txt_path))

	def detect(self):
		done = predict.predict(self.input_img_dir, self.output_img_dir, self.output_txt_dir)
		if done == "Done":
			QMessageBox.information(self, "检测完成",
                                "所有结果均已放入指定文件夹")
		else:
			QMessageBox.information(self, "检测失败",
                                "请检查输入图像是否有误！")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Single()
    sys.exit(app.exec_())
