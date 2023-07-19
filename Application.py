from GANime import Ui_GANime
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QStringListModel
import sys
import os
from tkinter.filedialog import askdirectory
from PyQt5.QtGui import QPainter, QPixmap, QColor
import csv


class GANimeUI(QMainWindow, Ui_GANime):
    def __init__(self, parent=None):
        super(GANimeUI, self).__init__(parent)
        self.current_image_path = "data\\images"
        self.setupUi(self)
        self.set_current_path_text(os.path.join(os.path.dirname(__file__), self.current_image_path))
        self.AIList.selectionModel().selectionChanged.connect(self.onSelectionChanged)
        self.trainingList.selectionModel().selectionChanged.connect(self.onSelectionChanged)
        self.current_list = 0
        self.current_Image = None
        self.csv_label = []
        self.get_csv_label()

    def set_current_path_text(self, text):
        self.folder_path.setText(text)

    def preview_change_path_text(self):
        path = askdirectory()
        if path != '':
            self.current_image_path = path
        self.set_current_path_text(self.current_image_path)

    def refresh_image_list(self):

        image_dir = os.listdir(self.current_image_path)
        array1 = [item for item in image_dir if "_" not in item]
        array2 = [item for item in image_dir if "_" in item]

        model1 = QStringListModel()
        model1.setStringList(array1)

        model2 = QStringListModel()
        model2.setStringList(array2)

        self.AIList.setModel(model1)
        self.trainingList.setModel(model2)

    def onSelectionChanged(self, selected, deselected):
        indexes = selected.indexes()
        for index in indexes:
            item_name = index.data()
            image_window_size = self.image.size()
            small_image_size = self.minIcon.size()

            # save image detail into class
            image = QPixmap(os.path.join(self.current_image_path, item_name))
            if self.current_list == 0:
                image_id = int(item_name.split('.')[0])
                label = self.csv_label[image_id]
                label = label[1].split(" ")
                hair = label[0]
                eyes = label[2]
                self.current_Image = ImageDetail(name=item_name, year=0, eyes=eyes, hair=hair,
                                                 width=image.width(), height=image.height())
            elif self.current_list == 1:
                name_year = item_name.split('.')[0]
                year = int(name_year.split('_')[1])
                self.current_Image = ImageDetail(name=item_name, year=year, eyes="", hair="",
                                                 width=image.width(), height=image.height())

            pixmap1 = QPixmap(image_window_size)
            pixmap1.fill(QColor("white"))
            pixmap2 = QPixmap(small_image_size)
            pixmap2.fill(QColor("white"))

            painter1 = QPainter(pixmap1)
            painter1.drawPixmap(0, 0, image_window_size.width(), image_window_size.height(), image)
            painter1.end()
            painter2 = QPainter(pixmap2)
            painter2.drawPixmap(0, 0, small_image_size.width(), small_image_size.height(), image)
            painter2.end()

            self.image.setPixmap(pixmap1)
            self.minIcon.setPixmap(pixmap2)
            self.update_detail_box()

    def update_detail_box(self):
        text = "Name: \t" + str(self.current_Image.name) + "\n" +\
               "Year: \t" + str(self.current_Image.year) + "\n" +\
               "Eyes: \t" + str(self.current_Image.eyes) + "\n" +\
               "Hair: \t" + str(self.current_Image.hair) + "\n" +\
               "Size: \t" + str(self.current_Image.width) + "x" + str(self.current_Image.height)
        self.ImageDetail.setText(text)

    def dataset_change(self, current):
        self.current_list = current
        if current == 0:
            self.trainingList.clearSelection()
        elif current == 1:
            self.AIList.clearSelection()

    def get_csv_label(self):
        csv_file = "data\\tags.csv"
        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.csv_label.append(row)


class ImageDetail:
    def __init__(self, name, year, eyes, hair, width, height):
        self.name = name
        self.year = year
        self.eyes = eyes
        self.hair = hair
        self.width = width
        self.height = height


def window():
    app = QApplication(sys.argv)
    win = GANimeUI()
    win.show()
    sys.exit(app.exec_())
