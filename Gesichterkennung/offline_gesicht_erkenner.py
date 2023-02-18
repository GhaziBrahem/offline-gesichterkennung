from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
import sys
import numpy as np
import cv2
import os
import dlib
from pkg_resources import resource_filename
import json
_schwellenwert = 0.57 #Schwellenwert

def pose_predictor_model_location(): # Definiere die Gesichtlocation funktion für den 68-Punkte-Modell-Detektor
    return resource_filename(__name__, "models/shape_predictor_68_face_landmarks.dat")
def pose_predictor_five_point_model_location(): # Definiere die Gesichtlocation funktion für den 5-Punkte-Modell-Detektor
    return resource_filename(__name__, "models/shape_predictor_5_face_landmarks.dat")
def face_recognition_model_location(): # Definiere die Gesichtlocation funktion für das Gesichtserkennungsmodell
    return resource_filename(__name__, "models/dlib_face_recognition_resnet_model_v1.dat")
def cnn_face_detector_model_location(): # Definiere die Gesichtlocation funktion für das CNN-Gesichtserkennungsmodell
    return resource_filename(__name__, "models/mmod_human_face_detector.dat")

face_detector = dlib.get_frontal_face_detector() # Lade den frontal_face_detector aus der DLIB-Bibliothek
predictor_68_point_model = pose_predictor_model_location() # Lade das 68-Punkte-Modell anhand von Gesichtmerkmale
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model) # Erstelle den 68-Punkte-Modell-Detektor
predictor_5_point_model = pose_predictor_five_point_model_location() # Lade das 5-Punkte-Modell anhand von Gesichtmerkmale
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model) # Erstelle den 5-Punkte-Modell-Detektor
cnn_face_detection_model = cnn_face_detector_model_location() # Lade das CNN-Gesichtserkennungsmodell anhand von Gesichtmerkmale
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model) # Erstelle den CNN-Gesichtserkennungsmodell-Detektor
face_recognition_model = face_recognition_model_location() # Lade das Gesichtserkennungsmodell anhand von Gesichtmerkmale
face_encoder = dlib.face_recognition_model_v1(face_recognition_model) # Erstelle den Gesichtserkennungsmodell-Encoder

# Funktion zum Rohgesichtererkennung basierend auf "hog" (für Gesichtserkennung auf CPU) oder "cnn" (für GPU)
def _raw_face_locations(img, number_of_times_to_upsample=1, model="hog"):
    if model == "cnn":
        return cnn_face_detector(img, number_of_times_to_upsample) # Verwendung des Modells cnn
    else:
        return face_detector(img, number_of_times_to_upsample) # Verwendung des Modells hog
# Funktion zur Rückgabe der Grenzen des erkannten Gesichts
def gesicht_tracker(img, number_of_times_to_upsample=1, model="hog"):
    if model == "cnn": # Gesichtserkennung mit dem cnn-Modell und anschließende Rückgabe der Grenzen des erkannten Gesichts
        return [_trim_css_to_bounds(_rect_to_css(face.rect), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample, "cnn")] 
    else:  # Gesichtserkennung mit dem hog-Modell und anschließende Rückgabe der Grenzen des erkannten Gesichts
        return [_trim_css_to_bounds(_rect_to_css(face), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample, model)]
# Funktion zur Rückgabe der Gesichtslandmarken 
def _raw_face_landmarks(face_image, face_locations=None, model="large"):
    if face_locations is None:
        face_locations = _raw_face_locations(face_image)
    else:
        face_locations = [_css_to_rect(face_location) for face_location in face_locations]
    pose_predictor = pose_predictor_68_point
    if model == "small":
        pose_predictor = pose_predictor_5_point
    return [pose_predictor(face_image, face_location) for face_location in face_locations]
# Funktion zur Rückgabe der Gesichts-Encodings
def gesicht_encoder(face_image, known_face_locations=None, num_jitters=1, model="small"):
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations, model) # Abrufen der Rohlandmarken des Gesichts
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks] # Berechnen der Gesichtsencodings
# Funktion zur Berechnung der Distanz zwischen dem Gesichts-Encoding des gegebenen Bildes und den Gesichts-Encodings des bekannten Gesichts
def gesicht_Diff(known_face_encodings, face_encoding_to_check, tolerance=_schwellenwert):
    return list(gesicht_Abstand(known_face_encodings, face_encoding_to_check) <= tolerance)  # Rückgabe der Distanzen zwischen den bekannten Gesichts-Encodings und dem gegebenen Gesichts-Encoding
# Hilfsfunktion zum Konvertieren eines Rechtecks in eine CSS-Stilbeschreibung
def _rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()
# Hilfsfunktion zum Konvertieren einer CSS-Stilbeschreibung in ein Rechteck
def _css_to_rect(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])
# Hilfsfunktion zum Kürzen der CSS-Stilbeschreibung auf die Bildgröße
def _trim_css_to_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)
# Berechne den Abstand zwischen beiden Gesichter.
def gesicht_Abstand(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)
# Eine leere Liste für Bilder und zugehorige Namen initialisieren
images = []
classNames = []
path = 'faces' # Der Pfad zum Ordner, der die JSON-Dateien enthält, wird festgelegt
myList = os.listdir(path) # Die Liste aller Dateinamen im Ordner wird erstellt
print('Gesichter von Json Format zur Bilde generieren...')
for cl in myList: # Eine Schleife, um durch jede JSON-Datei zu gehen und ein Bild aus der Numpy-Array-Darstellung zu erstellen
    with open(f'{path}/{cl}', 'r') as f: # Öffnen der JSON-Datei und Lesen der Daten
        img_json  = f.read()
    # Konvertieren des Numpy-Arrays aus der JSON-Datei in ein Bild 
    img_np = np.array(json.loads(img_json)) 
    img_np = img_np.astype(np.uint8)
    img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    images.append(img) # Das Bild wird gespeichert und in der Liste images hinzugefügt
    classNames.append(os.path.splitext(cl)[0]) # Die Klassenbezeichnung wird aus dem Dateinamen extrahiert und der Liste Namen hinzugefügt
print("Gesichter im Database ",classNames)
def findEncodings(images): # Eine Funktion, die die Gesichtskodierungen für eine Liste von Bildern zurückgibt
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = gesicht_encoder(img)[0]
        encodeList.append(encode)
    return encodeList
# Die Gesichtskodierungen für alle Bilder werden berechnet und in der Liste encodeListKnown gespeichert
encodeListKnown = findEncodings(images)
print('Gesichter von Json Format zur Bilde erfolgreich generiert !')
class VideoCapture(qtc.QThread):
    change_pixmap_signal = qtc.pyqtSignal(np.ndarray) # Signal für Pixmap-Änderung
    def __init__(self):
        super().__init__() # Ruft den Konstruktor der übergeordneten Klasse auf
        self.run_flag = True # Flag für den Thread-Lauf
        self.save_face = False # Flag für Gesichterkennung
        self.recognize_face = False # Flag für GesichtsSpeicherung
        self.input = None # Eingabe-Variable
    def run(self):
        cap = cv2.VideoCapture(0) # Video-Aufnahme-Objek
        while self.run_flag: # Schleife, solange der Thread-Lauf aktiv ist000000000
            success, img = cap.read() # Frame aus dem Video-Aufnahme-Objekt auslesen
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25) # Verkleinern des Bildes
            rgbSkale = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB) # Konvertierung des Bildes von BGR in RGB
            if self.recognize_face  is True: # Überprüfung, ob Gesichtserkennung aktiviert ist
                facesCurFrame = gesicht_tracker(rgbSkale) # Verfolgen von Gesichtern durch den Algorithmus des optischen Flusses
                encodesCurFrame = gesicht_encoder(rgbSkale, facesCurFrame) # Codieren von Gesichtern in einer Liste der Merkmalsvektoren
                for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame): # Schleife über die Merkmalsvektoren und Gesichtspositionen
                    _treffer = gesicht_Diff(encodeListKnown, encodeFace) # Berechnung der Differenzen zwischen dem Merkmalsvektor des aktuellen Gesichts und den bekannten Gesichtern in der Datenbank
                    _gesichtAbstand = gesicht_Abstand(encodeListKnown, encodeFace) # Berechnung des Abstands zwischen dem Merkmalsvektor des aktuellen Gesichts und den bekannten Gesichtern in der Datenbank
                    y1, x2, y2, x1 = faceLoc # Extrahieren der Koordinaten des erkannten Gesichts
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4 # Skalieren der Koordinaten auf die ursprüngliche Bildgröße
                    matchIndex = np.argmin(_gesichtAbstand) # Bestimmung des am besten passenden bekannten Gesichts durch Berechnung des minimalen Abstands
                    if _treffer[matchIndex]: # Wenn das Gesicht in der Datenbank vorhanden ist, wird der Name ausgegeben und das Gesicht mit einem grünen Rahmen markiert
                        name = classNames[matchIndex].upper()
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        print(name)    
                    else: # Wenn das Gesicht in der Datenbank nicht vorhanden ist, wird eine Nachricht ausgegeben und das Gesicht mit einem blauen Rahmen markiert
                        print("Gesicht ist nicht zu erkennen !")
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)                        
            if self.save_face is True: # Überprüfung, ob das Gesicht gespeichert werden soll
                detectedface = cv2.resize(img, (800, 400), interpolation=cv2.INTER_CUBIC) # Vergrößern des erkannten Gesichts
                img_np = np.array(detectedface) # Konvertierung des Gesichts in ein Numpy-Array
                img_json = json.dumps(img_np.tolist()) # Konvertierung des Numpy-Arrays in JSON
                with open(f"{path}/{self.input}.json", 'w') as f: # Speichern des Gesichts als JSON-Datei
                    f.write(img_json)    
                self.save_face = False  # Setzen des save_face-Flag auf False 
            if success == True: # Falls das aktuelle Frame erfolgreich ausgelesen wurde
                self.change_pixmap_signal.emit(img) # Signal für Pixmap-Änderung auslösen
        cap.release() # Video-Aufnahme beenden
        
    def stop(self):
        self.run_flag = False
        self.wait()
    def addface(self,name):
        self.save_face = True
        self.input = name
    def recognizeface(self):
        if self.recognize_face is False: 
            self.recognize_face = True  
        else: 
            self.recognize_face = False                    
class mainWindow(qtw.QWidget): # Definition der Hauptfenster-Klasse, die von QWidget abgeleitet wird
    def __init__(self):
        super().__init__() # Ruft den Konstruktor der übergeordneten Klasse auf
        self.setWindowIcon(qtg.QIcon('./logos/icon.png')) # Setzt das Fenster-Icon
        self.setWindowTitle('Offline Gesichtserkennung') # Setzt den Fenster-Titel
        self.setFixedSize(658, 690) # Setzt die feste Größe des Fensters
        header_image = qtw.QLabel()
        header_image.setPixmap(qtg.QPixmap('./logos/jadehs-logo.png'))
        header_layout = qtw.QHBoxLayout() # Legt ein Label für das Header-Bild fest
        header_layout.addStretch() 
        header_layout.addWidget(header_image)
        header_layout.addStretch()
        header_layout.setAlignment(qtc.Qt.AlignHCenter) # Setzt die Ausrichtung des Headers auf die Mitte
        header_label = qtw.QLabel('<h2>Offline Gesichtserkennung</h2>') # Legt das Label für den Header-Text fest
        header_label.setAlignment(qtc.Qt.AlignHCenter) # Setzt die Ausrichtung des Header-Texts auf die Mitte
        self.cameraButton = qtw.QPushButton('Open Camera', clicked=self.cameraButtonClick, checkable=True) # Legt den Kamera-Button fest
        self.saveFaceButton = qtw.QPushButton('Save Face', clicked=self.saveFaceButtonClick)# Legt den Speicher-Button fest
        self.recognizeFaceButton = qtw.QPushButton('Gesicht erkennen', clicked=self.recognizeFaceButtonClick) # Legt den Erkennungs-Button fest
        name_label = qtw.QLabel("Name Eingeben:") # Legt das Label für den Namen fest
        self.name_input = qtw.QLineEdit() # Legt das Eingabefeld für den Namen fest
        self.screen = qtw.QLabel() # Legt das Label für den Bildschirm fest
        self.img = qtg.QPixmap(700,480) # Legt das Pixmap für den Bildschirm fest
        self.img.fill(qtg.QColor('darkGrey')) # Füllt das Pixmap mit einem grauen Hintergrund
        self.screen.setPixmap(self.img)
        layout = qtw.QVBoxLayout()
        layout.addLayout(header_layout) # Legt das Haupt-Layout fest
        layout.addWidget(header_label)
        layout.addWidget(self.cameraButton)
        layout.addWidget(self.screen)
        form_layout = qtw.QFormLayout() # Legt das Formular-Layout fest
        form_layout.addRow(name_label, self.name_input)
        layout.addLayout(form_layout)
        layout.addWidget(self.saveFaceButton)
        layout.addWidget(self.recognizeFaceButton)
        self.setLayout(layout)
        self.show() #zeigt das Hauptfenster (mainWindow) an. Es ist eine Methode aus der Klasse "QWidget" und stellt das Fenster auf dem Bildschirm dar.
    def cameraButtonClick(self):
        print('clicked')
        status = self.cameraButton.isChecked()
        if status == True:
            self.cameraButton.setText('Close Camera') # Setze den Text des Buttons auf "Close Camera"
            self.capture = VideoCapture() # Initialisiere die Kamera
            self.capture.change_pixmap_signal.connect(self.updateImage) # Verbinde das change_pixmap_Signal mit updateImage
            self.capture.start()  # Starte die Kamera
        elif status == False:
            self.cameraButton.setText('Open Camera') # Setze den Text des Buttons auf "Open Camera"
            self.capture.stop() # Kamera stoppen
    def saveFaceButtonClick(self):
        print('Save Face button clicked')
        name = self.name_input.text() # Get the name entered by the user
        if name: # Check if a name is entered
            self.capture.addface(name)  # Pass the name to the addface method
        else:
            qtw.QMessageBox.warning(self, 'Warning', 'Please enter a name') # Show a warning message if no name is entered
    def recognizeFaceButtonClick(self):
        print('Recognize Face button clicked')
        self.capture.recognizeface()
    @qtc.pyqtSlot(np.ndarray) #decorator wird verwendet, um anzuzeigen, dass die Funktion als Slot in PyQt5 verwendet werden kann, einem Satz von Python-Bindungen für die Qt-Bibliotheken.
    def updateImage(self, image_array):
        rgb_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB) # Konvertiere das Bild von BGR nach RGB
        h,w, ch = rgb_img.shape    # Hole die Höhe, Breite und Anzahl der Kanäle des Bildes
        bytes_per_line = ch*w      # Berechne die Anzahl der Bytes pro Zeile
        convertedImage = qtg.QImage(rgb_img.data,w,h,bytes_per_line,qtg.QImage.Format_RGB888) # Konvertiere das Bild zu QImage
        scaledImage = convertedImage.scaled(700,480,qtc.Qt.KeepAspectRatio) # Skaliere das Bild
        qt_img = qtg.QPixmap.fromImage(scaledImage) # Konvertiere das skalierte Bild zu QPixmap
        self.screen.setPixmap(qt_img) # Aktualisiere das Pixmap auf dem Bildschirm
if __name__ == "__main__": # Überprüfen, ob das Skript als Hauptskript ausgeführt wird
    app = qtw.QApplication(sys.argv) # Erstelle eine QApplication
    mw = mainWindow() # Erstelle ein Hauptfenster
    sys.exit(app.exec()) # Beende die Anwendung mit dem Rückgabewert von app.exec()
