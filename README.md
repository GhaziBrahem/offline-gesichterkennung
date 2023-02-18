# Offline-Gesichterkennung
Der Code ist für die Offline-Gesichtserkennung implementiert, bei der Daten aus JSON-Dateien gelesen, in Bilder umgewandelt und die Erkennung am aktuellen Frame durchgeführt wird.

Der Code definiert mehrere Funktionen zum Codieren von Gesichtsmerkmalen und zum Berechnen des Abstands zwischen Gesichtscodierungen. Es initialisiert dann eine leere Liste für Bilder und ihre entsprechenden Namen und liest JSON-Dateien ein, die Gesichtsdaten enthalten. Das Skript konvertiert das Numpy-Array aus jeder JSON-Datei in ein Bild und fügt es der Liste der Bilder zusammen mit seinem entsprechenden Namen hinzu. Das Skript verwendet dann die Gesichts-Codierungsfunktion, um die Gesichts-Codierungen für alle Bilder zu berechnen und speichert sie in einer Liste.

Der Code initialisiert dann ein Videoaufnahmeobjekt und liest Frames von der Kamera ein. Er ändert die Größe des Frames und konvertiert ihn vom BGR- in das RGB-Format. Wenn die recognize_face-Flagge auf true gesetzt ist, führt der Code eine Gesichtserkennung durch, indem er die Gesichtscodierung des aktuellen Frames berechnet und sie mit den Gesichtscodierungen in der Datenbank vergleicht. Wenn eine Übereinstimmung gefunden wird, zeigt er den Namen der erkannten Person auf dem Bildschirm an.

folge die Installationsschritte 
mit command Prumpet erstelle eine virtuelle envirement und installiere alle requirements

      *envirement erstellen:  python -m venv gesichterkennung
      *envirement aktivieren: .\gesichterkennung\Scripts\activate     
      *direkt nach dem aktivieren den verzeichnen "Gesichterkennung" im envirement einfügen
        --lade die datei "shape_predictor_68_face_landmarks.dat"  --> https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat
        --füge sie nachher mit aktuellen Modelle unter \offline-gesichterkennung\Gesichterkennung\models
      *requirement insttallieren: pip install -r PyQt5 numpy opencv-python dlib      
      *führe offline_gesicht_erkenner.py  mit Visual Studio Code aus
