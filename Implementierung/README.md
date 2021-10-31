# Implementierung - Untersuchung der Auswirkungen von Quantisierung und Pruning auf Convolutional Neural Networks für Mobilgeräte
Dieser Ordner enthält den Code zu der Bachelorarbeit "Untersuchung der Auswirkungen von Quantisierung und Pruning auf Convolutional Neural Networks für Mobilgeräte" von Julian Hoever.


## Setup
Für diese Arbeit wird Python3 mit pip benötigt. Installieren der benötigten Pakete:
```
pip install -r requirements.txt
```


## Ablauf
Im folgenden wird beschrieben, wie die einzelnen Schritte in der Arbeit mittels dieses Codes durchgeführt wird.


### Modelle trainieren
Um ein Modell zu trainieren (Arbeit S. 22), gibt es in ```config.architectures``` für jede Architektur eine Konfiguration, welche definiert wie die Architektur erzeugt wird und wie die Daten vorverarbeitet werden. Bereits vorhandene Konfigurationen sind:

- mobilenetv1
- mobilenetv2
- mobilenetv3_large
- mobilenetv3_small
- efficientnet-b0
- mobilenetv3_large_minimalistic
- mobilenetv3_small_minimalistic

Damit kann anschließend eine Architektur wie folgt trainiert werden:
```
python3 train_model.py <model_config> <path_to_save_model>
```

Um eine der MobileNetV3 Minimalistic Architekturen zu trainieren muss folgender Befehl ausgeführt werden:

```
python3 train_mnv3_minimalistic.py <model_config> <path_to_save_model>
```


### Trainierte Modelle prunen
Sollen die trainierten Modelle gepruned werden (Arbeit S. 26), gibt es, wie in der Arbeit bereits beschrieben, 3 Pruningkonfigurationen (Arbeit S. 27 Tabelle 3.1) für jeden Sparsity-Wert, welche in dem Dictionary ```config.pruning.sparsity``` definiert sind:

- 30%
- 60%
- 90%

Außerdem sind in dem Modul ```config.pruning``` auch die regulären Ausdrücke für die ```prune_model.prune_layer``` Funktion definiert (Arbeit S. 26).

Um nun eine Architektur zu prunen muss folgender Befehl ausgeführt werden:

```
python3 prune_model.py <model_config> <pruning_config> <trained_model> <path_to_save_model>
```

Um eine der MobileNetV3 Minimalistic Architekturen zu prunen muss folgender Befehl ausgeführt werden:

```
python3 prune_mnv3_minimalistic.py <model_config> <pruning_config> <trained_model> <path_to_save_model>
```


### Modelle quantisieren
Um trainierte oder geprunte Modelle zu quantisieren muss folgender Befehl ausgeführt werden:

```
python3 quantize_model.py <model_config> <model_path> <quantized_tflite_path>
```


### Trainierte/Geprunte Modelle in TFLite konvertieren
Um ein trainiertes oder gepruntes Modell in das TFLite FlatBuffer Format zu konvertieren, ohne eine Quantisierung anzuwenden, muss folgender Befehl ausgeführt werden:
```
python3 model_to_tflite.py <model_path> <tflite_path>
```

### Modelle evaluieren
Nachdem alle Modelle im TFLite Format vorliegen, können diese wie folgt evaluiert werden:

#### TFLite evaluieren
```
python3 evaluate_tflite.py <model_config> <tflite_path> <json_output_path>
```

#### Klassenweise Evaluation der TFLite Modelle
```
python3 evaluate_tflite_classwise.py <model_config> <tflite_path> <json_output_path>
```

#### Komprimierte Dateigröße der Modelle ermitteln
```
python3 compressed_file_size.py <tflite_path>
```

#### Auswerten auf dem Raspberry Pi 4
Damit die Modelle auf dem Raspberry Pi 4 ausgewertet werden können (Inferenzzeit und Hauptspeicherbedarf), müssen die TFLite Modelle und die 2 Dateien aus dem Ordner ```platform_benchmark``` auf den Raspberry Pi übertragen werden. Anschließend kann folgender Befehl ausgeführt werden:

```
./run_benchmark.sh <tflite_path>
```


## Ergebnisse der Arbeit
Die trainierten Modelle, welche in dieser Arbeit verwendet wurden, befinden sich in dem Ordner ```saved_models``` und die daraus entstandenen TFLite Dateien, befinden sich in dem Ordner ```tflite_files```. Nach Anwendung sämtlicher Evaluationsschritte auf diese TFLite Modelle, wurden die daraus resultierenden Ergebnisse in die CSV-Dateien ```results/architecture_metrics.csv``` und ```results/mobilenetv3_minimalistic_metrics.csv``` übertragen. Mittels dieser CSV Dateien wurde in dem Jupyter Notebook ```evaluation_and_plots.ipynb``` die Auswertung der Ergebnisse vorgenommen und die in der Arbeit verwendeten Plots erstellt.


## Beispiel: DepthwiseConv2D Layer werden nicht gepruned
In der Arbeit wurde auf Seite 37 beschrieben, dass die ```tf.keras.layers.DepthwiseConv2D``` Schicht nicht gepruned werden konnte. Dazu ist in dem Jupyter Notebook ```prune_depthwiseconv2d_min_example.ipynb``` ein Minimalbeispiel enthalten, welches diese Problematik darstellt.


## Bash Skripte
Alle Bash Skripte dienten in dieser Arbeit lediglich als Hilfestellung, um gebündelt die einzelnen Schritte des Ablaufs durchführen zu können. Diese Skripte müssten für neu trainierte Modelle angepasst werden.
