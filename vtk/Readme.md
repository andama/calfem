# Readme

### vis_vtk.py

Innehåller funktioner som kopplar modellen till vtk. Denna är tänkt att främst använda vtk-funktioner för att man ska slippa göra detta i modellen man jobbar med.

### core_beam_extensions.py

Innehåller inget just nu, är tänkt att innehålla funktioner för att rita plotta tvärsnittskrafter eller liknande när det tillkommer.

### QtVTKMainWindow.ui

Fönster skapat i Qt Designer som krävs för att köra kod just nu. Väldigt lite funktionalitet är inbyggd i den just nu (endast en frame för vtk att rendera samt en knapp för att återställa kameran) men är tänkt att byggas ut med funktionalitet.

### 3Dbeam

Exempel på en 3D-balk med endast två element. Är tänkt som bas för att bygga viss funktionalitet för balkar, främst initiellt. Renderar endast noder och element samt koordinataxlar i nuläget. Calfem-modellen är i en separat fil i mappen.

### 3Dtruss

Exempel på en fackverksbro av 3D-balkar och 3D-stänger. Är tänkt som bas för att bygga ut mer avancerad funktionalitet för balkar, stänger, och fjädrar senare. Fungerar inte alls i nuläget.
