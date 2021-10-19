# Readme

Mapp för utveckling av visualiseringsverktyg i CALFEM för Python som bygger på *vtk* (Visualization ToolKit)

Innan något kan köras så måste både *calfem-python* och *vtk* vara installerat i aktuellt python-env med hjälp av pip:

- pip install calfem-python
- pip install vtk

**Kan inte få odeformerad mesh att funka för att det inte finns enkelt sätt att orientera element i _vtk_. Därför funkar inte _3Dbeam.py_ & _3Dtruss.py_ som de ska. Se _vedo_ & _pyvista_ för fungerande exempel. Vidare utväckling kan komma att ske i _vedo_/_pyvista_**

------

### vis_vtk.py

Huvudmodul för att göra allt relaterat till *vtk*. Innehåller funktioner som startar, kopplar modellen till samt interagerar med *vtk*.

Denna är tänkt att vara uppbyggd så att användaren som jobbar med en FE-modell ska behöva använda så få *vtk*-funktioner som möjligt direkt, och på så sätt förenkla visualisering genom att inte ändra sättet man jobbar i CALFEM för mycket.

En klass *MainWindow* interagerar med QtMainWindow.ui och en klass *beam* innehåller funktioner för visualisering av balkelement. Klasser för andra typer av element är tänkt att tillkomma här.

### core_beam_extensions.py

Innehåller inget just nu, är tänkt att innehålla funktioner för att möjliggöra plottning av tvärsnittskrafter eller liknande för balkar när det tillkommer.

### QtVTKMainWindow.ui

Fönster skapat i Qt Designer som krävs för att köra kod just nu. Väldigt lite funktionalitet är inbyggd i den just nu (endast en frame för *vtk* att rendera samt en knapp för att återställa kameran) men är tänkt att byggas ut med funktionalitet. Meny-objekten ger lite indikation på tänkt funktionalitet.

Fönstret funkar nu att startas oberoende av en modell, för att testa import av modeller från MATLAB & VTK-filer.

### empty.py

En fil som endast innehåller vad som är nödvändigt för att öppna Qt-fönstret med VTK-renderaren. För att testa funktionalitet utan en modell.

### 3Dbeam.py

Exempel på en 3D-balk med endast två element. Är tänkt som bas för att bygga viss funktionalitet för balkar, främst initiellt. Renderar endast noder och element samt koordinataxlar i nuläget.

Just nu används detta exempel för att implementera:

- Filter, ett filter för odeformerad mesh & deformerad mesh testas initiellt.
- Grundläggande interaktion med modellen *vtk*:s actor mode.
- Olika sätt att redovisa spänningar

### 3Dtruss.py

Exempel på en fackverksbro av 3D-balkar och 3D-stänger. Är tänkt som bas för att bygga ut mer avancerad funktionalitet för balkar, stänger, och fjädrar senare. Fungerar inte i nuläget eftersom *vis_vtk.py* inte kan hantera diskontinuerlig geometri ännu.
