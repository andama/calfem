import vtk
from vedo import Grid, Tensors, show

domain = Grid(resx=5, resy=5, c='gray')

print(domain)

# Generate random attributes on a plane
ag = vtk.vtkRandomAttributeGenerator()
ag.SetInputData(domain.polydata())
ag.GenerateAllDataOn()
ag.Update()

print(ag.GetOutput())

ts = Tensors(ag.GetOutput(), scale=0.1)
ts.print()

show(domain, ts).close()