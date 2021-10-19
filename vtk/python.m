%setenv('PATH', ['~/opt/anaconda3/envs/vtk-dev/bin/python', pathsep, getenv('PATH')])

%pe = pyenv('Version', '~/opt/anaconda3/envs/vtk-dev/bin/python')

%pyenv("ExecutionMode","OutOfProcess")

% pyrunfile("empty.py")

%pyrun("import sys")
%pyrun("vis_vtk as cfvv")
% pyrun("PyQt5.QtWidgets import *")
% pyrun("from PyQt5.QtCore import *")
% %if __name__ == "__main__":
% pyrun("app = QApplication(sys.argv)")
% pyrun("ex = cfvv.MainWindow()")
% pyrun("ex.show()")
% pyrun("sys.exit(app.exec_()")

system('python empty.py');