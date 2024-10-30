import py3Dmol
import math

def showxyz_fromxyzfile(xyz_filename, addlabel=True):
    with open(xyz_filename, 'r') as file:
        xyz_content = file.read()
 
    view = py3Dmol.view(width=400, height=300)
    view.addModel(xyz_content, 'xyz')
    view.setStyle({'stick': {'colorscheme': 'Jmol'}})
    if addlabel:
        view.addPropertyLabels('index',
                            {'not': {'elem': 'H'}}, 
                            {'fontSize': 10, 
                                'color':'black'})
                                # 'fontColor': 'black',
                                # 'showBackground': False});
    view.zoomTo()
    return view.show()

def showxyz_fromxyzstr(xyz_str, addlabel=True):
    xyz_content = xyz_str
    view = py3Dmol.view(width=400, height=300)
    view.addModel(xyz_content, 'xyz')
    view.setStyle({'stick': {'colorscheme': 'Jmol'}})
    if addlabel:
        view.addPropertyLabels('index',
                            {'not': {'elem': 'H'}}, 
                            {'fontSize': 10, 
                                'color':'black'})
                                # 'fontColor': 'black',
                                # 'showBackground': False});
    view.zoomTo()
    return view.show()

def showxyzs_fromxyzfile(xyz_files, columns=5):
    columns = 5
    w = 150
    h = 150
    rows = int(math.ceil(float(len(xyz_files)) / columns))
    w = w * columns
    h = h * rows
    
    # Initialize Layout
    view = py3Dmol.view(width=w, height=h, linked=False, viewergrid=(rows, columns))
    
    # Initialize starting positions
    x, y = 0, 0
    
    # Loop through XYZ files
    for xyz_file in xyz_files:
        with open(xyz_file, 'r') as f:
            xyz_content = f.read()
        
        view.addModel(xyz_content, 'xyz', viewer=(x, y))
        view.zoomTo(viewer=(x, y))
        
        # label_coord = {'x':0, 'y':0, 'z':0}
        # view.addLabel('Hi', {'fontColor': 'black', 'fontSize': 10, 'backgroundOpacity': 0, 'position': coords}, viewer=(x, y))
    
        # Update y and x for grid placement
        if y + 1 < columns:  # Fill in columns
            y += 1
        else:
            x += 1
            y = 0
    
    view.setStyle({'stick': {'colorscheme': 'Jmol'}})
    view.show()
    
def showxyzs_fromxyzstr(xyz_strs, columns=5):
    columns = 5
    w = 150
    h = 150
    rows = int(math.ceil(float(len(xyz_strs)) / columns))
    w = w * columns
    h = h * rows
    
    # Initialize Layout
    view = py3Dmol.view(width=w, height=h, linked=False, viewergrid=(rows, columns))
    
    # Initialize starting positions
    x, y = 0, 0
    
    # Loop through XYZ files
    for xyz_str in xyz_strs:
        xyz_content = xyz_str
        
        view.addModel(xyz_content, 'xyz', viewer=(x, y))
        view.zoomTo(viewer=(x, y))
        
        # label_coord = {'x':0, 'y':0, 'z':0}
        # view.addLabel('Hi', {'fontColor': 'black', 'fontSize': 10, 'backgroundOpacity': 0, 'position': coords}, viewer=(x, y))
    
        # Update y and x for grid placement
        if y + 1 < columns:  # Fill in columns
            y += 1
        else:
            x += 1
            y = 0
    
    view.setStyle({'stick': {'colorscheme': 'Jmol'}})
    view.show()
    

def showxyz_fromxyzfile(xyz_filename):
    with open(xyz_filename, 'r') as file:
        xyz_content = file.read()
 
    view = py3Dmol.view(width=400, height=300)
    view.addModel(xyz_content, 'xyz')
    view.setStyle({'stick': {'colorscheme': 'Jmol'}})
    view.addPropertyLabels('index',
                           {'not': {'elem': 'H'}}, 
                           {'fontSize': 10, 
                            'color':'black'})
                            # 'fontColor': 'black',
                            # 'showBackground': False});
    view.zoomTo()
    return view.show()

def showvib_fromxyzstr(xyz_str):
    xyz_content = xyz_str
    view = py3Dmol.view(width=400, height=300)
    view.addModel(xyz_str,'xyz',{'vibrate': {'frames':10,'amplitude':1}})
    view.setStyle({'stick': {'colorscheme': 'Jmol'}})
    view.addPropertyLabels('index',
                           {'not': {'elem': 'H'}}, 
                           {'fontSize': 10, 
                            'color':'black'})
                            # 'fontColor': 'black',
                            # 'showBackground': False});
    view.setBackgroundColor('0xeeeeee')
    view.animate({'loop': 'backAndForth'})
    view.zoomTo()
    return view.show()
