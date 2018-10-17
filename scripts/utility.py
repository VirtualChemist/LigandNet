# Utility script for feature generation using Mayachemtools. Generates TPATF features.
# Md Mahmudulla Hassan
# The University of Texas at El Paso
# Last Modified: 10/12/2018

import os
from rdkit import Chem
from rdkit.Chem import AllChem
import tempfile
import shutil
import subprocess
import errno
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
import base64
import cairosvg

MAYACHEMTOOLS_DIR = os.path.abspath("mayachemtools")

class SmilesToImage:
    def __init__(self, smiles):
        self.smiles = smiles
        self.temp_dir = tempfile.mkdtemp()
        self.png_file = os.path.join(self.temp_dir, "mol.png")
        self.svg_file = os.path.join(self.temp_dir, "mol.svg")

    def toPNG(self):
        # Set the drawing options
        DrawingOptions.atomLabelFontSize = 55
        DrawingOptions.dotsPerAngstrom = 100
        DrawingOptions.bondLineWidth = 3.0
        
        # Conver the SMILES into a mol object
        m = Chem.MolFromSmiles(self.smiles)
        # Calculate the coordinates
        AllChem.Compute2DCoords(m)
        # Draw the mol
        Draw.MolToFile(m, self.svg_file)
        # Convert the svg to png (for high quality image)
        cairosvg.svg2png(url=self.svg_file, write_to=self.png_file)
        # Convert into binary and return
        binary_image = None
        with open(self.png_file, "rb") as f:
            binary_image = base64.b64encode(f.read())
            shutil.rmtree(self.temp_dir)

        return binary_image


class FeatureGenerator:
    
    def __init__(self, smiles):
        self.smiles = smiles
        self.temp_dir = tempfile.mkdtemp()
    
    def toString(self):
        return self.smiles
    
    def toSDF(self):
        # Try to get the rdkit mol
        mol = Chem.MolFromSmiles(self.smiles)
        #if mol == None: raise("Error in mol object") 
        # Compute 2D coordinates
        AllChem.Compute2DCoords(mol)
        mol.SetProp("smiles", self.smiles)
        #self.sdf_filepath = os.path.join(self.temp_dir, "temp.sdf")
        w = Chem.SDWriter(os.path.join(self.temp_dir, "temp.sdf"))
        w.write(mol)
        w.flush()
    
    def toTPATF(self):
        features = []
        script_path = os.path.join(MAYACHEMTOOLS_DIR, "bin/TopologicalPharmacophoreAtomTripletsFingerprints.pl")

        # Generate the sdf file
        self.toSDF()

        # Now generate the TPATF features
        # Check if the sdf file exists
        if not os.path.isfile(os.path.join(self.temp_dir, "temp.sdf")): 
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), os.path.join(self.temp_dir, "temp.sdf"))

        # Execute perl script
        # Check if the sdf file exists
        command = "perl " + script_path + " -r " + os.path.join(self.temp_dir, "temp") + " --AtomTripletsSetSizeToUse FixedSize -v ValuesString -o " + os.path.join(self.temp_dir, "temp.sdf")
        os.system(command)      
        # Alternative (better/preferred) approach. Not worked when tried last time.  
        #pipe = subprocess.Popen(["perl", script_path, " -r " + os.path.join(self.temp_dir, "temp.sdf") + " --AtomTripletsSetSizeToUse FixedSize -v ValuesString -o temp.sdf"], stdin=subprocess.PIPE)
        #pipe.stdin.write("some/file/path")
        #pipe.stdin.close()

        # Check if the csv file is there.
        if not os.path.isfile(os.path.join(self.temp_dir, "temp.csv")): 
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), os.path.join(self.temp_dir, "temp.csv"))

        # Filter the output (features) and convert that into a list
        with open(os.path.join(self.temp_dir, "temp.csv"), 'r') as f:
            for line in f.readlines():
                if "Cmpd" in line:
                    line = line.split(';')[5].replace('"','')
                    features = [int(i) for i in line.split(" ")]

        # Clean up the temporary files
        self._cleanup()
        return features
       
    def _cleanup(self):
        shutil.rmtree(self.temp_dir)
