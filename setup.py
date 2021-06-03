from setuptools import setup

setup(
   name='vision_3d_utils',
   version='1.0',
   description='Modules to handle hypersim dataset and compute vision metrics.',
   author='Rik Bähnemann',
   author_email='brik@ethz.ch',
   packages=['vision_3d_utils'],  #same as name
   install_requires=['torch', 'opencv-python'], #external packages as dependencies
   scripts=[
            'examples/reprojection_analysis',
            'examples/reprojection_example_numpy',
            'examples/reprojection_example_torch',
           ]
)
