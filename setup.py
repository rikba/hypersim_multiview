from setuptools import setup

setup(
   name='hypersim_multiview',
   version='1.0',
   description='Implements and analyses frame-to-frame pixel reprojection for the Hypersim Evermotion data set.',
   author='Rik Bähnemann',
   author_email='brik@ethz.ch',
   packages=['hypersim_multiview'],
   install_requires=['torch', 'numpy', 'matplotlib', 'opencv-python', 'pandas'],
   scripts=[
            'examples/reprojection_analysis',
            'examples/reprojection_example',
           ]
)
