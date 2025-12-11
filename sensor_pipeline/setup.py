from setuptools import find_packages, setup
import os
from glob import glob 

package_name = 'sensor_pipeline'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # install launch and config
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('sensor_pipeline/config/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('sensor_pipeline/config/*.rviz')),
    ],
    install_requires=['setuptools', 'pyyaml'],
    zip_safe=True,
    maintainer='yunjinli',
    maintainer_email='yunjin.li0817@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_info_override = sensor_pipeline.nodes.camera_info_override:main',
            'colorize_pointcloud = sensor_pipeline.nodes.colorize_pointcloud:main',
            'grayscale_pointcloud = sensor_pipeline.nodes.grayscale_pointcloud:main',
            'depth_overlay = sensor_pipeline.nodes.depth_overlay:main',
        ],
    },
)
