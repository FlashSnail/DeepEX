import setuptools
setuptools.setup(name='deepex',
      version='0.0.4',
      description='DeepEX is a universal convenient frame with keras and Tensorflow. You can get well-known Wide&Deep model such as DeepFM here. Or, you can define you custom model use this frame.',
      url='https://github.com/FlashSnail/DeepEX',
      author='zhangzehua',
      author_email='zzh_0729@foxmail.com',
      packages=setuptools.find_packages(),
      classifiers=[
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 2",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      install_requires=[
        'numpy>=1.16.2',
        'keras>=2.2.4'
        #'Tensorflow>=1.12.0'
      ]      
      )
