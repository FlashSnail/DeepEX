import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
      name='deepex',
      version='0.0.15',
      description='DeepEX is a universal convenient frame with keras and Tensorflow. You can get well-known Wide&Deep model such as DeepFM here. Or, you can define you custom model use this frame.',
      long_description=long_description,
      long_description_content_type="text/markdown",	
      url='https://github.com/FlashSnail/DeepEX',
      download_url='https://github.com/FlashSnail/DeepEX/tags',
      author='zhangzehua',
      author_email='zzh_0729@foxmail.com',
	  python_requires=">=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*",
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
      ],
      keywords=['deep learning','keras','tensorflow','wide&deep','frame']      
      )
