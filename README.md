# contractors_toolbox

# Deploy the toolbox to the submission notebook 
It only matters when doing the submission.

In the root directory of the repo, do:

```sh
pip wheel . -w wheels --use-feature=in-tree-build
```

It generates a folder wheel, which contains the toolbox package and dependencies:

```
❯ ls
Deprecated-1.2.13-py2.py3-none-any.whl                        packaging-21.0-py3-none-any.whl
Markdown-3.3.4-py3-none-any.whl                               pandas-1.3.3-cp37-cp37m-macosx_10_9_x86_64.whl
Pillow-8.3.2-cp37-cp37m-macosx_10_10_x86_64.whl               protobuf-3.18.1-cp37-cp37m-macosx_10_9_x86_64.whl
PyPDF2-1.26.0-py3-none-any.whl                                pyDeprecate-0.3.1-py3-none-any.whl
PyYAML-5.4.1-cp37-cp37m-macosx_10_9_x86_64.whl                pyasn1-0.4.8-py2.py3-none-any.whl
SimpleITK-2.1.1-cp37-cp37m-macosx_10_9_x86_64.whl             pyasn1_modules-0.2.8-py2.py3-none-any.whl
Werkzeug-2.0.2-py3-none-any.whl                               pydicom-2.2.2-py3-none-any.whl
absl_py-0.14.1-py3-none-any.whl                               pyparsing-2.4.7-py2.py3-none-any.whl
aiohttp-3.7.4.post0-cp37-cp37m-macosx_10_14_x86_64.whl        python_dateutil-2.8.2-py2.py3-none-any.whl
async_timeout-3.0.1-py3-none-any.whl                          pytorch_lightning-1.4.9-py3-none-any.whl
attrs-21.2.0-py2.py3-none-any.whl                             pytz-2021.3-py2.py3-none-any.whl
cachetools-4.2.4-py3-none-any.whl                             requests-2.26.0-py2.py3-none-any.whl
certifi-2021.5.30-py2.py3-none-any.whl                        requests_oauthlib-1.3.0-py2.py3-none-any.whl
chardet-4.0.0-py2.py3-none-any.whl                            rsa-4.7.2-py3-none-any.whl
charset_normalizer-2.0.6-py3-none-any.whl                     scikit_learn-1.0-cp37-cp37m-macosx_10_13_x86_64.whl
click-8.0.1-py3-none-any.whl                                  scipy-1.7.1-cp37-cp37m-macosx_10_9_x86_64.whl
cycler-0.10.0-py2.py3-none-any.whl                            setuptools-58.2.0-py3-none-any.whl
fsspec-2021.10.0-py3-none-any.whl                             six-1.16.0-py2.py3-none-any.whl
future-0.18.2-py3-none-any.whl                                tensorboard-2.6.0-py3-none-any.whl
google_auth-1.35.0-py2.py3-none-any.whl                       tensorboard_data_server-0.6.1-py3-none-macosx_10_9_x86_64.whl
google_auth_oauthlib-0.4.6-py2.py3-none-any.whl               tensorboard_plugin_wit-1.8.0-py3-none-any.whl
grpcio-1.41.0-cp37-cp37m-macosx_10_10_x86_64.whl              threadpoolctl-3.0.0-py3-none-any.whl
humanize-3.12.0-py3-none-any.whl                              toolbox-0.0.1-py3-none-any.whl
idna-3.2-py3-none-any.whl                                     torch-1.9.1-cp37-none-macosx_10_9_x86_64.whl
importlib_metadata-4.8.1-py3-none-any.whl                     torchio-0.18.57-py2.py3-none-any.whl
joblib-1.0.1-py3-none-any.whl                                 torchmetrics-0.5.1-py3-none-any.whl
kiwisolver-1.3.2-cp37-cp37m-macosx_10_9_x86_64.whl            tqdm-4.62.3-py2.py3-none-any.whl
matplotlib-3.4.3-cp37-cp37m-macosx_10_9_x86_64.whl            typing_extensions-3.10.0.2-py3-none-any.whl
monai-0.7.0-202109240007-py3-none-any.whl                     urllib3-1.26.7-py2.py3-none-any.whl
multidict-5.2.0-cp37-cp37m-macosx_10_9_x86_64.whl             wheel-0.37.0-py2.py3-none-any.whl
natsort-7.1.1-py3-none-any.whl                                wrapt-1.13.1-cp37-cp37m-macosx_10_9_x86_64.whl
nibabel-3.2.1-py3-none-any.whl                                yarl-1.6.3-cp37-cp37m-macosx_10_14_x86_64.whl
numpy-1.21.2-cp37-cp37m-macosx_10_9_x86_64.whl                zipp-3.6.0-py3-none-any.whl
oauthlib-3.1.1-py2.py3-none-any.whl
```

Then we can upload the wheels as dataset.
I suggest we separate the wheels to two:
1. Our source code wheel toolbox-0.0.1-py3-none-any.whl
2. The other dependency wheels.

The size of all dependencies is 300 MB. But our source code wheel is only 63 KB.
If we separate them, then it's much much faster to re-upload our source code to the notebook.

In the notebook, the way to use it is to add the dataset, then `pip install` it.
```
!pip install --no-index --find-links ../input/{datasetname}/{wheel_location} toolbox
```

For example, my source code was put under the dataset `contractors-toolbox-v001-pohan`, and under the `wheels/` folder:

```
!pip install --no-index --find-links ../input/contractors-toolbox-v001-pohan/wheels/ toolbox
```

To add two different places of wheels, we can simply add another find links like:

```
!pip install --no-index \
 --find-links ../input/contractors-toolbox-v001-dependencies/ \
 --find-links ../input/contractors-toolbox-v001-source-code /
 toolbox
```