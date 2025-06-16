# In this file you will find information about the programs in this folder like libraries and other.<br>
For the programs to works on windows, you need (command to install):<br>
-pip (if you already have python installed python -m ensurepip --upgrade)<br>
-OpenCV (pip install opencv-python)<br>
-torch (pip install torch torchvision torchaudio)<br>
-ultralytics (pip install ultralytics)<br>
-deep-sort-realtime (pip install deep-sort-realtime)<br>
-numpy (pip install numpy)<br>

# If you are using a ZED camera<br>
You need "pyzed.sl"<br>
For it, you need to download the ZED SDK from the website: https://www.stereolabs.com/developers/ <br>
then:<br>
-pip install "C:\Program Files (x86)\ZED SDK\pyzed\whl\pyzed-3.8-cp310-cp310-win_amd64.whl"<br>
but you need to change some part so it works for you<br>
The most important one is cp310 because it is your python version<br>
cp38 for Python 3.8<br>
cp39 for Python 3.9<br>
cp310 for Python 3.10<br>
cp311 for Python 3.11<br>
and now you can test the install with this:<br>
import pyzed.sl as sl<br>
print(sl.Camera())<br>

