![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)
# Online Privacy Preservation for Person Re-Identification
This repository is the PyTorch source code implementation of 
[Online Privacy Preservation for Person Re-Identification]() and is currently being reviewed at CVPR 2023. In the following is an instruction to use the code
to train and evaluate the OPP model on the [Market-1501](
https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html) dataset.

### Requirements

Code was tested in virtual environment with Python 3.8 and 1 * RTX 3090 24G. 
The full installed packages in our virtual enviroment  were presented in the 'requirements.txt' file. 

### Data preparation
Download [Market1501 Dataset](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html) [[Google]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view) [[Baidu]](https://pan.baidu.com/s/1ntIi2Op)

Preparation: Open and edit the script `prepare_market.py` in the editor. Change the fifth line in `prepare_market.py` to your download path. Run the following script in the terminal to put the images with the same id in one folder:
```bash
python prepare_market.py
```
We use 'tree' command to show the prejoct's directory listing
in a neater format for different subdirectories, files and folders in our experiment as follows:
```
.
├── argpaser.py
├── builder.py
├── dreamer.py
├── DukeMTMC-reID
│   ├── bounding_box_test
│   ├── bounding_box_train
│   ├── CITATION.txt
│   ├── LICENSE_DukeMTMC-reID.txt
│   ├── LICENSE_DukeMTMC.txt
│   ├── pytorch
│   ├── query
│   └── README.md
├── evaluator.py
├── final_images
│   └── output_00106.png
├── log
│   └── events.out.tfevents.1661582743.server
├── lossfun.py
├── main.py
├── Market-1501
│   ├── bounding_box_test
│   ├── bounding_box_train
│   ├── gt_bbox
│   ├── gt_query
│   ├── pytorch
│   ├── query
│   └── readme.txt
├── model.py
├── net
│   ├── requirements.txt
│   └── teacher.pth
├── prepare_dukemtmc.py
├── prepare_market.py
├── __pycache__
│   ├── argpaser.cpython-38.pyc
│   ├── builder.cpython-38.pyc
│   ├── dreamer.cpython-38.pyc
│   ├── lossfun.cpython-38.pyc
│   ├── model.cpython-38.pyc
│   ├── trainer.cpython-38.pyc
│   └── util.cpython-38.pyc
├── README.md
├── requirements.txt
├── trainer.py
├── util.py
└── wget-log

```
Futhermore, you also can test our code on [DukeMTMC-reID Dataset]([GoogleDriver](https://drive.google.com/open?id=1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O) or ([BaiduYun](https://pan.baidu.com/s/1jS0XM7Var5nQGcbf9xUztw) password: bhbh)).
### Model preparation
Please find the pretrained teacher Re-ID model in
[BaiduPan](https://pan.baidu.com/s/15h4UAkAMghtVCZUcz24OFw) (password: bwsa).
After downloading *teacher.pth*, please put it into *./net/* folder.


### Run the code

Please enter the main folder, Train the OPP model by
```bash
python main.py --dream_person 1 --ms 5000 --T 2.0 --lamb 0.05 --sigma 1.0 --batch_size 32  --data_dir your_project_path/OPP-PersonReID/Market-1501/pytorch/
```
`--dream_person` num of person for dreaming.

`--ms` memory size of dreamer.

`--T` temperature for target generation

`--lamb` coefficient for the mix loss function

`--sigma` parameter of Gaussian Kernel

`--batch_size` training batch size.

`--data_dir` the path of the training data.

### Monitoring training progress
```
tensorboard.sh --port 6006 --logdir your_project_path/log
```

### Contact
If you have any problem please email me at honestwsh@gmail.com

