
# **Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix)**

 - Paper Id 41 : Implemented in partial fulfillment of the course BITS F312 - Neural Networks and Fuzzy Logic
 - Group Members :
	 - **Rohit Milind Rajhans** - 2017A7PS0105P
	 - **Kushagra Agrawal** - 2017A7PS0107P
	 - **Suraj Kumar** - 2017A8PS0519P
 - Assisted By : **Parth Patel**
 - Original pix2pix  (by Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros)
	 - **Github** : https://github.com/phillipi/pix2pix
	 - **Paper Link** : https://arxiv.org/pdf/1611.07004.pdf

## Objectives

 -   Read the paper thoroughly and understand all the concepts (like conditional GANs, generator’s architecture and its analysis, discriminator’s architecture and its analysis, choice of objective function and its analysis), therein (including the Appendix).
 - For the generator, implement only the U-Net Architecture as described in the paper (see Appendix).
 - For the discriminator, implement the 70x70 PatchGAN architecture as specified in the paper (see Appendix).
 - The loss function to be used is specified by equation 4 in the paper, with minor changes as specified in section 3.3.
 - Train a GAN model on any one dataset of choice (from among the ones mentioned in section 4) and show the resulting images generated.
## Implementation
 - We have used the **Keras** library for implementing the Pix2Pix architecture. The generator and discriminator architectures have been implemented as specified in the paper.
 - We have trained the model on two datasets : **Maps** (Aerial view -> Map view) and **Cityscapes** (Street View -> Label).
 - The model has been trained for 140 epochs on the Maps dataset and 100 epochs on the Cityscapes dataset.
## Usage
 - Install Python 3 and pip
 - Run `pip install -r requirements.txt` to install all the dependencies
 - **Directory Structure** :
	 - assets
		 - datasets
			 - cityscapes
				 - compressed/
				 - train/
				 - val/
			 - maps
				 - compressed/
				 - train/
				 - val/
		 - plots
			 - cityscapes
				 - epochs_20/
				 - epochs_100/
			 - maps
				 - epohs_20/
				 - epochs_140/
	 - models
		 - cityscapes
			 - `model_g_297500.h5` ....
		 - maps
			 - `model_g_153440.h5` ...
	 - networks
		 - `discriminator.py`
		 - `generator.py`
		 - `GAN.py`
	 - `main.py`
	 - `predict.py`
	 - `preprocessing.py`
	 - `pix2pix.ipynb`
 - **Training the data** : To start training from scratch, run execute `start_training()` from `main.py`. To resume training from a checkpoint, execute `resume_training()` from `main.py`. The model state is saved every 5 epochs in the folder models/dataset_name.
 - **Visualize predictions** : To run model on train data execute `generate_prediction()` from `predict.py`. To run model on validation data execute `load_and_plot()` from `predict.py`.
 - **All-in-one** : The file `pix2pix.ipynb` contains all the functions in one place for direct execution. Run using Jupyter Notebook.
## Results
All the generated results can be viewed in `assets/plots/dataset_name/`. A comparative plot between a model trained on less epochs and a model trained on higher number of epochs can also be found in the same folder. This results have been generated randomly from the training and validation datasets.
To generate more results, follow instructions mentioned above. 
**Note** : In comparative plots, the top image represents the model trained on lower epochs and the images further down represent models that are trained on higher epochs.

## Limitations

 - Due to low computing power at hand, the model could only be trained for 140 and 100 epochs on the Maps and Cityscapes datasets respectively.
 - Although, the training enables addition of random jitter as mentioned in the Appendix (Section 6.2.) of the paper, our saved models haven't been generated using random jitter. This is because, this preprocessing step consumed significantly more time.
 - The generated results have not been evaluated using the FCN score metric as mentioned in the paper.
## References :

 - Paper : https://arxiv.org/pdf/1611.07004.pdf
 - Original Work : https://github.com/phillipi/pix2pix
 - Cityscapes dataset : https://www.cityscapes-dataset.com
 - Pix2Pix datasets : http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/
 - https://keras.io/
 - https://www.tensorflow.org/tutorials/generative/pix2pix
 - https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/
 

