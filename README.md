# celestial-object-detection
An image classifier to detect celestial objects using fastai library and Transfer Learning. It is meant to be used as a template for building image classifiers using fastai library.

The data is collected by using duckduckgo image search.

**Note:** Currently the model can only detect 8 planets.


## Instructions:

Create a virtual environment (highly suggested), Python version: 3.10.x
```
python -m venv venv
```
Activate the enviroment
- Linux
```
source venv/bin/activate
```
- Windows
```
.\venv\Scripts\activate
```
Install the required libraries
```
pip install -r requirements.txt
```

## Training

Modify the `train.py` according your requirements and run it to train and export the model
```
python train.py
```
You can use `train.ipynb` for a more cellular approach.

The following are the steps involved in the training.

- Download the images for each category into respective folders.

**Note:** Ideally, you should do the above step separately so that you can have a better control over the data. But for the sake of simplicity, I have included the code in `train.py` to download the images.
- Create the `DataBlock` and load the data using `dataloaders` function.
- Fine-tune a model (I used the `resnet18` model) using `vision_learner` function for a certain number of epochs.
- Export the model to a pickle file to be be used for inference.

You can find the documentation `fast.ai` library [here](https://docs.fast.ai/). It is built on the top of `pytorch` library.

The most important part of the training is the `DataBlock` which is used to create the `DataLoaders` object. The `DataLoaders` object is used to load the data for training the model.

```
# Create a DataBlock and load the data
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms()
).dataloaders(path, bs=32)
```
Try to understand each parameter and modify it according to your requirement. You can find the documentation for `DataBlock` [here](https://docs.fast.ai/data.block.html).

## Inference

![Screenshot 1](/demo.png)

The User Interface is built using the `Gradio` library. `Gradio` is a very useful library to quickly prototype the UI for machine learning models. You can find the documentation [here](https://gradio.app/docs/).

Modify the `app.py` according to requirement and run it.

```
python app.py
```

In the `app.py` you can provide example images, categories etc. 

**Note:** The categories much match the categories used while training and they should be provided in ascending order in `app.py` as `fastai` will sort them in acending order while training.



## Demo

You can view the live demo of UI [here](https://huggingface.co/spaces/harshasurampudi/Which_Planet)

## TODO

- [ ] Add more categories.
- [ ] Collect more data.
- [ ] Add more data augmentation techniques.
- [ ] Compare the performance of different models.


## References

- [fast.ai](https://docs.fast.ai/)
- [pytorch](https://pytorch.org/docs/stable/index.html)
- [Gradio](https://gradio.app/docs/)
- [Transfer Learning](https://en.wikipedia.org/wiki/Transfer_learning)
- [ResNet](https://en.wikipedia.org/wiki/Residual_neural_network)

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Author

[Harsha Surampudi](https://github.com/HarshaSurampudi)


Feel free to open an issue or a pull request if you find any bugs or have any suggestions.

:star: this repo if you found it useful.
