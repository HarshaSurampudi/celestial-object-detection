from duckduckgo_search import ddg_images
from fastcore.all import *
from fastai.vision.all import *
from time import sleep

def search_images(term, max_images=100):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image')

# list of planets to search for
searches = 'Mercury','Venus','Earth','Mars','Jupiter','Saturn','Uranus','Neptune'

# folder for dataset
path = Path('dataset')

# Download images, save to disk, and resize
for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'Planet {o}'))
    resize_images(path/o, max_size=400, dest=path/o)
    # Sleep so that we don't get overload the search engine. it's not necessary, but polite
    sleep(1)

# Remove any images that can't be opened
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
print(f"Removed {len(failed)} images that couldn't be opened")

# Create a DataBlock and load the data
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms()
).dataloaders(path, bs=32)

# Train the model
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)

# Save the model
learn.export('planets.pkl')