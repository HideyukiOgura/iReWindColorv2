# Demo Software

We provide a GUI which can run on CPU-only devices as well as devices with a GPU. 

Try out iColoriT with your own images and color hints! 

<p align="center">
  <img width="90%" src="../docs/demo.gif">
</p>

## Pretrained 

Checkpoints for iReWindColor models are available in the links below.
- [iReWindColor](https://drive.google.com/file/d/12qVzjag87ynDLcaoKew8YzEQmx5zZIEz/view?usp=sharing)
- [iReWindColorv2](https://drive.google.com/file/d/1I-zHixTHQci7GaVB8_NxkFTUILy5Xf6s/view?usp=sharing)


## Installation

Our code is implemented in Python 3.8, torch>=1.8.2
```
git clone https://github.com/HideyukiOgura/iReWindColorv2.git
pip install -r requirements.txt
```

## Run

Once you have satisfied all the requirements, you can run the base iColoriT model by executing

```
bash demo.sh --target_image <path/to/image>
```


### Controls

<ul>

<li> Left click on Drawing Pad to select the hint location. 

<li> Left click on the ab Color Gamut to select a color.

<li> Right click on a hint location to undo the click.

<li> Press <em>Restart</em> to undo all hints. 

<li> Press <em>Save</em> to save the colorized image. 

<li> Press <em>Load</em> to load another image.

</ul>

## Acknowledgments

Our GUI is an updated version of the [interactive-deep-colorization](https://github.com/junyanz/interactive-deep-colorization).
Thanks for sharing the codes!
