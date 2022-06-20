[![CI](https://github.com/jloveric/stripe-layer-classification/actions/workflows/python-app.yml/badge.svg)](https://github.com/jloveric/stripe-layer-classification/actions/workflows/python-app.yml)
# Positional encoding layers for piecewise polynomial layers
Currently an experiment.  Will be moved to high-order-layers-torch when I determine this is valuable.

A high order approach to image classification.  Uses the "stripe" layers for "convolutional" and
fully connected layers.  However, neither of these are the same as the standard version.  Instead
of convolutional blocks, stripes across the entire domain are used along with a positional encoding.
The "fully connected" layer still shares weights along each stripe, but the output of that layer takes
the inputs from all of the stripes (and their rotations).

Basically, since each unit in the piecewise polynomial layers is split into a bunch of segments you can
add a position (x) to a color (c) to locate x, c as a single value in one of the segments (or bins) of the
the unit.  It's basically a positional encoding, but one that I hypothesize should work really well with
piecewise polynomial layers.

##
Convolutional like
```
python examples/cifar100.py segments=10 n=5 max_epochs=10 rotations=2 periodicity=2.0
```
##
Fully connected like
```
python examples/cifar100invariant.py segments=10 n=3 max_epochs=10 rotations=1 periodicity=2.0 out_features=50
```
